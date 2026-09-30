#!/usr/bin/env python3
"""정확한 파이프라인을 웹으로: python server.py  ->  http://localhost:8000"""
import subprocess, sys, threading, uuid
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory

HERE = Path(__file__).parent
JOBS = HERE / "jobs"; JOBS.mkdir(exist_ok=True)
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

PAGE = """<!doctype html><html lang=ko><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Song to Piano</title>
<style>body{font:16px/1.5 system-ui;max-width:640px;margin:32px auto;padding:0 16px}
button{background:#4f46e5;color:#fff;border:0;border-radius:8px;padding:10px 18px;font-size:15px;cursor:pointer}
button:disabled{opacity:.5}.c{background:#f3f3f7;border-radius:12px;padding:16px;margin:16px 0}a{display:block;margin:6px 0}</style>
<h1>🎹 Song to Piano</h1>
<p>노래 파일을 올리면 보컬·베이스·반주를 분리해 피아노 MIDI와 악보(MusicXML)로 만들어 줍니다. CPU에서는 곡당 몇 분 걸립니다.</p>
<div class=c><input type=file id=f accept="audio/*,.mp3,.m4a,.wav,.ogg,.flac,.aac"><div id=n style="color:#666;margin-top:6px"></div><br>
반주 방식 <select id=ac><option value=chords>코드 반주</option><option value=notes>음 전사(예전 방식)</option><option value=none>없음(멜로디+베이스만)</option></select><br><br>
박 추적 <select id=tr><option value=beat_this>Beat This!</option><option value=librosa>librosa(예전 방식)</option></select><br><br>
<label><input type=checkbox id=pv> 앞 60초만 빠르게 미리보기</label><br><br>
BPM 직접 지정 <input id=bp type=number placeholder="비우면 자동" style="width:110px"><br><br>
<button id=b>변환</button><p id=s></p><div id=r></div></div>
<script>
f.onchange=()=>{n.textContent=f.files[0]?'선택됨: '+f.files[0].name+' ('+(f.files[0].size/1048576).toFixed(1)+'MB)':''};
b.onclick=async()=>{if(!f.files[0])return s.textContent='파일을 선택하세요';
b.disabled=true;r.innerHTML='';s.textContent='변환 중… (창을 닫지 마세요)';
const d=new FormData();d.append('file',f.files[0]);d.append('accomp',ac.value);d.append('tracker',tr.value);d.append('bpm',bp.value);d.append('preview',pv.checked?'1':'');
try{const x=await fetch('/convert',{method:'POST',body:d}),t=await x.text();let j;
try{j=JSON.parse(t)}catch(e){throw Error('서버 응답 오류: '+t.slice(0,150))}
if(!x.ok)throw Error(j.error);const t0=Date.now();
for(;;){await new Promise(r=>setTimeout(r,3000));
const y=await fetch('/status/'+j.id),k=JSON.parse(await y.text());
if(k.status==='done'){s.textContent='완료!';r.innerHTML=k.files.map(n=>`<a href="/jobs/${j.id}/${n}" download>⬇ ${n}</a>`).join('');break}
if(k.status==='error')throw Error(k.error);
s.textContent='['+(k.stage||'준비 중')+'] '+Math.round((Date.now()-t0)/1000)+'초 경과 (창을 닫지 마세요)'}}
catch(e){s.textContent='오류: '+e.message}b.disabled=false}
</script></html>"""

@app.get("/")
def index():
    return PAGE

STATE = {}

def work(jid, cmd, d):
    tail = []
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        tail = (tail + [line])[-40:]
        if line.startswith("STAGE:"):
            STATE[jid] = dict(status="running", stage=line[6:].strip())
    p.wait()
    files = sorted(x.name for x in d.glob("*_piano.*"))
    if files:
        STATE[jid] = dict(status="done", files=files)
    else:
        STATE[jid] = dict(status="error", error="".join(tail)[-1500:])

@app.post("/convert")
def convert():
    up = request.files.get("file")
    if not up:
        return jsonify(error="파일이 없습니다"), 400
    jid = uuid.uuid4().hex[:10]
    d = JOBS / jid; d.mkdir()
    src = d / ("input" + Path(up.filename or "a.wav").suffix.lower()[:8])
    up.save(src)
    cmd = [sys.executable, str(HERE / "song2piano.py"), str(src), "-o", str(d)]
    acc = request.form.get("accomp", "chords")
    trk = request.form.get("tracker", "beat_this")
    cmd += ["--accomp", acc if acc in ("chords", "notes", "none") else "chords",
            "--tracker", trk if trk in ("beat_this", "librosa") else "beat_this"]
    if request.form.get("preview"):
        cmd += ["--max-sec", "60"]
    if request.form.get("bpm", "").replace(".", "", 1).isdigit():
        cmd += ["--bpm", request.form["bpm"]]
    STATE[jid] = dict(status="running")
    threading.Thread(target=work, args=(jid, cmd, d), daemon=True).start()
    return jsonify(id=jid)

@app.get("/status/<jid>")
def status(jid):
    return jsonify(STATE.get(jid, dict(status="error", error="작업을 찾을 수 없어요(서버가 재시작됐을 수 있어요)")))

@app.get("/jobs/<jid>/<name>")
def get(jid, name):
    return send_from_directory(JOBS / jid, name, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
