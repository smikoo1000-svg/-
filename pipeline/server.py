#!/usr/bin/env python3
"""정확한 파이프라인을 웹으로: python server.py  ->  http://localhost:8000"""
import subprocess, sys, uuid
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
<label><input type=checkbox id=na> 멜로디+베이스만 (반주 제외)</label><br><br>
<button id=b>변환</button><p id=s></p><div id=r></div></div>
<script>
f.onchange=()=>{n.textContent=f.files[0]?'선택됨: '+f.files[0].name+' ('+(f.files[0].size/1048576).toFixed(1)+'MB)':''};
b.onclick=async()=>{if(!f.files[0])return s.textContent='파일을 선택하세요';
b.disabled=true;r.innerHTML='';s.textContent='변환 중… (창을 닫지 마세요)';
const d=new FormData();d.append('file',f.files[0]);d.append('no_acc',na.checked?'1':'');
try{const x=await fetch('/convert',{method:'POST',body:d}),j=await x.json();
if(!x.ok)throw Error(j.error);s.textContent='완료!';
r.innerHTML=j.files.map(n=>`<a href="/jobs/${j.id}/${n}" download>⬇ ${n}</a>`).join('')}
catch(e){s.textContent='오류: '+e.message}b.disabled=false}
</script></html>"""

@app.get("/")
def index():
    return PAGE

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
    if request.form.get("no_acc"):
        cmd.append("--no-accompaniment")
    p = subprocess.run(cmd, capture_output=True, text=True)
    files = sorted(x.name for x in d.glob("*_piano.*"))
    if not files:
        return jsonify(error=(p.stderr or p.stdout)[-300:]), 500
    return jsonify(id=jid, files=files)

@app.get("/jobs/<jid>/<name>")
def get(jid, name):
    return send_from_directory(JOBS / jid, name, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
