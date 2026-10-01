#!/usr/bin/env python3
"""정확한 파이프라인을 웹으로: python server.py  ->  http://localhost:8000"""
import json, re, subprocess, sys, threading, time, uuid
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory

HERE = Path(__file__).parent
JOBS = HERE / "jobs"; JOBS.mkdir(exist_ok=True)
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

@app.get("/")
def index():
    return send_from_directory(HERE / "static", "index.html")


@app.get("/static/<path:name>")
def static_files(name):
    return send_from_directory(HERE / "static", name)


@app.get("/health")
def health():
    return jsonify(ok=True)


STATE = {}

def work(jid, cmd, d):
    tail = []
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        tail = (tail + [line])[-40:]
        if line.startswith("STAGE:"):
            STATE[jid] = dict(status="running", stage=line[6:].strip())
    p.wait()
    files = sorted({x.name for pat in ("*_piano*.*", "*_score.*") for x in d.glob(pat)})
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
    (d / "meta.json").write_text(json.dumps(dict(
        name=up.filename or "audio", ts=int(time.time()), accomp=request.form.get("accomp", "octave"),
        preview=bool(request.form.get("preview")), bpm=request.form.get("bpm", "")), ensure_ascii=False), encoding="utf-8")
    cmd = [sys.executable, str(HERE / "song2piano.py"), str(src), "-o", str(d)]
    acc = request.form.get("accomp", "octave")
    trk = request.form.get("tracker", "beat_this")
    cmd += ["--accomp", acc if acc in ("octave", "lead", "chords", "notes", "none") else "octave",
            "--tracker", trk if trk in ("beat_this", "librosa") else "beat_this"]
    for key in ("offsets", "dynamics", "harmony", "pedal", "legato"):              # 켜기/끄기 옵션
        val = request.form.get(key)
        if val in ("0", "1"):
            cmd.append(f"--{key}" if val == "1" else f"--no-{key}")
    grid = request.form.get("grid", "auto")
    cmd += ["--grid", grid if grid in ("auto", "8", "16", "3", "mixed", "32", "off") else "auto"]
    try:
        cmd += ["--quantize-strength", str(min(1.0, max(0.0, float(request.form.get("qstrength", "1")))))]
    except ValueError:
        pass
    if request.form.get("preview"):
        cmd += ["--max-sec", "60"]
    if request.form.get("bpm", "").replace(".", "", 1).isdigit():
        cmd += ["--bpm", request.form["bpm"]]
    STATE[jid] = dict(status="running")
    threading.Thread(target=work, args=(jid, cmd, d), daemon=True).start()
    return jsonify(id=jid)

@app.post("/score")
def score():
    """MIDI 파일만 올리면 PDF 악보(+MusicXML)를 만든다 (음원 변환 없이 몇 초)."""
    up = request.files.get("file")
    if not up or Path(up.filename or "").suffix.lower() not in (".mid", ".midi"):
        return jsonify(error="MIDI 파일(.mid)을 올려주세요"), 400
    jid = uuid.uuid4().hex[:10]
    d = JOBS / jid; d.mkdir()
    src = d / "input.mid"
    up.save(src)
    (d / "meta.json").write_text(json.dumps(dict(name=up.filename, ts=int(time.time()), kind="score"), ensure_ascii=False), encoding="utf-8")
    cmd = [sys.executable, str(HERE / "midi2score.py"), str(src), "-o", str(d), "--title", Path(up.filename).stem[:60]]
    grid = request.form.get("grid", "auto")
    cmd += ["--grid", grid if grid in ("auto", "8", "16", "3", "mixed", "32") else "auto"]
    if request.form.get("playable") == "0":
        cmd.append("--no-playable")
    STATE[jid] = dict(status="running")
    threading.Thread(target=work, args=(jid, cmd, d), daemon=True).start()
    return jsonify(id=jid)


JID = re.compile(r"^[0-9a-f]{10}$")


def job_info(jid):
    d = JOBS / jid
    if not JID.match(jid) or not d.is_dir():
        return None
    meta = {}
    try:
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    st = STATE.get(jid)
    files = sorted({x.name for pat in ("*_piano*.*", "*_score.*") for x in d.glob(pat)})
    if st is None:
        st = dict(status="done", files=files) if files else dict(status="error", error="작업을 찾을 수 없어요(서버가 재시작됐을 수 있어요)")
    return dict(id=jid, meta=meta, **st)


@app.get("/status/<jid>")
def status(jid):
    info = job_info(jid)
    return jsonify(info or dict(status="error", error="작업을 찾을 수 없어요"))


@app.get("/api/job/<jid>")
def api_job(jid):
    info = job_info(jid)
    return (jsonify(info), 200) if info else (jsonify(error="없는 작업"), 404)


@app.get("/jobs/<jid>/<name>")
def get(jid, name):
    if not JID.match(jid):
        return jsonify(error="잘못된 요청"), 400
    return send_from_directory(JOBS / jid, name)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
