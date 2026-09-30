#!/usr/bin/env python3
"""설치 + 서버 실행을 한 번에: python run.py  (가상환경이 켜져 있어도 OK)"""
import os, shutil, subprocess, sys
from pathlib import Path

root = Path(__file__).parent
venv = root / ".venv"
py = venv / "bin" / "python"
run = lambda *a: subprocess.run(list(map(str, a)), check=True, cwd=root)


def find_uv():
    if shutil.which("uv"):
        return shutil.which("uv")
    for pip in (shutil.which("pip3"), shutil.which("pip")):
        if pip:
            subprocess.run([pip, "install", "-q", "uv"])
            if shutil.which("uv"):
                return shutil.which("uv")
    sys.exit("uv 설치 실패: 터미널에서 'pip install uv' 를 먼저 실행해 주세요.")


uv = find_uv()
if not py.exists():
    run(uv, "venv", "--python", "3.11", venv)

check = "import pretty_midi, demucs, basic_pitch, music21, flask"
if subprocess.run([str(py), "-c", check], cwd=root, capture_output=True).returncode != 0:
    print("패키지 설치 중… (처음엔 몇 분 걸려요, Ctrl+C 누르지 마세요)")
    # GPU(CUDA)용 대용량 torch 대신 CPU용을 먼저 설치
    run(uv, "pip", "install", "--python", py, "torch", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cpu")
    run(uv, "pip", "install", "--python", py, "-r", "pipeline/requirements.txt")

print("서버 시작: 하단 '포트' 탭의 8000번을 브라우저로 여세요")
os.execv(str(py), [str(py), str(root / "pipeline" / "server.py")])
