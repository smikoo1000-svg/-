# Music to Piano 🎹

브라우저에서 동작하는 음악 → 피아노 변환 사이트. `index.html` 하나로 구성되며 서버·설치가 필요 없습니다.

- 오디오 업로드 → 다성 음높이 추정(STFT + 하모닉 합산 + 배음 제거 반복) → 음 추적 → 피아노 합성
- **가수곡 모드**: 보컬 음역의 멜로디 + 베이스 한 음씩만 추출해 피아노 편곡으로 변환
- 피아노롤 표시, 재생, MIDI/WAV 저장
- 템포 추정 → 16분음표 정렬 → 악보(그랜드 스태프) 표시, ABC 저장, 인쇄/PDF
- 파일은 외부로 전송되지 않습니다.

실행: `index.html`을 브라우저로 열거나 `python3 -m http.server` 후 접속. GitHub Pages로 바로 배포할 수 있습니다.

한계: 피아노/단일 악기에 가장 정확하며, 드럼·보컬이 섞인 곡은 근사치입니다. 정확도를 더 높이려면 Onsets and Frames, ByteDance Piano Transcription 같은 학습 모델(ONNX/TF.js)로 `transcribe()`를 교체하면 됩니다.

## 고정밀 파이프라인 (`pipeline/`)

브라우저 버전은 근사치라, 가수곡은 파이썬 파이프라인을 권장합니다.

```
pip install -r pipeline/requirements.txt
python pipeline/song2piano.py song.mp3 -o out/
```

Demucs로 보컬/베이스/반주 분리 → Basic Pitch로 각 파트 전사 → 피아노 한 대로 합쳐 `*_piano.mid`와 `*_piano.musicxml`(MuseScore 등에서 열기) 출력. 반주를 빼려면 `--no-accompaniment`. GPU가 없으면 곡당 몇 분 걸리며, Colab GPU에서 돌리면 빠릅니다.

### 정확한 버전을 사이트로 실행
```
pip install -r pipeline/requirements.txt
python pipeline/server.py      # http://localhost:8000
```
브라우저에서 파일을 올리면 MIDI와 악보(MusicXML)를 내려받을 수 있습니다. 서버가 필요하므로 GitHub Pages에는 올릴 수 없고, 본인 PC나 Colab/클라우드 서버에서 실행하세요.
