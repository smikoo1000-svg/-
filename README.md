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

### 파이프라인 구조 (Sheet Sage 방식 참고)
분리(Demucs) → **박/마디 추적(Beat This!)** → 보컬 멜로디·베이스 전사(Basic Pitch) → **코드 인식(크로마+Viterbi)** → 피아노 반주로 합성 → 16분음표 정렬 후 MIDI/MusicXML.
`--accomp chords|notes|none`(반주 방식), `--bpm 120`(템포 직접 지정).

### 정답과 비교해 점수 내기
`python tools/score_vs_truth.py 정답.mid 결과.mid` (P/R/F1). 반주 방식 `--accomp octave`(기본)는 정답 편곡과 비교해 점수가 가장 높았던 방식(멜로디를 한 옥타브 위로 겹침 + 베이스 접기).

### 출력 파일
- `*_piano.mid` : **재생용.** 원곡의 실제 리듬 그대로 + 박마다 템포 정보를 넣어서, 곡 내내 원곡과 박이 붙어 있음(스윙·3연음도 유지).
- `*_piano_quantized.mid` : 자동 감지한 박자 격자(8분/16분/3연음/혼합/32분)에 맞춘 판. `--quantize-strength 0~1`로 강도 조절.
- `*_piano_pedal.mid` : 서스테인 페달(CC64 127/0)만 담은 컨트롤러 트랙. 재생용 MIDI에도 같이 들어 있음.
- `*_piano_analysis.json` : BPM·신뢰도·템포 범위·격자·벨로시티·페달 통계.
- 옵션: `--harmony --offsets --dynamics --pedal --legato --grid {auto,8,16,3,mixed,32,off}`
- `*_piano.musicxml` : 악보(마디 정렬).

### 음이 빠지지 않게 하는 처리
원곡 스템에서 강한 소리가 시작하는데 음이 없는 지점을 찾아 음을 채우고(음높이가 분명한 소리만),
같은 음을 다시 발음한 경우(원곡에 실제 소리 시작이 있음)는 하나로 합치지 않음.
`NO_FILL=1` 환경변수로 끌 수 있음.
