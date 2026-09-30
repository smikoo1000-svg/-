# Music to Piano 🎹

브라우저에서 동작하는 음악 → 피아노 변환 사이트. `index.html` 하나로 구성되며 서버·설치가 필요 없습니다.

- 오디오 업로드 → 다성 음높이 추정(STFT + 하모닉 합산 + 배음 제거 반복) → 음 추적 → 피아노 합성
- **가수곡 모드**: 보컬 음역의 멜로디 + 베이스 한 음씩만 추출해 피아노 편곡으로 변환
- 피아노롤 표시, 재생, MIDI/WAV 저장
- 템포 추정 → 16분음표 정렬 → 악보(그랜드 스태프) 표시, ABC 저장, 인쇄/PDF
- 파일은 외부로 전송되지 않습니다.

실행: `index.html`을 브라우저로 열거나 `python3 -m http.server` 후 접속. GitHub Pages로 바로 배포할 수 있습니다.

한계: 피아노/단일 악기에 가장 정확하며, 드럼·보컬이 섞인 곡은 근사치입니다. 정확도를 더 높이려면 Onsets and Frames, ByteDance Piano Transcription 같은 학습 모델(ONNX/TF.js)로 `transcribe()`를 교체하면 됩니다.
