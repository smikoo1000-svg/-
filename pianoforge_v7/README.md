# PianoForge V7

V6 대비 변경점 (프로필 데이터/전사 엔진은 바이트 단위로 동일):
- `_suppress_high_register_noise()`: 오디오 스펙트럼에서 "안정적이고 날카로운 선(실제 음)"인지 검사해 잡음 음표 제거.
  강도: clean(기본, 실제 음 보존) / max / strong / standard / off
- 저음: 프레임보다 짧은 킥·클릭 잔음만 제거 (저음은 이 FFT 크기로 음정 검증이 불가능해서 보수적으로 처리)
- `_quantize_and_legato()`: 고정 템포(곡에서 자동 추정) 16분음표 격자에 온셋/오프셋 정렬 + 같은 음 조각 병합 + 레지스터별 레가토
  MIDI 템포도 추정 BPM으로 기록. 옵션: 박자 정렬 16분/8분/끄기, 음 이어붙이기 켜기/끄기
- 고음 음표 벨로시티 가산 제거, `sorted(Note)` 오류 수정

합치기: `cat PianoForge_V7_part1of4.py.part PianoForge_V7_part2of4.py.part PianoForge_V7_part3of4.py.part PianoForge_V7_part4of4.py.part > PianoForge_V7.py`
sha256(PianoForge_V7.py) = 85dba3a9c8d78b574fb076038f8bc97d4c7054d5887d5efe11aae3d321921b24
