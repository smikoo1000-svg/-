# PianoForge V7 — 고음 잡음 제거

V6 대비 변경점 (프로필 데이터/전사 엔진은 바이트 단위로 동일):
- `_suppress_high_register_noise()` 추가: 오디오 스펙트럼에서 "안정적이고 날카로운 선(실제 음)"인지 검사해 고음대 잡음 음표 제거
  (상한 초과 / 너무 짧음 / 비음정성 잡음 / 저음의 배음 / 동시발음 초과)
- UI에 "고음 잡음 제거 강도" 추가: 최대(기본) / 강함 / 보통 / 끄기
- 고음 음표 벨로시티 가산 제거, `sorted(Note)` 비교 오류 수정

합치기: `cat PianoForge_V7_part1of4.py.part PianoForge_V7_part2of4.py.part PianoForge_V7_part3of4.py.part PianoForge_V7_part4of4.py.part > PianoForge_V7.py`
sha256(PianoForge_V7.py) = 705c1d4cad9bfcaf010beacc53dca4bc6df6223036d13b42cf78fa68403457f6
