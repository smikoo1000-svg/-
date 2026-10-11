# PianoForge V7

V6 대비 변경점 (프로필 데이터/전사 엔진은 바이트 단위로 동일):
- 고음 잡음 제거: 오디오에서 "안정적이고 날카로운 음정선"인지 검사 (clean 기본 / max / strong / standard / off)
- 저음: 킥·클릭 잔음 수준의 아주 짧은 음표만 제거 (저음은 음정 검증 불가라 보수적)
- 박자 정렬: 고정 템포 자동 추정 -> 16분음표 격자, MIDI 템포 기록. 음 이어붙이기(같은 음 병합 + 레지스터별 레가토)
- 정확도: 자동 튜닝 보정(A=440 기준 어긋난 곡), 옥타브 오류 보정
- 내장 피아노 합성기(WAV): 비조화 배음, 1~3현 디튜닝, 타건 음색(세기별), 해머 노이즈, 2단 감쇠, 댐퍼/서스테인 페달, 스테레오, 룸 리버브
  - UI의 "고음질 피아노 WAV" 옵션 또는 `python PianoForge_V7.py --render in.mid out.wav`
- 고음 벨로시티 가산 제거, sorted(Note) 오류 수정

합치기: cat PianoForge_V7_part1of4.py.part PianoForge_V7_part2of4.py.part PianoForge_V7_part3of4.py.part PianoForge_V7_part4of4.py.part > PianoForge_V7.py
sha256(PianoForge_V7.py) = 01c9d8a1e971aef2023f6b55c52283c822e6035dc68265c87df510f612e249c1
