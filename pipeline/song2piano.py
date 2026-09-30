#!/usr/bin/env python3
"""가수곡 -> 피아노 편곡 MIDI + 악보(MusicXML).

1) Demucs(htdemucs)로 vocals / bass / other(반주) / drums 분리
2) Basic Pitch(Spotify)로 각 스템을 음표로 전사
3) 파트별 정리(보컬=단선율, 반주=동시음 제한, 베이스=단선율) 후 피아노 한 대로 합침
4) music21로 16분음표 정렬 + 그랜드 스태프 MusicXML 출력

사용:  python song2piano.py song.mp3 -o out/
"""
import argparse, hashlib, shutil, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from pathlib import Path

import pretty_midi
from basic_pitch.inference import predict

CACHE = Path(__file__).parent / "cache"       # 같은 파일은 분리/박 추적 결과를 재사용
_MODEL = None


def stage(msg):
    print(f"STAGE: {msg}", flush=True)         # 웹 화면이 진행 단계를 보여주는 데 사용


def to_wav(audio: Path, work: Path, max_sec=None) -> Path:
    """mp3/m4a 등을 wav로 변환 (Demucs의 디코더 의존성 문제 회피)."""
    import soundfile as sf
    out = work / f"{audio.stem}.wav"
    try:
        data, sr = sf.read(str(audio), always_2d=True)
    except Exception:
        import librosa
        y, sr = librosa.load(str(audio), sr=None, mono=False)
        data = (y if y.ndim > 1 else y[None]).T
    if max_sec:
        data = data[:int(max_sec * sr)]
    sf.write(str(out), data, sr, subtype="PCM_16")
    return out


def file_key(audio: Path, model: str, max_sec) -> str:
    h = hashlib.sha1()
    with open(audio, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    h.update(f"{model}|{max_sec}".encode())
    return h.hexdigest()[:16]


def prepare(audio: Path, model: str, max_sec):
    """wav 변환 + Demucs 분리를 하고 결과를 캐시에 저장. 같은 파일이면 바로 재사용."""
    d = CACHE / file_key(audio, model, max_sec)
    names = ("vocals", "bass", "other", "mix")
    stems = {n: d / f"{n}.wav" for n in names}
    if all(p.exists() for p in stems.values()):
        stage("이전 분리 결과 재사용 (분리 생략)")
        return stems, None
    d.mkdir(parents=True, exist_ok=True)
    stage("오디오 변환")
    to_wav(audio, d, max_sec).replace(stems["mix"])
    return stems, d


def separate(stems: dict, d: Path, model: str):
    with tempfile.TemporaryDirectory() as tmp:
        stage("음원 분리 중 (가장 오래 걸려요)")
        subprocess.run([sys.executable, "-m", "demucs", "-n", model, "--overlap", "0.1", "-o", tmp,
                        str(stems["mix"])], check=True)
        out = Path(tmp) / model / "mix"
        for n in ("vocals", "bass", "other"):
            shutil.move(str(out / f"{n}.wav"), str(stems[n]))
    old = sorted((p for p in CACHE.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)[:-3]
    for p in old:                                  # 최근 3곡만 보관
        shutil.rmtree(p, ignore_errors=True)


def transcribe(wav: Path, **kw) -> list:
    global _MODEL
    if _MODEL is None:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import Model
        _MODEL = Model(ICASSP_2022_MODEL_PATH)
    _, midi, _ = predict(str(wav), _MODEL, **kw)
    return [n for i in midi.instruments for n in i.notes]


def merge_same_pitch(notes, gap):
    """같은 높이의 음이 gap초 이내로 끊겼다 다시 시작하면 하나로 이어 붙인다."""
    out = []
    for n in sorted(notes, key=lambda n: (n.pitch, n.start)):
        if out and out[-1].pitch == n.pitch and n.start - out[-1].end <= gap:
            out[-1].end = max(out[-1].end, n.end)
            out[-1].velocity = max(out[-1].velocity, n.velocity)
        else:
            out.append(n)
    return sorted(out, key=lambda n: n.start)


def in_range(notes, lo, hi):
    return [n for n in notes if lo <= n.pitch <= hi]


def monophonic(notes, min_len, step=0.01):
    """단선율화: 매 순간(10ms) 활성 음 중 가장 센 음 하나만 남긴다.
    (긴 음 도중에 짧은 음이 끼어도 뒤쪽이 사라지지 않는다.) 같은 높이로 이어지면 다시 합친다."""
    notes = [n for n in notes if n.end > n.start]
    if not notes:
        return []
    T = int(max(n.end for n in notes) / step) + 2
    owner = np.full(T, -1, int)
    order = sorted(range(len(notes)), key=lambda i: (notes[i].velocity, notes[i].end - notes[i].start))
    for i in order:                                   # 센 음이 나중에 덮어쓴다
        owner[int(notes[i].start / step):int(notes[i].end / step) + 1] = i
    out, t = [], 0
    while t < T:
        if owner[t] < 0:
            t += 1
            continue
        i, u = owner[t], t
        while u < T and owner[u] == i:
            u += 1
        src = notes[i]
        out.append(pretty_midi.Note(velocity=src.velocity, pitch=src.pitch, start=t * step, end=u * step))
        t = u
    merged = []
    for n in out:                                     # 같은 높이가 바로 이어지면 하나로
        if merged and merged[-1].pitch == n.pitch and n.start - merged[-1].end <= 2 * step:
            merged[-1].end = n.end
        else:
            merged.append(n)
    return [n for n in merged if n.end - n.start >= min_len]


def limit_poly(notes, max_poly, min_len, min_vel):
    """반주: 짧은/약한 음 제거, 동시음 max_poly개로 제한."""
    notes = sorted((n for n in notes if n.end - n.start >= min_len and n.velocity >= min_vel), key=lambda n: n.start)
    keep = []
    for n in notes:
        live = [k for k in keep if k.end > n.start]
        if len(live) >= max_poly:
            weakest = min(live + [n], key=lambda k: k.velocity)
            if weakest is n:
                continue
            keep.remove(weakest)
        keep.append(n)
    return keep


def unify_beat_level(beats):
    """박 추적기가 구간마다 4분음표/8분음표 단위를 오락가락하면(예: 97↔200 BPM)
    시간 축이 뒤틀린다. 기준 간격(80~160 BPM 범위로 접은 중앙값)에 맞춰 촘촘한 구간의 박을 솎아 낸다."""
    beats = np.asarray(beats, float)
    if len(beats) < 8:
        return beats
    ref = float(np.median(np.diff(beats)))
    while 60 / ref > 160:
        ref *= 2
    while 60 / ref < 80:
        ref /= 2
    # 큰 간격(느린 단위)이 실제로 더 흔하면 그쪽을 기준으로 삼는다
    iv = np.diff(beats)
    big = iv[(iv > 0.75 * ref * 1.5 * 0.66) & (iv < ref * 1.4)]
    if len(big) > 0.25 * len(iv):
        ref = float(np.median(big))
    kept = [beats[0]]
    for b in beats[1:]:
        if b - kept[-1] >= 0.75 * ref:
            kept.append(b)
    return np.asarray(kept)


def beat_grid(mix: Path, bpm_override=None, tracker="beat_this"):
    """박/첫박(다운비트) 추적. 1순위 Beat This!, 실패하면 librosa. -> (bpm, beats, k0)
    k0 = 첫 마디의 첫 박이 beats 배열의 몇 번째인지."""
    beats = downs = None
    try:
        if tracker != "beat_this":
            raise ImportError("librosa 사용 선택")
        from beat_this.inference import File2Beats
        beats, downs = File2Beats(device="cpu", dbn=False)(str(mix))
        beats, downs = np.asarray(beats, float), np.asarray(downs, float)
        print("beat tracker: Beat This!")
    except Exception as e:
        print("Beat This! 사용 불가 -> librosa로 대체:", type(e).__name__)
    if beats is None or len(beats) < 8:
        import librosa
        y, sr = librosa.load(str(mix), sr=22050, mono=True)
        _, beats = librosa.beat.beat_track(y=y, sr=sr, units="time", tightness=100)
        beats, downs = np.asarray(beats, float), np.asarray([], float)
    beats = unify_beat_level(beats)
    bpm = 60 * (len(beats) - 1) / float(beats[-1] - beats[0])   # 평균 템포(곡 끝까지 어긋나지 않게)
    while bpm < 80 and len(beats) > 1:               # 너무 느리면 박을 반으로 쪼갬
        beats = np.sort(np.concatenate([beats, (beats[:-1] + beats[1:]) / 2])); bpm *= 2
    while bpm > 160:                                  # 너무 빠르면 박을 2개씩 묶음
        beats = beats[::2]; bpm /= 2
    if bpm_override:                                  # 사용자가 준 BPM: 첫 박부터 균일 격자
        bpm = float(bpm_override)
        beats = beats[0] + np.arange(0, beats[-1] - beats[0] + 60 / bpm, 60 / bpm)
    k0 = int(np.argmin(np.abs(beats - downs[0]))) if len(downs) else 0
    return round(bpm, 2), beats, k0        # MIDI에 저장되는 값과 동일하게 맞춰 격자 오차 방지


def cached_beat_grid(stems, a):
    if a.bpm:
        return beat_grid(stems["mix"], a.bpm, a.tracker)
    f = stems["mix"].with_name(f"beats_{a.tracker}.npz")
    if f.exists():
        z = np.load(f)
        return float(z["bpm"]), z["beats"], int(z["k0"])
    bpm, beats, k0 = beat_grid(stems["mix"], None, a.tracker)
    np.savez(f, bpm=bpm, beats=beats, k0=k0)
    return bpm, beats, k0


def snap(notes, bpm, beats, k0=0, div=4, shift=None):
    """실제 시각 -> 박 위치(첫 마디 첫 박 기준) -> 16분음표 칸 정렬 -> 고정 템포 시각."""
    if len(beats) < 4:
        beats = np.arange(0, 600, 60 / bpm)
    idx = np.arange(len(beats)) - k0
    spb = 60 / bpm

    def pos(t):
        if t <= beats[0]:
            return idx[0] + (t - beats[0]) / spb
        if t >= beats[-1]:
            return idx[-1] + (t - beats[-1]) / spb
        return float(np.interp(t, beats, idx))

    raw = [(pos(n.start), pos(n.end), n) for n in notes]
    if shift is None:
        shift = 4 * int(np.ceil(max(0, -min((r[0] for r in raw), default=0)) / 4))  # 마디 단위 여유
    out = []
    for s0, e0, n in raw:
        s = round((s0 + shift) * div) / div
        e = max(round((e0 + shift) * div) / div, s + 1 / div)
        out.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=s * spb, end=e * spb))
    return out


# ---------- 코드 인식 (Sheet Sage 방식: 반주는 음표 대신 코드로) ----------
_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _templates():
    T, lab = [], []
    for r in range(12):
        for kind, iv in (("", (0, 4, 7)), ("m", (0, 3, 7))):
            v = np.zeros(12)
            for k, w in zip(iv, (1.0, 0.8, 0.9)):
                v[(r + k) % 12] = w
            T.append(v / np.linalg.norm(v)); lab.append((r, kind))
    return np.array(T), lab


def recognize_chords(stems, beats):
    """박 단위 크로마 -> 24개 장/단3화음 템플릿 + Viterbi 평활. -> [(시작박, 끝박, root, kind)]"""
    import librosa
    sr, hop = 22050, 512
    size = int(beats[-1] * sr) + sr
    y = sum(librosa.util.fix_length(librosa.load(str(stems[k]), sr=sr, mono=True)[0], size=size)
            for k in ("other", "bass"))
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    fr = librosa.time_to_frames(beats, sr=sr, hop_length=hop)
    n = len(beats) - 1
    C = np.zeros((n, 12))
    for i in range(n):
        seg = chroma[:, fr[i]:max(fr[i] + 1, fr[i + 1])]
        C[i] = np.median(seg, axis=1)
    norm = np.linalg.norm(C, axis=1, keepdims=True)
    C = C / np.maximum(norm, 1e-9)
    T, lab = _templates()
    sc = C @ T.T                                       # (n, 24) 코사인 유사도
    prob = np.exp(sc * 12); prob /= prob.sum(axis=1, keepdims=True)
    stay = 0.9
    trans = np.full((24, 24), (1 - stay) / 23); np.fill_diagonal(trans, stay)
    path = librosa.sequence.viterbi(prob.T, trans, p_init=np.full(24, 1 / 24))
    spans, st = [], 0
    for i in range(1, n + 1):
        if i == n or path[i] != path[st]:
            if norm[st:i].mean() > 1e-3:
                spans.append((st, i, *lab[path[st]]))
            st = i
    return spans


def chord_notes(spans, beats, lo=55, hi=71):
    """코드 -> 오른손 화음(가까운 전위로 부드럽게 연결). 왼손 베이스는 별도 전사를 쓴다."""
    out, prev = [], None
    for s, e, root, kind in spans:
        pcs = [root, (root + (3 if kind == "m" else 4)) % 12, (root + 7) % 12]
        cands = [[p for p in range(lo - 12, hi + 13) if p % 12 == pc] for pc in pcs]
        best, bd = None, 1e9
        for a in cands[0]:
            for b in cands[1]:
                for c in cands[2]:
                    v = sorted([a, b, c])
                    if v[0] < lo or v[-1] > hi or v[-1] - v[0] > 12:
                        continue
                    d = 0 if prev is None else sum(abs(x - y) for x, y in zip(v, prev))
                    if d < bd:
                        best, bd = v, d
        if best is None:
            continue
        prev = best
        t0, t1 = beats[s], beats[e] if e < len(beats) else beats[-1] + (beats[-1] - beats[-2])
        out += [pretty_midi.Note(velocity=55, pitch=p, start=float(t0), end=float(t1)) for p in best]
    return out


def build(stems, a, grid):
    bpm, beats, k0 = grid
    print(f"tempo: {bpm:.1f} BPM, beats: {len(beats)}, 첫 마디 시작 박 #{k0}")
    pm = pretty_midi.PrettyMIDI(initial_tempo=round(bpm, 2))
    piano = pretty_midi.Instrument(0, name="Piano")

    def vocals():
        return monophonic(merge_same_pitch(in_range(transcribe(stems["vocals"], minimum_note_length=80,
                          onset_threshold=0.35, frame_threshold=0.2, melodia_trick=True,
                          minimum_frequency=100, maximum_frequency=1200), 48, 84), 0.12), a.min_len)

    def bass():
        return monophonic(merge_same_pitch(in_range(transcribe(stems["bass"], minimum_note_length=90,
                          onset_threshold=0.4, frame_threshold=0.25, minimum_frequency=35,
                          maximum_frequency=350), 28, 60), 0.15), 0.1)

    def other():
        return limit_poly(merge_same_pitch(in_range(transcribe(stems["other"], minimum_note_length=100,
                          onset_threshold=0.45, frame_threshold=0.3), 48, 88), 0.2),
                          a.accomp_poly, 0.2, 30)

    def lead():
        """반주 스템의 눈에 띄는 리드 선율(신스/기타/카우벨 등). 보컬과 겹치는 음은 뺀다."""
        return monophonic(merge_same_pitch(in_range(transcribe(stems["other"], minimum_note_length=80,
                          onset_threshold=0.4, frame_threshold=0.25), 60, 96), 0.1), 0.08)

    def chords():
        spans = recognize_chords(stems, beats)
        print("chords:", " ".join(f"{_NAMES[r]}{k}" for _, _, r, k in spans[:16]), "…")
        return chord_notes(spans, beats)

    stage("보컬·베이스·코드 분석 중 (동시에 처리)")
    tasks = {"vocals": vocals, "bass": bass}
    if a.accomp == "notes":
        tasks["other"] = other
    elif a.accomp in ("chords", "lead"):
        tasks["chords"] = chords
        if a.accomp == "lead":
            tasks["lead"] = lead
    with ThreadPoolExecutor(len(tasks)) as ex:
        futs = {k: ex.submit(f) for k, f in tasks.items()}
        parts = {k: f.result() for k, f in futs.items()}
    if "lead" in parts:                                   # 보컬과 같은 순간·같은 음이면 중복이라 제거
        v = parts["vocals"]
        parts["lead"] = [n for n in parts["lead"] if n.velocity >= 40 and not any(
            x.start < n.end and n.start < x.end and x.pitch % 12 == n.pitch % 12 for x in v)]
    shift = None
    allraw = [n for ns in parts.values() for n in ns]
    if allraw:
        first = min(np.interp(n.start, beats, np.arange(len(beats)) - k0) if beats[0] <= n.start <= beats[-1]
                    else (n.start - beats[0]) / (60 / bpm) - k0 for n in allraw)
        shift = 4 * int(np.ceil(max(0, -first) / 4))
    for name, ns in parts.items():
        for n in snap(ns, bpm, beats, k0, shift=shift):
            n.velocity = min(127, max(40, n.velocity))
            piano.notes.append(n)
        print(f"{name}: {len(ns)} notes")
    pm.instruments.append(piano)
    return pm


def to_score(midi_path: Path, xml_path: Path, bpm: float = 100):
    from music21 import converter, stream, clef, meter, tempo
    s = converter.parse(str(midi_path), quantizePost=True, quarterLengthDivisors=(4, 3))
    flat = s.flatten()
    right, left = stream.Part(), stream.Part()
    right.insert(0, clef.TrebleClef()); left.insert(0, clef.BassClef())
    from music21 import note, chord
    for el in flat.notes:
        ps = list(el.pitches) if el.isChord else [el.pitch]
        hi = [p for p in ps if p.midi >= 60]; lo = [p for p in ps if p.midi < 60]
        for group, part in ((hi, right), (lo, left)):
            if group:
                n = chord.Chord(group) if len(group) > 1 else note.Note(group[0])
                n.quarterLength = max(0.25, el.quarterLength)
                part.insert(el.offset, n)
    right.insert(0, tempo.MetronomeMark(number=round(bpm)))
    score = stream.Score([right, left])
    for p in score.parts:
        p.insert(0, meter.TimeSignature("4/4"))
        p.makeMeasures(inPlace=True)
    score.write("musicxml", fp=str(xml_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("out"))
    ap.add_argument("--model", default="htdemucs")
    ap.add_argument("--min-len", type=float, default=0.06, help="보컬 최소 음 길이(초)")
    ap.add_argument("--accomp-poly", type=int, default=2, help="반주 최대 동시음")
    ap.add_argument("--accomp", choices=["lead", "chords", "notes", "none"], default="lead",
                    help="반주 방식: lead=코드+리드 선율(기본), chords=코드만, notes=음 전사(복잡), none=멜로디+베이스만")
    ap.add_argument("--no-accompaniment", action="store_true", help="--accomp none 과 동일")
    ap.add_argument("--tracker", choices=["beat_this", "librosa"], default="beat_this", help="박 추적 방식")
    ap.add_argument("--max-sec", type=float, default=None, help="앞 N초만 변환(빠른 미리보기)")
    ap.add_argument("--bpm", type=float, default=None, help="BPM을 직접 지정(자동 추정이 틀릴 때)")
    a = ap.parse_args()
    if a.no_accompaniment:
        a.accomp = "none"
    a.out.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    stems, fresh = prepare(a.audio, a.model, a.max_sec)
    with ThreadPoolExecutor(1) as ex:
        stage("박·마디 추적 (음원 분리와 동시에)")
        beat_future = ex.submit(cached_beat_grid, stems, a)      # 분리와 병렬로 실행
        if fresh is not None:
            separate(stems, fresh, a.model)
        grid = beat_future.result()
    pm = build(stems, a, grid)
    stage("MIDI·악보 저장")
    mid = a.out / f"{a.audio.stem}_piano.mid"
    pm.write(str(mid)); print("MIDI:", mid)
    try:
        xml = a.out / f"{a.audio.stem}_piano.musicxml"
        to_score(mid, xml, pm.get_tempo_changes()[1][0]); print("악보:", xml)
    except Exception as e:
        print("악보 변환 실패(MIDI는 저장됨):", e)


if __name__ == "__main__":
    main()
