#!/usr/bin/env python3
"""가수곡 -> 피아노 편곡 MIDI + 악보(MusicXML).

1) Demucs(htdemucs)로 vocals / bass / other(반주) / drums 분리
2) Basic Pitch(Spotify)로 각 스템을 음표로 전사
3) 파트별 정리(보컬=단선율, 반주=동시음 제한, 베이스=단선율) 후 피아노 한 대로 합침
4) music21로 16분음표 정렬 + 그랜드 스태프 MusicXML 출력

사용:  python song2piano.py song.mp3 -o out/
"""
import argparse, subprocess, sys, tempfile
import numpy as np
from pathlib import Path

import pretty_midi
from basic_pitch.inference import predict


def to_wav(audio: Path, work: Path) -> Path:
    """mp3/m4a 등을 wav로 변환 (Demucs의 디코더 의존성 문제 회피)."""
    import soundfile as sf
    out = work / f"{audio.stem}.wav"
    try:
        data, sr = sf.read(str(audio), always_2d=True)
    except Exception:
        import librosa
        y, sr = librosa.load(str(audio), sr=None, mono=False)
        data = (y if y.ndim > 1 else y[None]).T
    sf.write(str(out), data, sr, subtype="PCM_16")
    return out


def separate(audio: Path, work: Path, model: str) -> dict:
    audio = to_wav(audio, work)
    subprocess.run([sys.executable, "-m", "demucs", "-n", model, "-o", str(work), str(audio)], check=True)
    d = work / model / audio.stem
    stems = {n: d / f"{n}.wav" for n in ("vocals", "bass", "other")}
    stems["mix"] = audio
    return stems


def transcribe(wav: Path, **kw) -> list:
    _, midi, _ = predict(str(wav), **kw)
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


def monophonic(notes, min_len):
    """겹치는 음 중 큰 음(velocity)만 남기는 단선율화."""
    notes = sorted((n for n in notes if n.end - n.start >= min_len), key=lambda n: n.start)
    out = []
    for n in notes:
        if out and n.start < out[-1].end:
            if n.start - out[-1].start >= 0.06:      # 새 음이 시작되면 이전 음을 끊는다
                out[-1].end = n.start
                out.append(n)
            elif n.velocity > out[-1].velocity:      # 거의 동시에 시작하면 큰 쪽만
                out[-1] = n
            continue
        out.append(n)
    return out


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
    bpm = 60 / float(np.median(np.diff(beats)))
    while bpm < 80 and len(beats) > 1:               # 너무 느리면 박을 반으로 쪼갬
        beats = np.sort(np.concatenate([beats, (beats[:-1] + beats[1:]) / 2])); bpm *= 2
    while bpm > 160:                                  # 너무 빠르면 박을 2개씩 묶음
        beats = beats[::2]; bpm /= 2
    if bpm_override:                                  # 사용자가 준 BPM: 첫 박부터 균일 격자
        bpm = float(bpm_override)
        beats = beats[0] + np.arange(0, beats[-1] - beats[0] + 60 / bpm, 60 / bpm)
    k0 = int(np.argmin(np.abs(beats - downs[0]))) if len(downs) else 0
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


def build(stems, a):
    bpm, beats, k0 = beat_grid(stems["mix"], a.bpm, a.tracker)
    print(f"tempo: {bpm:.1f} BPM, beats: {len(beats)}, 첫 마디 시작 박 #{k0}")
    pm = pretty_midi.PrettyMIDI(initial_tempo=round(bpm, 2))
    piano = pretty_midi.Instrument(0, name="Piano")
    parts = {
        "vocals": monophonic(merge_same_pitch(in_range(transcribe(stems["vocals"], minimum_note_length=80,
                             onset_threshold=0.35, frame_threshold=0.2, melodia_trick=True,
                             minimum_frequency=100, maximum_frequency=1200), 48, 84), 0.12), a.min_len),
        "bass": monophonic(merge_same_pitch(in_range(transcribe(stems["bass"], minimum_note_length=90,
                           onset_threshold=0.4, frame_threshold=0.25, minimum_frequency=35,
                           maximum_frequency=350), 28, 60), 0.15), 0.1),
    }
    if a.accomp == "notes":
        parts["other"] = limit_poly(merge_same_pitch(in_range(transcribe(stems["other"], minimum_note_length=100,
                                    onset_threshold=0.45, frame_threshold=0.3), 48, 88), 0.2),
                                    a.accomp_poly, 0.2, 30)
    elif a.accomp == "chords":
        spans = recognize_chords(stems, beats)
        print("chords:", " ".join(f"{_NAMES[r]}{k}" for _, _, r, k in spans[:16]), "…")
        parts["chords"] = chord_notes(spans, beats)
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
    ap.add_argument("--min-len", type=float, default=0.1, help="보컬 최소 음 길이(초)")
    ap.add_argument("--accomp-poly", type=int, default=2, help="반주 최대 동시음")
    ap.add_argument("--accomp", choices=["chords", "notes", "none"], default="chords",
                    help="반주 방식: chords=코드 인식(기본, 깔끔), notes=음 전사(복잡), none=멜로디+베이스만")
    ap.add_argument("--no-accompaniment", action="store_true", help="--accomp none 과 동일")
    ap.add_argument("--tracker", choices=["beat_this", "librosa"], default="beat_this", help="박 추적 방식")
    ap.add_argument("--bpm", type=float, default=None, help="BPM을 직접 지정(자동 추정이 틀릴 때)")
    a = ap.parse_args()
    if a.no_accompaniment:
        a.accomp = "none"
    a.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stems = separate(a.audio, Path(tmp), a.model)
        pm = build(stems, a)
    mid = a.out / f"{a.audio.stem}_piano.mid"
    pm.write(str(mid)); print("MIDI:", mid)
    try:
        xml = a.out / f"{a.audio.stem}_piano.musicxml"
        to_score(mid, xml, pm.get_tempo_changes()[1][0]); print("악보:", xml)
    except Exception as e:
        print("악보 변환 실패(MIDI는 저장됨):", e)


if __name__ == "__main__":
    main()
