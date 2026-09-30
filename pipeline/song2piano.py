#!/usr/bin/env python3
"""가수곡 -> 피아노 편곡 MIDI + 악보(MusicXML).

1) Demucs(htdemucs)로 vocals / bass / other(반주) / drums 분리
2) Basic Pitch(Spotify)로 각 스템을 음표로 전사
3) 파트별 정리(보컬=단선율, 반주=동시음 제한, 베이스=단선율) 후 피아노 한 대로 합침
4) music21로 16분음표 정렬 + 그랜드 스태프 MusicXML 출력

사용:  python song2piano.py song.mp3 -o out/
"""
import argparse, subprocess, sys, tempfile
from pathlib import Path

import pretty_midi
from basic_pitch.inference import predict


def separate(audio: Path, work: Path, model: str) -> dict:
    subprocess.run([sys.executable, "-m", "demucs", "-n", model, "-o", str(work), str(audio)], check=True)
    d = work / model / audio.stem
    return {n: d / f"{n}.wav" for n in ("vocals", "bass", "other")}


def transcribe(wav: Path, **kw) -> list:
    _, midi, _ = predict(str(wav), **kw)
    return [n for i in midi.instruments for n in i.notes]


def monophonic(notes, min_len):
    """겹치는 음 중 큰 음(velocity)만 남기는 단선율화."""
    notes = sorted((n for n in notes if n.end - n.start >= min_len), key=lambda n: n.start)
    out = []
    for n in notes:
        if out and n.start < out[-1].end:
            if n.velocity > out[-1].velocity and n.start - out[-1].start > 0.05:
                out[-1].end = n.start
                out.append(n)
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


def build(stems, a):
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(0, name="Piano")
    parts = {
        "vocals": monophonic(transcribe(stems["vocals"], minimum_note_length=100, onset_threshold=0.5,
                                        frame_threshold=0.3, melodia_trick=True,
                                        minimum_frequency=130, maximum_frequency=1100), a.min_len),
        "bass": monophonic(transcribe(stems["bass"], minimum_note_length=120, minimum_frequency=35,
                                      maximum_frequency=350), 0.12),
        "other": limit_poly(transcribe(stems["other"], minimum_note_length=150, onset_threshold=0.55,
                                       frame_threshold=0.35), a.accomp_poly, 0.15, 45),
    }
    for name, ns in parts.items():
        if name == "other" and a.no_accompaniment:
            continue
        for n in ns:
            piano.notes.append(pretty_midi.Note(velocity=min(127, max(40, n.velocity)), pitch=n.pitch,
                                                start=n.start, end=n.end))
        print(f"{name}: {len(ns)} notes")
    pm.instruments.append(piano)
    return pm


def to_score(midi_path: Path, xml_path: Path):
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
    ap.add_argument("--accomp-poly", type=int, default=3, help="반주 최대 동시음")
    ap.add_argument("--no-accompaniment", action="store_true", help="멜로디+베이스만")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stems = separate(a.audio, Path(tmp), a.model)
        pm = build(stems, a)
    mid = a.out / f"{a.audio.stem}_piano.mid"
    pm.write(str(mid)); print("MIDI:", mid)
    try:
        xml = a.out / f"{a.audio.stem}_piano.musicxml"
        to_score(mid, xml); print("악보:", xml)
    except Exception as e:
        print("악보 변환 실패(MIDI는 저장됨):", e)


if __name__ == "__main__":
    main()
