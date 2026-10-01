#!/usr/bin/env python3
"""MIDI 파일 -> PDF 악보 (+ MusicXML).

우리 변환기가 만든 `*_piano.mid`(박마다 템포가 바뀌는 MIDI)뿐 아니라 어떤 MIDI든 받는다.
 1) MIDI의 템포 정보에서 박 위치를 얻고, 첫 박(마디 시작)을 베이스 음 위치로 추정
 2) 음 위치를 보고 격자(8분/16분/3연음/혼합)와 위상 오차를 자동으로 정해 정렬
 3) 오른손/왼손 분리 + 한 손으로 칠 수 있게 정리(동시음 4개·한 옥타브 이내)
 4) MusicXML -> Verovio로 조판 -> PDF

사용:  python midi2score.py song_piano.mid -o out/ [--title 제목]
"""
import argparse, io, sys
from pathlib import Path
import numpy as np
import pretty_midi


def load_notes(midi_path):
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = [n for i in pm.instruments if not i.is_drum for n in i.notes]
    if not notes:
        raise ValueError("MIDI에 음이 없습니다")
    beats = np.asarray(pm.get_beats(), float)
    if len(beats) < 8:                                     # 템포 정보가 부실하면 평균 템포로 균일 박 생성
        bpm = float(pm.estimate_tempo()) if len(notes) > 20 else 100.0
        while bpm < 70:
            bpm *= 2
        while bpm > 160:
            bpm /= 2
        beats = np.arange(0, pm.get_end_time() + 60 / bpm, 60 / bpm)
    return notes, beats


def estimate_downbeat(notes, beats):
    """베이스(낮은 음)가 가장 자주 시작하는 박을 마디 첫 박으로 본다."""
    low = [n.start for n in notes if n.pitch < 55] or [n.start for n in notes]
    idx = np.arange(len(beats))
    pos = np.interp(low, beats, idx)
    near = np.abs(pos - np.round(pos)) < 0.15
    cnt = np.bincount(np.round(pos[near]).astype(int) % 4, minlength=4)
    return int(np.argmax(cnt)) if cnt.sum() >= 8 else 0


def make_bar_midi(notes, beats, bars_path, grid="auto"):
    import enhance as H
    from song2piano import snap
    k0 = estimate_downbeat(notes, beats)
    names = {"8": (2,), "16": (4,), "3": (3,), "mixed": (4, 3), "32": (8,)}
    if grid == "auto":
        divs, _ = H.choose_grid([n.start for n in notes], beats, k0)
    else:
        divs = names.get(grid, (4,))
    if tuple(divs) == (4, 3):                               # 16분+3연음 혼합은 잡음일 때가 많고 악보가 지저분해져서 16분으로
        divs = (4,)
    phase, _ = H.estimate_phase([n.start for n in notes], beats, max(divs))
    bpm = 60 * (len(beats) - 1) / float(beats[-1] - beats[0])
    first = min(np.interp(n.start, beats, np.arange(len(beats)) - k0) if beats[0] <= n.start <= beats[-1]
                else (n.start - beats[0]) / (60 / bpm) - k0 for n in notes)
    shift = 4 * int(np.ceil(max(0, -first) / 4))
    pm = pretty_midi.PrettyMIDI(initial_tempo=round(bpm, 2))
    piano = pretty_midi.Instrument(program=0)
    piano.notes = snap(notes, round(bpm, 2), beats, k0, div=divs, shift=shift, phase=phase)
    pm.instruments.append(piano)
    pm.write(str(bars_path))
    return bpm, divs


def render_pdf(xml_path, pdf_path):
    """MusicXML -> Verovio(SVG, A4 세로) -> PDF. 페이지를 한 파일로 합친다."""
    import verovio, cairosvg
    from pypdf import PdfReader, PdfWriter
    tk = verovio.toolkit()
    tk.setOptions({"pageWidth": 2100, "pageHeight": 2970, "scale": 45, "pageMarginLeft": 80, "pageMarginRight": 80,
                   "pageMarginTop": 100, "pageMarginBottom": 100, "adjustPageHeight": False,
                   "breaks": "auto", "svgViewBox": True, "footer": "none"})
    tk.loadData(Path(xml_path).read_text(encoding="utf-8"))
    w = PdfWriter()
    for pg in range(1, tk.getPageCount() + 1):
        svg = tk.renderToSVG(pg)
        buf = io.BytesIO()
        cairosvg.svg2pdf(bytestring=svg.encode("utf-8"), write_to=buf, output_width=793.7, output_height=1122.5)
        for page in PdfReader(io.BytesIO(buf.getvalue())).pages:
            w.add_page(page)
    with open(pdf_path, "wb") as f:
        w.write(f)
    return tk.getPageCount()


def midi_to_score(midi_path, out_dir, title=None, grid="auto", playable=True):
    from song2piano import to_score
    midi_path, out_dir = Path(midi_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = midi_path.stem
    notes, beats = load_notes(midi_path)
    bars = out_dir / f".{stem}_bars.mid"
    bpm, divs = make_bar_midi(notes, beats, bars, grid)
    xml = out_dir / f"{stem}_score.musicxml"
    to_score(bars, xml, bpm, playable=playable, title=title or stem.replace("_piano", "").replace("_", " "))
    bars.unlink(missing_ok=True)
    pdf = out_dir / f"{stem}_score.pdf"
    pages = render_pdf(xml, pdf)
    return pdf, xml, pages, bpm, divs


def main():
    ap = argparse.ArgumentParser(description="MIDI -> PDF 악보")
    ap.add_argument("midi", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("out"))
    ap.add_argument("--title", default=None)
    ap.add_argument("--grid", choices=["auto", "8", "16", "3", "mixed", "32"], default="auto")
    ap.add_argument("--playable", action=argparse.BooleanOptionalAction, default=True)
    a = ap.parse_args()
    pdf, xml, pages, bpm, divs = midi_to_score(a.midi, a.out, a.title, a.grid, a.playable)
    print(f"PDF 악보: {pdf} ({pages}쪽, BPM {bpm:.1f}, 격자 {divs})")
    print(f"MusicXML: {xml}")


if __name__ == "__main__":
    main()
