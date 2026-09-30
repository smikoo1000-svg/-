#!/usr/bin/env python3
"""정답 MIDI와 변환 결과를 음 단위로 비교해 점수(F1)를 낸다.
사용:  python tools/score_vs_truth.py 정답.mid 결과.mid [--from 0 --to 60]
P=정밀도(낸 음 중 정답인 비율), R=재현율(정답 음 중 맞힌 비율). 시작 시각 80ms + 음높이가 모두 맞아야 정답.
시간 이동은 -0.4~+0.4초를 자동 탐색해 가장 좋은 값을 쓴다(파일마다 시작 위치가 다를 수 있어서)."""
import argparse
import numpy as np, pretty_midi, mir_eval


def load(p, a, b):
    ns = [n for i in pretty_midi.PrettyMIDI(p).instruments for n in i.notes]
    return sorted((n for n in ns if a <= n.start < b), key=lambda n: n.start)


def f1(ref, est, shift, tol=0.08, pc=False):
    est = [n for n in est if n.start + shift >= 0]
    if not ref or not est:
        return 0.0, 0.0, 0.0
    iv = lambda ns, s: np.array([[n.start + s, max(n.end, n.start + .05) + s] for n in ns])
    hz = lambda ns: mir_eval.util.midi_to_hz(np.array([(n.pitch % 12 + 60) if pc else n.pitch for n in ns], float))
    P, R, F, _ = mir_eval.transcription.precision_recall_f1_overlap(
        iv(ref, 0), hz(ref), iv(est, shift), hz(est), onset_tolerance=tol, pitch_tolerance=50.0, offset_ratio=None)
    return P, R, F


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("truth"); ap.add_argument("result")
    ap.add_argument("--from", dest="a", type=float, default=0); ap.add_argument("--to", dest="b", type=float, default=1e9)
    x = ap.parse_args()
    ref, est = load(x.truth, x.a, x.b), load(x.result, x.a - .5, x.b + .5)
    F, s = max((f1(ref, est, s)[2], s) for s in np.arange(-0.4, 0.401, 0.01))
    P, R, F = f1(ref, est, s)
    print(f"정답 {len(ref)}음 / 결과 {len(est)}음 | P {P:.2f}  R {R:.2f}  F1 {F:.3f}  (시간 이동 {s*1000:+.0f}ms)")
    print(f"옥타브 무시 F1 {f1(ref, est, s, pc=True)[2]:.3f}")
