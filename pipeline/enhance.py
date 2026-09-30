"""음표 후처리 고급 기능 모음.

- suppress_overtones / chord_filter : 배음(오버톤) 오인식 제거, 코드 구성음 기반 다성음 정리
- refine_offsets                    : CQT 에너지 감쇠로 음이 끝나는 시점(오프셋) 정밀화
- assign_velocity                   : 어택 세기 + 소리 크기(dB)로 음마다 벨로시티(1~127) 산출
- make_pedal                        : 화성이 바뀌는 지점마다 서스테인 페달(CC64) 구간 생성
- choose_grid / snap_grid           : 박 격자(8분·16분·3연음·혼합) 자동 선택과 강도 조절 퀀타이즈
"""
import numpy as np
import librosa
import pretty_midi

SR = 22050
HOP = 512
_CACHE = {}


def _audio(wav):
    k = ("y", str(wav))
    if k not in _CACHE:
        _CACHE[k] = librosa.load(str(wav), sr=SR, mono=True)[0]
    return _CACHE[k]


def _cqt(wav):
    """피아노 음역 CQT 크기(C1=MIDI 24부터 84개 반음 구간), 95퍼센타일로 정규화."""
    k = ("cqt", str(wav))
    if k not in _CACHE:
        C = np.abs(librosa.cqt(_audio(wav), sr=SR, hop_length=HOP, fmin=librosa.midi_to_hz(24), n_bins=84,
                               bins_per_octave=12))
        _CACHE[k] = C / (np.percentile(C, 95) + 1e-9)
    return _CACHE[k]


def _frame(t):
    return int(t * SR / HOP)


# ---------------------------------------------------------------- 1) 다성음/화음 정리
_OVERTONE_INTERVALS = (12, 19, 24, 28, 31, 34)


def suppress_overtones(notes, ratio=0.8, min_overlap=0.5):
    """더 낮은 음의 배음(옥타브·12도·2옥타브 등)으로 설명되고 그 음보다 약한 음은 배음 오인식으로 보고 제거한다."""
    ns = sorted(notes, key=lambda n: n.start)
    keep = []
    for i, n in enumerate(ns):
        dur = max(n.end - n.start, 1e-3)
        explained = False
        for m in ns[max(0, i - 60):i + 60]:
            if m is n or n.pitch - m.pitch not in _OVERTONE_INTERVALS:
                continue
            ov = min(n.end, m.end) - max(n.start, m.start)
            if ov >= min_overlap * dur and m.velocity >= ratio * n.velocity and abs(m.start - n.start) <= 0.08:
                explained = True
                break
        if not explained:
            keep.append(n)
    return keep


def chord_filter(notes, spans, beats, strong_vel=80):
    """코드 구간의 구성음(3화음+7음)이 아닌 약한 음은 제거. 코드 없는 구간은 그대로 둔다."""
    if not spans:
        return notes
    bt = np.asarray(beats, float)

    def tones(root, kind):
        third = 3 if "m" in kind and "maj" not in kind else (5 if kind == "sus4" else 4)
        s = {root % 12, (root + third) % 12, (root + 7) % 12}
        if kind in ("7",):
            s.add((root + 10) % 12)
        if kind in ("maj7",):
            s.add((root + 11) % 12)
        if kind in ("m7",):
            s.add((root + 10) % 12)
        return s
    bounds = [(bt[min(s, len(bt) - 1)], bt[min(e, len(bt) - 1)], tones(r, k)) for s, e, r, k in spans]
    starts = np.array([b[0] for b in bounds])
    out = []
    for n in notes:
        j = int(np.searchsorted(starts, n.start, side="right")) - 1
        if j < 0 or n.start > bounds[j][1] or n.pitch % 12 in bounds[j][2] or n.velocity >= strong_vel:
            out.append(n)
    return out


# ---------------------------------------------------------------- 2) 오프셋 정밀화
def refine_offsets(notes, wav, decay=0.25, max_ext=0.25, min_len=0.06, pitch_class=False):
    """음의 끝을 '그 음높이 에너지가 최고치의 decay 배 아래로 떨어지는 시점'으로 맞춘다.
    원래 끝보다 최대 max_ext초 늘릴 수 있고, 같은 높이의 다음 음 시작 전에서 끊는다."""
    if not notes:
        return notes
    C = _cqt(wav)
    nfr = C.shape[1]
    ext = int(max_ext * SR / HOP)
    order = sorted(range(len(notes)), key=lambda i: notes[i].start)
    nxt = {}
    last = {}
    for i in reversed(order):
        n = notes[i]
        nxt[i] = last.get(n.pitch % 12 if pitch_class else n.pitch)
        last[n.pitch % 12 if pitch_class else n.pitch] = n.start
    out = []
    for i, n in enumerate(notes):
        idx = n.pitch - 24
        if idx < 1 or idx >= 83:
            out.append(n)
            continue
        if pitch_class:
            e = C[[b for b in range(84) if (b + 24) % 12 == n.pitch % 12]].sum(axis=0)
        else:
            e = C[idx - 1:idx + 2].max(axis=0)
        a = min(_frame(n.start), nfr - 2)
        b = min(_frame(n.end), nfr - 1)
        peak = float(e[a:a + max(3, int(0.12 * SR / HOP))].max())
        if peak < 1e-3:
            out.append(n)
            continue
        hi = min(nfr - 2, b + ext)
        t_end = None
        for t in range(a + 3, hi):
            if e[t] < decay * peak and e[t + 1] < decay * peak:
                t_end = t
                break
        if t_end is None:
            new_end = (hi * HOP / SR) if e[min(b, nfr - 1)] > 0.6 * peak else n.end
        else:
            new_end = t_end * HOP / SR
        new_end = max(n.start + min_len, new_end)
        if nxt.get(i) is not None:
            new_end = min(new_end, max(n.start + min_len, nxt[i] - 0.01))
        out.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=n.start, end=float(new_end)))
    return out


def offset_quality(notes, wav, pitch_class=False):
    """측정용: (음 끝 뒤에도 에너지가 남는 비율, 음 끝 직전에 이미 무음인 비율). 둘 다 낮을수록 좋다."""
    C = _cqt(wav)
    cut = ring = tot = 0
    for n in notes:
        idx = n.pitch - 24
        if idx < 1 or idx >= 83:
            continue
        e = C[[b for b in range(84) if (b + 24) % 12 == n.pitch % 12]].sum(axis=0) if pitch_class \
            else C[idx - 1:idx + 2].max(axis=0)
        a, b = _frame(n.start), _frame(n.end)
        if b + 5 >= C.shape[1] or b - a < 4:
            continue
        peak = float(e[a:a + max(3, int(0.12 * SR / HOP))].max())
        if peak < 1e-3:
            continue
        tot += 1
        cut += e[b:b + 4].mean() > 0.5 * peak          # 아직 울리는데 끊음
        ring += e[max(a, b - 4):b].mean() < 0.1 * peak  # 이미 사라졌는데 붙들고 있음
    return (cut / max(tot, 1), ring / max(tot, 1), tot)


# ---------------------------------------------------------------- 3) 벨로시티
def assign_velocity(notes, wav, base=80, spread=19, lo=28, hi=127):
    """음마다: 어택 세기(온셋 세기 피크) + 소리 크기(시작 후 약 120ms 평균 dB)를 z점수로 합쳐 벨로시티를 정한다."""
    if len(notes) < 8:
        return notes
    y = _audio(wav)
    k = ("env", str(wav))
    if k not in _CACHE:
        _CACHE[k] = (librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP),
                     librosa.amplitude_to_db(librosa.feature.rms(y=y, hop_length=HOP)[0] + 1e-6, ref=1.0))
    env, db = _CACHE[k]
    att, loud = [], []
    w = max(2, int(0.12 * SR / HOP))
    for n in notes:
        a = min(_frame(n.start), len(env) - 1)
        att.append(float(env[max(0, a - 1):a + 4].max()))
        loud.append(float(db[a:a + w].mean()))
    att = np.log1p(np.array(att))
    loud = np.array(loud)
    z = lambda x: (x - np.median(x)) / (1.4826 * np.median(np.abs(x - np.median(x))) + 1e-6)
    zz = np.clip(0.5 * z(att) + 0.5 * z(loud), -2.5, 2.5)
    return [pretty_midi.Note(velocity=int(np.clip(round(base + spread * v), lo, hi)), pitch=n.pitch,
                             start=n.start, end=n.end) for n, v in zip(notes, zz)]


# ---------------------------------------------------------------- 4) 서스테인 페달 (CC64)
def make_pedal(spans, beats, total_dur, min_len=0.25, lift=0.03, merge_beats=1.0):
    """화성(코드 구간)이 바뀔 때마다 페달을 뗐다 다시 밟는다(레가토 페달).
    짧은 코드 구간(merge_beats박 미만)은 이웃과 합쳐 페달이 떨리지 않게 한다. -> [(밟은 시각, 뗀 시각), ...]"""
    bt = np.asarray(beats, float)
    if not spans or len(bt) < 4:
        return []
    T = lambda b: float(bt[min(int(b), len(bt) - 1)])
    segs = []
    for s, e, r, k in spans:
        if segs and (e - s) < merge_beats:
            segs[-1][1] = e
        else:
            segs.append([s, e])
    out = []
    for s, e in segs:
        t0, t1 = T(s) + 0.015, min(T(e), total_dur) - lift
        if t1 - t0 >= min_len:
            out.append((t0, t1))
    return out


def pedal_events(pedal):
    """[(down, up)] -> [(시각, CC64 값)] 시간순."""
    ev = []
    for d, u in pedal:
        ev += [(d, 127), (u, 0)]
    return sorted(ev)


# ---------------------------------------------------------------- 5) 격자(퀀타이즈)
_GRID_NAMES = {(2,): "8분음표", (3,): "3연음", (4,): "16분음표", (4, 3): "16분+3연음 혼합", (8,): "32분음표"}


def choose_grid(times, beats, k0=0, tol=0.028):
    """박 사이 음의 위치(박 안에서의 소수 위치)를 세어 격자를 고른다. 박 바로 위의 음은 어떤 격자에나 맞으므로 제외하고,
    서로 겹치지 않는 위치끼리 비교한다: 16분음표 위치(1/4, 3/4) · 8분 위치(1/2) · 3연음 위치(1/3, 2/3).
    반환: (격자 분할 튜플, 통계). 예: (2,)=8분, (4,)=16분, (3,)=3연음, (4, 3)=16분+3연음 혼합."""
    bt = np.asarray(beats, float)
    idx = np.arange(len(bt), dtype=float)
    ts = np.asarray([t for t in times if bt[0] + 0.1 <= t <= bt[-1] - 0.1], float)
    if len(ts) < 30:
        return (4,), {}
    pos = np.interp(ts, bt, idx)
    spb = np.interp(ts, bt[:-1], np.diff(bt)) if len(bt) > 2 else np.full(len(ts), 0.5)
    frac = pos - np.floor(pos)
    tolb = tol / spb                                              # 허용 오차(박 단위)
    near = lambda c: np.abs(frac - c) <= tolb
    on_beat = near(0.0) | near(1.0)
    off = ~on_beat
    nS = int(np.sum(off & (near(0.25) | near(0.75))))             # 16분 위치
    nH = int(np.sum(off & near(0.5)))                              # 8분(엇박) 위치
    nT = int(np.sum(off & (near(1 / 3) | near(2 / 3))))           # 3연음 위치
    st = dict(on_beat=int(on_beat.sum()), n16=nS, n8=nH, n3=nT, total=len(ts))
    if nT >= 25 and nT > 1.5 * nS:
        return ((4, 3) if nS > 0.4 * nT else (3,)), st
    if nS >= 0.15 * max(nS + nH, 1) and nS >= 20:
        return (4,), st
    return ((2,) if nH >= 20 else (4,)), st


def snap_grid(notes, beats, k0, divs=(4,), strength=1.0, shift=0):
    """박 위치를 divs(박당 분할 수들의 합집합) 격자의 가장 가까운 선으로 옮긴다. strength 0~1: 0=그대로, 1=완전 정렬.
    반환: 박 위치 기준 (start, end)를 가진 음 리스트(단위=박)."""
    bt = np.asarray(beats, float)
    idx = np.arange(len(bt), dtype=float) - k0

    def pos(t):
        if t <= bt[0]:
            return idx[0] + (t - bt[0]) / (bt[1] - bt[0])
        if t >= bt[-1]:
            return idx[-1] + (t - bt[-1]) / (bt[-1] - bt[-2])
        return float(np.interp(t, bt, idx))

    def near(p):
        best = min((round(p * d) / d for d in divs), key=lambda g: abs(g - p))
        return p + strength * (best - p)
    out = []
    mind = 1.0 / max(divs)
    for n in notes:
        s = near(pos(n.start) + shift)
        e = max(near(pos(n.end) + shift), s + (mind if strength >= 0.5 else 0.05))
        out.append((s, e, n))
    return out


def estimate_phase(times, beats, div=4, min_r=0.3, max_ms=60):
    """음 시작들이 박 격자선에서 일정하게 벗어나 있는 양(박 단위)을 원형 평균으로 구한다.
    박 추적 결과가 실제 소리보다 수십 ms 앞서는 경우가 많아서, 격자를 이만큼 옮기면 음이 격자에 맞는 비율이 크게 오른다.
    음들이 격자에 모여 있지 않으면(집중도 < min_r) 0을 돌려준다. 반환: (phase_beats, 집중도)"""
    bt = np.asarray(beats, float)
    ts = np.asarray([t for t in times if bt[0] + 0.1 <= t <= bt[-1] - 0.1], float)
    if len(ts) < 40:
        return 0.0, 0.0
    pos = np.interp(ts, bt, np.arange(len(bt), dtype=float))
    z = np.exp(2j * np.pi * div * pos).mean()
    r, ph = float(abs(z)), float(np.angle(z) / (2 * np.pi * div))
    spb = float(np.median(np.diff(bt)))
    if r < min_r or abs(ph) * spb * 1000 > max_ms:
        return 0.0, r
    return ph, r


def quantize_real_time(notes, beats, k0, divs=(4,), strength=1.0, phase=0.0):
    """실제 시간축에서 음의 시작/끝을 박 격자선 쪽으로 strength(0~1)만큼 옮긴다.
    격자선은 박 추적 결과(박마다 달라지는 템포)를 따라가므로 원곡과 붙은 채로 정렬된다."""
    if strength <= 0 or not notes:
        return list(notes)
    bt = np.asarray(beats, float)
    idx = np.arange(len(bt), dtype=float)

    def to_pos(t):
        if t <= bt[0]:
            return (t - bt[0]) / (bt[1] - bt[0])
        if t >= bt[-1]:
            return (len(bt) - 1) + (t - bt[-1]) / (bt[-1] - bt[-2])
        return float(np.interp(t, bt, idx))

    def to_time(p):
        if p <= 0:
            return bt[0] + p * (bt[1] - bt[0])
        if p >= len(bt) - 1:
            return bt[-1] + (p - (len(bt) - 1)) * (bt[-1] - bt[-2])
        return float(np.interp(p, idx, bt))

    def near(p):
        g = min((round(p * d) / d for d in divs), key=lambda q: abs(q - p))
        return p + strength * (g - p)
    out = []
    for n in notes:
        s = to_time(near(to_pos(n.start) - phase) + phase)
        e = max(to_time(near(to_pos(n.end) - phase) + phase), s + 0.03)
        out.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=max(0.0, s), end=max(e, s + 0.03)))
    return out


def legato(notes, max_gap=0.15, overlap=0.0):
    """단선율 층: 다음 음이 max_gap초 안에 시작하면 지금 음의 끝을 다음 음 시작까지 이어 준다(피아노 레가토).
    소리가 끝난 뒤에도 다음 음까지 이어 치는 건 서스테인 페달(CC64)이 받쳐 주는 전제다."""
    ns = sorted(notes, key=lambda n: n.start)
    out = []
    for i, n in enumerate(ns):
        end = n.end
        if i + 1 < len(ns):
            gap = ns[i + 1].start - n.end
            if 0 < gap <= max_gap:
                end = ns[i + 1].start + overlap
        out.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=n.start, end=end))
    return out
