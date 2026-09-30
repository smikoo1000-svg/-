#!/usr/bin/env python3
"""가수곡 -> 피아노 편곡 MIDI + 악보(MusicXML).

1) Demucs(htdemucs)로 vocals / bass / other(반주) / drums 분리
2) Basic Pitch(Spotify)로 각 스템을 음표로 전사
3) 파트별 정리(보컬=단선율, 반주=동시음 제한, 베이스=단선율) 후 피아노 한 대로 합침
4) music21로 16분음표 정렬 + 그랜드 스태프 MusicXML 출력

사용:  python song2piano.py song.mp3 -o out/
"""
import argparse, hashlib, os, shutil, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
from pathlib import Path

import pretty_midi
from basic_pitch.inference import predict
import enhance as H

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


_ONS = {}


def stem_onsets(wav: Path, delta=0.1):
    """스템의 (온셋 시각들, 온셋 세기 포락선, 세기 80퍼센타일). 같은 파일이면 재사용."""
    import librosa
    k = (str(wav), delta)
    if k not in _ONS:
        y, sr = librosa.load(str(wav), sr=22050, mono=True)
        env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
        on = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=512, units="time", delta=delta)
        _ONS[k] = (on, env, float(np.percentile(env, 80)), y, sr)
    return _ONS[k]


def merge_same_pitch(notes, gap, onsets=None):
    """같은 높이의 음이 gap초 이내로 끊겼다 다시 시작하면 하나로 이어 붙인다.
    단, 두 번째 음의 시작 지점에 원곡의 실제 소리 시작(온셋)이 있으면 '다시 발음한 음'이라 합치지 않는다."""
    on = np.asarray(onsets if onsets is not None else [], float)
    out = []
    for n in sorted(notes, key=lambda n: (n.pitch, n.start)):
        reattack = bool(len(on)) and out and out[-1].pitch == n.pitch and n.start - out[-1].start > 0.08 \
            and np.min(np.abs(on - n.start)) <= 0.05
        if out and out[-1].pitch == n.pitch and n.start - out[-1].end <= gap and not reattack:
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


def fill_missing(notes, wav: Path, lo, hi, default, dur_max=0.3, voiced=False):
    """원곡에서 강한 소리가 시작하는데 음이 없는 지점에 음을 채운다.
    음이름은 스템 크로마 최대값, 옥타브는 주변 음에 가장 가까운 곳. 음높이가 불분명한 소리(타악·잡음)는 제외."""
    import librosa
    on, env, thr, y, sr = stem_onsets(wav)
    hop = 512
    t = np.arange(len(env)) * hop / sr
    st = np.array(sorted(n.start for n in notes)) if notes else np.array([])
    ch = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rthr = 0.15 * float(np.percentile(rms, 95))
    strong = [o for o in on if float(np.interp(o, t, env)) >= thr]
    added = []
    for i, o in enumerate(strong):
        if len(st) and np.min(np.abs(st - o)) <= 0.06:
            continue
        fr = min(int(o * sr / hop), ch.shape[1] - 1)
        fe = min(fr + max(2, int(0.12 * sr / hop)), ch.shape[1])
        c = ch[:, fr:fe].mean(axis=1)
        if c.max() < 0.22 * c.sum():                      # 음높이가 분명하지 않은 소리는 건너뜀
            continue
        if voiced and rms[min(fr, len(rms) - 1)] < rthr:
            continue
        pc = int(np.argmax(c))
        near = [n.pitch for n in notes if abs(n.start - o) <= 2.0]
        ref = float(np.median(near)) if near else default
        cands = [p for p in range(lo, hi + 1) if p % 12 == pc]
        if not cands:
            continue
        p = min(cands, key=lambda x: abs(x - ref))
        nxt = strong[i + 1] if i + 1 < len(strong) else o + dur_max
        added.append(pretty_midi.Note(velocity=64, pitch=int(p), start=float(o),
                                      end=float(max(o + 0.06, min(nxt, o + dur_max)))))
    return sorted(list(notes) + added, key=lambda n: n.start)


def refine_onsets(notes, wav: Path, win=0.07):
    """Basic Pitch가 잡은 음 시작 시각은 ±50ms쯤 흔들린다. 해당 스템에서 실제로 소리가 시작하는
    지점(온셋)을 찾아 win초 안이면 그 위치로 옮겨서 16분 칸 정렬이 밀리지 않게 한다."""
    import librosa
    y, sr = librosa.load(str(wav), sr=22050, mono=True)
    ref = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    if len(ref) == 0:
        return notes
    out = []
    for n in notes:
        t = float(ref[np.argmin(np.abs(ref - n.start))])
        s0 = t if abs(t - n.start) <= win and t < n.end - 0.02 else n.start
        out.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=s0, end=n.end))
    return sorted(out, key=lambda n: n.start)


def chroma_correct(notes, wav: Path, margin=1.5):
    """Basic Pitch 음의 음이름이 보컬 스템 크로마와 크게 어긋나면(크로마 최대 음이름의 세기가
    현재 음이름의 margin배 이상) 가장 가까운 옥타브의 크로마 최대 음이름으로 바꾼다."""
    import librosa
    sr, hop = 22050, 512
    y = librosa.load(str(wav), sr=sr, mono=True)[0]
    ch = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    out, changed = [], 0
    for n in notes:
        a = int(n.start * sr / hop)
        b = max(a + 2, int(n.end * sr / hop))
        if b >= ch.shape[1]:
            out.append(n); continue
        c = ch[:, a:b].mean(axis=1)
        best = int(np.argmax(c)); cur = n.pitch % 12
        if best != cur and c[best] >= margin * max(c[cur], 1e-6):
            p = n.pitch + ((best - cur + 6) % 12 - 6)          # 가장 가까운 음으로
            out.append(pretty_midi.Note(velocity=n.velocity, pitch=int(p), start=n.start, end=n.end)); changed += 1
        else:
            out.append(n)
    return out


def fix_octave_outliers(notes):
    """단선율에서 앞뒤 음과 8반음 이상 떨어진 채 혼자 튀는 음은 옥타브 오인식일 가능성이 크다.
    앞뒤가 가까운데(5반음 이내) 가운데만 튀면 옥타브를 옮겨 가장 가까운 곳으로 되돌린다."""
    ns = sorted(notes, key=lambda n: n.start)
    fixed = 0
    for i in range(1, len(ns) - 1):
        p, c, n = ns[i - 1].pitch, ns[i].pitch, ns[i + 1].pitch
        if abs(c - p) >= 8 and abs(c - n) >= 8 and abs(p - n) <= 5:
            target = (p + n) / 2
            cand = min((c - 12, c + 12), key=lambda x: abs(x - target))
            if abs(cand - target) <= 6:
                ns[i].pitch, fixed = cand, fixed + 1
    return ns


def lowline(stems, lo=None, hi=None, maxdur=0.46):
    """낮은 음 파트(리듬 위주의 저음 라인). 소리가 시작하는 지점은 반주 스템에서, 음이름은 베이스 스템의
    크로마(가장 센 음이름)에서 잡아 lo~hi 음역에 놓는다. Basic Pitch로 베이스를 전사하는 것보다
    정답 편곡과의 일치도가 훨씬 높았다."""
    import librosa
    lo = int(os.environ.get("LOW_LO", 48)) if lo is None else lo
    sr, hop = 22050, 512
    yb = librosa.load(str(stems["bass"]), sr=sr, mono=True)[0]
    yo = librosa.load(str(stems["other"]), sr=sr, mono=True)[0]
    env = librosa.onset.onset_strength(y=yo, sr=sr)
    on = librosa.onset.onset_detect(onset_envelope=env, sr=sr, units="time", delta=0.07)
    if not os.environ.get("NO_FILL"):                    # 베이스 스템에서만 시작하는 소리도 포함
        envb = librosa.onset.onset_strength(y=yb, sr=sr)
        onb = librosa.onset.onset_detect(onset_envelope=envb, sr=sr, units="time", delta=0.1)
        merged = np.sort(np.concatenate([on, onb]))
        on = np.array([merged[0]] + [t for p, t in zip(merged, merged[1:]) if t - p > 0.05]) if len(merged) else merged
    if len(on) == 0:
        return []
    ch = librosa.feature.chroma_cqt(y=yb, sr=sr, hop_length=hop)
    out = []
    for i, t in enumerate(on):
        a = int(t * sr / hop)
        b = a + max(2, int(0.12 * sr / hop))
        if b >= ch.shape[1]:
            continue
        pc = int(np.argmax(ch[:, a:b].mean(axis=1)))
        end = min(on[i + 1] if i + 1 < len(on) else t + maxdur, t + maxdur)
        out.append(pretty_midi.Note(velocity=70, pitch=lo + ((pc - lo) % 12), start=float(t),
                                    end=float(max(end, t + 0.05))))
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


def unify_beat_level(beats):
    """박 추적기가 구간마다 4분음표/8분음표 단위를 오락가락하면(예: 97↔200 BPM)
    시간 축이 뒤틀린다. 기준 간격(80~160 BPM 범위로 접은 중앙값)에 맞춰 촘촘한 구간의 박을 솎아 낸다."""
    beats = np.asarray(beats, float)
    if len(beats) < 8:
        return beats
    ref = float(np.median(np.diff(beats)))
    while 60 / ref > float(os.environ.get('BPM_MAX', 160)):
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


def raw_beats(mix: Path, tracker="beat_this"):
    """박 추적기의 원본 출력(박, 다운비트). 접기/보정 전 값을 캐시에 저장해 둔다."""
    f = mix.with_name(f"rawbeats_{tracker}.npz")
    if f.exists():
        z = np.load(f)
        return z["beats"], z["downs"]
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
    np.savez(f, beats=beats, downs=downs)
    return beats, downs


def beat_grid(mix: Path, bpm_override=None, tracker="beat_this"):
    """박/첫박(다운비트) 추적. 1순위 Beat This!, 실패하면 librosa. -> (bpm, beats, k0)
    k0 = 첫 마디의 첫 박이 beats 배열의 몇 번째인지."""
    beats, downs = raw_beats(mix, tracker)
    beats = unify_beat_level(beats)
    bpm = 60 * (len(beats) - 1) / float(beats[-1] - beats[0])   # 평균 템포(곡 끝까지 어긋나지 않게)
    while bpm < 80 and len(beats) > 1:               # 너무 느리면 박을 반으로 쪼갬
        beats = np.sort(np.concatenate([beats, (beats[:-1] + beats[1:]) / 2])); bpm *= 2
    while bpm > float(os.environ.get('BPM_MAX', 160)):                                  # 너무 빠르면 박을 2개씩 묶음
        beats = beats[::2]; bpm /= 2
    if bpm_override:                                  # 사용자가 준 BPM: 첫 박부터 균일 격자
        bpm = float(bpm_override)
        beats = beats[0] + np.arange(0, beats[-1] - beats[0] + 60 / bpm, 60 / bpm)
    k0 = int(np.argmin(np.abs(beats - downs[0]))) if len(downs) else 0
    return round(bpm, 2), beats, k0        # MIDI에 저장되는 값과 동일하게 맞춰 격자 오차 방지


def cached_beat_grid(stems, a):
    return beat_grid(stems["mix"], a.bpm, a.tracker)


def grid_phase(stems, bpm, beats, k0):
    """박 추적 격자선과 실제 소리 시작 사이의 평균 어긋남(초). 재생용 MIDI를 원곡에 맞추는 데 쓴다."""
    import librosa
    ons = []
    for k in ("vocals", "other"):
        y, sr = librosa.load(str(stems[k]), sr=22050, mono=True)
        ons.append(librosa.onset.onset_detect(y=y, sr=sr, units="time"))
    on = np.sort(np.concatenate(ons))
    on = on[(on >= beats[0]) & (on <= beats[-1])]
    if len(on) < 30:
        return 0.0
    idx = np.arange(len(beats)) - k0
    sl = 60 / bpm / 4
    dev = np.array([((float(np.interp(t, beats, idx)) * 4 + .5) % 1 - .5) * sl for t in on])
    return float(np.clip(np.median(dev), -0.05, 0.05))


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
    divs = tuple(div) if isinstance(div, (tuple, list)) else (div,)
    near = lambda p: min((round(p * d) / d for d in divs), key=lambda g: abs(g - p))
    out = []
    for s0, e0, n in raw:
        s = near(s0 + shift)
        e = max(near(e0 + shift), s + 1 / max(divs))
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


def chord_notes(spans, beats, lo=48, hi=64):
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
        von = stem_onsets(stems["vocals"])[0]
        v = monophonic(merge_same_pitch(in_range(transcribe(stems["vocals"],
                          minimum_note_length=float(os.environ.get("BP_MINLEN", 80)),
                          onset_threshold=float(os.environ.get("BP_ONSET", 0.35)),
                          frame_threshold=float(os.environ.get("BP_FRAME", 0.2)), melodia_trick=True,
                          minimum_frequency=100, maximum_frequency=1200), 48, 84), 0.12, von), a.min_len)
        v = fix_octave_outliers(v)
        if os.environ.get("CHROMA_FIX"):
            v = chroma_correct(v, stems["vocals"], float(os.environ["CHROMA_FIX"]))
        v = refine_onsets(v, stems["vocals"])
        v = fill_missing(v, stems["vocals"], 48, 84, 67, voiced=True) if not os.environ.get("NO_FILL") else v
        if a.offsets:
            v = H.refine_offsets(v, stems["vocals"])
        if a.legato:
            v = H.legato(v, a.legato_gap)
        return H.assign_velocity(v, stems["vocals"], base=84) if a.dynamics else v

    def bass():
        b = monophonic(merge_same_pitch(in_range(transcribe(stems["bass"], minimum_note_length=90,
                          onset_threshold=0.4, frame_threshold=0.25, minimum_frequency=35,
                          maximum_frequency=350), 28, 60), 0.15), 0.1)
        return refine_onsets(b, stems["bass"])

    def other():
        return limit_poly(merge_same_pitch(in_range(transcribe(stems["other"], minimum_note_length=100,
                          onset_threshold=0.45, frame_threshold=0.3), 48, 88), 0.2),
                          a.accomp_poly, 0.2, 30)

    def lead():
        """반주 스템의 눈에 띄는 리드 선율(신스/기타/카우벨 등). 보컬과 겹치는 음은 뺀다."""
        oon = stem_onsets(stems["other"])[0]
        l = monophonic(merge_same_pitch(in_range(transcribe(stems["other"], minimum_note_length=80,
                          onset_threshold=0.4, frame_threshold=0.25), 60, 96), 0.1, oon), 0.08)
        l = refine_onsets(fix_octave_outliers(l), stems["other"])
        l = fill_missing(l, stems["other"], 60, 96, 72) if not os.environ.get("NO_FILL") else l
        if a.offsets:
            l = H.refine_offsets(l, stems["other"])
        if a.legato:
            l = H.legato(l, a.legato_gap)
        return H.assign_velocity(l, stems["other"], base=74) if a.dynamics else l

    def low():
        b = lowline(stems)
        if a.offsets:
            b = H.refine_offsets(b, stems["bass"], pitch_class=True)
        return H.assign_velocity(b, stems["bass"], base=70) if a.dynamics else b

    def harm_raw():
        """반주 스템의 다성음(화음) 후보. 배음 오인식을 지우고 코드 구성음 위주로 남기는 건 코드 인식 뒤에 한다."""
        return in_range(transcribe(stems["other"], minimum_note_length=90, onset_threshold=0.4,
                                   frame_threshold=0.25), 40, 84)

    def chords():
        spans = recognize_chords(stems, beats)
        print("chords:", " ".join(f"{_NAMES[r]}{k}" for _, _, r, k in spans[:16]), "…")
        return spans

    stage("보컬·베이스·코드 분석 중 (동시에 처리)")
    tasks = {"vocals": vocals, "bass": bass}
    if a.accomp == "octave":
        tasks = {"vocals": vocals, "lead": lead, "bass": low}                      # 베이스 파트 = 저음 리듬 라인
        if os.environ.get("OCT_CHORDS"):
            tasks["chords"] = chords
    elif a.accomp == "notes":
        tasks["other"] = other
    elif a.accomp in ("chords", "lead"):
        tasks["chords"] = chords
        if a.accomp == "lead":
            tasks["lead"] = lead
    if "chords" not in tasks:
        tasks["_spans"] = chords                          # 페달(CC64)과 하모니 정리에 쓰는 코드 구간
    if a.harmony and a.accomp == "octave":
        tasks["_harm"] = harm_raw
    with ThreadPoolExecutor(len(tasks)) as ex:
        futs = {k: ex.submit(f) for k, f in tasks.items()}
        parts = {k: f.result() for k, f in futs.items()}
    spans = parts["chords"] if "chords" in parts else parts.pop("_spans", None)
    harm_notes = parts.pop("_harm", None)
    if "chords" in parts:                                 # 코드 음역을 멜로디 바로 아래로 (뭉개짐 방지)
        mel = [n.pitch for k in ("vocals", "lead") for n in parts.get(k, [])]
        hi = int(np.clip(np.percentile(mel, 25) - 2, 55, 66)) if mel else 64
        parts["chords"] = chord_notes(parts["chords"], beats, lo=hi - 14, hi=hi)
    if a.accomp == "octave":
        # 정답 편곡(사람이 만든 하프 솔로)과 비교해 찾은 스타일: 멜로디를 한 옥타브 위로 겹쳐 치고,
        # 베이스는 너무 낮지 않게 48~60 음역으로 접고, 지속 코드 층은 넣지 않는다.
        v = parts["vocals"]
        lead_ = [n for n in parts["lead"] if n.velocity >= 40 and not any(
            x.start < n.end and n.start < x.end and x.pitch % 12 == n.pitch % 12 for x in v)]
        melody = sorted(v + lead_, key=lambda n: n.start)
        _ms = int(os.environ.get("MEL_SHIFT", 0))
        if _ms:
            melody = [pretty_midi.Note(velocity=n.velocity, pitch=int(min(max(n.pitch + _ms, 28), 100)),
                                       start=n.start, end=n.end) for n in melody]
        parts["vocals"] = melody
        parts["lead"] = [pretty_midi.Note(velocity=int(max(20, n.velocity * 0.9)), pitch=min(n.pitch + 12, 100),
                                          start=n.start, end=n.end) for n in melody]
    if a.accomp == "octave" and parts.get("bass"):
        # 저음 라인은 마디마다 반복되는 리듬 패턴이므로, 드물게만 나타나는 위치의 소리 시작은 잡음으로 보고 뺀다
        idx_ = np.arange(len(beats)) - k0
        slot = lambda n: int(round(float(np.interp(n.start, beats, idx_)) * 4)) % 16
        h = np.bincount([slot(n) for n in parts["bass"]], minlength=16)
        parts["bass"] = [n for n in parts["bass"] if h[slot(n)] >= 0.3 * h.max()]
    if "lead" in parts and a.accomp != "octave":          # 보컬과 같은 순간·같은 음이면 중복이라 제거
        v = parts["vocals"]
        parts["lead"] = [n for n in parts["lead"] if n.velocity >= 40 and not any(
            x.start < n.end and n.start < x.end and x.pitch % 12 == n.pitch % 12 for x in v)]
    if "chords" in parts:                                 # 멜로디/베이스와 같은 높이를 동시에 치는 코드 음은 뺀다
        others = [n for k, ns_ in parts.items() if k != "chords" for n in ns_]
        parts["chords"] = [c for c in parts["chords"] if not any(
            o.pitch == c.pitch and o.start < c.end and c.start < o.end for o in others)]
    if harm_notes is not None and spans is not None:       # 다성음(화음): 배음 제거 + 코드 구성음 필터 + 중복 제거
        hn = H.chord_filter(H.suppress_overtones(harm_notes), spans, beats)
        hn = limit_poly(merge_same_pitch(hn, 0.1, stem_onsets(stems["other"])[0]), 3, 0.12, 25)
        hn = refine_onsets(hn, stems["other"])
        if a.offsets:
            hn = H.refine_offsets(hn, stems["other"])
        if a.dynamics:
            hn = [pretty_midi.Note(velocity=int(max(20, n.velocity * 0.78)), pitch=n.pitch, start=n.start, end=n.end)
                  for n in H.assign_velocity(hn, stems["other"], base=70)]
        others = [n for k, ns_ in parts.items() for n in ns_]
        parts["harm"] = [h for h in hn if not any(o.pitch == h.pitch and o.start < h.end and h.start < o.end
                                                  for o in others)]
    allraw = [n for ns in parts.values() for n in ns]
    # --- 격자(퀀타이즈) 선택: 자동이면 음 위치를 분석해서 8분/16분/3연음/혼합 중에서 고른다
    names = {"8": (2,), "16": (4,), "3": (3,), "mixed": (4, 3), "32": (8,)}
    if a.grid == "auto":
        divs, gst = H.choose_grid([n.start for n in allraw], beats, k0)
    else:
        divs, gst = names.get(a.grid, (4,)), {}
    print("격자:", H._GRID_NAMES.get(tuple(divs), divs), gst)
    shift = None
    if allraw:
        first = min(np.interp(n.start, beats, np.arange(len(beats)) - k0) if beats[0] <= n.start <= beats[-1]
                    else (n.start - beats[0]) / (60 / bpm) - k0 for n in allraw)
        shift = 4 * int(np.ceil(max(0, -first) / 4))
    for name, ns in parts.items():
        for n in snap(ns, bpm, beats, k0, div=divs, shift=shift):
            n.velocity = min(127, max(1, n.velocity))
            piano.notes.append(n)
        print(f"{name}: {len(ns)} notes")
    pm.instruments.append(piano)
    raw_notes = [pretty_midi.Note(velocity=min(127, max(1, n.velocity)), pitch=n.pitch, start=n.start, end=n.end)
                 for ns in parts.values() for n in ns]
    # 격자판(재생용): 실제 시간축에서 격자선 쪽으로 strength만큼 이동 (1.0=완전 정렬, 0=그대로)
    quant_notes = H.quantize_real_time(raw_notes, beats, k0, divs, 0.0 if a.grid == "off" else a.quantize_strength)
    # --- 서스테인 페달(CC64): 화성이 바뀌는 지점마다 밟았다 뗌
    pedal = H.make_pedal(spans, beats, float(beats[-1])) if (a.pedal and spans) else []
    vel = np.array([n.velocity for n in raw_notes]) if raw_notes else np.array([0])
    ivs = np.diff(beats)
    ok = float(np.mean(np.abs(ivs / np.median(ivs) - 1) <= 0.04)) if len(ivs) else 0.0
    covered = sum(u - d for d, u in pedal)
    info = dict(
        bpm=float(bpm), bpm_confidence=round(ok, 3),
        tempo_range=[round(float(60 / np.quantile(ivs, .95)), 1), round(float(60 / np.quantile(ivs, .05)), 1)],
        beats=int(len(beats)), first_downbeat_beat=int(k0),
        grid=H._GRID_NAMES.get(tuple(divs), str(divs)), grid_stats=gst, quantize_strength=float(a.quantize_strength),
        layers={k: len(v) for k, v in parts.items()},
        velocity=dict(min=int(vel.min()), median=int(np.median(vel)), max=int(vel.max()), std=round(float(vel.std()), 1)),
        pedal=dict(enabled=bool(pedal), segments=len(pedal), coverage=round(covered / max(float(beats[-1]), 1e-6), 3),
                   events=[[round(d, 3), round(u, 3)] for d, u in pedal]),
        options=dict(accomp=a.accomp, offsets=a.offsets, dynamics=a.dynamics, harmony=a.harmony, pedal=a.pedal, grid=a.grid))
    return pm, quant_notes, raw_notes, beats, pedal, info


def write_timed_midi(notes, beats, path, tpb=480, pedal=None):
    """음을 원곡의 실제 시각 그대로(16분 칸 정렬 없이) 저장한다. 박 추적 결과를 템포 정보(박마다 템포 변화)로
    함께 써서, 어떤 플레이어로 재생해도 곡 내내 원곡과 박이 붙어 있고, 악보 프로그램에서도 박 줄이 맞는다."""
    import mido
    beats = np.asarray(beats, float)
    ivs = np.diff(beats)
    idx = np.arange(len(beats), dtype=float)

    def pos(t):                                            # 실제 시각 -> 박 위치(박 단위)
        if t < beats[0]:
            return (t - beats[0]) / ivs[0]
        if t >= beats[-1]:
            return (len(beats) - 1) + (t - beats[-1]) / ivs[-1]
        return float(np.interp(t, beats, idx))

    p0 = pos(0.0)                                             # 원곡 0초의 박 위치 (첫 박 앞이면 음수)
    tick = lambda t: max(0, int(round((pos(t) - p0) * tpb)))   # MIDI 0틱 = 원곡 0초
    mid = mido.MidiFile(type=1, ticks_per_beat=tpb)
    tempo_tr = mido.MidiTrack(); mid.tracks.append(tempo_tr)
    ev = [(0, mido.MetaMessage("set_tempo", tempo=int(round(ivs[0] * 1e6))))]
    for j, iv in enumerate(ivs):
        ev.append((max(0, int(round((j - p0) * tpb))), mido.MetaMessage("set_tempo", tempo=int(round(iv * 1e6)))))
    ev.append((0, mido.MetaMessage("time_signature", numerator=4, denominator=4)))
    ev.sort(key=lambda e: e[0])
    last = 0
    for t_, m in ev:
        tempo_tr.append(m.copy(time=t_ - last)); last = t_
    tr = mido.MidiTrack(); mid.tracks.append(tr)
    tr.append(mido.MetaMessage("track_name", name="Piano", time=0))
    tr.append(mido.Message("program_change", program=0, channel=0, time=0))
    evs = []
    for n in notes:
        on_, off_ = tick(n.start), tick(max(n.end, n.start + 0.03))
        off_ = max(off_, on_ + 1)
        v = int(min(127, max(1, n.velocity)))
        evs.append((on_, 1, mido.Message("note_on", note=int(n.pitch), velocity=v, channel=0)))
        evs.append((off_, 0, mido.Message("note_off", note=int(n.pitch), velocity=0, channel=0)))
    for d_, u_ in (pedal or []):                            # 서스테인 페달(CC64): 127=밟음, 0=뗌
        evs.append((tick(d_), 2, mido.Message("control_change", control=64, value=127, channel=0)))
        evs.append((max(tick(u_), tick(d_) + 1), -1, mido.Message("control_change", control=64, value=0, channel=0)))
    evs.sort(key=lambda e: (e[0], e[1]))
    last = 0
    for t_, _, m in evs:
        tr.append(m.copy(time=t_ - last)); last = t_
    mid.save(str(path))


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
    ap.add_argument("--accomp", choices=["octave", "lead", "chords", "notes", "none"], default="octave",
                    help="반주 방식: octave=멜로디 옥타브 겹침(기본), lead=코드+리드 선율, chords=코드만, notes=음 전사(복잡), none=멜로디+베이스만")
    ap.add_argument("--no-accompaniment", action="store_true", help="--accomp none 과 동일")
    ap.add_argument("--tracker", choices=["beat_this", "librosa"], default="beat_this", help="박 추적 방식")
    ap.add_argument("--max-sec", type=float, default=None, help="앞 N초만 변환(빠른 미리보기)")
    ap.add_argument("--bpm", type=float, default=None, help="BPM을 직접 지정(자동 추정이 틀릴 때)")
    B = argparse.BooleanOptionalAction
    ap.add_argument("--offsets", action=B, default=True, help="음이 끝나는 시점을 소리 에너지로 정밀화")
    ap.add_argument("--legato", action=B, default=False, help="멜로디를 다음 음까지 이어 치기(페달과 함께 쓰면 피아노다움)")
    ap.add_argument("--legato-gap", type=float, default=0.15, help="레가토로 이을 최대 틈(초)")
    ap.add_argument("--dynamics", action=B, default=True, help="어택·소리 크기로 음마다 벨로시티 산출")
    ap.add_argument("--harmony", action=B, default=True, help="반주의 화음(다성음) 층을 추가(배음 제거+코드 구성음 필터)")
    ap.add_argument("--pedal", action=B, default=True, help="서스테인 페달(CC64)을 화성 변화에 맞춰 생성")
    ap.add_argument("--grid", choices=["auto", "8", "16", "3", "mixed", "32", "off"], default="auto",
                    help="격자: auto=음 위치로 자동 선택, 8=8분, 16=16분, 3=3연음, mixed=16분+3연음, 32=32분, off=정렬 안 함")
    ap.add_argument("--quantize-strength", type=float, default=1.0, help="격자판 MIDI 정렬 강도 0~1 (1=완전, 0.5=반쯤)")
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
    pm, quant_notes, raw_notes, beats_, pedal, info = build(stems, a, grid)
    stage("MIDI·악보 저장")
    stem_name = a.audio.stem
    mid = a.out / f"{stem_name}_piano.mid"
    write_timed_midi(raw_notes, beats_, mid, pedal=pedal); print("MIDI:", mid)              # 원곡 실제 리듬 + 박마다 템포 + 페달
    write_timed_midi(quant_notes, beats_, a.out / f"{stem_name}_piano_quantized.mid", pedal=pedal)   # 격자 정렬판
    if pedal:                                                                                      # 페달만 따로 (컨트롤러 데이터)
        write_timed_midi([], beats_, a.out / f"{stem_name}_piano_pedal.mid", pedal=pedal)
    (a.out / f"{stem_name}_piano_analysis.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        bars = a.out / f".{a.audio.stem}_bars.mid"                    # 악보용: 마디가 정확히 맞는 버전
        pm.write(str(bars))
        xml = a.out / f"{a.audio.stem}_piano.musicxml"
        to_score(bars, xml, pm.get_tempo_changes()[1][0]); print("악보:", xml)
        bars.unlink(missing_ok=True)
    except Exception as e:
        print("악보 변환 실패(MIDI는 저장됨):", e)


if __name__ == "__main__":
    main()
