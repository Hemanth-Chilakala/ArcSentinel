"""
ArcSentinel - pre-proposal feasibility model (golden reference).

Purpose
-------
Before hardware is available, this script checks that the proposed
detection chain can separate high-impedance faults (HIFs) from the
everyday feeder events that fool simpler detectors, and that it still
works after int8 quantization (the form it will take in PolarFire SoC
fabric via SmartHLS).

Pipeline (mirrors the planned hardware partition)
  1. Waveform generator     : synthetic 11 kV feeder, physics-based HIF
                              (Emanuel two-diode arc model) + 6 confuser
                              classes.            -> later: HIL replay source
  2. Per-cycle features      : DFT harmonics, Haar-wavelet HF energy,
                              half-cycle asymmetry, cycle-to-cycle
                              difference energy, 3rd-harmonic angle.
                                                   -> FABRIC (SmartHLS/CoreFFT)
  3. Window statistics       : mean / std over a 10-cycle sliding window.
                                                   -> FABRIC
  4. Classifier              : 28-32-16-7 MLP, float and int8.
                                                   -> FABRIC (SmartHLS int8)
  5. Security logic          : K consecutive HIF windows -> alarm/trip.
                                                   -> bare-metal RISC-V hart
  Baseline                   : conventional residual overcurrent (51N) with
                              2nd-harmonic inrush restraint.

All data here are SYNTHETIC. The results establish algorithmic
feasibility only; field validation is part of the development plan.

Usage:  python arcsentinel_model.py            (numpy + matplotlib only)
"""

import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------
# System constants
# ----------------------------------------------------------------------------
F_NOM = 50.0                 # Hz (60 Hz supported by changing F_NOM / N)
N = 512                      # samples per nominal cycle
FS = F_NOM * N               # 25.6 kS/s per channel
CYCLES_PER_EVENT = 50        # 1.0 s per event record
WIN = 10                     # cycles per classification window
HOP = 2                      # cycles between windows (40 ms)
K_SECURITY = 5               # consecutive HIF windows needed to declare HIF
V_PK = 11e3 / np.sqrt(3) * np.sqrt(2)   # phase-to-neutral peak, 11 kV feeder
PICKUP_51N = 30.0            # A rms, sensitive residual overcurrent setting
DELAY_51N_CYC = 10           # definite-time delay for 51N baseline
H2_RESTRAINT = 0.15          # 51N blocked while I2/I1 exceeds this (inrush)
# HIF arc current is drawn log-uniformly over this rms range. The low end covers
# vegetation-contact faults, which can draw well under 1 A.
ARC_RMS_RANGE = (0.5, 50.0)

CLASSES = ["normal", "load_switch", "cap_switch", "inrush",
           "nonlinear_load", "welder", "HIF"]
HIF = CLASSES.index("HIF")
NORMAL = CLASSES.index("normal")


# ----------------------------------------------------------------------------
# 1. Waveform generator
# ----------------------------------------------------------------------------
def _base_feeder(rng, n_samp):
    """Healthy feeder: phase-A voltage, phase-A current, residual current."""
    f = F_NOM + rng.uniform(-0.1, 0.1)          # small frequency drift
    t = np.arange(n_samp) / FS
    w = 2 * np.pi * f
    ph_v = rng.uniform(0, 2 * np.pi)

    va = V_PK * np.sin(w * t + ph_v)
    va += V_PK * rng.uniform(0, 0.02) * np.sin(5 * (w * t + ph_v))   # 5th harm.

    i_load = rng.uniform(40, 300) * np.sqrt(2)
    pf_ang = rng.uniform(0.1, 0.5)
    # slow random load variation (+/- a few %)
    drift = 1 + np.cumsum(rng.normal(0, 2e-4, n_samp))
    drift = np.clip(drift, 0.9, 1.1)
    ia = i_load * drift * np.sin(w * t + ph_v - pf_ang)
    ia += i_load * rng.uniform(0, 0.05) * np.sin(5 * (w * t + ph_v) + rng.uniform(0, 6.28))

    # residual current: unbalance + triplen harmonics from single-phase loads
    i0_1 = rng.uniform(1, 12) * np.sqrt(2)
    ph0 = rng.uniform(0, 2 * np.pi)
    i0 = i0_1 * drift * np.sin(w * t + ph0)
    i0 += i0_1 * rng.uniform(0, 0.3) * np.sin(3 * (w * t) + rng.uniform(0, 6.28))

    return t, w, ph_v, va, ia, i0


def _damped(t, t0, f, tau, amp, rng):
    x = np.zeros_like(t)
    m = t >= t0
    x[m] = amp * np.exp(-(t[m] - t0) / tau) * np.sin(2 * np.pi * f * (t[m] - t0) + rng.uniform(0, 6.28))
    return x


def generate_event(cls, rng):
    """Return (va, ia, i0, active_mask, arc_rms) for one event of class `cls`.
    active_mask marks samples where the labelled phenomenon is present;
    arc_rms is the rms HIF arc current after onset (0 for other classes)."""
    n_samp = CYCLES_PER_EVENT * N
    t, w, ph_v, va, ia, i0 = _base_feeder(rng, n_samp)
    active = np.zeros(n_samp, bool)
    t_on = rng.uniform(0.1, 0.35)                 # event onset (s)
    k_on = int(t_on * FS)
    name = CLASSES[cls]
    arc_rms = 0.0

    if name == "normal":
        pass

    elif name == "load_switch":
        step = rng.uniform(-0.5, 0.8)
        i_step = np.abs(ia).max() / 1.1 * step
        s = np.where(t >= t_on, 1.0, 0.0)
        ia += i_step * s * np.sin(w * t + ph_v - rng.uniform(0.1, 0.6))
        i0 += rng.uniform(1, 15) * np.sqrt(2) * s * np.sin(w * t + rng.uniform(0, 6.28))
        active[k_on:k_on + 3 * N] = True

    elif name == "cap_switch":
        f_osc = rng.uniform(300, 1500)
        tau = rng.uniform(0.004, 0.03)
        amp = rng.uniform(50, 400)
        ia += _damped(t, t_on, f_osc, tau, amp, rng)
        i0 += _damped(t, t_on, f_osc, tau, amp * rng.uniform(0.05, 0.3), rng)
        va += _damped(t, t_on, f_osc, tau, V_PK * rng.uniform(0.05, 0.3), rng)
        active[k_on:k_on + int(5 * tau * FS) + N] = True

    elif name == "inrush":
        # unipolar, decaying, 2nd-harmonic-rich magnetising current
        i_pk = rng.uniform(100, 800)
        tau = rng.uniform(0.1, 0.6)
        m = t >= t_on
        env = np.zeros_like(t)
        env[m] = i_pk * np.exp(-(t[m] - t_on) / tau)
        sat = np.clip(np.sin(w * t + ph_v + rng.uniform(-0.5, 0.5)), 0, None) ** 4
        inr = env * sat
        ia += inr
        i0 += inr * rng.uniform(0.3, 1.0)
        active[m & (env > 0.1 * i_pk)] = True

    elif name == "nonlinear_load":
        # rectifier-type loads: strong, STABLE odd harmonics, 3rd in neutral
        s = np.where(t >= t_on, 1.0, 0.0)
        for h, lo, hi in [(3, 2, 12), (5, 1, 8), (7, 0.5, 5), (9, 0.5, 3)]:
            i0 += s * rng.uniform(lo, hi) * np.sqrt(2) * np.sin(h * (w * t + ph_v) + rng.uniform(0, 6.28))
        for h in (5, 7, 11, 13):
            ia += s * np.abs(ia).max() * rng.uniform(0.02, 0.15) / h * 5 * np.sin(h * (w * t + ph_v) + rng.uniform(0, 6.28))
        active[k_on:] = True

    elif name == "welder":
        # intermittent single-phase arc-type load behind a transformer:
        # symmetric clamp, on/off duty cycles, larger & regular current
        i_w = rng.uniform(10, 45) * np.sqrt(2)
        v_clamp = rng.uniform(0.1, 0.35)
        v_norm = np.sin(w * t + ph_v)
        shape = np.sign(v_norm) * np.clip(np.abs(v_norm) - v_clamp, 0, None) / (1 - v_clamp)
        period = rng.uniform(0.12, 0.4)
        duty = rng.uniform(0.4, 0.8)
        on = ((t - t_on) % period) < duty * period
        on &= t >= t_on
        amp = i_w * (1 + 0.05 * rng.standard_normal(CYCLES_PER_EVENT).repeat(N))
        weld = shape * amp * on
        ia += weld
        i0 += weld
        active[t >= t_on] = True

    elif name == "HIF":
        # Emanuel model: arc conducts when |v| exceeds a (random, asymmetric)
        # breakdown voltage; resistance varies cycle to cycle (randomness),
        # build-up over several cycles, intermittent extinction, HF bursts
        # at re-ignition.
        i_target = np.exp(rng.uniform(*np.log(ARC_RMS_RANGE))) * np.sqrt(2)   # peak
        vp_frac = rng.uniform(0.15, 0.55)
        asym = rng.uniform(0.03, 0.25) * rng.choice([-1, 1])
        vn_frac = np.clip(vp_frac * (1 + asym), 0.05, 0.8)
        vp, vn = vp_frac * V_PK, vn_frac * V_PK
        n_cyc = CYCLES_PER_EVENT
        r_var = rng.uniform(0.05, 0.35)
        rp = (V_PK - vp) / i_target * (1 + r_var * rng.standard_normal(n_cyc)).clip(0.3)
        rn = (V_PK - vn) / i_target * (1 + r_var * rng.standard_normal(n_cyc)).clip(0.3)
        p_on = rng.uniform(0.6, 1.0)
        arc_on = rng.random(n_cyc) < p_on
        buildup = np.clip(np.arange(n_cyc) / rng.uniform(1, 8), 0, 1)

        i_arc = np.zeros(n_samp)
        cyc = np.minimum(np.arange(n_samp) // N, n_cyc - 1)
        pos = va > vp
        neg = va < -vn
        i_arc[pos] = (va[pos] - vp) / rp[cyc[pos]]
        i_arc[neg] = (va[neg] + vn) / rn[cyc[neg]]
        i_arc *= (arc_on * buildup)[cyc]
        i_arc[t < t_on] = 0.0

        # HF burst at each conduction start (arc re-ignition)
        starts = np.flatnonzero(np.diff((np.abs(i_arc) > 0).astype(int)) == 1)
        f_hf = rng.uniform(2e3, 8e3)
        L = int(0.001 * FS)
        burst = np.exp(-np.arange(L) / (0.0002 * FS)) * np.sin(2 * np.pi * f_hf * np.arange(L) / FS)
        hf_amp = rng.uniform(0.05, 0.2) * i_target
        for s0 in starts:
            e = min(n_samp, s0 + L)
            i_arc[s0:e] += hf_amp * burst[: e - s0] * rng.uniform(0.5, 1.5)

        ia += i_arc
        i0 += i_arc
        active[t >= t_on] = True
        arc_rms = float(np.sqrt(np.mean(i_arc[t >= t_on] ** 2)))

    # sensor / ADC noise (CT + 24-bit ADC, referred to primary)
    ia += rng.normal(0, rng.uniform(0.05, 0.4), n_samp)
    i0 += rng.normal(0, rng.uniform(0.05, 0.3), n_samp)
    va += rng.normal(0, V_PK * 1e-3, n_samp)
    return va, ia, i0, active, arc_rms


# ----------------------------------------------------------------------------
# 2. Per-cycle features (bit-level detail finalised in SmartHLS C++ model)
# ----------------------------------------------------------------------------
HARM = [1, 2, 3, 5, 7]
FEATURE_NAMES = [
    "I0_h1", "I0_h2", "I0_h3", "I0_h5", "I0_h7",
    "I0_nonharmonic", "I0_haarD1", "I0_haarD2",
    "I0_asymmetry", "I0_cycle_diff", "I0_h3_angle_cos", "I0_h3_angle_sin",
    "Ia_h2_ratio", "Ia_h1_change",
]


def per_cycle_features(va, ia, i0):
    n_cyc = len(va) // N
    V = np.fft.rfft(va[: n_cyc * N].reshape(n_cyc, N), axis=1) * (np.sqrt(2) / N)
    IA = np.fft.rfft(ia[: n_cyc * N].reshape(n_cyc, N), axis=1) * (np.sqrt(2) / N)
    X0 = i0[: n_cyc * N].reshape(n_cyc, N)
    I0 = np.fft.rfft(X0, axis=1) * (np.sqrt(2) / N)          # rms-scaled phasors

    feats = np.zeros((n_cyc, len(FEATURE_NAMES)))
    mag0 = np.abs(I0)
    for j, h in enumerate(HARM):
        feats[:, j] = np.log1p(mag0[:, h])
    harmonic_bins = np.arange(1, 26)
    total = np.sum(mag0[:, 1:] ** 2, axis=1)
    harm_e = np.sum(mag0[:, harmonic_bins] ** 2, axis=1)
    feats[:, 5] = np.log1p(np.sqrt(np.clip(total - harm_e, 0, None)))

    # Haar DWT detail levels 1 and 2 (adders/subtractors only in hardware)
    a1 = (X0[:, 0::2] + X0[:, 1::2]) / np.sqrt(2)
    d1 = (X0[:, 0::2] - X0[:, 1::2]) / np.sqrt(2)
    d2 = (a1[:, 0::2] - a1[:, 1::2]) / np.sqrt(2)
    feats[:, 6] = np.log1p(np.sqrt(np.mean(d1 ** 2, axis=1)))
    feats[:, 7] = np.log1p(np.sqrt(np.mean(d2 ** 2, axis=1)))

    # half-cycle asymmetry of residual current
    pos = np.sum(np.clip(X0, 0, None), axis=1)
    neg = np.sum(np.clip(-X0, 0, None), axis=1)
    feats[:, 8] = (pos - neg) / (pos + neg + 1e-6)

    # cycle-to-cycle difference energy ("randomness")
    diff = np.zeros(n_cyc)
    diff[1:] = np.sqrt(np.mean((X0[1:] - X0[:-1]) ** 2, axis=1))
    feats[:, 9] = np.log1p(diff)

    # angle of I0 3rd harmonic relative to 3x voltage fundamental angle
    ang = np.angle(I0[:, 3]) - 3 * np.angle(V[:, 1])
    wgt = mag0[:, 3] / (mag0[:, 3] + 1.0)                 # fade when h3 ~ 0
    feats[:, 10] = np.cos(ang) * wgt
    feats[:, 11] = np.sin(ang) * wgt

    # inrush discriminator + load-change indicator on phase current
    ia1 = np.abs(IA[:, 1])
    feats[:, 12] = np.abs(IA[:, 2]) / (ia1 + 1e-3)
    chg = np.zeros(n_cyc)
    chg[1:] = np.abs(np.log((ia1[1:] + 1) / (ia1[:-1] + 1)))
    feats[:, 13] = chg

    i0_rms = np.sqrt(np.mean(X0 ** 2, axis=1))
    h2_ratio = np.maximum(feats[:, 12], mag0[:, 2] / (mag0[:, 1] + 1e-3))
    return feats, i0_rms, h2_ratio


# ----------------------------------------------------------------------------
# 3. Window statistics + labels
# ----------------------------------------------------------------------------
def windows(feats, active, cls):
    n_cyc = feats.shape[0]
    act_cyc = active[: n_cyc * N].reshape(n_cyc, N).mean(axis=1)
    X, y, end_cyc = [], [], []
    for s in range(1, n_cyc - WIN + 1, HOP):      # skip cycle 0 (no diff)
        f = feats[s: s + WIN]
        X.append(np.concatenate([f.mean(axis=0), f.std(axis=0)]))
        frac = act_cyc[s: s + WIN].mean()
        y.append(cls if frac >= 0.5 or (cls in (1, 2) and frac > 0) else NORMAL)
        end_cyc.append(s + WIN)
    return np.array(X), np.array(y), np.array(end_cyc)


def build_dataset(n_per_class, seed):
    rng = np.random.default_rng(seed)
    X, y, ev_id, ev_cls, ev_end, ev_rms, ev_onset = [], [], [], [], [], [], []
    ev = 0
    ev_meta = []
    for c in range(len(CLASSES)):
        for _ in range(n_per_class):
            va, ia, i0, active, arc_rms = generate_event(c, rng)
            feats, i0_rms, h2_ratio = per_cycle_features(va, ia, i0)
            Xw, yw, endc = windows(feats, active, c)
            X.append(Xw); y.append(yw); ev_end.append(endc)
            ev_id.append(np.full(len(yw), ev))
            onset_cyc = int(np.argmax(active) // N) if active.any() else 0
            ev_meta.append(dict(cls=c, i0_rms=i0_rms, h2_ratio=h2_ratio,
                                onset=onset_cyc, arc_rms=arc_rms))
            ev += 1
    return (np.vstack(X), np.concatenate(y), np.concatenate(ev_id),
            np.concatenate(ev_end), ev_meta)


# ----------------------------------------------------------------------------
# 4. MLP (numpy) + int8 quantized inference
# ----------------------------------------------------------------------------
LAYERS = [28, 32, 16, len(CLASSES)]


def init_mlp(rng):
    P = []
    for a, b in zip(LAYERS[:-1], LAYERS[1:]):
        P.append([rng.normal(0, np.sqrt(2 / a), (a, b)), np.zeros(b)])
    return P


def forward(P, X):
    acts = [X]
    h = X
    for i, (W, b) in enumerate(P):
        h = h @ W + b
        if i < len(P) - 1:
            h = np.maximum(h, 0)
        acts.append(h)
    return acts


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_mlp(X, y, rng, epochs=400, lr=3e-3, batch=256, wd=1e-4):
    P = init_mlp(rng)
    m = [[np.zeros_like(W), np.zeros_like(b)] for W, b in P]
    v = [[np.zeros_like(W), np.zeros_like(b)] for W, b in P]
    counts = np.bincount(y, minlength=len(CLASSES))
    cw = (len(y) / (len(CLASSES) * np.maximum(counts, 1)))
    Y = np.eye(len(CLASSES))[y]
    step = 0
    for ep in range(epochs):
        idx = rng.permutation(len(y))
        for s in range(0, len(y), batch):
            bi = idx[s: s + batch]
            acts = forward(P, X[bi])
            p = softmax(acts[-1])
            g = (p - Y[bi]) * cw[y[bi]][:, None] / len(bi)
            step += 1
            for li in range(len(P) - 1, -1, -1):
                W, b = P[li]
                gW = acts[li].T @ g + wd * W
                gb = g.sum(axis=0)
                if li > 0:
                    g = (g @ W.T) * (acts[li] > 0)
                for k, grad in enumerate((gW, gb)):
                    m[li][k] = 0.9 * m[li][k] + 0.1 * grad
                    v[li][k] = 0.999 * v[li][k] + 0.001 * grad ** 2
                    mh = m[li][k] / (1 - 0.9 ** step)
                    vh = v[li][k] / (1 - 0.999 ** step)
                    P[li][k] -= lr * mh / (np.sqrt(vh) + 1e-8)
    return P


def quantize_mlp(P, X_cal):
    """Symmetric per-layer int8 weights, int8 activations, int32 accumulate.
    Requantisation uses a fixed-point multiplier + right shift (hardware)."""
    q = {"layers": []}
    acts = forward(P, X_cal)
    s_in = np.percentile(np.abs(X_cal), 99.9) / 127
    q["s_in"] = s_in
    s_prev = s_in
    for i, (W, b) in enumerate(P):
        s_w = np.abs(W).max() / 127
        Wq = np.clip(np.round(W / s_w), -127, 127).astype(np.int32)
        bq = np.round(b / (s_w * s_prev)).astype(np.int64)
        s_out = np.percentile(np.abs(acts[i + 1]), 99.9) / 127 if i < len(P) - 1 else None
        if s_out is not None:
            M = s_w * s_prev / s_out
            shift = 24
            mult = int(np.round(M * (1 << shift)))
        else:
            mult, shift = None, None
        q["layers"].append(dict(Wq=Wq, bq=bq, mult=mult, shift=shift))
        if s_out is not None:
            s_prev = s_out
    return q


def int8_infer(q, X):
    h = np.clip(np.round(X / q["s_in"]), -127, 127).astype(np.int64)
    for i, L in enumerate(q["layers"]):
        acc = h @ L["Wq"].astype(np.int64) + L["bq"]
        if L["mult"] is not None:
            acc = np.maximum(acc, 0)
            h = np.clip((acc * L["mult"] + (1 << (L["shift"] - 1))) >> L["shift"], 0, 127)
        else:
            return acc                      # int32 logits; argmax in HW
    return h


# ----------------------------------------------------------------------------
# 5. Event-level decisions: ArcSentinel security logic vs 51N baseline
# ----------------------------------------------------------------------------
def event_decisions(pred, ev_id, ev_end, meta):
    out = []
    for e, md in enumerate(meta):
        m = ev_id == e
        p = pred[m]
        ends = ev_end[m]
        run, det_cyc = 0, None
        for k in range(len(p)):
            run = run + 1 if p[k] == HIF else 0
            if run >= K_SECURITY:
                det_cyc = int(ends[k]); break
        # baseline: 51N definite time on per-cycle residual rms, blocked
        # (timer reset) while the 2nd-harmonic ratio indicates inrush
        over = (md["i0_rms"] > PICKUP_51N) & (md["h2_ratio"] < H2_RESTRAINT)
        run51, trip51 = 0, False
        for o in over:
            run51 = run51 + 1 if o else 0
            if run51 >= DELAY_51N_CYC:
                trip51 = True; break
        out.append(dict(cls=md["cls"], arc=det_cyc is not None, det_cyc=det_cyc,
                        onset=md["onset"], trip51=trip51, arc_rms=md["arc_rms"]))
    return out


BANDS = [0.25, 1, 2, 5, 10, 20, 50]


def summarize(dec):
    hif = [d for d in dec if d["cls"] == HIF]
    non = [d for d in dec if d["cls"] != HIF]
    lat = [(d["det_cyc"] - d["onset"]) / F_NOM for d in hif if d["arc"]]
    bands = {}
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        m = [d for d in hif if lo <= d["arc_rms"] < hi]
        if m:
            bands[f"{lo:g}-{hi:g} A"] = dict(n=len(m),
                                             arcsentinel=float(np.mean([d["arc"] for d in m])),
                                             baseline51N=float(np.mean([d["trip51"] for d in m])))
    res = dict(
        hif_events=len(hif), nonhif_events=len(non),
        dependability_by_arc_current=bands,
        arcsentinel_dependability=float(np.mean([d["arc"] for d in hif])),
        arcsentinel_security=float(1 - np.mean([d["arc"] for d in non])),
        baseline51N_dependability=float(np.mean([d["trip51"] for d in hif])),
        baseline51N_security=float(1 - np.mean([d["trip51"] for d in non])),
        median_detection_time_s=float(np.median(lat)) if lat else None,
        p95_detection_time_s=float(np.percentile(lat, 95)) if lat else None,
        false_alarms_by_class={CLASSES[c]: int(sum(d["arc"] for d in non if d["cls"] == c))
                               for c in range(len(CLASSES)) if c != HIF},
        baseline51N_false_trips_by_class={CLASSES[c]: int(sum(d["trip51"] for d in non if d["cls"] == c))
                                          for c in range(len(CLASSES)) if c != HIF},
    )
    return res


def confusion(y, p):
    C = np.zeros((len(CLASSES), len(CLASSES)), int)
    for a, b in zip(y, p):
        C[a, b] += 1
    return C


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def make_plots(C, dec, rng_seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    ink, muted, accent, alt = "#1b2330", "#6b7684", "#d9480f", "#1c7ed6"

    # --- example waveforms
    rng = np.random.default_rng(rng_seed)
    fig, axs = plt.subplots(2, 3, figsize=(10, 4.6), sharex=True)
    show = ["normal", "cap_switch", "inrush", "nonlinear_load", "welder", "HIF"]
    for ax, name in zip(axs.flat, show):
        va, ia, i0, _, _ = generate_event(CLASSES.index(name), rng)
        k0 = int(0.45 * FS)
        tt = np.arange(4 * N) / FS * 1e3
        ax.plot(tt, i0[k0:k0 + 4 * N], color=accent if name == "HIF" else alt, lw=0.8)
        ax.set_title(name.replace("_", " "), color=ink, loc="left", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=muted)
    for ax in axs[1]:
        ax.set_xlabel("time (ms)", color=muted)
    for ax in axs[:, 0]:
        ax.set_ylabel("residual current (A)", color=muted)
    fig.suptitle("Synthetic residual-current signatures (4 cycles, 25.6 kS/s)", x=0.01, ha="left", color=ink)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_waveforms.png"), dpi=200)
    plt.close(fig)

    # --- confusion matrix (int8 model, window level)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    Cn = C / np.maximum(C.sum(axis=1, keepdims=True), 1)
    ax.imshow(Cn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), [c.replace("_", "\n") for c in CLASSES], fontsize=7)
    ax.set_yticks(range(len(CLASSES)), [c.replace("_", " ") for c in CLASSES], fontsize=7)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if C[i, j]:
                ax.text(j, i, f"{Cn[i, j]*100:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if Cn[i, j] > 0.5 else ink)
    ax.set_xlabel("predicted", color=muted); ax.set_ylabel("true", color=muted)
    ax.set_title("int8 classifier, held-out windows (% of row)", loc="left", color=ink, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_confusion.png"), dpi=200)
    plt.close(fig)

    # --- detection rate vs fault current: ArcSentinel vs 51N
    hif = [d for d in dec if d["cls"] == HIF]
    cur = np.array([d["arc_rms"] for d in hif])
    bins = np.array(BANDS)
    centers, a_rate, b_rate = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (cur >= lo) & (cur < hi)
        if m.sum() < 3:
            continue
        centers.append(f"{lo:g}-{hi:g}")
        a_rate.append(np.mean([d["arc"] for d, mm in zip(hif, m) if mm]) * 100)
        b_rate.append(np.mean([d["trip51"] for d, mm in zip(hif, m) if mm]) * 100)
    x = np.arange(len(centers))
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(x - 0.2, b_rate, 0.38, color="#adb5bd", label=f"51N + inrush restraint (pickup {PICKUP_51N:.0f} A)")
    ax.bar(x + 0.2, a_rate, 0.38, color=accent, label="ArcSentinel (int8)")
    ax.set_xticks(x, centers)
    ax.set_xlabel("HIF arc current (A rms, log-spaced bands)", color=muted)
    ax.set_ylabel("faults detected (%)", color=muted)
    ax.set_ylim(0, 128)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    ax.set_title("High-impedance fault detection vs arc current (held-out events)", loc="left", color=ink, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_detection_vs_current.png"), dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    t0 = time.time()
    n_train, n_test = 240, 120
    print(f"fs = {FS/1e3:.1f} kS/s, {N} samples/cycle, classes = {CLASSES}")
    Xtr, ytr, _, _, _ = build_dataset(n_train, seed=1)
    Xte, yte, ev_te, end_te, meta_te = build_dataset(n_test, seed=2)   # independent seed
    print(f"windows: train {len(ytr)}, test {len(yte)}  ({time.time()-t0:.1f}s)")

    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-6
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd      # folded into layer-1 in HW

    rng = np.random.default_rng(7)
    P = train_mlp(Ztr, ytr, rng)
    pf = forward(P, Zte)[-1].argmax(axis=1)
    q = quantize_mlp(P, Ztr)
    pq = int8_infer(q, Zte).argmax(axis=1)

    acc_f = float(np.mean(pf == yte)); acc_q = float(np.mean(pq == yte))
    print(f"window accuracy: float {acc_f:.4f}  int8 {acc_q:.4f}")

    dec = event_decisions(pq, ev_te, end_te, meta_te)
    res = summarize(dec)
    C = confusion(yte, pq)

    n_mac = sum(a * b for a, b in zip(LAYERS[:-1], LAYERS[1:]))
    n_par = sum(a * b + b for a, b in zip(LAYERS[:-1], LAYERS[1:]))
    res.update(dict(
        window_accuracy_float=acc_f, window_accuracy_int8=acc_q,
        int8_vs_float_agreement=float(np.mean(pf == pq)),
        mlp_layers=LAYERS, mlp_macs_per_inference=n_mac, mlp_params=n_par,
        mlp_weight_bytes_int8=n_par,
        fs_hz=FS, samples_per_cycle=N, window_cycles=WIN, hop_cycles=HOP,
        k_security=K_SECURITY, pickup_51N_A=PICKUP_51N, h2_restraint=H2_RESTRAINT,
        arc_rms_range_A=list(ARC_RMS_RANGE),
        train_events=n_train * len(CLASSES), test_events=n_test * len(CLASSES),
        confusion_int8=C.tolist(), classes=CLASSES,
        runtime_s=round(time.time() - t0, 1),
    ))
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(res, f, indent=2)

    make_plots(C, dec, rng_seed=11)
    print(json.dumps({k: v for k, v in res.items() if k != "confusion_int8"}, indent=2))


if __name__ == "__main__":
    main()
