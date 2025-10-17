# app.py — NeurobIQs EEG Dashboard (Production)
from __future__ import annotations

import io
import os
import re
import logging
import warnings
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio
from scipy.signal import welch, detrend, savgol_filter, butter, filtfilt, iirnotch

# ──────────────────────────────────────────────────────────────────────────────
# APP META / WARNING HYGIENE
# ──────────────────────────────────────────────────────────────────────────────
APP_NAME     = "NeurobIQs EEG Dashboard"
APP_VERSION  = "1.7.0"
BUILD_STAMP  = datetime.now().strftime("%Y-%m-%d %H:%M")

# Silence noisy warnings (helps JSON-based runners and keeps logs clean)
SILENCE_WARNINGS = True
if SILENCE_WARNINGS:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.captureWarnings(True)
    os.environ.setdefault("PYTHONWARNINGS", "ignore")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & THEME
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title=APP_NAME, layout="wide")
pio.templates.default = "simple_white"

st.markdown(f"""
<style>
:root {{
  --brand: #6d4aff;
  --ok: #4caf50;
  --warn: #e6c850;
  --bad: #d44848;
  --ink: rgba(0,0,0,0.85);
  --muted: rgba(0,0,0,0.6);
  --panel: #ffffff;
  --shadow: 0 1px 2px rgba(0,0,0,0.06), 0 6px 24px rgba(0,0,0,0.08);
}}
html, body, [class*="css"] {{ font-variant-numeric: tabular-nums; }}
h1, h2, h3 {{ letter-spacing: .2px; }}
.app-card {{ background: var(--panel); border-radius: 16px; padding: 16px 18px;
  box-shadow: var(--shadow); border: 1px solid rgba(0,0,0,.06); margin-bottom: 14px; }}
.card-title {{ display:flex; align-items:center; gap:10px; font-weight:700; color:var(--ink); margin-bottom: 8px; }}
.card-sub {{ color: var(--muted); font-size: .92rem; margin-top:-4px; }}
.kpi {{ background: linear-gradient(180deg, rgba(109,74,255,.05), rgba(109,74,255,.02));
  border:1px solid rgba(109,74,255,.12); border-radius:16px; padding:14px; box-shadow: var(--shadow); }}
.kpi .label {{ color: var(--muted); font-size:.85rem; margin-bottom:4px;}}
.kpi .value {{ font-weight:800; font-size:1.25rem;}}
.badge {{ display:inline-block; padding:3px 10px; border-radius:999px; background:rgba(0,0,0,.06);
  color:var(--ink); font-weight:700; font-size:.90rem; }}
.checkgrid {{ display:grid; grid-template-columns: 140px 130px 1fr; gap:6px 12px; }}
.checkitem {{ display:flex; align-items:center; gap:8px; padding:8px 10px; border-radius:10px;
  background: rgba(0,0,0,.03); border:1px solid rgba(0,0,0,.05); }}
.tick-ok {{ color: var(--ok); font-weight:800;}}
.tick-bad {{ color: var(--bad); font-weight:800;}}
.qbar-row {{ display:flex; align-items:center; justify-content:space-between; margin: 8px 4px 2px 4px; }}
.qbar-title {{ font-weight:600; font-size:.96rem; line-height:1.2; color: var(--ink); white-space:normal; max-width: calc(100% - 130px); }}
.qbar-badge {{ font-weight:700; font-size:.90rem; padding:4px 10px; border-radius:999px;
  background:rgba(0,0,0,.06); color:var(--ink); }}
.smallnote {{ color: var(--muted); font-size:.85rem; }}
.footer {{ color: var(--muted); font-size:.80rem; }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"<h1 style='margin-bottom:4px'>{APP_NAME}</h1>", unsafe_allow_html=True)
st.caption(f"v{APP_VERSION} · Built {BUILD_STAMP}")

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
VALID_CHANNELS = {"Cz", "F3", "F4", "O1"}
DEFAULT_FS = 250.0

BANDS = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 12.0),
    "Beta":  (13.0, 21.0),   # TBR denominator
    "LowBeta": (12.0, 15.0), # Theta/Low-Beta
    "Beta15_20": (15.0, 20.0),
    "HiBeta": (20.0, 30.0),
}

# Robust trapezoid integrator (no deprecation warnings)
try:
    from numpy import trapezoid as _trapint
except Exception:
    try:
        from scipy.integrate import trapezoid as _trapint
    except Exception:
        from numpy import trapz as _trapint  # last-resort fallback

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Processing")
    N_PERSEG = st.slider("Welch nperseg (FFT window length)", 256, 4096, 2048, step=256)
    N_OVERLAP = st.slider("Welch noverlap (overlap between windows)", 0, 2048, 1024, step=128)
    F_MAX_VIEW = st.slider("Max frequency shown (Hz)", 20, 60, 45, step=5)
    SYM_TOL = st.number_input("Symmetry tolerance (± around 1.0)", value=0.15, min_value=0.05, max_value=0.5, step=0.01)

    st.header("Input & scaling")
    unit = st.selectbox("Signal input unit", ["µV", "mV"], index=0)
    EXTRA_SCALE = st.number_input("Extra scale factor (×)", value=1.0, min_value=0.0001, step=0.1, format="%.4f")
    FS_OVERRIDE = st.number_input("Override sampling rate (Hz, 0 = auto)", value=0.0, min_value=0.0, step=25.0)

    st.header("Artifact handling")
    ART_ENABLE = st.checkbox("Enable artifact cleaning", value=True)
    BP_ENABLE  = st.checkbox("Band-pass filter", value=True)
    BP_LOW, BP_HIGH = st.slider("Band-pass (Hz)", 0.1, 60.0, (1.0, 45.0), step=0.1)
    NOTCH = st.selectbox("Notch", ["Off", "50 Hz", "60 Hz"], index=1)
    st.caption("EU powerline is 50 Hz; US is 60 Hz.")

    st.markdown("**Epoching & thresholds**")
    EPOCH_SEC = st.slider("Epoch length (s)", 1.0, 4.0, 2.0, step=0.5)
    Z_AMP = st.slider("Amplitude (robust z) reject >", 2.0, 6.0, 3.5, step=0.5)
    BLINK_RATIO = st.slider("Blink ratio (1–3 Hz / 8–12 Hz) >", 0.5, 5.0, 2.0, step=0.1)
    MUSCLE_RATIO = st.slider("Muscle ratio (25–45 Hz / 8–12 Hz) >", 0.5, 5.0, 1.5, step=0.1)
    FLAT_STD_MIN = st.number_input("Flatline min std", value=1e-6, format="%.1e")
    MIN_GOOD_EPOCHS = st.number_input("Min good epochs required", value=6, min_value=1, step=1)

    st.divider()
    SHOW_DIAL_DEBUG = st.checkbox("Show dial component details", value=False)
    ENABLE_NOTES = st.checkbox("Enable '+ Add note' on sliders", value=True)
    DISPLAY_INDICES = st.radio("Display indices (badges)?", ["Yes", "No"], index=0, horizontal=True) == "Yes"

if "notes" not in st.session_state:
    st.session_state["notes"] = {}

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def parse_meta_from_filename(name: str) -> Tuple[Optional[str], Optional[str]]:
    base = os.path.basename(name)
    chan = None
    for c in VALID_CHANNELS:
        if re.search(rf"(^|[_\-]){c}([_\-]|[0-9])", base, re.IGNORECASE) or base.startswith(c):
            chan = c; break
    cond = "EyesOpen" if re.search(r"eyes\s*open", base, re.IGNORECASE) else (
           "EyesClosed" if re.search(r"eyes\s*closed", base, re.IGNORECASE) else None)
    return chan, cond

def estimate_fs(ts: np.ndarray) -> float:
    """Estimate sampling rate from timestamps (ms or s)."""
    ts = np.asarray(ts).astype(float)
    if ts.size < 4: return DEFAULT_FS
    d = np.diff(ts)
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0: return DEFAULT_FS
    dt = float(np.median(d))
    fs = 1.0/dt if dt < 0.2 else 1000.0/dt
    return float(fs) if 100.0 <= fs <= 1024.0 else DEFAULT_FS

def safe_welch_psd(x: np.ndarray, fs: float, nperseg: int, noverlap: int) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD with guards. Uses FFT under the hood."""
    nseg = max(8, min(int(nperseg), len(x)))
    nov  = min(int(noverlap), max(0, nseg - 1))
    f, pxx = welch(
        detrend(x, type="constant"),
        fs=fs, window="hann", nperseg=nseg, noverlap=nov,
        detrend=False, return_onesided=True, scaling="density",
    )
    return f, pxx

def band_power(f: np.ndarray, pxx: np.ndarray, lo: float, hi: float) -> float:
    m = (f >= lo) & (f < hi)
    return float(_trapint(pxx[m], f[m])) if np.any(m) else 0.0

def pct_change(ec: float, eo: float) -> float:
    if eo == 0 or not np.isfinite(eo) or not np.isfinite(ec): return np.nan
    return 100.0*(ec - eo)/eo

def score_range(x: float, lo: float, hi: float) -> float:
    if hi <= lo: return 0.0
    mid = (lo+hi)/2.0
    return max(0.0, 1.0 - 2.0*abs(x-mid)/(hi-lo))

def score_sym1(r: float, tol: float) -> float:
    if tol <= 0: return 0.0
    return max(0.0, 1.0 - abs(r-1.0)/tol)

def score_shift(pct: float) -> float:
    return float(np.clip((pct + 25.0)/125.0, 0.0, 1.0))

def mean_finite(xs: List[Optional[float]]) -> float:
    arr = np.array([x for x in xs if x is not None and np.isfinite(x)], dtype=float)
    return float(arr.mean()) if arr.size else np.nan

# Filters
def _butter_bandpass(low, high, fs, order=4):
    nyq = 0.5 * fs
    lowc = max(0.0001, low / nyq)
    highc = min(0.9999, high / nyq)
    if highc <= lowc:
        lowc, highc = 0.01, 0.9
    b, a = butter(order, [lowc, highc], btype="band")
    return b, a

def apply_bandpass(x: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    try:
        b, a = _butter_bandpass(low, high, fs, order=4)
        return filtfilt(b, a, x)
    except Exception:
        return x

def apply_notch(x: np.ndarray, fs: float, freq: float, Q: float = 30.0) -> np.ndarray:
    try:
        if freq >= 0.5*fs:  # guard if chosen notch is above Nyquist
            return x
        b, a = iirnotch(w0=freq, Q=Q, fs=fs)
        return filtfilt(b, a, x)
    except Exception:
        return x

# Epoching & artifact rules
def epoch_bounds(n: int, fs: float, epoch_sec: float) -> List[Tuple[int,int]]:
    L = max(1, int(round(epoch_sec * fs)))
    starts = list(range(0, n, L))
    bounds = [(s, min(n, s + L)) for s in starts]
    return [(s, e) for (s, e) in bounds if (e - s) >= 0.5 * L]

def robust_amp_threshold(x: np.ndarray, z_amp: float) -> float:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    robust_std = 1.4826 * mad
    return float(z_amp * robust_std)

def artifact_clean(
    x: np.ndarray,
    fs: float,
    epoch_sec: float,
    z_amp: float,
    blink_ratio_thr: float,
    muscle_ratio_thr: float,
    flat_std_min: float,
) -> Tuple[np.ndarray, Dict]:
    """
    Returns (x_clean concatenated good epochs, qc dict).
    QC includes total/good epochs and reason counts.
    """
    n = len(x)
    if n < 8:
        return x.copy(), {"total": 0, "good": 0, "used_original": True}

    epochs = epoch_bounds(n, fs, epoch_sec)
    thr_amp = robust_amp_threshold(x, z_amp)

    good_segments: List[np.ndarray] = []
    reasons = {"amp": 0, "blink": 0, "muscle": 0, "flat": 0}
    total = len(epochs)

    for (s, e) in epochs:
        ep = x[s:e]
        # flatline?
        if np.std(ep) < flat_std_min:
            reasons["flat"] += 1
            continue
        # amplitude spike?
        if np.max(np.abs(ep - np.median(ep))) > thr_amp:
            reasons["amp"] += 1
            continue
        # quick PSD
        f_ep, pxx_ep = safe_welch_psd(ep, fs, N_PERSEG, N_OVERLAP)
        def pwr(lo, hi): return band_power(f_ep, pxx_ep, lo, hi)
        alpha = max(pwr(8, 12), 1e-12)
        blink = pwr(1, 3) / alpha
        muscle = pwr(25, 45) / alpha
        if blink > blink_ratio_thr:
            reasons["blink"] += 1
            continue
        if muscle > muscle_ratio_thr:
            reasons["muscle"] += 1
            continue
        good_segments.append(ep)

    if len(good_segments) == 0:
        return x.copy(), {"total": total, "good": 0, **reasons, "used_original": True}

    x_clean = np.concatenate(good_segments, axis=0)
    return x_clean, {"total": total, "good": len(good_segments), **reasons, "used_original": False}

# Slider figure (UI & PDF)
def build_slider_figure(
    title: str,
    value: float | None,
    axis_min: float,
    axis_max: float,
    green: List[Tuple[float,float]] = (),
    yellow: List[Tuple[float,float]] = (),
    red: List[Tuple[float,float]] = (),
    ticks: List[float] = (),
    unit: str = "",
    height: int = 88,
    bar_thickness: float = 0.14,
    marker_size: int = 9
) -> go.Figure:
    fig = go.Figure()
    y_mid = 0.50
    y0 = y_mid - bar_thickness/2.0
    y1 = y_mid + bar_thickness/2.0

    def add_bands(ranges, color):
        for lo, hi in ranges:
            lo = max(lo, axis_min); hi = min(hi, axis_max)
            if hi <= lo: continue
            fig.add_shape(type="rect", x0=lo, x1=hi, y0=y0, y1=y1,
                          xref="x", yref="y", line=dict(width=0),
                          fillcolor=color, opacity=1.0, layer="below")
    if not (green or yellow or red):
        add_bands([(axis_min, axis_max)], "rgba(160,160,160,0.35)")
    else:
        add_bands(red,    "rgb(212,72,72)")
        add_bands(yellow, "rgb(230,200,80)")
        add_bands(green,  "rgb(80,180,80)")

    if value is not None and np.isfinite(value):
        x = float(np.clip(value, axis_min, axis_max))
        fig.add_shape(type="line", x0=x, x1=x, y0=y0+0.02, y1=y1-0.02,
                      xref="x", yref="y", line=dict(color="rgba(0,0,0,0.65)", width=1.3), layer="above")
        fig.add_trace(go.Scatter(
            x=[x], y=[y_mid], mode="markers",
            marker=dict(size=marker_size, color="rgba(109,74,255,.95)", line=dict(width=1.2, color="white")),
            hovertemplate=f"{title}: %{x:.2f}{unit}<extra></extra>", showlegend=False
        ))
    else:
        fig.add_annotation(xref="paper", yref="paper", x=0.5, y=y_mid, showarrow=False,
                           text="n/a", font=dict(color="rgba(0,0,0,0.6)"))
    fig.update_xaxes(range=[axis_min, axis_max], showline=False, showgrid=False, zeroline=False,
                     tickmode="array", tickvals=ticks or [], tickfont=dict(size=10))
    fig.update_yaxes(visible=False, range=[0,1])
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=2, b=26),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    for t in (ticks or []):
        if axis_min <= t <= axis_max:
            fig.add_annotation(x=t, y=y0-0.12, xref="x", yref="y",
                               text=f"{t:g}{unit}", showarrow=False,
                               font=dict(size=10, color="rgba(0,0,0,0.85)"))
    return fig

def ui_slider_row(
    title: str,
    value: float | None,
    axis_min: float,
    axis_max: float,
    green: List[Tuple[float,float]] = (),
    yellow: List[Tuple[float,float]] = (),
    red: List[Tuple[float,float]] = (),
    ticks: List[float] = (),
    unit: str = "",
    key: str | None = None,
    show_note: bool = False,
    show_badge: bool = True
):
    badge = "n/a" if value is None or not np.isfinite(value) else f"{value:.2f}{unit}"
    right = f'<div class="qbar-badge">{badge}</div>' if show_badge else ""
    st.markdown(f'<div class="qbar-row"><div class="qbar-title">{title}</div>{right}</div>', unsafe_allow_html=True)
    fig = build_slider_figure(title, value, axis_min, axis_max, green, yellow, red, ticks, unit)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    if show_note:
        note_key = f"note_{key or title}"
        txt = st.text_area("Add note", key=note_key, label_visibility="collapsed", placeholder="Type your observation…")
        st.session_state["notes"][key or title] = txt

# Dials
def gauge_figure(
    title: str,
    score01: Optional[float],
    invert: bool=False,
    height: int = 230,
    width: int  = 260,
    thickness: float = 0.18,
    title_size: int = 14
) -> go.Figure:
    v = 0 if (score01 is None or not np.isfinite(score01)) else float(100*score01)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=v, number={'suffix':'%'},
        gauge={'axis': {'range':[0,100]},
               'bar': {'thickness': thickness},
               'steps': [
                   {'range':[0,60],  'color': "rgb(220,80,80)" if invert else "rgb(80,180,80)"},
                   {'range':[60,80], 'color': "rgb(230,200,80)"},
                   {'range':[80,100],'color': "rgb(80,180,80)" if invert else "rgb(220,80,80)"},
               ]}
    ))
    fig.update_layout(
        title={'text': title, 'y': 0.98, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top', 'font': {'size': title_size}},
        margin=dict(l=10, r=10, t=50, b=6),
        height=height, width=width
    )
    return fig

def ui_gauge(title: str, score01: Optional[float], invert: bool=False):
    st.plotly_chart(gauge_figure(title, score01, invert), use_container_width=True)

def make_export_basename(raw: Optional[str]) -> str:
    if not raw:
        return "neurobiqs_report"
    s = str(raw).strip()
    s = re.sub(r"[\s\.]+", "-", s)
    s = re.sub(r"[^A-Za-z0-9\-_]+", "", s)
    s = s.strip("-_")
    return s or "neurobiqs_report"

def hash_psd_input(x: np.ndarray, fs: float, nperseg: int, noverlap: int) -> str:
    """Content hash for PSD caching (independent of filename)."""
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(x).view(np.uint8))
    h.update(np.array([fs, nperseg, noverlap], dtype=np.float64).view(np.uint8))
    return h.hexdigest()

# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
tab_upload, tab_metrics, tab_qc, tab_dials, tab_sliders, tab_psd, tab_export = st.tabs(
    ["📁 Upload", "📊 Metrics", "🧪 QC", "🧭 Dials", "🎚️ Sliders", "📈 PSD", "⬇️ Export"]
)

# ──────────────────────────────────────────────────────────────────────────────
# 1) UPLOAD + DUPLICATE RESOLVER
# ──────────────────────────────────────────────────────────────────────────────
with tab_upload:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📁 Data upload</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload the CSVs (Cz, F3, F4, O1 × EyesOpen/EyesClosed). Columns: timestamp and one signal column.",
        type=["csv"], accept_multiple_files=True
    )
    auto_paths = [p for p in os.listdir(".")
                  if re.search(r"(Cz|F3|F4|O1).*(Eyes(Open|Closed))", p, re.IGNORECASE)
                  and p.lower().endswith(".csv")]
    auto_files = []
    if not uploaded and auto_paths:
        for p in sorted(auto_paths):
            try: auto_files.append({"name": p, "data": pd.read_csv(p)})
            except Exception: pass
        st.info("Auto-loaded CSVs from working directory.")

    # Input scaling
    UNIT_SCALE = 1_000.0 if unit == "mV" else 1.0
    UNIT_SCALE *= float(EXTRA_SCALE)

    data_rows: List[Dict] = []
    def push_one(name: str, df: pd.DataFrame):
        chan, cond = parse_meta_from_filename(name)
        if chan is None or cond is None:
            st.warning(f"Could not parse channel/condition from filename: **{name}** — expected Cz/F3/F4/O1 and EyesOpen/EyesClosed."); return
        if df.shape[1] < 2:
            st.warning(f"{name}: expected 2 columns: 'timestamp' and signal."); return
        ts = df.iloc[:,0].to_numpy()
        x  = df.iloc[:,1].to_numpy(dtype=float) * UNIT_SCALE
        fs = float(FS_OVERRIDE) if FS_OVERRIDE > 0 else estimate_fs(ts)
        data_rows.append({"file":name, "channel":chan, "condition":cond, "fs":fs, "timestamp":ts, "signal":x})

    if uploaded:
        for f in uploaded:
            try: df = pd.read_csv(f)
            except Exception as e: st.error(f"Failed to read {f.name}: {e}"); continue
            push_one(f.name, df)
    elif auto_files:
        for entry in auto_files: push_one(entry["name"], entry["data"])
    st.markdown('</div>', unsafe_allow_html=True)

    if not data_rows:
        st.stop()

    # Show completeness
    need = [(c, "EyesOpen") for c in ["Cz","F3","F4","O1"]] + [(c, "EyesClosed") for c in ["Cz","F3","F4","O1"]]
    have = {}
    for r in data_rows:
        have.setdefault((r["channel"], r["condition"]), []).append(r)

    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">✅ Completeness & duplicate resolver</div>', unsafe_allow_html=True)

    # Duplicate resolver UI (choose one per Channel×Condition)
    chosen_rows: List[Dict] = []
    with st.expander("Resolve duplicates (if any)", expanded=any(len(v)>1 for v in have.values())):
        for (ch, cond) in sorted(have.keys()):
            rows = have[(ch,cond)]
            if len(rows) == 1:
                chosen_rows.append(rows[0])
                st.markdown(f"- **{ch} / {'EO' if cond=='EyesOpen' else 'EC'}** — using: `{rows[0]['file']}`")
            else:
                opts = [r["file"] for r in rows]
                key = f"dup_{ch}_{cond}"
                default_idx = len(opts)-1  # default to last (often latest)
                pick = st.selectbox(f"{ch} / {'EO' if cond=='EyesOpen' else 'EC'}", opts, index=default_idx, key=key)
                chosen = next(r for r in rows if r["file"] == pick)
                chosen_rows.append(chosen)

    # Checklist grid
    st.markdown('<div class="checkgrid">', unsafe_allow_html=True)
    st.markdown("<div><strong>Channel</strong></div><div><strong>Condition</strong></div><div><strong>Status</strong></div>", unsafe_allow_html=True)
    for ch, cond in need:
        ok = any((row["channel"]==ch and row["condition"]==cond) for row in chosen_rows)
        tick = "<span class='tick-ok'>✔</span> Ready" if ok else "<span class='tick-bad'>✖</span> Missing"
        st.markdown(f"<div class='checkitem'><div>{ch}</div><div>{'EO' if cond=='EyesOpen' else 'EC'}</div><div>{tick}</div></div>", unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

    # File index for selected files
    idx_df = pd.DataFrame([{"file":r["file"], "channel":r["channel"], "condition":r["condition"],
                            "fs":r["fs"], "n":len(r["signal"])} for r in chosen_rows]) \
                .sort_values(["channel","condition","file"])
    with st.expander("View selected file index"):
        st.dataframe(idx_df, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# 2) ARTIFACT CLEANING & PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────
processed_rows: List[Dict] = []
qc_rows: List[Dict] = []

for r in chosen_rows:
    x = r["signal"].astype(float)
    fs = float(r["fs"])

    # Optional filtering (safe fallbacks)
    if BP_ENABLE:
        x = apply_bandpass(x, fs, BP_LOW, BP_HIGH)
    if NOTCH != "Off":
        notch_freq = 50.0 if NOTCH.startswith("50") else 60.0
        x = apply_notch(x, fs, notch_freq, Q=30.0)

    # Optional artifact cleaning
    if ART_ENABLE:
        x_clean, qc = artifact_clean(
            x=x, fs=fs, epoch_sec=EPOCH_SEC, z_amp=Z_AMP,
            blink_ratio_thr=BLINK_RATIO, muscle_ratio_thr=MUSCLE_RATIO,
            flat_std_min=FLAT_STD_MIN
        )
    else:
        x_clean, qc = x, {"total": 0, "good": 0, "used_original": True}

    r2 = {**r, "signal_clean": x_clean}
    processed_rows.append(r2)

    qc_rows.append({
        "file": r["file"],
        "channel": r["channel"],
        "condition": r["condition"],
        "fs": r["fs"],
        "samples_raw": len(r["signal"]),
        "samples_clean": len(x_clean),
        "epochs_total": qc.get("total", 0),
        "epochs_good": qc.get("good", 0),
        "clean_pct": (100.0 * qc.get("good", 0) / max(1, qc.get("total", 0))) if qc.get("total", 0) > 0 else np.nan,
        "rej_amp": qc.get("amp", 0),
        "rej_blink": qc.get("blink", 0),
        "rej_muscle": qc.get("muscle", 0),
        "rej_flat": qc.get("flat", 0),
        "used_original": qc.get("used_original", False),
    })

qc_df = pd.DataFrame(qc_rows).sort_values(["channel", "condition", "file"])

# ──────────────────────────────────────────────────────────────────────────────
# 3) PSD CACHE (content hashing) & METRICS
# ──────────────────────────────────────────────────────────────────────────────
psd_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
psd_key_map: Dict[Tuple[str,str], str] = {}  # (channel,cond) -> hash key

for r in processed_rows:
    key = hash_psd_input(r["signal_clean"], float(r["fs"]), N_PERSEG, N_OVERLAP)
    if key not in psd_cache:
        f, pxx = safe_welch_psd(r["signal_clean"], float(r["fs"]), N_PERSEG, N_OVERLAP)
        psd_cache[key] = (f, pxx)
    psd_key_map[(r["channel"], r["condition"])] = key

# Band powers & ratios
records = []
for r in processed_rows:
    key = psd_key_map[(r["channel"], r["condition"])]
    f, pxx = psd_cache[key]
    bands = {name: band_power(f, pxx, lo, hi) for name,(lo,hi) in BANDS.items()}
    theta, alpha = bands["Theta"], bands["Alpha"]
    beta, lowb   = bands["Beta"], bands["LowBeta"]
    b1520, hib   = bands["Beta15_20"], bands["HiBeta"]
    TBR = theta/beta if beta>0 else np.nan
    ThetaLowBeta = theta/lowb if lowb>0 else np.nan
    ThetaAlpha   = theta/alpha if alpha>0 else np.nan
    BetaSplit    = b1520/hib if hib>0 else np.nan
    records.append({
        "file": r["file"], "channel": r["channel"], "condition": r["condition"], "fs": r["fs"],
        **bands, "TBR": TBR, "ThetaLowBeta": ThetaLowBeta, "ThetaAlpha": ThetaAlpha,
        "BetaSplit_15_20_over_20_30": BetaSplit
    })
bands_df = pd.DataFrame.from_records(records).sort_values(["channel","condition","file"])

with tab_metrics:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📊 Band powers & ratios</div>', unsafe_allow_html=True)
    st.dataframe(bands_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Derived metrics & helpers
def get_val(ch: str, cond: str, col: str) -> Optional[float]:
    try:
        v = float(bands_df[(bands_df["channel"]==ch) & (bands_df["condition"]==cond)][col].iloc[0])
        return v if np.isfinite(v) else None
    except IndexError:
        return None

def alpha_shift(ch: str) -> Optional[float]:
    eo, ec = get_val(ch,"EyesOpen","Alpha"), get_val(ch,"EyesClosed","Alpha")
    return None if eo is None or ec is None else pct_change(ec, eo)

def tbr_shift(ch: str) -> Optional[float]:
    eo, ec = get_val(ch,"EyesOpen","TBR"), get_val(ch,"EyesClosed","TBR")
    if eo is None or ec is None or not np.isfinite(eo) or eo==0: return None
    return 100.0*(ec - eo)/eo

def symmetry(band: str, cond: str="EyesClosed") -> Optional[float]:
    f3, f4 = get_val("F3",cond,band), get_val("F4",cond,band)
    if f3 is None or f4 is None or not np.isfinite(f3) or not np.isfinite(f4) or f4==0: return None
    return f3/f4

def peak_alpha(channel="O1", cond="EyesClosed") -> Optional[float]:
    key = psd_key_map.get((channel, cond))
    if key is None: return None
    f, pxx = psd_cache[key]
    m = (f>=8.0) & (f<=12.0)
    if np.any(m):
        fx, px = f[m], pxx[m]
        if fx.size >= 11:
            try: px = savgol_filter(px, window_length=11, polyorder=2, mode="interp")
            except Exception: pass
        idx = int(np.argmax(px))
        return float(fx[idx])
    m2 = (f>=7.0) & (f<=13.0)
    if not np.any(m2): return None
    idx = int(np.argmax(pxx[m2]))
    return float(f[m2][idx])

alpha_shift_O1 = alpha_shift("O1")
alpha_shift_Cz = alpha_shift("Cz")
tbr_shift_O1   = tbr_shift("O1")
theta_sym = symmetry("Theta","EyesClosed")
alpha_sym = symmetry("Alpha","EyesClosed")
beta_sym  = symmetry("Beta","EyesClosed")

# Peak Alpha for all channels (EC)
paf_o1_ec = peak_alpha("O1","EyesClosed")
paf_cz_ec = peak_alpha("Cz","EyesClosed")
paf_f3_ec = peak_alpha("F3","EyesClosed")
paf_f4_ec = peak_alpha("F4","EyesClosed")

def alpha_flex_O1() -> Optional[float]:
    eo_rows = bands_df[(bands_df["channel"]=="O1") & (bands_df["condition"]=="EyesOpen")].sort_values("file")
    if len(eo_rows) < 2: return None
    eo1, eo2 = float(eo_rows["Alpha"].iloc[0]), float(eo_rows["Alpha"].iloc[1])
    if not np.isfinite(eo1) or eo1==0 or not np.isfinite(eo2): return None
    return 100.0*(eo2 - eo1)/eo1

alpha_flex_o1 = alpha_flex_O1()

tbr_cz_eo = get_val("Cz","EyesOpen","TBR")
tlb_cz_ec = get_val("Cz","EyesClosed","ThetaLowBeta")
tbr_o1_eo = get_val("O1","EyesOpen","TBR")
tbr_o1_ec = get_val("O1","EyesClosed","TBR")
f3_tbr_ec, f4_tbr_ec = get_val("F3","EyesClosed","TBR"), get_val("F4","EyesClosed","TBR")
frontal_tbr_sym = None if f3_tbr_ec is None or f4_tbr_ec is None or f4_tbr_ec==0 else f3_tbr_ec/f4_tbr_ec
frontal_tbr_avg = mean_finite([f3_tbr_ec, f4_tbr_ec])

derived_df = pd.DataFrame([
    {"Metric":"AlphaShift_O1_pct (EO→EC)", "Value": alpha_shift_O1},
    {"Metric":"TBR_Shift_O1_pct (EO→EC)", "Value": tbr_shift_O1},
    {"Metric":"AlphaShift_Cz_pct (EO→EC)", "Value": alpha_shift_Cz},
    {"Metric":"Theta_Symmetry F3/F4 (EC)", "Value": theta_sym},
    {"Metric":"Alpha_Symmetry F3/F4 (EC)", "Value": alpha_sym},
    {"Metric":"Beta(13–21) Symmetry F3/F4 (EC)", "Value": beta_sym},
    {"Metric":"Peak Alpha O1 (EC) [Hz]", "Value": paf_o1_ec},
    {"Metric":"Peak Alpha Cz (EC) [Hz]", "Value": paf_cz_ec},
    {"Metric":"Peak Alpha F3 (EC) [Hz]", "Value": paf_f3_ec},
    {"Metric":"Peak Alpha F4 (EC) [Hz]", "Value": paf_f4_ec},
    {"Metric":"Alpha Flexibility O1 (EO↔EO) [%]", "Value": alpha_flex_o1},
])

with tab_metrics:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🧮 Derived metrics</div>', unsafe_allow_html=True)
    st.dataframe(derived_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# KPI tiles
with tab_metrics:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("<div class='kpi'><div class='label'>Peak Alpha O1 (EC)</div><div class='value'>{}</div></div>"
                    .format("n/a" if paf_o1_ec is None or not np.isfinite(paf_o1_ec) else f"{paf_o1_ec:.2f} Hz"),
                    unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='kpi'><div class='label'>Alpha Shift O1 (EO→EC)</div><div class='value'>{}</div></div>"
                    .format("n/a" if alpha_shift_O1 is None or not np.isfinite(alpha_shift_O1) else f"{alpha_shift_O1:.1f}%"),
                    unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='kpi'><div class='label'>TBR O1 (EC)</div><div class='value'>{}</div></div>"
                    .format("n/a" if tbr_o1_ec is None or not np.isfinite(tbr_o1_ec) else f"{tbr_o1_ec:.2f}"),
                    unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='kpi'><div class='label'>Frontal TBR Symmetry</div><div class='value'>{}</div></div>"
                    .format("n/a" if frontal_tbr_sym is None or not np.isfinite(frontal_tbr_sym) else f"{frontal_tbr_sym:.2f}"),
                    unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# 4) QC TAB
# ──────────────────────────────────────────────────────────────────────────────
with tab_qc:
    st.markdown('<div class="app-card"><div class="card-title">🧪 Artifact quality control</div>', unsafe_allow_html=True)
    if qc_df.empty:
        st.info("No QC available.")
    else:
        st.dataframe(qc_df, use_container_width=True)
        low = qc_df[(qc_df["epochs_total"] > 0) & (qc_df["epochs_good"] < MIN_GOOD_EPOCHS)]
        if not low.empty:
            st.warning(f"{len(low)} file(s) have fewer than {MIN_GOOD_EPOCHS} good epochs. Consider longer/cleaner recordings.")
    st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# 5) DIALS
# ──────────────────────────────────────────────────────────────────────────────
sleep_score = mean_finite([
    None if alpha_shift_O1 is None else score_range(alpha_shift_O1, 50.0, 70.0),
    None if tbr_shift_O1   is None else score_shift(tbr_shift_O1),
])
emotional_score = mean_finite([
    None if beta_sym        is None else score_sym1(beta_sym, SYM_TOL),
    None if alpha_sym       is None else score_sym1(alpha_sym, SYM_TOL),
    None if frontal_tbr_sym is None else score_sym1(frontal_tbr_sym, 0.20),
])
cognitive_score = mean_finite([
    None if tbr_cz_eo       is None else score_range(tbr_cz_eo, 1.8, 2.2),
    None if tlb_cz_ec       is None else score_range(tlb_cz_ec, 1.6, 2.4),
    None if frontal_tbr_avg is None else score_range(frontal_tbr_avg, 1.8, 2.2),
])
stress_score = np.nan if all(v is None for v in [
    beta_sym, frontal_tbr_avg, alpha_shift_Cz
]) else 1.0 - mean_finite([
    None if beta_sym        is None else score_sym1(beta_sym, SYM_TOL),
    None if frontal_tbr_avg is None else score_range(frontal_tbr_avg, 1.8, 2.2),
    None if alpha_shift_Cz  is None else score_range(alpha_shift_Cz, 50.0, 70.0),
])

with tab_dials:
    st.markdown('<div class="app-card"><div class="card-title">🧭 Dials</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        ui_gauge("Emotional", emotional_score)
        ui_gauge("Cognitive", cognitive_score)
    with c2:
        ui_gauge("Sleep", sleep_score)
        ui_gauge("Stress/Trauma", stress_score, invert=True)
    if SHOW_DIAL_DEBUG:
        with st.expander("See dial component details", expanded=False):
            st.write({
                "Sleep": {"alpha_shift_O1%": alpha_shift_O1, "tbr_shift_O1%": tbr_shift_O1},
                "Cognitive": {"TBR_Cz_EO": tbr_cz_eo, "Theta/LowBeta_Cz_EC": tlb_cz_ec, "Frontal_TBR_avg_EC": frontal_tbr_avg},
                "Emotional": {"Beta_sym_F3/F4": beta_sym, "Alpha_sym_F3/F4": alpha_sym, "TBR_sym_F3/F4": frontal_tbr_sym}
            })
    st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# 6) SLIDERS
# ──────────────────────────────────────────────────────────────────────────────
with tab_sliders:
    st.markdown('<div class="app-card"><div class="card-title">🎚️ Sliders</div>', unsafe_allow_html=True)
    region = st.radio("Region", ["Posterior","Central","Anterior","Symmetry"], index=0, horizontal=True)

    def posterior():
        ui_slider_row("Alpha Shift O1 (EO → EC)", alpha_shift_O1, -100, 200, unit="%",
                      green=[(50,70)], yellow=[(40,50),(70,85)], red=[(-100,40),(85,200)], ticks=[-100,50,70,200],
                      key="alpha_shift_o1", show_note=ENABLE_NOTES, show_badge=DISPLAY_INDICES)
        ui_slider_row("Alpha Flexibility O1 (EO ↔ EO)", alpha_flex_o1, -100, 250, unit="%",
                      green=[(0,25)], yellow=[(-10,0),(25,40)], red=[(-100,-10),(40,250)], ticks=[-100,0,25,250],
                      key="alpha_flex_o1", show_note=ENABLE_NOTES, show_badge=DISPLAY_INDICES)
        ui_slider_row("Peak Alpha O1 (EC)", paf_o1_ec, 7, 13, unit=" Hz",
                      green=[(9.5,13.0)], yellow=[(9.0,9.5)], red=[(7.0,9.0)], ticks=[7,9.5,13],
                      key="paf_o1_ec", show_note=ENABLE_NOTES, show_badge=DISPLAY_INDICES)
        ui_slider_row("TBR Shift O1 (EO → EC)", tbr_shift_O1, -100, 100, unit="%",
                      green=[(0,100)], yellow=[(-25,0)], red=[(-100,-25)], ticks=[-100,-25,0,100],
                      key="tbr_shift_o1", show_note=ENABLE_NOTES, show_badge=DISPLAY_INDICES)
        ui_slider_row("TBR O1 (EO)", tbr_o1_eo, 0, 5,
                      green=[(1.2,2.7)], yellow=[(0.9,1.2),(2.7,3.0)], red=[(0.0,0.9),(3.0,5.0)],
                      ticks=[0,0.9,1.2,2.7,3.0,5.0], key="tbr_o1_eo", show_note=ENABLE_NOTES, show_badge=DISPLAY_INDICES)
        ui_slider_row("TBR O1 (EC)", tbr_o1_ec, 0, 5,
                      green=[(1.2,2.7)], yellow=[(0.9,1.2),(2.7,3.0)], red=[(0.0,0.9),(3.0,5.0)],
                      ticks=[0,0.9,1.2,2.7,3.0,5.0], key="tbr_o1_ec", show_note=ENABLE_NOTES, show_badge=DISPLAY_INDICES)

    def central():
        ui_slider_row("Alpha Shift Cz (EO → EC)", alpha_shift_Cz, -100, 200, unit="%",
                      green=[(50,70)], yellow=[(40,50),(70,85)], red=[(-100,40),(85,200)], ticks=[-100,50,70,200],
                      key="alpha_shift_cz", show_note=ENABLE_NOTES, show_badge=DISPLAY_INDICES)
        ui_slider_row("Peak Alpha Cz (EC)", paf_cz_ec, 7, 13, unit=" Hz",
                      green=[(9.5,13.0)], yellow=[(9.0,9.5)], red=[(7.0,9.0)], ticks=[7,9.5,13],
                      key="paf_cz_ec", show_note=ENABLE_NOTES, show_badge=DISPLAY_INDICES)
        ui_slider_row("TBR Cz (EO)", tbr_cz_eo, 0, 5,
                      green=[(1.8,2.2)], yellow=[(1.6,1.8),(2.2,2.4)], red=[(0.0,1.6),(2.4,5.0)],
                      ticks=[0,1.6,1.8,2.2,2.4,5.0], key="tbr_cz_eo", show_note=ENABLE_NOTES, show_badge=DISPLAY_INDICES)
        ui_slider_row("Theta/Low-Beta Cz (EC)", tlb_cz_ec, 0.5, 3.5,
                      green=[(1.6,2.4)], yellow=[(1.4,1.6),(2.4,2.6)], red=[(0.5,1.4),(2.6,3.5)],
                      ticks=[0.5,1.6,2.4,3.5], key="tlb_cz_ec", show_note=ENABLE_NOTES, show_badge=DISPLAY_INDICES)

    def anterior():
        for ch in ["F3","F4"]:
            ta     = get_val(ch, "EyesClosed", "ThetaAlpha")
            tbr    = get_val(ch, "EyesClosed", "TBR")
            bsplit = get_val(ch, "EyesClosed", "BetaSplit_15_20_over_20_30")
            paf    = paf_f3_ec if ch=="F3" else paf_f4_ec
            st.markdown(f"<div class='badge'>Frontal {ch} (EC)</div>", unsafe_allow_html=True)
            ui_slider_row(f"Theta/Alpha {ch} (EC)", ta, 0.2, 3.0, ticks=[0.2,1.0,3.0],
                          key=f"ta_{ch}_ec", show_note=ENABLE_NOTES, show_badge=DISPLAY_INDICES)
            ui_slider_row(f"Peak Alpha {ch} (EC)", paf, 7, 13, unit=" Hz",
                          green=[(9.5,13.0)], yellow=[(9.0,9.5)], red=[(7.0,9.0)], ticks=[7,9.5,13],
                          key=f"paf_{ch}_ec", show_note=ENABLE_NOTES, show_badge=DISPLAY_INDICES)
            ui_slider_row(f"TBR {ch} (EC)", tbr, 0.5, 4.0,
                          green=[(1.8,2.2)], yellow=[(1.6,1.8),(2.2,2.4)], red=[(0.5,1.6),(2.4,4.0)],
                          ticks=[0.5,1.6,1.8,2.2,2.4,4.0], key=f"tbr_{ch}_ec",
                          show_note=ENABLE_NOTES, show_badge=DISPLAY_INDICES)
            ui_slider_row(f"Beta 15–20 / 20–30 {ch} (EC)", bsplit, 0.2, 2.0,
                          ticks=[0.2,1.0,2.0], key=f"betasplit_{ch}_ec",
                          show_note=ENABLE_NOTES, show_badge=DISPLAY_INDICES)

    def symmetry_sliders():
        ui_slider_row("Theta Symmetry F3/F4 (EC)", theta_sym, 0.4, 1.6,
                      green=[(1.0-SYM_TOL, 1.0+SYM_TOL)],
                      yellow=[(1.0-1.25*SYM_TOL, 1.0-SYM_TOL), (1.0+SYM_TOL, 1.0+1.25*SYM_TOL)],
                      red=[(0.4, 1.0-1.25*SYM_TOL), (1.0+1.25*SYM_TOL, 1.6)],
                      ticks=[0.4, 1.0-SYM_TOL, 1.0, 1.0+SYM_TOL, 1.6],
                      key="theta_sym", show_note=ENABLE_NOTES, show_badge=DISPLAY_INDICES)
        ui_slider_row("Alpha Symmetry F3/F4 (EC)", alpha_sym, 0.4, 1.6,
                      green=[(1.0-SYM_TOL, 1.0+SYM_TOL)],
                      yellow=[(1.0-1.25*SYM_TOL, 1.0-SYM_TOL), (1.0+SYM_TOL, 1.0+1.25*SYM_TOL)],
                      red=[(0.4, 1.0-1.25*SYM_TOL), (1.0+1.25*SYM_TOL, 1.6)],
                      ticks=[0.4, 1.0-SYM_TOL, 1.0, 1.0+SYM_TOL, 1.6],
                      key="alpha_sym", show_note=ENABLE_NOTES, show_badge=DISPLAY_INDICES)
        ui_slider_row("Beta (13–21) Symmetry F3/F4 (EC)", beta_sym, 0.4, 1.6,
                      green=[(1.0-SYM_TOL, 1.0+SYM_TOL)],
                      yellow=[(1.0-1.25*SYM_TOL, 1.0-SYM_TOL), (1.0+SYM_TOL, 1.0+1.25*SYM_TOL)],
                      red=[(0.4, 1.0-1.25*SYM_TOL), (1.0+1.25*SYM_TOL, 1.6)],
                      ticks=[0.4, 1.0-SYM_TOL, 1.0, 1.0+SYM_TOL, 1.6],
                      key="beta_sym", show_note=ENABLE_NOTES, show_badge=DISPLAY_INDICES)

    if region == "Posterior": posterior()
    elif region == "Central": central()
    elif region == "Anterior": anterior()
    else: symmetry_sliders()
    st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# 7) PSD VIEWER
# ──────────────────────────────────────────────────────────────────────────────
with tab_psd:
    st.markdown('<div class="app-card"><div class="card-title">📈 Power Spectral Density</div>', unsafe_allow_html=True)
    sel_channel = st.selectbox("Channel", sorted(list(VALID_CHANNELS)))
    cols = st.columns(2)
    for i, cond in enumerate(["EyesOpen", "EyesClosed"]):
        with cols[i]:
            key = psd_key_map.get((sel_channel, cond))
            if key is None or key not in psd_cache:
                st.info(f"No data for {sel_channel} {cond}"); continue
            f, pxx = psd_cache[key]
            m = f <= F_MAX_VIEW
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=f[m], y=pxx[m], mode="lines", name=f"{sel_channel} {cond}"))
            fig.update_layout(margin=dict(l=10,r=10,t=30,b=10),
                              title=f"{sel_channel} — {cond}",
                              xaxis_title="Frequency (Hz)", yaxis_title="PSD (µV²/Hz)")
            st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# 8) EXPORT (CSV + PDF with captions & aligned sliders)
# ──────────────────────────────────────────────────────────────────────────────
with tab_export:
    st.markdown('<div class="app-card"><div class="card-title">⬇️ Export all metrics</div>', unsafe_allow_html=True)

    c0, c1 = st.columns([0.7, 0.3])
    with c0:
        export_name_input = st.text_input("Export name", value=st.session_state.get("export_name", "neurobiqs_report"))
        st.session_state["export_name"] = export_name_input
    with c1:
        append_ts = st.checkbox("Append date & time", value=True)

    base = make_export_basename(export_name_input)
    if append_ts:
        base = f"{base}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    csv_filename = f"{base}.csv"
    pdf_filename = f"{base}.pdf"

    out = bands_df.copy()
    out["AlphaShift_O1_pct"] = alpha_shift_O1
    out["TBR_Shift_O1_pct"]  = tbr_shift_O1
    out["AlphaShift_Cz_pct"] = alpha_shift_Cz
    out["Theta_Symmetry_F3_over_F4_EC"] = theta_sym
    out["Alpha_Symmetry_F3_over_F4_EC"]  = alpha_sym
    out["Beta_Symmetry_F3_over_F4_EC"]   = beta_sym
    out["PeakAlpha_O1_EC_Hz"] = paf_o1_ec
    out["PeakAlpha_Cz_EC_Hz"] = paf_cz_ec
    out["PeakAlpha_F3_EC_Hz"] = paf_f3_ec
    out["PeakAlpha_F4_EC_Hz"] = paf_f4_ec
    out["AlphaFlexibility_O1_EO_EO_pct"] = alpha_flex_o1
    out["Dial_Emotional_0_1"] = emotional_score
    out["Dial_Sleep_0_1"]     = sleep_score
    out["Dial_Cognitive_0_1"] = cognitive_score
    out["Dial_StressTrauma_0_1_higher_worse"] = stress_score

    # Merge QC into export
    if not qc_df.empty:
        out = out.merge(
            qc_df[["file","epochs_total","epochs_good","clean_pct","rej_amp","rej_blink","rej_muscle","rej_flat","used_original"]],
            on="file", how="left"
        )

    st.download_button("Download CSV", data=out.to_csv(index=False).encode("utf-8"),
                       file_name=csv_filename, mime="text/csv")

    # Capability probes for PDF
    try:
        import reportlab  # noqa: F401
        PDF_HAS_REPORTLAB = True
    except Exception:
        PDF_HAS_REPORTLAB = False
    try:
        _ = go.Figure().to_image(format="png")  # kaleido probe
        PDF_HAS_KALEIDO = True
    except Exception:
        PDF_HAS_KALEIDO = False

    st.caption(
        f"PDF export status — ReportLab: {'✅' if PDF_HAS_REPORTLAB else '❌'} · "
        f"Plots (kaleido): {'✅' if PDF_HAS_KALEIDO else '❌'}"
    )

    def build_pdf_bytes() -> Optional[bytes]:
        if not PDF_HAS_REPORTLAB:
            st.info("Install ReportLab to enable PDF:  pip install reportlab")
            return None

        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors

        # helper: df -> wrapped table
        def pdf_table_from_df(
            df: pd.DataFrame,
            page_width: float,
            col_ratios: Optional[List[float]] = None,
            header_bg=colors.lightgrey,
            font_size: int = 8
        ) -> Table:
            sty_h = ParagraphStyle("hdr", fontSize=font_size, leading=font_size+1, wordWrap="CJK", spaceAfter=0)
            sty_b = ParagraphStyle("bod", fontSize=font_size, leading=font_size+1, wordWrap="CJK", spaceAfter=0)
            data: List[List] = [[Paragraph(str(c), sty_h) for c in df.columns]]
            for _, row in df.iterrows():
                cells = []
                for c in df.columns:
                    v = row[c]
                    if isinstance(v, (int, float, np.floating)) and np.isfinite(v):
                        cells.append(Paragraph(f"{float(v):.4g}", sty_b))
                    else:
                        cells.append(Paragraph(str(v), sty_b))
                data.append(cells)
            col_w = [page_width/len(df.columns)] * len(df.columns) if col_ratios is None else [page_width*r for r in col_ratios]
            t = Table(data, colWidths=col_w, repeatRows=1, hAlign="LEFT")
            ts = [
                ("BACKGROUND",(0,0),(-1,0), header_bg),
                ("GRID",(0,0),(-1,-1), 0.25, colors.grey),
                ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
                ("FONTSIZE",(0,0),(-1,-1), font_size),
                ("VALIGN",(0,0),(-1,-1),"TOP"),
            ]
            for j, c in enumerate(df.columns):
                if pd.api.types.is_numeric_dtype(df[c]):
                    ts.append(("ALIGN",(j,1),(j,-1),"RIGHT"))
            t.setStyle(TableStyle(ts))
            return t

        # Build document
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                topMargin=2.0*cm, bottomMargin=2.0*cm,
                                leftMargin=1.6*cm, rightMargin=1.6*cm)
        styles = getSampleStyleSheet()
        H  = styles["Heading1"]; H.fontSize = 16
        H2 = styles["Heading2"]; H2.spaceBefore = 10; H2.fontSize = 13
        P  = styles["BodyText"]; P.leading = 14
        page_width = doc.width

        elems: List = []
        elems.append(Paragraph(f"{APP_NAME}", H))
        elems.append(Paragraph(f"v{APP_VERSION} · Exported {datetime.now().strftime('%Y-%m-%d %H:%M')}", P))
        elems.append(Paragraph(
            "Disclaimer: Research/education only. Not a medical device or diagnostic. "
            "Welch PSD with Hann windows; artifact screening via amplitude/blink/muscle/flat rules.", P))
        elems.append(Spacer(1, 8))

        elems.append(Paragraph(
            f"Sampling: fs={'override ' + str(int(FS_OVERRIDE)) + ' Hz' if FS_OVERRIDE > 0 else 'auto-estimated'}, "
            f"Welch: nperseg={int(N_PERSEG)}, noverlap={int(N_OVERLAP)}, "
            f"Filters: {'BP ' + str(BP_LOW) + '–' + str(BP_HIGH) + ' Hz' if BP_ENABLE else 'none'}, "
            f"Notch: {NOTCH}, Units: signals processed in µV.", P))

        # Dial scores table
        dial_rows = [
            ["Dial", "Score (0–1)", "Score %"],
            ["Emotional", f"{emotional_score:.2f}" if np.isfinite(emotional_score) else "n/a",
             f"{100*emotional_score:.1f}%" if np.isfinite(emotional_score) else "n/a"],
            ["Sleep", f"{sleep_score:.2f}" if np.isfinite(sleep_score) else "n/a",
             f"{100*sleep_score:.1f}%" if np.isfinite(sleep_score) else "n/a"],
            ["Cognitive", f"{cognitive_score:.2f}" if np.isfinite(cognitive_score) else "n/a",
             f"{100*cognitive_score:.1f}%" if np.isfinite(cognitive_score) else "n/a"],
            ["Stress/Trauma", f"{stress_score:.2f}" if np.isfinite(stress_score) else "n/a",
             f"{100*stress_score:.1f}%" if np.isfinite(stress_score) else "n/a"],
        ]
        dial_tbl = Table(dial_rows, hAlign="LEFT",
                         colWidths=[page_width*0.45, page_width*0.22, page_width*0.22])
        dial_tbl.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0), colors.lightgrey),
            ("GRID",(0,0),(-1,-1), 0.25, colors.grey),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("FONTSIZE",(0,0),(-1,-1), 9),
            ("VALIGN",(0,0),(-1,-1),"TOP"),
        ]))
        elems.append(Paragraph("Dial scores", H2))
        elems.append(dial_tbl)
        elems.append(Spacer(1, 8))

        # Dial plots (with titles) — slim
        try:
            _ = go.Figure().to_image(format="png")
            HAVE_KALEIDO = True
        except Exception:
            HAVE_KALEIDO = False

        if HAVE_KALEIDO:
            def fig_to_img(fig, width_pts: float):
                png = fig.to_image(format="png", scale=2)
                img = Image(io.BytesIO(png))
                img._restrictSize(width_pts, 10000)
                return img

            dials = [
                gauge_figure("Emotional", emotional_score, height=140, width=160, thickness=0.16, title_size=11),
                gauge_figure("Sleep",      sleep_score,     height=140, width=160, thickness=0.16, title_size=11),
                gauge_figure("Cognitive",  cognitive_score, height=140, width=160, thickness=0.16, title_size=11),
                gauge_figure("Stress/Trauma", stress_score, invert=True,
                             height=140, width=160, thickness=0.16, title_size=11),
            ]
            col_w = page_width/3 - 4
            imgs = [fig_to_img(f, col_w) for f in dials]
            rows = [imgs[:3], imgs[3:] + [Spacer(1,1),]] if len(imgs) > 3 else [imgs]
            for r in rows:
                elems.append(Table([r], colWidths=[col_w]*len(r)))
                elems.append(Spacer(1, 4))
        else:
            elems.append(Paragraph("Dial plots unavailable (install kaleido to embed).", P))

        # Derived metrics table
        elems.append(Paragraph("Derived metrics", H2))
        elems.append(pdf_table_from_df(pd.DataFrame(derived_df), page_width, col_ratios=[0.65, 0.30], font_size=9))
        elems.append(Spacer(1, 6))

        # QC table
        if not qc_df.empty:
            elems.append(Paragraph("Artifact QC (per file)", H2))
            qc_cols = ["file","channel","condition","epochs_total","epochs_good","clean_pct","rej_amp","rej_blink","rej_muscle","rej_flat","used_original"]
            qct = pdf_table_from_df(qc_df[qc_cols], page_width, font_size=8)
            elems.append(qct)
            elems.append(Spacer(1, 6))

        # Band powers (slim columns)
        elems.append(Paragraph("Band powers & core ratios", H2))
        slim = bands_df[["file","channel","condition","Theta","Alpha","Beta","LowBeta","TBR"]].copy()
        elems.append(pdf_table_from_df(
            slim, page_width,
            col_ratios=[0.33, 0.08, 0.10, 0.11, 0.11, 0.11, 0.08, 0.08],
            font_size=8
        ))
        elems.append(PageBreak())

        # PSD overlays
        if HAVE_KALEIDO:
            def psd_overlay_fig(channel: str):
                fig = go.Figure()
                for cond, label in [("EyesOpen","EO"), ("EyesClosed","EC")]:
                    key = psd_key_map.get((channel, cond))
                    if key in psd_cache:
                        f, pxx = psd_cache[key]
                        m = f <= F_MAX_VIEW
                        fig.add_trace(go.Scatter(x=f[m], y=pxx[m], mode="lines", name=f"{channel} {label}"))
                fig.update_layout(
                    title=f"PSD — {channel} (EO vs EC)",
                    xaxis_title="Frequency (Hz)", yaxis_title="PSD (µV²/Hz)",
                    margin=dict(l=20,r=10,t=40,b=30),
                    height=260, width=740,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0)
                )
                return fig

            elems.append(Paragraph("PSD overlays (EO vs EC)", H2))
            col_w = page_width*0.49
            row = []
            for ch in sorted(list(VALID_CHANNELS)):
                fig = psd_overlay_fig(ch)
                img = Image(io.BytesIO(fig.to_image(format="png", scale=2)))
                img._restrictSize(col_w, 10000)
                row.append(img)
                if len(row) == 2:
                    elems.append(Table([row], colWidths=[col_w, col_w]))
                    elems.append(Spacer(1, 6))
                    row = []
            if row:
                elems.append(Table([row], colWidths=[col_w]))
        else:
            elems.append(Paragraph("PSD plots unavailable (install kaleido to embed).", P))

        # Slider snapshots — captions + fixed heights for perfect alignment
        if HAVE_KALEIDO:
            elems.append(PageBreak())
            elems.append(Paragraph("Slider snapshots", H2))

            slider_img_w = page_width * 0.49
            slider_img_h = 82
            cap_style = ParagraphStyle("cap", fontSize=9, leading=10, alignment=1)

            def sfig(title, value, lo, hi, **kw):
                return build_slider_figure(title, value, lo, hi,
                                           height=78, bar_thickness=0.12, marker_size=8, **kw)

            specs: List[Tuple[str, go.Figure]] = []
            # Posterior
            specs += [
                ("Alpha Shift O1 (EO → EC)", sfig("Alpha Shift O1 (EO → EC)", alpha_shift_O1, -100, 200,
                                                  green=[(50,70)], yellow=[(40,50),(70,85)],
                                                  red=[(-100,40),(85,200)], ticks=[-100,50,70,200], unit="%")),
                ("Alpha Flexibility O1 (EO ↔ EO)", sfig("Alpha Flexibility O1 (EO ↔ EO)", alpha_flex_o1, -100, 250,
                                                        green=[(0,25)], yellow=[(-10,0),(25,40)],
                                                        red=[(-100,-10),(40,250)], ticks=[-100,0,25,250], unit="%")),
                ("Peak Alpha O1 (EC)", sfig("Peak Alpha O1 (EC)", paf_o1_ec, 7, 13,
                                            green=[(9.5,13.0)], yellow=[(9.0,9.5)], red=[(7.0,9.0)],
                                            ticks=[7,9.5,13], unit=" Hz")),
                ("TBR Shift O1 (EO → EC)", sfig("TBR Shift O1 (EO → EC)", tbr_shift_O1, -100, 100,
                                                green=[(0,100)], yellow=[(-25,0)], red=[(-100,-25)],
                                                ticks=[-100,-25,0,100], unit="%")),
                ("TBR O1 (EO)", sfig("TBR O1 (EO)", tbr_o1_eo, 0, 5,
                                     green=[(1.2,2.7)], yellow=[(0.9,1.2),(2.7,3.0)],
                                     red=[(0.0,0.9),(3.0,5.0)], ticks=[0,0.9,1.2,2.7,3.0,5.0])),
                ("TBR O1 (EC)", sfig("TBR O1 (EC)", tbr_o1_ec, 0, 5,
                                     green=[(1.2,2.7)], yellow=[(0.9,1.2),(2.7,3.0)],
                                     red=[(0.0,0.9),(3.0,5.0)], ticks=[0,0.9,1.2,2.7,3.0,5.0])),
            ]
            # Central
            specs += [
                ("Alpha Shift Cz (EO → EC)", sfig("Alpha Shift Cz (EO → EC)", alpha_shift_Cz, -100, 200,
                                                  green=[(50,70)], yellow=[(40,50),(70,85)],
                                                  red=[(-100,40),(85,200)], ticks=[-100,50,70,200], unit="%")),
                ("Peak Alpha Cz (EC)", sfig("Peak Alpha Cz (EC)", paf_cz_ec, 7, 13,
                                            green=[(9.5,13.0)], yellow=[(9.0,9.5)], red=[(7.0,9.0)],
                                            ticks=[7,9.5,13], unit=" Hz")),
                ("TBR Cz (EO)", sfig("TBR Cz (EO)", tbr_cz_eo, 0, 5,
                                     green=[(1.8,2.2)], yellow=[(1.6,1.8),(2.2,2.4)],
                                     red=[(0.0,1.6),(2.4,5.0)], ticks=[0,1.6,1.8,2.2,2.4,5.0])),
                ("Theta/Low-Beta Cz (EC)", sfig("Theta/Low-Beta Cz (EC)", tlb_cz_ec, 0.5, 3.5,
                                                green=[(1.6,2.4)], yellow=[(1.4,1.6),(2.4,2.6)],
                                                red=[(0.5,1.4),(2.6,3.5)], ticks=[0.5,1.6,2.4,3.5])),
            ]
            # Anterior
            specs += [
                ("Theta/Alpha F3 (EC)", sfig("Theta/Alpha F3 (EC)", get_val("F3","EyesClosed","ThetaAlpha"), 0.2, 3.0, ticks=[0.2,1.0,3.0])),
                ("Peak Alpha F3 (EC)",  sfig("Peak Alpha F3 (EC)", paf_f3_ec, 7, 13,
                                             green=[(9.5,13.0)], yellow=[(9.0,9.5)], red=[(7.0,9.0)],
                                             ticks=[7,9.5,13], unit=" Hz")),
                ("TBR F3 (EC)",         sfig("TBR F3 (EC)", get_val("F3","EyesClosed","TBR"), 0.5, 4.0,
                                             green=[(1.8,2.2)], yellow=[(1.6,1.8),(2.2,2.4)],
                                             red=[(0.5,1.6),(2.4,4.0)], ticks=[0.5,1.6,1.8,2.2,2.4,4.0])),
                ("Beta 15–20 / 20–30 F3 (EC)", sfig("Beta 15–20 / 20–30 F3 (EC)",
                                                    get_val("F3","EyesClosed","BetaSplit_15_20_over_20_30"),
                                                    0.2, 2.0, ticks=[0.2,1.0,2.0])),
                ("Theta/Alpha F4 (EC)", sfig("Theta/Alpha F4 (EC)", get_val("F4","EyesClosed","ThetaAlpha"), 0.2, 3.0, ticks=[0.2,1.0,3.0])),
                ("Peak Alpha F4 (EC)",  sfig("Peak Alpha F4 (EC)", paf_f4_ec, 7, 13,
                                             green=[(9.5,13.0)], yellow=[(9.0,9.5)], red=[(7.0,9.0)],
                                             ticks=[7,9.5,13], unit=" Hz")),
                ("TBR F4 (EC)",         sfig("TBR F4 (EC)", get_val("F4","EyesClosed","TBR"), 0.5, 4.0,
                                             green=[(1.8,2.2)], yellow=[(1.6,1.8),(2.2,2.4)],
                                             red=[(0.5,1.6),(2.4,4.0)], ticks=[0.5,1.6,1.8,2.2,2.4,4.0])),
                ("Beta 15–20 / 20–30 F4 (EC)", sfig("Beta 15–20 / 20–30 F4 (EC)",
                                                    get_val("F4","EyesClosed","BetaSplit_15_20_over_20_30"),
                                                    0.2, 2.0, ticks=[0.2,1.0,2.0])),
            ]

            # render 2 per row with captions
            row_cells = []
            from reportlab.platypus import Table, TableStyle, Paragraph  # type: ignore
            from reportlab.lib.styles import ParagraphStyle  # type: ignore
            for title, fig in specs:
                png = fig.to_image(format="png", scale=2)
                from reportlab.platypus import Image  # type: ignore
                img = Image(io.BytesIO(png))
                img.drawWidth = slider_img_w
                img.drawHeight = slider_img_h
                cell = Table(
                    [[img], [Paragraph(title, cap_style)]],
                    colWidths=[slider_img_w],
                    rowHeights=[slider_img_h, 14],
                    hAlign="CENTER"
                )
                cell.setStyle(TableStyle([
                    ("ALIGN",(0,0),(-1,-1),"CENTER"),
                    ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                    ("TOPPADDING",(0,0),(-1,-1),2),
                    ("BOTTOMPADDING",(0,0),(-1,-1),2),
                ]))
                row_cells.append(cell)
                if len(row_cells) == 2:
                    elems.append(Table([row_cells], colWidths=[slider_img_w, slider_img_w]))
                    elems.append(Spacer(1, 4))
                    row_cells = []
            if row_cells:
                elems.append(Table([row_cells], colWidths=[slider_img_w]))

        # Footer note
        elems.append(Spacer(1, 6))
        elems.append(Paragraph(
            f"Generated by {APP_NAME} v{APP_VERSION}. This report aggregates spectral features and heuristics; clinical interpretation is required.",
            P
        ))

        # build
        doc.build(elems)
        return buf.getvalue()

    try:
        pdf_bytes = build_pdf_bytes()
        if pdf_bytes:
            st.download_button("📄 Download PDF report", data=pdf_bytes,
                               file_name=pdf_filename, mime="application/pdf")
        else:
            if PDF_HAS_REPORTLAB:
                st.warning("PDF was not generated (no data or a silent error).")
    except Exception as e:
        st.error(f"PDF generation failed: {e}")

    st.markdown('</div>', unsafe_allow_html=True)
