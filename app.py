from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Iterable, List

from scipy.signal import butter, filtfilt, iirnotch, welch

st.set_page_config(page_title="EEG — FFT Spectra", layout="wide")

# ---- Fixed recording parameters ----
FS: float = 250.0  # Hz
DT: float = 1.0 / FS
NYQUIST: float = FS / 2.0

# ---- EEG bands for x-axis annotation ----
EEG_BANDS: Dict[str, Tuple[float, float]] = {
    "Delta": (1.0, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 12.0),
    "Beta":  (13.0, 30.0),
    "Gamma": (30.0, 100.0),
}
st.set_page_config(
    page_title="EEG — FFT Spectra",
    layout="wide",
    menu_items={
        "Get help": None,       # hides the "Get help" link
        "Report a Bug": None,   # hides the "Report a bug" link
        "About": "EEG Spectra Dashboard"
    }
)
# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------
st.sidebar.header("Upload & Cuts")

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# --- Preprocessing ---
st.sidebar.markdown("### Pre-processing")
do_notch = st.sidebar.checkbox("Apply 50 Hz notch", value=True,
                               help="Use to filter out powerline noise in europe")
do_bandpass = st.sidebar.checkbox("Apply 0.5–45 Hz band-pass", value=True,
                                  help="4th-order Butterworth. Typical EEG passband.")
st.sidebar.caption("Filters are applied before FFT and (if enabled) Welch PSD.")

# --- Raw preview ---
show_time_preview = st.sidebar.checkbox("Show raw/preprocessed time-series preview", value=False)
preview_secs = st.sidebar.number_input("Preview seconds", min_value=0.0, value=180.0, step=10.0)

# --- FFT window ---
win_choice = st.sidebar.selectbox(
    "FFT window",
    ["None (rectangular)", "Hann (recommended)"],
    index=1,
    help="Hann reduces leakage between bands."
)

# --- Frequency plotting ---
fmax_plot = st.sidebar.number_input(
    "Max frequency shown (Hz, 0 = full Nyquist)",
    min_value=0.0, value=50.0, step=5.0
)

show_band_summary = st.sidebar.checkbox("Show band peak summary (FFT amplitude)", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Fixed-Length Cuts")
cut_len_s = st.sidebar.number_input(
    "Cut every X seconds",
    min_value=0.1, value=60.0, step=10.0,
    help="Back-to-back cuts (no overlap)."
)
include_last_partial = st.sidebar.checkbox(
    "Include final partial cut", value=True,
    help="If duration isn't a multiple of X, include the last shorter piece."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Welch PSD")
do_welch = st.sidebar.checkbox("Also compute Welch PSD", value=True,
                               help="Computes PSD.")
welch_secs = st.sidebar.number_input("Welch window length (s)", min_value=1.0, value=4.0, step=1.0)
welch_overlap = st.sidebar.slider("Welch overlap (%)", min_value=0, max_value=90, value=50, step=5)

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------
def to_seconds_from_timestamp(t_raw: np.ndarray, fs: float) -> np.ndarray:
    t_raw = t_raw.astype(float)
    if t_raw.size < 2:
        return t_raw
    dt_med = np.median(np.diff(t_raw))
    if np.isfinite(dt_med) and np.isclose(dt_med, 1.0/fs, rtol=0.05, atol=1e-6):
        return t_raw
    if np.isfinite(dt_med) and np.isclose(dt_med, 1.0, rtol=0.05, atol=1e-6):
        return t_raw / fs
    if np.isfinite(dt_med) and np.isclose(dt_med, 1000.0/fs, rtol=0.05, atol=1e-3):
        return t_raw / 1000.0
    st.warning("Timestamp scale unrecognized; using uniform time built from sampling rate.")
    n = t_raw.size
    return np.arange(n) / fs

def coherent_gain(window: np.ndarray) -> float:
    return float(np.sum(window)) / float(window.size)

def amplitude_spectrum_uV(x_uV: np.ndarray, fs: float, window: str = "hann") -> Tuple[np.ndarray, np.ndarray]:
    x = x_uV.astype(np.float64) - np.mean(x_uV)
    N = x.size
    if N < 2:
        return np.array([0.0]), np.array([0.0])
    if window.lower().startswith("hann"):
        w = np.hanning(N); cg = coherent_gain(w); xw = x * w
    else:
        cg = 1.0; xw = x
    X = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    amp = (np.abs(X) / (N / 2.0)) / cg
    amp[0] /= 2.0
    if N % 2 == 0 and amp.size > 1:
        amp[-1] /= 2.0
    return freqs, amp

def design_bandpass(low_hz: float, high_hz: float, fs: float, order: int = 4):
    ny = fs / 2.0
    low = max(1e-6, low_hz / ny)
    high = min(0.999999, high_hz / ny)
    b, a = butter(order, [low, high], btype="bandpass")
    return b, a

def apply_filters(x: np.ndarray, fs: float, notch_50: bool, bandpass_05_45: bool) -> np.ndarray:
    y = np.asarray(x, dtype=np.float64)
    if notch_50:
        b0, a0 = iirnotch(w0=50.0/(fs/2.0), Q=30.0)
        y = filtfilt(b0, a0, y, method="pad")
    if bandpass_05_45:
        b1, a1 = design_bandpass(0.5, 45.0, fs, order=4)
        y = filtfilt(b1, a1, y, method="pad")
    return y

def cut_indices_uniform(total_samples: int, fs: float, cut_len_s: float, include_last_partial: bool) -> Iterable[Tuple[int, int]]:
    samples_per_cut = max(1, int(round(cut_len_s * fs)))
    starts = np.arange(0, total_samples, samples_per_cut)
    for s in starts:
        e = int(min(total_samples, s + samples_per_cut))
        if e - s == samples_per_cut:
            yield int(s), int(e)
        else:
            if include_last_partial and e > s:
                yield int(s), int(e)

def style_band_axis(ax, bands: Dict[str, Tuple[float, float]], fmax: float | None):
    fmax_use = fmax if (fmax and fmax > 0) else ax.get_xlim()[1]
    for name, (f1, f2) in bands.items():
        if f2 <= 0 or f1 >= fmax_use:
            continue
        x1 = max(0.0, f1); x2 = min(fmax_use, f2)
        ax.axvspan(x1, x2, alpha=0.10)
        ax.axvline(x1, color="black", alpha=0.12, linewidth=0.8)
        ax.axvline(x2, color="black", alpha=0.12, linewidth=0.8)
    top = ax.twiny(); top.set_xlim(ax.get_xlim())
    centers, labels = [], []
    for name, (f1, f2) in bands.items():
        c = 0.5 * (f1 + f2)
        if 0 <= c <= fmax_use:
            centers.append(c); labels.append(name)
    top.set_xticks(centers, labels); top.tick_params(axis="x", direction="out", pad=0)
    top.xaxis.set_ticks_position("top")
    for spine in top.spines.values(): spine.set_alpha(0.2)

def plot_amp(freqs: np.ndarray, amp: np.ndarray, fmax: float | None, title_suffix: str = ""):
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.plot(freqs, amp)
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Amplitude (µV)")
    ax.set_title(f"Amplitude Spectrum {title_suffix}")
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    if fmax and fmax > 0: ax.set_xlim(0, min(fmax, NYQUIST))
    else: ax.set_xlim(0, NYQUIST)
    style_band_axis(ax, EEG_BANDS, fmax)
    st.pyplot(fig, use_container_width=True)

def band_peak_table(freqs: np.ndarray, amp: np.ndarray) -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []
    for name, (f1, f2) in EEG_BANDS.items():
        mask = (freqs >= f1) & (freqs <= f2)
        if not mask.any():
            rows.append({"Band": name, "Peak Freq (Hz)": np.nan, "Peak Amp (µV)": np.nan}); continue
        i = np.argmax(amp[mask]); rows.append({"Band": name,
            "Peak Freq (Hz)": float(freqs[mask][i]), "Peak Amp (µV)": float(amp[mask][i])})
    return pd.DataFrame(rows)

# ---- Welch helpers (µV²/Hz) ----
def welch_psd(signal_uV: np.ndarray, fs: float, win_secs: float, overlap_pct: int):
    nperseg = max(8, int(round(win_secs * fs)))
    noverlap = int(round(nperseg * (overlap_pct / 100.0)))
    f, Pxx = welch(
        signal_uV.astype(np.float64), fs=fs, window="hann",
        nperseg=nperseg, noverlap=noverlap, detrend="constant",
        return_onesided=True, scaling="density", average="median"
    )
    return f, Pxx

def welch_psd_for_segment(signal_uV: np.ndarray, fs: float, win_secs: float, overlap_pct: int):
    """
    Welch for a single cut: clamp nperseg to cut length and handle tiny segments gracefully.
    """
    seg_N = int(len(signal_uV))
    if seg_N < 8:
        return np.array([0.0]), np.array([np.nan])
    requested = int(round(win_secs * fs))
    nperseg = max(8, min(requested, seg_N))
    if requested > seg_N:
        st.warning(f"Welch window ({win_secs:.2f}s) is longer than this cut ({seg_N/fs:.2f}s). "
                   f"Using {nperseg/fs:.2f}s for this cut.", icon="⚠️")
    noverlap = int(round(nperseg * (overlap_pct / 100.0)))
    f, Pxx = welch(
        signal_uV.astype(np.float64), fs=fs, window="hann",
        nperseg=nperseg, noverlap=noverlap, detrend="constant",
        return_onesided=True, scaling="density", average="median"
    )
    return f, Pxx

def plot_psd(f: np.ndarray, Pxx: np.ndarray, fmax: float | None, title_suffix: str = ""):
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.semilogy(f, Pxx)
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (µV²/Hz)")
    ax.set_title(f"Welch PSD {title_suffix}")
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    if fmax and fmax > 0: ax.set_xlim(0, min(fmax, NYQUIST))
    else: ax.set_xlim(0, NYQUIIST)
    style_band_axis(ax, EEG_BANDS, fmax)
    st.pyplot(fig, use_container_width=True)

def band_power_table(f: np.ndarray, Pxx: np.ndarray) -> pd.DataFrame:
    rows = []
    for name, (f1, f2) in EEG_BANDS.items():
        mask = (f >= f1) & (f <= f2)
        power = float(np.trapz(Pxx[mask], f[mask])) if mask.any() else 0.0
        rows.append({"Band": name, "Power (µV²)": power})
    return pd.DataFrame(rows)

# -------------------------------------------------------
# Main
# -------------------------------------------------------
st.title("🧠 EEG — FFT Spectra (µV)")

if uploaded is None:
    st.info("Upload your CSV to begin (columns: timestamp, signal[µV]).")
    st.stop()

# Read CSV
try:
    df = pd.read_csv(uploaded, header=0)
except Exception as e:
    st.error(f"Could not read CSV: {e}"); st.stop()

if df.shape[1] < 2:
    st.error("Expected two columns: [timestamp, signal]."); st.stop()

time_col = df.columns[0]; signal_col = df.columns[1]

t_raw = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
x_uV  = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
mask_ok = np.isfinite(t_raw) & np.isfinite(x_uV)
if not mask_ok.any():
    st.error("No valid numeric data found after parsing. Check your file."); st.stop()
t_raw = t_raw[mask_ok]; x_uV = x_uV[mask_ok]

t_s = to_seconds_from_timestamp(t_raw, FS)

# Preprocess
x_proc = apply_filters(x_uV, FS, notch_50=do_notch, bandpass_05_45=do_bandpass)

# Preview
if show_time_preview:
    st.subheader("Time-Series Preview")
    n_prev = int(min(len(x_proc), preview_secs * FS)) if preview_secs > 0 else len(x_proc)
    fig_prev, ax_prev = plt.subplots(figsize=(12, 3))
    ax_prev.plot(t_s[:n_prev], x_uV[:n_prev], alpha=0.4, label="Raw")
    ax_prev.plot(t_s[:n_prev], x_proc[:n_prev], linewidth=1.2, label="Preprocessed")
    ax_prev.set_xlabel("Time (s)"); ax_prev.set_ylabel("Amplitude (µV)")
    ax_prev.set_title("EEG preview (raw vs preprocessed)")
    ax_prev.grid(True, which="both", linestyle="--", alpha=0.6)
    ax_prev.legend()
    st.pyplot(fig_prev, use_container_width=True)
    st.markdown("---")

# Summary
duration = float((t_s[-1] - t_s[0])) if t_s.size > 1 else 0.0
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Samples", f"{len(x_proc):,}")
c2.metric("Duration (s)", f"{duration:.3f}")
c3.metric("Sampling rate (Hz)", f"{FS:g}")
c4.metric("Δt (ms)", f"{(1000.0/FS):.3f}")
c5.metric("Nyquist (Hz)", f"{NYQUIST:g}")

# -----------------------------
# FULL SIGNAL — FFT Amplitude
# -----------------------------
st.header("Full-Signal Amplitude Spectrum")
window_name = "hann" if win_choice.startswith("Hann") else "rect"
freqs_full, amp_full = amplitude_spectrum_uV(x_proc, FS, window=window_name)
plot_amp(freqs_full, amp_full, fmax=fmax_plot if fmax_plot > 0 else None, title_suffix=f"— Full Signal ({win_choice})")

spec_df = pd.DataFrame({"Frequency (Hz)": freqs_full, "Amplitude (µV)": amp_full})
st.download_button("⬇️ Download amplitude spectrum (full signal)",
                   data=spec_df.to_csv(index=False).encode("utf-8"),
                   file_name="full_signal_amplitude_spectrum.csv", mime="text/csv")

if show_band_summary:
    st.subheader("Band Peak Summary — Full Signal (FFT amplitude)")
    st.dataframe(band_peak_table(freqs_full, amp_full), use_container_width=True)

st.markdown("---")

# -----------------------------
# FULL SIGNAL — Welch PSD (optional)
# -----------------------------
if do_welch:
    st.header("Welch PSD — Full Signal")
    f_psd, Pxx = welch_psd(x_proc, FS, win_secs=welch_secs, overlap_pct=welch_overlap)
    plot_psd(f_psd, Pxx, fmax=fmax_plot if fmax_plot > 0 else None,
             title_suffix=f"(win={welch_secs:g}s, overlap={welch_overlap}%)")
    psd_df = pd.DataFrame({"Frequency (Hz)": f_psd, "PSD (µV^2/Hz)": Pxx})
    st.download_button("⬇️ Download Welch PSD (full signal)",
                       data=psd_df.to_csv(index=False).encode("utf-8"),
                       file_name="full_signal_welch_psd.csv", mime="text/csv")
    st.subheader("Band Power Summary — Full Signal (from PSD)")
    st.dataframe(band_power_table(f_psd, Pxx), use_container_width=True)
    st.markdown("---")

# -----------------------------
# CUTS — FFT + Welch per cut
# -----------------------------
st.header("Cuts: Spectra & Downloads")

if cut_len_s <= 0:
    st.error("Cut length must be > 0 seconds."); st.stop()

total_samples = len(x_proc)
cuts = list(cut_indices_uniform(total_samples, FS, cut_len_s, include_last_partial))

if len(cuts) == 0:
    st.info("No cuts generated. Increase recording length or decrease the cut length.")
else:
    st.write(f"Generated **{len(cuts)}** cuts of **{cut_len_s:g} s** each"
             f"{' (including final partial cut)' if include_last_partial else ''}.")

    for i, (s_idx, e_idx) in enumerate(cuts, start=1):
        x_cut = x_proc[s_idx:e_idx]
        t_start = s_idx / FS
        t_end = (e_idx - 1) / FS if e_idx > s_idx else s_idx / FS
        t_dur = (e_idx - s_idx) / FS

        st.subheader(f"Cut {i} — {t_start:.3f}s → {t_end:.3f}s (≈ {t_dur:.3f}s)")

        # --- FFT amplitude for this cut ---
        freqs_c, amp_c = amplitude_spectrum_uV(x_cut, FS, window=window_name)
        plot_amp(freqs_c, amp_c, fmax=fmax_plot if fmax_plot > 0 else None,
                 title_suffix=f"— Cut {i} ({win_choice})")

        if show_band_summary:
            st.dataframe(band_peak_table(freqs_c, amp_c), use_container_width=True)

        cut_fft_df = pd.DataFrame({"Frequency (Hz)": freqs_c, "Amplitude (µV)": amp_c})
        st.download_button(
            label=f"⬇️ Download amplitude spectrum — Cut {i}",
            data=cut_fft_df.to_csv(index=False).encode("utf-8"),
            file_name=f"cut_{i}_amplitude_spectrum.csv",
            mime="text/csv",
            key=f"dl_fft_cut_{i}"
        )

        # --- Welch PSD for this cut (uses same Welch settings) ---
        if do_welch:
            f_c_psd, Pxx_c = welch_psd_for_segment(x_cut, FS, win_secs=welch_secs, overlap_pct=welch_overlap)
            plot_psd(f_c_psd, Pxx_c, fmax=fmax_plot if fmax_plot > 0 else None,
                     title_suffix=f"— Cut {i} (win={min(welch_secs, t_dur):.2f}s, overlap={welch_overlap}%)")

            # Band powers for this cut
            st.dataframe(band_power_table(f_c_psd, Pxx_c), use_container_width=True)

            # Download per-cut PSD
            cut_psd_df = pd.DataFrame({"Frequency (Hz)": f_c_psd, "PSD (µV^2/Hz)": Pxx_c})
            st.download_button(
                label=f"⬇️ Download Welch PSD — Cut {i}",
                data=cut_psd_df.to_csv(index=False).encode("utf-8"),
                file_name=f"cut_{i}_welch_psd.csv",
                mime="text/csv",
                key=f"dl_psd_cut_{i}"
            )

