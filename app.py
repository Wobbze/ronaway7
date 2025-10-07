from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

from scipy.signal import butter, filtfilt, iirnotch, welch

st.set_page_config(
    page_title="EEG — FFT Spectra",
    layout="wide",
    menu_items={
        "Get help": None,
        "Report a Bug": None,
        "About": "EEG Spectra Dashboard"
    }
)

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

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------
st.sidebar.header("Upload")

uploaded_files = st.sidebar.file_uploader(
    "Upload one or more CSV files (2 columns: timestamp, signal[µV])",
    type=["csv"],
    accept_multiple_files=True
)

# --- Preprocessing (global) ---
st.sidebar.markdown("### Pre-processing")
do_notch = st.sidebar.checkbox(
    "Apply 50 Hz notch", value=True,
    help="Filter out 50 Hz powerline noise (EU)."
)
do_bandpass = st.sidebar.checkbox(
    "Apply 0.5–45 Hz band-pass", value=True,
    help="4th-order Butterworth. Typical EEG passband."
)
st.sidebar.caption("Filters are applied before FFT and (if enabled) Welch PSD.")

# --- Preview (global) ---
show_time_preview = st.sidebar.checkbox("Show raw/preprocessed time-series preview", value=False)
preview_secs = st.sidebar.number_input("Preview seconds", min_value=0.0, value=60.0, step=10.0)

# --- FFT window (global) ---
win_choice = st.sidebar.selectbox(
    "FFT window",
    ["None (rectangular)", "Hann (recommended)"],
    index=1,
    help="Hann reduces leakage; amplitudes corrected."
)

# --- Frequency plotting (global) ---
fmax_plot = st.sidebar.number_input(
    "Max frequency shown (Hz, 0 = full Nyquist)",
    min_value=0.0, value=50.0, step=5.0
)
show_band_summary = st.sidebar.checkbox("Show band peak summary (FFT amplitude)", value=True)

# --- Welch PSD (global) ---
st.sidebar.markdown("---")
st.sidebar.subheader("Welch PSD")
do_welch = st.sidebar.checkbox("Compute Welch PSD", value=True,
                               help="Averaged PSD (µV²/Hz) for smoother, comparable spectra.")
welch_secs = st.sidebar.number_input("Welch window length (s)", min_value=1.0, value=4.0, step=1.0)
welch_overlap = st.sidebar.slider("Welch overlap (%)", min_value=0, max_value=90, value=50, step=5)

# --- FFT Smoothing (global) ---
st.sidebar.markdown("---")
st.sidebar.subheader("FFT Smoothing")
do_fft_smooth = st.sidebar.checkbox("Smooth FFT amplitude", value=True)
smooth_bw_hz = st.sidebar.number_input("Smoothing bandwidth (Hz)", min_value=0.0, value=1.0, step=0.5)
smooth_method_label = st.sidebar.selectbox(
    "Smoothing method",
    ["Moving average", "Savitzky–Golay", "Median"],
    help="Moving average = simple; Savitzky–Golay preserves peak shapes; Median is robust to spikes."
)
_METHOD_KEY = {"Moving average": "moving_avg", "Savitzky–Golay": "savgol", "Median": "median"}[smooth_method_label]
savgol_poly = None
if _METHOD_KEY == "savgol":
    savgol_poly = st.sidebar.slider("Savitzky–Golay polyorder", 1, 5, 2)

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

def plot_psd(f: np.ndarray, Pxx: np.ndarray, fmax: float | None, title_suffix: str = ""):
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.semilogy(f, Pxx)
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD (µV²/Hz)")
    ax.set_title(f"Welch PSD {title_suffix}")
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    if fmax and fmax > 0: ax.set_xlim(0, min(fmax, NYQUIST))
    else: ax.set_xlim(0, NYQUIST)
    style_band_axis(ax, EEG_BANDS, fmax)
    st.pyplot(fig, use_container_width=True)

def band_power_table(f: np.ndarray, Pxx: np.ndarray) -> pd.DataFrame:
    rows = []
    for name, (f1, f2) in EEG_BANDS.items():
        mask = (f >= f1) & (f <= f2)
        power = float(np.trapz(Pxx[mask], f[mask])) if mask.any() else 0.0
        rows.append({"Band": name, "Power (µV²)": power})
    return pd.DataFrame(rows)

# ---- FFT smoothing helper ----
def smooth_amp(freqs: np.ndarray, amp: np.ndarray, bandwidth_hz: float,
               method: str = "moving_avg", polyorder: int = 2) -> np.ndarray:
    """
    Smooth a magnitude spectrum over a given frequency bandwidth (Hz).

    Methods:
      - moving_avg: boxcar mean
      - savgol: Savitzky–Golay polynomial smoothing (preserves peak shape better)
      - median: median filter (robust to spikes)
    """
    if bandwidth_hz <= 0 or amp.size < 3:
        return amp

    df = float(np.median(np.diff(freqs))) if amp.size > 1 else 0.0
    if df <= 0:
        return amp

    w = max(1, int(round(bandwidth_hz / df)))
    if w >= amp.size:
        w = amp.size - 1
    w = max(1, w)

    if method == "moving_avg":
        kernel = np.ones(w, dtype=float) / float(w)
        return np.convolve(amp, kernel, mode="same")

    elif method == "savgol":
        win = w if w % 2 == 1 else w + 1
        win = max(polyorder + 3, win)
        if win >= amp.size:
            win = amp.size - (1 - amp.size % 2)  # largest odd < size
        from scipy.signal import savgol_filter
        return savgol_filter(amp, window_length=win, polyorder=polyorder, mode="interp")

    elif method == "median":
        from scipy.signal import medfilt
        win = w if w % 2 == 1 else w + 1
        win = min(win, amp.size - (1 - amp.size % 2))
        return medfilt(amp, kernel_size=win)

    return amp

# -------------------------------------------------------
# Main
# -------------------------------------------------------
st.title("🧠 EEG — FFT Spectra (µV)")

if not uploaded_files:
    st.info("Upload one or more CSV files to begin (columns: timestamp, signal[µV]).")
    st.stop()

tab_labels = [f.name for f in uploaded_files]
tabs = st.tabs(tab_labels)

for tab, up in zip(tabs, uploaded_files):
    with tab:
        st.subheader(f"File: {up.name}")

        # Read CSV safely
        try:
            df = pd.read_csv(up, header=0)
        except Exception as e:
            st.error(f"Could not read CSV '{up.name}': {e}")
            continue

        if df.shape[1] < 2:
            st.error("Expected two columns: [timestamp, signal].")
            continue

        # Parse to numeric & drop NaNs
        t_raw = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
        x_uV  = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()
        mask_ok = np.isfinite(t_raw) & np.isfinite(x_uV)
        if not mask_ok.any():
            st.error("No valid numeric data found after parsing. Check your file.")
            continue
        t_raw = t_raw[mask_ok]; x_uV = x_uV[mask_ok]

        # Convert timestamp to seconds (seconds / ticks / ms)
        t_s = to_seconds_from_timestamp(t_raw, FS)

        # Preprocess
        x_proc = apply_filters(x_uV, FS, notch_50=do_notch, bandpass_05_45=do_bandpass)

        # Summary metrics
        duration = float((t_s[-1] - t_s[0])) if t_s.size > 1 else 0.0
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Samples", f"{len(x_proc):,}")
        c2.metric("Duration (s)", f"{duration:.3f}")
        c3.metric("Sampling rate (Hz)", f"{FS:g}")
        c4.metric("Δt (ms)", f"{(1000.0/FS):.3f}")
        c5.metric("Nyquist (Hz)", f"{NYQUIST:g}")

        # Preview (optional)
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

        # -----------------------------
        # FULL SIGNAL — FFT Amplitude (with optional smoothing)
        # -----------------------------
        st.header("Full-Signal Amplitude Spectrum")
        window_name = "hann" if win_choice.startswith("Hann") else "rect"
        freqs_full, amp_full = amplitude_spectrum_uV(x_proc, FS, window=window_name)

        amp_to_plot = amp_full
        if do_fft_smooth and smooth_bw_hz > 0:
            amp_to_plot = smooth_amp(
                freqs_full, amp_full, smooth_bw_hz,
                method=_METHOD_KEY, polyorder=(savgol_poly or 2)
            )

        plot_amp(
            freqs_full, amp_to_plot,
            fmax=fmax_plot if fmax_plot > 0 else None,
            title_suffix=f"— {up.name} ({win_choice})"
        )

        # Download (use the plotted version so it matches the graph)
        out_fft_name = f"{up.name}_amplitude_spectrum"
        if do_fft_smooth and smooth_bw_hz > 0:
            out_fft_name += f"_smoothed_{smooth_method_label.replace(' ', '')}_{smooth_bw_hz:g}Hz"
        spec_df = pd.DataFrame({"Frequency (Hz)": freqs_full, "Amplitude (µV)": amp_to_plot})
        st.download_button(
            "⬇️ Download amplitude spectrum",
            data=spec_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{out_fft_name}.csv",
            mime="text/csv",
            key=f"dl_fft_{up.name}"
        )

        if show_band_summary:
            st.subheader("Band Peak Summary (FFT amplitude)")
            st.dataframe(band_peak_table(freqs_full, amp_to_plot), use_container_width=True)

        st.markdown("---")

        # -----------------------------
        # FULL SIGNAL — Welch PSD (optional)
        # -----------------------------
        if do_welch:
            st.header("Welch PSD — Full Signal")
            f_psd, Pxx = welch_psd(x_proc, FS, win_secs=welch_secs, overlap_pct=welch_overlap)
            plot_psd(
                f_psd, Pxx,
                fmax=fmax_plot if fmax_plot > 0 else None,
                title_suffix=f"{up.name} (win={welch_secs:g}s, overlap={welch_overlap}%)"
            )
            psd_df = pd.DataFrame({"Frequency (Hz)": f_psd, "PSD (µV^2/Hz)": Pxx})
            st.download_button(
                "⬇️ Download Welch PSD",
                data=psd_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{up.name}_welch_psd.csv",
                mime="text/csv",
                key=f"dl_psd_{up.name}"
            )
            st.subheader("Band Power Summary (from PSD)")
            st.dataframe(band_power_table(f_psd, Pxx), use_container_width=True)

