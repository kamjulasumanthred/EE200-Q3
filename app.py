"""
app.py
=========================================================================
EE200: Signals, Systems and Networks — Course Project
Q3A 'Magical Mystery Tune' + Q3B 'Zapptain America'

A Shazam-style audio fingerprinting and song identification system,
wrapped in an interactive Gradio app with Single-Clip and Batch modes.

Pipeline: audio -> spectrogram -> constellation (peak-picking) ->
          combinatorial hashes -> offset-histogram matching

=========================================================================
HOW TO RUN
=========================================================================
1. Put your song library in a folder named "EE200 Project Song Database"
   (same folder as this script), containing .wav / .mp3 files.
   Do NOT rename the files — the filename (without extension) is the
   label your identifier must output.

2. Install dependencies:
       pip install -r requirements.txt

3. Run locally:
       python app.py

   The app indexes the database automatically on startup, then launches
   the Gradio UI with two tabs: "Single-Clip Mode" and "Batch Mode".

4. Deploy on Hugging Face Spaces (free, recommended for Gradio):
   - Create a new Space, SDK = Gradio.
   - Upload app.py, requirements.txt, packages.txt, and the
     "EE200 Project Song Database" folder (so the index ships with the app).
   - The Space will auto-build and launch; share the Space URL.
=========================================================================
"""

import os
import glob
import warnings

import numpy as np
import scipy.signal as signal
from scipy.ndimage import maximum_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import gradio as gr

# -------------------------------------------------------------------------
# COMPATIBILITY PATCH (Hugging Face Spaces / current Gradio + Pydantic)
# -------------------------------------------------------------------------
import gradio_client.utils as _gc_utils

_orig_json_schema_to_python_type = _gc_utils._json_schema_to_python_type


def _safe_json_schema_to_python_type(schema, defs):
    if isinstance(schema, bool):
        return "Any"
    return _orig_json_schema_to_python_type(schema, defs)


_gc_utils._json_schema_to_python_type = _safe_json_schema_to_python_type

_no_proxy_hosts = {"127.0.0.1", "localhost", "0.0.0.0"}
_existing_no_proxy = {
    h.strip() for h in os.environ.get("NO_PROXY", os.environ.get("no_proxy", "")).split(",") if h.strip()
}
os.environ["NO_PROXY"] = ",".join(sorted(_existing_no_proxy | _no_proxy_hosts))
os.environ["no_proxy"] = os.environ["NO_PROXY"]

try:
    import librosa
    _HAVE_LIBROSA = True
except ImportError:
    _HAVE_LIBROSA = False
    warnings.warn("librosa not found — falling back to scipy.io.wavfile "
                   "(.wav only, no mp3 support, no accurate pitch/time effects).")


# #########################################################################
# CONFIGURATION & CONSTANTS
# #########################################################################

DB_PATH = "EE200 Project Song Database"
FS = 22050                  # downsampling rate for faster processing
NPERSEG = 2048               # STFT window length (samples)
HOP = NPERSEG // 4            # 75% overlap
PEAK_NEIGHBORHOOD_SIZE = 20   # local-max neighborhood for peak picking
MIN_PEAK_DB_BELOW_MAX = 45    # peak must be within this many dB of the
                               # clip's OWN loudest bin
FAN_VALUE = 8                  # how many forward peaks to pair with each anchor
MAX_TIME_DELTA = 200            # max anchor-target time gap (frames) -> "target zone"
MIN_MATCHING_HASHES = 5          # min aligned hash votes to accept a match

# Global database: hash_val -> list of (song_name, time_offset)
SONG_DB = {}


# #########################################################################
# AUDIO LOADING (librosa if available, scipy fallback otherwise)
# #########################################################################

def load_audio(path, sr=FS, mono=True):
    """Load an audio file as a float32 numpy array at sample rate `sr`."""
    if _HAVE_LIBROSA:
        y, sr_out = librosa.load(path, sr=sr, mono=mono)
        return y.astype(np.float32), sr_out

    # scipy fallback: .wav only
    if path.lower().endswith(".wav"):
        from scipy.io import wavfile
        sr_in, y = wavfile.read(path)
        y = y.astype(np.float32)
        if np.issubdtype(y.dtype, np.integer):
            y = y / np.iinfo(y.dtype).max
        if mono and y.ndim > 1:
            y = y.mean(axis=1)
        if sr_in != sr:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(sr_in, sr)
            y = resample_poly(y, sr // g, sr_in // g)
        return y.astype(np.float32), sr
    raise ImportError("librosa is required to load non-.wav files. "
                       "pip install librosa")


# #########################################################################
# FINGERPRINTING CORE
# #########################################################################

def get_spectrogram(audio_path):
    """Computes the magnitude spectrogram (dB) of an audio file."""
    y, sr = load_audio(audio_path, sr=FS, mono=True)
    f, t, Sxx = signal.spectrogram(y, fs=sr, nperseg=NPERSEG,
                                    noverlap=NPERSEG - HOP, window='hann')
    S_db = 10 * np.log10(Sxx + 1e-10)
    return f, t, S_db


def get_spectrogram_from_array(y, sr=FS, nperseg=NPERSEG, noverlap=None):
    """Same as get_spectrogram but starting from an in-memory waveform."""
    if noverlap is None:
        noverlap = nperseg - nperseg // 4
    f, t, Sxx = signal.spectrogram(y, fs=sr, nperseg=nperseg,
                                    noverlap=noverlap, window='hann')
    S_db = 10 * np.log10(Sxx + 1e-10)
    return f, t, S_db


def extract_peaks(S_db):
    """Extracts the 'constellation' of peaks from the spectrogram."""
    local_max = maximum_filter(S_db, size=PEAK_NEIGHBORHOOD_SIZE) == S_db
    threshold = S_db.max() - MIN_PEAK_DB_BELOW_MAX
    peaks = local_max & (S_db > threshold)
    freq_idx, time_idx = np.where(peaks)
    return freq_idx, time_idx


def generate_hashes(freq_idx, time_idx, fan_value=FAN_VALUE,
                     max_dt=MAX_TIME_DELTA):
    """Pairs nearby peaks into compact combinatorial hashes (f1, f2, delta_t)."""
    sort_idx = np.argsort(time_idx)
    freq_idx = freq_idx[sort_idx]
    time_idx = time_idx[sort_idx]

    hashes = []
    n = len(time_idx)
    for i in range(n):
        count = 0
        for j in range(i + 1, n):
            dt = time_idx[j] - time_idx[i]
            if dt > max_dt:
                break  # time-sorted, so no more candidates beyond this
            f1, f2 = freq_idx[i], freq_idx[j]
            hash_val = f"{f1}_{f2}_{dt}"
            hashes.append((hash_val, time_idx[i]))
            count += 1
            if count >= fan_value:
                break
    return hashes


def generate_single_peak_hashes(freq_idx, time_idx):
    """Alternative fingerprinting scheme using single peaks alone."""
    return [(f"{f}", t) for f, t in zip(freq_idx, time_idx)]


def fingerprint_array(y, sr=FS, use_pairs=True):
    """Full pipeline starting from an in-memory waveform (for experiments)."""
    f, t, S_db = get_spectrogram_from_array(y, sr=sr)
    freq_idx, time_idx = extract_peaks(S_db)
    if use_pairs:
        hashes = generate_hashes(freq_idx, time_idx)
    else:
        hashes = generate_single_peak_hashes(freq_idx, time_idx)
    return hashes, {"f": f, "t": t, "S_db": S_db,
                     "freq_idx": freq_idx, "time_idx": time_idx}


# #########################################################################
# DATABASE MANAGEMENT
# #########################################################################

def index_database(db_path=DB_PATH, use_pairs=True):
    """Indexes all songs in `db_path` into a hash -> [(song, offset), ...] dict."""
    db = {}
    if not os.path.exists(db_path):
        return db, f"Database path '{db_path}' not found! Check your folder name."

    song_files = sorted(glob.glob(os.path.join(db_path, "*.wav")) +
                         glob.glob(os.path.join(db_path, "*.mp3")))
    for file in song_files:
        song_name = os.path.splitext(os.path.basename(file))[0]
        f, t, S_db = get_spectrogram(file)
        f_idx, t_idx = extract_peaks(S_db)
        if use_pairs:
            hashes = generate_hashes(f_idx, t_idx)
        else:
            hashes = generate_single_peak_hashes(f_idx, t_idx)

        for hash_val, t_offset in hashes:
            db.setdefault(hash_val, []).append((song_name, t_offset))

    return db, f"Successfully indexed {len(song_files)} songs."


# #########################################################################
# MATCHING LOGIC & PLOTTING
# #########################################################################

def _offset_histogram_scores(query_hashes, db):
    """Record delta_t = db_offset - query_offset for matching hashes."""
    matches = {}  # song_name -> {delta: count}
    
    # Check if we are using the new NumPy-based database
    if isinstance(db, dict) and 'keys' in db and 'values' in db:
        keys_arr = db['keys']
        values_arr = db['values']
        song_list = db['songs']
        
        # Pre-allocate packed hash values and query offsets
        hash_ints = []
        q_offsets = []
        for hash_val, q_t_offset in query_hashes:
            try:
                f1, f2, dt = map(int, hash_val.split('_'))
                hash_int = (f1 & 0x3FF) | ((f2 & 0x3FF) << 10) | ((dt & 0xFFF) << 20)
                hash_ints.append(hash_int)
                q_offsets.append(q_t_offset)
            except ValueError:
                continue
                
        if not hash_ints:
            return matches
            
        hash_ints = np.array(hash_ints, dtype=np.uint32)
        q_offsets = np.array(q_offsets, dtype=np.uint32)
        
        # Vectorized binary search for all query hashes
        left_indices = np.searchsorted(keys_arr, hash_ints, side='left')
        right_indices = np.searchsorted(keys_arr, hash_ints, side='right')
        
        # Populate match histogram
        for i in range(len(hash_ints)):
            l = left_indices[i]
            r = right_indices[i]
            if l < r:
                q_t_offset = q_offsets[i]
                packed_matches = values_arr[l:r]
                for packed_match in packed_matches:
                    song_id = packed_match >> 26
                    db_t_offset = packed_match & 0x3FFFFFF
                    db_song = song_list[song_id]
                    
                    delta_t = int(db_t_offset) - int(q_t_offset)
                    hist = matches.setdefault(db_song, {})
                    hist[delta_t] = hist.get(delta_t, 0) + 1
    else:
        # Fallback to the original dictionary lookup
        for hash_val, q_t_offset in query_hashes:
            for db_song, db_t_offset in db.get(hash_val, []):
                delta_t = db_t_offset - q_t_offset
                hist = matches.setdefault(db_song, {})
                hist[delta_t] = hist.get(delta_t, 0) + 1
                
    return matches




def match_query(query_path, db=None, use_pairs=True, generate_plots=True):
    """Matches a query clip and returns result label and diagnostic plots."""
    if db is None:
        db = SONG_DB

    f, t_arr, S_db = get_spectrogram(query_path)
    f_idx, t_idx = extract_peaks(S_db)
    if use_pairs:
        query_hashes = generate_hashes(f_idx, t_idx)
    else:
        query_hashes = generate_single_peak_hashes(f_idx, t_idx)

    matches = _offset_histogram_scores(query_hashes, db)

    if not matches:
        if generate_plots:
            from matplotlib.figure import Figure
            empty_fig = Figure()
            return "No Match Found", 0, empty_fig, empty_fig, empty_fig
        return "No Match Found", 0, None, None, None

    best_song, max_score, best_hist = None, 0, {}
    for song, hist in matches.items():
        peak_count = max(hist.values())
        if peak_count > max_score:
            max_score = peak_count
            best_song = song
            best_hist = hist

    matched = max_score >= MIN_MATCHING_HASHES
    result_label = best_song if matched else "No Match Found"

    if not generate_plots:
        return result_label, max_score, None, None, None

    from matplotlib.figure import Figure

    # --- Plot 1: Spectrogram ---
    fig1 = Figure(figsize=(8, 4))
    ax1 = fig1.subplots()
    ax1.pcolormesh(t_arr, f, S_db, shading='gouraud', cmap='magma',
                    vmin=S_db.max() - 80, vmax=S_db.max())
    ax1.set_title("1. Spectrogram")
    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylim([0, 5000])
    fig1.tight_layout()

    # --- Plot 2: Constellation ---
    fig2 = Figure(figsize=(8, 4))
    ax2 = fig2.subplots()
    ax2.scatter(t_arr[t_idx], f[f_idx], c='cyan', s=10, marker='x')
    ax2.set_facecolor('black')
    ax2.set_title(f"2. Constellation of Peaks ({len(f_idx)} peaks)")
    ax2.set_ylabel("Frequency (Hz)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylim([0, 5000])
    fig2.tight_layout()

    # --- Plot 3: Offset histogram for the best match ---
    fig3 = Figure(figsize=(8, 4))
    ax3 = fig3.subplots()
    if best_hist:
        deltas = sorted(best_hist.keys())
        counts = [best_hist[d] for d in deltas]
        ax3.bar(deltas, counts, width=1.0, color='green', edgecolor='black')
    ax3.set_title(f"3. Offset Histogram — Best Match: {best_song} "
                  f"(peak votes = {max_score})")
    ax3.set_xlabel("Time Offset Delta (frames)")
    ax3.set_ylabel("Vote Count")
    fig3.tight_layout()

    return result_label, max_score, fig1, fig2, fig3



# #########################################################################
# Q3A EXPERIMENTS
# #########################################################################

def add_noise(y, snr_db):
    """Add white Gaussian noise to achieve a target SNR (dB)."""
    sig_power = np.mean(y ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power) * np.random.randn(len(y))
    return (y + noise).astype(np.float32)


def pitch_shift_audio(y, sr, n_semitones):
    """Shift pitch by n_semitones."""
    if _HAVE_LIBROSA:
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_semitones).astype(np.float32)
    factor = 2 ** (n_semitones / 12.0)
    from scipy.signal import resample
    n_new = int(len(y) / factor)
    shifted = resample(y, n_new)
    if len(shifted) < len(y):
        shifted = np.pad(shifted, (0, len(y) - len(shifted)))
    else:
        shifted = shifted[:len(y)]
    return shifted.astype(np.float32)


def time_stretch_audio(y, rate):
    """Stretch/compress duration by `rate`."""
    if _HAVE_LIBROSA:
        return librosa.effects.time_stretch(y, rate=rate).astype(np.float32)
    from scipy.signal import resample
    n_new = int(len(y) / rate)
    return resample(y, n_new).astype(np.float32)


def _identify_array(y, sr, db, use_pairs=True):
    """Identify an in-memory waveform against an in-memory database dict."""
    hashes, meta = fingerprint_array(y, sr=sr, use_pairs=use_pairs)
    matches = _offset_histogram_scores(hashes, db)
    if not matches:
        return "No Match Found", 0, meta, {}
    best_song, max_score, best_hist = None, 0, {}
    for song, hist in matches.items():
        peak_count = max(hist.values())
        if peak_count > max_score:
            max_score = peak_count
            best_song = song
            best_hist = hist
    matched = max_score >= MIN_MATCHING_HASHES
    label = best_song if matched else "No Match Found"
    return label, max_score, meta, matches


def run_q3a_experiments(db_path=DB_PATH, query_song_path=None, out_dir="figures"):
    """Runs Q3A experiments and saves report figures to `out_dir`."""
    os.makedirs(out_dir, exist_ok=True)

    song_files = sorted(glob.glob(os.path.join(db_path, "*.wav")) +
                         glob.glob(os.path.join(db_path, "*.mp3")))
    if not song_files:
        raise FileNotFoundError(f"No songs found in '{db_path}'.")
    if query_song_path is None:
        query_song_path = song_files[0]

    print(f"Using query song: {query_song_path}")
    full_audio, sr = load_audio(query_song_path, sr=FS)
    clip_len = min(10 * sr, len(full_audio))
    start = max(0, len(full_audio) // 2 - clip_len // 2)
    query_clip = full_audio[start:start + clip_len]

    db_pairs, _ = index_database(db_path, use_pairs=True)

    # ---- 1. Whole-song DFT ----
    print("[1/6] Whole-song DFT...")
    X = np.fft.fft(full_audio)
    freqs = np.fft.fftfreq(len(full_audio), d=1 / sr)
    half = len(full_audio) // 2
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs[:half], 20 * np.log10(np.abs(X[:half]) + 1e-6),
            linewidth=0.5, color='navy')
    ax.set_xlim([0, 5000])
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Whole-Song DFT — Frequencies present, but WHEN is lost', fontweight='bold')
    ax.grid(True, alpha=0.3); plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '01_whole_song_dft.png'), dpi=150)
    plt.close(fig)

    # ---- 2. Window length comparison ----
    print("[2/6] Window length comparison...")
    short_win, long_win = 512, 8192
    f_s, t_s, S_s = get_spectrogram_from_array(query_clip, sr=sr, nperseg=short_win)
    f_l, t_l, S_l = get_spectrogram_from_array(query_clip, sr=sr, nperseg=long_win)
    fig, axes = plt.subplots(2, 1, figsize=(11, 9))
    for ax, f_, t_, S_, title in [
            (axes[0], f_s, t_s, S_s, f'Short Window (nperseg={short_win}) — Good Time Resolution'),
            (axes[1], f_l, t_l, S_l, f'Long Window (nperseg={long_win}) — Sharp Frequency Lines')]:
        ax.pcolormesh(t_, f_, S_, shading='gouraud', cmap='magma',
                       vmin=S_.max() - 80, vmax=S_.max())
        ax.set_ylim([0, 5000]); ax.set_xlabel('Time (s)'); ax.set_ylabel('Frequency (Hz)')
        ax.set_title(title, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '02_window_comparison.png'), dpi=150)
    plt.close(fig)

    # ---- 3. Constellation ----
    print("[3/6] Constellation extraction...")
    f, t, S_db = get_spectrogram_from_array(query_clip, sr=sr)
    f_idx, t_idx = extract_peaks(S_db)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.pcolormesh(t, f, S_db, shading='gouraud', cmap='magma',
                  vmin=S_db.max() - 80, vmax=S_db.max())
    ax.scatter(t[t_idx], f[f_idx], s=18, facecolors='none', edgecolors='cyan',
               linewidths=1.0, label='Constellation peaks')
    ax.legend(loc='upper right')
    ax.set_ylim([0, 5000]); ax.set_xlabel('Time (s)'); ax.set_ylabel('Frequency (Hz)')
    ax.set_title(f'Spectrogram + Constellation ({len(f_idx)} peaks)', fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '03_constellation.png'), dpi=150)
    plt.close(fig)
    print(f"      {len(f_idx)} peaks extracted.")

    # ---- 4. Pairs vs single-peak hashing ----
    print("[4/6] Paired hashes vs. single-peak hashes...")
    db_single, _ = index_database(db_path, use_pairs=False)
    label_pairs, score_pairs, _, matches_pairs = _identify_array(query_clip, sr, db_pairs, use_pairs=True)
    label_single, score_single, _, matches_single = _identify_array(query_clip, sr, db_single, use_pairs=False)
    print(f"      Paired -> {label_pairs} (score={score_pairs})")
    print(f"      Single -> {label_single} (score={score_single})")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, matches, label, scr in zip(
            axes, [matches_pairs, matches_single],
            ['Paired Hashes (anchor-target)', 'Single-Peak Hashes'],
            [score_pairs, score_single]):
        if matches:
            best_song = max(matches, key=lambda s: max(matches[s].values()))
            hist = matches[best_song]
            deltas = sorted(hist.keys()); counts = [hist[d] for d in deltas]
            ax.bar(deltas, counts, width=1.0, color='teal')
        ax.set_title(f"{label}\nPeak votes = {scr}", fontweight='bold', fontsize=10)
        ax.set_xlabel('Offset delta (frames)'); ax.set_ylabel('Vote count')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '04_pairs_vs_single.png'), dpi=150)
    plt.close(fig)

    # ---- 5. Noise robustness ----
    print("[5/6] Noise robustness sweep...")
    snr_levels = (30, 20, 10, 5, 0, -5)
    results = []
    for snr in snr_levels:
        noisy = add_noise(query_clip, snr_db=snr)
        label, score, _, _ = _identify_array(noisy, sr, db_pairs, use_pairs=True)
        matched = (label != "No Match Found")
        results.append((snr, matched, score))
        print(f"      SNR={snr:+4d} dB -> matched={matched}, score={score}")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ['green' if m else 'red' for (_, m, _) in results]
    ax.bar([str(s) for s, _, _ in results], [sc for _, _, sc in results], color=colors)
    ax.axhline(MIN_MATCHING_HASHES, color='orange', linestyle='--',
              label=f'Min hashes to accept ({MIN_MATCHING_HASHES})')
    ax.set_xlabel('SNR (dB)'); ax.set_ylabel('Best-match aligned hash count')
    ax.set_title('Noise Robustness: Score vs. SNR\n(green=matched, red=failed)', fontweight='bold')
    ax.legend(); plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '05_noise_robustness.png'), dpi=150)
    plt.close(fig)

    # ---- 6. Pitch-shift / time-stretch robustness ----
    print("[6/6] Pitch-shift / time-stretch robustness sweep...")
    semitone_shifts = (0, 0.5, 1, 2, 4)
    stretch_rates = (1.0, 1.02, 1.05, 1.10, 1.20)
    pitch_results, stretch_results = [], []
    for st in semitone_shifts:
        shifted = query_clip if st == 0 else pitch_shift_audio(query_clip, sr, st)
        label, score, _, _ = _identify_array(shifted, sr, db_pairs, use_pairs=True)
        pitch_results.append((st, label != "No Match Found", score))
        print(f"      pitch={st:+.1f} st -> matched={label != 'No Match Found'}, score={score}")
    for rate in stretch_rates:
        stretched = query_clip if rate == 1.0 else time_stretch_audio(query_clip, rate)
        label, score, _, _ = _identify_array(stretched, sr, db_pairs, use_pairs=True)
        stretch_results.append((rate, label != "No Match Found", score))
        print(f"      rate={rate:.2f} -> matched={label != 'No Match Found'}, score={score}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    sts = [r[0] for r in pitch_results]; pscores = [r[2] for r in pitch_results]
    pmatched = [r[1] for r in pitch_results]
    axes[0].bar([f"{s:+.1f}" for s in sts], pscores,
               color=['green' if m else 'red' for m in pmatched])
    axes[0].axhline(MIN_MATCHING_HASHES, color='orange', linestyle='--')
    axes[0].set_xlabel('Pitch shift (semitones)'); axes[0].set_ylabel('Best-match hash count')
    axes[0].set_title('Pitch-Shift Robustness', fontweight='bold')

    rates = [r[0] for r in stretch_results]; sscores = [r[2] for r in stretch_results]
    smatched = [r[1] for r in stretch_results]
    axes[1].bar([f"{r:.2f}" for r in rates], sscores,
               color=['green' if m else 'red' for m in smatched])
    axes[1].axhline(MIN_MATCHING_HASHES, color='orange', linestyle='--')
    axes[1].set_xlabel('Time-stretch rate'); axes[1].set_ylabel('Best-match hash count')
    axes[1].set_title('Time-Stretch Robustness', fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, '06_pitch_time_robustness.png'), dpi=150)
    plt.close(fig)

    print(f"\nAll experiment plots saved to: {out_dir}/")


# #########################################################################
# GRADIO INTERFACE FUNCTIONS
# #########################################################################

import pickle
import gzip

DB_GZ_FILE = "database.pkl.gz"

if os.path.exists(DB_GZ_FILE):
    try:
        with gzip.open(DB_GZ_FILE, "rb") as f:
            SONG_DB = pickle.load(f)
        if isinstance(SONG_DB, dict) and 'keys' in SONG_DB and 'values' in SONG_DB:
            num_songs = len(SONG_DB['songs'])
            index_status = f"Successfully loaded pre-indexed database from {DB_GZ_FILE} ({num_songs} songs)."
        else:
            num_songs = len(set(song for list_val in SONG_DB.values() for song, _ in list_val))
            index_status = f"Successfully loaded pre-indexed database from {DB_GZ_FILE} ({num_songs} songs)."
    except Exception as e:
        SONG_DB, index_status = index_database()
        try:
            # Save in the new NumPy format to save space and memory
            song_list = sorted(list(set(song for matches in SONG_DB.values() for song, _ in matches)))
            song_to_id = {name: idx for idx, name in enumerate(song_list)}
            flat_keys = []
            flat_values = []
            for hash_str, matches in SONG_DB.items():
                f1, f2, dt = map(int, hash_str.split('_'))
                hash_int = (f1 & 0x3FF) | ((f2 & 0x3FF) << 10) | ((dt & 0xFFF) << 20)
                for song_name, offset in matches:
                    song_id = song_to_id[song_name]
                    packed = (song_id << 26) | (offset & 0x3FFFFFF)
                    flat_keys.append(hash_int)
                    flat_values.append(packed)
            keys_arr = np.array(flat_keys, dtype=np.uint32)
            values_arr = np.array(flat_values, dtype=np.uint32)
            sort_idx = np.argsort(keys_arr)
            keys_arr = keys_arr[sort_idx]
            values_arr = values_arr[sort_idx]
            db_data = {'songs': song_list, 'keys': keys_arr, 'values': values_arr}
            with gzip.open(DB_GZ_FILE, "wb") as f:
                pickle.dump(db_data, f, protocol=4)
            SONG_DB = db_data
        except:
            pass
else:
    SONG_DB, index_status = index_database()
    try:
        song_list = sorted(list(set(song for matches in SONG_DB.values() for song, _ in matches)))
        song_to_id = {name: idx for idx, name in enumerate(song_list)}
        flat_keys = []
        flat_values = []
        for hash_str, matches in SONG_DB.items():
            f1, f2, dt = map(int, hash_str.split('_'))
            hash_int = (f1 & 0x3FF) | ((f2 & 0x3FF) << 10) | ((dt & 0xFFF) << 20)
            for song_name, offset in matches:
                song_id = song_to_id[song_name]
                packed = (song_id << 26) | (offset & 0x3FFFFFF)
                flat_keys.append(hash_int)
                flat_values.append(packed)
        keys_arr = np.array(flat_keys, dtype=np.uint32)
        values_arr = np.array(flat_values, dtype=np.uint32)
        sort_idx = np.argsort(keys_arr)
        keys_arr = keys_arr[sort_idx]
        values_arr = values_arr[sort_idx]
        db_data = {'songs': song_list, 'keys': keys_arr, 'values': values_arr}
        with gzip.open(DB_GZ_FILE, "wb") as f:
            pickle.dump(db_data, f, protocol=4)
        SONG_DB = db_data
    except:
        pass

print(index_status)




def process_single_clip(audio_file):
    if audio_file is None:
        empty_fig = plt.figure()
        return "Please upload or record an audio file.", empty_fig, empty_fig, empty_fig

    file_path = (audio_file.path if hasattr(audio_file, 'path')
                 else (audio_file['name'] if isinstance(audio_file, dict) else audio_file))

    song, score, fig1, fig2, fig3 = match_query(file_path, db=SONG_DB,
                                                  use_pairs=True, generate_plots=True)
    label = f"Predicted Song: {song}" if song != "No Match Found" \
        else f"No confident match (best score {score} < threshold {MIN_MATCHING_HASHES})"
    return label, fig1, fig2, fig3


def process_batch(audio_files):
    if not audio_files:
        return None

    results = []
    for file in audio_files:
        if hasattr(file, 'orig_name'):
            filename = file.orig_name
        elif isinstance(file, dict) and 'orig_name' in file:
            filename = file['orig_name']
        else:
            filename = os.path.basename(file)

        file_path = (file.path if hasattr(file, 'path')
                     else (file['name'] if isinstance(file, dict) else file))

        song, score, _, _, _ = match_query(file_path, db=SONG_DB,
                                             use_pairs=True, generate_plots=False)

        results.append({
            "filename": filename,
            "prediction": song if song != "No Match Found" else "no_match"
        })

    df = pd.DataFrame(results, columns=["filename", "prediction"])
    csv_path = "results.csv"
    df.to_csv(csv_path, index=False)
    plt.close('all')
    return csv_path


# #########################################################################
# GRADIO LAYOUT
# #########################################################################

with gr.Blocks(title="Zapptain America Identifier") as app:
    gr.Markdown("# 🎶 Zapptain America — Audio Fingerprinting App")
    gr.Markdown("Identify songs using frequency-domain fingerprints and offset histograms.")
    gr.Markdown(f"**Database status:** {index_status}")

    with gr.Tabs():
        # Single-Clip Mode
        with gr.Tab("Single-Clip Mode"):
            gr.Markdown("Upload or record a single query clip to identify the song and "
                        "visualize the intermediate DSP steps.")

            audio_input = gr.Audio(
                type="filepath",
                label="Upload or Record Query Clip",
                sources=["upload", "microphone"]
            )

            match_btn = gr.Button("Identify Song")

            output_text = gr.Textbox(label="Result")
            with gr.Row():
                plot1 = gr.Plot(label="Spectrogram")
                plot2 = gr.Plot(label="Constellation")
                plot3 = gr.Plot(label="Offset Histogram")

            match_btn.click(
                fn=process_single_clip,
                inputs=audio_input,
                outputs=[output_text, plot1, plot2, plot3]
            )

        # Batch Mode
        with gr.Tab("Batch Mode"):
            gr.Markdown("Upload multiple query clips to generate the required results.csv file.")
            batch_input = gr.File(file_count="multiple", file_types=["audio"],
                                  label="Upload Query Clips")
            batch_btn = gr.Button("Process Batch")
            batch_output = gr.File(label="Download results.csv")

            batch_btn.click(
                fn=process_batch,
                inputs=batch_input,
                outputs=batch_output
            )


# #########################################################################
# ENTRY POINT
# #########################################################################

if __name__ == "__main__":
    import sys
    if "--experiments" in sys.argv:
        args = [a for a in sys.argv[1:] if a != "--experiments"]
        query_song = args[0] if len(args) > 0 else None
        out_dir = args[1] if len(args) > 1 else "figures"
        run_q3a_experiments(query_song_path=query_song, out_dir=out_dir)
    else:
        is_spaces = os.environ.get("SPACE_ID") is not None
        launch_kwargs = dict(
            server_name="0.0.0.0",
            server_port=int(os.environ.get("PORT", 7860)),
            share=not is_spaces,
        )
        try:
            app.launch(ssr_mode=False, **launch_kwargs)
        except TypeError:
            app.launch(**launch_kwargs)
