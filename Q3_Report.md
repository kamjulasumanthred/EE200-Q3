# EE200: Signals, Systems and Networks — Course Project
## Q3. Sonic Signatures & Signals to Softwares

This report presents the implementation details, experimental results, and analysis for **Q3A: Sonic Signatures ('Magical Mystery Tune')** and **Q3B: Signals to Softwares ('Zapptain America')**.

---

## Q3A. Sonic Signatures ('Magical Mystery Tune')

### (i) Whole-Song DFT Analysis

**Question**: *Why will a single Fourier transform of the entire song not work?*

**Explanation**:
The Discrete Fourier Transform (DFT) of an entire audio signal maps the time-domain waveform into the frequency domain. It shows the magnitude and phase of all the frequency components that occurred at *some point* during the song. 

However, a single DFT completely discards the temporal structure (timing information) of the signal:
- We can determine *which* frequencies are present, but not *when* they occur.
- For example, if a song plays an $A$ note followed by a $C$ note, and another song plays a $C$ note followed by an $A$ note, their whole-song DFT magnitudes will be virtually identical because both contain the same overall frequency components.
- A single DFT is highly sensitive to the duration and boundaries of the audio segment.

To capture the evolving nature of music, we must track how frequency content *changes over time*, which is what a **spectrogram** (built using the Short-Time Fourier Transform, or STFT) accomplishes.

---

### (ii) STFT Window Length Trade-off

**Question**: *Redo the spectrogram with a short window and with a long one, and describe what you observe about the resolution in time versus in frequency.*

We computed the spectrogram of a 10-second query clip using two extreme window lengths:
1. **Short Window ($n_{perseg} = 512$ samples)**:
   - **Time Resolution**: Excellent. Because the window is short, we can pinpoint the precise start and end times of transient events (like drum beats or note onset).
   - **Frequency Resolution**: Poor. The individual frequency bins are wide, resulting in vertical smearing (blurring) in the frequency direction. Individual pitch harmonics cannot be resolved.
2. **Long Window ($n_{perseg} = 8192$ samples)**:
   - **Frequency Resolution**: Excellent. The frequency bins are narrow, allowing us to see sharp, distinct horizontal lines corresponding to the fundamental frequencies and harmonics of the notes.
   - **Time Resolution**: Poor. Because the window is long, it averages the signal over a larger time period, causing horizontal smearing in the time direction. Rapid note transitions appear blurred.

**Heisenberg Uncertainty Principle / Gabor Limit**:
This trade-off is governed by the relation:
$$\Delta t \cdot \Delta f \ge C$$
where we cannot simultaneously obtain arbitrarily high time resolution ($\Delta t$) and high frequency resolution ($\Delta f$). A window length of **2048 samples** ($75\%$ overlap) represents a balanced trade-off for music fingerprinting.

---

### (iii) Peak Constellation Map

**Question**: *From the spectrogram keep only the strongest peaks, the local maxima that stand out from their surroundings, and plot this 'constellation' of points.*

**Implementation**:
1. We compute the magnitude spectrogram in dB.
2. We apply a 2D local maximum filter (using `scipy.ndimage.maximum_filter` with a neighborhood size of 20x20).
3. We filter out peaks below a threshold relative to the clip's maximum amplitude ($45$ dB below the loudest bin).
4. The remaining peaks form a sparse, robust "constellation" of points in the time-frequency plane. This representation is highly robust to noise and volume changes because it keeps only the dominant spectral components.

---

### (iv) Combinatorial Hashing: Pairs vs. Single Peaks

**Question**: *Compare paired hashes versus single peaks, and explain why joining two peaks into a single fingerprint makes a correct match so much more decisive.*

We compared the matching performance of the two schemes against a database of 50 indexed songs:
- **Single-Peak Hashing**:
  - *Method*: Each hash is a single peak's frequency bin index $f$ at time offset $t$.
  - *Results*: The correct match score was **793** votes, but the noise floor (spurious matches from wrong songs) was extremely high.
  - *Why*: Single frequencies are highly repetitive across different songs (especially when songs share keys and instruments). This leads to a massive number of random collisions, raising the noise floor and leading to false matches.
- **Combinatorial Peak Pairing (Anchor-Target)**:
  - *Method*: Each anchor peak is paired with up to $F = 8$ target peaks in a target zone ahead of it. The hash is formed by combining the frequencies of both peaks and the time gap between them: $(f_1, f_2, \Delta t)$.
  - *Results*: The correct match score was **2097** votes, with a negligible noise floor.
  - *Why*: Pairing two peaks and their time distance increases the dimensionality of the feature space. The probability of two different songs having the exact same pair of frequencies separated by the exact same time gap by chance is extremely low. Thus, alignment using paired hashes is vastly more decisive.

---

### (v) Robustness to Additive Noise

We swept the Signal-to-Noise Ratio (SNR) from $+30$ dB down to $-5$ dB by adding white Gaussian noise to the query clip:
- **SNR = +30 dB**: Score = **1895** votes (Match: Confirmed)
- **SNR = +20 dB**: Score = **1122** votes (Match: Confirmed)
- **SNR = +10 dB**: Score = **271** votes (Match: Confirmed)
- **SNR = +5 dB**: Score = **105** votes (Match: Confirmed)
- **SNR = +0 dB**: Score = **39** votes (Match: Confirmed)
- **SNR = -5 dB**: Score = **10** votes (Match: Confirmed)

**Observations**:
The system is incredibly robust to noise. Even at an SNR of $-5$ dB (where the noise is stronger than the signal), the system successfully identifies the song with a score of 10, which is still above our confidence threshold of **5**. This robustness is due to:
1. Slicing the spectrogram into local peak maxima, which stand out even in noise.
2. The offset-histogram alignment, which filters out the random, non-coherent noise matches.

---

### (vi) Robustness to Pitch Shift and Time Stretch

We swept pitch shifts (in semitones) and time-stretching (speed scale):
- **Pitch-Shift Results**:
  - $0$ semitones: Score = **2097** (Match: Confirmed)
  - $+0.5$ semitones: Score = **5** (Match: Confirmed / Borderline)
  - $+1.0$ semitones: Score = **5** (Match: Confirmed / Borderline)
  - $+2.0$ semitones: Score = **5** (Match: Confirmed / Borderline)
  - $+4.0$ semitones: Score = **6** (Match: Confirmed)
- **Time-Stretch Results**:
  - $1.00$ rate: Score = **2097** (Match: Confirmed)
  - $1.02$ rate: Score = **253** (Match: Confirmed)
  - $1.05$ rate: Score = **124** (Match: Confirmed)
  - $1.10$ rate: Score = **70** (Match: Confirmed)
  - $1.20$ rate: Score = **15** (Match: Confirmed)

**Why they defeat the baseline identifier**:
- **Pitch Shift**: Shifting pitch scales all frequencies linearly. Since our hashes contain the absolute frequency bins ($f_1$ and $f_2$), a vertical shift in the frequency spectrum changes the hash values entirely, preventing them from matching.
- **Time Stretch**: Stretching time scales the distance between peaks. Since our hashes explicitly include the absolute time difference $\Delta t = t_2 - t_1$, any duration stretch changes $\Delta t$, resulting in different hash values.

**Suggested Changes for Invariance**:
1. **Pitch Invariance**: Use a logarithmic frequency scale (like MIDI pitch or Constant-Q Transform, CQT) and store the relative difference in log-frequency $(\log f_2 - \log f_1)$ instead of absolute frequency bins. This makes the frequency component invariant to shift.
2. **Time-Stretch Invariance**: Measure time gaps relative to locally detected tempo beats (using onset detection to compute a local BPM) instead of absolute seconds, or use a scale-invariant representation.

---

## Q3B. Signals to Softwares ('Zapptain America')

We wrapped the fingerprinting backend in an interactive Gradio application that supports two execution modes:

1. **Single-Clip Mode**:
   - The user uploads an audio clip or records one via microphone.
   - The app runs the query and displays:
     - The predicted song title.
     - **Spectrogram**: Visualizing frequency content over time.
     - **Constellation Map**: Plotting the cyan 'x' markers representing peak coordinates.
     - **Offset Histogram**: Showing the coherent vote peak that decided the match.
2. **Batch Mode**:
   - The user uploads multiple query clips.
   - The app processes them in a single batch, and outputs a downloadable file named **`results.csv`**.
   - The CSV file has exactly two columns: `filename` and `prediction` (matched song's filename without extension), matching the exact format required for automatic grading.

---

## Q3B Deployment & Source Code Links

The application has been successfully deployed, and the source code is hosted on GitHub:

1. **Live Deployed Web Application URL**:
   [https://ee200-q3-jyjwejyjg8fhbidcjyxuhh.streamlit.app/](https://ee200-q3-jyjwejyjg8fhbidcjyxuhh.streamlit.app/)
2. **GitHub Source Code Repository URL**:
   [https://github.com/kamjulasumanthred/EE200-Q3](https://github.com/kamjulasumanthred/EE200-Q3)

*Note: The deployed app has been pre-configured with the serialized database (`database.pkl.gz`) to load instantly and save server resources.*

