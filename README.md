# FFT Audio Fingerprinting & Spectral Matching Engine

An acoustic signal processing and audio localization engine implemented in Python (`NumPy`, `SciPy`, `Matplotlib`). Given a full-length environmental or musical recording and a short query snippet, the engine downmixes multi-channel PCM streams, extracts one-sided Fast Fourier Transform (FFT) magnitude spectra, and executes a sliding-window spectral cross-correlation search using L2-normalized cosine similarity to pinpoint the exact temporal offset of the query signal.

## Core Capabilities

- **Multi-Channel PCM Ingestion & Downmixing:** Reads uncompressed `.wav` audio streams (`scipy.io.wavfile`), converts stereo channels into a unified mono floating-point signal via cross-channel arithmetic averaging, and normalizes sample indexing across arbitrary sampling rates.
- **One-Sided FFT Spectral Extraction:** Transforms discrete time-domain windows into the frequency domain (`scipy.fftpack.fft`), computing scaled single-sided amplitude spectra over `[0, fs/2]` up to the Nyquist limit.
- **Sliding-Window Cosine Similarity Search:** Sweeps a configurable sliding window (50ms stride resolution) across the target signal, evaluating the L2-normalized dot product between the query spectrum and each candidate window spectrum to achieve amplitude-invariant temporal localization.
- **Diagnostic Telemetry & Visualization:** Generates four diagnostic plots comparing full time-domain waveforms, single-sided frequency spectra, temporal similarity trajectories, and direct waveform overlays of the query versus detected segments.
