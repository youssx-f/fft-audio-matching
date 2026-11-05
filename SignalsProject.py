import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft

file_name = "451046__kyles__city-square-cafe-ext-calm-birds-mopeds-ventilation-or-city-tone-light-walla-marseille-ms.wav"
fs, data = wavfile.read(file_name)
if len(data.shape) > 1:
    data = np.mean(data, axis=1)

data = data.astype(float)
total_length_sec = len(data) / fs
time_axis = np.linspace(0, total_length_sec, len(data))

plt.figure(figsize=(10, 4))
plt.plot(time_axis, data)
plt.title("Full Audio Signal (Time Domain)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

actual_start_time = 2.0
clip_duration = 0.5
start_index = int(actual_start_time * fs)
clip_length_samples = int(clip_duration * fs)
end_index = start_index + clip_length_samples
clip = data[start_index:end_index]

def get_fft_magnitude(segment):
    N = len(segment)
    freq_data = fft(segment)
    mag = (2 / N) * np.abs(freq_data[0:int(N / 2)])
    return mag

clip_fft_mag = get_fft_magnitude(clip)
N_clip = len(clip)
freq_axis = np.linspace(0.0, fs / 2, int(N_clip / 2))

plt.figure(figsize=(10, 4))
plt.plot(freq_axis, clip_fft_mag)
plt.title("Frequency Spectrum of the Query Clip")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()

step_size = int(fs * 0.05)
similarity_scores = []
time_steps = []
clip_norm = np.linalg.norm(clip_fft_mag)

for i in range(0, len(data) - clip_length_samples, step_size):
    window = data[i : i + clip_length_samples]
    window_fft_mag = get_fft_magnitude(window)
    dot_product = np.dot(window_fft_mag, clip_fft_mag)
    window_norm = np.linalg.norm(window_fft_mag)
    score = 0 if (window_norm == 0 or clip_norm == 0) else dot_product / (window_norm * clip_norm)
    similarity_scores.append(score)
    time_steps.append(i / fs)

best_match_idx = np.argmax(similarity_scores)
detected_time = time_steps[best_match_idx]
best_score = similarity_scores[best_match_idx]
detected_start_sample = int(detected_time * fs)
detected_segment = data[detected_start_sample : detected_start_sample + clip_length_samples]

print("--- Project Expected Outputs ---")
print(f"Sampling frequency: {fs} Hz")
print(f"Length of full signal: {total_length_sec:.2f} seconds")
print(f"Clip length: {clip_duration:.2f} seconds")
print(f"Original clip position: {actual_start_time:.2f} seconds")
print(f"Detected position: {detected_time:.2f} seconds")
print(f"Best similarity score: {best_score:.4f}")
