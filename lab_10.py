import numpy as np
import matplotlib.pyplot as plt
import librosa
voice, sr = librosa.load('voice.wav', sr=None)

segment_length = 1100
num_segments = len(voice) // segment_length

segment_energies = []

for i in range(num_segments):
    segment = voice[i * segment_length:(i + 1) * segment_length]
    energy = np.sum(segment**2)
    segment_energies.append(energy)


overall_energy = np.sum(voice**2)
plt.figure(figsize=(10, 6))
plt.plot(range(num_segments), segment_energies, marker='o')
plt.xlabel('Segment Number')
plt.ylabel('Energy')
plt.title('Segment-wise Energy of Speech Signal')
plt.grid(True)
plt.show()

print(f"Overall Energy of Speech Signal: {overall_energy}")

import numpy as np
import matplotlib.pyplot as plt

fs = 1000
duration = 1
t = np.arange(0, duration, 1/fs)
f1 = 50
f2 = 120
signal1 = np.sin(2 * np.pi * f1 * t)
signal2 = np.sin(2 * np.pi * f2 * t)

synthetic_signal = signal1 + signal2

noise_f1 = 50
noise_f2 = 2000
noise_amp = 0.25
noise1 = noise_amp * np.sin(2 * np.pi * noise_f1 * t)
noise2 = noise_amp * np.sin(2 * np.pi * noise_f2 * t)
noisy_signal = synthetic_signal + noise1 + noise2


plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, synthetic_signal)
plt.title('Synthetic Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(t, noisy_signal)
plt.title('Noisy Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

frequencies = np.fft.fftfreq(len(synthetic_signal), 1/fs)
synthetic_fft = np.fft.fft(synthetic_signal)
noisy_fft = np.fft.fft(noisy_signal)

noise1_fft = np.fft.fft(noise1)
noise2_fft = np.fft.fft(noise2)

plt.figure(figsize=(12, 6))
plt.plot(frequencies, np.abs(synthetic_fft), label='Synthetic Signal')
plt.plot(frequencies, np.abs(noisy_fft), label='Noisy Signal')
plt.plot(frequencies, np.abs(noise1_fft), label='Noise (50Hz)')
plt.plot(frequencies, np.abs(noise2_fft), label='Noise (2000Hz)')
plt.title('Magnitude Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim(0, 1000)
plt.legend()
plt.grid(True)
plt.show()
