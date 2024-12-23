import numpy as np
import matplotlib.pyplot as plt

def fft_ifft(signal):
    N = len(signal)
    fft_result = np.fft.fft(signal)
    magnitude = np.abs(fft_result)
    phase = np.angle(fft_result)
    ifft_result = np.fft.ifft(fft_result)
    return fft_result, magnitude, phase, ifft_result

f = 0.1
T = 0.1
n = np.arange(0, 100)
x1 = np.sin(2 * np.pi * f * n * T)
fft_x1, magnitude_x1, phase_x1, ifft_x1 = fft_ifft(x1)

x2 = np.exp(-n * T / 5)
fft_x2, magnitude_x2, phase_x2, ifft_x2 = fft_ifft(x2)

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.plot(n, x1)
plt.title('Original Signal x1[n]')
plt.xlabel('n')
plt.ylabel('Amplitude')

plt.subplot(2, 3, 2)
plt.plot(magnitude_x1)
plt.title('FFT Magnitude of x1[n]')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')

plt.subplot(2, 3, 3)
plt.plot(phase_x1)
plt.title('FFT Phase of x1[n]')
plt.xlabel('Frequency')
plt.ylabel('Phase')

plt.subplot(2, 3, 4)
plt.plot(n, ifft_x1.real)
plt.title('Reconstructed x1[n]')
plt.xlabel('n')
plt.ylabel('Amplitude')

plt.subplot(2, 3, 5)
plt.plot(n, x2)
plt.title('Original Signal x2[n]')
plt.xlabel('n')
plt.ylabel('Amplitude')

plt.subplot(2, 3, 6)
plt.plot(magnitude_x2)
plt.title('FFT Magnitude of x2[n]')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')


plt.tight_layout()
plt.show()


# code for recording audio in raspberry pi--- arecord -5audio.wav
import librosa
import numpy as np
import matplotlib.pyplot as plt

audio_file = 'audio.wav'
y, sr = librosa.load(audio_file)


D = np.fft.fft(y)

y_reconstructed = np.fft.ifft(D)

plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(y)
plt.title('Original Audio Signal')
plt.xlabel('Samples')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 2)
plt.plot(np.abs(D))
plt.title('Magnitude Spectrum')
plt.xlabel('Frequency (Bins)')
plt.ylabel('Magnitude')


plt.subplot(3, 1, 3)
plt.plot(y_reconstructed.real)
plt.title('Reconstructed Audio Signal')
plt.xlabel('Samples')
plt.ylabel('Amplitude')


plt.tight_layout()
plt.show()
