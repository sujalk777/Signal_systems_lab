import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 0.01, 1000)
x = np.cos(4000 * np.pi * t)

plt.plot(t, x, color='blue')
plt.title("Continuous-Time Signal $x(t) = \cos(4000 \pi t)$")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

t_continuous = np.linspace(0, 0.01, 1000)
x_continuous = np.cos(4000 * np.pi * t_continuous)

fs = 6000
Ts = 1 / fs
t_sampled = np.arange(0, 0.01, Ts)
x_sampled = np.cos(4000 * np.pi * t_sampled)

plt.stem(t_sampled, x_sampled, linefmt='blue', markerfmt='bo', basefmt=" ", label='Sampled Points')

plt.title("Sampling of Continuous-Time Signal at 6000 Hz")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

x_c=x_sampled
X_n = np.fft.fft(x_c)
N = np.linspace(-6000, 6000, 60)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.stem(N, np.abs(X_n))
plt.xlabel('k-->')
plt.ylabel('|X(k)|-->')
plt.title('Magnitude Spectrum')
plt.grid()

plt.subplot(1, 2, 2)
plt.stem(N, np.angle(X_n))
plt.xlabel('k-->')
plt.ylabel('Phase of X(k)-->')
plt.title('Phase Spectrum')
plt.grid()

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

x_c=x_sampled
X_n = np.fft.fft(x_c)
N = np.linspace(-6000,6000,60)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.stem(N, np.abs(X_n)/np.max(np.abs(X_n)))
plt.xlabel('k-->')
plt.ylabel('|X(k)|-->')
plt.title('Magnitude Spectrum')
plt.grid()

plt.subplot(1, 2, 2)
plt.stem(N, np.angle(X_n))
plt.xlabel('k-->')
plt.ylabel('Phase of X(k)-->')
plt.title('Phase Spectrum')
plt.grid()

plt.tight_layout()
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter

fs = 6000
fc = 2000
numtaps = 101

t = np.linspace(-numtaps // 2, numtaps // 2, numtaps) / fs
ideal_lp = np.sinc(2 * fc * t)
ideal_lp *= np.hamming(numtaps)

ideal_lp /= np.sum(ideal_lp)

w, h = np.fft.fft(ideal_lp, 2048), np.fft.fftshift(np.fft.fft(ideal_lp, 2048))

frequencies = np.linspace(-fs / 2, fs / 2, len(h))

plt.plot(frequencies, np.abs(h), color='red')
plt.title("Frequency Response of the Reconstruction Filter")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(-5000, 5000)
plt.grid()

plt.tight_layout()
plt.show()


# next part is for personal use not a part of the experiment
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

fs = 6000
fc = 2000
numtaps = 101

t_continuous = np.linspace(0, 0.01, 1000)
x_continuous = np.cos(4000 * np.pi * t_continuous)

t_sampled = np.arange(0, 0.01, 1/fs)
x_sampled = np.cos(4000 * np.pi * t_sampled)

t = np.linspace(-numtaps // 2, numtaps // 2, numtaps) / fs
ideal_lp = np.sinc(2 * fc * t)
ideal_lp *= np.hamming(numtaps)
ideal_lp /= np.sum(ideal_lp)

x_reconstructed = lfilter(ideal_lp, 1, x_sampled)

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t_continuous, x_continuous, color='blue', label='Original Signal')
plt.title("Original Signal $x(t) = \cos(4000 \pi t)$")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()

plt.subplot(3, 1, 2)
plt.stem(t_sampled, x_sampled, linefmt='red', markerfmt='ro', basefmt=" ", label='Sampled Signal')
plt.title("Sampled Signal at 6000 Hz")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t_sampled, x_reconstructed, color='green', label='Reconstructed Signal')
plt.title("Reconstructed Signal after Filtering")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
