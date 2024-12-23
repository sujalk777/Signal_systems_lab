import numpy as np
import matplotlib.pyplot as plt

f = 20
t = np.linspace(0, 4, 500)
x = np.sin(2 * np.pi * f * t)

delay = 1
delayed_t = t + delay

delayed_x = np.sin(2 * np.pi * f * delayed_t)
delayed_x = delayed_x[delayed_t <= 10]
delayed_t = delayed_t[delayed_t <= 10]

X = np.fft.fft(x)
delayed_X = np.fft.fft(delayed_x)


delayed_x_reconstructed = np.fft.ifft(delayed_X)


plt.figure(figsize=(12, 8))
plt.subplot(3, 2, 1)
plt.plot(t, x)
plt.title('Original Signal x(t)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')


plt.subplot(3, 2, 2)
plt.plot(np.abs(X))
plt.title('Magnitude Spectrum of x(t)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')


plt.subplot(3, 2, 3)
plt.plot(np.angle(X))
plt.title('Phase Spectrum of x(t)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase')

plt.subplot(3, 2, 4)
plt.plot(delayed_t, delayed_x)
plt.title('Delayed Signal x(t-5)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(3, 2, 5)
plt.plot(np.abs(delayed_X))
plt.title('Magnitude Spectrum of delayed x(t)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')


plt.subplot(3, 2, 6)
plt.plot(delayed_t, delayed_x_reconstructed.real)
plt.title('Reconstructed Delayed Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')


plt.tight_layout()
plt.show()



import numpy as np
import matplotlib.pyplot as plt

f = 20
t = np.linspace(0, 1, 500)
x = np.sin(2 * np.pi * f * t)

scaled_x = 4 * x

X = np.fft.fft(x)
scaled_X = np.fft.fft(scaled_x)

scaled_x_reconstructed = np.fft.ifft(scaled_X)

plt.figure(figsize=(12, 8))
plt.subplot(3, 2, 1)
plt.plot(t, x)
plt.title('Original Signal x(t)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(3, 2, 2)
plt.plot(t, scaled_x)
plt.title('Scaled Signal 4*x(t)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(3, 2, 3)
plt.plot(np.abs(X))
plt.title('Magnitude Spectrum of x(t)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

plt.subplot(3, 2, 4)
plt.plot(np.angle(X))
plt.title('Phase Spectrum of x(t)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase')


plt.subplot(3, 2, 5)
plt.plot(np.abs(scaled_X))
plt.title('Magnitude Spectrum of scaled x(t)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.subplot(3, 2, 6)
plt.plot(t, scaled_x_reconstructed.real)
plt.title('Reconstructed Scaled Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')


plt.tight_layout()
plt.show()
