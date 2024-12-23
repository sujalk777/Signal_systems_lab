
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5])
h = np.array([0, 0.5, 1, 1.5])

y = np.convolve(x, h, mode='full')

plt.figure(figsize=(8, 6))

plt.subplot(3, 1, 1)
plt.stem(range(len(x)), x)
plt.title('x[n]')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True)


plt.subplot(3, 1, 2)
plt.stem(range(len(h)), h)
plt.title('h[n]')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True)


plt.subplot(3, 1, 3)
plt.stem(range(len(y)), y)
plt.title('y[n] = x[n] * h[n]')
plt.xlabel('n')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()


import librosa
import numpy as np
import matplotlib.pyplot as plt

voice, sr_voice = librosa.load('voice.wav', sr=8000)
noise, sr_noise = librosa.load('noice.wav', sr=8000)

convolved_audio = np.convolve(voice, noise, mode='full')

convolved_audio = convolved_audio / np.max(np.abs(convolved_audio))

plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(voice)
plt.title('Voice Signal')

plt.subplot(3, 1, 2)
plt.plot(noise)
plt.title('Noise Signal')

plt.subplot(3, 1, 3)
plt.plot(convolved_audio)
plt.title('Convolved Audio')

plt.tight_layout()
plt.show()

