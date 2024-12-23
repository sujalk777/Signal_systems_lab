# prompt: Upsampling and downsampling for a sine wave function. Upsample by
# factor 5, then downsample by factor 3.

import numpy as np
import matplotlib.pyplot as plt

# Assuming 'x' is your sine wave signal (replace with your actual signal)
# Example sine wave:
f = 20  # Frequency
t = np.linspace(0, 1, 500)  # Time vector
x = np.sin(2 * np.pi * f * t)


def upsample(signal, factor):
  upsampled_signal = np.zeros(len(signal) * factor)
  upsampled_signal[::factor] = signal
  return upsampled_signal

def downsample(signal, factor):
  downsampled_signal = signal[::factor]
  return downsampled_signal

# Upsample by a factor of 5
upsampled_x = upsample(x, 5)

# Downsample by a factor of 3
downsampled_x = downsample(upsampled_x, 3)

# Plot the original, upsampled, and downsampled signals
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.stem(t, x)
plt.title('Original Signal')

plt.subplot(3, 1, 2)
plt.stem(np.linspace(0, 1, len(upsampled_x)), upsampled_x)
plt.title('Upsampled Signal (factor=5)')

plt.subplot(3, 1, 3)
plt.stem(np.linspace(0, 1, len(downsampled_x)), downsampled_x)
plt.title('Downsampled Signal (factor=3)')

plt.tight_layout()
plt.show()
