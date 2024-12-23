import numpy as np
import matplotlib.pyplot as plt

def dirac_delta_continuous(t, t0, epsilon=0.1):
  return 1 / (epsilon * np.sqrt(np.pi)) * np.exp(-((t - t0) / epsilon)**2)

def dirac_delta_discrete(n, n0):
  return 1 if n == n0 else 0

t = np.linspace(-2, 2, 1000)

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, dirac_delta_continuous(t, 0))
plt.title('Continuous-time Dirac Delta Function')
plt.xlabel('t')
plt.ylabel('δ(t)')
plt.grid(True)

n = np.arange(-5, 6)
delta_n = [dirac_delta_discrete(i, 0) for i in n]
plt.subplot(2, 1, 2)
plt.stem(n, delta_n)
plt.title('Discrete-time Dirac Delta Function')
plt.xlabel('n')
plt.ylabel('δ[n]')
plt.grid(True)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def unit_step(t):
  return np.where(t >= 0, 1, 0)

def ramp(t):
  return np.where(t >= 0, t, 0)

t = np.linspace(-2, 2, 1000)

plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.plot(t, unit_step(t))
plt.title('Continuous-time Unit Step Function')
plt.xlabel('t')
plt.ylabel('u(t)')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(t, ramp(t))
plt.title('Continuous-time Ramp Function')
plt.xlabel('t')
plt.ylabel('r(t)')
plt.grid(True)
n = np.arange(-5, 6)

plt.subplot(2, 2, 3)
plt.stem(n, unit_step(n))
plt.title('Discrete-time Unit Step Function')
plt.xlabel('n')
plt.ylabel('u[n]')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.stem(n, ramp(n))
plt.title('Discrete-time Ramp Function')
plt.xlabel('n')
plt.ylabel('r[n]')
plt.grid(True)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def unit_parabolic(t):
  return np.where(t >= 0, 0.5 * t**2, 0)

t = np.linspace(-2, 2, 1000)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(t, unit_parabolic(t))
plt.title('Continuous-time Unit Parabolic Function')
plt.xlabel('t')
plt.ylabel('p(t)')
plt.grid(True)

n = np.arange(-5, 6)
plt.subplot(1, 2, 2)
plt.stem(n, unit_parabolic(n))
plt.title('Discrete-time Unit Parabolic Function')
plt.xlabel('n')
plt.ylabel('p[n]')
plt.grid(True)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
def exponential_continuous(t, a):
  return np.exp(a * t)

def exponential_discrete(n, a):
  return a**n

a = -0.5
t = np.linspace(-2, 2, 1000)

n = np.arange(-5, 6)

exp_continuous = exponential_continuous(t, a)
exp_discrete = exponential_discrete(n, np.exp(a))

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(t, exp_continuous)
plt.title('Continuous-time Exponential Function')
plt.xlabel('t')
plt.ylabel('e^(at)')
plt.grid(True)


plt.subplot(1, 2, 2)
plt.stem(n, exp_discrete)
plt.title('Discrete-time Exponential Function')
plt.xlabel('n')
plt.ylabel('a^n')
plt.grid(True)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def random_function_continuous(t):
  amplitude = np.random.uniform(0.5, 1.5)
  frequency = np.random.uniform(1, 3)
  return amplitude * np.sin(2 * np.pi * frequency * t)

def random_function_discrete(n):
    amplitude = np.random.uniform(0.5, 1.5)
    frequency = np.random.uniform(0, 0.5)
    return amplitude * np.sin(2 * np.pi * frequency * n)

t = np.linspace(0, 2, 500)
n = np.arange(0, 10)
continuous_signal = random_function_continuous(t)
discrete_signal = random_function_discrete(n)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(t, continuous_signal)
plt.title('Continuous-time Random Function')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.stem(n, discrete_signal)
plt.title('Discrete-time Random Function')
plt.xlabel('n')
plt.ylabel('x[n]')
plt.grid(True)

plt.tight_layout()
plt.show()

