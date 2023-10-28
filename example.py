import numpy as np
import hilbert_transform as ht
from scipy.signal import hilbert
from matplotlib import pyplot as plt


rng = np.random.default_rng(123456)

n_samples = 2**9
l_window = 2**7
n_lag = 50

#
# a low-frequency sine with a varying envelope
#
f = 1.0 / 20
t = np.arange(n_samples)
# an envelope which varies both...
env = np.ones(t.shape)
len_5 = int(t.size / 5) + int(t.size / 5) % 2
# ... discontinuously and ...
env[len_5 : 2 * len_5] = 0.2
# ... smoothly
env[3 * len_5 : int(3.5 * len_5)] = np.linspace(1.0, 0.3, int(0.5 * len_5))
env[int(3.5 * len_5) : 4 * len_5] = np.linspace(0.3, 1.0, int(0.5 * len_5))
# a noisy sine with this envelope
x = env * np.sin(2 * np.pi * f * t)
x += (np.random.rand(t.size) * 2 - 1) * 0.2

#
# full-length vs. sliding window Hilbert transform
#

# full-length Hilbert transform
H = hilbert(x)

# sliding window Hilbert transform
H_slide = ht.sliding_transform(x, l_window, n_lag)

plt.plot(x, "k", linewidth=0.8, label="A * sin + noise")

plt.plot(np.abs(H), "b:", linewidth=1.9, label="full-length transform")
plt.plot(-np.abs(H), "b:", linewidth=1.9)

plt.plot(np.abs(H_slide), "-.", color="orange", linewidth=1.7, label="sliding window transform")
plt.plot(-np.abs(H_slide), "-.", color="orange", linewidth=1.7)
plt.plot(env, "g--", label="A")
plt.plot(-env, "g--")

plt.legend()
plt.savefig('example.png')

print('generated example.png.')
