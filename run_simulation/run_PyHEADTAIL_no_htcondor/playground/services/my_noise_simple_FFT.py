import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

# Noise parameters
mu = 0
sigma = 1
n_points = 10000
# Noise kicks
y = np.random.normal(mu, sigma, n_points)
# Cumulative kicks
yy = np.array([np.sum(y[:i]) for i in range(n_points)])

# Frequency bins for given FFT parameters.
freq = np.fft.fftfreq(y.shape[-1])

plt.figure(1)
plt.plot(y, label='white gaussian noise')
plt.plot(yy, label='brownian/red noise')
plt.grid()
plt.legend()
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.tight_layout()
#plt.savefig('my_noise.png')
plt.figure(2)
# Compute the one-dimensional discrete Fourier Transform
plt.plot(freq, np.fft.fft(y), '.', label='white gaussian noise')
plt.plot(freq, np.fft.fft(yy), '.', label='brownian/red noise')
plt.plot(freq, 1/(freq**2), c='k', label=r'$1/f^2$')
plt.yscale('log')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.grid()
plt.legend()
#plt.savefig('my_noise_fft_2.png')
plt.show()