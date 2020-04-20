# inspired by : https://www.ritchievink.com/blog/2017/04/23/understanding-the-fourier-transform-by-example/
import numpy as np
import matplotlib.pyplot as plt

freq_list = [2.05, 2.1, 2.15,2.2]
freq_list = np.arange(2.05, 5.05, 0.5)
t = np.linspace(0, 10, 500)  # Number of sample points

signals = []

for freq in freq_list:
    s = np.cos(2 * np.pi * freq * t) # + 0.5 * np.sin(90 * 2 * np.pi * t)
    signals.append(s)
    # plt.plot(t, s, label='freq={} Hz'.format(freq))

y_mean1 = sum(signals)/len(freq_list)
plt.plot(t, y_mean1, c='k', linewidth=5, label='average wave, {} freqs'.format(len(freq_list)))


plt.ylabel("Amplitude")
plt.xlabel("Time [s]")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('decoherence_2fres.png')
plt.show()
