import numpy as np

import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

npoints=10000
tune = 400
amp = 0.8
t = np.linspace(0,1000,npoints)
noise_y = np.exp(-2.e-3*t)
tune_y = amp*np.exp(-(t-tune)**2/(2*50**2))
two_y = noise_y+tune_y

t1 = argrelextrema(tune_y, np.greater)
t2 = argrelextrema(two_y, np.greater)

plt.plot(t,noise_y,'k', label='noise spectrum')
plt.plot(t,tune_y,'C0', label='tune spectrum')
plt.plot(t,noise_y+tune_y,'C1', label='tune + noise spectrum')
plt.axvline(x=t[t1],color='C0')
plt.axvline(x=t[t2],color='C1')
plt.xlabel('samples')
plt.ylabel('Amplitude')
plt.grid()
plt.legend()
plt.tight_layout()
#plt.savefig('simple_signal_analysis_A0_8.png')
plt.show()
