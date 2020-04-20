import numpy as np
import matplotlib.pyplot as plt


signals1 = []
signals2 = []
# freq_list1 = [2.05, 2.1, 2.15, 2.5]
freq_list1 = np.arange(2.05, 10000.05, 0.5)
#freq_list1 = np.linspace(2.05, 2.5, 10)
freq_list1 = np.arange(2.05, 10000.05, 500)
#freq_list1 = np.arange(2.05, 2.5, 0.005)  # smaller tune spread
print(freq_list1)
Fs = 1000
sample = 1000
t = np.arange(sample)

for f in freq_list1:
    y = np.cos(2 * np.pi * f * t / Fs)
    signals1.append(y)
    #plt.plot(t, y)#, label='freq={}'.format(f))
#for f in freq_list2:
#    y = np.sin(2 * np.pi * f * t / Fs)
#    signals2.append(y)
    # plt.plot(t, y, label='freq={}'.format(f))

y_mean1 = sum(signals1)/len(freq_list1)
#y_mean2 = sum(signals2)/len(freq_list2)

plt.plot(t, y_mean1, c='k', linewidth=5, label='mean {} freqs'.format(len(freq_list1)))
#plt.plot(t, y_mean2, c='r', linewidth=2, label='mean 20 freqs')
plt.xlabel('time (s)')
plt.ylabel('voltage(V)')
plt.legend()
plt.grid()
plt.tight_layout()
#plt.savefig('cos_signals_decoherence.png')
plt.show()