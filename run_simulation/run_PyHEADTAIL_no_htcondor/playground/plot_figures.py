import pickle
import numpy as np
import matplotlib.pyplot as plt


mytunes_AN = pickle.load(open('mytunes_AN.pkl', 'rb'))
mytunes_noKick = pickle.load(open('mytunes_noKick.pkl', 'rb'))
mytunes_noKick_50 = pickle.load(open('mytunes_noKick_window50.pkl', 'rb'))

mylen_50 = len(mytunes_noKick_50['Qx'][0])
myx_50 = np.linspace(0, 1000, mylen_50)
mylen = len(mytunes_AN['Qx'][0])
myx = np.linspace(0, 1000, mylen)

plt.plot(myx, np.array(mytunes_noKick['Qx'][7576]), '.-', label='window 100')
plt.plot(myx_50, np.array(mytunes_noKick_50['Qx'][7576]), '.-', label='window 50')
plt.xlabel('turns')
plt.ylabel('Qx')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

plt.plot(myx, mytunes_AN['Qy'][7576], '.-', label='AN')
plt.plot(myx, mytunes_noKick['Qy'][7576], '.-', label='no AN')
plt.xlabel('turns')
plt.ylabel('Qy')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()



y1 =  np.array(mytunes_noKick_50['Qx'][7576])
y2 =  np.array(mytunes_AN['Qy'][7576])
# subtract the mean , in order to remove the DC component
average_Qx = np.average(y1)
average_Qy = np.average(y2)

y1[:] = [x - average_Qx for x in y1]
y2[:] = [x - average_Qy for x in y2]

fourier_qx = np.fft.fft(y1)
fourier_qy = np.fft.fft(y2)
step=10
freqs_qx = np.fft.fftfreq(len(fourier_qx))
freqs_qy = np.fft.fftfreq(len(fourier_qy))

fig, ax = plt.subplots()
plt.plot(freqs_qx/step, 2*abs(fourier_qx) / len(fourier_qx), 'o', label='Qx')#, label='FFT whole signal')
#plt.plot(freqs_qy/step, 2*abs(fourier_qy) / len(fourier_qy), 'o', label='Qy')#, label='FFT whole signal')

#plt.plot(freqs_qx_2, 2*abs(fourier_qx_2) / len(fourier_qx_2), 'or', label='FFT half signal')
plt.xlabel('Frequency ')
plt.ylabel('Amplitude [Arbitrary units]')
plt.tight_layout()
plt.legend()
plt.grid()

plt.show()


plt.grid()
plt.show()


quit()


plt.show()
print(max(abs(spectrum)))
sp = abs(spectrum)
index = np.argmax(np.array(sp))
print(freq[index])
print(freq)

'''
# do fft
import scipy.fftpack

# Number of samplepoints
N = int(mylen)
# sample spacing
T = int(1000 /mylen)
x = myx
y = mytunes_noKick['Qx'][7576]
yf = scipy.fftpack.fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), int(N/2))

plt.subplot(2, 1, 1)
plt.plot(xf, 2.0/N * np.abs(yf[0:int(N/2)]))
plt.subplot(2, 1, 2)
plt.plot(xf[1:], 2.0/N * np.abs(yf[0:int(N/2)])[1:])
plt.show()
'''