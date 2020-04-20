# inspired by : https://www.ritchievink.com/blog/2017/04/23/understanding-the-fourier-transform-by-example/
import numpy as np
import matplotlib.pyplot as plt

f = 40.
t = np.linspace(0, 0.5, 500)  # Number of sample points

s = 2*np.sin(2 * np.pi * f * t) # + 0.5 * np.sin(90 * 2 * np.pi * t)

plt.ylabel("Amplitude")
plt.xlabel("Time [s]")
plt.plot(t, s)
plt.show()

fft = np.fft.fft(s)  # type: numpy.array
indices_of_interest = [0, 1, 2, int(len(fft)/2), 498, 499]
for index in indices_of_interest:
    print('Value at index {}: {} '.format(index, fft[index]))

# Plot the frequency spectrum of the sin wave function
'''
We plot only half of the spectrum, because that is the only 
half giving us real information.
'''
T = t[1] - t[0]  # sampling interval
N = s.size

f = np.linspace(0, 1 / T, N) # 1/T = frequency

plt.ylabel("Amplitude")
plt.xlabel("Frequency [Hz]")
plt.plot(f[:N // 2], np.abs(fft)[:N // 2] * 1 / N) # 1 / N is a normalization factor
#plt.bar(f[:N // 2], np.abs(fft)[:N // 2] * 1 / N, width=1.5)  # 1 / N is a normalization factor
plt.show()

'''
Note: The normalisation with 1/N results to an FFT amplitude equal to
to the amplitude signal that you import to the FFT. However, following
the process above you split the power between the positive and negative sides.
Therefore the amplitude of the positive side only (which is the one that you plot)
will have half the amplitude of the signal you input to the FFT. Specifically
here the amplitude is 0.5 instead of 1.  
'''