import matplotlib.pyplot as plt
import numpy as np


n_turns = int(1e4)            # number of cycles to run the simulation for
decTurns = int(1)           # how often to record data
circumference  = 6911.5623 # [m]
f_rev  = 299792458/circumference  # [Hz] The revolution frequency of the machine
separationTime = 1 / f_rev * decTurns #There are decTurns revolutions before each recording of data
tspan = (np.arange(n_turns / decTurns) + 1.0) * separationTime #total time spanned by the simulation


meanX, meanY, meanXsq, meanYsq, emitX, emitY = np.loadtxt('file_globalCC_PN_ramp200.txt', delimiter=",", unpack=True)
meanX, meanY, meanXsq, meanYsq, emitX, emitY = np.loadtxt('file_noNoise.txt', delimiter=",", unpack=True)

plt.plot(emitY[200:]*1e6)
plt.ylim(2.05, 2.1)

[m, b], cov = np.polyfit(tspan[200:], emitY[200:], 1, cov=True)
# compute the error of the gradient of the fit slope
err = np.sqrt(np.diag(cov))
print(m)
print(err[0])
plt.plot(np.arange(n_turns), np.arange(n_turns)*m+b*1e6, label=f'y={m*1e9:.2f}e-9x')
plt.xlabel('turns')
plt.ylabel(r'$\mathrm{\epsilon_y} \ [\mu m]$')

plt.legend()
plt.savefig('dey_globalCC_noNoise.png', bbox_inches='tight')
plt.show()