import simulation_parameters as pp
import time
import numpy as np
import pickle
import csv
import NAFFlib as pnf
from scipy.constants import c
from PyHEADTAIL.trackers.detuners import Chromaticity, AmplitudeDetuning
from PyHEADTAIL.trackers.transverse_tracking import TransverseMap
from PyHEADTAIL.trackers.simple_long_tracking import RFSystems, LinearMap
import matplotlib.pyplot as plt


n_turns = 100000 # 1000 turns are enough for the NAFF algorithm to compute the tune

# 1. CREATE ONE TURN MAP
transverse_map = TransverseMap(pp.s, pp.alpha_x, pp.beta_x, pp.D_x, pp.alpha_y, pp.beta_y, pp.D_y, pp.Q_x, pp.Q_y,
    [Chromaticity(pp.Qp_x, pp.Qp_y),
    AmplitudeDetuning(pp.app_x, pp.app_y, pp.app_xy)])

longitudinal_map = LinearMap([pp.alpha], pp.circumference, pp.Q_s)

one_turn_map = [transverse_map[0]] + [longitudinal_map]

# 2. LOAD OBJECTS FROM FILES, BUNCH AND NOISE KICKS
bfile = open('bunch', 'rb')
bunch = pickle.load(bfile)
bfile.close()

bunch.x = bunch.x[0]
bunch.y = bunch.y[0]
bunch.xp = bunch.xp[0]
bunch.yp = bunch.yp[0]
bunch.z = bunch.z[0]
bunch.dp = bunch.dp[0]

print(bunch.x)
bfile = open('ampKicks', 'rb')
ampKicks = pickle.load(bfile)
bfile.close()

# 3. CREATE LISTS TO SAVE THE TBT DATA
X = []
Y = []

# 4. SET UP THE ACCELERATOR AND START TRACKING
for i in range(n_turns):

    #bunch.yp += ampKicks[i] * np.sin(2 * np.pi * 400e6 / (bunch.beta * c) * bunch.z)

    # The next two lines actually run the simulation
    for m in one_turn_map:
        m.track(bunch)

    X.append(bunch.x[0])


y= X
# subtract the mean , in order to remove the DC component
average_Qx = np.average(y)


y[:] = [x - average_Qx for x in y]


fourier_qx = np.fft.fft(y)

freqs_qx = np.fft.fftfreq(len(fourier_qx))


fig, ax = plt.subplots()
plt.plot(freqs_qx, 2*abs(fourier_qx) / len(fourier_qx), 'o', label='Qx')#, label='FFT whole signal')
plt.yscale('log')
plt.show()