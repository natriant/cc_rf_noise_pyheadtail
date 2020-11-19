import simulation_parameters as pp
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
import csv
from scipy.constants import c
from PyHEADTAIL.trackers.detuners import Chromaticity, AmplitudeDetuning
from PyHEADTAIL.trackers.transverse_tracking import TransverseMap
from PyHEADTAIL.trackers.simple_long_tracking import RFSystems, LinearMap


# 1. Crete transverse and longitudinal map
transverse_map = TransverseMap(pp.s, pp.alpha_x, pp.beta_x, pp.D_x, pp.alpha_y, pp.beta_y, pp.D_y, pp.Q_x, pp.Q_y,
                               [Chromaticity(pp.Qp_x, pp.Qp_y),
                                AmplitudeDetuning(pp.app_x, pp.app_y, pp.app_xy)])

#longitudinal_map = LinearMap([pp.alpha], pp.circumference, 1)
longitudinal_map = LinearMap([pp.alpha], pp.circumference, pp.Q_s)

# Create one turn map
one_turn_map = [transverse_map[0]] + [longitudinal_map]

# 2. Reload objects from files
bfile = open('input/bunch', 'rb')
bunch = pickle.load(bfile)
bfile.close()

bfile = open('input/ampKicks', 'rb')
ampKicks = pickle.load(bfile)
bfile.close()

# 3. Lists for tbt data
meanX = []
meanY = []
emitX = []
emitY = []

# 4. Set up accelerator map and start tracking
t0 = time.clock()
for i in range(pp.n_turns):
    # Gaussian Amplitude noise
    bunch.yp += ampKicks[i] * np.sin(2 * np.pi * 400e6 / (bunch.beta * c) * bunch.z)

    # Gaussian Phase noise
    # bunch.yp += phaseKicks[i]*np.cos(2*np.pi*400e6/(bunch.beta*c)*bunch.z)

    # start tracking
    for m in one_turn_map:
        m.track(bunch)

    if i % pp.decTurns is 0:
        j = int(i / pp.decTurns)
        meanX.append(np.mean(bunch.x))
        meanY.append(np.mean(bunch.y))
        emitX.append(bunch.epsn_x())
        emitY.append(bunch.epsn_y())


dataExport = [meanX, meanY, emitX, emitY]

# Plotting
plt.plot(np.array(emitY)*1e6)
plt.xlabel('turns')
plt.ylabel('ey')
plt.tight_layout()
plt.grid()
plt.show()
plt.plot(emitX)
plt.xlabel('turns')
plt.ylabel('ex')
plt.show()
plt.close()
plt.show()


save_tbt = True
if save_tbt:
    f = open('output/ayy0_axx0_noAN_emit.txt', 'w')
    with f:
        out = csv.writer(f, delimiter=',')
        out.writerows(zip(*dataExport))

print('--> Done.')

print("Simulation time in seconds: " + str(time.clock() - t0))