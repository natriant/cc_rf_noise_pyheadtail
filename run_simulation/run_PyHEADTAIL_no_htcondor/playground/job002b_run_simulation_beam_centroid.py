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

longitudinal_map = LinearMap([pp.alpha], pp.circumference, pp.Q_s)

# Create one turn map
one_turn_map = [transverse_map[0]] + [longitudinal_map]

# 2. Reload objects from files
bfile = open('input/bunch', 'rb')
bunch = pickle.load(bfile)
bfile.close()

#z_temp = np.split(bunch.z, 2)[0]
#z_temp_new = z_temp*(-1)
#bunch.z = np.concatenate([z_temp, z_temp_new])


bfile = open('input/ampKicks', 'rb')
ampKicks = pickle.load(bfile)
bfile.close()

# 3. Lists for tbt data
meanX = []
meanY = []
emitX = []
emitY = []
emitY_2 = []
emitY3 = []
stdY = []
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

    meanX.append(np.mean(bunch.x))
    stdY.append(np.std(bunch.y))
    meanY.append(np.mean(bunch.y))
    emitX.append(bunch.epsn_x())
    emitY.append(bunch.epsn_y())
    emitY_2.append((np.std(bunch.y)**2)/pp.beta_y)
    yy = bunch.y-np.mean(bunch.y)
    ypp = bunch.yp - np.mean(bunch.yp)
    emitY3.append(np.sqrt(np.mean(yy**2)*np.mean(ypp**2)-np.mean(yy*ypp)**2))


dataExport = [meanX, meanY, emitX, emitY]

sigma_y = 0.0007270874265680602
plt.plot(np.array(meanY)/sigma_y,  c='b')
plt.xlabel('turns')
plt.ylabel('<y>/sigma_y)')
plt.tight_layout()
plt.grid()
plt.show()

plt.plot(emitY, c='b', label='pyheadtail method')
plt.plot(np.array(emitY3)*pp.beta*pp.gamma, c='r', label='method 2')
plt.legend()
plt.xlabel('turns')
plt.ylabel('ey')
plt.tight_layout()
plt.grid()
plt.show()

quit()
f = open('test1.txt', 'w')
with f:
    out = csv.writer(f, delimiter=',')
    out.writerows(zip(*dataExport))

print('--> Done.')

print("Simulation time in seconds: " + str(time.clock() - t0))