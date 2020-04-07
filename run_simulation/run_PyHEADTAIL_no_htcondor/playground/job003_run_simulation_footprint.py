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


n_turns = 10000 # 1000 turns are enough for the NAFF algorithm to compute the tune

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

bfile = open('ampKicks', 'rb')
ampKicks = pickle.load(bfile)
bfile.close()

# 3. CREATE LISTS TO SAVE THE TBT DATA
X = []
Y = []

ampKicks = (np.random.normal(0, pp.stdAmpNoise, n_turns))
# 4. SET UP THE ACCELERATOR AND START TRACKING
for i in range(n_turns):

    bunch.yp += ampKicks[i] * np.sin(2 * np.pi * 400e6 / (bunch.beta * c) * bunch.z)

    # The next two lines actually run the simulation
    for m in one_turn_map:
        m.track(bunch)

    X.append(bunch.x)
    Y.append(bunch.y)

# 5. COMPUTE THE TUNE
x_data = {}
y_data = {}

for particle in range(pp.macroparticlenumber):
    x_data[particle] = []
    y_data[particle] = []
# maybe even 100 turns are enough
for particle in range(pp.macroparticlenumber):
    for turn in range(n_turns):
        x_data[particle].append(X[turn][particle])
        y_data[particle].append(Y[turn][particle])

Qx_list = []
Qy_list = []


for particle in range(pp.macroparticlenumber):
    signal_x = x_data[particle]
    Qx_list.append(pnf.get_tune(np.array(signal_x), 0))

    signal_y = y_data[particle]
    Qy_list.append(pnf.get_tune(np.array(signal_y), 0))

print('--> Computing tunes Done.')
dataExport = [Qx_list, Qy_list]

save_tunes = True
if save_tunes:
    f = open('mytunes_noTuneSpread_noAN_1e4.txt', 'w')
    with f:
        out = csv.writer(f, delimiter=',')
        out.writerows(zip(*dataExport))


plt.scatter(Qx_list, Qy_list)
plt.ylim(0.179, 0.1805)
plt.xlim(0.124, 0.131)
plt.xlabel('Qx')
plt.ylabel('Qy')
plt.grid()
plt.show()
