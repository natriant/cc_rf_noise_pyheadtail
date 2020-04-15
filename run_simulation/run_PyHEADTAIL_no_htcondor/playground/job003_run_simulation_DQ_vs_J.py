import simulation_parameters as pp
import numpy as np
import pickle
import csv
import NAFFlib as pnf
from scipy.constants import c
from PyHEADTAIL.trackers.detuners import Chromaticity, AmplitudeDetuning
from PyHEADTAIL.trackers.transverse_tracking import TransverseMap
from PyHEADTAIL.trackers.simple_long_tracking import RFSystems, LinearMap
import matplotlib.pyplot as plt


n_turns = 1000 # 1000 turns are enough for the NAFF algorithm to compute the tune

# 1. CREATE ONE TURN MAP
transverse_map = TransverseMap(pp.s, pp.alpha_x, pp.beta_x, pp.D_x, pp.alpha_y, pp.beta_y, pp.D_y, pp.Q_x, pp.Q_y,
    [Chromaticity(pp.Qp_x, pp.Qp_y),
    AmplitudeDetuning(pp.app_x, pp.app_y, pp.app_xy)])

longitudinal_map = LinearMap([pp.alpha], pp.circumference, pp.Q_s)

one_turn_map = [transverse_map[0]] + [longitudinal_map]

# 2. LOAD OBJECTS FROM FILES, BUNCH AND NOISE KICKS
bfile = open('input/bunch', 'rb')
bunch = pickle.load(bfile)
bfile.close()

# calculate initial actions
Jx = 1 / 2 * (
        (1 + pp.alpha_x ** 2) / pp.beta_x * bunch.x ** 2
        + 2 * pp.alpha_x * bunch.x * bunch.xp
        + pp.beta_x * bunch.xp ** 2)
Jy = 1 / 2 * (
        (1 + pp.alpha_y ** 2) / pp.beta_y * bunch.y ** 2
        + 2 * pp.alpha_y * bunch.y * bunch.yp
        + pp.beta_y * bunch.yp ** 2)


bfile = open('input/ampKicks', 'rb')
ampKicks = pickle.load(bfile)
bfile.close()

# 3. CREATE LISTS TO SAVE THE TBT DATA
X = []
Y = []


# 4. SET UP THE ACCELERATOR AND START TRACKING
for i in range(n_turns):

    bunch.yp += ampKicks[i] #* np.sin(2 * np.pi * 400e6 / (bunch.beta * c) * bunch.z)

    # The next  -1.1e-11two lines actually run the simulation
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

print(np.std(Qy_list))
print(np.std(Qx_list))

print('--> Computing tunes Done.')
dataExport = [Qx_list, Qy_list]

save_tunes = False
if save_tunes:
    f = open('mytunes_noTuneSpread_noAN_1e4.txt', 'w')
    with f:
        out = csv.writer(f, delimiter=',')
        out.writerows(zip(*dataExport))

plt.scatter(np.array(Jy)*1e9, Qy_list)
plt.xlabel('Jy (nm)')
plt.ylabel('Qy')
plt.ylim(0.179, 0.1805)
plt.tight_layout()
plt.grid()
plt.savefig('Dqy_vs_Jy_AN_ayy0.png')
plt.show()
