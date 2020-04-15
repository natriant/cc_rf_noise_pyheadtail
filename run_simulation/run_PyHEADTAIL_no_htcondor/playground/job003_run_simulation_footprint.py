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


n_turns = 1000 # 1000 turns are enough for the NAFF algorithm to compute the tune

# 1. LOAD OBJECTS FROM FILES, BUNCH AND NOISE KICKS
bfile = open('input/bunch', 'rb')
bunch = pickle.load(bfile)
bfile.close()

bfile = open('input/ampKicks', 'rb')
ampKicks = pickle.load(bfile)
bfile.close()

# 2. CREATE ONE TURN MAP

# Set the flag True if the detuning coefficients are given to 1/m you need to convert
# them to PyHEADTAIL units.
convert_hanarmonicities = True
if convert_hanarmonicities:
    scale_factor = 2*bunch.p0
    app_x = pp.app_x*scale_factor
    app_y = pp.app_y * scale_factor
    app_xy = pp.app_xy * scale_factor
else:
    app_x = pp.app_x
    app_y = pp.app_y
    app_xy = pp.app_xy

transverse_map = TransverseMap(pp.s, pp.alpha_x, pp.beta_x, pp.D_x, pp.alpha_y, pp.beta_y, pp.D_y, pp.Q_x, pp.Q_y,
    [Chromaticity(pp.Qp_x, pp.Qp_y),
    AmplitudeDetuning(app_x, app_y, app_xy)])

longitudinal_map = LinearMap([pp.alpha], pp.circumference, pp.Q_s)

one_turn_map = [transverse_map[0]] + [longitudinal_map]

# 3. CREATE LISTS TO SAVE THE TBT DATA
X = []
Y = []
DP = []

# 4. SET UP THE ACCELERATOR AND START TRACKING
for i in range(n_turns):

    # bunch.yp += ampKicks[i] * np.sin(2 * np.pi * 400e6 / (bunch.beta * c) * bunch.z)

    # The next two lines actually run the simulation
    for m in one_turn_map:
        m.track(bunch)

    X.append(bunch.x)
    Y.append(bunch.y)
    DP.append(bunch.dp)

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

plt.plot()
save_tunes = False
if save_tunes:
    f = open('./output/mytunes_nominalSPS_QpxQPy2.txt', 'w')
    with f:
        out = csv.writer(f, delimiter=',')
        out.writerows(zip(*dataExport))


plt.scatter(Qx_list, Qy_list)
plt.ylim(0.1797, 0.1797+0.0004)
plt.xlim(0.128, 0.128+0.004)
plt.xlabel('Qx')
plt.ylabel('Qy')
plt.grid()
plt.tight_layout()
# plt.savefig('footprint_AN_ayy0.png')
plt.show()
