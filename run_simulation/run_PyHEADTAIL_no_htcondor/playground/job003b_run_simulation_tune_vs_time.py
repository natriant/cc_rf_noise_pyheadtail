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
bfile = open('bunch', 'rb')
bunch = pickle.load(bfile)
bfile.close()

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

    X.append(bunch.x)
    Y.append(bunch.y)

print('--> Tracking Done.')

# 5. COMPUTE THE TUNE
x_data = {}
y_data = {}

for particle in range(pp.macroparticlenumber):
    x_data[particle] = []
    y_data[particle] = []
for particle in range(pp.macroparticlenumber):
    for turn in range(n_turns):
        x_data[particle].append(X[turn][particle])
        y_data[particle].append(Y[turn][particle])

# Parameters for sliding window
window_size = 50
step = 10

# Dictionaries for Qx, Qy. Each entry corresponds to one window
Qx_dict = {}
Qy_dict = {}

print('--> Start computing tunes')

for particle in range(pp.macroparticlenumber):
    Qx_dict[particle] = []
    Qy_dict[particle] = []
    counter = 0
    while True:
        window_signal_x = x_data[particle][counter:window_size+counter]
        window_signal_y = y_data[particle][counter:window_size + counter]
        counter = counter + step
        # Compute tune
        Qx_dict[particle].append(pnf.get_tune(np.array(window_signal_x), 0))
        Qy_dict[particle].append(pnf.get_tune(np.array(window_signal_y), 0))
        if x_data[particle][-1] in window_signal_x:
            break

plt.plot(np.arange(0, n_turns, n_turns/len(Qx_dict[0])), Qy_dict[0], '.', label='Qy')
plt.plot(np.arange(0, n_turns, n_turns/len(Qx_dict[0])), Qx_dict[0], '.', label='Qx')
plt.ylabel('Qx,y')
plt.xlabel('turns')
plt.grid()
plt.legend()
plt.show()

print('--> Computing tunes Done.')

save_tunes = True
if save_tunes:
    my_tunes = {'Qx': Qx_dict, 'Qy': Qy_dict}
    pickle.dump(my_tunes, open('mytunes_noKick_window50.pkl', 'wb'))
