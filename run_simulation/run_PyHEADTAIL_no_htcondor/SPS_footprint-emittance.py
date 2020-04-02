#==========================================================
#                       Imports
#==========================================================
from __future__ import division

import os, sys, time
import h5py as hp
import numpy as np
import matplotlib.pyplot as plt
import pickle
import csv
import NAFFlib as pnf

# Added by Themis
import scipy
#sys.path.append('/Applications/anaconda/pkgs/')
#sys.path.append('/nfsbigdata1/tmastorigrp/src/')

from scipy.constants import m_p, c, e
from mpl_toolkits.mplot3d import Axes3D
from math import *

from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.particles.generators import generate_Gaussian6DTwiss 
from PyHEADTAIL.trackers.detuners import Chromaticity, AmplitudeDetuning
from PyHEADTAIL.trackers.transverse_tracking import TransverseMap
from PyHEADTAIL.trackers.simple_long_tracking import RFSystems, LinearMap
from PyHEADTAIL.monitors.monitors import BunchMonitor, SliceMonitor, ParticleMonitor
from PyHEADTAIL.feedback.transverse_damper import TransverseDamper
from PyHEADTAIL.impedances.wakes import CircularResonator, WakeTable, WakeField


#==========================================================
#               Variables We Change
#==========================================================
n_turns = int(1e5)            #number of cycles to run the simulation for
decTurns = int(100)             #how often to record data
n_integrals = int(1e3)

Q_x = 26.13                 #How many times the particles oscillate in phase space each turn

Q_y = 26.18                           # Will need it to be 16.25 IF CC feedback is used
                            # For this version of PyHEADTAIL Q_x should be an array

ampGain = 0               #strength of amplitude feedback (usually between 0 and 0.15)
phaseGain = 0             #strength of phase feedback (usually between 0 and 0.15)

filename = 'file.txt'      #Where the data for the run is saved

numDelay = 1                #Turns of delay between measuring and acting with the feedback system
                            #Make sure to adjust Q_x if adjusting numDelay

ampNoiseOn = 1             #Turns on the amplitude noise - 0 is off, 1 is on
phaseNoiseOn = 0            #Turns on the phase noise - 0 is off, 1 is on
stdAmpNoise = 1e-8          #Size of amplitude noise (1e-8 for ~22nm/s at 0 ampGain)
stdPhaseNoise = 1e-8      #Size of phase noise (1e-8 for ~24nm/s at 0 phaseGain)

damperOn = 0                #Turns on the damper - 0 is off, 1 is on
dampingrate_x = 50          #Strength of the damper (note it must be turned on further down in the code)
                            #(40 is the "standard" value)

measNoiseOn = 0             #Turns on the measurement noise - 0 is off, 1 is on
stdMeasNoise = 1000e-9       #standard deviation of measurement noise

#==========================================================
#           Variables We (Usually) Do Not Change
#==========================================================

gamma          = 287.8
p0             = m_p*c*np.sqrt(gamma**2 - 1)
beta           = np.sqrt(1 - 1/gamma**2)
circumference  = 6911.5623
frev           = 299792458/circumference

# PARAMETERS FOR TRANSVERSE MAP
# =====================
n_segments     = 1
s              = np.arange(0, n_segments + 1) * circumference / n_segments
alpha_x        = -0.8757651182 * np.ones(n_segments) 
beta_x         = 29.23897404 * np.ones(n_segments) 
D_x            = -0.4837377902 * np.ones(n_segments)
alpha_y        = 1.898525134  * np.ones(n_segments) 
beta_y         = 76.07315729 * np.ones(n_segments) 
D_y            = 0 * np.ones(n_segments)

Qp_x           = 0 #10
Qp_y           = 0

# PARAMETERS FOR LONGITUDINAL MAP
# =======================
alpha           = 1.9e-3
Q_s             = 0.0035
h1, h2          = 4620, 9240
V1, V2          = 4.5e6, 0e6
dphi1, dphi2    = 0, np.pi
p_increment     = 0 * e/c * circumference/(beta*c)


# CREATE BEAM
# ===========
macroparticlenumber = int(1e4)

charge    = e
mass      = m_p
intensity = 1.5e11

R         = circumference/(2*np.pi)
eta       = alpha-1/gamma**2
beta_z    = np.abs(eta)*R/Q_s

epsn_x    = 2e-6
epsn_y    = 2e-6
    
epsn_z    = 2.5
sigma_z   = 0.155 #2*0.0755

sigma_x   = np.sqrt(epsn_x/(beta*gamma) * beta_x[0])
sigma_xp  = sigma_x/beta_x[0]
sigma_y   = np.sqrt(epsn_y/(beta*gamma) * beta_y[0])
sigma_yp  = sigma_y/beta_y[0]
sigma_dp  = sigma_z/beta_z
epsn_z    = 4*np.pi * p0/e * sigma_z*sigma_dp


# Create bunch just to create the object. The redifine all the coordinates
bunch     = generate_Gaussian6DTwiss(
        macroparticlenumber, intensity, charge, mass, circumference, gamma,
        alpha_x, alpha_y, beta_x, beta_y, beta_z, epsn_x, epsn_y, epsn_z)

# Create the initial condition
steps = int(np.sqrt(macroparticlenumber))
J_min = 10**(-12)
Jx_max = 10**(-7)
Jy_max = 10**(-7)
Jx = np.linspace(J_min, Jx_max, steps)
Jy = np.linspace(J_min, Jy_max, steps)

x = np.sqrt(Jx*beta_x[0]*2)
y = np.sqrt(Jy*beta_y[0]*2)

# meshgrid
xx, yy = np.meshgrid(x, y)


bunch.x = xx.flatten()
bunch.y = yy.flatten()


bunch.xp = 0 * np.ones(macroparticlenumber)
bunch.yp = 0 * np.ones(macroparticlenumber)
bunch.z = 0 * np.ones(macroparticlenumber)
bunch.dp = 0 * np.ones(macroparticlenumber)


afile = open('./input/bunch', 'wb')
pickle.dump(bunch, afile)
afile.close()

scale_factor = 2*bunch.p0


### CREATE THE MAPS

# Define the detuning coefficients, in PyHEADTAIL units
app_x = 179.3610966*scale_factor
app_y = 0.0 *scale_factor
app_xy = 0.0
######



### CREATE THE TRANSVERSE MAP
transverse_map = TransverseMap(s, alpha_x, beta_x, D_x, alpha_y, beta_y, D_y, Q_x, Q_y,
    [Chromaticity(Qp_x, Qp_y),
    AmplitudeDetuning(app_x, app_y, app_xy)])


### CREATE THE LONGITUDINAL MAP
longitudinal_map = LinearMap([alpha], circumference, Q_s)

# CREATE DAMPER
# =============
dampingrate_y = 10 #40
damper = TransverseDamper(dampingrate_x, dampingrate_y)

# ======================================================================
# SET UP ACCELERATOR MAP AND START TRACKING
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

if ampNoiseOn == 1:
    ampKicks = np.random.normal(0, stdAmpNoise, n_turns)
else:
    ampKicks = np.zeros(n_turns)
if phaseNoiseOn == 1:
    phaseKicks = np.random.normal(0, stdPhaseNoise, n_turns)
else:
    phaseKicks = np.zeros(n_turns)
if measNoiseOn == 1:
    noise = np.random.normal(0, stdMeasNoise, n_turns)# / beta_x[0] #Remove beta_x[0] when measuring in x
else:
    noise = np.zeros(n_turns)

delayAmp = np.zeros(numDelay + 1)
delayPhase = np.zeros(numDelay + 1)

t0 = time.clock()

#reload object from file
file2 = open('./input/bunch', 'rb')
bunch = pickle.load(file2)
file2.close()

print(bunch.x)
print(bunch.epsn_x())
print(bunch.epsn_y())
quit()

print('--> Begin tracking...')
if damperOn == 1:
    one_turn_map = [transverse_map[0]] + [longitudinal_map] + [damper]
else:
    one_turn_map = [transverse_map[0]] + [longitudinal_map]

n_damped_turns = int(n_turns/decTurns) # The total number of turns at which the data are damped.
                       # We want this number as an integer, so it can be used in the next functions.
Y1 = []
Y2 = []
X1 = []
X2 = []
emitX = []
emitY = []

for i in range(n_turns):

    # Crab cavity
    Vcc = 1e6
    p_cc = Vcc/(gamma*.938e9) # Vo/Eb
    #  bunch.xp += (i/n_turns)*p_cc*np.sin(2*np.pi*400e6/(bunch.beta*c)*bunch.z)

    # Gaussian Amplitude noise
    bunch.yp += ampKicks[i]*np.sin(2*np.pi*400e6/(bunch.beta*c)*bunch.z)

    # Gaussian Phase noise
    bunch.yp += phaseKicks[i]*np.cos(2*np.pi*400e6/(bunch.beta*c)*bunch.z)

    #These next two lines actually "run" the simulation - the computationally heavy part
    for m in one_turn_map:
        m.track(bunch)

    if i < n_integrals:
        X1.append(bunch.x)
        Y1.append(bunch.y)
    if i>int(n_turns-n_integrals-1):
        X2.append(bunch.x)
        Y2.append(bunch.y)
    emitX.append(bunch.epsn_x())
    emitY.append(bunch.epsn_y())

    if i%decTurns is  0:
        print('turn {}'.format(i))

print(len(Y1))
print(len(Y2))
# calculate the tune spread
y_data_1 = {}
y_data_2 = {}
x_data_1 = {}
x_data_2 = {}

for particle in range(macroparticlenumber):
    y_data_1[particle] = []
    y_data_2[particle] = []
    x_data_1[particle] = []
    x_data_2[particle] = []
# maybe even 100 turns are enough
for particle in range(macroparticlenumber):
    for turn in range(0, n_integrals):
        y_data_1[particle].append(Y1[turn][particle])
        y_data_2[particle].append(Y2[turn][particle])
        x_data_1[particle].append(X1[turn][particle])
        x_data_2[particle].append(X2[turn][particle])

y1_lost_particles = []
Qy1_list = []
y2_lost_particles = []
Qy2_list = []
Qx1_list = []
Qx2_list = []

for particle in range(macroparticlenumber):
    if np.isnan(y_data_1[particle]).any() :
        y1_lost_particles.append(particle)
        print('particle {} lost'.format(particle))
    else:
        signal_y1 = y_data_1[particle]
        signal_x1 = x_data_1[particle]
        Qy1_list.append(pnf.get_tune(np.array(signal_y1)))
        Qx1_list.append(pnf.get_tune(np.array(signal_x1)))
    if np.isnan(y_data_2[particle]).any():
        y2_lost_particles.append(particle)
        print('particle {} lost'.format(particle))
    else:
        signal_y2 = y_data_2[particle]
        Qy2_list.append(pnf.get_tune(np.array(signal_y2)))
        signal_x2 = x_data_2[particle]
        Qx2_list.append(pnf.get_tune(np.array(signal_x2)))


dataExport = [Qy1_list, Qy2_list, Qx1_list, Qx2_list, emitX, emitY]

f = open('./output/'+filename, 'w')

with f:
    out = csv.writer(f, delimiter=',')
    out.writerows(zip(*dataExport))

print('--> Done.')

print("Simulation time in seconds: " + str(time.clock() - t0))

plt.scatter(Qx1_list, Qy1_list, 'C0', label='first 1000 turns')
plt.scatter(Qx2_list, Qy2_list, 'C1', label='last 1000 turns')
plt.ylim(0.179, 0.1797+0.0004)
plt.xlim(0.128, 0.128+0.004)
plt.xlabel('Qx')
plt.ylabel('Qy')
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('./output/tune_spread.png')
plt.close()

plt.plot(np.arange(n_turns), emitY)
plt.xlabel('turns')
plt.ylabel('dey/dt')
plt.grid()
plt.tight_layout()
plt.savefig('./output/emittance_growth.png')
plt.close()

