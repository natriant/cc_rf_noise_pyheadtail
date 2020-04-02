from __future__ import division

import os, sys, time
import h5py as hp
import numpy as np
import matplotlib.pyplot as plt
import csv
import pickle
import scipy
from scipy.constants import m_p, c, e
import NAFFlib as pnf

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
n_turns = int(1e5)           #number of cycles to run the simulation for
decTurns =int(1)               #how often to record data
n_integral = int(1e3)

Q_x = 26.13 #18                 #How many times the particles oscillate in phase space each turn 
                            # Will need it to be 16.25 IF CC feedback is used
                            # For this version of PyHEADTAIL Q_x should be an array

ampGain = 0               #strength of amplitude feedback (usually between 0 and 0.15)
phaseGain = 0             #strength of phase feedback (usually between 0 and 0.15)

filename = 'file.txt'      #Where the data for the run is saved

numDelay = 1                #Turns of delay between measuring and acting with the feedback system
                            #Make sure to adjust Q_x if adjusting numDelay

ampNoiseOn = 1              #Turns on the amplitude noise - 0 is off, 1 is on
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
alpha_x        = -0.8757651182* np.ones(n_segments) #0 * np.ones(n_segments)
beta_x         = 29.23* np.ones(n_segments) #75 * np.ones(n_segments) 
D_x            = -0.4837377902 * np.ones(n_segments) #0 * np.ones(n_segments)
alpha_y        = 1.898525134 * np.ones(n_segments) #0 * np.ones(n_segments)
beta_y         = 76.07315729*np.ones(n_segments) #72 * np.ones(n_segments) 
D_y            = 0 * np.ones(n_segments)

Qp_x           = 0 #10
Qp_y           = 0

Q_y            = 26.18 # 13

app_x          =  179.3610966
app_y          =  0.0 # already scaled for PyHEADTAIL (ie includes the factor 2*bunch.p0)
app_xy         =  0.0 #-441.3397664


#transverse_map = TransverseMap(s, alpha_x, beta_x, D_x, alpha_y, beta_y, D_y, Q_x, Q_y,
#    [Chromaticity(Qp_x, Qp_y),
#    AmplitudeDetuning(app_x, app_y, app_xy)]) 

# PARAMETERS FOR LONGITUDINAL MAP
# =======================
alpha           = 1.9e-3
Q_s             = 0.0035
h1, h2          = 4620, 9240
V1, V2          = 4.5e6, 0e6
dphi1, dphi2    = 0, np.pi
p_increment     = 0 * e/c * circumference/(beta*c)

#longitudinal_map = LinearMap([alpha], circumference, Q_s)

# CREATE DAMPER
# =============
dampingrate_y = 10 #40
damper = TransverseDamper(dampingrate_x, dampingrate_y)

# CREATE BEAM
# ===========
macroparticlenumber = 100000

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

bunch     = generate_Gaussian6DTwiss(
    macroparticlenumber, intensity, charge, mass, circumference, gamma,
    alpha_x, alpha_y, beta_x, beta_y, beta_z, epsn_x, epsn_y, epsn_z)
xoffset = 0e-4
yoffset = 0e-4
bunch.x += xoffset
bunch.y += yoffset


afile = open('bunch', 'wb')
pickle.dump(bunch, afile)
afile.close()

# ===================================
# CREATE TRANVERSE AND LONGITUDINAL MAPS
# ==================================
scale_factor = 2*bunch.p0 # for detuning coefficients

transverse_map = TransverseMap(s, alpha_x, beta_x, D_x, alpha_y, beta_y, D_y, Q_x, Q_y,
    [Chromaticity(Qp_x, Qp_y),
    AmplitudeDetuning(app_x*scale_factor, app_y, app_xy*scale_factor)])

longitudinal_map = LinearMap([alpha], circumference, Q_s)




# ======================================================================
# SET UP ACCELERATOR MAP AND START TRACKING
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
t0 = time.clock()

#reload object from file
file2 = open('bunch', 'rb')
bunch = pickle.load(file2)
file2.close()

print('--> Begin tracking...')
if damperOn == 1:
    one_turn_map = [transverse_map[0]] + [longitudinal_map] + [damper]
else:
    one_turn_map = [transverse_map[0]] + [longitudinal_map]

n_damped_turns = int(n_turns/decTurns) # The total number of turns at which the data are damped.
                       # We want this number as an integer, so it can be used in the next functions. 
X= []
Y= []
emitX = []
emitY = []
    
ampKicks = np.random.normal(0, stdAmpNoise, n_turns)


for i in range(n_turns):
    
    # Crab cavity
    Vcc = 1e6
    p_cc = Vcc/(gamma*.938e9) # Vo/Eb

    # Gaussian Amplitude noise
    bunch.yp += ampKicks[i]*np.sin(2*np.pi*400e6/(bunch.beta*c)*bunch.z)

    # Gaussian Phase noise
    #bunch.xp += phaseKicks[i]*np.cos(2*np.pi*400e6/(bunch.beta*c)*bunch.z)
    #bunch.yp += phaseKicks[i]*np.cos(2*np.pi*400e6/(bunch.beta*c)*bunch.z)

    #These next two lines actually "run" the simulation - the computationally heavy part
    for m in one_turn_map:
        m.track(bunch)
        
        

    emitX.append(bunch.epsn_x())
    emitY.append(bunch.epsn_y())

    if i < n_integral or i > (n_turns-n_integral-1):
        X.append(bunch.x)
        Y.append(bunch.y)


# calculate the tune spread
y_data = {}
x_data = {}
for particle in range(macroparticlenumber):
    y_data[particle] = []
    x_data[particle] = []
# maybe even 100 turns are enough
for particle in range(macroparticlenumber):
    for index in range(0, len(X)):
        y_data[particle].append(Y[index][particle])
        x_data[particle].append(X[index][particle])

Qy1_list = [] # first integral
Qy2_list = [] # second integral
Qx1_list = []
Qx2_list = []

for particle in range(macroparticlenumber):
    signal_y1 = y_data[particle][:len(X)//2]
    signal_y2 = y_data[particle][len(X)//2:] 
    signal_x1 = x_data[particle][:len(X)//2]
    signal_x2 = x_data[particle][len(X)//2:]
    
    Qy1_list.append(pnf.get_tune(np.array(signal_y1)))
    Qy2_list.append(pnf.get_tune(np.array(signal_y2)))
    Qx1_list.append(pnf.get_tune(np.array(signal_x1)))
    Qx2_list.append(pnf.get_tune(np.array(signal_x2)))
        

dataExport = [Qy1_list, Qy2_list, Qx1_list, Qx2_list, emitX, emitY]

f = open(filename, 'w')

with f:
    out = csv.writer(f, delimiter=',')
    out.writerows(zip(*dataExport))

print('--> Done.')

print("Simulation time in seconds: " + str(time.clock() - t0))
