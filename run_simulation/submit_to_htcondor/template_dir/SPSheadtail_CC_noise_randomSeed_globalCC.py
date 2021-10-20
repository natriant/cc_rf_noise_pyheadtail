from __future__ import division

import os, sys, time
import h5py as hp
import numpy as np
import csv
import pickle
import scipy
from scipy.constants import m_p, c, e

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/afs/cern.ch/work/n/natriant/private/PyHEADTAIL/')

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
n_turns = int(1e5)            # number of cycles to run the simulation for
decTurns = int(100)               # how often to record data

ampGain = 0  # strength of amplitude feedback (usually between 0 and 0.15)
phaseGain = 0  # strength of phase feedback (usually between 0 and 0.15)

filename = 'file.txt'  # Where the data for the run is saved

numDelay = 1 #Turns of delay between measuring and acting with the feedback system
            #Make sure to adjust Q_x if adjusting numDelay

ampNoiseOn = 0  # Turns on the amplitude noise - 0 is off, 1 is on
phaseNoiseOn = 1  # Turns on the phase noise - 0 is off, 1 is on
stdAmpNoise = 1e-8 #*np.sqrt(1.82)  # Size of amplitude noise (1e-8 for ~22nm/s at 0 ampGain)
stdPhaseNoise = 1e-8 #*np.sqrt(0.72)  # Size of phase noise (1e-8 for ~24nm/s at 0 phaseGain)

damperOn = 0  # Turns on the damper - 0 is off, 1 is on
dampingrate_x = 0 #50  # Strength of the damper (note it must be turned on further down in the code)
                            #(40 is the "standard" value)

wakefieldOn = 1         # Turns on the wakefields

measNoiseOn = 0             # Turns on the measurement noise - 0 is off, 1 is on
stdMeasNoise = 1000e-9       # standard deviation of measurement noise



#==========================================================
#           Variables We (Usually) Do Not Change
#==========================================================

gamma = 213.16 #287.8
P0C = 199.9e9 # eV *c (?)

p0 = m_p*c*np.sqrt(gamma**2 - 1)
beta = np.sqrt(1 - 1/gamma**2)
circumference = 6911.5623
frev = 299792458/circumference


# PARAMETERS FOR TRANSVERSE MAP
# =====================
n_segments = 2
s = np.arange(0, n_segments+1)*circumference/n_segments
# optics at CC2
alpha_x = 0 * np.ones(n_segments)  # -0.9120242128
beta_x = 30.31164764 * np.ones(n_segments)
D_x = 0 * np.ones(n_segments) # -0.4954033073
alpha_y = 0 * np.ones(n_segments)  # 1.862209583
beta_y = 73.81671646 * np.ones(n_segments)
D_y = 0 * np.ones(n_segments)

i_wake = 1
# beta_x[i_wake] = 54.65 #### (for Q20)
# beta_y[i_wake] = 54.51 #### (for Q20)
beta_x[i_wake] = 42.0941 #### (for Q26)
beta_y[i_wake] = 42.0137 #### (for Q26)

Q_x, Q_y = 26.13, 26.18
Qp_x, Qp_y = 1.0, 1.0 #10

# detuning coefficients in (1/m)
app_x = 0.0  #2.4705e-15 #4e-11
app_xy = %axy #0.0 #-0*2.25e-11
app_y = %ayy #2000.0 #15000  #-7.31-14 #0*3e-11


# PARAMETERS FOR LONGITUDINAL MAP
# =======================
alpha = 1.9e-3
Q_s = 0.0051 #35
h1, h2 = 4620, 9240
V1, V2 = 5.008e6, 0e6
dphi1, dphi2 = 0, np.pi
p_increment = 0 * e/c * circumference/(beta*c)

# CREATE DAMPER
# =============
dampingrate_y = 40 #40
damper = TransverseDamper(dampingrate_x, dampingrate_y)

# CREATE BEAM
# ===========
macroparticlenumber = int(5e5) # 100000

charge = e
mass = m_p
intensity = 3.0e10

R = circumference/(2*np.pi)
eta = alpha-1/gamma**2
beta_z = np.abs(eta)*R/Q_s
epsn_x = 2.3e-6  # m
epsn_y = 2.3e-6  # m
#epsn_z = 2.5  # m
tau = 2.2e-9 # 4 sigma_t [s]
sigma_z = c*tau/4 #0.155  # m
#sigma_z = %sigmaz #0.155
print(f'sigma_z= {sigma_z} [m]')

sigma_x = np.sqrt(epsn_x/(beta*gamma) * beta_x[0])
sigma_xp = sigma_x/beta_x[0]
sigma_y = np.sqrt(epsn_y/(beta*gamma) * beta_y[0])
sigma_yp = sigma_y/beta_y[0]
sigma_dp = sigma_z/beta_z
epsn_z = 4*np.pi * p0/e * sigma_z*sigma_dp
print(f'epsn_z={epsn_z}')

bunch = generate_Gaussian6DTwiss(
    macroparticlenumber, intensity, charge, mass, circumference, gamma,
    alpha_x[0], alpha_y[0], beta_x[0], beta_y[0], beta_z, epsn_x, epsn_y, epsn_z)
xoffset = 0e-4
yoffset = 0e-4
bunch.x += xoffset
bunch.y += yoffset


#afile = open('bunch', 'wb')
#pickle.dump(bunch, afile)
#afile.close()

# SLICER FOR WAKEFIELDS
# =====================
n_slices = 500 # 500
slicer_for_wakefields = UniformBinSlicer(n_slices, z_cuts=(-3.*sigma_z, 3.*sigma_z))#,circumference=circumference, h_bunch=h1)

# WAKEFIELD
# ==========
n_turns_wake = 1 # for the moment we consider that the wakefield decays after 1 turn
#wakefile1 = ('/afs/cern.ch/work/n/natriant/private/pyheadtail_example_crabcavity/wakefields/newkickers_Q26_2018_modified.txt')
wakefile1 = ('/afs/cern.ch/work/n/natriant/private/pyheadtail_example_crabcavity/wakefields/SPS_complete_wake_model_2018_Q26.txt')
ww1 = WakeTable(wakefile1, ['time', 'dipole_x', 'dipole_y', 'quadrupole_x', 'quadrupole_y'], n_turns_wake=n_turns_wake)
# only dipolar kick
#my_length = len(ww1.wake_table['quadrupole_x'])
#ww1.wake_table['quadrupole_x'] = np.zeros(my_length)
#ww1.wake_table['quadrupole_y'] = np.zeros(my_length)

# only quadrupolar kick
#my_length = len(ww1.wake_table['dipole_x'])
#ww1.wake_table['dipole_x'] = np.zeros(my_length)
#ww1.wake_table['dipole_y'] = np.zeros(my_length)

wake_field_kicker = WakeField(slicer_for_wakefields, ww1)#, beta_x=beta_x, beta_y=beta_y)

# CREATE TRANSVERSE AND LONGITUDINAL MAPS
# =======================================
scale_factor = 2*bunch.p0  # scale the detuning coefficients in pyheadtail units
transverse_map = TransverseMap(s, alpha_x, beta_x, D_x, alpha_y, beta_y, D_y, Q_x, Q_y,
    [Chromaticity(Qp_x, Qp_y),
    AmplitudeDetuning(app_x*scale_factor, app_y*scale_factor, app_xy*scale_factor)])

longitudinal_map = LinearMap([alpha], circumference, Q_s)



# ======================================================================
# SET UP ACCELERATOR MAP AND START TRACKING
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
seed = 12345 + %seed_step
rng = np.random.default_rng(seed)

_ = rng.normal(size=100000) 


if ampNoiseOn == 1:
    ampKicks = rng.normal(loc=0, scale=stdAmpNoise, size=n_turns)
else:
    ampKicks = np.zeros(n_turns)
if phaseNoiseOn == 1:
    phaseKicks = rng.normal(loc=0, scale=stdPhaseNoise, size=n_turns)
else:
    phaseKicks = np.zeros(n_turns)
if measNoiseOn == 1:
    noise = rng.normal(loc=0, scale=stdMeasNoise, size=n_turns) # / beta_x[0] #Remove beta_x[0] when measuring in x
else:
    noise = np.zeros(n_turns)

delayAmp = np.zeros(numDelay + 1)
delayPhase = np.zeros(numDelay + 1)

t0 = time.clock()

#reload object from file
#file2 = open('bunch', 'rb')
#bunch = pickle.load(file2)
#file2.close()

print('--> Begin tracking...')
one_turn_map = []
for i, segment in enumerate(transverse_map):
    one_turn_map.append(segment)
    if wakefieldOn:
        if i+1 == i_wake:
            one_turn_map.append(wake_field_kicker)
one_turn_map.append(longitudinal_map)

n_damped_turns = int(n_turns/decTurns) # The total number of turns at which the data are damped.
                       # We want this number as an integer, so it can be used in the next functions. 
meanX = np.zeros(n_damped_turns)
meanY = np.zeros(n_damped_turns)
meanXsq = np.zeros(n_damped_turns)
meanYsq = np.zeros(n_damped_turns)
emitX = np.zeros(n_damped_turns)
emitY = np.zeros(n_damped_turns)

Vcc = 1e6
cc_voltage = lambda turn: np.interp(turn, [0, 200, 1e12], Vcc * np.array([0, 1, 1])) # ramp up during the first 200 turns
deg2rad = np.pi/180
 
for i in range(n_turns):
    
    # Crab cavity
    Vcc = 1e6
    p_cc = Vcc/(gamma*.938e9)  # Vo/Eb

    # Gaussian Amplitude noise
    #bunch.xp += ampKicks[i]*np.sin(2*np.pi*400e6/(bunch.beta*c)*bunch.z)
    bunch.yp += [cc_voltage(i)+ ampKicks[i]] * np.sin(2 * np.pi * 400e6 / (bunch.beta * c) * bunch.z)/(gamma * .938e9)
     
    # Gaussian Phase noise
    if i == 0:
        bunch.yp += [cc_voltage(i)] * np.sin(2 * np.pi * 400e6 / (bunch.beta * c) * bunch.z)/(gamma * .938e9) 
    else:
        bunch.yp += [cc_voltage(i)] * np.sin(2 * np.pi * 400e6 / (bunch.beta * c) * bunch.z+phaseKicks[i]*P0C/cc_voltage(i))/(gamma * .938e9)  # phase and amplitude noise
    

    

    #These next two lines actually "run" the simulation - the computationally heavy part
    for m in one_turn_map:
        m.track(bunch)
    
    '''     
    negavg = np.mean(bunch.x[bunch.z < 0.0])
    posavg = np.mean(bunch.x[bunch.z > 0.0])
        
    #Amplitude Correction
    posCorr = (posavg-negavg)/2
    posCorr = posCorr + noise[i]
    momCorr = (ampGain)*posCorr/beta_x[0]
    delayAmp[0:-1] = delayAmp[1:]
    delayAmp[numDelay] = momCorr
    #bunch.xp += delayAmp[0]*np.sin(2*np.pi*400e6/(bunch.beta*c)*bunch.z)
    
    #Phase Correction
    posCorr = (posavg+negavg)/2
    posCorr = posCorr + noise[i]
    momCorr = (phaseGain)*posCorr/beta_x[0]
    delayPhase[0:-1] = delayPhase[1:]
    delayPhase[numDelay] = momCorr
    #bunch.xp += delayPhase[0]*np.cos(2*np.pi*400e6/(bunch.beta*c)*bunch.z)
    '''

    if i%decTurns is  0:
        j = int(i/decTurns)
        meanX[j] = np.mean(bunch.x)
        meanY[j] = np.mean(bunch.y)
        meanXsq[j] = np.mean((bunch.x-np.mean(bunch.x))**2)
        meanYsq[j] = np.mean((bunch.y-np.mean(bunch.y))**2)
        emitX[j] = bunch.epsn_x()
        emitY[j] = bunch.epsn_y()

dataExport = [meanX, meanY, meanXsq, meanYsq, emitX, emitY]

f = open(filename, 'w')

with f:
    out = csv.writer(f, delimiter=',')
    out.writerows(zip(*dataExport))

print('--> Done.')

print("Simulation time in seconds: " + str(time.clock() - t0))
