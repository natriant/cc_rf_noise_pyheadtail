from __future__ import division

import os, sys, time
import h5py as hp
import numpy as np
import csv
import pickle
import scipy
from scipy.constants import m_p, c, e

from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.particles.generators import generate_Gaussian6DTwiss 
from PyHEADTAIL.trackers.detuners import Chromaticity, AmplitudeDetuning
from PyHEADTAIL.trackers.transverse_tracking import TransverseMap
from PyHEADTAIL.trackers.simple_long_tracking import RFSystems, LinearMap

import matplotlib.pyplot as plt


#==========================================================
#               Variables We Change
#==========================================================
n_turns = 1000 # int(1e5)            # number of cycles to run the simulation for
decTurns = 1#int(100)               # how often to record data

# CC noise
ampNoiseOn = 0  # Turns on the amplitude noise - 0 is off, 1 is on
phaseNoiseOn = 0  # Turns on the phase noise - 0 is off, 1 is on
stdAmpNoise = 1e-8  # Size of amplitude noise (1e-8 for ~22nm/s at 0 ampGain)
stdPhaseNoise = 1e-8  # Size of phase noise (1e-8 for ~24nm/s at 0 phaseGain)


#==========================================================
#           Variables We (Usually) Do Not Change
#==========================================================

gamma = 287.8
p0 = m_p*c*np.sqrt(gamma**2 - 1)
beta = np.sqrt(1 - 1/gamma**2)
circumference = 6911.5623
frev = 299792458/circumference


# PARAMETERS FOR TRANSVERSE MAP
# =====================
n_segments = 1
s = np.arange(0, n_segments+1)*circumference/n_segments
# optics at CC2
alpha_x = 0 * np.ones(n_segments)  # -0.9120242128
beta_x = 30.31164764 * np.ones(n_segments)
D_x = 0 * np.ones(n_segments) # -0.4954033073
alpha_y = 0 * np.ones(n_segments)  # 1.862209583
beta_y = 73.81671646 * np.ones(n_segments)
D_y = 0 * np.ones(n_segments)

Q_x, Q_y = 26.13, 26.18
Qp_x, Qp_y = 1.0, 1.0 #10

# detuning coefficients in (1/m)
app_x = 0.0  #2.4705e-15 #4e-11
app_xy = 0.0 #-0*2.25e-11
app_y = 0.0 #15000 #%ayy #15000  #-7.31-14 #0*3e-11


# PARAMETERS FOR LONGITUDINAL MAP
# =======================
alpha = 1.9e-3
Q_s = 0.0051 #35
h1, h2 = 4620, 9240
V1, V2 = 5.008e6, 0e6
dphi1, dphi2 = 0, np.pi
p_increment = 0 * e/c * circumference/(beta*c)

# CREATE BEAM
# ===========
macroparticlenumber = 1000#int(20e3) #int(5e5) # 100000

charge = e
mass = m_p
intensity = 3.5e10

R = circumference/(2*np.pi)
eta = alpha-1/gamma**2
beta_z = np.abs(eta)*R/Q_s
epsn_x = 2e-6  # m
epsn_y = 2e-12 #2e-6  # m # we need a very small vertical emittance so we practically have no offset
epsn_z = 2.5  # m
sigma_z = 0.155  # m

sigma_x = np.sqrt(epsn_x/(beta*gamma) * beta_x[0])
sigma_xp = sigma_x/beta_x[0]
sigma_y = np.sqrt(epsn_y/(beta*gamma) * beta_y[0])
sigma_yp = sigma_y/beta_y[0]
sigma_dp = sigma_z/beta_z
epsn_z = 4*np.pi * p0/e * sigma_z*sigma_dp

bunch = generate_Gaussian6DTwiss(
    macroparticlenumber, intensity, charge, mass, circumference, gamma,
    alpha_x[0], alpha_y[0], beta_x[0], beta_y[0], beta_z, epsn_x, epsn_y, epsn_z)
xoffset = 0e-4
yoffset = 0e-4
bunch.x += xoffset
bunch.y += yoffset




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
seed = 12345 + 10 #%seed_step
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


t0 = time.clock()


print('--> Begin tracking...')
one_turn_map = []
for i, segment in enumerate(transverse_map):
    one_turn_map.append(segment)
    #if wakefieldOn:
    #    if i+1 == i_wake:
    #        one_turn_map.append(wake_field_kicker)
one_turn_map.append(longitudinal_map)

n_damped_turns = int(n_turns/decTurns) # The total number of turns at which the data are damped.
                       # We want this number as an integer, so it can be used in the next functions.
meanX = np.zeros(n_damped_turns)
meanY = np.zeros(n_damped_turns)
meanXsq = np.zeros(n_damped_turns)
meanYsq = np.zeros(n_damped_turns)
emitX = np.zeros(n_damped_turns)
emitY = np.zeros(n_damped_turns)

# Crab cavity
Vcc = 1e6 # V
f_cc = 400e6 # Hz
ps = 0# 90.  # phase of the cavity


# Compute theoretically the transverse kick from the CC
E_0 = 270e9 # total energy of the reference particle [eV]
clight = 299792458 # speed of light [m/s]
k = 2 * np.pi * f_cc / clight # wavenumber of the cavity

n_particles = 1000.
start, stop = -4*sigma_z, 4*sigma_z
step = (stop-start)/n_particles
initial_sigmas = np.arange(start, stop, step)
muy = 0
delta_py_cc = Vcc * np.sin(ps + k * np.array(initial_sigmas))/E_0
y_co_cc = (np.sqrt(beta_y * beta_y)) * np.array(delta_py_cc) * np.cos(2 * np.pi * muy - np.pi * Q_y) / (
                2 * np.sin(np.pi * Q_y))

# Ramping up the CC
p_cc = Vcc / (gamma * .938e9)  # Vo/Eb
cc_voltage = lambda turn: np.interp(turn, [0, 200, 1e12], Vcc * np.array([0, 1, 1]))



for i in range(n_turns):

    # bunch.xp += (i/n_turns)*p_cc*np.sin(2*np.pi*400e6/(bunch.beta*c)*bunch.z)
    # bunch.yp += (i/n_turns)*p_cc*np.sin(2*np.pi*400e6/(bunch.beta*c)*bunch.z)
    bunch.yp += cc_voltage(i)*np.sin(2 * np.pi * f_cc / (bunch.beta * c) * bunch.z)/(gamma * .938e9)

    # Gaussian Amplitude noise
    # bunch.xp += ampKicks[i]*np.sin(2*np.pi*400e6/(bunch.beta*c)*bunch.z)
    bunch.yp += ampKicks[i] * np.sin(2 * np.pi * f_cc / (bunch.beta * c) * bunch.z)

    # Gaussian Phase noise
    # bunch.xp += phaseKicks[i]*np.cos(2*np.pi*400e6/(bunch.beta*c)*bunch.z)
    bunch.yp += phaseKicks[i] * np.cos(2 * np.pi * f_cc / (bunch.beta * c) * bunch.z)

    # These next two lines actually "run" the simulation - the computationally heavy part
    for m in one_turn_map:
        m.track(bunch)

    if i % decTurns is 0:
        j = int(i / decTurns)
        meanX[j] = np.mean(bunch.x)
        meanY[j] = np.mean(bunch.y)
        meanXsq[j] = np.mean((bunch.x - np.mean(bunch.x)) ** 2)
        meanYsq[j] = np.mean((bunch.y - np.mean(bunch.y)) ** 2)
        emitX[j] = bunch.epsn_x()
        emitY[j] = bunch.epsn_y()

        plot_turn = 500
        if i == plot_turn:
            plt.plot(bunch.z, bunch.y, 'o', c='C0', label='tracking')

dataExport = [meanX, meanY, meanXsq, meanYsq, emitX, emitY]

plt.plot(initial_sigmas, y_co_cc, c='C1', label='theory')
plt.xlabel('z [m]')
plt.ylabel(r'$\mathrm{y_{CO} \ [m]}$')
plt.grid(linestyle='--')
plt.legend(loc=1)
plt.tight_layout()

plt.savefig(f'CC_kick_tracking_VS_theory_{plot_turn}.png', bbox_inches='tight')

plt.show()

print('--> Done.')

print("Simulation time in seconds: " + str(time.clock() - t0))


