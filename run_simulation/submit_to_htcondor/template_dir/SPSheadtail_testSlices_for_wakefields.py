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
n_turns = int(2000)            #number of cycles to run the simulation for
decTurns = int(1)               #how often to record data

Q_y = 26.18                #How many times the particles oscillate in phase space each turn 
Q_x = 26.13                          # Will need it to be 16.25 IF CC feedback is used
                            # For this version of PyHEADTAIL Q_x should be an array

filename = 'file'      #Where the data for the run is saved

wakefieldOn = 1         # Turns on the wakefields
#wakeContribution = 'step'

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
Qp_x, Qp_y = 0, 0 #10

# detuning coefficients in (1/m)
app_x = 0.0  #2.4705e-15 #4e-11
app_xy = 0.0 #-0*2.25e-11
app_y = 0.0  #-7.31-14 #0*3e-11

# PARAMETERS FOR LONGITUDINAL MAP
# =======================
alpha = 1.9e-3
Q_s = 0.0051
h1, h2 = 4620, 9240
V1, V2 = 5.008e6, 0e6
dphi1, dphi2 = 0, np.pi
p_increment = 0 * e/c * circumference/(beta*c)

# CREATE BEAM
# ===========
macroparticlenumber = int(5e5) #int(4e6) #int(5e5) # at least 5e5 particles are needed in the presence of the wakefields

charge = e
mass = m_p

R = circumference/(2*np.pi)
eta = alpha-1/gamma**2
beta_z = np.abs(eta)*R/Q_s


epsn_x = 2e-6
epsn_y = 2e-6     
#epsn_z    = 2.5
sigma_z   = 0.155 #2*0.0755

sigma_x = np.sqrt(epsn_x/(beta*gamma) * beta_x[0])
sigma_xp = sigma_x/beta_x[0]
sigma_y = np.sqrt(epsn_y/(beta*gamma) * beta_y[0])
sigma_yp = sigma_y/beta_y[0]
sigma_dp = sigma_z/beta_z
epsn_z = 4*np.pi * p0/e * sigma_z*sigma_dp

# generate bunches for the different intensities
intensity_list = np.linspace(0, 5e10, 5)
for intensity in intensity_list[1:]:
    print(f'Createing bunch for intensity:{intensity}')

    bunch = generate_Gaussian6DTwiss(
        macroparticlenumber, intensity, charge, mass, circumference, gamma,
        alpha_x[0], alpha_y[0], beta_x[0], beta_y[0], beta_z, epsn_x, epsn_y, epsn_z)
    xoffset = 0.25*sigma_x #5e-4
    yoffset = 0.25*sigma_y #0.5*sigma_y #5e-4
    
    bunch.x += xoffset
    bunch.y += yoffset
    
    afile = open(f'bunch_intensity{intensity/1e10}1e10', 'wb')
    pickle.dump(bunch, afile)
    afile.close()



# SLICER FOR WAKEFIELDS
# ============
n_slices = %slices  # 500
slicer_for_wakefields = UniformBinSlicer(n_slices, z_cuts=(-3.*sigma_z, 3.*sigma_z))#,circumference=circumference, h_bunch=h1)

# WAKEFIELD
# ==========
n_turns_wake = 1 # for the moment we consider that the wakefield decays after 1 turn

#wakefile1=('SPS_complete_wake_model_2018_Q26.txt')
wakefile1=('/afs/cern.ch/work/n/natriant/private/pyheadtail_example_crabcavity/wakefields/SPS_complete_wake_model_2018_Q26.txt')

ww1 = WakeTable(wakefile1, ['time', 'dipole_x', 'dipole_y', 'quadrupole_x', 'quadrupole_y'], n_turns_wake=n_turns_wake)

# only dipolar kick
my_length = len(ww1.wake_table['quadrupole_x'])
ww1.wake_table['quadrupole_x'] = np.zeros(my_length)
ww1.wake_table['quadrupole_y'] = np.zeros(my_length)



wake_field = WakeField(slicer_for_wakefields, ww1)#, beta_x=beta_x, beta_y=beta_y)

# 4) Create transverese and longitudinal map
scale_factor = 2*bunch.p0  # scale the detuning coefficients in pyheadtail units
transverse_map = TransverseMap(s, alpha_x, beta_x, D_x, alpha_y, beta_y, D_y, Q_x, Q_y,
    [Chromaticity(Qp_x, Qp_y),
    AmplitudeDetuning(app_x*scale_factor, app_y*scale_factor, app_xy*scale_factor)])

longitudinal_map = LinearMap([alpha], circumference, Q_s)

# 5) Set up the accelerator
t0 = time.clock()

one_turn_map = []
for i, segment in enumerate(transverse_map):
    one_turn_map.append(segment)
    if wakefieldOn:
        if i+1 == i_wake:
            one_turn_map.append(wake_field)
            #one_turn_map.append(wake_field_wall)
one_turn_map.append(longitudinal_map)

n_damped_turns = int(n_turns/decTurns) # The total number of turns at which the data are damped.
                       # We want this number as an integer, so it can be used in the next functions. 

# 6) Start tracking
meanX = np.zeros(n_damped_turns)
meanY = np.zeros(n_damped_turns)
#meanXsq = np.zeros(n_damped_turns)
#meanYsq = np.zeros(n_damped_turns)
#emitX = np.zeros(n_damped_turns)
#emitY = np.zeros(n_damped_turns)
    
# Iterate over the intensities, reload bunch object from file for each intensisty
#intensity_list = np.linspace(0, 5e10, 5)
for intensity in intensity_list[1:]:
    print(f'Tracking for intensity:{intensity}')
    
    file2 = open(f'bunch_intensity{intensity/1e10}1e10', 'rb')
    bunch = pickle.load(file2)
    file2.close()
    for i in range(n_turns):
        #These next two lines actually "run" the simulation - the computationally heavy part
        for m in one_turn_map:
            m.track(bunch)
            
      
        if i%decTurns is  0:
            j = int(i/decTurns)
            meanX[j] = np.mean(bunch.x)
            meanY[j] = np.mean(bunch.y)
            #meanXsq[j] = np.mean((bunch.x-np.mean(bunch.x))**2)
            #meanYsq[j] = np.mean((bunch.y-np.mean(bunch.y))**2)
            #emitX[j] = bunch.epsn_x()
            #emitY[j] = bunch.epsn_y()


    dataExport = [meanX, meanY]#, meanXsq, meanYsq, emitX, emitY]

    f = open(filename+f'_intensity{intensity/1e10:.2f}e10_ayy{app_y}_QpyQpx{Qp_x}.txt', 'w')

    with f:
        out = csv.writer(f, delimiter=',')
        out.writerows(zip(*dataExport))

    print('--> Done.')

    print("Simulation time in seconds: " + str(time.clock() - t0))

# 7) Compute tunes
Qx_list, Qy_list = [], []
# Add tunes for intensity 0, there should be no tune shift
Qx_list.insert(0, 0.13)
Qy_list.insert(0, 0.18)

for intensity in intensity_list[1:]:
    # Load the file with all of the saved data from the run
    meanX, meanY = np.loadtxt(filename+f'_intensity{intensity/1e10:.2f}e10_ayy{app_y}_QpyQpx{Qp_x}.txt', delimiter = ",", unpack = True)
    
    Qx_list.append(pnf.get_tune(meanX))
    Qy_list.append(pnf.get_tune(meanY))

Qy_coherent = {}
Qx_coherent = {}
for i, intensity in enumerate(intensity_list):
    Qy_coherent[f'intensity {intensity}'] = Qy_list[i]
    Qx_coherent[f'intensity {intensity}'] = Qx_list[i]
    
save2pickle = True
if save2pickle:
    with open(f'Qy_coherent_vs_Intensity_6D_ayy{app_y}_wakesON_QpyQpx{Qp_x}_{wakeContribution}.pkl', 'wb') as ff:
        pickle.dump(Qy_coherent, ff, pickle.HIGHEST_PROTOCOL)
    ff.close()
    
if save2pickle:
    with open(f'Qx_coherent_vs_Intensity_6D_ayy{app_y}_wakesON_QpyQpx{Qp_x}_{wakeContribution}.pkl', 'wb') as ff:
        pickle.dump(Qx_coherent, ff, pickle.HIGHEST_PROTOCOL)
    ff.close()
