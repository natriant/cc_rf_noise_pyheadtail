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

from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.particles.generators import generate_Gaussian6DTwiss 
from PyHEADTAIL.trackers.detuners import Chromaticity, AmplitudeDetuning
from PyHEADTAIL.trackers.transverse_tracking import TransverseMap
from PyHEADTAIL.trackers.simple_long_tracking import RFSystems, LinearMap, Drift
from PyHEADTAIL.monitors.monitors import BunchMonitor, SliceMonitor, ParticleMonitor
from PyHEADTAIL.feedback.transverse_damper import TransverseDamper
from PyHEADTAIL.impedances.wakes import CircularResonator, WakeTable, WakeField

#%matplotlib inline

###### Plotting parameters #####
params = {'legend.fontsize': 20,
          'figure.figsize': (8, 7),
          'axes.labelsize': 25,
          'axes.titlesize': 21,
          'xtick.labelsize': 23,
          'ytick.labelsize': 23,
          'image.cmap': 'jet',
          'lines.linewidth': 2,
          'lines.markersize': 7,
          'font.family': 'sans-serif'}


plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.rcParams.update(params)

###### Useful functions #######
def get_values_per_slice(particle_coordinates, parts_id):
    value = lambda index: particle_coordinates[index]
    values_per_slice = lambda slice_indices: list(map(value, slice_indices))
    return list(map(values_per_slice, parts_id))

def mean_coordiantes_per_slice(bunch_coord, parts_id):
    coordinate_per_slice = np.array(get_values_per_slice(bunch_coord, parts_id))
    mean_coord = []
    for i in range(len(coordinate_per_slice)):
        mean_temp = np.mean(coordinate_per_slice[i])
        mean_coord.append(mean_temp)
    return mean_coord

###### Variables we change ######

#==========================================================
#               Variables We Change
#==========================================================
n_turns = int(1)            #number of cycles to run the simulation for

Q_y = 26.18                #How many times the particles oscillate in phase space each turn
Q_x = 26.13                          # Will need it to be 16.25 IF CC feedback is used
                            # For this version of PyHEADTAIL Q_x should be an array

filename = 'file'      #Where the data for the run is saved

wakefieldOn = 1         # Turns on the wakefields


##==========================================================
#           Variables We (Usually) Do Not Change
#==========================================================
gamma = 287.8
p0 = m_p*c*np.sqrt(gamma**2 - 1)
beta = np.sqrt(1 - 1/gamma**2)
circumference = 6911.5623
frev = 299792458/circumference

### Transverse paraamters
### Needed for the createion of the  beam not for the tracking. 
### Heren_segments = 1, no tracking in the longitudinal plane
n_segments = 1
beta_x = 42.0941 * np.ones(n_segments) #### (for Q26)
beta_y = 42.0137 * np.ones(n_segments) #### (for Q26)
alpha_x, alpha_y = 0* np.ones(n_segments), 0* np.ones(n_segments)

# PARAMETERS FOR LONGITUDINAL MAP
# =======================
alpha = [1.9e-3]
Q_s = 0.0051
h1, h2 = 4620, 9240
V1, V2 = 5.008e6, 0e6
dphi1, dphi2 = 0, np.pi
p_increment = 0 * e/c * circumference/(beta*c)


L = 10000 # [m] length of drift space

one_turn_map = [Drift(alpha, L)] # longitudinal map only

# CREATE BEAM
# ===========
macroparticlenumber = int(1e6) # at least 5e5 particles are needed in the presence of the wakefields
charge = e
mass = m_p

R = circumference/(2*np.pi)
eta = np.array(alpha)-1/gamma**2
beta_z = np.abs(eta)*R/Q_s


epsn_x = 2e-6
epsn_y = 2e-6
#epsn_z    = 2.5
#sigma_z   = 0.155 #2*0.0755

tau = 1.7e-9 # 4 sigma_t [s]
sigma_z = c*tau/4 #0.155  # m
#sigma_z = 0.#27
#sigma_z = %sigmaz #0.155
print(f'sigma_z= {sigma_z} [m]')


sigma_x = np.sqrt(epsn_x/(beta*gamma) * beta_x[0])
sigma_xp = sigma_x/beta_x[0]
sigma_y = np.sqrt(epsn_y/(beta*gamma) * beta_y[0])
sigma_yp = sigma_y/beta_y[0]
sigma_dp = sigma_z/beta_z
epsn_z = 4*np.pi * p0/e * sigma_z*sigma_dp

intensity = 3.0e11

bunch = generate_Gaussian6DTwiss(macroparticlenumber, intensity, charge, mass, circumference, gamma, alpha_x[0], alpha_y[0], beta_x[0], beta_y[0], beta_z, epsn_x, epsn_y, epsn_z)

xoffset = 0.0 #0.15*sigma_x #5e-4
yoffset = 0.25*sigma_y #0.5*sigma_y #5e-4

bunch.x += xoffset
bunch.y += yoffset

afile = open(f'bunch_4_drift_example', 'wb')
pickle.dump(bunch, afile)
afile.close()

# SLICER FOR WAKEFIELDS
# ============
n_slices = 500
n_sigma = 3.
slicer_for_wakefields = UniformBinSlicer(n_slices, z_cuts=(-n_sigma*sigma_z, n_sigma*sigma_z))#,circumference=circumference, h_bunch=h1)

# WAKEFIELD
# ==========
n_turns_wake = 1 # for the moment we consider that the wakefield decays after 1 turn
wakefile1=('SPS_complete_wake_model_2018_Q26.txt')


ww1 = WakeTable(wakefile1, ['time', 'dipole_x', 'dipole_y', 'quadrupole_x', 'quadrupole_y'], n_turns_wake=n_turns_wake)

# only dipolar kick
my_length = len(ww1.wake_table['quadrupole_x'])
ww1.wake_table['quadrupole_x'] = np.zeros(my_length)
ww1.wake_table['quadrupole_y'] = np.zeros(my_length)
ww1.wake_table['dipole_x'] = np.zeros(my_length)

#ww1.wake_table['dipole_y'] = ww1.wake_table['dipole_y']*100


wake_field = WakeField(slicer_for_wakefields, ww1)#, beta_x=beta_x, beta_y=beta_y)

one_turn_map.append(wake_field)


# Open bunch file
file2 = open(f'bunch_4_drift_example', 'rb')
bunch = pickle.load(file2)
file2.close()


print(f'Start tracking')
#t0 = time.clock()
for i in range(n_turns):
    for m in one_turn_map:
        m.track(bunch)
        X, XP = bunch.x, bunch.xp
        Y, YP = bunch.y, bunch.yp
        Z, DP = bunch.z, bunch.dp



dataExport = [X, XP, Y, YP, Z, DP]

f = open(filename+f'_test_drift.txt', 'w')

with f:
    out = csv.writer(f, delimiter=',')
    out.writerows(zip(*dataExport))

print('--> Done.')

#print("Simulation time in seconds: " + str(time.clock() - t0))


file2 = open(f'bunch_4_drift_example', 'rb')
bunch = pickle.load(file2)
file2.close()

plt.scatter(Z, YP-bunch.yp)


#plt.scatter(bunch.z, bunch.yp)

plt.ylim(-2e-7, 2e-7)

plt.show()

plt.close()

####  Simulation end ###

############ Theory ##########
# convert wake table data to SI
convert_to_s = 1e-9
convert_to_V_per_Cm = 1e15 

time = convert_to_s * ww1.wake_table['time'] # [s]
wake_strength =  -convert_to_V_per_Cm*ww1.wake_table['dipole_y'] # dz = beta*c*dt, dz<0 for ultrarelativistic 

# The wake(dt) uses the scipy.interpolate.interp1d linear interpolation to calculate the wake strength at an arbitrary
# value of dt (provided it is in the valid range). The valid range of dt is given by the time range from the wake table.

# convert time to z
z_wakes = time*beta*c # [m]

'''
z_cut_head = my_sliceSet.z_cut_head
z_cut_tail = my_sliceSet.z_cut_tail
print(z_cut_head, z_cut_tail)
z_boundary = 2*z_cut_head


# discard all the values of the wake for > bunch length. > 2*np.max(z_centers)
z_wakes_boundaries = [x for x in z_wakes if x <=2*np.max(z_boundary)]
print(len(z_wakes_boundaries))




wake_strength = wake_strength[:len(z_wakes_boundaries)]
'''

####  Simulation end ###3
my_sliceSet = bunch.get_slices(slicer_for_wakefields, statistics=['mean_x', 'mean_y']) # type; PyHEADTAIL.particles.slicing.SliceSet
z_centers = my_sliceSet.z_centers


print(len(wake_strength))


# interpolate to the z of the bunch distribution
#wake_interpolated = np.interp(z_centers, z_wakes_boundaries, wake_strength)

wake_interpolated = np.interp(z_centers, z_wakes, wake_strength)

print(wake_interpolated)
plt.plot(z_centers, wake_interpolated)
plt.xlabel('z centers')
plt.ylabel(wake_interpolated)
plt.show()
plt.close()
#plt.plot(z_wakes, wake_strength)

# create a list with the mean of bunch y for all slices

parts_id = []
for my_slice in np.arange(my_sliceSet.n_slices):
    parts_id.append(my_sliceSet.particle_indices_of_slice(slice_index=my_slice))




# convolution

#conv_dipole_y = []
#for my_slice in np.arange(my_sliceSet.n_slices):
#    mp_per_slice = my_sliceSet.n_macroparticles_per_slice[my_slice]
#    conv_dipole_temp = np.convolve(mp_per_slice*mean_y[my_slice], wake_interpolated[::-1], 'valid')
#    conv_dipole_y.append(conv_dipole_temp)

#moments_list = [s.n_macroparticles_per_slice*s.mean_y for s in my_sliceSet]
#print(moments_list)

#quit()
conv_dipole_y = np.convolve(my_sliceSet.n_macroparticles_per_slice*my_sliceSet.mean_y, wake_interpolated[::-1], 'same')
#len(conv_dipole_y)


wake_kicks = -charge**2/(mass*gamma*(beta**2)*(c**2))*np.array(conv_dipole_y)
#len(wake_kicks)

file2 = open(f'bunch_4_drift_example', 'rb')
bunch = pickle.load(file2)
file2.close()

plt.scatter(Z, YP-bunch.yp, label='simulation')

plt.plot(z_centers, 3*wake_kicks*1e5, c='r', label='theory')
plt.xlabel('z [m]')
plt.ylabel(r'$\Delta yp \ [rad]$')
plt.tight_layout()

plt.ylim(-2e-7, 2e-7)
plt.grid()
plt.legend()
#plt.show()
#plt.close()

plt.savefig('dipole_kick.png', bbox_inches='tight')


