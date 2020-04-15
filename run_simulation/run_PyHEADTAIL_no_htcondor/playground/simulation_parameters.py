import numpy as np
from scipy.constants import m_p, c, e

n_turns = int(1e4)
decTurns = int(1)  # how often to record data

Q_x = 26.13
Q_y = 26.18

gamma = 287.8
p0 = m_p * c * np.sqrt(gamma ** 2 - 1)
beta = np.sqrt(1 - 1 / gamma ** 2)
circumference = 6911.5623
frev = 299792458 / circumference

# PARAMETERS FOR TRANSVERSE MAP
# =====================
n_segments = 1
s = np.arange(0, n_segments + 1)*circumference / n_segments
alpha_x = -0.8757651182*np.ones(n_segments)
beta_x = 29.23 * np.ones(n_segments)
D_x = 0 * np.ones(n_segments)
alpha_y = 1.898525134 * np.ones(n_segments)
beta_y = 76.07315729*np.ones(n_segments)
D_y = 0 * np.ones(n_segments)

Qp_x = 2.
Qp_y = 2.

app_x = 179.3585107  # -1.1e-11  # 2*-6.1333e-13#179.3610966
app_xy = -441.3397664
app_y = -30.78659311


# PARAMETERS FOR LONGITUDINAL MAP
# =======================
alpha = 1.9e-3
Q_s = 0.0035
h1, h2 = 4620, 9240
V1, V2 = 4.5e6, 0e6
dphi1, dphi2 = 0, np.pi
p_increment = 0 * e / c * circumference / (beta * c)

# BEAM PARAMETERS
# ===========
macroparticlenumber = 900 #10000

charge = e
mass = m_p
intensity = 1.5e11

R = circumference / (2 * np.pi)
eta = alpha - 1 / gamma ** 2
beta_z = np.abs(eta) * R / Q_s
epsn_x = 2e-6  # m
epsn_y = 2e-6
epsn_z = 2.5
sigma_z = 0.155  # 2*0.0755

xoffset = 0e-4
yoffset = 0e-4

# Noise parameters

ampNoiseOn = True
phaseNoiseOn = False
stdAmpNoise = 1e-8
stdPhaseNoise = 1e-8
