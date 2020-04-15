import simulation_parameters as pp
from scipy.constants import e
import numpy as np
import pickle
from PyHEADTAIL.particles.generators import generate_Gaussian6DTwiss


sigma_x = np.sqrt(pp.epsn_x / (pp.beta * pp.gamma) * pp.beta_x[0])
sigma_xp = sigma_x / pp.beta_x[0]
sigma_y = np.sqrt(pp.epsn_y / (pp.beta * pp.gamma) * pp.beta_y[0])
sigma_yp = sigma_y / pp.beta_y[0]
sigma_dp = pp.sigma_z / pp.beta_z
epsn_z = 4 * np.pi*pp.p0 / e * pp.sigma_z * sigma_dp

bunch = generate_Gaussian6DTwiss(
    pp.macroparticlenumber, pp.intensity, pp.charge, pp.mass, pp.circumference, pp.gamma,
    pp.alpha_x[0], pp.alpha_y[0], pp.beta_x[0], pp.beta_y[0], pp.beta_z, pp.epsn_x, pp.epsn_y, pp.epsn_z)

bunch.x += pp.xoffset
bunch.y += pp.yoffset
bunch.z = np.abs(bunch.z)
bunch.dp = np.zeros(pp.macroparticlenumber)

afile = open('input/bunch_3', 'wb')
pickle.dump(bunch, afile)
afile.close()
