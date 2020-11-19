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
epsn_z = 2.5

bunch = generate_Gaussian6DTwiss(
    pp.macroparticlenumber, pp.intensity, pp.charge, pp.mass, pp.circumference, pp.gamma,
    pp.alpha_x[0], pp.alpha_y[0], pp.beta_x[0], pp.beta_y[0], pp.beta_z, pp.epsn_x, pp.epsn_y, epsn_z)

bunch.x += pp.xoffset
bunch.y += pp.yoffset

print('rms x = {} m'.format(np.std(bunch.x)))
print('rms y = {} m'.format(np.std(bunch.y)))
print('rms z = {} m'.format(np.std(bunch.z)))
print('rms delta = {} m'.format(np.std(bunch.dp)))

afile = open('input/bunch', 'wb')
pickle.dump(bunch, afile)
afile.close()
