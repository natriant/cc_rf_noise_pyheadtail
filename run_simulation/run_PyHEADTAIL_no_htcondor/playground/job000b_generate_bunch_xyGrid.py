import simulation_parameters as pp
import matplotlib.pyplot as plt
import numpy as np
import pickle
from PyHEADTAIL.particles.generators import generate_Gaussian6DTwiss

# Compute the geometric emittances
ex_geom = pp.epsn_x/(pp.beta*pp.gamma)
ey_geom = pp.epsn_y/(pp.beta*pp.gamma)
print('ex_geom={} m, ey_geom={} m'.format(ex_geom, ey_geom))

# Compute the sigmas (rms beam size)
sigma_x = np.sqrt(ex_geom*pp.beta_x[0])
sigma_y = np.sqrt(ey_geom*pp.beta_y[0])
print('sigma_x={} m, sigma_y = {} m'.format(sigma_x, sigma_y))

steps = int(np.sqrt(pp.macroparticlenumber))

# Define the upper limit of the distribution
xmax, ymax = 3*sigma_x, 3*sigma_y

# Change to action phase variables just to create the linspace
Jxmin, Jymin = 10**(-13), 10**(-13) # for zero you cannot calculate the tune
Jxmax, Jymax = xmax**2/(2*pp.beta_x[0]), ymax**2/(2*pp.beta_y[0])

Jx = np.linspace(10**(-13), Jxmax, steps)
Jy = np.linspace(10**(-13), Jymax, steps)

# Return to x-y
x = np.sqrt(Jx*2*pp.beta_x[0])
y = np.sqrt(Jy*2*pp.beta_y[0])

# meshgrid
xx, yy = np.meshgrid(x, y)

# Generate a bunch just to create the object.
# Then redefine the coordinates according to the distribution you want.
bunch = generate_Gaussian6DTwiss(
    pp.macroparticlenumber, pp.intensity, pp.charge, pp.mass, pp.circumference, pp.gamma,
    pp.alpha_x, pp.alpha_y, pp.beta_x, pp.beta_y, pp.beta_z, pp.epsn_x, pp.epsn_y, pp.epsn_z)

# Now redefine the coordinates.
bunch.x = xx.flatten()
bunch.y = yy.flatten()

bunch.xp = 0 * np.ones(pp.macroparticlenumber)
bunch.yp = 0 * np.ones(pp.macroparticlenumber)
bunch.z = 0 * np.ones(pp.macroparticlenumber)
bunch.dp = 0 * np.ones(pp.macroparticlenumber)

afile = open('./input/bunch', 'wb')
pickle.dump(bunch, afile)
afile.close()

plt.plot(bunch.x, bunch.y, '.')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('{} particles'.format(pp.macroparticlenumber))
plt.tight_layout()
savefig = False
if savefig:
    plt.savefig('initial_distribution_pyheadtail_6sigma.png')
plt.show()