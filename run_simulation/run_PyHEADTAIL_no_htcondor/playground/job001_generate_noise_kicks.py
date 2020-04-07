import simulation_parameters as pp
import numpy as np
import pickle

if pp.ampNoiseOn:
    ampKicks = (np.random.normal(0, pp.stdAmpNoise, pp.n_turns))
    bfile = open('ampKicks', 'wb')
    pickle.dump(ampKicks, bfile)
    bfile.close()

if pp.phaseNoiseOn:
    phaseKicks = (np.random.normal(0, pp.stdPhaseNoise, pp.n_turns))
    bfile = open('phaseKicks', 'wb')
    pickle.dump(phaseKicks, bfile)
    bfile.close()


