import numpy as np
import pickle

turns = int(1e5)
runs = int(30)
stdNoise = 1e-8

noise_list = []

for run in range(runs):
	noise_list.append(np.random.normal(0, stdNoise, turns))

with open('my_noise_kicks.pkl', 'wb') as f:
	pickle.dump(noise_list, f) 
