import numpy as np
import matplotlib.pyplot as plt



gamma = 287.8
# Crab cavity
Vcc = 1e6
p_cc = Vcc / (gamma * .938e9)  # Vo/Eb
print(p_cc)
cc_voltage = lambda turn: np.interp(turn, [0, 200, 1e12], p_cc * np.array([0, 1, 1]))

for i in range(1000):
    plt.plot(i, cc_voltage(i), '.')
#plt.plot(cc_voltage)
plt.show()