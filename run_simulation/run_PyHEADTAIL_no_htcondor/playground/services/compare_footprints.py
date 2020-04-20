import matplotlib.pyplot as plt
import matplotlib
import numpy as np
plt.style.use('natalia')

Qx_list1, Qy_list1 = np.loadtxt('../output/mytunes_nominalSPS_QpxQPy0.txt', delimiter=",", unpack=True)
Qx_list2, Qy_list2 = np.loadtxt('../output/mytunes_nominalSPS_QpxQPy2.txt', delimiter=",", unpack=True)


f, ax = plt.subplots(figsize=(14,14))
ax.scatter(Qx_list2, Qy_list2, c='C1', label='Qpx=Qpy=2')
ax.scatter(Qx_list1, Qy_list1, c='C0', label='Qpx=Qpy=0')
ax.set_ylim(0.17993, 0.18007)
ax.set_xlim(0.12993, 0.13007)
# neglect the offset in the axis
y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
ax.yaxis.set_major_formatter(y_formatter)
x_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
ax.xaxis.set_major_formatter(y_formatter)
plt.xlabel('Qx')
plt.ylabel('Qy')
plt.grid()
plt.legend()
plt.tight_layout()
#plt.savefig('gaussian_footprint_nominalSPStunespread_chroma.png')
plt.show()