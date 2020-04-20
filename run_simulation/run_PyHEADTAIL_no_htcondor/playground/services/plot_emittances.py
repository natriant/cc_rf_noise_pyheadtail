import os
import matplotlib.pyplot as plt
import numpy as np

# iterate over all the files in directory
#entries = os.listdir('output/')

# selected files
entries = ['ayy_noAN', 'ayy0_axx_AN']

linewidths = [4, 1]
colors = ['C0', 'C1']
linestyles = ['-', '--']
for index, file in enumerate(entries):
    meanX, meanY, emitX, emitY = np.loadtxt('output/'+file+'_emit.txt', delimiter=",", unpack=True)
    plt.plot(np.array(emitY)*1e6, color=colors[index], linestyle = linestyles[index], linewidth=linewidths[index], label=file)

plt.xlabel('turns')
plt.ylabel('ex,y [um]')
plt.tight_layout()
plt.grid()
plt.legend()
#plt.savefig('eyx_An_ayy_axx_notzeroStudies.png')
plt.show()