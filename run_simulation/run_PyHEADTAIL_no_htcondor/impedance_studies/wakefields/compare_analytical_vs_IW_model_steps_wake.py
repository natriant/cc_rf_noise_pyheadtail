import matplotlib.pyplot as plt
from PyHEADTAIL.impedances.wakes import WakeTable

# Plotting parameters
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


# Import wakefields
wakefile1 = 'step_analytical_wake.txt'
wakefile2 = 'SPS_wake_model_with_steps_2018_Q26.txt'

ww1 = WakeTable(wakefile1, ['time', 'dipole_y'], n_turns_wake=1)
ww2 = WakeTable(wakefile2, ['time', 'dipole_x', 'dipole_y', 'quadrupole_x', 'quadrupole_y'], n_turns_wake=1)


fig, ax = plt.subplots()
ax.plot(ww1.wake_table['time'], ww1.wake_table['dipole_y'], '-o', label='analytical step wake')
ax.plot(ww2.wake_table['time'], ww2.wake_table['dipole_y'], '-o', label='SPS IW model, step')


# styling
ax.set_xlabel('Waketime [ns]')
ax.set_ylabel('DipolY [V/pC/mm]')
ax.legend(loc=1)
ax.grid(ls='--')


plt.tight_layout()
savefig = True 
if savefig:
    plt.savefig('dipoleY_steps_analytical_vs_SPS_IW_model_Q26.png', bbox_inches='tight') 
plt.show()



