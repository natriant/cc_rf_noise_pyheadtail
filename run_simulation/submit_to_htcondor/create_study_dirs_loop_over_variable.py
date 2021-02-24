import numpy as np
import shutil 
from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove

def replace(file_path, pattern, subst):
    #Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    #Copy the file permissions from the old file to the new file
    copymode(file_path, abs_path)
    #Remove original file
    remove(file_path)
    #Move new file
    move(abs_path, file_path)


# Set the limits for the variables you want to iterate over
my_start = -2e4  
my_stop = 2.01e4 
my_step = 2000.0

my_variables = list(np.arange(my_start, my_stop, my_step))
#my_variables = list(np.linspace(0.0, 5.0e10, 25))
print(my_variables)

n_sets = 20 # How many different sets of noise kicks.
n_runs = 3 # How many times the simulation is repeated for each set of noise kciks.

# Create the study direcotries
src = './template_dir' # source directory
dest = './' # destinationdirectory
dir_name = 'SPS_emit_270GeV_PN1e-8_WakesON_resonCircON_QpxQpy1_6D_Nb5e5_ayy'

# choose script from source directory
script_name = 'SPSheadtail_CC_noise_randomSeed_circularResonator.py'

for my_set in range(n_sets):
    for run in range(n_runs):
    
        for index, var in enumerate(my_variables):
            # copy the entire directory
            try:
                destination = shutil.copytree(src, dest+dir_name+f'{var}_fixedKicksSet{my_set}_run{run}', copy_function = shutil.copy)
                # replace the variable value in each file
                replace(dest+dir_name+f'{var}_fixedKicksSet{my_set}_run{run}/{script_name}', '%ayy', f'{var}')
	
                replace(dest+dir_name+f'{var}_fixedKicksSet{my_set}_run{run}/{script_name}', '%seed_step', f'{10+my_set}')
            except OSError as err:
                print(f'OS error: {err}')

