import os
import shutil
import numpy as np
import job000_create_fixed_noise_kick as pp
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





noise_types =['AN', 'PN']

# scan ayy
ayy_min, ayy_max, num = 1e-15, 1e-11, 15
ayy_scan = np.linspace(ayy_min, ayy_max, num)


for index, ayy in enumerate(ayy_scan):
    for noise_type in noise_types:
        for i in range(pp.runs):
            j = i+1
            # Create the necessary directories
            os.mkdir('../{}_mytest{}_ayy{}'.format(noise_type, j, index)) # the index of ayy is included in the name of the directory
            os.mkdir('../{}_mytest{}_ayy{}/error'.format(noise_type, j, index))
            os.mkdir('../{}_mytest{}_ayy{}/output'.format(noise_type, j, index))
            os.mkdir('../{}_mytest{}_ayy{}/log'.format(noise_type, j, index))
        
        
            if noise_type == 'AN':
                source = './templates/SPSheadtail_fixed_AmplitudeNoise_kick_ayyScan.py'
                destination = '../AN_mytest{}_ayy{}/SPSheadtail_CC_new_version_python.py'.format(j, index)
                shutil.copyfile(source, destination)
                replace(destination, '%run_number', '{}'.format(i)) # replace the number of run in the script, i cause the indices of the list start from 0
                replace(destination, '%ayy', '{}'.format(ayy)) # replace the detuning strength     
           
 
            if noise_type == 'PN':
                source = './templates/SPSheadtail_fixed_PhaseNoise_kick_ayyScan.py'
                destination = '../PN_mytest{}_ayy{}/SPSheadtail_CC_new_version_python.py'.format(j, index)
                shutil.copyfile(source, destination)
                replace(destination, '%run_number', '{}'.format(i)) # replace the number of run in the script, i cause the indices of the list start from 0
                replace(destination, '%ayy', '{}'.format(ayy)) # replace the detuning strength
    
    

