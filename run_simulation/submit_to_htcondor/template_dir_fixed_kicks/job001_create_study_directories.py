import os
import shutil
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

for noise_type in noise_types:
    for i in range(pp.runs):
        j = i+1
        # Create the necessary directories
        os.mkdir('../{}_mytest{}'.format(noise_type, j))
        os.mkdir('../{}_mytest{}/error'.format(noise_type, j))
        os.mkdir('../{}_mytest{}/output'.format(noise_type, j))
        os.mkdir('../{}_mytest{}/log'.format(noise_type, j))
        
        
        if noise_type == 'AN':
            source = './templates/SPSheadtail_fixed_AmplitudeNoise_kick.py'
            destination = '../AN_mytest{}/SPSheadtail_CC_new_version_python.py'.format(j)
            shutil.copyfile(source, destination)
            replace(destination, '%run_number', '{}'.format(i)) # i cause the indeces of the list start from 0

 
 
        if noise_type == 'PN':
            source = './templates/SPSheadtail_fixed_PhaseNoise_kick.py'
            destination = '../PN_mytest{}/SPSheadtail_CC_new_version_python.py'.format(j)
            shutil.copyfile(source, destination)
            replace(destination, '%run_number', '{}'.format(i)) # i cause the indeces of the list start from 0

    
    

