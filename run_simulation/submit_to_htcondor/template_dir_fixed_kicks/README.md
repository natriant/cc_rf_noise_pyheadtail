1) job000 --> creates a fixed list of kicks for each turn over a given number of runs. The kicks are saved in apickle file. This file is loaded in the script that actually runs the simulation (). The purpose of this is to help us study the effect of noise for different parameters without being affected from the random factor of the noise. ATTENTION! Run it only once for a specific case of studies.
2) job001 --> create the directories required for PyHEADTAIL simulations. Also it adds in the scripts that run the simulations the run number so they load the corresponding list of kicks.
3) job002 --> prepatre directories for runnins the simulations scanning a range of detuning strength values. ayy is chosen here as we apply the kick on the vertical plane. Note that axy chould have been choosen as well but ayy is better option. 