#!/bin/bash
echo "uname -r:" `uname -r`

DIR=/afs/cern.ch/work/n/natriant/private/pyheadtail_example_crabcavity

export MYPYTHON=~/miniconda2

source $MYPYTHON/bin/activate ""
export PATH=$MYPYTHON/bin:$PATH

echo "which python:" `which python`


date
#python $DIR/%index1/SPSheadtail_CC_noise.py
python $DIR/%index1/SPSheadtail_CC_noise_randomSeed.py
date
