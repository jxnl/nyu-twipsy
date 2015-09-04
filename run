#!/bin/bash

#PBS -l nodes=1:ppn=12
#PBS -l walltime=4:00:00
#PBS -l mem=128GB
#PBS -N twipsy-rd 
#PBS -M jl7500@nyu.edu
#PBS -m abe

module purge
module load python3
module load scikit-learn 
module load scipy
module load pandas

SRCDIR=$HOME/Twispy/

RUNDIR=$SCRATCH/Twsipy/run-${PBS_JOBID/.*}
mkdir -p $RUNDIR

# python /home/jl7500/Twipsy/gridsearch_linearsvc.py
# python /home/jl7500/Twipsy/gridsearch_logisticregression.py
python /home/jl7500/Twipsy/gridsearch_rf.py
# python /home/jl7500/Twipsy/gridsearch_svc.py
