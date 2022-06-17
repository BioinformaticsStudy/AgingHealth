#!/usr/bin/sh

##PBS directives
#PBS -N train_DJIN_id100
#PBS -l nodes ... (number of nodes / cores / gpus)
#PBS -l walltime=72:00:00 (maximum time the job should run, default 1 hour)
#PBS -M email@address.com (for updates)
#PBS -m abe

##commands

## parse data
## calculate averages / stds
## train model
## predictions
## comparison model
