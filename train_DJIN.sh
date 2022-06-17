#!/usr/bin/sh

##PBS directives
#PBS -N train_DJIN_id100
#PBS -l nodes=1:ppn=12:gpus=1 (number of nodes / cores / gpus)
#PBS -l walltime=72:00:00 (maximum time the job should run, default 1 hour)
#PBS -M lmartin9@trinity.edu
#PBS -m abe

##commands
## parse data
sh /data/zhanglab/lmartin9/AgingHealth/Data_Parser/create_elsa_data.sh 
## calculate averages / stds
python3 /data/zhanglab/lmartin9/AgingHealth/population_average.py
python3 /data/zhanglab/lmartin9/AgingHealth/population_std.py
## train model
python3 /data/zhanglab/lmartin9/AgingHealth/train.py --job_id=100
## predictions
python3 /data/zhanglab/lmartin9/AgingHealth/predict.py --job_id=100 --epoch=1999
## comparison model
