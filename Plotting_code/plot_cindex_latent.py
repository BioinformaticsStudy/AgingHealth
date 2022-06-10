# plots time-dependent c-index of multiple latent space models depending on their N
# latent models should be trained with train_full_multiple.py
from pathlib import Path
import sys
file = Path(__file__). resolve()  
package_root_directory = file.parents [1]  
sys.path.append(str(package_root_directory))  

import argparse
import torch
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Utils.cindex import cindex_td
from Alternate_models.dataset_dim import Dataset
from torch.utils.data import DataLoader
from DataLoader.collate import custom_collate


parser = argparse.ArgumentParser('Cindex_Latent')
parser.add_argument('--job_id',type=int)
parser.add_argument('--epoch',type=int,default=1999)
parser.add_argument('--start', type=int,default=2,help='lowest N')
parser.add_argument('--step', type=int, default=5,help='difference in N between models')
parser.add_argument('--stop', type=int,default=35,help='highest N')
parser.add_argument('--dataset', type=str, default='elsa',choices=['elsa','sample'],help='what dataset was used to train the models')
args = parser.parse_args()

postfix = '_sample' if args.dataset=='sample' else ''
test_name = f'../Data/test{postfix}.csv'

Ns = list(np.arange(args.start,args.stop,args.step)) + [args.stop]
results = pd.DataFrame(index=Ns,columns=['C-index'])

for N in Ns:
    survival = np.load('../Analysis_Data/Survival_trajectories_job_id%d_epoch%d_latent%d%s.npy'%(args.job_id,args.epoch,N,postfix))    

    pop_avg = np.load(f'../Data/Population_averages{N}{postfix}.npy')
    pop_avg_env = np.load(f'../Data/Population_averages_env{N}{postfix}.npy')
    pop_std = np.load(f'../Data/Population_std{N}{postfix}.npy')
    pop_avg_ = torch.from_numpy(pop_avg[...,1:]).float()
    pop_avg_env = torch.from_numpy(pop_avg_env).float()
    pop_std = torch.from_numpy(pop_std[...,1:]).float()

    min_count = N // 3
    if min_count < 1:
        prune=False
    test_set = Dataset(test_name, N,  pop=False, min_count=min_count,prune=prune)
    num_test = test_set.__len__()
    test_generator = DataLoader(test_set, batch_size = num_test, shuffle = False, collate_fn = lambda x: custom_collate(x, pop_avg_, pop_avg_env, pop_std, 1.0))
    for data in test_generator:
        break

    death_ages = data['death age'].numpy()
    censored = data['censored'].numpy()
    times = data['times'].numpy()
    ages = times[:,0]
    death_ages = np.array([death_ages[m] if death_ages[m] > 0 else times[m].max() for m in range(death_ages.size)])
    sample_weight = data['weights'].numpy()

    cindex = cindex_td(death_ages, survival[:,:,1], survival[:,:,0], 1 - censored)
    results['C-index'][N] = cindex


results.index.name = 'N'
results.reset_index(inplace=True)    
plot = sns.scatterplot(data=results,x='N',y='C-index')
plot.set_xlabel('Model dimension')
plot.set_ylabel('Survival C-index')
plt.ylim(.5,1)
fig = plot.get_figure()
fig.savefig(f'../Plots/latent_cindex_by_dim_job_id{args.job_id}_epoch{args.epoch}{postfix}.pdf')

