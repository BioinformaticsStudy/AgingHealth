#plots the integrated brier score for multiple latent space models depending on N
from pathlib import Path
import sys
file = Path(__file__). resolve()  
package_root_directory = file.parents [1]  
sys.path.append(str(package_root_directory))  

import argparse
import torch
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import sem, binned_statistic
from Alternate_models.dataset_dim import Dataset
from torch.utils.data import DataLoader
from DataLoader.collate import custom_collate
from lifelines import KaplanMeierFitter

parser = argparse.ArgumentParser('Brier_score_latent')
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
results = pd.DataFrame(index=Ns,columns=['IBS'])

for N in Ns:
    survival = np.load('../Analysis_Data/Survival_trajectories_job_id%d_epoch%d_latent%d%s.npy'%(args.job_id,args.epoch,N,postfix))
    
    pop_avg = np.load(f'../Data/Population_averages{N}{postfix}.npy')
    pop_avg_env = np.load(f'../Data/Population_averages_env{N}{postfix}.npy')
    pop_std = np.load(f'../Data/Population_std{N}{postfix}.npy')
    pop_avg_ = torch.from_numpy(pop_avg[...,1:]).float()
    pop_avg_env = torch.from_numpy(pop_avg_env).float()
    pop_std = torch.from_numpy(pop_std[...,1:]).float()

    test_set = Dataset(test_name, N,  pop=False, min_count=10)
    num_test = test_set.__len__()
    test_generator = DataLoader(test_set, batch_size = num_test, shuffle = False, collate_fn = lambda x: custom_collate(x, pop_avg_, pop_avg_env, pop_std, 1.0))
    for data in test_generator:
        break

    death_ages = data['death age'].numpy()
    censored = data['censored'].numpy()
    times = data['times'].numpy()
    dead_mask = data['survival_mask'].numpy()
    ages = times[:,0]

    death_ages = np.array([death_ages[m] if death_ages[m] > 0 else times[m].max() for m in range(death_ages.size)])

    sample_weight = data['weights'].numpy()
    dead_mask = np.concatenate((dead_mask, np.zeros((dead_mask.shape[0], dead_mask.shape[1]*3))), axis = 1)
    
    survival_prob = survival[:,:,1]
    survival_ages = survival[:,:,0]

    observed = 1 - np.array(censored,dtype=int)

    kmf_G = KaplanMeierFitter()
    kmf_G.fit(death_ages, event_observed = 1 - observed, timeline = np.arange(0, 200, 1))
    G = kmf_G.survival_function_.values.flatten()

    G = np.array([G[int(np.nanmin(survival_ages[i])):][:len(survival_ages[i])] for i in range(len(censored))])

    bin_edges = np.arange(30.5, 130.5, 1)
    bin_centers = bin_edges[1:] - np.diff(bin_edges)

    BS = np.zeros(bin_centers.shape)
    BS_count = np.zeros(bin_centers.shape)

    BS_S = np.zeros(bin_centers.shape)
    BS_S_count = np.zeros(bin_centers.shape)

    for i in range(len(survival_ages)):
        if censored[i] == 0:
            ages = survival_ages[i, ~np.isnan(survival_ages[i])]
            mask = dead_mask[i, ~np.isnan(survival_ages[i])]
            prob = survival_prob[i, ~np.isnan(survival_ages[i])]
            G_i = G[i][~np.isnan(survival_ages[i])]

            ages = ages[~np.isnan(prob)]
            mask = mask[~np.isnan(prob)]
            G_i = G_i[~np.isnan(prob)]
            prob = prob[~np.isnan(prob)]

            G_alive = G_i[mask==1]
            G_dead = G_i[mask==0]
            G_dead = G_alive[-1]*np.ones(G_dead.shape)
            G_i = np.concatenate((G_alive, G_dead))
        
        
            G_i[G_i < 1e-5] = np.nan

            if len(ages[~np.isnan(G_i)]) != 0:
                BS += sample_weight[i]*binned_statistic(ages[~np.isnan(G_i)], ((mask - prob)**2/G_i)[~np.isnan(G_i)], bins = np.arange(30.5, 130.5, 1), statistic = np.nansum)[0]
                BS_count += binned_statistic(ages[~np.isnan(G_i)], ((mask - prob)**2/G_i)[~np.isnan(G_i)], bins = np.arange(30.5, 130.5, 1), statistic = 'count')[0]
        else:
            ages = survival_ages[i, ~np.isnan(survival_ages[i])]
            mask = dead_mask[i, ~np.isnan(survival_ages[i])]
            prob = survival_prob[i, ~np.isnan(survival_ages[i])]
            G_i = G[i][~np.isnan(survival_ages[i])[:len(G[i])]]
            
            ages = ages[~np.isnan(prob)]
            mask = mask[~np.isnan(prob)]
            G_i = G_i[~np.isnan(prob)]
            prob = prob[~np.isnan(prob)]
            
            ages = ages[mask==1]
            prob = prob[mask==1]
            G_i = G_i[mask==1]
            mask = mask[mask==1]
            
            G_i[G_i < 1e-5] = np.nan

            if len(ages[~np.isnan(G_i)]) != 0:
                BS += sample_weight[i]*binned_statistic(ages[~np.isnan(G_i)], ((mask - prob)**2/G_i)[~np.isnan(G_i)], bins = np.arange(30.5, 130.5, 1), statistic = np.nansum)[0]
                BS_count += sample_weight[i]*binned_statistic(ages[~np.isnan(G_i)], ((mask-prob)**2/G_i)[~np.isnan(G_i)], bins = np.arange(30.5, 130.5, 1), statistic = 'count')[0]

    BS_t = (BS/BS_count)
    min_death_age = death_ages[censored==0].min()
    max_death_age = death_ages[censored==0].max()

    IBS = np.trapz(y = BS_t[ (bin_centers>=min_death_age) & (bin_centers<=max_death_age) ], x = bin_centers[ (bin_centers>=min_death_age) & (bin_centers<=max_death_age) ])/(max_death_age-min_death_age)
    results['IBS'][N] = IBS


# plotting code
results.index.name = 'N'
results.reset_index(inplace=True)    
plot = sns.scatterplot(data=results,x='N',y='IBS')
plot.set_xlabel('Model dimension')
plot.set_ylabel('Integrated Brier Score')
fig = plot.get_figure()
fig.savefig(f'../Plots/latent_brier_score_by_dim_job_id{args.job_id}_epoch{args.epoch}{postfix}.pdf')