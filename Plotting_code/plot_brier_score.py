import argparse
import torch
import numpy as np
from scipy.stats import sem, binned_statistic
from pandas import read_csv
from torch.utils import data
import os

from pathlib import Path
import sys
file = Path(__file__). resolve()  
package_root_directory = file.parents [1]  
sys.path.append(str(package_root_directory))  

from DataLoader.collate import custom_collate
from Utils.cindex import cindex_td

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
cm = plt.get_cmap('Set1')

parser = argparse.ArgumentParser('Brier score')
parser.add_argument('--job_id', type=int)
parser.add_argument('--epoch', type=int)
parser.add_argument('--dataset',type=str,choices=['elsa','sample'],default='elsa',help='the dataset that was used to train the model; either \'elsa\' or \'sample\'')
parser.add_argument('--no_compare',action='store_true',help='whether or not to plot the comparison model')
parser.add_argument('--minscore', type=float, default=None, help='minimum brier score showed on plot')
parser.add_argument('--latentN', type=int, default=None, help='N if plotting latent model; can be left None if plotting DJIN')
parser.add_argument('--compare_id',type=int)
args = parser.parse_args()
postfix = '_sample' if args.dataset == 'sample' else ''
model_name = 'DJIN' if args.latentN==None else f'latent{args.latentN}'

if args.latentN == None:
    from DataLoader.dataset import Dataset
else:
    from Alternate_models.dataset_dim import Dataset

dir = os.path.dirname(os.path.realpath(__file__))

device = 'cpu'

N = 29 if args.latentN==None else args.latentN
dt = 0.5
length = 50

pop_avg = np.load(f'{dir}/../Data/Population_averages{postfix}.npy')
pop_avg_env = np.load(f'{dir}/../Data/Population_averages_env{postfix}.npy')
pop_std = np.load(f'{dir}/../Data/Population_std{postfix}.npy')
pop_avg_ = torch.from_numpy(pop_avg[...,1:]).float()
pop_avg_env = torch.from_numpy(pop_avg_env).float()
pop_std = torch.from_numpy(pop_std[...,1:]).float()

test_name = f'{dir}/../Data/test{postfix}.csv'
test_set = Dataset(test_name, N, pop=False, min_count=10)
num_test = test_set.__len__()
test_generator = data.DataLoader(test_set, batch_size = num_test, shuffle = False, collate_fn = lambda x: custom_collate(x, pop_avg_, pop_avg_env, pop_std, 1.0))

with torch.no_grad():
    
    survival = np.load(dir+'/../Analysis_Data/Survival_trajectories_job_id%d_epoch%d_%s%s.npy'%(args.job_id,args.epoch,model_name,postfix))
    
    if not args.no_compare: 
        linear = np.load(f'{dir}/../Comparison_models/Predictions/Survival_trajectories_baseline_id{args.compare_id}_rfmice{postfix}.npy')
    
    start = 0
    for data in test_generator:
        break

    death_ages = data['death age'].numpy()
    censored = data['censored'].numpy()
    times = data['times'].numpy()
    dead_mask = data['survival_mask'].numpy()
    ages = times[:,0]

    death_ages = np.array([death_ages[m] if death_ages[m] > 0 else times[m].max() for m in range(death_ages.size)])
    
    sample_weight = data['weights'].numpy()
    sex_index = data['env'][:,-1].long().numpy()
    
    
#dead_mask = np.concatenate((dead_mask, np.zeros(dead_mask.shape)), axis = 1)
dead_mask = np.concatenate((dead_mask, np.zeros((dead_mask.shape[0], dead_mask.shape[1]*3))), axis = 1)

survival_prob = survival[:,:,1]
survival_ages = survival[:,:,0]

if not args.no_compare:
    linear_prob = linear[:,:,1]
    linear_ages = linear[:,:,0]


from lifelines import KaplanMeierFitter
    
observed = 1 - np.array(censored,dtype=int)

kmf_S = KaplanMeierFitter()
kmf_S.fit(death_ages, event_observed = observed, timeline = np.arange(0, 200, 1))
S = kmf_S.survival_function_.values.flatten()
S = np.array([S[int(np.nanmin(survival_ages[i])):][:len(survival_ages[i])] for i in range(len(censored))])

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
    # die
    if censored[i] == 0:
        ages = survival_ages[i, ~np.isnan(survival_ages[i])]
        mask = dead_mask[i, ~np.isnan(survival_ages[i])]
        prob = survival_prob[i, ~np.isnan(survival_ages[i])]
        S_i = S[i][~np.isnan(survival_ages[i])]
        G_i = G[i][~np.isnan(survival_ages[i])]

        ages = ages[~np.isnan(prob)]
        mask = mask[~np.isnan(prob)]
        S_i = S_i[~np.isnan(prob)]
        G_i = G_i[~np.isnan(prob)]
        prob = prob[~np.isnan(prob)]
        
        
        G_alive = G_i[mask==1]
        G_dead = G_i[mask==0]
        G_dead = G_alive[-1]*np.ones(G_dead.shape)
        G_i = np.concatenate((G_alive, G_dead))
        
        
        G_i[G_i < 1e-5] = np.nan
        
        if len(ages[~np.isnan(G_i)]) != 0:
            BS += binned_statistic(ages[~np.isnan(G_i)], ((mask - prob)**2/G_i)[~np.isnan(G_i)], bins = np.arange(30.5, 130.5, 1), statistic = np.nansum)[0]
            BS_count += binned_statistic(ages[~np.isnan(G_i)], ((mask - prob)**2/G_i)[~np.isnan(G_i)], bins = np.arange(30.5, 130.5, 1), statistic = 'count')[0]

            BS_S += sample_weight[i]*binned_statistic(ages[~np.isnan(G_i)], ((mask - prob)**2/G_i)[~np.isnan(G_i)], bins = np.arange(30.5, 130.5, 1), statistic = np.sum)[0]
            BS_S_count += sample_weight[i]*binned_statistic(ages[~np.isnan(G_i)], ((mask - prob)**2/G_i)[~np.isnan(G_i)], bins = np.arange(30.5, 130.5, 1), statistic = 'count')[0]
    
    else:
        ages = survival_ages[i, ~np.isnan(survival_ages[i])]
        mask = dead_mask[i, ~np.isnan(survival_ages[i])]
        prob = survival_prob[i, ~np.isnan(survival_ages[i])]
        S_i = S[i][~np.isnan(survival_ages[i])[:len(S[i])]]
        G_i = G[i][~np.isnan(survival_ages[i])[:len(G[i])]]

        
        ages = ages[~np.isnan(prob)]
        mask = mask[~np.isnan(prob)]
        S_i = S_i[~np.isnan(prob)]
        G_i = G_i[~np.isnan(prob)]
        prob = prob[~np.isnan(prob)]
        
        ages = ages[mask==1]
        prob = prob[mask==1]
        S_i = S_i[mask==1]
        G_i = G_i[mask==1]
        mask = mask[mask==1]
        
        G_i[G_i < 1e-5] = np.nan

        if len(ages[~np.isnan(G_i)]) != 0:
            BS += sample_weight[i]*binned_statistic(ages[~np.isnan(G_i)], ((mask - prob)**2/G_i)[~np.isnan(G_i)], bins = np.arange(30.5, 130.5, 1), statistic = np.nansum)[0]
            BS_count += sample_weight[i]*binned_statistic(ages[~np.isnan(G_i)], ((mask-prob)**2/G_i)[~np.isnan(G_i)], bins = np.arange(30.5, 130.5, 1), statistic = 'count')[0]
            
            BS_S += sample_weight[i]*binned_statistic(ages[~np.isnan(G_i)], ((mask - S_i)**2/G_i)[~np.isnan(G_i)], bins = np.arange(30.5, 130.5, 1), statistic = np.nansum)[0]
            BS_S_count += sample_weight[i]*binned_statistic(ages[~np.isnan(G_i)], ((mask - S_i)**2/G_i)[~np.isnan(G_i)], bins = np.arange(30.5, 130.5, 1), statistic = 'count')[0]

if not args.no_compare:
    BS_linear = np.zeros(bin_centers.shape)
    BS_linear_count = np.zeros(bin_centers.shape)

    for i in range(len(survival_ages)):

        # die
        if censored[i] == 0:
            if not args.no_compare:
                ages = linear_ages[i, ~np.isnan(linear_ages[i])]
                mask = dead_mask[i, ~np.isnan(linear_ages[i])]
                prob = linear_prob[i, ~np.isnan(linear_ages[i])]
                

                S_i = S[i][~np.isnan(linear_ages[i])[:len(S[i])]]
                G_i = G[i][~np.isnan(linear_ages[i])[:len(G[i])]]

            ages = ages[~np.isnan(prob)]
            mask = mask[~np.isnan(prob)]
            S_i = S_i[~np.isnan(prob)]
            G_i = G_i[~np.isnan(prob)]
            prob = prob[~np.isnan(prob)]
            
            G_alive = G_i[mask==1]
            G_dead = G_i[mask==0]
            G_dead = G_alive[-1]*np.ones(G_dead.shape)
            G_i = np.concatenate((G_alive, G_dead))

            G_i[G_i < 1e-5] = np.nan
            
            BS_linear += binned_statistic(ages[~np.isnan(G_i)], ((mask - prob)**2/G_i)[~np.isnan(G_i)], bins = np.arange(30.5, 130.5, 1), statistic = np.sum)[0]
            BS_linear_count += binned_statistic(ages[~np.isnan(G_i)], ((mask - prob)**2/G_i)[~np.isnan(G_i)], bins = np.arange(30.5, 130.5, 1), statistic = 'count')[0]
        
        else:
            ages = linear_ages[i, ~np.isnan(linear_ages[i])]
            mask = dead_mask[i, ~np.isnan(linear_ages[i])]
            prob = linear_prob[i, ~np.isnan(linear_ages[i])]
            S_i = S[i][~np.isnan(linear_ages[i])[:len(S[i])]]
            G_i = G[i][~np.isnan(linear_ages[i])[:len(G[i])]]

            
            ages = ages[~np.isnan(prob)]
            mask = mask[~np.isnan(prob)]
            S_i = S_i[~np.isnan(prob)]
            G_i = G_i[~np.isnan(prob)]
            prob = prob[~np.isnan(prob)]
            
            ages = ages[mask==1]
            prob = prob[mask==1]
            S_i = S_i[mask==1]
            G_i = G_i[mask==1]
            mask = mask[mask==1]
            
            G_i[G_i < 1e-5] = np.nan
            
            BS_linear += sample_weight[i]*binned_statistic(ages[~np.isnan(G_i)], ((mask - prob)**2/G_i)[~np.isnan(G_i)], bins = np.arange(30.5, 130.5, 1), statistic = np.nansum)[0]
            BS_linear_count += sample_weight[i]*binned_statistic(ages[~np.isnan(G_i)], ((mask-prob)**2/G_i)[~np.isnan(G_i)], bins = np.arange(30.5, 130.5, 1), statistic = 'count')[0]


BS_t = (BS/BS_count)
if not args.no_compare:
    BS_linear_t = (BS_linear/BS_linear_count)
BS_S_t = (BS_S/BS_S_count)

fig,ax = plt.subplots(figsize=(4.5,4.5))

plt.plot(bin_centers[bin_centers>=60], BS_t[bin_centers>=60], color = cm(0), label = 'DJIN model', linewidth = 2.5)
if not args.no_compare:
    plt.plot(bin_centers[bin_centers>=60], BS_linear_t[bin_centers>=60], color = cm(2), label = 'Elastic net Cox', linewidth = 2.5, linestyle = '--')


min_death_age = death_ages[censored==0].min()
max_death_age = death_ages[censored==0].max()


IBS = np.trapz(y = BS_t[ (bin_centers>=min_death_age) & (bin_centers<=max_death_age) ], x = bin_centers[ (bin_centers>=min_death_age) & (bin_centers<=max_death_age) ])/(max_death_age-min_death_age)
if not args.no_compare:
    IBS_linear = np.trapz(y = BS_linear_t[ (bin_centers>=min_death_age) & (bin_centers<=max_death_age) ], x = bin_centers[ (bin_centers>=min_death_age) & (bin_centers<=max_death_age) ])/(max_death_age-min_death_age)
ax.text(.84, 0.55, r'IBS = %.2f'%IBS, horizontalalignment='center', verticalalignment='center',transform=ax.transAxes, color=cm(0),fontsize = 14, zorder=1000000)

if not args.no_compare:
    ax.text(.25, 0.55, r'IBS = %.2f'%IBS_linear, horizontalalignment='center', verticalalignment='center',transform=ax.transAxes, color=cm(2),fontsize = 14, zorder=1000000)

ax.text(-0.05, 1.05, 'b', horizontalalignment='left', verticalalignment='center',transform=ax.transAxes, color='k',fontsize = 16, zorder=1000000,
        fontweight='bold')

if args.minscore is not None:
    plt.ylim(args.minscore*.1,20)
plt.xlim(60, 100)
plt.xlabel('Death age (years)', fontsize = 14)
plt.ylabel('Survival Brier score', fontsize = 14)
ax.tick_params(labelsize=12)

ax.xaxis.set_minor_locator(MultipleLocator(5))

plt.yscale('log')
plt.legend(loc = 'lower right')
plt.tight_layout()
plt.savefig(dir+'/../Plots/Brier_score_job_id%d_epoch%d%s.pdf'%(args.job_id, args.epoch,postfix))

with open(f'{dir}/../Analysis_Data/IBS_job_id{args.job_id}_epoch{args.epoch}{postfix}.txt','w') as outfile:
    outfile.writelines(str(IBS))
    if not args.no_compare:
        outfile.writelines(',' + str(IBS_linear))