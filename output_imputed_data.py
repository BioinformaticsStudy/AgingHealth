import pandas as pd
import torch
import argparse
from os import getcwd
from DJIN_Model.model import Model
from DataLoader.dataset import Dataset
from DataLoader.collate import custom_collate
from torch.utils import data
import numpy as np
from os import getcwd

parser = argparse.ArgumentParser('Output_Imputed_Data')
parser.add_argument('--trainset', default='elsa', choices=['sample','elsa'], type=str)
parser.add_argument('--dataset', default='ELSA_cleaned',type=str)
parser.add_argument('--job_id', type=int)
parser.add_argument('--epoch', type=int)
args = parser.parse_args()
train_postfix = '_sample' if args.trainset=='sample' else ''


torch.set_num_threads(6)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N = 29
sims = 250
dt = 0.5
length = 50

# getting hyperparameters
hp = dict()
hyperparameters_file = f'{getcwd()}/Output/train{args.job_id}{train_postfix}.hyperparams'
lines = open(hyperparameters_file, 'r').readlines()
for line in lines:
    line = line.replace('\n','')
    parts = line.split(', ')
    hp[parts[0]] = float(parts[1]) if '.' in parts[1] else int(parts[1])

# getting avg / std
initial_pop_avg = np.load(f'Data/Population_averages{train_postfix}.npy')
initial_pop_avg_env = np.load(f'Data/Population_averages_env{train_postfix}.npy')
initial_pop_std = np.load(f'Data/Population_std{train_postfix}.npy')
initial_pop_avg_ = torch.from_numpy(initial_pop_avg[...,1:]).float()
initial_pop_avg_env = torch.from_numpy(initial_pop_avg_env).float()
initial_pop_std = torch.from_numpy(initial_pop_std[...,1:]).float()
pop_avg_bins = np.arange(40, 105, 3)[:-2]

# preparing data
train_name = f'Data/{args.dataset}.csv'
train_set = Dataset(train_name, N, pop=False, min_count=10)
num_test = 400
train_generator = data.DataLoader(train_set, batch_size = num_test, shuffle = False, collate_fn = lambda x: custom_collate(x, initial_pop_avg_, initial_pop_avg_env, initial_pop_std, 1.0))

# getting mean and std ages from data
mean_T = train_set.mean_T
std_T = train_set.std_T

# creating model
model = Model(device, N, hp['gamma_size'], hp['z_size'], hp['decoder_size'], hp['Nflows'], hp['flow_hidden'], hp['f_nn_size'], mean_T, std_T, dt, length).to(device)
model.load_state_dict(torch.load('Parameters/train%d_Model_DJIN_epoch%d%s.params'%(args.job_id, args.epoch,train_postfix),map_location=device))
model = model.eval()

columns = ['gait speed', 'grip dom', 'grip ndom', 'FI ADL', 'FI IADL', 'chair','leg raise', 'full tandem', 'srh', 'eye',
           'hear', 'func', 'dias', 'sys', 'pulse', 'trig',
           'crp','hdl','ldl','glucose','igf1','hgb','fib','fer', 'chol', 'wbc', 'mch', 'hba1c', 'vitd']
columnIndexes = dict()
for i,column in enumerate(columns):
    columnIndexes[i] = column

imputed_baselines = pd.DataFrame(columns=columns)
# iterating through data
with torch.no_grad():
    for data in train_generator:
        y = data['Y'].to(device)
        times = data['times'].to(device)
        mask = data['mask'].to(device)
        mask0 = data['mask0'].to(device)
        survival_mask = data['survival_mask'].to(device)
        dead_mask = data['dead_mask'].to(device)
        after_dead_mask = data['after_dead_mask'].to(device)
        censored = data['censored'].to(device)
        env = data['env'].to(device)
        med = data['med'].to(device)
        sample_weights = data['weights'].to(device)
        predict_missing = data['missing'].to(device)
        pop_std = data['pop std'].to(device)
        
        batch_size = y.shape[0]
        
        # create initial timepoints
        y0_ = y[:, 0, :]
        t0 = times[:, 0]
        med0 = med[:,0,:]
        trans_t0 = (t0.unsqueeze(-1) - mean_T)/std_T
        

        # fill in missing for input
        y0 = mask[:,0,:]*(y0_) \
        + (1 - mask[:,0,:])*(predict_missing + pop_std*torch.randn_like(y0_))

        #sample VAE
        sample0, z_sample, mu0, logvar0, prior_entropy, log_det = model.impute(trans_t0, y0, mask[:,0,:], env, med0)

        # impute
        recon_mean_x0 = model.impute.decoder(torch.cat((z_sample, trans_t0, env, med0), dim=-1))
        
        # baseline state
        x0 = mask[:,0,:] * (y0_ ) + (1 - mask[:,0,:]) * recon_mean_x0
        
        #adding to output
        x0 = pd.DataFrame(x0.numpy()).rename(columns=columnIndexes)
        imputed_baselines = imputed_baselines.append(x0,ignore_index=True)

imputed_baselines.to_csv(f'{getcwd()}/Data/{args.dataset}_imputed_id{args.job_id}_epoch{args.epoch}.csv')
