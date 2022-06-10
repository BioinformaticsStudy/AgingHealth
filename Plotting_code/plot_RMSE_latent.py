import argparse
import pandas as pd
import numpy as np
import torch
from Alternate_models.dataset_dim import Dataset
from torch.utils.data import DataLoader
from DataLoader.collate import custom_collate
from Utils.transformation import Transformation

parser = argparse.ArgumentParser('Cindex_Latent')
parser.add_argument('--job_id',type=int)
parser.add_argument('--epoch',type=int,default=1999)
parser.add_argument('--start', type=int,default=2,help='lowest N')
parser.add_argument('--step', type=int, default=5,help='difference in N between models')
parser.add_argument('--stop', type=int,default=35,help='highest N')
parser.add_argument('--dataset', type=str, default='elsa',choices=['elsa','sample'],help='what dataset was used to train the models')
args = parser.parse_args()

postfix = '_sample' if args.dataset=='sample' else ''
device = 'cpu'

dt = 0.5
length = 50


Ns = list(np.arange(args.start,args.stop,args.step)) + [args.stop]
results = pd.DataFrame(index=Ns,columns=['Mean RMSE']) 
for N in Ns:
    pop_avg = np.load(f'../Data/Population_averages{N}{postfix}.npy')
    pop_avg_env = np.load(f'../Data/Population_averages_env{N}{postfix}.npy')
    pop_std = np.load(f'../Data/Population_std{postfix}{N}.npy')
    pop_avg_ = torch.from_numpy(pop_avg[...,1:]).float()
    pop_avg_env = torch.from_numpy(pop_avg_env).float()
    pop_std = torch.from_numpy(pop_std[...,1:]).float()
    pop_avg_bins = np.arange(40, 105, 3)[:-2]

    test_name = f'../Data/test{postfix}.csv'
    test_set = Dataset(test_name, N, pop=False, min_count = 10)
    num_test = test_set.__len__()
    test_generator = DataLoader(test_set, batch_size = num_test, shuffle = False, collate_fn = lambda x: custom_collate(x, pop_avg_, pop_avg_env, pop_std, 1.0))

    mean_deficits = pd.read_csv(f'../Data/mean_deficits_latent.txt',sep=',',header=None, names = ['variable','value'])[1:N+1]
    std_deficits = pd.read_csv(f'../Data/std_deficits_latent.txt',sep=',',header=None, names = ['variable','value'][1:N+1])
    mean_deficits.reset_index(inplace=True,drop=True)
    std_deficits.reset_index(inplace=True,drop=True)

    # get indexes of log scaled variables to be used in Transformation
    log_scaled_variables = ['fer','trig','crp', 'wbc', 'mch', 'vitd', 'dheas','leg raise','full tandem']
    log_scaled_indexes = []
    for variable in log_scaled_variables:
        row = mean_deficits.loc[mean_deficits['variable']==variable]
        if len(row) > 0:
            index = row.index[0]
            log_scaled_indexes.append(index)

    mean_deficits.drop(['variable'],axis='columns', inplace=True)
    std_deficits.drop(['variable'],axis='columns', inplace=True)
    mean_deficits = mean_deficits.values.flatten()
    std_deficits = std_deficits.values.flatten()

    psi = Transformation(mean_deficits, std_deficits, log_scaled_indexes)