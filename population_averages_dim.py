import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader
from scipy.stats import binned_statistic

from scipy.signal import savgol_filter
import argparse
from pathlib import Path
import sys
import os
file = Path(__file__). resolve()  
package_root_directory = file.parents [1]  
sys.path.append(str(package_root_directory))

from Alternate_models.dataset_dim import Dataset

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def run(dataset, N):
    postfix = f'_latent{N}_sample' if dataset == 'sample' else f'_latent{N}'
    dir = os.path.dirname(os.path.realpath(__file__))

    device = 'cpu'

    dt = 0.5

    train_name = f'{dir}/Data/train{postfix}.csv' 
    training_set = Dataset(train_name, N, pop=True)
    num_train = training_set.__len__()
    training_generator = DataLoader(training_set,
                                                batch_size = num_train,
                                                shuffle = False, drop_last = False)

    mean_T = training_set.mean_T
    std_T = training_set.std_T


    age_bins = np.arange(40, 105, 3)
    bin_centers = age_bins[1:] - np.diff(age_bins)/2.0

    avg = np.zeros((2, bin_centers.shape[0], N + 1))
    avg_smooth = np.zeros((2, bin_centers.shape[0], N + 1))

    avg_env = np.zeros((2, bin_centers.shape[0], 2))
    avg_env_smooth = np.zeros((2, bin_centers.shape[0], 2))

    for batch_data, batch_times, batch_mask, batch_survival_mask, batch_dead_mask, _, batch_censored, _, batch_env, batch_med, batch_weights in training_generator:

        times = batch_times.numpy()
        data = batch_data.numpy()
        mask = batch_mask.numpy()
        env = batch_env.numpy()

        env_times = batch_times.numpy()[:,0]

        num_env = 29+19-N-5 # total variables - deficits - medications
        sex_index = num_env-1
        bmi_index = num_env-3
        height_index = num_env-4
        
        for sex in [0,1]:
            selected = (env[:,sex_index] == sex)
            size = np.sum(selected).astype(int)*batch_data.shape[1]

            curr_times = times[selected].reshape(size)
            curr_data = data[selected].reshape(size, N)
            curr_mask = mask[selected].reshape(size, N)
            
            for evid, ev in enumerate([height_index,bmi_index]):
                avg_env[sex, 3:-4, evid] = binned_statistic(env_times[selected][env[selected, ev]>-100], env[selected][env[selected, ev]>-100, ev], bins = age_bins)[0][3:-4]
            
                avg_env_smooth[sex, 3:-4, evid] = savgol_filter(avg_env[sex, 3:-4, evid], 9, 3)
                
                nans, x = nan_helper(avg_env[sex, 3:-4, evid])
                avg_env[sex, 3:-4, evid][nans] = np.interp(x(nans), x(~nans), avg_env[sex, 3:-4, evid][~nans])
            
                avg_env_smooth[sex, 3:-4, evid] = savgol_filter(avg_env[sex, 3:-4, evid], 9, 3)
            
            for n in range(N):
                avg[sex, 3:-4,1+n] = binned_statistic(curr_times[curr_mask[:, n]>0], curr_data[curr_mask[:, n]>0,n], bins= age_bins)[0][3:-4]
            
                nans, x= nan_helper(avg[sex, 3:-4,1+n])
                avg[sex, 3:-4,1+n][nans]= np.interp(x(nans), x(~nans), avg[sex, 3:-4,1+n][~nans])
            
                avg_smooth[sex, 3:-4,1+n] = savgol_filter(avg[sex, 3:-4,1+n], 9, 3)


    for sex in [0, 1]:
        avg[sex, :3] = avg[sex,3]
        avg[sex,-4:] = avg[sex,-5]
        
        avg_smooth[sex,:3] = avg_smooth[sex, 3]
        avg_smooth[sex,-4:] = avg_smooth[sex,-5]
        
        avg_env[sex,:3] = avg_env[sex,3]
        avg_env[sex,-4:] = avg_env[sex,-5]
        
        avg_env_smooth[sex,:3] = avg_env_smooth[sex,3]
        avg_env_smooth[sex,-4:] = avg_env_smooth[sex,-5]
        
        avg[sex, :,0] = bin_centers
        avg_smooth[sex, :,0] = bin_centers


    np.save(f'{dir}/Data/Population_averages{postfix}.npy', avg_smooth)
    np.save(f'{dir}/Data/Population_averages_env{postfix}.npy', avg_env_smooth)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pop_avg')
    parser.add_argument('--dataset',type=str,choices=['elsa','sample'],default='elsa',help='the dataset that will be used to train the model; either \'elsa\' or \'sample\'')
    parser.add_argument('--N', type=int, default=29,  help='how many deficits to use')
    args = parser.parse_args()

    run(args.dataset,args.N)