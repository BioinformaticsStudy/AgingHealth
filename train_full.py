# training for latent model located in /Alternate_Models
# run split_data.py (with latentN argument), population_averages_dim.py, and population_std_dim.py before running this

import argparse
import os
import numpy as np
import itertools
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from Alternate_models.model_full import Model

from Alternate_models.dataset_dim import Dataset
from DataLoader.collate import custom_collate
from Utils.schedules import LinearScheduler

from Alternate_models.loss_full import loss, sde_KL_loss


def train(job_id, batch_size, niters, learning_rate, corruption, gamma_size, z_size, decoder_size, Nflows, flow_hidden, dataset, N):
    postfix = f'_latent{N}_sample' if dataset=='sample' else f'_latent{N}'
    dir = os.path.dirname(os.path.realpath(__file__))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        num_workers = 8
        torch.set_num_threads(10)
        test_after = 10
        test_average = 5
    else:
        num_workers = 0
        torch.set_num_threads(4)
        test_after = 10
        test_average = 3

    # folders for output
    params_folder = dir+'/Parameters/'
    output_folder = dir+'/Output/'

    # setting up file for loss outputs
    loss_file = '%svalidation%d.loss'%(output_folder, job_id)
    open(loss_file, 'w')

    # output hyperparameters
    hyperparameters_file = '%strain%d_full%d.hyperparams'%(output_folder, job_id, N)
    with open(hyperparameters_file, 'w') as hf:
        hf.writelines('batch_size, %d\n'%batch_size)
        hf.writelines('niters, %d\n'%niters)
        hf.writelines('learning_rate, %.3e\n'%learning_rate)
        hf.writelines('corruption, %.3f\n'%corruption)
        hf.writelines('gamma_size, %d\n'%gamma_size)
        hf.writelines('z_size, %d\n'%z_size)
        hf.writelines('decoder_size, %d\n'%decoder_size)
        hf.writelines('Nflows, %d\n'%Nflows)
        hf.writelines('flow_hidden, %d\n'%flow_hidden)

    dt = 0.5

    pop_avg = np.load(f'{dir}/Data/Population_averages{postfix}.npy')
    pop_avg_env = np.load(f'{dir}/Data/Population_averages_env{postfix}.npy')
    pop_std = np.load(f'{dir}/Data/Population_std{postfix}.npy')
    pop_avg = torch.from_numpy(pop_avg[...,1:]).float()
    pop_avg_env = torch.from_numpy(pop_avg_env).float()
    pop_std = torch.from_numpy(pop_std[...,1:]).float()

    min_count = N // 4
    prune = min_count >= 1

    train_name = f'{dir}/Data/train{postfix}.csv'
    training_set = Dataset(train_name, N, pop=False, min_count = min_count, prune=prune)
    training_generator = DataLoader(training_set,
                                        batch_size = batch_size,
                                        shuffle = True, drop_last = True, num_workers = num_workers, pin_memory=True,
                                        collate_fn = lambda x: custom_collate(x, pop_avg, pop_avg_env, pop_std, corruption))

    valid_name = f'{dir}/Data/valid{postfix}.csv'
    validation_set = Dataset(valid_name, N, pop=False, min_count = min_count,prune=prune)
    validation_generator = DataLoader(validation_set,
                                          batch_size = 4000,
                                          shuffle = False, drop_last = False,pin_memory=True,
                                          collate_fn = lambda x: custom_collate(x, pop_avg, pop_avg_env, pop_std, 1.0))

    print('Data loaded: %d training examples and %d validation examples'%(training_set.__len__(), validation_set.__len__()))


    mean_T = training_set.mean_T
    std_T = training_set.std_T

    print(mean_T, std_T)

    # initialize ae, model, solver, and optimizer
    model = Model(device, N, gamma_size, z_size, decoder_size, Nflows, flow_hidden, mean_T, std_T, dt).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor = 0.5, threshold = 1000, threshold_mode ='abs', patience = 4, min_lr = 1e-5, verbose=True)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Model has %d parameters'%params)


    matrix_mask = (1 - torch.eye(N).to(device))


    kl_scheduler_dynamics = LinearScheduler(300)
    kl_scheduler_vae = LinearScheduler(500)

    sigma_prior = torch.distributions.gamma.Gamma(torch.tensor(1.0).to(device), torch.tensor(25000.0).to(device))
    vae_prior = torch.distributions.normal.Normal(torch.tensor(0.0).to(device), torch.tensor(1.0).to(device))

    niters = niters
    for epoch in range(niters):

        beta_dynamics = kl_scheduler_dynamics()
        beta_vae = kl_scheduler_vae()
        
        for data in training_generator:
            
            optimizer.zero_grad()
            
            sigma_posterior = torch.distributions.gamma.Gamma(model.logalpha.exp(), model.logbeta.exp())
            sigma_y = sigma_posterior.rsample((data['Y'].shape[0], data['Y'].shape[1]))
            
            pred_X, pred_Z, t, pred_S, pred_logGamma, pred_sigma_X, context, y, times, mask, survival_mask, dead_mask, after_dead_mask, censored, sample_weights, med, env, z_sample, prior_entropy, log_det, recon_mean_x0, mask0 = model(data, sigma_y)
            summed_weights = torch.sum(sample_weights)
            kl_term = torch.sum(torch.sum(sample_weights*((mask*sigma_posterior.log_prob(sigma_y)).permute(1,2,0)),dim=(1,2)) - torch.sum(sample_weights*((mask*sigma_prior.log_prob(sigma_y)).permute(1,2,0)),dim=(1,2))) - beta_vae*torch.sum(sample_weights*vae_prior.log_prob(z_sample).permute(1,0)) - torch.sum(sample_weights*(prior_entropy.permute(1,0))) - torch.sum(sample_weights*log_det)
            
            # calculate loss
            l = loss(pred_X[:,::2], recon_mean_x0, pred_logGamma[:,::2], pred_S[:,::2], survival_mask, dead_mask, after_dead_mask, times, y, censored, mask, sigma_y[:,1:], sigma_y[:,0], sample_weights) + beta_dynamics*sde_KL_loss(pred_Z, t, context, dead_mask, model.dynamics.posterior_drift, model.dynamics.prior_drift, pred_sigma_X, dt, mean_T, std_T, sample_weights, med) + kl_term
            
            # calculate gradients and update params
            l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1E4)
            optimizer.step()
        
        # loss for validation set
        if epoch % test_after == 0:

            model = model.eval()
            
            with torch.no_grad():

                total_loss = 0.
                recon_loss = 0.
                kl_loss = 0.
                sde_loss = 0.
                for i in range(test_average):
                    
                    for data in validation_generator:
                        
                        sigma_posterior = torch.distributions.gamma.Gamma(model.logalpha.exp(), model.logbeta.exp())
                        sigma_y = sigma_posterior.rsample((data['Y'].shape[0], data['Y'].shape[1]))
                        
                        pred_X, pred_Z, t, pred_S, pred_logGamma, pred_sigma_X, context, y, times, mask, survival_mask, dead_mask, after_dead_mask, censored, sample_weights, med, env, z_sample, prior_entropy, log_det, recon_mean_x0, mask0 = model(data, sigma_y, test=True)
                        summed_weights = torch.sum(sample_weights)
                        
                        
                        kl_term = torch.sum(torch.sum(sample_weights*((mask*sigma_posterior.log_prob(sigma_y)).permute(1,2,0)),dim=(1,2)) - torch.sum(sample_weights*((mask*sigma_prior.log_prob(sigma_y)).permute(1,2,0)),dim=(1,2))) - beta_vae*torch.sum(sample_weights*vae_prior.log_prob(z_sample).permute(1,0)) - torch.sum(sample_weights*(prior_entropy.permute(1,0))) - torch.sum(sample_weights*log_det)
                        
                        # calculate loss
                        recon_l = loss(pred_X[:,::2], recon_mean_x0, pred_logGamma[:,::2], pred_S[:,::2], survival_mask, dead_mask, after_dead_mask, t, y, censored, mask, sigma_y[:,1:], sigma_y[:,0], sample_weights)
                        full_l = sde_KL_loss(pred_Z, t, context, dead_mask, model.dynamics.posterior_drift, model.dynamics.prior_drift, pred_sigma_X, dt, mean_T, std_T, sample_weights, med)
                        
                        kl_loss += kl_term
                        total_loss += full_l + recon_l + kl_term
                        recon_loss += recon_l
                        sde_loss += full_l
                        
                # output loss
                with open(loss_file, 'a') as lf:
                    lf.writelines('%d, %.3f, %.3f\n'%(epoch, recon_loss.cpu().numpy()/test_average, total_loss.cpu().numpy()/test_average))
                print('N %d, Epoch %d, recon loss %.3f, total loss %.3f, kl loss %.3f, SDE loss %.3f, beta dynamics %.3f, beta vae %.3f) '%(N, epoch, recon_loss.cpu().numpy()/test_average, total_loss.cpu().numpy()/test_average, kl_loss.cpu().numpy()/test_average, sde_loss.cpu().numpy(), beta_dynamics, beta_vae), pred_sigma_X.cpu().mean(), sigma_y.cpu().mean())
                
            model = model.train()
            
            # step learning rate
            scheduler.step(total_loss/test_average)
            
        # output params
        if epoch % 20 ==0:
            torch.save(model.state_dict(), '%strain%d_Model%s_epoch%d.params'%(params_folder, job_id, postfix, epoch))

        kl_scheduler_dynamics.step()
        kl_scheduler_vae.step()

    torch.save(model.state_dict(), '%strain%d_Model%s_epoch%d.params'%(params_folder, job_id, postfix, epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train')
    parser.add_argument('--job_id', type=int)
    parser.add_argument('--batch_size', type=int, default = 1000)
    parser.add_argument('--niters', type=int, default = 2000)
    parser.add_argument('--learning_rate', type=float, default = 1e-2)
    parser.add_argument('--corruption', type=float, default = 0.9)
    parser.add_argument('--gamma_size', type=int, default = 25)
    parser.add_argument('--z_size', type=int, default = 30)
    parser.add_argument('--decoder_size', type=int, default = 65)
    parser.add_argument('--Nflows', type=int, default = 3)
    parser.add_argument('--flow_hidden', type=int, default = 24)
    parser.add_argument('--dataset',type=str, default='elsa',choices=['elsa','sample'])
    parser.add_argument('--N', type=int, default=29, help='number of deficits to use')
    args = parser.parse_args()

    train(args.job_id, args.batch_size, args.niters, args.learning_rate, args.corruption, args.gamma_size, args.z_size, args.decoder_size, args.Nflows, args.flow_hidden, args.dataset, args.N)