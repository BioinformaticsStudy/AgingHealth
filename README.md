
Parsing Data:
- Place ELSA data in directory that the repository is in
- Execute Data_Parser/create_elsa_data.sh
- This reads the ELSA data, cleans it up, and splits it into three files for training, validation, and testing
- Also creates mean_deficits.txt and std_deficits.txt, used for calculating relative RMSE
- Execute data_info.py to create a file with information about the data

Model:
- located in /DJIN_Model
- model.py: main code for model
- diagonal_func.py: contains N neural networks for each deficit
- dynamics.py: calculates dynamics at a particular time step
- solver.py: uses dynamics on every time set
- loss.py: contains functions for calculating loss
- memory_model.py: contains nueral net used for calculating inital h for survival RNN
- vae_flow.py: contains variational autoencoder (VAE) for imputing data
- realnvp_flow.py: contains nvp flows used in VAE

Training:
- generate averages and standard deviations with population_average.py and population_std.py
- Execute train.py with a job_id
- Optionally set the hyperparameters, which are output to /Output
- Outputs trained parameters to /Parameters

Predicting:
- execute predict.py with a job_id and an epoch
- generates file with survival trajectories, used for c-index, brier score, and d-callibration
- generates file with mean trajectories for deficits, used for longitudinal predictions

Comparison model:
- located in /Comparison_model
- execute longitudinal.py to generate mean trajectories for deficits
- execute survival.py to generate survival trajectories

Latent model:
- located in /Alternate_model
- trained with train_full.py, predictions made with predict_full.py
- option to specify N
- generate variables to be used with generate_variables.py, orders variables by the amount they change in the data
- can also manually create variables.txt in /Data, height, bmi, ethnicity, and sex must be the last four variables
- create population averages and standard deviations with population_average_latent.py and population_std_latent.py
