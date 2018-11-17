#!/usr/bin/python
import os
import sys


# Data distribution:
# "gaussian": AR(1) model, defined by a correlation coeffecient parameter
# "gmm"     : mixture of 3 AR(1) models with different correlation coefficients
# "sparse"  : heavy tailed and weakly correlated distribusion
# "mstudent": heavy tailed correlated multivariate-t distribution
#model = "gaussian"
#base_path = "/scratch/users/yromano/CJRepo_Remote/train_machine/0ed13f755e24f3692b8c90add3618acfb84369fd"

#model = "gmm"
#base_path = "/scratch/users/yromano/CJRepo_Remote/train_machine/2d7995d06617d5668174c4b1736f2354938a0f5a"
#
#model = "sparse"
#base_path = "/scratch/users/yromano/CJRepo_Remote/train_machine/f2513e0586885f675fad6af849623623f7ff9a71"
#
model = "mstudent"
base_path = "/scratch/users/yromano/CJRepo_Remote/train_machine/a25684feb1e47b94e91a9bce70421a1f8a1aaec9"

#######################################
# Location of the trained machine     #
#######################################

# Location to save the results 
results_dir  = base_path + "/paper_results/" + model
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Location to store the trained machine
machine_dir  = results_dir + "/machines/"
if not os.path.exists(machine_dir):
    os.makedirs(machine_dir)
machine_loadfile = machine_dir + "/network_checkpoint.pth.tar"

#######################################
# Location to store the test results  #
#######################################

out_dir  = results_dir + "/tests/"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
out_logfile = out_dir + "fdr_power.txt"
out_plots = out_dir
####################################
# Load all required python modules #
####################################

print("Loading python modules... "); sys.stdout.flush()

import data
import matplotlib
import experiments
import numpy as np
import pandas as pd
import parameter_setting as parameters
from DeepKnockoffs import KnockoffMachine

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

print("done."); sys.stdout.flush()

np.set_printoptions(precision=4)
matplotlib.use('Agg')
print("done."); sys.stdout.flush()


#############################
# Define the Synthetic Data #
#############################

# number of features
p = 100

# Parameters of the synthestic data distribution
distribution_params = parameters.GetDistributionParams(model, p)

# Define the data generation function
DataSampler = data.DataSampler(distribution_params, standardize=True)

#############################
# Load the knockoff machine #
#############################

# Hyperparameters for training procedure, accrrding to the published manuscript
training_params = parameters.GetTrainingHyperParams(model)

# Set the parameters for training deep knockoffs
pars = dict()

# Data type, either "continuous" or "binary
pars['family'] = "continuous"
# Dimensions of data
pars['p'] = p                            
# How many times running over all training observations
pars['epochs'] = 1000
# Period between printing learning status
pars['num_replications'] = 100
# Number of variables to swap at each iteration
pars['num_swaps'] = int(0.5*p)
# Batch size
pars['batch_size'] = int(50*p)
# Size of test set
pars['test_size']  = int(0*p)
# Learning rate for main training loop
pars['lr'] = 0.01
# When to decrease learning rate
pars['lr_milestones'] = [1000]
# Optimizer, either "SGD" or "adam"
pars['optimizer'] = 'SGD'
#Width of the network (~6 layers are fixed)
pars['dim_h'] = int(10*p)
# Target correlation
pars['target_corr'] = np.zeros(p) # dummy input
# Penalty encouraging second-order knockoffs
pars['LAMBDA'] = training_params['LAMBDA']
# Decorrelation penalty hyperparameter
pars['DELTA'] = training_params['DELTA']
# Penalty for MMD distance
pars['GAMMA'] = training_params['GAMMA']
# Kernel widths for the MMD measure (uniform weights)
pars['alphas'] = [1.,2.,4.,8.,16.,32.,64.,128.]

# Load the machine
machine = KnockoffMachine(pars, machine_dir)
assert os.path.isfile(machine_loadfile), "File " + machine_loadfile + " does not exist!"
machine.load(machine_loadfile)


##############################
# Define the test parameters #
##############################

# Number of non-zero coefficients 
signal_n = 30
# Amplitude of the non-zero coefficients 
signal_amplitude_vec = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25]
# Compute the FDR as the average FDP over n_experiments
n_experiments = 1000
# Target FDR level
nominal_fdr = 0.1


test_params = parameters.GetFDRTestParams(model)


######################
# Test FDR and Power #
######################

# Initialize table of results
results = pd.DataFrame(columns=['Model','Experiment', 'Method', 'FDP', 'Power', 'Corr', \
                                'Amplitude', 'Signals', 'Alpha', 'FDR.nominal'])

print("Preparing to run %d experiments. " %(n_experiments))
print("Detailed results will be saved in: %s " %(out_logfile))
print("Plots will be saved in directory: %s " %(out_plots))

sys.stdout.flush()

for amp_id in range(len(signal_amplitude_vec)):
    for exp_id in range(n_experiments):
        signal_amplitude = signal_amplitude_vec[amp_id]
        print("Running experiment %d of %d ..." %(exp_id+1, n_experiments))
        sys.stdout.flush()
        # Sample X
        X = DataSampler.sample(test_params["n"], test=True)
        
        # Sample Y|X
        y,theta = experiments.sample_Y(X, signal_n=signal_n, signal_a=signal_amplitude)
    
        # Deep knockoffs
        Xk_m = machine.generate(X)
        W_m  = experiments.lasso_stats(X,Xk_m,y,alpha=test_params["elasticnet_alpha"],scale=False)
        selected_m, FDP_m, POW_m = experiments.select(W_m, theta, nominal_fdr=nominal_fdr)
        print("  Machine : power = %.3f, fdp = %.3f" %(POW_m, FDP_m))
        sys.stdout.flush()
        G_m = np.corrcoef(X,Xk_m,rowvar=False)
        corr_XXk_m = np.diag(G_m[0:p,p:(2*p)])
        corr_XXk_m = [1 if np.isnan(x) else x for x in corr_XXk_m]
        results = results.append({'Model':model,'Experiment':exp_id, 'Method':'deep', 'Power':POW_m, 'FDP':FDP_m, \
                                  'Corr':np.mean(corr_XXk_m), 'Amplitude':signal_amplitude, 'Signals':signal_n, \
                                  'Alpha':0.1, 'FDR.nominal':nominal_fdr}, ignore_index=True)
    
        # Save table of results
        results.to_csv(out_logfile, sep=" ", index=False)


# Plot results

# compute the average FDP and power per each signal amplitude
avg_fdr_vec = np.zeros(len(signal_amplitude_vec))
avg_power_vec = np.zeros(len(signal_amplitude_vec))
for amp_id in range(len(signal_amplitude_vec)):
    curr_results = results[results['Amplitude']==signal_amplitude_vec[amp_id]]
    avg_fdr_vec[amp_id] = np.mean(curr_results['FDP'])
    avg_power_vec[amp_id] = np.mean(curr_results['Power'])
    
# Plot FDR vs. Amplitude
plt.plot(signal_amplitude_vec, avg_fdr_vec, 'red', label = 'Machine', linewidth=4.0)
plt.plot(signal_amplitude_vec, nominal_fdr*np.ones(len(signal_amplitude_vec)), 'orange', label = 'Target FDR', linestyle="--" ,linewidth=4.0)
plt.xlabel("Amplitude")
plt.ylabel("FDP")
plt.legend(loc=2)
plt.ylim(ymin=0, ymax=0.5)
plt.grid(True)
plt.tight_layout()
plt.savefig(out_plots + 'fdr.png', dpi=300)
plt.show()


# Plot Power vs. Amplitude
plt.clf(); plt.plot(signal_amplitude_vec, avg_power_vec, 'red', linewidth=4.0)
plt.xlabel("Amplitude")
plt.ylabel("Power")
plt.ylim(ymin=0, ymax=1)
plt.grid(True)
plt.tight_layout()
plt.savefig(out_plots + '_power.png', dpi=300)
plt.show()


