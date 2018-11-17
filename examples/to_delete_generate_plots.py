#!/usr/bin/python
import os
import sys


# Data distribution:
# "gaussian": AR(1) model, defined by a correlation coeffecient parameter
# "gmm"     : mixture of 3 AR(1) models with different correlation coefficients
# "sparse"  : heavy tailed and weakly correlated distribusion
# "mstudent": heavy tailed correlated multivariate-t distribution
#model = "gaussian"
#base_path = "/media/sf_results/CJ_get_tmp/0ed13f755e24f3692b8c90add3618acfb84369fd"

model = "gmm"
base_path = "/media/sf_results/CJ_get_tmp/2d7995d06617d5668174c4b1736f2354938a0f5a"
#
#model = "sparse"
#base_path = "/media/sf_results/CJ_get_tmp/f2513e0586885f675fad6af849623623f7ff9a71"
#
#model = "mstudent"
#base_path = "/scratch/users/yromano/CJRepo_Remote/train_machine/0ed13f755e24f3692b8c90add3618acfb84369fd"

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

import matplotlib
import numpy as np
import pandas as pd
import parameter_setting as parameters

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
    
# Load table of results
results = pd.read_csv(out_logfile, sep=" ")


# Plot results

# compute the average FDP and power per each signal amplitude
avg_fdr_vec = np.zeros(len(signal_amplitude_vec))
avg_power_vec = np.zeros(len(signal_amplitude_vec))
for amp_id in range(len(signal_amplitude_vec)):
    curr_results = results[results['Amplitude']==signal_amplitude_vec[amp_id]]
    print(len(curr_results['FDP']))
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


