#!/usr/bin/python
import os
import sys


# Data distribution:
# "gaussian": AR(1) model, defined by a correlation coeffecient parameter
# "gmm"     : mixture of 3 AR(1) models with different correlation coefficients
# "sparse"  : heavy tailed and weakly correlated distribusion
# "mstudent": heavy tailed correlated multivariate-t distribution
model = "gaussian"

# set to 'True' to load a pretrained machine
load_machine = False

#######################################
# Define the output folders and files #
#######################################

# Location to save the results 
results_dir  = "./paper_results/" + model
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Location to store the trained machine
machine_dir  = results_dir + "/machines/"
if not os.path.exists(machine_dir):
    os.makedirs(machine_dir)
machine_file = machine_dir + "/network"
machine_loadfile = machine_file + "_checkpoint.pth.tar"

# Location to store the logfile
logs_dir  = results_dir + "/logs/"
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
logs_file = logs_dir + "/log.txt"

# Location to store basic diagnostic plots
plt_dir  = results_dir + "/plots/"
if not os.path.exists(plt_dir):
    os.makedirs(plt_dir)
plt_file = plt_dir + "_scatter"

####################################
# Load all required python modules #
####################################

print("Loading python modules... ", end=''); sys.stdout.flush()
import data
import numpy as np
from DeepKnockoffs import utils
from DeepKnockoffs import KnockoffMachine
from DeepKnockoffs import GaussianKnockoffs
import parameter_setting as parameters
print("done."); sys.stdout.flush()

np.set_printoptions(precision=4)


###########################
# Generate Synthetic Data #
###########################

# number of features
p = 100
# number of training examples
n = 100*p

# Parameters of the synthestic data distribution
distribution_params = parameters.GetDistributionParams(model, p)

# Define the data generation function
DataSampler = data.DataSampler(distribution_params, standardize=True)

# Sample training data
print("Sampling %d training observations from %s distribution... " 
      %(n, model), end=''); sys.stdout.flush()
X_train = DataSampler.sample(n)
print("done."); sys.stdout.flush()

###################################
# Generate second-order knockoffs #
###################################

print("Solving SDP for second-order knockoffs... ", end=''); sys.stdout.flush()
SigmaHat = np.cov(X_train,rowvar=False)
second_order = GaussianKnockoffs(SigmaHat, method="sdp",mu=np.mean(X_train,0))
corr_g = (np.diag(SigmaHat) - np.diag(second_order.Ds)) / np.diag(SigmaHat)
print("done.");
print('Average second-order correlation: %.3f' %(np.mean(np.abs(corr_g))))
sys.stdout.flush()

##############################
# Train the knockoff machine #
##############################

# Hyperparameters for training procedure, accrrding to the published manuscript
training_params = parameters.GetTrainingHyperParams(model)


# Set the parameters for training deep knockoffs
pars = dict()

# Data type, either "continous" or "binary
pars['family'] = "continous"
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
pars['lr'] = 0.001
# When to decrease learning rate
pars['lr_milestones'] = [pars['epochs']]
#Width of the network (~6 layers are fixed)
pars['dim_h'] = int(10*p)
# Target correlation
pars['target_corr'] = corr_g
# Penalty encouraging second-order knockoffs
pars['LAMBDA'] = training_params['LAMBDA']
# Decorrelation penalty hyperparameter
pars['DELTA'] = training_params['DELTA']
# Penalty for MMD score
pars['GAMMA'] = training_params['GAMMA']
# Kernel widths for the MMD measure (uniform weights)
pars['alphas'] = [1.,2.,4.,8.,16.,32.,64.,128.]

# define the machine
machine = KnockoffMachine(pars, machine_file, logs_file)

# load or train a deep knockioff machine
if(load_machine==1):
    print("Loading knockoff machine...")
    sys.stdout.flush()
    assert os.path.isfile(machine_loadfile), "File " + machine_loadfile + " does not exist!"
    machine.load(machine_loadfile)
else:
    print("Fitting knockoff machine...")
    sys.stdout.flush()
    machine.train(X_train)

###################################################################
# Test and compare the knockoff machine to second-order knockoffs #
###################################################################
    
# Sample test data
print("Sampling %d test observations from %s distribution... " %(n, model), end=''); sys.stdout.flush()
X_test = DataSampler.sample(n, test=True)
print("done."); sys.stdout.flush()

# Sampling knockoffs
print("Sampling knockoffs copies... ", end=''); sys.stdout.flush()
Xk_m = machine.generate(X_test)
Xk_g = second_order.generate(X_test)
print("done."); sys.stdout.flush()

# Plot second-order diagnostics
print("Making second-order diagnostic plots... ", end=''); sys.stdout.flush()
utils.ScatterOriginality(X_test, Xk_m, title='deep', directory=plt_file)
utils.ScatterExchengability(X_test, Xk_m, title='deep', directory=plt_file)
utils.ScatterOriginality(X_test, Xk_g, title='gaussian', directory=plt_file)
utils.ScatterExchengability(X_test, Xk_g, title='gaussian', directory=plt_file)
print("done."); sys.stdout.flush()

