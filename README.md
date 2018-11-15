# DeepKnockoffs

#################################
#################################
#         Dependencies          #
#################################
#################################

numpy=1.14.2
lapack
scipy=1.1.0
pytorch=0.4.1
sklearn=0.19.1
cvxpy=1.0.10
cvxopt=1.2.0
glmnet=0.2.0 (use glmnet_python package)
pandas=0.23.4

#################################
#################################
#      Installation Guide       #
#################################
#################################

$ cd Local/Path/deepknockoffs/DeepKnockoffs
$ python setup.py install --user

#################################
#################################
#           Examples            #
#################################
#################################

Basic experiment:

Run a toy example on a small dataset and plot second-order diagnostics.
The samples follow Gaussian AR(1) model.

$ cd Local/Path/deepknockoffs/examples
$ python toy_example.py


Advanced experiments:

The script train_machine.py fits a machine to the distributions defined in
Section 6 in the manuscript and save the result to a local directory.
The distributions include: "Gaussian model", "Gaussian mixture model",
"Multivariate Student’s t-distribution", and "Sparse Gaussian variables".

The script test_machine.py loads a pre-trained machine and estimates the FDR and
Power in a controlled synthetic setting, see Section 6 in the manuscript.
Detailed results and plots are saved to a local directory.
