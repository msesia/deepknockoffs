import torch
import torch.nn as nn
import torch.optim as optim
import os, sys, math, pdb
import numpy as np
import scipy.stats as stats
from DeepKnockoffs.mmd import mix_rbf_mmd2_loss, KL_loss
np.warnings.filterwarnings('ignore')
import pandas as pd
from torch.nn import functional as tf

def covariance_diff_unbiased(Z1, Z2, offset=0, scale=1.0):
    """
    Unbiased estimate of the squared Frobenius 
    norm difference of Sigma(Z1) and Sigma(Z2)
    """
    n1 = Z1.shape[0]
    n2 = Z2.shape[0]
    ## Center the data
    Z1 = Z1 - Z1.mean(0)
    Z2 = Z2 - Z2.mean(0)
    # Estimate the trace of Sigma1^2
    ZZ1 = torch.mm(Z1, Z1.t())
    A1 = (ZZ1-torch.diag(torch.diag(ZZ1))).pow(2).mean() * n1 / (n1-1.0)
    # Estimate the trace of Sigma2^2
    ZZ2 = torch.mm(Z2, Z2.t())
    A2 = (ZZ2-torch.diag(torch.diag(ZZ2))).pow(2).mean() * n2 / (n2-1.0)
    # Estimate  the trace of Sigma1 * Sigma2
    C = torch.mm(Z1,Z2.t()).pow(2).mean()
    # Compute statistic
    T = A1 + A2 - 2.0 * C
    # Make the loss function positive
    return torch.abs(T)/scale, offset/scale
    #return tf.relu(T+offset)/scale, offset/scale
    #return tf.relu(T)/scale, 0

def covariance_diff_biased(X, Xk, SigmaHat, Mask, scale=1.0):
    # Center X,Xk
    mX  = X  - torch.mean(X,0,keepdim=True)
    mXk = Xk - torch.mean(Xk,0,keepdim=True)
    # Compute covariance matrices
    SXkXk = torch.mm(torch.t(mXk),mXk)/mXk.shape[0]
    SXXk  = torch.mm(torch.t(mX),mXk)/mXk.shape[0]

    # Compute loss
    T  = (SigmaHat-SXkXk).pow(2).sum() / scale
    T += (Mask*(SigmaHat-SXXk)).pow(2).sum() / scale
    return T

def create_checkpoint_name(pars):
    checkpoint_name = 'net'
    for key, value in pars.items():
        checkpoint_name += '_' + key
        if key == 'alphas':
            for i in range(len(pars['alphas'])):
                checkpoint_name += '_' + str(pars['alphas'][i])
        else:
            checkpoint_name += '_' + str(value)
    return checkpoint_name

def train_net_helper(trainset, pars, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    checkpoint_name = 'net'
    for key, value in pars.items():
        checkpoint_name += '_' + key
        if key == 'alphas':
            for i in range(len(pars['alphas'])):
                checkpoint_name += '_' + str(pars['alphas'][i])
        else:
            checkpoint_name += '_' + str(value)

    full_checkpoint_name = checkpoint_path + checkpoint_name + '_checkpoint.pth.tar'

    deep = KnockoffMachine(pars = pars,
                           checkpoint_name = full_checkpoint_name)

    resume = False
    if os.path.isfile(full_checkpoint_name):
        resume = True

    deep.train(trainset, resume)

def save_checkpoint(state, filename):
    # keep the previous model
    if os.path.isfile(filename):
        os.rename(filename, filename + '_prev.pth.tar')
    # save new model
    torch.save(state, filename)

def gen_batches(n_samples, batch_size, n_reps):
    """ Divide input data into batches.
    :param data: input data
    :param batch_size: size of each batch
    :return: data divided into batches
    """
    batches = []
    for rep_id in range(n_reps):
        idx = np.random.permutation(n_samples)
        for i in range(0, math.floor(n_samples/batch_size)*batch_size, batch_size):
            window = np.arange(i,i+batch_size)
            new_batch = idx[window]
            batches += [new_batch]
    return(batches)

class Net(nn.Module):
    def __init__(self, p, dim_h, family="continuous"):
        super(Net, self).__init__()

        self.p = p
        self.dim_h = dim_h
        if (family=="continous"):
            self.main = nn.Sequential(
                nn.Linear(2*self.p, self.dim_h, bias=False),
                nn.BatchNorm1d(self.dim_h),
                nn.PReLU(),
                nn.Linear(self.dim_h, self.dim_h, bias=False),
                nn.BatchNorm1d(self.dim_h),
                nn.PReLU(),
                nn.Linear(self.dim_h, self.dim_h, bias=False),
                nn.BatchNorm1d(self.dim_h),
                nn.PReLU(),
                nn.Linear(self.dim_h, self.dim_h, bias=False),
                nn.BatchNorm1d(self.dim_h),
                nn.PReLU(),
                nn.Linear(self.dim_h, self.dim_h, bias=False),
                nn.BatchNorm1d(self.dim_h),
                nn.PReLU(),
                nn.Linear(self.dim_h, self.dim_h, bias=False),
                nn.BatchNorm1d(self.dim_h),
                nn.PReLU(),
                nn.Linear(self.dim_h, self.p),
            )
        elif (family=="binary"):
            self.main = nn.Sequential(
                nn.Linear(2*self.p, self.dim_h, bias=False),
                nn.BatchNorm1d(self.dim_h, eps=1e-02),
                nn.PReLU(),
                nn.Linear(self.dim_h, self.dim_h, bias=False),
                nn.BatchNorm1d(self.dim_h, eps=1e-02),
                nn.PReLU(),
                nn.Linear(self.dim_h, self.dim_h, bias=False),
                nn.BatchNorm1d(self.dim_h, eps=1e-02),
                nn.PReLU(),
                nn.Linear(self.dim_h, self.dim_h, bias=False),
                nn.BatchNorm1d(self.dim_h, eps=1e-02),
                nn.PReLU(),
                nn.Linear(self.dim_h, self.dim_h, bias=False),
                nn.BatchNorm1d(self.dim_h, eps=1e-02),
                nn.PReLU(),
                nn.Linear(self.dim_h, self.dim_h, bias=False),
                nn.BatchNorm1d(self.dim_h, eps=1e-02),
                nn.PReLU(),
                nn.Linear(self.dim_h, self.p),
                nn.Sigmoid(),
                nn.BatchNorm1d(self.p, eps=1e-02),
            )
        else:
            sys.exit("Error: unknown family");

    def forward(self, x, noise):
        x_cat = torch.cat((x,noise),1)
        x_cat[:,0::2] = x
        x_cat[:,1::2] = noise
        return self.main(x_cat)

def norm(X, p=2):
    if(p==np.inf):
        return(torch.max(torch.abs(X)))
    else:
        return(torch.norm(X,p))

class KnockoffMachine:
    def __init__(self, pars, checkpoint_name, logs_name):
        # architecture
        self.p = pars['p']
        self.dim_h = pars['dim_h']
        self.family = pars['family']

        # training
        self.epochs = pars['epochs']
        self.num_replications = pars['num_replications']
        self.batch_size = pars['batch_size']
        self.test_size = pars['test_size']
        self.lr = pars['lr']
        self.lr_milestones = pars['lr_milestones']
        self.optimizer = pars['optimizer']

        # parameters
        self.alphas = pars['alphas']
        self.num_swaps = pars['num_swaps']
        self.target_corr = torch.from_numpy(pars['target_corr']).float()
        self.BETA = pars['BETA']
        self.GAMMA_K = pars['GAMMA_K']
        self.GAMMA_S = pars['GAMMA_S']
        self.DELTA = pars['DELTA']
        self.noise_std = pars['noise_std']
        self.dim_noise = pars['dim_noise']
        self.method = pars['method']
        if self.method == 'kl':
            self.matching_loss = KL_loss
            self.matching_param = []
        elif self.method == 'mmd':
            self.matching_loss = mix_rbf_mmd2_loss
            self.matching_param = self.alphas
        else:
            raise        

        # Normalize learning rate to avoid numerical issues
        self.lr = self.lr / np.max([self.BETA, self.GAMMA_K, self.GAMMA_S, self.DELTA, 1.0])

        self.pars = pars
        self.checkpoint_name = checkpoint_name + "_checkpoint.pth.tar"
        self.best_checkpoint_name = checkpoint_name + "_best.pth.tar"
        self.logs_name = logs_name
        self.resume_epoch = 0
        self.net = Net(self.p, self.dim_h, family=self.family)

    def estimate_entropy(self, X, noise):
        idx = np.random.choice(X.shape[0],1)
        X = X[idx].repeat(1,X.shape[0]).view(-1, X.shape[1])
        Xk = self.net(X, self.noise_std*noise.normal_())
        Xk = Xk.data.cpu().numpy()
        cent = 0.0
        for j in range(X.shape[1]):
            histogram = np.histogram(Xk[:,j], bins=50, range=(-5,5), density=True)[0]
            cent += stats.entropy(histogram, base=2)
        return cent/X.shape[1]

    def compute_diagnostics(self, X, Xk, noise, test=False):
        # Initialize dictionary of diagnostics
        diagnostics = dict()
        if test:
            diagnostics["Data"] = "test"
        else:
            diagnostics["Data"] = "train"

        ##############################
        # Second-order moments
        ##############################
        # Difference in means
        D_mean = X.mean(0) - Xk.mean(0)
        D_mean = (D_mean*D_mean).mean()
        diagnostics["Mean"] = D_mean.data.cpu().item()

        # Center and scale X, Xk
        mX = X - torch.mean(X,0,keepdim=True)
        mXk = Xk - torch.mean(Xk,0,keepdim=True)
        scaleX  = (mX*mX).mean(0,keepdim=True)
        scaleXk = (mXk*mXk).mean(0,keepdim=True)
        
        # Correlation between X and Xk
        scaleX[scaleX==0] = 1.0   # Prevent division by 0
        scaleXk[scaleXk==0] = 1.0 # Prevent division by 0
        mXs  = mX  / torch.sqrt(scaleX)
        mXks = mXk / torch.sqrt(scaleXk)
        corr = (mXs*mXks).mean()
        diagnostics["Corr-diag"] = corr.data.cpu().item()

        # Corr(Xk,Xk)
        Sigma = torch.mm(torch.t(mXs),mXs)/mXs.shape[0]
        Sigma_ko = torch.mm(torch.t(mXks),mXks)/mXk.shape[0]
        DK_i = norm(Sigma_ko-Sigma, p=np.inf)
        DK_2 = norm(Sigma_ko-Sigma) / norm(Sigma)
        diagnostics["Corr-K"] = DK_2.data.cpu().item()

        # Corr(Xk,X) excluding the diagonal elements
        SigIntra_est = torch.mm(torch.t(mXks),mXs)/mXk.shape[0]
        DS_i = norm(self.Mask*(SigIntra_est-Sigma), p=np.inf)
        DS_2 = norm(self.Mask*(SigIntra_est-Sigma)) / norm(Sigma)
        diagnostics["Corr-S"] = DS_2.data.cpu().item()

        ##############################
        # Loss function
        ##############################
        _, loss_display, mmd_k, mmd_s = self.loss(X[:noise.shape[0]], Xk[:noise.shape[0]], test=True)
        diagnostics["Loss"]  = loss_display.data.cpu().item()
        diagnostics["MMD-K"] = mmd_k.data.cpu().item()
        diagnostics["MMD-S"] = mmd_s.data.cpu().item()

        ##############################
        # Conditional entropy
        ##############################
        cond_entropy = self.estimate_entropy(X[:noise.shape[0]], noise)
        diagnostics["Entropy"] = cond_entropy

        # Return dictionary of diagnostics
        return diagnostics

    def loss(self, X, Xk, test=False):
        # Divide the observations into two batches
        n = int(X.shape[0]/2)
        X1,Xk1 = X[:n], Xk[:n]
        X2,Xk2 = X[n:(2*n)], Xk[n:(2*n)]
        
        # Joint variables
        Z1 = torch.cat((X1,Xk1),1)
        Z2 = torch.cat((Xk2,X2),1)
        Z3 = torch.cat((X2,Xk2),1).clone()
        swap_inds = torch.randperm(self.p)[0:self.num_swaps]
        Z3[:,swap_inds] = Xk2[:,swap_inds]
        Z3[:,swap_inds+self.p] = X2[:,swap_inds]

        # (X,Xk) =d (Xk,X)
        mmd_k = 0.0
        if(self.GAMMA_K>0 or test):
            mmd_k = self.matching_loss(Z1, Z2, self.matching_param)

        # (X,Xk) =d (X,Xk)_s
        mmd_s = 0.0
        if(self.GAMMA_S>0 or test):
            mmd_s = self.matching_loss(Z1, Z3, self.matching_param)

        # Match first two moments
        cov_intra_loss = 0.0
        if self.DELTA>0:
            # First moment
            D_mean = X.mean(0) - Xk.mean(0)
            loss_1m = D_mean.pow(2).sum()
            # Second moments
            loss_2m = covariance_diff_biased(X, Xk, self.SigmaHat, self.Mask, scale=self.Sigma_norm)
            #loss_2m_scale = 4.0*self.Sigma_norm
            #loss_2m_k = covariance_diff_unbiased(Z1,Z2,scale=loss_2m_scale)
            #loss_2m_s = covariance_diff_unbiased(Z1,Z3,scale=loss_2m_scale)
            #loss_2m = loss_2m_k + loss_2m_s
            # Combine moments
            loss_moments = loss_1m + loss_2m
        else:
            loss_moments = 0.0

        # Penalize correlations between variables and knockoffs
        loss_corr = 0.0
        if self.BETA>0:
            # Center X,Xk
            mX  = X  - torch.mean(X,0,keepdim=True)
            mXk = Xk - torch.mean(Xk,0,keepdim=True)
            # Correlation between X and Xk
            eps = 1e-3
            scaleX  = mX.pow(2).mean(0,keepdim=True)
            scaleXk = mXk.pow(2).mean(0,keepdim=True)
            mXs  = mX / (eps+torch.sqrt(scaleX))
            mXks = mXk / (eps+torch.sqrt(scaleXk))
            corr_XXk = (mXs*mXks).mean(0)
            #loss_corr = torch.clamp(corr_XXk.abs()-self.target_corr.abs(), min=0).pow(2).mean()
            loss_corr = (corr_XXk-self.target_corr).pow(2).mean()

        # Combine the loss functions
        loss = self.GAMMA_K*mmd_k + self.GAMMA_S*mmd_s + self.DELTA*loss_moments + self.BETA*loss_corr
        loss_display = loss
        return loss, loss_display, mmd_k, mmd_s


    def train(self, X_in, resume = False):
        # Divide data into training/test set
        X = torch.from_numpy(X_in[self.test_size:]).float()
        if(self.test_size>0):
            X_test = torch.from_numpy(X_in[:self.test_size]).float()
        else:
            X_test = torch.zeros(0, self.p)

        # used to display compute statistics
        self.SigmaHat = np.cov(X,rowvar=False)
        self.SigmaHat = torch.from_numpy(self.SigmaHat).float()
        self.Mask = torch.ones(self.p, self.p) - torch.eye(self.p)

        # allocate a matrix for the noise realization
        noise = torch.zeros(self.batch_size,self.dim_noise)
        noise_test = torch.zeros(X_test.shape[0],self.dim_noise)
        use_cuda = torch.cuda.is_available()

        if resume == True:
            self.load(self.checkpoint_name)
            self.net.train()
        else:
            self.net.train()
            if(self.optimizer=="SGD"):
                self.net_optim = optim.SGD(self.net.parameters(), lr = self.lr, momentum=0.9)
                self.net_sched = optim.lr_scheduler.MultiStepLR(self.net_optim, gamma=0.1,
                                                                milestones=self.lr_milestones)
            else:
                self.net_optim = optim.Adam(self.net.parameters())
                self.net_sched = optim.lr_scheduler.MultiStepLR(self.net_optim, gamma=0.1,
                                                                milestones=self.lr_milestones)

        if self.method == 'kl':
            self.matching_param = torch.eye(self.batch_size,self.batch_size)
            if use_cuda:
                self.matching_param = matching_param.cuda()
        elif self.method == 'mmd':
            self.matching_param = self.alphas

        if use_cuda:
            self.SigmaHat = self.SigmaHat.cuda()
            self.Mask = self.Mask.cuda()
            self.net = self.net.cuda()
            X      = X.cuda()
            X_test = X_test.cuda()
            noise = noise.cuda()
            noise_test = noise_test.cuda()
            self.target_corr = self.target_corr.cuda()

        Xk = 0*X
        self.Sigma_norm = self.SigmaHat.pow(2).sum()
        self.Sigma_norm_cross = (self.Mask*self.SigmaHat).pow(2).sum()

        # Store diagnostics
        diagnostics = pd.DataFrame()
        losses_test = []

        for epoch in range(self.resume_epoch, self.epochs):
            self.net.train() # Prepare for training phase
            self.net_sched.step()
            batches = gen_batches(X.size(0), self.batch_size, self.num_replications)

            losses = []
            losses_dist_s = []
            losses_dist_k = []

            for batch in batches:
                # Extract data for this batch
                X_batch  = X[batch,:]

                self.net_optim.zero_grad()

                # Run the network
                Xk_batch = self.net(X_batch, self.noise_std*noise.normal_())

                # Compute the loss function
                loss, loss_display, mmd_k, mmd_s = self.loss(X_batch, Xk_batch)

                # Compute the gradient
                loss.backward()

                # Minimize the loss
                self.net_optim.step()

                # Save history
                losses.append(loss_display.data.cpu().item())
                if self.GAMMA_S>0:
                    losses_dist_s.append(mmd_s.data.cpu().item())
                if self.GAMMA_K>0:
                    losses_dist_k.append(mmd_k.data.cpu().item())

                # Save the knockoffs
                Xk[batch, :] = Xk_batch.data

            ##############################
            # Compute diagnostics
            ##############################
            self.net.eval() # Prepare for testing phase
            # Training data
            diagnostics_train = self.compute_diagnostics(X, Xk, noise, test=False)
            diagnostics_train["Loss"]  = np.mean(losses)
            if(self.GAMMA_K>0 and self.GAMMA_S>0):
                diagnostics_train["MMD-K"] = np.mean(losses_dist_k)
                diagnostics_train["MMD-S"] = np.mean(losses_dist_s)
            diagnostics_train["Epoch"] = epoch
            diagnostics = diagnostics.append(diagnostics_train, ignore_index=True)
            # Test data
            if(self.test_size>0):
                Xk_test = self.net(X_test, self.noise_std*noise_test.normal_())
                diagnostics_test = self.compute_diagnostics(X_test, Xk_test, noise_test, test=True)
            else:
                diagnostics_test = {key:np.nan for key in diagnostics_train.keys()}
            diagnostics_test["Epoch"] = epoch
            diagnostics = diagnostics.append(diagnostics_test, ignore_index=True)

            # If the test loss is at a minimum, save the machine
            losses_test.append(diagnostics_test["Loss"])
            if((self.test_size>0) and (diagnostics_test["Loss"] == np.min(losses_test))):
                best_machine = True
                save_checkpoint({
                    'epochs': epoch+1,
                    'pars'  : self.pars,
                    'state_dict': self.net.state_dict(),
                    'optimizer' : self.net_optim.state_dict(),
                    'scheduler' : self.net_sched.state_dict(),
                }, self.best_checkpoint_name)
            else:
                best_machine = False

            ##############################
            # Show progress
            ##############################
            print("[%4d/%4d], Lte: %.4f, L: %.4f, H: %.4f" %
                  (epoch + 1, self.epochs, diagnostics_test["Loss"], diagnostics_train["Loss"],
                   diagnostics_test["Entropy"]), end=", ")
            print("C: (%.3f,%.3f), XkXk: (%.3f,%.3f), XXk: (%.3f,%.3f)" %
                  (diagnostics_train["Corr-diag"], diagnostics_test["Corr-diag"],
                   diagnostics_train["Corr-K"], diagnostics_test["Corr-K"],
                   diagnostics_train["Corr-S"], diagnostics_test["Corr-S"]), end=", ")
            print("MMDS: (%.4f,%.4f), MMDK: (%.4f,%.4f)" %
                  (diagnostics_train["MMD-K"], diagnostics_test["MMD-K"],
                   diagnostics_train["MMD-S"], diagnostics_test["MMD-S"]), end="")
            if best_machine:
                print(" *", end="")
            print("")
            sys.stdout.flush()

            # Save diagnostics file
            diagnostics.to_csv(self.logs_name, sep=" ", index=False)

            save_checkpoint({
                'epochs': epoch+1,
                'pars'  : self.pars,
                'state_dict': self.net.state_dict(),
                'optimizer' : self.net_optim.state_dict(),
                'scheduler' : self.net_sched.state_dict(),
            }, self.checkpoint_name)

    def load(self, filename):
        flag = 1
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            sys.stdout.flush()
            try:
                checkpoint = torch.load(filename, map_location='cpu')
            except:
                print("error loading saved model, trying the previous version")
                sys.stdout.flush()
                flag = 0

            if flag == 0:
                try:
                    checkpoint = torch.load(filename + '_prev.pth.tar', map_location='cpu')
                    flag = 1
                except:
                    print("error loading prev model, starting from scratch")
                    sys.stdout.flush()
                    flag = 0
        else:
            print("=> no checkpoint found at '{}'".format(filename))
            sys.stdout.flush()
            flag = 0

        if flag == 1:
                self.net.load_state_dict(checkpoint['state_dict'])
                if torch.cuda.is_available():
                    self.net = self.net.cuda()

                self.net_optim = optim.SGD(self.net.parameters(), lr = self.lr, momentum=0.9)
                self.net_optim.load_state_dict(checkpoint['optimizer'])
                self.net_sched = optim.lr_scheduler.MultiStepLR(self.net_optim, gamma=0.1,
                                                                milestones=self.lr_milestones)
                #self.net_sched.load_state_dict(checkpoint['scheduler'])
                self.resume_epoch = checkpoint['epochs']

                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(filename, checkpoint['epochs']))
                sys.stdout.flush()
        else:
            self.net.train()
            self.net_optim = optim.SGD(self.net.parameters(), lr = self.lr, momentum=0.9)
            self.net_sched = optim.lr_scheduler.MultiStepLR(self.net_optim, gamma=0.1,
                                                            milestones=self.lr_milestones)

            self.resume_epoch = 0

    def generate(self, X_in):
        X = torch.from_numpy(X_in).float()
        self.net = self.net.cpu()
        self.net.eval()
        Xk = self.net(X, self.noise_std*torch.randn(X.size(0),self.dim_noise))
        Xk = Xk.data.cpu().numpy()
        return Xk
