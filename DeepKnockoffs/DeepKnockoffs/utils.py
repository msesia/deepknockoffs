import torch
import numpy as np
from DeepKnockoffs.mmd import mix_rbf_mmd2
 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#from torch_two_sample.torch_two_sample import statistics_diff as tests
#from torch_two_sample.torch_two_sample import statistics_nondiff as nondiff_tests
import scipy.stats as stats

from random import shuffle


def gen_batches(n_samples, batch_size, n_reps):
    """ Divide input data into batches.
    :param data: input data
    :param batch_size: size of each batch
    :return: data divided into batches
    """
    batches = []
    for rep_id in range(n_reps):
        idx = np.arange(0, n_samples)
        shuffle(idx)
        for i in range(0, n_samples, batch_size):
            new_batch = idx[np.arange(i,i+batch_size)]
            #new_batch = sorted(new_batch)
            batches += [new_batch]
    return(batches)

def PlotScatterHelper(X,Y, offset):
    fig, ax = plt.subplots()
    
    ax.set_xlim([-0.2, 1.1])
    ax.set_ylim([-0.2, 1.1])
    
    for i in range(offset,X.shape[0]-1):
        ax.scatter(X.diagonal(i),Y.diagonal(i), alpha=0.2)
    
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    
    for i in range(offset,X.shape[0]-1):
        meanValX = np.mean(X.diagonal(i))
        meanValXk = np.mean(Y.diagonal(i))
        ax.plot([meanValX, meanValX],lims, 'k-', alpha=0.2, zorder=0)
        if i==0:
            color = 'r-'
            alpha = 1
        else:
            color = 'k-'
            alpha = 0.2
        ax.plot(lims, [meanValXk, meanValXk], color, alpha=alpha, zorder=0)
    
    # now plot both limits against eachother
    ax.plot(lims, lims, 'k-', dashes=[2,2], alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    return fig, ax
    
def ScatterOriginality(X, Xk, filename='', directory='./', show=False):
    XX = np.corrcoef(X.T)
    XkXk = np.corrcoef(Xk.T)
    fig, ax = PlotScatterHelper(XX, XkXk, offset=0)
    plt.title(filename + ': Originality')
    plt.xlabel(r'$\Sigma_{\mathbf{X},\mathbf{X}}(ij)$')
    plt.ylabel(r'$\Sigma_{\tilde{\mathbf{X}},\tilde{\mathbf{X}}}(ij)$')
    plt.savefig(directory + filename + '_orig.png', dpi=300) 
    if show:
        plt.show()
        
def ScatterExchengability(X, Xk, filename='', directory='./', show=False):
    p = X.shape[1]
    G = np.corrcoef(X.T, Xk.T)
    XX  = G[:p,:p]
    XXk = G[:p,p:(2*p)]

    fig, ax = PlotScatterHelper(XX, XXk, offset=0)
    plt.title(filename + ': Exchengability & Power')
    plt.xlabel(r'$\Sigma_{\mathbf{X},\mathbf{X}}(ij)$')
    plt.ylabel(r'$\Sigma_{{\mathbf{X}},\tilde{\mathbf{X}}}(ij)$')
    plt.savefig(directory + filename + '_exch.png', dpi=300)
    if show:
        plt.show()

def PlotQQ(X):
    stats.probplot(X.flatten(), dist="norm", plot=plt)
    plt.title("Normal Q-Q plot")
    
def PlotHist(X, Xk, num_bins = 1000, filename = '', directory='./', show=False):
    X = X.flatten()
    Xk = Xk.flatten()
    
    minval = np.min(np.concatenate((X,Xk)))
    maxval = np.min(np.concatenate((X,Xk)))
    val = np.maximum(np.abs(minval),np.abs(maxval))
    bins = np.linspace(-val, val, num_bins)

    plt.hist(X.flatten(), bins, alpha=0.5, label=r'$\mathbf{X}$')
    plt.hist(Xk.flatten(), bins, alpha=0.5, label=r'$\mathbf{\tilde{X}}$')

    plt.legend(loc='upper right')
    plt.title(filename + ': Histogram')
    plt.savefig(directory + filename + '_hist.png', dpi=300)
    if show:
        plt.show()
