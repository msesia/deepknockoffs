import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def PlotScatterHelper(A,B):
    """ Plot the entires of a matrix A vs those of B
    :param X: a matrix
    :param Y: a matrix
    :return fig: handle to the generated scatter plot
    """
    fig, ax = plt.subplots()

    ax.set_xlim([-0.2, 1.1])
    ax.set_ylim([-0.2, 1.1])

    for i in range(0,A.shape[0]-1):
        ax.scatter(A.diagonal(i),B.diagonal(i), alpha=0.2)

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    for i in range(0,A.shape[0]-1):
        meanValA = np.mean(A.diagonal(i))
        meanValB = np.mean(B.diagonal(i))
        ax.plot([meanValA, meanValA],lims, 'k-', alpha=0.2, zorder=0)
        if i==0:
            color = 'r-'
            alpha = 1
        else:
            color = 'k-'
            alpha = 0.2
        ax.plot(lims, [meanValB, meanValB], color, alpha=alpha, zorder=0)

    # Plot both limits against each other
    ax.plot(lims, lims, 'k-', dashes=[2,2], alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    return fig

def ScatterOriginality(X, Xk, title="", directory=None, show=True):
    """ Plot the entires of Cov(Xk,Xk) vs Cov(X,X)
    :param X: input data
    :param Xk: knockoffs
    :param title: title of the figure and the filename to store the plot
    :param directory: path to store the plot
    :param show: present the figurte
    """
    XX = np.corrcoef(X.T)
    XkXk = np.corrcoef(Xk.T)
    PlotScatterHelper(XX, XkXk)
    plt.title(title + ': Originality')
    plt.xlabel(r'$\Sigma_{\mathbf{X},\mathbf{X}}(i,j)$')
    plt.ylabel(r'$\Sigma_{\tilde{\mathbf{X}},\tilde{\mathbf{X}}}(i,j)$')
    if directory is not None:
        plt.savefig(directory + title + '_orig.png', dpi=300)
    if show:
        plt.show()

def ScatterExchengability(X, Xk, title="", directory=None, show=True):
    """ Plot the entires of Cov(X,Xk) vs Cov(X,X)
    :param X: input data
    :param Xk: knockoffs
    :param filename: title of the figure and the filename to store the plot
    :param directory: path to store the plot
    :param show: present the figure
    """
    p = X.shape[1]
    G = np.corrcoef(X.T, Xk.T)
    XX  = G[:p,:p]
    XXk = G[:p,p:(2*p)]

    PlotScatterHelper(XX, XXk)
    plt.title(title + ': Exchengability & Power')
    plt.xlabel(r'$\Sigma_{\mathbf{X},\mathbf{X}}(i,j)$')
    plt.ylabel(r'$\Sigma_{{\mathbf{X}},\tilde{\mathbf{X}}}(i,j)$')
    if directory is not None:
        plt.savefig(directory + title + '_exch.png', dpi=300)
    if show:
        plt.show()
