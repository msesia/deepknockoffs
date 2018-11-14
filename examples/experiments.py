import numpy as np
# IOS users may find https://github.com/bbalasub1/glmnet_python/issues/13  
# useful to resolve the error: :GLMnet.so: unknown file type"
#import glmnet_python
#from glmnet import glmnet
#from cvglmnet import cvglmnet
#from cvglmnetCoef import cvglmnetCoef

# Alternative solution for conda users is by installing glmnet:
# conda install -c conda-forge glmnet 
from glmnet import ElasticNet

from sklearn import preprocessing
from DeepKnockoffs import knockoffs

def sample_Y(X, signal_n=20, signal_a=10.0):
    n,p = X.shape
    beta = np.zeros((p,1))
    beta_nonzero = np.random.choice(p, signal_n)
    beta[beta_nonzero,0] = (2*np.random.choice(2,signal_n)-1) * signal_a / np.sqrt(n)
    y = np.dot(X,beta) + np.random.normal(size=(n,1))
    return y,beta

#def lasso_stats(X,Xk,y,alpha=0.1,scale=True):
#    X  = X.astype("float")
#    Xk = Xk.astype("float")
#    p = X.shape[1]
#    if scale:
#        X_concat = preprocessing.scale(np.concatenate((X,Xk),1))
#    else:
#        X_concat = np.concatenate((X,Xk),1)
#    cols_order = np.random.choice(X_concat.shape[1],X_concat.shape[1],replace=False)
#    cvfit = cvglmnet(x=X_concat[:,cols_order].copy(), y=y.copy(), family='gaussian', alpha=alpha)
#    Z = np.zeros((2*p,))
#    Z[cols_order] = cvglmnetCoef(cvfit, s = 'lambda_min').squeeze()[1:]
#    W = np.abs(Z[0:p]) - np.abs(Z[p:(2*p)])
#    return(W.squeeze())


def lasso_stats(X,Xk,y,alpha=0.1,scale=True):
    X  = X.astype("float")
    Xk = Xk.astype("float")
    p = X.shape[1]
    if scale:
        X_concat = preprocessing.scale(np.concatenate((X,Xk),1))
    else:
        X_concat = np.concatenate((X,Xk),1)
        
    Z = ElasticNet(alpha=alpha,n_splits=10)        
    Z = Z.fit(X_concat, y)
    W = np.abs(Z.coef_[0:p]) - np.abs(Z.coef_[p:2*p])
    
    return(W.squeeze())

def select(W, beta, nominal_fdr=0.1,offset=1):
    W_threshold = knockoffs.kfilter(W, q=nominal_fdr, threshold=offset)
    selected = np.where(W >= W_threshold)[0]
    nonzero = np.where(beta!=0)[0]
    TP = len(np.intersect1d(selected, nonzero))
    FP = len(selected) - TP
    FDP = FP / max(TP+FP,1.0)
    POW = TP / max(len(nonzero),1.0)
    return selected, FDP, POW
