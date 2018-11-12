import numpy as np
import cvxpy as cvx
from scipy import linalg
import warnings
from glmnet import ElasticNet

def kfilter(W,q=0.1):
    t = np.insert(np.abs(W[W!=0]),0,0)
    t = np.sort(t)
    ratio = np.zeros(len(t));
    for i in range(len(t)):
        ratio[i] = (1 + np.sum(W <= -t[i])) / np.maximum(1.0, np.sum(W >= t[i]))
        
    index = np.where(ratio <= q)[0]
    if len(index)==0:
        thresh = float('inf')
    else:
        thresh = t[index[0]]
       
    return thresh

def RidgeCoefDiff(y, X, Xk, q, n_splits = 3):
    
    p = X.shape[1]
    
    X_input = np.concatenate((X,Xk),1)
        
    m = ElasticNet(alpha=0,n_splits=n_splits)        
    m = m.fit(X_input, y)
        
    W = np.abs(m.coef_[0:p]) - np.abs(m.coef_[p:2*p])
    thresh = knockFilter(W,q=q)
    S = np.argwhere(W>=thresh)
    
    return S

def LassoCoefDiff(y, X, Xk, q, n_splits = 3):
    
    p = X.shape[1]
    
    X_input = np.concatenate((X,Xk),1)
        
    m = ElasticNet(alpha=1,n_splits=n_splits)        
    m = m.fit(X_input, y)
        
    W = np.abs(m.coef_[0:p]) - np.abs(m.coef_[p:2*p])
    thresh = knockFilter(W,q=q)
    S = np.argwhere(W>=thresh)
    
    return S

def OLSCoefDiff(y, X, Xk, q):
    
    p = X.shape[1]
    
    X_input = np.concatenate((X,Xk),1)
    coef = np.linalg.lstsq(X_input, y, rcond = None)[0]
    W = np.abs(coef[0:p]) - np.abs(coef[p:2*p])
    thresh = knockFilter(W,q=q)
    S = np.argwhere(W>=thresh)
    
    return S
