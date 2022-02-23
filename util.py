import torch
import numpy as np
from geomloss import SamplesLoss



def wasserstein_torch(X,t ):
    """ Returns the Wasserstein distance between treatment groups """

    it = torch.where(t==1)[0]
    ic = torch.where(t==0)[0]
    Xc = X[ic]
    Xt = X[it]
    samples_loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.05, backend="tensorized")
    imbalance_loss = samples_loss(Xt, Xc)

    return imbalance_loss

def mmd2_torch(X, t):
    it = torch.where(t==1)[0]
    ic = torch.where(t==0)[0]
    Xc = X[ic]
    Xt = X[it]

    samples_loss = SamplesLoss(loss="energy", p=2, blur=0.05, backend="tensorized")
    imbalance_loss = samples_loss(Xt, Xc)

    return imbalance_loss
