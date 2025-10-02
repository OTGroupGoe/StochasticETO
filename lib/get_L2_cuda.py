import numpy as np
import scipy as sp
import pandas as pd
import torch
# import ot
from misc.torchot import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import itertools
dev = torch.device('cuda')


def getDistSqrTorus(x,y):
    dim = 1
    if (x.ndim == 2):
        dim = x.shape[1]
    m = x.shape[0]
    n = y.shape[0]
    return torch.sum(((x.reshape((m,1,dim))-y.reshape((1,n,dim)) + 0.5)%1 - 0.5)**2, dim = 2)

def cost(X,Y):
    
    return getDistSqrTorus(X,Y)


def dens_gauss_shift(X, Y, shift, std, shift_prob=0.5):
    """
    X, Y same shape numpy arrays to describe point clouds
    shift = shift of second diagonal
    std = standard deviation of both diagonals
    shift_prob = probability of going to the se
    """
    X = torch.tensor(X, device = dev)
    Y = torch.tensor(Y,device = dev)
    o = torch.ceil(torch.tensor(4 * std).to(dev))
    
    a = (Y - X).reshape((*X.shape, 1))
    z = torch.arange(-o, o+1, dtype=float, device = dev).reshape((*[1 for _ in X.shape], -1))
    
    d0 = 1 / ((2 * torch.pi)**.5 * std) * torch.sum(torch.exp(-(a - z)**2 / (2 * std**2)), axis=-1)
    d1 = 1 / ((2 * torch.pi)**.5 * std) * torch.sum(torch.exp(-(a - shift - z)**2 / (2 * std**2)), axis=-1)
    
    return shift_prob * d1 + (1 - shift_prob) * d0



def sample_Gau(gen,num ,std, shift,dim = 1, shift_prob = 0.5):
    x = gen.random(num)
    shift_ind = gen.choice([0,1], p=[1-shift_prob,shift_prob],size = num)
    gau = gen.normal(0, std, size = num)
    y = x + shift_ind*shift + gau
    y = y%1
    if(dim == 1):
        return torch.tensor(x.T).to(dev)[:,None],torch.tensor(y.T).to(dev)[:,None]
    while (dim > 1):
        xx = gen.random(num)
        yy = gen.random(num)
        x = np.vstack((x, xx))
        y = np.vstack((y, yy))
        dim -= 1
    return torch.tensor(x.T).to(dev),torch.tensor(y.T).to(dev)



# N_sim = 100
def get_torus_simulation(gen, Nlist, stdL, epsL, shift, shift_prob, dim, N_sim = 100, n_sim = 500):
    df = []
    epsL = torch.sort(torch.tensor(epsL, device = dev), descending = True)[0]
    for N, std in tqdm(list(itertools.product(Nlist, stdL))):
        b = np.clip(1e10/(2*8*N**2), 1, N_sim).astype(int)
        sta_ptr = np.arange(0,N_sim, b, dtype=int)
        for ii in sta_ptr:
            bb = min(b, N_sim - ii)
            costXXL = []
            costYYL = []
            xL = []
            yL = []
            for _ in range(bb):
                x,y = sample_Gau(gen = gen, num = N, std = std, shift = shift, dim=dim, shift_prob=shift_prob)
                xL.append(x)
                yL.append(y)
                costXXL.append(cost(x,x))
                costYYL.append(cost(y,y))

            costXXL = torch.stack(costXXL).cpu()
            costYYL = torch.stack(costYYL).cpu()
            xL = torch.stack(xL)
            yL = torch.stack(yL)
            _, _, betaX = SolveEOT(costXXL.to(dev), epsL)
            _, alphaY, _ = SolveEOT(costYYL.to(dev), epsL)


            ePts = 200
            res_mcmc = []
            res_mcmc2 = []
            for _ in range(n_sim):
                x_e = torch.rand(size = (ePts, dim), device = dev)
                y_e = torch.rand(size = (ePts, dim), device = dev)
                costX = torch.stack([cost(x_e,xL[i]) for i in range(xL.shape[0])])
                costY = torch.stack([cost(y_e,yL[i]) for i in range(xL.shape[0])])
                pot_x = -epsL[None,:,None] * (torch.logsumexp((-costX[:,None,...] + betaX[:,:,None,:])/epsL[None,:,None,None], dim = -1) - torch.log(torch.tensor(N,device = dev))) 
                pot_y = -epsL[None,:,None] * (torch.logsumexp((-costY[:,None,...] + alphaY[:,:,None,:])/epsL[None,:,None,None], dim = -1) - torch.log(torch.tensor(N,device = dev))) 
                KX = torch.exp((-costX[:,None,...] + betaX[...,None,:] + pot_x[...,None])/epsL[None,:,None,None])
                KY = torch.exp((-costY[:,None,...] + alphaY[...,None,:] + pot_y[...,None])/epsL[None,:,None,None]).permute(0,1,3,2)
                Res = (KX@KY/N).permute(0,1,3,2)
                xgrid, ygrid = torch.meshgrid(x_e[:,0], y_e[:,0], indexing='xy')
                dens = dens_gauss_shift(xgrid, ygrid, shift, std, shift_prob)
                dens2 = []
                for e in epsL:
                    dens2.append(dens_gauss_shift(xgrid, ygrid, shift, (std**2 + e)**.5, shift_prob))
                dens2 = torch.stack(dens2)
                interres = ((Res - dens[None,None,:,:])**2).mean([2,3])
                interres2 = ((Res - dens2[None,...])**2).mean([2,3])
                res_mcmc.append(interres.cpu())
                res_mcmc2.append(interres2.cpu())
            otpt = torch.stack(res_mcmc).mean(0)**.5 #b,3
            otpt2 = torch.stack(res_mcmc2).mean(0)**.5 #b,3
            
            for i, sig in enumerate(epsL):
                df.append(pd.DataFrame({'N' : N, 'dim' : dim, 'std' : std, 'jump' : shift, 'jump_prob' : shift_prob , 've' : sig.item(), 'L2' : otpt[:,i], 'L2_true' : otpt2[:,i]}))
    return pd.concat(df)



def get_Bais(std:float,jump:float,jump_prob:float,ve:float,Res = 500):
    """
    Return L2 distance between true density and blured true density.
    std = standard deviation for distance between two species
    jump = shift distance
    jump_prob = shift probability
    ve = sinkhorn regulariser
    Res = resolution, default 500.    
    """
    x_e = y_e = np.linspace(0,1,Res,endpoint = False)
    M2 = dens_gauss_shift(*np.meshgrid(y_e,x_e), shift = jump, std = (std**2 + ve)**.5, shift_prob=jump_prob)
    M1 = dens_gauss_shift(*np.meshgrid(y_e,x_e), shift = jump, std =std, shift_prob=jump_prob)

    return (torch.sum((M1-M2)**2))**.5/Res