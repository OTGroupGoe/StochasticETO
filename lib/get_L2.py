import numpy as np
import scipy
# import ot
from lib.SinkhornNP import SolveOT
import os
import pandas as pd
def getDistSqrTorus(x,y):
    dim = 1
    if (x.ndim == 2):
        dim = np.shape(x)[1]
    m = np.shape(x)[0]
    n = np.shape(y)[0]
    return scipy.linalg.norm((x.reshape((m,1,dim))-y.reshape((1,n,dim)) + 0.5)%1 - 0.5,axis = 2)**2

def cost(X,Y):
    
    return getDistSqrTorus(X,Y)

def dens_gauss_shift(X, Y, shift, std, shift_prob=0.5):
    """
    X, Y same shape numpy arrays to describe point clouds
    shift = shift of second diagonal
    std = standard deviation of both diagonals
    shift_prob = probability of going to the se
    """
    
    o = np.ceil(4 * std)
    
    a = (Y - X).reshape((*X.shape, 1))
    z = np.arange(-o, o+1, dtype=float).reshape((*[1 for _ in X.shape], -1))
    
    d0 = 1 / ((2 * np.pi)**.5 * std) * np.sum(np.exp(-(a - z)**2 / (2 * std**2)), axis=-1)
    d1 = 1 / ((2 * np.pi)**.5 * std) * np.sum(np.exp(-(a - shift - z)**2 / (2 * std**2)), axis=-1)
    
    return shift_prob * d1 + (1 - shift_prob) * d0

# def slow_Gau(gen,num,std,shift,shift_prob=0.5):
#     x = np.zeros(num)
#     y = np.zeros(num)
#     for i in range(num):
#         xp = gen.random()
#         sft = gen.choice([0,1], p=[1-shift_prob,shift_prob])
#         ggau = gen.normal(0, std)
#         yp = xp + sft*shift + ggau
#         x[i] = xp%1
#         y[i] = yp%1
#     return x.T, y.T

def sample_Gau(gen,num ,std, shift,dim = 1, shift_prob = 0.5):
    x = gen.random(num)
    shift_ind = gen.choice([0,1], p=[1-shift_prob,shift_prob],size = num)
    gau = gen.normal(0, std, size = num)
    y = x + shift_ind*shift + gau
    y = y%1
    while (dim > 1):
        xx = gen.random(num)
        yy = gen.random(num)
        x = np.vstack((x, xx))
        y = np.vstack((y, yy))
        dim -= 1
    return x.T,y.T

def Get_L2_loss(gen,N:int,std:float,jump:float,jump_prob:float,ve:float,dim:int,MtCl = 10000):
    """
    Simulate data points and Return Monte-Carlo approximated L_2 distance between True density and estimated density, as well as
    the approximated distance between Blured True density and estimated density.
    N = Number of simulation points to be sampled
    std = standard deviation for distance between two species
    jump = shift distance
    jump_prob = shift probability
    ve = sinkhorn regulariser
    dim = torus dimention (with uniform distribution on extra dimentions)
    10 * MtCl = total Monte-Carlo Sampling points
    """
    x,y = sample_Gau(gen, num = N ,std = std, shift = jump, dim = dim, shift_prob = jump_prob)
    Res_X = SolveOT(np.ones(N)/N,np.ones(N)/N,cost(x,x),1e-6,ve,10,returnSolver = True)
    Res_Y = SolveOT(np.ones(N)/N,np.ones(N)/N,cost(y,y),1e-6,ve,10,returnSolver = True) 
    ePts = 10
    MS = 0 #Monte-Carlo Result for True vs estimated
    MS2 = 0 #Blured True vs estimated
    for _ in range(MtCl):
        ddim = dim
        x_e = gen.uniform(0,1,size=ePts)%1
        y_e = gen.uniform(0,1,size=ePts)%1
        x_cloud,y_cloud = np.meshgrid(x_e,y_e)
        while (ddim > 1):
            xx = gen.random(ePts)
            yy = gen.random(ePts)
            x_e = np.vstack((x_e, xx))
            y_e = np.vstack((y_e, yy))
            ddim -= 1
        x_e = x_e.T
        y_e = y_e.T
        pot_x = 1/(np.sum(np.exp((-cost(x_e,x) + Res_X[2].beta)/ve),axis=1))
        pot_y = 1/(np.sum(np.exp((-cost(y_e,y) + Res_Y[2].beta)/ve),axis=1))
        F_X = pot_x[:,np.newaxis]*np.exp((-cost(x_e,x) + Res_X[2].beta)/ve)*N
        F_Y = pot_y[:,np.newaxis]*np.exp((-cost(y_e,y) + Res_Y[2].beta)/ve)*N

        M = dens_gauss_shift(X = x_cloud, Y = y_cloud, shift = jump, std =std, shift_prob=jump_prob)
        M2 = dens_gauss_shift(X = x_cloud, Y = y_cloud, shift = jump, std = np.sqrt(std**2 + ve), shift_prob=jump_prob)
        MS2 += np.sum((M2-F_Y@F_X.T/N)**2)
        MS += np.sum((M-F_Y@F_X.T/N)**2)
    return np.sqrt(MS/(ePts**2*MtCl)), np.sqrt(MS2/(ePts**2*MtCl))



def get_Bais(std:float,jump:float,jump_prob:float,ve:float,Res = 5000):
    """
    Return L2 distance between true density and blured true density.
    std = standard deviation for distance between two species
    jump = shift distance
    jump_prob = shift probability
    ve = sinkhorn regulariser
    Res = resolution, default 5000.    
    """
    x_e = y_e = np.linspace(0,1,Res,endpoint = False)
    M2 = dens_gauss_shift(*np.meshgrid(y_e,x_e), shift = jump, std = np.sqrt(std**2 + ve), shift_prob=jump_prob)
    M1 = dens_gauss_shift(*np.meshgrid(y_e,x_e), shift = jump, std =std, shift_prob=jump_prob)

    return np.sqrt(np.sum((M1-M2)**2))/Res




def get_M(gen,N:int,std:float,jump:float,ve:float,jump_prob = .5):
    """
    Simulate data points in one dim setting, and return the estimated density with resolution 1k by 1k
    N = Number of simulation points to be sampled
    std = standard deviation for distance between two species
    jump = shift distance
    jump_prob = shift probability
    ve = sinkhorn regulariser
    """
    x,y = sample_Gau(gen, num = N ,std = std, shift = jump,dim = 1, shift_prob = jump_prob)
    Res_X = SolveOT(np.ones(N)/N,np.ones(N)/N,cost(x,x),1e-6,ve,10,returnSolver = True)
    Res_Y = SolveOT(np.ones(N)/N,np.ones(N)/N,cost(y,y),1e-6,ve,10,returnSolver = True) 
    x_e = y_e = np.linspace(0,1,1000,endpoint=False)
    pot_x = 1/(np.sum(np.exp((-cost(x_e,x) + Res_X[2].beta)/ve),axis=1))
    pot_y = 1/(np.sum(np.exp((-cost(y_e,y) + Res_Y[2].beta)/ve),axis=1))
    F_X = pot_x[:,np.newaxis]*np.exp((-cost(x_e,x) + Res_X[2].beta)/ve)*N
    F_Y = pot_y[:,np.newaxis]*np.exp((-cost(y_e,y) + Res_Y[2].beta)/ve)*N
    return F_Y@F_X.T/N

filenameBase="./results_torus/"
def Sample_torus_L2(N,std,jump,jump_prob,ve,dim):
    seed = np.random.SeedSequence()
    gen = np.random.Generator(np.random.MT19937(seed))
    L2, L2_true = Get_L2_loss(gen,N,std,jump,jump_prob,ve,dim,MtCl = 10000)
    # resulting data frame
    res = {"L2" : [L2],
        "L2_true" : [L2_true],
        "N" : [N],
        "std" : [std],
        "jump" : [jump],
        "jump_prob" : [jump_prob],
        "ve" : [ve],
        "dim" : [dim]
        }
    # path to sample file
    fname = os.path.join(filenameBase, "sample_{}.csv".format(np.random.SeedSequence().entropy))
    pd.DataFrame(res).to_csv(fname)


def get_extension(seed:int, NPts:int, std:float, jump:float, ve:float, res = 2000,jump_prob = .5, dim = 1):
    gen = np.random.Generator(np.random.MT19937(seed))
    nev = 1
    x,y = sample_Gau(gen, num = NPts ,std = std, shift = jump, dim = dim, shift_prob = jump_prob)
    res_x = SolveOT(np.ones(NPts)/NPts,np.ones(NPts)/NPts,cost(x,x),1e-6,ve,10,returnSolver = True)
    EK_x = res_x[1].toarray()
    res_y = SolveOT(np.ones(NPts)/NPts,np.ones(NPts)/NPts,cost(x,y),1e-6,ve,10,returnSolver = True)
    EK_y = res_y[1].toarray()
    grid = res
    x_e = np.linspace(0,1,grid,endpoint=False)
    EK_x *= NPts**2
    EK_y *= NPts**2
    pot_y = 1/(np.sum(np.exp((-cost(x_e,y) + res_y[2].beta)/ve),axis=1))
    pot_y = pot_y.T
    F_Y = pot_y[:,np.newaxis]*np.exp((-cost(x_e,y) + res_y[2].beta)/ve)*NPts
    B = EK_y@EK_x.T
    eigval,eigvet = np.linalg.eig(B)
    temp = eigvet[:,nev]
    temp *= np.abs(temp[0])/temp[0]
    return x_e, (F_Y@EK_x.T)@np.real(temp/eigval[nev]), x, np.real(eigvet[:,nev])