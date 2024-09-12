import numpy as np
import scipy
import scipy.sparse.linalg
from lib.SinkhornKeops import TKeopsSinkhornSolverStandard as SinkhornSolver
import time
import sys
#from pykeops.numpy import LazyTensor
import itertools
#import LinOT.SinkhornNP as S
def Get_data(rotlist,epslist,laglist,skiplist,t0list = [2000]):
    for lag,rot,eps,skip,t0 in itertools.product(laglist,rotlist,epslist,skiplist,t0list):
        t1=time.time()
        filename="data_convection/1102271_sw_T.data"
        if (rot == 0):
            filename = "data_convection/1003261_sw_T.data"
        posFull=np.loadtxt(filename,comments="#",delimiter="\t")
        posFull=posFull[t0:,2:].astype(np.float32)
        nFull=posFull.shape[0]
        nPts=1+(nFull-1-lag)//skip
        posX=posFull[0:skip*nPts:skip].copy()
        posY=posFull[lag:skip*nPts+lag:skip].copy()
        mu=np.ones(nPts,dtype=np.float32)
        params={}
        params["errorGoal"]=1E-3*np.sum(mu)
        params["verbose"]=True

        params["epsInit"]=None
        params["eps"]=eps
        params["verbose"]=False
        # -

        solverXX=SinkhornSolver(posX,posX,mu,mu,**params)
        solverXX.cfg["innerIterations"]=20
        # be very careful about the axis oder in solverYX
        solverYX=SinkhornSolver(posX,posY,mu,mu,**params)
        solverYX.cfg["innerIterations"]=20
        # print(solverXX.epsList)
        solverXX.solve()

        solverYX.solve()

        # ## Compute eigenvalues

        gammaXX=scipy.sparse.linalg.aslinearoperator(solverXX.SinkhornPi.exp())
        gammaYX=scipy.sparse.linalg.aslinearoperator(solverYX.SinkhornPi.exp())

        gamma=gammaYX@gammaXX

        nEigval=10

        eigval,eigvec=scipy.sparse.linalg.eigs(gamma,k=nEigval,which="LM",\
                return_eigenvectors=True,tol=1E-3)

        # ## Dump to file

        filenameBase="./results_convection/"

        import os

        i=0
        filename=filenameBase+f"{i:04d}.npz"
        while os.path.isfile(filename):
            i+=1
            filename=filenameBase+f"{i:04d}.npz"
        print("\nsaving to " + filename)
        t2=time.time()
        np.savez(filename,
                alphaXX=solverXX.alpha,betaXX=solverXX.beta,
                alphaYX=solverYX.alpha,betaYX=solverYX.beta,
                eps=solverXX.eps,
                eigval=eigval,eigvec=eigvec,
                initOffset=t0,skip=skip,lag=lag,rot = rot,time = t1 - t2)
    return