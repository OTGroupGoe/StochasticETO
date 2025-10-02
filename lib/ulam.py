# Copyright © 2025 Thilo Stier
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import numpy as np
from scipy.sparse import coo_array

def ulam(X, Y, bincnt=None, binsize=None):
    """
    calculate ulam transition matrix from data with given bincount (per dimension) or binsize (per dimension)
    
    params:
    X = array of shape (n, d) containing pre-transition points where n is the number of points and d is the point dimension
    Y = array of shape (n, d) containint post-transition points
    bincnt = int or array of shape (d,) type int giving number of bins (per dimension); do not combine with binsz
    binsize = float or array of shape (d,) type float giving size of bins (per dimension); do not combine with bincnt
    """
    
    Z = np.stack((X, Y))
    _, n, d = Z.shape
    
    lo = np.min(Z, axis=(0,1))
    hi = np.max(Z, axis=(0,1))
    rg = hi - lo
    
    if bincnt is not None:
        shift = -lo
        scale = bincnt / rg
        binsize = 1. / scale
        
    elif binsize is not None:
        bincnt = np.ceil(rg / binsize).astype(np.int64)
        scale = 1. / binsize
        shift = -lo + 0.5 * (binsize * bincnt - rg)

    if type(bincnt) == int:
        bincnt = np.full(d, bincnt)
        
    Z = np.clip(((Z + shift) * scale).astype(np.int64), 0, bincnt-1).transpose(1,0,2)
   
    u, iv = np.unique(Z.reshape(-1,d), axis=0, return_inverse=True)
    iv = iv.reshape(-1,2)
    
    a = coo_array((np.ones(iv.shape[0]), iv.T), shape=(u.shape[0], u.shape[0]))
    a.sum_duplicates()
    
    a /= np.maximum(a.sum(0)[None,:], 1e-9)
    
    return {"T" : a,
            "idx" : u,
            "scale" : scale,
            "shift" : shift,
            "bincnt" : bincnt,
            "binsize" : binsize}

