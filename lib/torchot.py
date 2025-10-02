# Copyright © 2025 Thilo Stier
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import torch
import numpy as np

def get_onb(n, dtype=torch.float32, device=torch.device("cpu")):
    """
    Construct an onb of size n where the first vector is diagonal (all entries the same).
    Return result B as matrix where basis vectors appear in rows, i.e. B[i] = i'th basis vector.
    """
    nn = n
    B = torch.zeros((n,n), dtype=dtype)
    B[0,:] = 1
    j = 0
    while nn > 0:
        bz = (int(np.floor(np.log2(nn))))
        k = 2**bz
        bj = j
        B[j,j:j+k] = 1
        if j > 0:
            B[j,:j] = -k/j
        B[j] /= torch.linalg.norm(B[j])
        j += 1
        for ibz in range(bz)[::-1]:
            ik = 2**ibz
            for l in range(2**(bz - ibz - 1)):
                B[j,bj+2*l*ik:bj+(2*l+1)*ik] = 1
                B[j,bj+(2*l+1)*ik:bj+(2*l+2)*ik] = -1
                B[j] /= torch.linalg.norm(B[j])
                j += 1
        nn -= k
    return B.to(device)

class Sinkhorn_c:
    def __init__(self, c, mu, nu, epsis, verbose=False):
        """
        construct Sinkhorn solver for cost given as matrix
        params:
        c = cost matrixm shape (b,n,m) or (1,n,m)
        mu = weights for first marginal, shape (b,n) or (1,n)
        nu = weights for second marginal, shape (b,m) or (1,m)
        epsis = regularization strengths to calculate values for, shape (b,e) or (1,e)
        """

        self.verbose = verbose
        
        assert len(c.shape) == 3, "c has invalid shape"
        assert len(mu.shape) == 2 and mu.shape[1] == c.shape[1], "mu has invalid shape"
        assert len(nu.shape) == 2 and nu.shape[1] == c.shape[2], "nu has invalid shape"
        assert len(epsis.shape) == 2, "epsis has invalid shape"
        
        # const values
        self.ptype = c.dtype             # type to use for primal-type objects
        self.dtype = torch.float64       # type to use for duals
        self.device = c.device           # device to use (all parameters are already assumed to be on the same device)
        
        self.n = c.shape[1]              # first marginal size
        self.m = c.shape[2]              # second marginal size
        bs = [c.shape[0], mu.shape[0], nu.shape[0], epsis.shape[0]]
        self.b = max(bs)                 # batch size
        assert all([(b == 1 or b == self.b) for b in bs]), "batch dimensions are not compatible"
        
        self.c = c                       # cost matrix (shape=([1b],n,m))
        self.mu = mu[:,:,None]           # first marginal measure (shape=([1b],n,1))
        self.nu = nu[:,None,:]           # second marginal measure (shape=([1b],1,m))
        self.e = epsis.shape[-1]         # number of epsis
        self.target_epsis = epsis        # target epsis (shape=([1b],e))
        
        # runtime values
        self.epsi = epsis[:,0]           # current epsi (shape ([1b],))
        self.alpha = torch.zeros((self.b,self.n,1), dtype=self.dtype, device=self.device) # first dual (shape=(b,n,1))
        self.beta  = torch.zeros((self.b,1,self.m), dtype=self.dtype, device=self.device) # second dual (shape=(b,1,m))
        self.k1 = None                   # approx k @ nu (used to estimate error on first marginal, would be 1 if exact)
        self.k2 = None                   # approx k.T @ mu (used to estimate error on second marginal, would be 1 if exact)
        self.k = None                    # kernel (for mvp, equal transport plan)
        self.u = torch.ones((self.b,self.n), dtype=self.ptype, device=self.device) # first scaling factor
        self.v = torch.ones((self.b,self.m), dtype=self.ptype, device=self.device) # second scaling factor
        self.absorbed = True             # scaling factors are not in use currently
        
        # output values
        self.alphas = torch.empty((self.b, self.e, self.n), dtype=self.dtype, device=self.device) # output first duals
        self.betas  = torch.empty((self.b, self.e, self.m), dtype=self.dtype, device=self.device) # output second duals
        
    @property
    def pi(self):
        """
        get transport plan (as dense matrix), shape (b,n,m)
        """
        self.absorb()
        return torch.exp((self.alpha + self.beta - self.c).to(self.ptype) / self.epsi[:,None,None]) * self.mu * self.nu
    
    @property
    def pis(self):
        """
        get transport plan for all target epsis (as dense tensor)
        """
        return torch.exp((self.alphas[:,:,:,None] + self.betas[:,:,None,:] - self.c[:,None,:,:]) / self.target_epsis[:,:,None,None]) * self.mu[:,None,:,:] * self.nu[:,None,:,:]
    
    @property
    def cost(self):
        """
        get approximate transport cost (assuming alpha, beta are optimal)
        """
        self.absorb()
        return torch.sum(self.mu * self.alpha) + torch.sum(self.nu * self.beta)
    
    @property
    def costs(self):
        """
        get approximate transport costs for all target epsis (assuming alphas, betas are optimal)
        """
        return (torch.sum(self.mu.squeeze(2)[:,None,:] * self.alphas, dim=-1).to(self.ptype) +
                torch.sum(self.nu.squeeze(1)[:,None,:] * self.betas, dim=-1).to(self.ptype))
        
    def step_lse(self):
        """
        perform one step (both sides) using lse; alpha is updated first
        """
        self.absorb()
        
        #self.alpha = -self.epsi * torch.logsumexp((self.beta - self.c).to(self.ptype) / self.epsi + self.lognu, dim=1, keepdim=True)
        a = (self.beta - self.c) / self.epsi[:,None,None]  # shape (b,n,m)
        m = torch.amax(a, dim=2, keepdim=True)             # shape (b,n,1)
        a = torch.exp((a - m).to(self.ptype))              # shape (b,n,m)
        s = torch.sum(a * self.nu, dim=2, keepdim=True)    # shape (b,n,1)
        del a
        self.k1 = s * torch.exp((m + self.alpha / self.epsi[:,None,None]).to(self.ptype)) # shape (b,n,1)
        self.alpha = -self.epsi[:,None,None] * (m + torch.log(s)) # shape (b,n,1)
        
        #self.beta = -self.epsi * torch.logsumexp((self.alpha - self.c) / self.epsi + self.logmu, dim=0, keepdim=True)
        a = (self.alpha - self.c) / self.epsi[:,None,None] # shape (b,n,m)
        m = torch.amax(a, dim=1, keepdim=True)             # shape (b,1,m)
        a = torch.exp((a - m).to(self.ptype))              # shape (b,n,m)
        s = torch.sum(a * self.mu, dim=1, keepdim=True)    # shape (b,1,m)
        del a
        self.k2 = s * torch.exp((m + self.beta / self.epsi[:,None,None]).to(self.ptype)) # shape (b,1,m)
        self.beta = -self.epsi[:,None,None] * (m + torch.log(s)) # shape (b,1,m)
        
    def step_mvp(self):
        """
        perform one step (both sides) using mvp; u is updated first
        """
        if self.k is None:
            self.calc_kernel()
        
        a = self.mu.squeeze(2) / torch.einsum("bnm,bm->bn", self.k, self.v) # shape (b,n)
        self.k1 = (self.u / a)[:,:,None]                   # shape (b,n,1)
        self.u = a                                         # shape (b,n)
        
        a = self.nu.squeeze(1) / torch.einsum("bnm,bn->bm", self.k, self.u) # shape (b,m)
        self.k2 = (self.v / a)[:,None,:]                   # shape (b,1,m)
        self.v = a                                         # shape (b,m)
        
        self.absorbed = False
        
    def absorb(self):
        """
        absorb scaling factors into duals, invalidate kernel
        """
        if not self.absorbed:
            if self.verbose:
                print("absorbing")
            self.alpha += self.epsi[:,None,None] * torch.log(self.u[:,:,None])
            self.beta += self.epsi[:,None,None] * torch.log(self.v[:,None,:])
            self.u[...] = 1
            self.v[...] = 1
            self.k = None
            self.absorbed = True
        self.shift_stabilize()
            
    def shift_stabilize(self):
        """
        shift duals such that they have the same mean
        """
        sh = 0.5 * (torch.mean(self.alpha, dim=(1,2)) - torch.mean(self.beta, dim=(1,2)))
        self.alpha -= sh[:,None,None]
        self.beta += sh[:,None,None]
            
    def calc_kernel(self):
        """
        calculate kernel given current parameters
        """
        self.k = self.pi
        
    def set_epsi(self, epsi):
        """
        set current epsi to given value
        """
        self.absorb()
        if self.verbose:
            print(f"set epsi to {float(epsi.item()):.3e}")
        self.epsi = epsi
        self.k = None
    
    def run_inner(self, max_err=1e-4, max_its=10000, lse_its=1):
        """
        run for current self.epsi until approx. marginal error is below max_err or number of iterations reaches max_its
        """
        if self.verbose:
            print(f"start inner loop for epsi = {float(self.epsi.item()):.3e}")

        for it in range(max_its):
            if it < lse_its:
                self.step_lse()
            else:
                uu = self.u
                vv = self.v
                self.step_mvp()
                if not (torch.all(self.u > 1e-10) and
                        torch.all(self.v > 1e-10) and
                        torch.all(self.u < 1e10) and
                        torch.all(self.v < 1e10)) or (it % 1000 == 0):
                    
                    if torch.any(torch.isnan(self.u)) or torch.any(torch.isnan(self.v)):
                        if self.verbose:
                            print("nan in mvp dual, fall back to lse")
                        self.u = uu
                        self.v = vv
                        self.absorb()
                        self.step_lse()
                    else:
                        #print("absorb due to large value or number of iterations")
                        self.absorb()
                
            err = max(torch.max(torch.abs(self.k1 - 1)).item(), torch.max(torch.abs(self.k2 - 1)).item())
            if err < max_err:
                if self.verbose:
                    print(f"break inner loop at iteration {it} since error is reached: {err} < {max_err}")
                break
        self.absorb()
        if it + 1 == max_its:
            print(f"WARNING: Sinkhorn did not converge for epsi = {float(self.epsi.item()):.3e}, err = {err:.3e}, max_err = {max_err:.3e}, its = {it + 1}, dtype = {self.dtype}, ptype = {self.ptype}")
        return it + 1, err
    
    @staticmethod
    def get_next_epsi(current, target, epsi_scale):
        """
        given we are (with epsi) at current and next want to scale to target, with scaling speed limit epsi_scale,
        return what the next value for epsi should be (using numpy)
        """
        cnt = np.ceil((np.log(target) - np.log(current)) / np.log(epsi_scale) - 1e-4).astype(np.int32)
        return np.where(cnt <= 1, target, current * np.exp(np.divide(np.log(target) - np.log(current), cnt, where = (cnt > 0))))

    
    def get_epsi_steps(self, init_epsi=None, epsi_scale=0.5):
        """
        get an epsi scaling schedule, starting at init_epsi (shape=([1b],))
        """
        t = self.target_epsis.shape[0]
        target_epsis = self.target_epsis.detach().cpu().numpy() # shape (t,e)
        if init_epsi is None:
            init_epsi = np.maximum(target_epsis[:,0], torch.amax(self.c, dim=(1,2)).detach().cpu().numpy()**2) # shape ([1bt],)
        elif type(init_epsi) is not type(np.zeros(1)):
            init_epsi = np.array([init_epsi])
        
        # make sure init_epsi.shape == t (i.e. init_epsi is batched iff target_epsis is batched)
        if t == 1 and init_epsi.shape[0] > 1:
            init_epsi = np.max(init_epsi, axis=0, keepdims=True)
        if t > 1 and init_epsi.shape[0] == 1:
            init_epsi = np.repeat(init_epsi, target_epsis.shape[0], axis=0) 
        
        epsis = [init_epsi]                                # epsis to solve for
        i = np.where(init_epsi == target_epsis[:,0], 1, 0) # number of target_epsis already in epsis (i.e. target_epsis[i] == next epsi to reach)
        oidx = [i - 1]                                     # index of target_epsis corresponding to epsi (or -1 if it does not correspond)
        
        while np.any(i < self.e):
            cur, target = epsis[-1], target_epsis[np.arange(t), np.minimum(i, self.e-1)]
            nxt = self.get_next_epsi(cur, target, epsi_scale)
            #print(cur, target, nxt, i)
            epsis += [nxt]
            oidx += [np.where(nxt == target, i, -1)]
            i += (nxt == target)
            
        return (torch.tensor(np.array(epsis), dtype=self.ptype, device=self.device), 
                torch.tensor(np.minimum(np.array(oidx), self.e-1), dtype=torch.int32, device=self.device))
        
    
    def run(self, init_epsi=None, epsi_scale=0.5, max_err=1e-4, max_its_per_epsi=10000):
        """
        run Sinkhorn starting at init_epsi, scaling by approx. epsi_scale, 
        until max_err relative marginal error is reached
        performing at most max_its_per_epsi iterations for each epsi step
        """
        
        if torch.max(torch.abs(self.mu.sum((-1,-2)) - self.nu.sum((-1,-2)))).item() > 0.1 * max_err:
            print("WARNING: measure total masses differ significantly")
        
        epsis, oidx = self.get_epsi_steps(init_epsi, epsi_scale)

        if self.verbose:
            epsis_str = ", ".join([f"{float(e.item()):.3e}" for e in epsis])
            print(f"start sinkhorn, epsi steps: {epsis_str}")

        
        ba = torch.arange(self.b, device=self.device)
        
        for epsi, i in zip(epsis, oidx):
            self.set_epsi(epsi)
            its, err = self.run_inner(max_err, max_its_per_epsi)
            
            self.alphas[ba,i,:] = self.alpha.squeeze(2)
            self.betas[ba,i,:]  = self.beta.squeeze(1)
    

class Sinkhorn_W2(Sinkhorn_c):
    def __init__(self, X, Y, mu, nu, epsis, verbose=False):
        """
        construct Sinkhorn solver for quadratic Euclidean cost between weighted point clouds
        params:
        X = support points of first marginal, shape (b,n,d) or (1,n,d)
        Y = support points of second marginal, shape (b,m,d) or (1,m,d)
        mu = weights for first marginal, shape (b,n) or (1,n)
        nu = weights for second marginal, shape (b,m) or (1,m)
        epsis = regularization strengths to calculate values for, shape (b,e) or (1,e)
        """
        Sinkhorn_c.__init__(self, torch.cdist(X, Y, 2)**2, mu, nu, epsis, verbose)

class EOT_c(torch.autograd.Function):
    
    @staticmethod
    def forward(c, epsis, mu=None, nu=None, err=1e-4, verbose=False):
        """
        Calculate EOT transport between measures mu and nu for cost c with regularization strength(s) epsis.
        
        parameters:
        c = cost matrix, shape (n,m) or (b,n,m)
        epsis = entropic penalization strength, shape () or (e,) or (b,e) if the result should be calculated for multiple epsis
        mu = first marginal point masses, shape (n,) or (b,n), > 0, default uniform
        nu = second marginal point masses, shape (m,) or (b,m), > 0, default uniform
        
        returns:
        (cost, alpha, beta) - EOT transport cost, first dual, second dual, shapes (), (n,), (m,) or (e,), (e,n), (e,m) depending of shape of epsi
        """
        
        device = c.device
        
        b = None
        if len(c.shape) == 3:
            b = c.shape[0]
        elif mu is not None and len(mu.shape) == 2:
            b = mu.shape[0]
        elif nu is not None and len(nu.shape) == 2:
            b = nu.shape[0]
        elif len(epsis.shape) == 2:
            b = epsis.shape[0]
        
        if b is None:
            b = 1
            explicit_batch = False
        else:
            explicit_batch = True
        
        if len(c.shape) == 3: # batch c
            _, n, m = c.shape
        elif len(c.shape) == 2: # non-batch c
            n, m = c.shape
            c = c[None,:,:]
        else: # invalid c
            assert False, "c has invalid shape"
            
        if mu is None:
            mu = torch.full((1,n), 1./n, device=device)
        elif len(mu.shape) == 2: # batch mu
            pass
        elif len(mu.shape) == 1: # non-batch mu
            mu = mu[None,:]
        else: # invalid mu
            assert False, "mu has invalid shape"
          
        if nu is None:
            nu = torch.full((1,m), 1./m, device=device)
        elif len(nu.shape) == 2: # batch nu
            pass
        elif len(nu.shape) == 1: # non-batch nu
            nu = nu[None,:]
        else: # invalid nu
            assert False, "nu has invalid shape"
            
        if len(epsis.shape) == 2: # batch multi-epsis
            e = epsis.shape[1]
            explicit_multi_epsi = True
        elif len(epsis.shape) == 1: # non-batch multi-epsis
            e = epsis.shape[0]
            epsis = epsis[None,:]
            explicit_multi_epsi = True
        elif len(epsis.shape) == 0: # non-batch single epsis
            e = 1
            epsis = epsis[None,None]
            explicit_multi_epsi = False
        else: # invalid epsis
            assert False, "epsis has invalid shape"

        assert c.shape in [(1,n,m), (b,n,m)], f"c has invalid shape {c.shape}"
        assert mu.shape in [(1,n), (b,n)], f"mu has invalid shape {mu.shape}"
        assert nu.shape in [(1,m), (b,m)], f"nu has invalid shape {nu.shape}"
        assert epsis.shape in [(b,e), (1,e)], f"epsis has invalid shape {epsis.shape}"
        
        solv = Sinkhorn_c(c, mu, nu, epsis, verbose)
        solv.run(max_err=err)
        
        if explicit_batch:
            return solv.costs, solv.alphas, solv.betas
        else:
            if explicit_multi_epsi:
                return solv.costs[0], solv.alphas[0], solv.betas[0]
            else:
                return solv.costs[0,0], solv.alphas[0,0], solv.betas[0,0]

      
    @staticmethod
    def setup_context(ctx, inputs, outputs):
        #ctx.set_materialize_grads(False)
        ctx.save_for_backward(*(inputs[:4]), *(outputs[1:]))
        
    
    @staticmethod
    def backward(ctx, dcost, dalpha, dbeta):
        c, epsis, mu, nu, alpha, beta = ctx.saved_tensors

        device, dtype = c.device, c.dtype
        ptype = dcost.dtype
        
        b = None
        if len(c.shape) == 3:
            b = c.shape[0]
        elif mu is not None and len(mu.shape) == 2:
            b = mu.shape[0]
        elif nu is not None and len(nu.shape) == 2:
            b = nu.shape[0]
        elif len(epsis.shape) == 2:
            b = epsis.shape[0]
        
        if b is None:
            b = 1
            explicit_batch = False
        else:
            explicit_batch = True
        
        if len(c.shape) == 3: # batch c
            _, n, m = c.shape
            explicit_c_batch = True
        elif len(c.shape) == 2: # non-batch c
            n, m = c.shape
            c = c[None,:,:]
            explicit_c_batch = False
        else: # invalid c
            assert False, "c has invalid shape"

        if mu is None:
            mu = torch.full((1,n), 1./n, device=device)
            implicit_mu = True
        elif len(mu.shape) == 2: # batch mu
            explicit_mu_batch = True
            implicit_mu = False
        elif len(mu.shape) == 1: # non-batch mu
            mu = mu[None,:]
            explicit_mu_batch = False
            implicit_mu = False
          
        if nu is None:
            nu = torch.full((1,m), 1./m, device=device)
            implicit_nu = True
        elif len(nu.shape) == 2: # batch nu
            explicit_nu_batch = True
            implicit_nu = False
        elif len(nu.shape) == 1: # non-batch nu
            nu = nu[None,:]
            explicit_nu_batch = False
            implicit_nu = False
            
        if len(epsis.shape) == 2: # batch multi-epsis
            e = epsis.shape[1]
            explicit_epsi_batch = True
            explicit_multi_epsi = True
        elif len(epsis.shape) == 1: # non-batch multi-epsis
            e = epsis.shape[0]
            epsis = epsis[None,:]
            explicit_epsi_batch = False
            explicit_multi_epsi = True
        elif len(epsis.shape) == 0: # non-batch single epsis
            e = 1
            epsis = epsis[None,None]
            explicit_epsi_batch = False
            explicit_multi_epsi = False
        
        if explicit_batch:
            pass
        else:
            if explicit_multi_epsi:
                alpha, beta = alpha[None,...], beta[None,...]
                dalpha, dbeta, dcost = dalpha[None,...], dbeta[None,...], dcost[None,...]
            else:
                alpha, beta = alpha[None,None,...], beta[None,None,...]
                dalpha, dbeta, dcost = dalpha[None,None,...], dbeta[None,None,...], dcost[None,None,...]
        
        dc, depsi, dmu, dnu = None, None, None, None
        
        assert c.shape in [(1,n,m), (b,n,m)], f"c has invalid shape {c.shape}"
        assert mu.shape in [(1,n), (b,n)], f"mu has invalid shape {mu.shape}"
        assert nu.shape in [(1,m), (b,m)], f"nu has invalid shape {nu.shape}"
        assert epsis.shape in [(b,e), (1,e)], f"epsis has invalid shape {epsis.shape}"

        assert dcost.shape == (b,e), f"dcost has invalid shape {dcost.shape} != {(b,e)}"
        assert dalpha.shape == (b,e,n), f"dalpha has invalid shape {dalpha.shape} != {(b,e,n)}"
        assert dbeta.shape == (b,e,m), f"dbeta has invalid shape {dbeta.shape} != {(b,e,m)}"

        k = torch.exp((alpha[:,:,:,None] + beta[:,:,None,:] - c[:,None,:,:]).to(dtype) / epsis[:,:,None,None])
        B = get_onb(n, dtype=dtype, device=device)[1:]

        assert(c.shape == (b,n,m))
        assert(k.shape == (b,e,n,m))
        assert(B.shape == (n-1,n))

        A = torch.eye(n-1, dtype=dtype, device=device) - torch.einsum("ij,bj,bejk,bk,belk,ml->beim", B, mu, k, nu, k, B)
        assert(A.shape == (b,e,n-1,n-1))

        rhs = torch.einsum("ij,bej->bei", B, dalpha.to(dtype)) - torch.einsum("ij,bj,bejk,bek->bei", B, mu, k, dbeta.to(dtype))
        assert(rhs.shape == (b,e,n-1))

        delta = torch.einsum("ji,bej->bei", B, torch.linalg.solve(A, rhs))
        assert(delta.shape == (b,e,n))

        # c needs grad
        if ctx.needs_input_grad[0]:
            pnkd = dbeta.to(dtype) - torch.einsum("bj,beij,bei->bej", nu, k, delta)
            dc = (torch.einsum("bep,beij,bi,jp->beij", pnkd, k, mu, torch.eye(m, dtype=dtype, device=device))
                + torch.einsum("bep,beij,bj,ip->beij", delta, k, nu, torch.eye(n, dtype=dtype, device=device))
                + torch.einsum("be,beij,bi,bj->beij", dcost.to(dtype), k, mu, nu))
            assert(dc.shape == (b,e,n,m))
            
        # epsi needs grad
        if ctx.needs_input_grad[1]:
            klogk = k * ((alpha[:,:,:,None] + beta[:,:,None,:] - c[:,None,:,:]).to(dtype) / epsis[:,:,None,None])

            assert(klogk.shape == (b,e,n,m))

            depsi = (torch.einsum("bei,beij,bj->be", delta, klogk, nu)
                   - torch.einsum("bei,beij,bj,belj,bl->be", delta, k, nu, klogk, mu)
                   + torch.einsum("bej,beij,bi->be", dbeta.to(dtype), klogk, mu)
                   + torch.einsum("be,beij,bi,bj->be", dcost, klogk, mu, nu))

            assert(depsi.shape == (b,e))

        # mu needs grad
        if ctx.needs_input_grad[2]:
            dmu = (torch.einsum("be,bel,belj,bj,beij->bei", epsis, delta, k, nu, k)
                 - torch.einsum("be,bej,beij->bei", epsis, dbeta.to(ptype), k)
                 + torch.einsum("be,bei->bei", dcost, alpha.to(ptype)))
            dmu -= torch.mean(dmu, dim=-1, keepdim=True)

            assert(dmu.shape == (b,e,n))

        # nu needs grad
        if ctx.needs_input_grad[3]:
            dnu = (-torch.einsum("be,bei,beij->bej", epsis, delta, k)
                 + torch.einsum("be,bei->bei", dcost, beta.to(ptype)))
            dnu -= torch.mean(dnu, dim=-1, keepdim=True)

            assert(dnu.shape == (b,e,m))

        if explicit_c_batch: # batch c
            pass
        elif dc is not None: # non-batch c
            dc = dc.squeeze(0)
        
        if implicit_mu: # no input mu
            dmu = None
        elif explicit_mu_batch: # batch mu
            pass
        elif dmu is not None: # non-batch mu
            dmu = dmu.squeeze(0)
            
        if implicit_nu:
            dnu = None
        elif explicit_nu_batch: # batch nu
            pass
        elif dnu is not None: # non-batch nu
            dnu = dnu.squeeze(0)
            
        if explicit_epsi_batch: # batch multi-epsis
            pass
        elif explicit_multi_epsi and depsi is not None: # non-batch multi-epsis
            depsi = depsi.squeeze(0)
        elif depsi is not None: # non-batch single epsis
            depsi = depsi.squeeze(0).squeeze(0)
            
        return dc, depsi, dmu, dnu, None, None



def SolveEOT(c, epsis, mu=None, nu=None, err=1e-4, verbose=False):
    """
    Calculate EOT transport between measures mu and nu for cost c with regularization strength(s) epsis.
    
    - Works both for single problems and batch data.
    - Works both for single epsi and sequence of epsis.
    - Reference measure is mu x nu.
    
    Parameters:
    c (tensor of shape (n,m) or (b,n,m)): cost matrices
    epsis (tensor of shape (), (e,) or (b,e)): entropic penalization strength(s). If multiple epsis are given (e > 1), they should always be sorted largest to smallest.
    mu (None or tensor of shape (n,) or (b,n)): first marginal point masses, default uniform probability measure
    nu (None or tensor of shape (m,) or (b,m)):  second marginal point masses, default uniform probability measure
    err (float): (approximate) allowable maximal relative marginal error, default 1e-4
    
    Returns:
    tuple of tensors (cost, alpha, beta): EOT transport cost, first dual, second dual
    - If used with batches, cost will have shape (b,e), alpha (b,e,n) and beta (b,e,m).
    - If used with single input and multiple epsis, cost will have shape (e,), alpha (e,n) and beta (e,m).
    - If used with single input and single epsi, cost will have shape (), alpha (n,) and beta (m,)
    """
    
    return EOT_c.apply(c, epsis, mu, nu, err, verbose)
    

        
class EOT_L2sqr(torch.autograd.Function):
    
    @staticmethod
    def forward(X, Y, epsis, mu=None, nu=None, err=1e-4, verbose=False):
        """
        Calculate EOT transport between (weighted) point clouds (X, mu) and (Y, nu) with regularization strength epsi.
        
        parameters:
        X = first marginal points, shape (n,d) or (b,n,d)
        Y = second marginal points, shape (m,d) or (b,m,d)
        epsis = entropic penalization strength, shape () or (e,) or (b,e) if the result should be calculated for multiple epsis
        mu = first marginal point masses, shape (n,) or (b,n), > 0, default uniform
        nu = second marginal point masses, shape (m,) or (b,m), > 0, default uniform
        
        returns:
        (cost, alpha, beta) - EOT transport cost, first dual, second dual, shapes (), (n,), (m,) or (e,), (e,n), (e,m) depending of shape of epsi
        """
        
        device = X.device
        
        b = None
        if len(X.shape) == 3:
            b = X.shape[0]
        elif len(Y.shape) == 3:
            b = Y.shape[0]
        elif mu is not None and len(mu.shape) == 2:
            b = mu.shape[0]
        elif nu is not None and len(nu.shape) == 2:
            b = nu.shape[0]
        elif len(epsis.shape) == 2:
            b = epsis.shape[0]
        
        if b is None:
            b = 1
            explicit_batch = False
        else:
            explicit_batch = True
        
        if len(X.shape) == 3: # batch X
            n, d = X.shape[1], X.shape[2]
        elif len(X.shape) == 2: # non-batch X
            n, d = X.shape[0], X.shape[1]
            X = X[None,:,:]
        else: # invalid X
            assert False, "X has invalid shape"
            
        if len(Y.shape) == 3: # batch Y
            m, d = Y.shape[1], Y.shape[2]
        elif len(Y.shape) == 2: # non-batch Y
            m, d = Y.shape[0], Y.shape[1]
            Y = Y[None,:,:]
        else: # invalid Y
            assert False, "Y has invalid shape"
        
        if mu is None:
            mu = torch.full((1,n), 1./n, device=device)
        elif len(mu.shape) == 2: # batch mu
            pass
        elif len(mu.shape) == 1: # non-batch mu
            mu = mu[None,:]
        else: # invalid mu
            assert False, "mu has invalid shape"
          
        if nu is None:
            nu = torch.full((1,m), 1./m, device=device)
        elif len(nu.shape) == 2: # batch nu
            pass
        elif len(nu.shape) == 1: # non-batch nu
            nu = nu[None,:]
        else: # invalid nu
            assert False, "nu has invalid shape"
            
        if len(epsis.shape) == 2: # batch multi-epsis
            e = epsis.shape[1]
            explicit_multi_epsi = True
        elif len(epsis.shape) == 1: # non-batch multi-epsis
            e = epsis.shape[0]
            epsis = epsis[None,:]
            explicit_multi_epsi = True
        elif len(epsis.shape) == 0: # non-batch single epsis
            e = 1
            epsis = epsis[None,None]
            explicit_multi_epsi = False
        else: # invalid epsis
            assert False, "epsis has invalid shape"

        assert X.shape in [(1,n,d), (b,n,d)], f"X has invalid shape {X.shape}"
        assert Y.shape in [(1,m,d), (b,m,d)], f"Y has invalid shape {Y.shape}"
        assert mu.shape in [(1,n), (b,n)], f"mu has invalid shape {mu.shape}"
        assert nu.shape in [(1,m), (b,m)], f"nu has invalid shape {nu.shape}"
        assert epsis.shape in [(b,e), (1,e)], f"epsis has invalid shape {epsis.shape}"
        
        solv = Sinkhorn_W2(X, Y, mu, nu, epsis, verbose)
        solv.run(max_err=err)
        
        if explicit_batch:
            return solv.costs, solv.alphas, solv.betas
        else:
            if explicit_multi_epsi:
                return solv.costs[0], solv.alphas[0], solv.betas[0]
            else:
                return solv.costs[0,0], solv.alphas[0,0], solv.betas[0,0]
    
    @staticmethod
    def setup_context(ctx, inputs, outputs):
        #ctx.set_materialize_grads(False)
        ctx.save_for_backward(*(inputs[:5]), *(outputs[1:]))
        
    
    @staticmethod
    def backward(ctx, dcost, dalpha, dbeta):
        X, Y, epsis, mu, nu, alpha, beta = ctx.saved_tensors

        device, dtype = X.device, X.dtype
        ptype = dcost.dtype
        
        b = None
        if len(X.shape) == 3:
            b = X.shape[0]
        elif len(Y.shape) == 3:
            b = Y.shape[0]
        elif mu is not None and len(mu.shape) == 2:
            b = mu.shape[0]
        elif nu is not None and len(nu.shape) == 2:
            b = nu.shape[0]
        elif len(epsis.shape) == 2:
            b = epsis.shape[0]
            
        if b is None:
            b = 1
            explicit_batch = False
        else:
            explicit_batch = True
        
        if len(X.shape) == 3: # batch X
            n, d = X.shape[1], X.shape[2]
            explicit_X_batch = True
        elif len(X.shape) == 2: # non-batch X
            n, d = X.shape[0], X.shape[1]
            X = X[None,:,:]
            explicit_X_batch = False
            
        if len(Y.shape) == 3: # batch Y
            m, d = Y.shape[1], Y.shape[2]
            explicit_Y_batch = True
        elif len(Y.shape) == 2: # non-batch Y
            m, d = Y.shape[0], Y.shape[1]
            Y = Y[None,:,:]
            explicit_Y_batch = False
        
        if mu is None:
            mu = torch.full((1,n), 1./n, device=device)
            implicit_mu = True
        elif len(mu.shape) == 2: # batch mu
            explicit_mu_batch = True
            implicit_mu = False
        elif len(mu.shape) == 1: # non-batch mu
            mu = mu[None,:]
            explicit_mu_batch = False
            implicit_mu = False
          
        if nu is None:
            nu = torch.full((1,m), 1./m, device=device)
            implicit_nu = True
        elif len(nu.shape) == 2: # batch nu
            explicit_nu_batch = True
            implicit_nu = False
        elif len(nu.shape) == 1: # non-batch nu
            nu = nu[None,:]
            explicit_nu_batch = False
            implicit_nu = False
            
        if len(epsis.shape) == 2: # batch multi-epsis
            e = epsis.shape[1]
            explicit_epsi_batch = True
            explicit_multi_epsi = True
        elif len(epsis.shape) == 1: # non-batch multi-epsis
            e = epsis.shape[0]
            epsis = epsis[None,:]
            explicit_epsi_batch = False
            explicit_multi_epsi = True
        elif len(epsis.shape) == 0: # non-batch single epsis
            e = 1
            epsis = epsis[None,None]
            explicit_epsi_batch = False
            explicit_multi_epsi = False
        
        if explicit_batch:
            pass
        else:
            if explicit_multi_epsi:
                alpha, beta = alpha[None,...], beta[None,...]
                dalpha, dbeta, dcost = dalpha[None,...], dbeta[None,...], dcost[None,...]
            else:
                alpha, beta = alpha[None,None,...], beta[None,None,...]
                dalpha, dbeta, dcost = dalpha[None,None,...], dbeta[None,None,...], dcost[None,None,...]
        
        dX, dY, depsi, dmu, dnu = None, None, None, None, None
        
        assert dcost.shape == (b,e), f"dcost has invalid shape {dcost.shape} != {(b,e)}"
        assert dalpha.shape == (b,e,n), f"dalpha has invalid shape {dalpha.shape} != {(b,e,n)}"
        assert dbeta.shape == (b,e,m), f"dbeta has invalid shape {dbeta.shape} != {(b,e,m)}"

        c = torch.cdist(X, Y, 2)**2
        k = torch.exp((alpha[:,:,:,None] + beta[:,:,None,:] - c[:,None,:,:]).to(dtype) / epsis[:,:,None,None])
        B = get_onb(n, dtype=dtype, device=device)[1:]

        assert(c.shape == (b,n,m))
        assert(k.shape == (b,e,n,m))
        assert(B.shape == (n-1,n))

        A = torch.eye(n-1, dtype=dtype, device=device) - torch.einsum("ij,bj,bejk,bk,belk,ml->beim", B, mu, k, nu, k, B)
        assert(A.shape == (b,e,n-1,n-1))

        rhs = torch.einsum("ij,bej->bei", B, dalpha.to(dtype)) - torch.einsum("ij,bj,bejk,bek->bei", B, mu, k, dbeta.to(dtype))
        assert(rhs.shape == (b,e,n-1))

        delta = torch.einsum("ji,bej->bei", B, torch.linalg.solve(A, rhs))
        assert(delta.shape == (b,e,n))

        # X needs grad
        if ctx.needs_input_grad[0]:
            kd1c = 2 * k[...,None] * (X[:,:,None,:] - Y[:,None,:,:])[:,None,...]

            assert(kd1c.shape == (b,e,n,m,d))

            dX = (torch.einsum("bei,beijk,bj->bik", delta, kd1c, nu)
                - torch.einsum("bel,belj,bj,beijk,bi->bik", delta, k, nu, kd1c, mu)
                + torch.einsum("bej,beijk,bi->bik", dbeta.to(dtype), kd1c, mu)
                + torch.einsum("be,beijk,bi,bj->bik", dcost, kd1c, mu, nu))

            assert(dX.shape == (b,n,d))

        # Y needs grad
        if ctx.needs_input_grad[1]:
            kd2c = -2 * k[...,None] * (X[:,:,None,:] - Y[:,None,:,:])[:,None,...]

            assert(kd2c.shape == (b,e,n,m,d))

            dY = (torch.einsum("bei,beijk,bj->bjk", delta, kd2c, nu)
                - torch.einsum("bel,belj,bj,beijk,bi->bjk", delta, k, nu, kd2c, mu)
                + torch.einsum("bej,beijk,bi->bjk", dbeta.to(dtype), kd2c, mu)
                + torch.einsum("be,beijk,bi,bj->bjk", dcost, kd2c, mu, nu))

            assert(dY.shape == (b,m,d))

        # epsi needs grad
        if ctx.needs_input_grad[2]:
            klogk = k * ((alpha[:,:,:,None] + beta[:,:,None,:] - c[:,None,:,:]).to(dtype) / epsis[:,:,None,None])

            assert(klogk.shape == (b,e,n,m))

            depsi = (torch.einsum("bei,beij,bj->be", delta, klogk, nu)
                   - torch.einsum("bei,beij,bj,belj,bl->be", delta, k, nu, klogk, mu)
                   + torch.einsum("bej,beij,bi->be", dbeta.to(dtype), klogk, mu)
                   + torch.einsum("be,beij,bi,bj->be", dcost, klogk, mu, nu))

            assert(depsi.shape == (b,e))

        # mu needs grad
        if ctx.needs_input_grad[3]:
            dmu = (torch.einsum("be,bel,belj,bj,beij->bei", epsis, delta, k, nu, k)
                 - torch.einsum("be,bej,beij->bei", epsis, dbeta.to(ptype), k)
                 + torch.einsum("be,bei->bei", dcost, alpha.to(ptype)))
            dmu -= torch.mean(dmu, dim=-1, keepdim=True)

            assert(dmu.shape == (b,e,n))

        # nu needs grad
        if ctx.needs_input_grad[4]:
            dnu = (-torch.einsum("be,bei,beij->bej", epsis, delta, k)
                 + torch.einsum("be,bei->bei", dcost, beta.to(ptype)))
            dnu -= torch.mean(dnu, dim=-1, keepdim=True)

            assert(dnu.shape == (b,e,m))

        if explicit_X_batch: # batch X
            pass
        elif dX is not None: # non-batch X
            dX = dX.squeeze(0)
            
        if explicit_Y_batch: # batch Y
            pass
        elif dY is not None: # non-batch Y
            dY = dY.squeeze(0)
        
        if implicit_mu: # no input mu
            dmu = None
        elif explicit_mu_batch: # batch mu
            pass
        elif dmu is not None: # non-batch mu
            dmu = dmu.squeeze(0)
            
        if implicit_nu:
            dnu = None
        elif explicit_nu_batch: # batch nu
            pass
        elif dnu is not None: # non-batch nu
            dnu = dnu.squeeze(0)
            
        if explicit_epsi_batch: # batch multi-epsis
            pass
        elif explicit_multi_epsi and depsi is not None: # non-batch multi-epsis
            depsi = depsi.squeeze(0)
        elif depsi is not None: # non-batch single epsis
            depsi = depsi.squeeze(0).squeeze(0)
            
        return dX, dY, depsi, dmu, dnu, None, None

def SolveEW2(X, Y, epsis, mu=None, nu=None, err=1e-4, verbose=False):
    """
    Calculate EOT transport between (weighted) point clouds (X, mu) and (Y, nu) with regularization strength epsi.
    - Works both for single problems and batch data.
    - Works both for single epsi and sequence of epsis.
    - Reference measure is mu x nu.
    - Cost ist squared Euclidean, equivalent to c = torch.cdist(X, Y, 2)**2.
    
    Parameters:
    X (tensor of shape (n,d) or (b,n,d)): first marginal points
    Y (tensor of shape (m,d) or (b,m,d)): second marginal points
    epsis (tensor of shape (), (e,) or (b,e)): entropic penalization strength(s). If multiple epsis are given (e > 1), they should always be sorted largest to smallest.
    mu (None or tensor of shape (n,) or (b,n)): first marginal point masses, default uniform probability measure
    nu (None or tensor of shape (m,) or (b,m)):  second marginal point masses, default uniform probability measure
    err (float): (approximate) allowable maximal relative marginal error, default 1e-4
    
    Returns:
    tuple of tensors (cost, alpha, beta): EOT transport cost, first dual, second dual
    - If used with batches, cost will have shape (b,e), alpha (b,e,n) and beta (b,e,m).
    - If used with single input and multiple epsis, cost will have shape (e,), alpha (e,n) and beta (e,m).
    - If used with single input and single epsi, cost will have shape (), alpha (n,) and beta (m,)
    """
    
    return EOT_L2sqr.apply(X, Y, epsis, mu, nu, err, verbose)
    
