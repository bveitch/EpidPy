#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 19:44:26 2020

description : Wrapping SIR modeller into nonlinear least-squares data fitting 
author      : bveitch
version     : 1.0
project     : EpidPy (epidemic modelling in python)

Supported operations:
    
Objfn: 
    Least squares objective function
    obj(x)=|W( S*sir_fx(x) -d )|_2^2
Jacobian:
    jac(x)=Grad_x obj(x)    

x      : model space from sir_modeller.py
d      : data to fit
sir_fx : (sirmod) sir operator from sir_modeller.py, function from x to sir space
S      : (sampler) sampling function (sampler.py) from sir space to space of d
W      : (wts) diagonal matrix - weighting function of time samples
     
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import check_grad

import time

class FixParameters:
    
    def __init__(self, fixpars):
        self.fix  = fixpars

    def F(self,params):
        for ikey in self.fix.keys:
            params[ikey]=self.fix[ikey]

    def FT(self,params):
        for ikey in self.fix.keys:
            params[ikey]=0
            
class SIR_Solver:    
    
    obj_list=[]
    xk_list=[]
    k_iter=0
    
    def __init__(self, d, sirmod, sampler = None,wts=None):
           
        self.d      = d
        self.sirmod = sirmod
        self.S      = sampler
                
        try:
            if( wts.any() ) :  
                assert (wts.shape == self.d.shape),"data shape doesnt match wts"
                self.wts=wts
        except AttributeError: 
            self.wts=np.ones(d.shape)   
        if(self.S == None):
            assert ((self.sirmod.nt,self.sirmod.dsize) == self.d.shape),"data shape doesnt match Fm"
        else:
            assert ((self.sirmod.nt,self.S.outsize) == self.d.shape),"data shape doesnt match SFm"
        
        
    def callback(self,xk, ):
        self.xk_list.append(xk)
        obj=self.Objfn(xk)
        self.k_iter+=1
        print(self.k_iter,obj)
        self.obj_list.append(obj)
    
    def Objfn(self,x,args=None):
        fx=self.sirmod.f(x)
        if(self.S == None):
            rfx=self.wts*(fx-self.d)
            r=np.ravel(rfx)
        else:
            Sfx=self.S.F(fx)
            rSfx=self.wts*(Sfx-self.d)
            r=np.ravel(rSfx)
        return 0.5*sum(r**2)

    def Jacobian(self,x,args=None): 
        fx=self.sirmod.f(x)
        if(self.S == None):
            r=self.wts*(fx-self.d)     
            return self.sirmod.Ft(x,self.wts*r)
        else:
            Sfx=self.S.F(fx)
            r=self.wts*(Sfx-self.d)
            Str = self.S.Ft(self.wts*r)
        return self.sirmod.Ft(x,Str)
    
    def test(self,args):
        print("Objective function test")
        x0 = args['x0']
        np.random.seed(0)
        dx =np.random.randn(len(x0))
        niter=args['niter']  
        tol  =args['tol']  
        grad=self.Jacobian(x0)
        eps=1
        err_prev=1
        for i in range(niter):
            xp   = [x + eps*y for x,y in zip(x0,dx)]
            xm   = [x - eps*y for x,y in zip(x0,dx)]
            objp = self.Objfn(xp)
            objm = self.Objfn(xm)
            diff = (objp-objm)/(2*eps)
            err  = diff - np.dot(grad,dx)
            rat  = err_prev/err
            print(eps, diff, np.dot(grad,dx), err,rat)
            if(abs(rat-4.0) < tol):
                return True
            else:
                err_prev= err
                eps /= 2
        return False
        
    def check_grad(self,args):
        print("SciPy's Gradient test")
        x = args['x0']
        v = check_grad(self.Objfn, self.Jacobian, x)
        print(v)
        if(v < args['tol']):
            return True
        else:
            return False
        
    def init_mod(self):
        mod=[0 for ic in range(self.xsize)]
        return mod
                
    
    def solve(self,x0,options0):
        print('Starting solver')
        start = time.time()
        ##options={'gtol': 1e-05, 'norm': inf, 'eps': 1.4901161193847656e-08, 'maxiter': None, 'disp': False, 'return_all': False}
        res=minimize(self.Objfn, x0, args=None, method='CG', jac=self.Jacobian, tol=None, callback=self.callback, options=options0)
        end = time.time()
        runtime = end - start
        print('Solver finished')
        objmin=self.obj_list[-1]/self.obj_list[0]
        xsol=res["x"]     
        sol = {'xsol':xsol, 'runtime':runtime,'objmin':objmin}
        return sol

    def plot_convergence(self,fname=None):
        fig, ax=plt.subplots(1,1,squeeze=False, figsize=(18,10))
        obj0 =self.obj_list[0]
        niter=len(self.obj_list)
        iters=np.arange(0,niter,1)
        ax[0][0].set_title("Convergence for {:} fit".format(self.sirmod.type),fontsize=10)
        ax[0][0].set_xlabel('Iterations')
        ax[0][0].set_ylabel('Error')
        ax[0][0].plot(iters, self.obj_list/obj0, 'b-')
        plt.show()
        if(fname is not None):
            plt.savefig(fname)
        
        
