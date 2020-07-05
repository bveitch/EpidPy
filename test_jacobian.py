#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
description : General purpose tests for nonlinear operators and Jacobians
author      : bveitch
version     : 1.0
project     : EpidPy (epidemic modelling in python)

lin_test: test that Jacobian approximates derivative of nonlinear operator

    F(x) approx [f(x+eps del_x) - f(x-eps del_x)]/eps

dp_test : test that Jacobian matrix passed dot product test
    
    d^T F(x) x = x^T F(x)^T d    

"""
import numpy as np;
import sir_update as sir
import math

#def lin_test_sir(sir_object, args): 
#    print("Linearization test")
#    mod0 =args['mod0']
#    dmod =args['dmod']
#    niter=args['niter']  
#    tol  =args['lintol']  
#    dataL=sir_object.F(mod0,dmod)
#    eps=1
#    err_prev=1
#    for i in range(niter):
#        modp=[x + eps*y for x,y in zip(mod0,dmod)]
#        modm=[x - eps*y for x,y in zip(mod0,dmod)]
#        datap = sir_object.f(modp)
#        datam = sir_object.f(modm)
#        diff = (datap-datam)/(2*eps)
#        j0=math.sqrt(np.sum(diff**2))
#        err   = j0 - math.sqrt(np.sum(dataL**2))
#        ratio = err_prev/err
#        lin_test_pass = abs(ratio-2.0) < tol
#        print(eps, j0, math.sqrt(np.sum(dataL**2)), err,lin_test_pass)
#        err_prev= err
#        eps /= 2
#    return lin_test_pass
#    
#def dp_test_sir(sir_object,args):  
#    print("Dot-product test")
#    mod0 = args['mod0']
#    m1   = sir_object.msize
#    nt   = sir_object.nt
#    d1   = sir_object.dsize
#    tol  = args['dptol']
#    #tol=1e-12
#    #m1=len(m)
#    dmod = np.random.randn(m1)
#    #d = 2*np.random.random_sample(d.shape) -1  
#    data = 2*np.random.random_sample((nt,d1)) -1 
#    Fm = sir_object.F( mod0,dmod)
#    Ftd= sir_object.Ft(mod0,data)
#    dTFm  = np.dot(np.ravel(data).T,np.ravel(Fm))
#    mTFtd = np.dot(Ftd.T,dmod) 
#    diff  = dTFm - mTFtd
#    print(dTFm, mTFtd, abs(diff) )
#    return ( abs(diff) < tol )

def lin_test(nlobject, args): 
    print("Linearization test")
    mod0 =args['mod0']
    dmod =args['dmod']
    niter=args['niter']  
    tol  =args['lintol']  
    dataL=nlobject.F(mod0,dmod)
    eps=1
    err_prev=1
    for i in range(niter):
        modp=[x + eps*y for x,y in zip(mod0,dmod)]
        modm=[x - eps*y for x,y in zip(mod0,dmod)]
        datap = nlobject.f(modp)
        datam = nlobject.f(modm)
        diff = (datap-datam)/(2*eps)
        j0=math.sqrt(np.sum(diff**2))
        err   = j0 - math.sqrt(np.sum(dataL**2))
        ratio = err_prev/err
        
        print(eps, j0, math.sqrt(np.sum(dataL**2)), err,abs(ratio-4.0))
        if(abs(ratio-4.0) < tol):
            return True
        else:
            err_prev= err
            eps /= 2
    return False

def dp_test(F,Ft,args):  
    print("Dot-product test")
    m0 = args['mod']
    d0 = args['data']
    tol  = args['dptol']
    dmod = np.random.random_sample(m0.shape)
    data = 2*np.random.random_sample(d0.shape) -1 
    Fm = F( dmod)
    Ftd= Ft(data)
    dTFm  = np.dot(np.ravel(data).T,np.ravel(Fm))
    mTFtd = np.dot(np.ravel(Ftd).T,np.ravel(dmod)) 
    diff  = dTFm - mTFtd
    print(dTFm, mTFtd, abs(diff) )
    return ( abs(diff) < tol )