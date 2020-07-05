#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 08:46:30 2020

@author: myconda

description : Convenience functions for building sir models for testing  
author      : bveitch
version     : 1.0
project     : EpiPy (epidemic modelling in python)
"""

import numpy as np
import random
import math
from scipy.special import gamma as Gamma
from plottingutils import plot_beta_gamma

I0=0.01
I1=0.001
beta0=0.14
gamma0=0.057
delta0=0.1
nstage=30
    
def eval_Gamma(n,a,b):
    y=math.pow(a,b) *math.exp(-a*n)*math.pow(n,b-1)/(Gamma(b))
    return y  

def build_nstage_model(nstage,args): 
    beta=np.zeros((nstage))
    gamma=np.zeros((nstage))

    shI=args['peakI']
    shR=args['peakR']
    a  =args['rateI']
    c  =args['rateR']
    b  =a*shI+1
    d  =c*shR+1
    scI=args['scaleI']
    scR=args['scaleR']
    i0 =args['I0']
    
    for istage in range(nstage):
        beta[istage]  = eval_Gamma(istage,a,b)
        gamma[istage] = eval_Gamma(istage,c,d)
    
    plot_beta_gamma(beta,gamma)
    
    mod=np.zeros(2*nstage+1) 
    mod[0:nstage]=scI*beta/sum(beta)
    mod[nstage:2*nstage]=scR*gamma/sum(gamma)
    mod[2*nstage]=i0
    return mod

def sir_mod():
    return [beta0,gamma0,I0]
            
def seir_mod():
    return [beta0,delta0,gamma0,I0,I1]

def sirs_mod():
     A=delta0
     return [beta0,gamma0,A,I0]

def si2r_mod():
     return [0.5*beta0,0.5*beta0,delta0,gamma0,I0,I1]
 
def sir_nstage_mod():
    args={'peakI':5,'peakR':15,'rateI':2,'rateR':0.5,'scaleI':(beta0/gamma0),'scaleR':1,'I0':I0}
    return build_nstage_model(nstage,args)   

def init_mod(sir_type):
     models = {
            'sir'        : sir_mod,
            'seir'       : seir_mod,
            'sirs'       : sirs_mod,
            'si2r'       : si2r_mod,
            'sir_nstage' : sir_nstage_mod
        }
     mod = models.get(sir_type, lambda: "Invalid modelling type")
     return mod()

def init_dmod(mod):
    dmod=[]
    for m in mod:
        dmod.append( m * random.random() )
    return dmod    