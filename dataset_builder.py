#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 11:55:56 2020

description : Models and adds sample bias to epidemic data. 
author      : bveitch
version     : 1.0
project     : EpidPy (epidemic modelling in python)

Usage: 
    Create synthetics for data fitting
    
"""

import numpy as np;
import sir_modeller as sirm
import model_builder as mbuild
import sir_sampler as samplr
from plottingutils import plot_rsampled
from test_jacobian import dp_test

nstage=30
margs={'peakI':5,'peakR':15,'rateI':2,'rateR':0.5,'scaleI':2,'scaleR':1,'I0':1e-4}
istage=3
hstage=21
ntestmin=100
ntestmax=10000
pTT=0.95
pTF=0.01

#NOT used at present    
def estimate_sir_from_nstage(sir_type,nstage,model,scale=0.01):
    
    def est_sir(beta,gamma):
       return [sum[beta],sum[gamma]]
   
    def params_from_jstage(beta,gamma,jstage):
        if(jstage>0):
            b1=sum(beta[0:jstage])/(jstage)
            b2=sum(beta[jstage:nstage])/(nstage-jstage)
            g =sum(gamma[jstage:nstage])/(nstage-jstage)
            e =1/jstage
        else:
            b1=0
            b2=sum(beta)/nstage
            g=sum(gamma)/nstage
            e=1
        return [b1,b2,g,e]
    
    def est_seir(beta,gamma):
        boolb=beta < scale*np.max(beta)
        boolg=gamma < scale*np.max(gamma)
        both = boolb and boolg
        #find first False value
        jstage = next( (istage for istage, b in enumerate(both) if b == False), nstage)
        [b1,b2,g,e] = params_from_jstage(beta,gamma,jstage)
        return [b2,g,e]
                
    def est_si2r(beta,gamma):
        boolg=gamma < scale*np.max(gamma)
        #find first False value
        jstage = next((istage for istage, b in enumerate(boolg) if b == False), nstage)
        [b1,b2,g,e] = params_from_jstage(beta,gamma,jstage)
        return [b1,b2,g,e]
                
    estimates = {
            'sir'        : est_sir,
            'seir'       : est_seir,
            'si2r'       : est_si2r
        }
    
    estimator = estimates.get(sir_type, lambda: "Invalid modelling type")
    
    beta=model[0:nstage]
    gamma=model[nstage:2*nstage]
    
    return estimator(beta,gamma)

def build_data_nstage(args,Test=True):

    mod=mbuild.build_nstage_model(nstage,margs)
    nt =args['nt']
    dtype='sir_nstage'
    dargs={'type':dtype,'dt':args['dt'], 'nt':nt, 'ntsub': args['ntsub'],'nstage':nstage}
    sirdata=sirm.sir_mod(dargs)

    #Sampling operator with test
    truedata=sirdata.f(mod)
  
    d_intv=[[istage,hstage],[hstage,nstage+1]]
    samp  = samplr.Sampler(sirdata.dsize,d_intv)
    data_sum_intv = samp.F(truedata)
    
    infect_genpop = data_sum_intv[:,0]
    hospital_pop  = data_sum_intv[:,1]
    dtest=(ntestmax-ntestmin)/(nt-1)
    ntests = ntestmin+dtest*np.arange(0,nt)
    data_rsamp = samplr.randSampler(infect_genpop,pTT,pTF,ntests)
    
    recorded_data  = (data_rsamp+hospital_pop).reshape(nt,1)
    
    t=np.arange(0,nt,1)
    plot_rsampled(t,[infect_genpop,hospital_pop],[data_rsamp,hospital_pop],recorded_data,np.sum(truedata[:,1:nstage+1],1))
    if(Test):
        dpargs={'mod': truedata, 'data': data_sum_intv, 'dptol':1e-11}
        dp_pass=dp_test(samp.F,samp.Ft,dpargs)
        if(dp_pass):
            print('dotproduct passed')
        else:
            print('dotproduct failed')
    return recorded_data