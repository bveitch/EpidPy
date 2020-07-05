#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 10:17:51 2020

description : Sampling methods from sir space to data space 
author      : bveitch
version     : 1.0
project     : EpidPy (epidemic modelling in python)

Usage:
    Data fitting in least squares fit
    Also acts on labels for data description and fitting
    randSampler: adds random noise/sampling bias to synthetics
"""

import numpy as np;

def labels_from_type(sir_type,nstage=0):
    if(sir_type == 'sir'):
         return list(sir_type)
    elif(sir_type == 'seir'):
        return list(sir_type)
    elif(sir_type == 'sirs'):
        return list(sir_type) 
    elif(sir_type == 'si2r'):
        return ['s','i1','i2','r'] 
    elif(sir_type == 'sir_nstage'):
        assert(nstage>0),'nstage must be >0'
        labels=['s']
        [labels.append('i'+str(istage)) for istage in range(nstage)]
        labels.append('r')
        return labels
    else:
       raise ValueError("Invalid modelling type")

class Sampler:
    
    def __init__(self, nsize,intervals): 
        for ival in intervals:
            assert(ival[1] > ival[0]),'interval must be increasing'
        self.insize     = nsize
        self.intervals  = intervals
        self.outsize    = len(intervals)
        
    def F(self,data):
        [nt,nsize]=data.shape
        assert(nsize == self.insize),"data size doesnt match insize"
        data_samp=np.zeros((nt,self.outsize))
        icount=0
        for ival in self.intervals:
            isum = np.zeros(nt) 
            for i in range(ival[0],ival[1]):
                isum += data[:,i]
#           isum=np.sum(data[:,ival[0]:ival[1]],1)
            data_samp[:,icount]=isum 
            icount+=1
        return data_samp

    def Ft(self,data_samp):
        [nt,isize]=data_samp.shape
        assert(isize == self.outsize),"data sampled size doesnt match outsize"
        data=np.zeros((nt,self.insize))
        icount=0
        for ival in self.intervals:
            isum = data_samp[:,icount]
#            data[:,ival[0]:ival[1]] +=isum
            for i in range(ival[0],ival[1]):
                data[:,i]+=isum
            icount+=1
        return data

    def Flabels(self,labels):
        sampled_labels=[]
        for ival in self.intervals:
            if(ival[1] == ival[0]+1):
                l0=labels[ival[0]]
                sampled_labels.append(l0)
            else:    
                l0=labels[ival[0]  ]
                l1=labels[ival[1]-1]
                sampled_labels.append(l0 + '-' + l1) 
        return sampled_labels

    def Flabels_from_type(self,sir_type,nstage=0):
        labels=labels_from_type(sir_type,nstage)
        return self.Flabels(labels)
    
#def draw_samples(pT,n,Ntests):
#    np.random.seed(0) 
#    s         = np.random.normal(0, 1,n)
#    mu        = Ntests*pT
#    sigma     = np.sqrt(Ntests*pT*(1-pT))
#    data_samp = (sigma*s+mu)/Ntests
#    return data_samp
    
def randSampler(data,pTT,pTF,Ntests):
    
    def draw_samples(pT,n,Ntests):
        np.random.seed(0) 
        s         = np.random.normal(0, 1,n)
        mu        = Ntests*pT
        sigma     = np.sqrt(Ntests*pT*(1-pT))
        data_samp = (sigma*s+mu)/Ntests
        return data_samp
    
    if(len(data.shape)==1):
        n=data.shape[0]
        pT=pTT*data+pTF*(1-data)
        
        data_samp=draw_samples(pT,n,Ntests)
        return data_samp
    else:
        [n,m]=data.shape
        assert (m == len(pTT)),"data size doesnt match size pTT"
        pT=np.zeros(n)
        for i in range(m): 
            pT=pTT[i]*data[:,i]+pTF*(1-np.sum(data,1))
       
        data_samp=draw_samples(pT,n,Ntests)
        return data_samp


   