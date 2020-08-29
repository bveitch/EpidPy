# -*- coding: utf-8 -*-
"""

description : Wrapper for modelling SIR on networks 
author      : bveitch
version     : 1.0
project     : EpidPy (epidemic modelling in python)

Supported modelling types:
    sir        : susceptible, infected ,            recovered
    seir       : susceptible, exposed  , infected , recovered
    sirs       : susceptible, infected ,            recovered, susceptible 
"""

import numpy as np
import math
import network_update as sirNup
from functools import partial

def check_nan(sir):
        for v in sir:
            if(math.isnan(v)):
                return True
            elif(math.isinf(v)):
                return True
        return False  




class sir_mod:  
    
    def __init__(self, args,Lapl,pops):
        self.type  = args['type']
        self.dt    = args['dt']
        self.nt    = args['nt']
        self.ntsub = args['ntsub']
        self.L     = Lapl
        self.nv    = Lapl.shape[0]
        self.pops  = pops
        assert(self.L.shape[1] == self.nv),'Laplacian must be square'
        if( self.type == 'sir'):
            self.dsize =3
            self.msize =3
            self.upd   = partial(sirNup.netw_sir_nonlinear,self.dt,self.L)
        elif( self.type == 'seir'):
            self.dsize =4
            self.msize =4  
            self.upd   = partial(sirNup.netw_seir_nonlinear,self.dt,self.L)
        elif( self.type == 'sirs'):
            self.dsize =3
            self.msize =4 
            #self.upd   = partial(sirNup.netw_sirs_nonlinear,self.dt,self.L) 
            self.upd   = partial(sirNup.netw_sirs_nonlinear,self.dt,self.L)
        else:
            raise ValueError("Invalid modelling type")
        self.ntot=self.nv*self.dsize
   
    def push_ic(self,icases,sir):
        sir[0      :  self.nv] -= icases
        sir[self.nv:2*self.nv] += icases
            
    def update_f(self,mod,sir0):
        self.upd(mod,sir0)
   
    def f(self,mod):
        sir0=np.zeros(self.ntot)
        data=np.zeros((self.nt,self.ntot))
        
        mtsize = len(mod)
        assert ( (mtsize == self.msize + 1) ),"model must have one initial condition!" 
        
        m=mod[0:self.msize]
        ic=mod[self.msize]
        assert ( (len(ic) == self.nv) ),"each node must have one initial condition!" 
        sir0[0:self.nv]=self.pops
        self.push_ic(ic,sir0)
        
        nt_tot=self.nt*self.ntsub
        for it in range(nt_tot):
            if(it % self.ntsub==0):
                jt=int(it/self.ntsub)
                data[jt,:]=sir0
            
            self.update_f(m,sir0)
            if(check_nan(sir0)):
                del sir0
                return np.zeros((self.nt,self.ntot))
        del sir0
        return data
    


    