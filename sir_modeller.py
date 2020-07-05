# -*- coding: utf-8 -*-
"""

description : Wrapper for SIR modelling modes 
author      : bveitch
version     : 1.0
project     : EpiPy (epidemic modelling in python)

Supported modelling types:
    sir        : susceptible, infected ,            recovered
    seir       : susceptible, exposed  , infected , recovered
    sirs       : susceptible, infected ,            recovered, susceptible 
    si2r       : susceptible, infected1, infected2, recovered
    sir_nstage : susceptible, infected (n stages) , recovered
"""

import numpy as np;
import math
import sir_update as sirup;
from functools import partial

def check_nan(sir):
        for v in sir:
            if(math.isnan(v)):
                return True
            elif(math.isinf(v)):
                return True
        return False  

def push_ic(mod,sir): 
    for ic in range(len(mod)):
        sir[0]    -=mod[ic]
        sir[ic+1] +=mod[ic]
                
def pull_ic(sir,mod):
    for ic in range(len(mod)):
        mod[ic]    += sir[ic+1] -sir[0]

class sir_mod:  
    
    def __init__(self, args):
        self.type  = args['type']
        self.dt    = args['dt']
        self.nt    = args['nt']
        self.ntsub = args['ntsub']
        if( self.type == 'sir'):
            self.dsize =3
            self.msize =2
            self.upd   = sirup.sir_nonlinear
            self.jac   = sirup.jacobian_sir
            self.jac_m = partial(sirup.jacobian_mod, self.msize)
        elif(  self.type == 'seir'):
            self.dsize =4
            self.msize =3  
            self.upd   = sirup.seir_nonlinear 
            self.jac   = sirup.jacobian_seir
            self.jac_m = partial(sirup.jacobian_mod_seir, self.msize)
        elif(  self.type == 'sirs'):
            self.dsize =3
            self.msize =3 
            self.upd  = sirup.sirs_nonlinear 
            self.jac  = sirup.jacobian_sirs
            self.jac_m= partial(sirup.jacobian_mod_sirs, self.msize)
        elif(self.type == 'si2r'):
            self.dsize =4
            self.msize =4   
            self.upd   = sirup.si2r_nonlinear
            self.jac   = sirup.jacobian_si2r
            self.jac_m = partial(sirup.jacobian_mod_si2r, self.msize)
        elif(self.type == 'sir_nstage'):
            nstage = args['nstage']
            self.dsize = nstage+2
            self.msize = 2*nstage
            Diag  = np.concatenate( ( - np.ones((nstage -1)) , np.array([0]) ) )
            Imat  = np.diag( Diag )
            Imat += np.diag( np.ones((nstage-1)) , -1 )
            self.upd   = partial(sirup.sir_nstage_nonlinear,Imat)
            self.jac   = partial(sirup.jacobian_sir_nstage, Imat)
            self.jac_m = partial(sirup.jacobian_mod_sir_nstage, Imat, self.msize)
        else:
            raise ValueError("Invalid modelling type")

    
    def update_f(self,mod,sir0):
        self.upd(self.dt,mod,sir0)
   
    def f(self,mod):
        sir0=np.zeros(self.dsize)
        data=np.zeros((self.nt,self.dsize))
        
        mtsize = len(mod)
        assert ( (mtsize > self.msize) ),"must have at least one initial condition!" 
        assert ( (mtsize < self.msize + self.dsize) ),"too many initial conditions for data!"
        m=mod[0:self.msize]
        ic=mod[self.msize:mtsize]
        
        sir0[0]=1
        push_ic(ic,sir0)
        
        ntot=self.nt*self.ntsub
        for it in range(ntot):
            if(it % self.ntsub==0):
                jt=int(it/self.ntsub)
                data[jt,:]=sir0
            
            self.update_f(m,sir0)
            if(check_nan(sir0)):
                del sir0
                return np.zeros((self.nt,self.dsize))
        del sir0
        return data
    
    def update_F(self,mod0,sir0,dmod,dsir):
        dsir += self.dt*np.dot(self.jac( mod0,sir0),dsir)
        dsir += self.dt*np.dot(self.jac_m(    sir0),dmod)
            
    def F(self,mod0,dmod):
        
        sir0 = np.zeros(self.dsize)
        dsir = np.zeros(self.dsize)
        data = np.zeros((self.nt,self.dsize))

        mtsize = len(mod0)
        assert ( (mtsize > self.msize) ),"must have at least one initial condition!" 
        assert ( (mtsize < self.msize + self.dsize) ),"too many initial conditions for data!"
        assert (len(dmod) == mtsize),"dmod size doesnt match mod0 size!"
        m0=mod0[0:self.msize]
        dm=dmod[0:self.msize]
        ic0=mod0[self.msize:mtsize]
        dic=dmod[self.msize:mtsize]
     
        sir0[0]=1
        push_ic(ic0,sir0)
        push_ic(dic,dsir)

        ntot=self.nt*self.ntsub
        for it in range(ntot): 
            if((it % self.ntsub)==0):
                jt=int(it/self.ntsub)
                data[jt,:]=dsir
            self.update_F(m0,sir0,dm,dsir)
            self.update_f(m0,sir0)
            #self.update_F(m0,sir0,dm,dsir) -minor bug
       
        del sir0
        del dsir
        return data

    def update_Ft(self,mod0,sir0,dsir,dmod):
        dmod += self.dt*np.dot(self.jac_m(    sir0).T,dsir)
        dsir += self.dt*np.dot(self.jac( mod0,sir0).T,dsir)
    
    def Ft(self,mod0,data): 
       
        sir0    = np.zeros(self.dsize)   
        mtsize = len(mod0)
        assert ( (mtsize > self.msize) ),"must have at least one initial condition!" 
        assert ( (mtsize < self.msize + self.dsize) ),"too many initial conditions for data!"
      
        m0=mod0[0:self.msize]
        ic0=mod0[self.msize:mtsize]
     
        sir0[0]=1
        push_ic(ic0,sir0)
        
        ntot=self.nt*self.ntsub
        sirdata = np.zeros((ntot,self.dsize))
        for it in range(ntot):
            sirdata[it,:]= sir0
            self.update_f(m0,sir0)

        dmod = np.zeros(mtsize)
        dsir = np.zeros(self.dsize)
        for it in reversed(range(ntot)):
            #self.update_Ft(m0,sir0,dsir,dmod[0:self.msize])-minor bug corrected for above
        
            sir0 = sirdata[it,:]
            self.update_Ft(m0,sir0,dsir,dmod[0:self.msize])
            if(it% self.ntsub==0):
                jt=int(it/self.ntsub)
                dsir += data[jt,:]
            if(it==0):
                pull_ic(dsir,dmod[self.msize:mtsize])
            
        del sir0
        del dsir
        del sirdata
        return dmod
    
    