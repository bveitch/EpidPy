#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 20:37:25 2020

description : Provides nonlinear operators, and Jacobians for SIR Networks, wrapping SIR updates. Wrapped into sir_modeller 
author      : bveitch
version     : 1.0
project     : EpiPy (epidemic modelling in python)
"""
import numpy as np
import sir_update as sirup
from functools import partial

class GenericNetworkUpdater:
    
    def __init__(self, args):
        self.sirtype = args['sirtype']
        assert(self.L.shape[1] == self.nv),'Laplacian must be square'
        if( self.sirtype == 'sir'):
            self.dsize =3
            self.msize =2
            self.upd   = sirup.sir_nonlinear
            self.jac   = sirup.jacobian_sir
            self.jac_m = partial(sirup.jacobian_mod, self.msize)
        elif(self.sirtype == 'seir'):
            self.dsize =4
            self.msize =3  
            self.upd   = sirup.seir_nonlinear 
            self.jac   = sirup.jacobian_seir
            self.jac_m = partial(sirup.jacobian_mod_seir, self.msize)
        elif(  self.sirtype == 'sirs'):
            self.dsize =3
            self.msize =3 
            self.upd  = sirup.sirs_nonlinear 
            self.jac  = sirup.jacobian_sirs
            self.jac_m= partial(sirup.jacobian_mod_sirs, self.msize)
        elif(self.sirtype == 'si2r'):
            self.dsize =4
            self.msize =4   
            self.upd   = sirup.si2r_nonlinear
            self.jac   = sirup.jacobian_si2r
            self.jac_m = partial(sirup.jacobian_mod_si2r, self.msize)
        elif(self.sirtype == 'sir_nstage'):
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
        self.ntot=self.nv*self.dsize

    def network_nonlinear(self,dt,mod,sir):
        ntot_=self.ntot
        assert(sir.shape[0]==ntot_),'sir size doesnt match number of graph vertices'
        assert(len(mod)==self.msize),'mod size doesnt match for sir class type'

        nv_   =self.nv
        dsize_=self.dsize
        msize_=self.msize
        sir2=sir.reshape(nv_,dsize_)
        print(sir2.shape)
        for iv in range(nv_):
            print(sir2[iv,:])
            self.upd(dt,mod,sir2[iv,:])
    
        
        Li=np.zeros(ntot_)
        diffusivity=0
        for j in range(1,dsize_-1):
            j0=j*nv_
            j1=(j+1)*nv_
            Li[j0:j1]=-diffusivity*np.dot(self.L,sir[j0:j1])
        
        sir  = sir2.reshape(dsize_*nv_) 
        print(sir.shape)
        sir += dt*Li
    
    def jacobian_network(self,mod,sir): 
        raise Exception("jacobian_network untested") 
        ntot_=self.nv*self.dsize
        nv_=self.nv 
        dsize_=self.dsize

        sir2=sir.reshape(dsize_,nv_)
        jac0=sir.zeros((dsize_,dsize_,nv_))
    
        for iv in range(nv_):
            jac0[:,:,iv]=self.jac(mod,sir2[:,iv])
    
        jac=np.zeros((ntot_,ntot_))
        irange=range(1,dsize_-1)
        for j in range(dsize_):
            for k in range(dsize_):
                j0=j*nv_
                j1=(j+1)*nv_
                k0=k*nv_
                k1=(k+1)*nv_
                jac[j0:j1,k0:k1]=np.diag(jac0[j,k,:])
                if(j in irange and k in irange):
                    jac[j0:j1,k0:k1]+=self.L
    
        return jac;

    def jacobian_mod_network(self,m,sir):   
        raise Exception("jacobian_mod_network untested") 
        ntot_=self.nv*self.dsize
        nv_=self.nv 
        dsize_=self.dsize
        sir2=sir.reshape(dsize_,nv_)
    
        jacm=sir.zeros((nv_,dsize_,m))
        for i in range(nv_):
            jacm[i,:,0:m-1] += self.jac_m(m,sir2[:,i])
            jacm[i,:,m]     += self.L
    
        jacm.reshape(ntot_,m)
    
        return jacm; 