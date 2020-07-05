# -*- coding: utf-8 -*-
"""
description : Provides nonlinear operators, and Jacobians for SIR updates. Wrapped into sir_modeller 
author      : bveitch
version     : 1.0
project     : EpiPy (epidemic modelling in python)

"""

import numpy as np;

"""
SIR:
sir[3]: susceptible, infected, recovered
mod[2]: infectivity, recovery (as rates per unit population)
"""

def sir_nonlinear(dt,mod,sir):
    s=sir[0]
    i=sir[1]
    sir[0] += -dt*mod[0]*s         *i
    sir[1] +=  dt*(mod[0]*s-mod[1])*i
    sir[2] +=  dt*mod[1]           *i

def jacobian_sir(mod,sir):
    n=sir.shape[0]
    jac=np.zeros((n,n))
    jac[0,0] = - mod[0]*sir[1]
    jac[0,1] = - mod[0]*sir[0]
    jac[1,0] =   mod[0]*sir[1]
    jac[1,1] =   mod[0]*sir[0]-mod[1]
    jac[2,1] =   mod[1]
    return jac;

def jacobian_mod(m,sir):
    n=sir.shape[0]
    jacm=np.zeros((n,m))
    jacm[0,0] = - sir[0]*sir[1]
    jacm[1,0] =   sir[0]*sir[1]
    jacm[1,1] = -        sir[1]
    jacm[2,1] =          sir[1]
    return jacm;   

"""
SEIR:
seir[4]: susceptible, exposed, infected, recovered
mod[3] : infectivity, progression (exposed to infected), recovery (as rates per unit population)
"""           
def seir_nonlinear(dt,mod,seir):
    s =seir[0]
    e =seir[1]
    i =seir[2]
    seir[0] += -dt*mod[0]*s*i 
    seir[1] +=  dt*mod[0]*s*i - dt*mod[1]*e
    seir[2] +=  dt*mod[1]*e   - dt*mod[2]*i 
    seir[3] +=  dt*mod[2]*i

def jacobian_seir(mod,seir):
    n=seir.shape[0]
    jac=np.zeros((n,n))
    jac[0,0] = - mod[0]*seir[2] 
    jac[0,2] = - mod[0]*seir[0]
    jac[1,0] =   mod[0]*seir[2]
    jac[1,1] = - mod[1]
    jac[1,2] =   mod[0]*seir[0]
    jac[2,1] =   mod[1]
    jac[2,2] = - mod[2]
    jac[3,2] =   mod[2]
    return jac;

def jacobian_mod_seir(m,seir):
    n=seir.shape[0]
    jacm=np.zeros((n,m))
    jacm[0,0] = - seir[0]*seir[2]
    jacm[1,0] =   seir[0]*seir[2]
    jacm[1,1] = - seir[1]
    jacm[2,1] =   seir[1]
    jacm[2,2] = - seir[2]
    jacm[3,2] =   seir[2]
    return jacm;   
"""
SI2R
si2r[4]: susceptible, early stage infectives, late stage infectives, recovered
mod[4] : infectivity (early stage), infectivity (late stage), progression (early to late), recovery (as rates per unit population)
"""           
def si2r_nonlinear(dt,mod,si2r):
    s =si2r[0]
    i1=si2r[1]
    i2=si2r[2]
   
    si2r[0] += -dt*s*(mod[0]*i1 + mod[1]*i2)
    si2r[1] +=  dt*s*(mod[0]*i1 + mod[1]*i2) - dt*mod[2]*i1
    si2r[2] +=  dt*mod[2]*i1                 - dt*mod[3]*i2 
    si2r[3] +=                                 dt*mod[3]*i2

def jacobian_si2r(mod,si2r):
    n=si2r.shape[0]
    jac=np.zeros((n,n))
    jac[0,0] = - mod[0]*si2r[1] - mod[1]*si2r[2]
    jac[0,1] = - mod[0]*si2r[0]
    jac[0,2] = - mod[1]*si2r[0]
    jac[1,0] =   mod[0]*si2r[1] + mod[1]*si2r[2]
    jac[1,1] =   mod[0]*si2r[0] - mod[2]
    jac[1,2] =   mod[1]*si2r[0]
    jac[2,1] =   mod[2] 
    jac[2,2] = - mod[3]
    jac[3,2] =   mod[3]
    return jac;

def jacobian_mod_si2r(m,si2r):
    n=si2r.shape[0]
    jacm=np.zeros((n,m))
    jacm[0,0] = - si2r[0]*si2r[1]
    jacm[0,1] = - si2r[0]*si2r[2]
    jacm[1,0] =   si2r[0]*si2r[1]
    jacm[1,1] =   si2r[0]*si2r[2]
    jacm[1,2] = - si2r[1]
    jacm[2,2] =   si2r[1]
    jacm[2,3] = - si2r[2]
    jacm[3,3] =   si2r[2]
    return jacm;   

"""
SIR_Nstage
sir_nstage[nstage+2]: susceptible, infectives (n stages), recovered
mod[2 nstage]       : infectivity (nstages), recovery (nstages), [progression=fixed] (as rates per unit population)
Slow and fast infection update modes
"""  

def updateI(I, S, Upmat, gamma, bI, dt):
    I2    = np.zeros(I.shape)
    I2   += I
    I2[0]+= dt*S*bI
    I2   += dt*Upmat.dot(I) - dt*np.multiply(gamma,I)
    return I2

def jac_updateI(I, S, Upmat, gamma, beta):
    n=I.shape[0] 
    jacI=np.zeros((n,n+1))
    
    jacI[0,0]   = float(np.dot( beta.T ,I))
    jacI[0,1:n+1] = S*beta.T
    jacI[:,1:n+1] += Upmat- np.diag(gamma)
    return jacI

def jac_m_updateI(m,I, S):
    n=I.shape[0] 
    jacm=np.zeros((n,m))
    jacm[0,0:n]   += S*I.T
    jacm[0:n,n:m] -= np.diag(I)
    return jacm

def updateI_slow(I,S, gamma, bI, dt):
    
    nstage = I.shape[0]
    I[0] = (1-dt)*I[0] + dt*S*bI - dt*gamma[0]*I[0]
    for istage in range(1,nstage-1):
        I[istage] = dt*I[istage-1] +(1-dt)*I[istage]- dt*gamma[istage]*I[istage]
    I[nstage-1] = dt*I[nstage-2] + I[nstage-1]- dt*gamma[nstage-1]*I[nstage-1]

def jac_updateI_slow(I,S, gamma, beta):
    n = I.shape[0]
    jacI=np.zeros((n,n+1))
    jacI[0,0]   += float(np.dot( beta.T ,I))
    jacI[0,1:n+1] += S*beta.T
    jacI[0,1]   += -1-gamma[0] 
   
    for istage in range(1,n):
        jacI[istage,istage  ] +=  1 
        jacI[istage,istage+1] += -1-gamma[istage]
    jacI[n-1,n-1] += 1 
    jacI[n-1,n  ] += -gamma[n-1]
    return jacI
    
def jac_m_updateI_slow(m,I,S):
    
    n = I.shape[0]
    jacm=np.zeros((n,m))
    jacm[0,0:n] = S*I.T
   
    for istage in range(0,n):
        jstage=n+istage
        jacm[istage,jstage] =  -I[istage] 
    
def sir_nstage_nonlinear(Imat, dt,mod,sir):
    
    nstage=sir.shape[0]-2
    if(Imat is not None):
        [n,m]=Imat.shape
        assert ([nstage,nstage] != Imat.shape),"Imat shape doesnt match nstages!"
    msize=len(mod)
    beta=np.array(mod[0:nstage])
    gamma=np.array(mod[nstage:msize])
    s=sir[0]
    i=sir[1:nstage+1]
    
    b=float(np.dot( beta.T ,i))
    c=float(np.dot(gamma.T ,i))
       
    sir[0] -= dt*s*b 
    if (Imat is not None):
        sir[1:nstage+1]  = updateI(i,s, Imat, gamma, b, dt)
    else:
        updateI_slow(sir[1:nstage+1],s, gamma, b, dt)
    sir[nstage+1] += dt*c  

def jacobian_sir_nstage(Imat,mod0,sir0):
    n=sir0.shape[0]
    nstage=n-2
    if(Imat is not None):
        assert ([nstage,nstage] != Imat.shape),"Imat shape doesnt match nstages!"
    
    msize=len(mod0)
    beta0=np.array(mod0[0:nstage])
    gamma0=np.array(mod0[nstage:msize])
    s0=sir0[0]
    i0=sir0[1:nstage+1]
    
    b0=float(np.dot( beta0.T ,i0))
     
    jac=np.zeros((n,n))
    jac[0,0]  = -b0
    jac[0,1:nstage+1] =- s0*beta0.T
   
    if (Imat is not None):
        jac[1:nstage+1,0:nstage+1]  = jac_updateI(i0,s0, Imat, gamma0, beta0)
    else:
        jac[1:nstage+1,0:nstage+1]  = jac_updateI_slow(i0,s0, gamma0, beta0)
    jac[nstage+1,1:nstage+1] += gamma0.T
    
    return jac;
    
def jacobian_mod_sir_nstage(Imat,m,sir0):
    n=sir0.shape[0]
    nstage=n-2
   
    s0=sir0[0]
    i0=sir0[1:nstage+1]
    
    jacm=np.zeros((n,m))
       
    jacm[0,0:nstage] -= s0*i0.T 
    if (Imat is not None):
        jacm[1:nstage+1,0:2*nstage]  = jac_m_updateI(2*nstage,i0,s0)
    else:
        jacm[1:nstage+1,0:2*nstage]  = jac_m_updateI_slow(2*nstage,i0,s0)
    jacm[nstage+1,nstage:2*nstage] = i0.T 
    
    return jacm;    
  
"""
Partition updater for infectives for SIR_Nstage
p[nstage]    : partition of infectives (n stages) (sum to 1)
gamma[nstage]: rate recovery (nstages)
delta[nstage]: rate of progression (nstages)
Used in threshold computation for SIR_nstage and SI2R
"""  

def update_p(dt,p,I,gamma,delta,tol=1.0e-8):
    n=p.shape[0]
    assert(gamma.shape == p.shape)
    assert(delta.shape[0] == gamma.shape[0] - 1)
    
    D=gamma +I
#   D[1:n-1] = (gamma[1:n-1]+1)  + I
    D[n-1] = gamma[n-1] + I
    D[0  ] = 0 
    D[1:n-1] += delta[1:n-1]
    M  = np.diag( D )
    M -= np.diag( delta[0:n-1], -1)
  
    A  = np.concatenate( ( sum(M).reshape((1,n)) , np.zeros((n-1,n) ) ) )
    A -= M
    p +=dt*np.dot(A,p)
    converge=False
    if(np.sum(abs(np.dot(A,p))) <= tol):
        converge=True
    return [p,converge]

      
"""
SIRS
sirs[3]: susceptible, infectived, recovered, susceptible
mod[3] : infectivity, recovery, relapse (as rates per unit population)
"""  
def sirs_nonlinear(dt,mod,sirs):
    s =sirs[0]
    i =sirs[1]
    r =sirs[2]

    sirs[0] += - dt*(mod[0]*(s*i) - mod[2]*r)
    sirs[1] +=   dt*(mod[0]*s     - mod[1]  )*i
    sirs[2] +=   dt*(mod[1]*i     - mod[2]*r)
    
def jacobian_sirs(mod,sirs):
    n=sirs.shape[0]
    jac=np.zeros((n,n))
    jac[0,0] = - mod[0]*sirs[1]
    jac[0,1] = - mod[0]*sirs[0]
    jac[0,2] =   mod[2]
    jac[1,0] =   mod[0]*sirs[1]
    jac[1,1] =   mod[0]*sirs[0]-mod[1]
    jac[2,1] =   mod[1]
    jac[2,2] = - mod[2]
    return jac;

def jacobian_mod_sirs(m,sirs):
    n=sirs.shape[0]
    jacm=np.zeros((n,m))
    jacm[0,0] = - sirs[0]*sirs[1]
    jacm[0,2] =   sirs[2]
    jacm[1,0] =   sirs[0]*sirs[1]
    jacm[1,1] = -         sirs[1]
    jacm[2,1] =           sirs[1]
    jacm[2,2] = - sirs[2]
    return jacm;   