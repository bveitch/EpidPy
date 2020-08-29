# -*- coding: utf-8 -*-
"""
description : Provides nonlinear operators for SIR updates on networks. Wrapped into sir_network_modeller 
author      : bveitch
version     : 1.0
project     : EpiPy (epidemic modelling in python)

"""

import numpy as np

"""
SIR:
sir[3]: susceptible, infected, recovered
mod[2]: infectivity, recovery (as rates per unit population)
"""

def netw_sir_nonlinear(dt,L,mod,sir):
    N=L.shape[0]
    
    S=sir[0:N]
    I=sir[N:2*N]
    
    LI = L.dot(I)
    sir[0:N]    +=                 - dt* mod[0]*S*I
    sir[N:2*N]  += - mod[2]*dt*LI  + dt*( mod[0]*S - mod[1])*I
    sir[2*N:3*N]+=                 + dt*             mod[1] *I

"""
SEIR:
seir[4]: susceptible, exposed, infected, recovered
mod[3] : infectivity, progression (exposed to infected), recovery (as rates per unit population)
"""           
def netw_seir_nonlinear(dt,L,mod,seir):
    N=L.shape[0]
    S =seir[0:N]
    E =seir[N:2*N]
    I =seir[2*N:3*N]
    
    LI = L.dot(I)
    seir[0:N]    +=                 -dt*mod[0]*S*I 
    seir[N:2*N]  +=                  dt*mod[0]*S*I - dt*mod[1]*E
    seir[2*N:3*N]+= - mod[3]*dt*LI + dt*mod[1]*E   - dt*mod[2]*I 
    seir[3*N:4*N]+=                                  dt*mod[2]*I
 


      
"""
SIRS
sirs[3]: susceptible, infectived, recovered, susceptible
mod[3] : infectivity, recovery, relapse (as rates per unit population)
"""  
def netw_sirs_nonlinear(dt,L,mod,sirs):
    N=L.shape[0]
    
    S =sirs[0:N]
    I =sirs[N:2*N]
    R =sirs[2*N:3*N]
    
    LI = L.dot(I)
    sirs[0:N]    +=                - dt*mod[0]*(S*I)                + dt*mod[2]*R
    sirs[N:2*N]  += - mod[3]*dt*LI + dt*(mod[0]*S    -    mod[1])*I
    sirs[2*N:3*N]+=                                    dt*mod[1] *I - dt*mod[2]*R


