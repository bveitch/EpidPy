# -*- coding: utf-8 -*-
"""

description : Test that nonlinear sir operators evolves to its theoretical threshold 
author      : bveitch
version     : 1.0
project     : EpidPy (epidemic modelling in python)

Note:
    In simple cases (SIR, SEIR, SIRS) this is straightforward
    SIR and SEIR:
        (at threshold) r - 1 = exp(-R0 r)
    SIRS: 
        known algebraic relationship
    SI2R, and SIR_Nstage [SIRS_Nstage]:
        Not trivial, R0 depends on partitioning of infectives
        Apply the above threshold criterion once partitions cease evolving
"""

import numpy as np;
import sir_update as sirup;
from functools import partial

def eval_f(x,R0):
    y=1-x-np.exp(-R0*x)
    return y

def eval_df(x,R0):
    dy=-1+R0*np.exp(-R0*x)
    return dy
    
def newton_step(x,R0):
    x-=eval_f(x,R0)/eval_df(x,R0)
    return x

def calibrate_i(sir,verb=False):
    s0=float(sir[0])
    i0=float(sir[1])
    r0=float(sir[2])
    if(i0 != 0):
        if(verb):
            print('Warning: input data is not at threshold')
        return [s0,0,i0+r0]
    else:
        return [s0,i0,r0]   

def sir_threshold(mod,sir0,args,verb=False):
    R0=mod[0]
    if(R0 < 1):
        return calibrate_i(sir0)
    else:
        x1=0.5
        for i in range(args['niter']):
            x2 = newton_step(x1,R0)
            err= np.abs(x2-x1)
            if(verb):
                print("istep = {:}, x= {:#.3g}, err={:#.6g}".format(i,x2,err))
            if err < args['newton_tol']:
                return [1-x2,0,x2]
            else:
                x1=x2
        print("oops, I didnt converge!")        
        return [1-x2,0,x2]    

def sirs_threshold(mod,args):
    R0=mod[0]
    A=mod[1]
    s0=1/R0
    i0=A*(R0-1)/(R0*(1+A))
    r0=(R0-1)/(R0*(1+A))
    return [s0,i0,r0]

def iterative_threshold(mod,sir_ic,thresholdfn,args):
    n=len(sir_ic)
    s0=sir_ic[0]
    r0=sir_ic[n-1]
    i0=sir_ic[1:n-1]
    beta =mod[0]
    gamma=mod[1]
    delta=mod[2]
    p=i0/sum(i0)
    sir0=[s0,sum(i0),r0]
    dt=args['dt']
    ntot=args['nt']*args['ntsub']
    for it in range(ntot):
        b=float(np.dot( beta.T ,p))
        c=float(np.dot(gamma.T ,p))
        sirup.sir_nonlinear(dt,[b, c],sir0)
        st=b*sir0[0]-c
        [p,conv]=sirup.update_p(dt,p,st,gamma,delta)   
        if(conv):
            m0=[b/c]
            sir=thresholdfn(m0,sir0,args['niter'],args['newton_tol'])
            return sir
    return sir0

def get_threshold(args,mod,ics):
    
    def build_model(sir_type,mod):
        if(sir_type == 'sir'):
            assert ( len(mod) == 2 ),"model size must be 2!" 
            return [mod[0]/mod[1]]
        elif(sir_type == 'seir'):
            assert ( len(mod) == 3 ),"model size must be 3!" 
            return [mod[0]/mod[2]]
        elif(sir_type == 'sirs'):
            assert ( len(mod) == 3 ),"model size must be 3!" 
            return [mod[0]/mod[1],mod[2]/mod[1]] 
        elif(sir_type == 'si2r'):
            assert ( len(mod) == 4 ),"model size must be 4!" 
            return [np.array(mod[0:2]),np.array([0,mod[2]]),np.array([mod[3]])]
        elif(sir_type == 'sir_nstage'):
            nstage = int(len(mod)/2)
            return [mod[0:nstage],mod[nstage:2*nstage],np.ones(nstage-1)]
        else:
             raise ValueError("Invalid modelling type")
    
    def thresholdfn(sir_type,mod,ics):
        modt=build_model(sir_type,mod)
        methods = {
            'sir'        : partial(sir_threshold,modt,ics),
            'seir'       : partial(sir_threshold,modt,ics),
            'sirs'       : partial(sirs_threshold,modt),
            'si2r'       : partial(iterative_threshold,modt,ics,sir_threshold),
            'sir_nstage' : partial(iterative_threshold,modt,ics,sir_threshold)
        }
        func = methods.get(sir_type, lambda: "Invalid modelling type")
        return func
    
    thrfn=thresholdfn(args['type'],mod,ics)
    return thrfn(args)


def compare(a,b,tol):
    diff = abs( a - b )
    print(a,b,diff)
    if(diff > tol):
        return False
    else:
        return True
    
def compare_sir(sir0,sir1,tol):
    cmp=[compare(x,y,tol) for x,y in zip(sir0,sir1)]
    return all(cmp)
        
def test(args,ics,mod,data):
    print("Threshold test")
    if(args['type'] != 'sirs'):
        sir0=calibrate_i(data,True)
    else:
        sir0=data
    sir=get_threshold(args,mod,ics)
    return [compare_sir(sir,sir0,0.005),sir]
