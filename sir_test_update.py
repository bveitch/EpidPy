# -*- coding: utf-8 -*-
"""
description : Run and test sir_modeller
author      : bveitch
version     : 1.0
project     : EpidPy (epidemic modelling in python)
"""

import numpy as np
import plottingutils as plotter
import model_builder as mbuild
import sir_modeller as sirm
import test_threshold as threshold
import test_jacobian as tests
from functools import partial

"""
Testing sir modes and operators
"""
nt=350
ntsub=100
dt=0.01
t=np.arange(0,nt,1)
sir_types=['sir','seir','sirs','si2r','sir_nstage']
passed=True

def data_to_sir(data):
    [nt,nc]=data.shape
    sir_data=np.zeros((nt,3))
    sir_data[:,0]=data[:,0]
    for i in range(1,nc-1):
        sir_data[:,1]=data[:,i]
    sir_data[:,2]=data[:,nc-1]
    return sir_data

for sir_type in sir_types:
    print('Testing {:} modelling ...'.format(sir_type))
    
    #Build model and data
    mod=mbuild.init_mod(sir_type)
    modargs={'type':sir_type,'dt':dt, 'nt':nt, 'ntsub': ntsub,'nstage':30}
    sirmod=sirm.sir_mod(modargs)
    data=sirmod.f(mod)

    passed=True
    #Test thresholds for epidemic
    thresh_args={'type':sir_type,'dt':dt, 'nt':nt, 'ntsub': ntsub,'niter':20,'newton_tol':0.0001,'thresh_tol':0.01}
    m0=mod[0:sirmod.msize]
    ics=np.zeros(sirmod.dsize)
    m=len(mod)-sirmod.msize
    ics[0]=1
    sirm.push_ic(mod[sirmod.msize:len(mod)],ics)
    sir_data=data_to_sir(data)
    [thresh_pass,sir_t]=threshold.test(thresh_args,ics,m0,sir_data[nt-1,:])
    print(thresh_pass)
    #Plot a comparison
    nstage = data.shape[1] - 2 
    plotter.plot_sirdata_with_threshold(sir_type,t,nstage,sir_data,sir_t)

    passed = (passed and thresh_pass)

    #Linearization test (for Jacobian)
    dmod=mbuild.init_dmod(mod)
    linargs={'mod0': mod, 'dmod':dmod, 'niter':20, 'lintol':0.005}
    lin_pass=tests.lin_test(sirmod,linargs)
    print(lin_pass)
    passed = (passed and lin_pass)
    #Dot product test (for Jacobian)
    dpargs={'mod': np.array(mod), 'data': data, 'dptol':1e-11}
    sirF =partial(sirmod.F ,mod)
    sirFt=partial(sirmod.Ft,mod)
    dp_pass=tests.dp_test(sirF,sirFt,dpargs)
    print(dp_pass)
    passed = (passed and dp_pass)
    if(passed):
        print("... Well done, all {:} tests passed".format(sir_type))
    else:
        print("... Check! An {:} test has failed".format(sir_type))
    if(not passed):
        break