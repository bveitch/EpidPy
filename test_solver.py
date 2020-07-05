#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 19:44:26 2020

description : Run and test solvers on trial data
author      : bveitch
version     : 1.0
project     : EpidPy (epidemic modelling in python)

"""
import numpy as np
import sir_modeller as sirm
import sir_solver as sirs
import model_builder as mbuild
import dataset_builder as dbuild
import sir_sampler as samplr
import plottingutils as plotter

def test_solver(mod,sirtype):
    soltest_args={'x0':mod,'niter':30, 'tol':1.0e-5}
    passed = solver.test(soltest_args)
    if(passed):
        print("solver test passed")
    passed2 = solver.check_grad(soltest_args)
    if(passed2):
        print("scipy test passed") 
    return passed and passed2
 
nt=350
ntsub=100
dt=0.01
t=np.arange(0,nt,1)
modargs={'dt':dt, 'nt':nt, 'ntsub': ntsub}
#Build test dataset
data = dbuild.build_data_nstage(modargs)
data_dict=plotter.LabelData(['infected_cases'],data)

# Modelling setup:
sirtype='si2r'
fmargs={'type':sirtype,'dt':dt, 'nt':nt, 'ntsub': ntsub}
sirfm=sirm.sir_mod(fmargs)
samp_intv=[[1,3]]
Fmsamp  = samplr.Sampler(sirfm.dsize,samp_intv)
mod0=mbuild.init_mod(sirtype)
Sfm0_dict=plotter.LabeledData_from_model(mod0,sirfm,Fmsamp)
   
title="Initial Epidemic fit for {:}".format(sirtype)
plotter.plot_fit(t,data_dict,sirtype,Sfm0_dict,title,'si2r_init_fit.jpg')

#Setup solvers
w=np.sqrt(np.arange(nt)/(nt-1))
W=w.reshape(nt,1)*np.ones((nt,len(samp_intv)))
solver=sirs.SIR_Solver(data,sirfm,Fmsamp,W)
if(True):
    passed=test_solver(mod0,sirtype)
    if(passed):
        print("... Well done, all {:} tests passed".format(sirtype))
    else:
        print("... Check! An {:} test has failed".format(sirtype))
        exit()

#Run
options0={'maxiter':100, 'disp': True}
solution=solver.solve(mod0,options0)

#Print diagnostics
Sfmsol_dict=plotter.LabeledData_from_model(solution['xsol'],sirfm,Fmsamp)
title="Final epidemic fit for {:}".format(sirtype)
plotter.plot_fit(t,data_dict,sirtype,Sfmsol_dict,'si2r_final_fit.jpg')    
print('solver runtime:', solution['runtime']/60,'mins')
obj=solution['objmin']
print('final objective function: ',obj)
solver.plot_convergence()
