# -*- coding: utf-8 -*-
"""
description : Run and test sir_network_modeller
author      : bveitch
version     : 1.0
project     : EpidPy (epidemic modelling in python)
"""

import numpy as np
import plottingutils as plotter
import sys
sys.path.insert(1, 'network')
import networked_model_builder as mNbuild
import sir_network_modeller as sirNm
import graph_builder as gbuild
from functools import partial

"""
Testing sir modes and operators
"""
nt=350
ntsub=100
dt=0.01
t=np.arange(0,nt,1)
sir_types=['sir','seir','sirs']

passed=True

V= list(range(6))
E= ((0,1),(0,2),(0,3),(0,4),(0,5),(1,2),(1,4),(2,3),(3,4))
G=gbuild.Graph(V,E,'network')
Nv=len(V)
pops=np.array([8.9,4.3,2.8,1.5, 0.8, 0.5])
#pops=np.ones(Nv)
icases=np.array([0.0010,0,0.00002,0,0,0])

if(len(icases) != Nv):
    raise RuntimeError("Cases and graph vertices must be same dim");

#def set_NetLaplacian(G,pops):
#    L=G.get_laplacian()
#    Sc=np.diag(pops)
#    invSc=np.diag(1./pops)
#    LN=invSc.dot(L.dot(Sc))
#    return LN
def test_sirdata_sum(sirdata,total_pop,tol=5e-4):
    [nt,nv]=sirdata[0].shape
    dsum=np.zeros(nt)
    for data in sirdata:
        dsum +=np.sum(data,axis=1)
    print(dsum)    
    return np.all((dsum - total_pop) < tol)
   
for sir_type in sir_types:
    print('Testing {:} modelling ...'.format(sir_type))
    
    #Build model and data
    mod=mNbuild.init_netmod(sir_type,icases)
    L=G.get_laplacian()
    modargs={'type':sir_type,'dt':dt, 'nt':nt, 'ntsub': ntsub}
    sirmod=sirNm.sir_mod(modargs,L,pops)
    data=sirmod.f(mod)
    sirdata=plotter.data_network_to_sir(sir_type,data,Nv)
    test_passed= test_sirdata_sum(sirdata,sum(pops))
    if(test_passed):
        print("... Well done, {:} sum test passed".format(sir_type))
    else:
        print("... Check! {:} sum test has failed".format(sir_type))
    plotter.plot_sirdata_for_node(sir_type,t,sirdata)
    plotter.plot_nodedata_for_compartment(t,sir_type,sirdata)
    #movie maker is slow:
    plotter.movie_maker_network(G,sir_type,sirdata,200,comp='i')