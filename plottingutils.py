#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 09:05:42 2020

description : Plotting methods and utilities
author      : bveitch
version     : 1.0
project     : EpidPy (epidemic modelling in python)

"""

import matplotlib.pyplot as plt
import numpy as np
import sir_sampler as samplr
import numpy as np
import networkx as nx
import matplotlib.animation as animation

linestyles=[':','-.','--','-']
colors1  =['b','r','g','c','m','k']
colors2  =['pink','darkred','olive','chartreuse','navy','skyblue']
markers1 =['x','o','*','v']
markers2 =['+','s','d','^']

def LabelData(label_list,data):
    [nt,nc]=data.shape
    assert(len(label_list) == nc),'Label size must match data size'
    data_list=[]
    for ic in range(nc):
        data_list.append(data[:,ic])
    return dict(zip(label_list,data_list))

def LabeledData_from_model(m,sir_object,data_sampler=None):
     fm=sir_object.f(m)
     nstage=sir_object.dsize-2
     if(data_sampler):
         Sfm=data_sampler.F(fm)
         Sfm_labels=data_sampler.Flabels_from_type(sir_object.type,nstage)
         Sfm_dict=LabelData(Sfm_labels,Sfm)
         return Sfm_dict
     else:
         labels=samplr.labels_from_type(sir_object.type,nstage)
         fm_dict=LabelData(labels,fm)
         return fm_dict

def plot_sirdata_with_threshold(sirtype,t,nstage,sir_data,sir_thresholds):
    
    def sirlabels_from_type(nstage):
        labels = {
            'sir'        : ['s','i','r'],
            'seir'       : ['s','ei','r'],
            'sirs'       : ['s','i','r'],
            'si2r'       : ['s','i:1-2','r'],
            'sir_nstage' : ['s','i:1-'+str(nstage),'r']
        }
        label = labels.get(sirtype, lambda: "Invalid modelling type")
        return label
        
    labels  = sirlabels_from_type(nstage)
    [nt,nc] = sir_data.shape
    assert( nt == len(t))
    assert( nc == len(sir_thresholds))
    
    fig, ax=plt.subplots()
    for ic in range(sir_data.shape[1]):
        dat=sir_data[:,ic]
        tval=sir_thresholds[ic]
        thold=[tval for it in range(len(t))]
        
        ax.set_title("{:} mod with theoretical threshold".format(sirtype))
        ax.set_xlabel('days')
        ax.set_ylabel('Incidence per unit Population')
        labelstr=labels[ic]
        lstyle=linestyles[ic%len(linestyles)] 
        col=colors1[ic%len(colors1)]
        mark=markers1[ic%len(markers1)]
        ax.plot(t, dat, lstyle ,color=col, marker=mark,label=labelstr)
        ax.plot(t, thold, 'k-.', marker='None')
        ax.text(300, tval, "r={:#.3g}".format(tval))
    ax.legend(loc='center right',shadow=True)
    plt.show()
        
def plot_beta_gamma(beta,gamma):
    nstage=beta.shape[0]
    fig, ax=plt.subplots(1,1,squeeze=False, figsize=(18,10))
    ax[0,0].set_title("Infectivity and Recovery functionals",fontsize=10)
    ax[0,0].set_xlabel('days')
    ax[0,0].xaxis.labelpad=-25
    ax[0,0].set_ylabel('')
    ax[0,0].plot(range(nstage), beta , 'r',label='infectivity')
    ax[0,0].plot(range(nstage), gamma, 'b',label='recovery'   )
    ax[0,0].legend(loc='upper right',shadow=True)       

def plot_rsampled(t,sdata2,rdata2,tdata,Tdata): 
    fig, ax=plt.subplots(1,2,squeeze=False, figsize=(18,10))
    ax[0,0].set_title("Sampling of multi-stage infectives",fontsize=10)
    ax[0,0].set_xlabel('days')
    ax[0,0].xaxis.labelpad=-25
    ax[0,0].set_ylabel('')
    ax[0,0].plot(t, sdata2[0], 'r-',marker='x',label='i1')
    ax[0,0].plot(t, rdata2[0], 'b-',marker='o',label='sampled i1')
    ax[0,0].plot(t, sdata2[1], 'g-',marker='+',label='i2')
    ax[0,0].plot(t, rdata2[1], 'k:',marker='x',label='sampled i2')
    ax[0,0].legend(loc='lower right',shadow=True)
    
    ax[0,1].set_title("Sampling of total infectives",fontsize=10)
    ax[0,1].set_xlabel('days')
    ax[0,1].xaxis.labelpad=-25
    ax[0,1].set_ylabel('')
    ax[0,1].plot(t, sdata2[0] + sdata2[1], 'r-',marker='+',label='sum_i1i2' )
    ax[0,1].plot(t, tdata, 'b-',marker='x',label='sampled_i1i2')
    ax[0,1].plot(t, Tdata, 'g-',marker='None',label='total_true_i')
    ax[0,1].legend(loc='lower right',shadow=True)
    plt.show()



def plot_fit(t,d_dict,sir_type,fm_dict,title=None,fname=None):
   
    def clip(d):
        if(d<0):
            return 0
        elif(d>1):
            return 1
        else:
            return d
        
    def clip_data(data):
        return [clip(d) for d in data]
                
    def plot_from_dict(axis,i,j,d,markers,colors,prefix):
        ii=0
        for key, value in d.items():
            assert(len(value) == len(t)),"data size doesnt match time samples"
            label=prefix + key
            lstyle=linestyles[ii % len(linestyles)] 
            col=colors[ ii % len(colors) ]
            mrk=markers[ii % len(markers)]
            ax[i][j].plot(t, clip_data(value), lstyle ,color=col, marker=mrk,label=label)
            ii+=1
        
    assert(len(d_dict) == len(fm_dict)),"data size doesnt match modelled size"
  
    fig, ax=plt.subplots(1,1,squeeze=False, figsize=(18,10))
    if(title is not None): 
        ax[0][0].set_title(title,fontsize=10)
    ax[0][0].set_xlabel('days')
    ax[0][0].set_ylabel('Incidence per unit Population')
    plot_from_dict(ax,0,0, d_dict,markers1 ,colors1 ,'data')
    plot_from_dict(ax,0,0,fm_dict,markers2 ,colors2 ,'modelled')
    ax[0,0].legend(loc='center right',shadow=True) 
    plt.show()
    if(fname is not None):  
        plt.savefig(fname)


 
#Network plotting utils
        
def data_network_to_sir(sirtype,data,nvertices):
        
        def get_ncomponents_from_type():
            ncomps = {
            'sir'        : 3,
            'seir'       : 4,
            'sirs'       : 3
            }
            ncomp = ncomps.get(sirtype, lambda: "Invalid modelling type")
            return ncomp
        
        [nt,nn]=data.shape
        nc=get_ncomponents_from_type()
        assert(nn == nc*nvertices),'data dimension 1 must match ncompartments * nvertices'
        sirdata=data.reshape((nt,nc,nvertices))
        sirlist=[]
        for j in range(nc):
            sirlist.append(sirdata[:,j,:])
        return sirlist
    
def plot_sirdata_for_node(sirtype,tt,sirdata,ivertex=None,fname=None):
  
    fig, ax = plt.subplots(figsize=(18,10))
    ax.set_title(" {:} epidemic on node {:}".format(sirtype.upper(),ivertex))
    ax.set_xlabel('days') 
    ax.set_ylabel('Incidence per unit Population')
    
    labels = list(sirtype)
    ic=0
    for data in sirdata:
            
            if(ivertex is None):
                d_plot=np.sum(data,axis=1)
            else:
                d_plot=data[:,ivertex]
                
            lstyle=linestyles[ic%len(linestyles)] 
            col=colors1[ic %len(colors1)] 
            mrk=markers1[ic % len(markers1)]
            label=labels[ic]
            ax.plot(tt, d_plot, lstyle ,color=col, marker=mrk,label=label)
            ic+=1
       
    ax.legend(loc='upper right',shadow=True)  
    plt.show()      
    if(fname is not None):  
        plt.savefig(fname)
        
def plot_nodedata_for_compartment(tt,sirtype,sirdata,comp='i',fname=None):     
    
    icomp=sirtype.find(comp)
    assert(icomp >=0 ),'compartment name not in modelling type' 
    assert(icomp < len(sirtype) ),'compartment index must be less than {:}'.format(len(sirtype)) 
    compartment_data=sirdata[icomp]
    [nt,nv]=compartment_data.shape
    fig, ax=plt.subplots(1,1,squeeze=False, figsize=(18,10))
    ax[0][0].set_title("{:}: Compartment {:} over all nodes".format(sirtype.upper(),comp.upper()))
    ax[0][0].set_xlabel('days') 
    ax[0][0].set_ylabel('Incidence per unit Population')
    for iv in range(nv):
        dat= compartment_data[:,iv]
        
        labelstr='node:'+str(iv)
        lstyle=linestyles[iv%len(linestyles)] 
        col=colors1[iv%len(colors1)]
        mark=markers1[iv%len(markers1)]
        ax[0][0].plot(tt, dat, lstyle ,color=col, marker=mark,label=labelstr)
            
    ax[0][0].legend(loc='center right',shadow=True)
    plt.show() 
    if(fname is not None):  
        plt.savefig(fname)
        
def movie_maker_network(Netw,sirtype,sirdata,nt,comp='i'): 
    icomp=sirtype.find(comp)
    assert(icomp >=0 ),'compartment name not in modelling type' 
    assert(icomp < len(sirtype) ),'compartment index must be less than {:}'.format(len(sirtype)) 
    
    compartment_data=sirdata[icomp]
    [ntot,nv]=compartment_data.shape 
    assert(nv == len(Netw.V) ),'number of compartment nodes must match number of graph vertices'      
  
    fig, ax = plt.subplots(figsize=(8,5))

    Writer = animation.writers['ffmpeg']
    ffwriter = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800, codec='mpeg4')
    G=nx.Graph()
    
    
    for v in Netw.V:
        G.add_node(v)
    
    for e in Netw.E:
        G.add_edge(e[0],e[1])

    Nt=min(nt,ntot)
    fname="Epid_{:}_{:}_cases.mp4".format(sirtype.upper(),comp.upper())
    with ffwriter.saving(fig, "movies/"+fname, Nt):
        for it in range(Nt):
            ax.clear()

            # Background nodes
            nx.draw_shell(G,with_labels=True, font_weight='bold',node_size=200*compartment_data[it,:])

            # Scale plot ax
            title="{:} Epidemic. {:} cases at timestep {:}:    ".format(sirtype.upper(),comp.upper(),it)
            ax.set_title(title, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            ffwriter.grab_frame()