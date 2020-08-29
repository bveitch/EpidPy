# -*- coding: utf-8 -*-
"""
description : Tools for building graphs; some algebraic constructions for completeness (and fun)
author      : bveitch
version     : 1.0
project     : EpiPy (epidemic modelling in python)

"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def set_adjacency(V,E):  
    nv = len(V)  
    A=np.zeros((nv,nv))
    D=np.zeros(nv)
    for v in V: 
        for w in V:
            if ( (v,w) in E and v != w):
                A[v,w]=1
                D[v]+=1 
                D[w]+=1
    A+=A.T
    return[A,D]
    
class Graph:
    
    def __init__(self, V,E, name=None):
        self.V    = V
        self.E    = E
        self.name = name
     
        [A,D]=set_adjacency(V,E)
      
        self.A = A
        self.degree=D
              
    def print_Adj(self):
          print("adjacency matrix =", self.A)   

    def is_regular(self):
        result = False;
        if len(self.degree) > 0 :
            result = all(elem == self.degree[0] for elem in self.degree)
            return [result,self.degree[0]] 
        else:
            return [result,0]
     
    def get_adjacency(self):
        return self.A
    
    def get_laplacian(self):  
        return np.diag(self.degree)-self.A
    
    def get_pop_laplacian(self,popcounts): 
        L = np.diag(self.degree)-self.A
        assert(len(popcounts) == len(self.V)),"Population size must match number of graph vertices"
        Sc=np.diag(popcounts)
        invSc=np.diag(1./popcounts)
        return invSc.dot(L.dot(Sc))
    
    def draw( self ):
        plt.close()
        G=nx.Graph()
        for v in self.V:
            G.add_node(v)
    
        for e in self.E:
            G.add_edge(e[0],e[1])

        print ( "number of nodes =", G.number_of_nodes() ) 
        plt.subplot()
        if(self.name is not None):
            plt.title(self.name)
        nx.draw_shell(G,with_labels=True, font_weight='bold')
        plt.show()
        plt.savefig("mygraph.png") 


def Empty(n):
    V=list(range(n))
    E=[]
    return Graph(V,E,'E{a}'.format(a=n))
 
def Complete(n):
    V=list(range(n))
    E=[]
    for v in V:
        E+=[ (v,w) for w in range(v+1,n)]
    return Graph(V,E,'K{a}'.format(a=n))

def Bipartite(n1,n2):
    V=list(range(n1+n2))
    E=[]
    for i in range(n1):  
#        [ E.append((i,j+n1)) for j in range(n2)]
        E+=[ (i,j+n1) for j in range(n2)]
#        for j in range(n2):
#            E.append((i,j+n1))
    return Graph(V,E,'B({a},{b})'.format(a=n1,b=n2))

def Multipartite(nlist):
    ntot=sum(nlist)
   
        
    V=list(range(ntot))
    E=[]
    nn=0 
    name='M('    
    for ilist in range(len(nlist)):
        for i in range(nlist[ilist]):
            v=nn+i
            mm=nn+nlist[ilist]
            for m in nlist[ilist+1:len(nlist)]:
#                for j in range(m):
#                    w=mm+j
#                    [ E.append((v,w))]
                E+=[ (v,w) for w in range(mm,mm + m)]
                mm+=m
        nn+=nlist[ilist]
        
        name+='{}'.format(nlist[ilist])
        
        if(ilist==(len(nlist)-1)):
            name+=')'
        else:
            name+=','
   
    return Graph(V,E,name)

def Cycle(n):
    V=list(range(n))

    E=[(v,v+1) for v in range(n-1)]
    E+=[(0,n-1)]
    return Graph(V,E,'C{a}'.format(a=n))

def Star(n):
    V=list(range(n))
    E=[ (0,v) for v in range(1,n) ]
    return Graph(V,E,'S{a}'.format(a=n))

def Wheel(n):
    V=list(range(n))

    E=[ (0,v) for v in range(1,n) ]
    E+=[(v,v+1) for v in range(1,n-1)]
    E+=[(1,n-1)]
    return Graph(V,E,'W{a}'.format(a=n))

def Expt():
    V= (1, 2, 3, 4, 5, 6)
    E= ((1,2),(1,3),(1,4),(1,5),(1,6),(2,3),(2,5),(3,4),(4,5))
    
    return Graph(V,E)

def Graph_from_Adjacency(A):
    [nv,nv2]=A.shape
    V=range(nv)
    E=[]
    for i in range(nv):
        E+=[(i,j) for j in range(i+1,nv) if ( A[i,j] == 1)]

    return Graph(V,E)

def Complement(G):
    V=G.V
    E=G.E
    Ec=[]
    n=len(V)
    for v in V:  
        Ec+=[(v,w) for w in range(v+1,n) if ( (v,w) not in E)]

    return Graph(V,Ec,'('+G.name+')^C')
    
def GraphJoin(G1,G2):

    V1=G1.V
    E1=G1.E
    nv = len(V1) 
    V2=G2.V
    E2=G2.E
     
    V0=V1.copy()
    E0=E1.copy()

    V0+=[v+nv for v in V2]
 
    E0+=[(e[0]+nv,e[1]+nv) for e in E2]
       
    for v1 in V1:
        E0+=[(v1,v2+nv) for v2 in V2]

    return Graph(V0,E0,'J('+G1.name+','+G2.name+')')

def Product(G1,G2,ptype='x'):

    def order(e):
        if(e[0] > e[1]): 
            return (e[1],e[0])
        else:
            return e
        
    def tensor(e1,E1,e2,E2):
        condition = ( (order(e1) in E1) and (order(e2) in E2) )
        return condition
    
    def cartesian(e1,E1,e2,E2):
        condition = (order(e1) in E1 and e2[0] == e2[1]) or ( e1[0] == e1[1] and order(e2) in E2)
        return condition
    
    def get_condition():
        conditions = {
            'x'  : cartesian,
            '*'  : tensor
            }
        condition = conditions.get(ptype, lambda: "Invalid product option")
        return condition
    
    V1=G1.V
    E1=G1.E
    V2=G2.V
    E2=G2.E
    n1 = len(V1)  
    n2 = len(V2)     
    W=[]


    for v1 in V1:
        W+=[n2*v1+v2 for v2 in V2]
   
    F=[]
    #cond=get_condition()      
    for v1 in V1:
        for v2 in V2: 
            for w1 in range(v1,n1): 
                if(ptype == '*'):
                    F+=[(n2*v1+v2,n2*w1+w2) for w2 in V2 if ( tensor((v1,w1),E1,(v2,w2),E2) )]  
                else:
                    F+=[(n2*v1+v2,n2*w1+w2) for w2 in range(v2,n2) if ( cartesian((v1,w1),E1,(v2,w2),E2) )]

                
    return Graph(W,F,'('+G1.name+')'+ptype+'('+G2.name +')')


  