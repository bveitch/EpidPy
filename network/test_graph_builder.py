#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
description : Testing graph builder
author      : bveitch
version     : 1.0
project     : EpiPy (epidemic modelling in python)

"""
import graph_builder as gbuild

def build_graph(n,name):
   if(name == 'empty'):
       return gbuild.Empty(n)
   elif(name == 'complete'):
       return gbuild.Complete(n)
   elif(name == 'cycle'):
       return gbuild.Cycle(n)
   elif(name == 'star'):
       return gbuild.Star(n)
   elif(name == 'wheel'):
       return gbuild.Wheel(n)
   elif(name == 'bipartite'):
       return gbuild.Bipartite(n[0],n[1])
   elif(name == 'multipartite'):
       return gbuild.Multipartite(n) 
   else:
       return gbuild.Empty(0)

def get_result(passed):
    if(passed):
        return 'passed'
    else:
        return 'failed'

def vertex_test(n,name,nV):
   if(name == 'bipartite'):
       return ( (n[0] + n[1]) == nV)
   elif(name == 'multipartite'):
       return ( sum(n) == nV )
   else:
       return ( n == nV)
   
    
def nedges_multipartite(nlist):
    nE=0
    while len(nlist) > 0 :
        nl  = nlist.pop()
        nE += nl*sum(nlist)
    return nE

def edge_test(n,name,nE):
   if(name == 'empty'):
       nedges = 0
   elif(name == 'complete'):
       nedges = n*(n-1)/2
   elif(name == 'cycle'):
       nedges = n
   elif(name == 'star'): 
       nedges = n-1
   elif(name == 'wheel'):
       nedges = 2*(n-1)
   elif(name == 'bipartite'):
       nedges = n[0]*n[1]
   elif(name == 'multipartite'):
       nedges = nedges_multipartite(n)
   else:
       nedges = -1
   return (nedges==nE)  

def regularity_test(nv,is_reg,k,name):
   if(name == 'empty'):
       return (is_reg and k==0)
   elif(name == 'complete'):
       return (is_reg and k==nv-1)
   elif(name == 'cycle'):
       return (is_reg and k==2)
   elif(name == 'star'): 
       if( nv <= 2):
           return (is_reg and k == nv-1)
       else:
           return (not is_reg)
   elif(name == 'wheel'):
       if( nv <= 3):
           return (is_reg and k == nv-1)
       else:
           return (not is_reg) 
   elif(name == 'bipartite'):
       if( nv[0] == nv[1]):
           return (is_reg and k == nv[0])
       else:
           return (not is_reg) 
   elif(name == 'multipartite'):
       reg_mp = all(elem == nv[0] for elem in nv)
       if(reg_mp):
           return (is_reg and k==sum(nv[0:len(nv)-2]) )
       else:
           return (not is_reg)
   else:
       return False 
   
def general_test(n,name):
    G=build_graph(n,name)
    print("testing {:}".format(G.name))
    pass_nV = vertex_test(n,name,len(G.V))
    result  = get_result(pass_nV)
    print("Vertex test {0} for {1}. Has {2} vertices".format(result,G.name,len(G.V)))
    pass_nE = edge_test(n,name,len(G.E))
    result  = get_result(pass_nE)
    print("Edge test {0} for {1}. Has {2} edges".format(result,G.name,len(G.E)) )
    return (pass_nV and pass_nE)

               
def test_complement(G): 
    print("testing complement of {:}".format(G.name))
    n=len(G.V) 
    Gc=gbuild.Complement(G)
    pass_nV = (len(Gc.V) == n )
    result  = get_result(pass_nV)
    print("Vertex test {0} for {1}. Has {2} vertices".format(result,Gc.name,len(Gc.V)))
    pass_nE = (len(Gc.E) == n*(n-1)/2 - len(G.E) )
    result  = get_result(pass_nE)
    print("Edge test {0} for {1}. Has {2} edges".format(result,Gc.name,len(Gc.E)) )
#    AG =G.get_adjacency()
#    AGc=Gc.get_adjacency()
#    pass_adj = (AGc == (np.ones(n) - AG) ) 
    return (pass_nV and pass_nE)

def test_join(G1,G2):
    print("testing join of {0} and {1}".format(G1.name,G2.name))
    n1=len(G1.V) 
    n2=len(G2.V) 
    G=gbuild.GraphJoin(G1,G2)
    pass_nV = (len(G.V) == n1 + n2 ) 
    result  = get_result(pass_nV)
    print("Vertex test {0} for {1}. Has {2} vertices".format(result,G.name,len(G.V)))
    pass_nE = (len(G.E) == ( n1*n2 + len(G2.E) + len(G1.E) ) )
    result  = get_result(pass_nE)
    print("Edge test {0} for {1}. Has {2} edges".format(result,G.name,len(G.E)) )
    return (pass_nV and pass_nE)

def test_product(G1,G2,opt="x"): 
    
    def get_prodtype(opt):
        if(opt=='*'):
            return 'tensor'
        else:
            return 'cartesian'
        
    print("testing {0} product of {1} and {2}".format(get_prodtype(opt),G1.name,G2.name))
    n1=len(G1.V) 
    n2=len(G2.V) 
    G=gbuild.Product(G1,G2,opt)
    pass_nV = (len(G.V) == n1*n2 )
    result  = get_result(pass_nV)
    print("Vertex test {0} for {1}. Has {2} vertices".format(result,G.name,len(G.V)))
    if(opt == "x"):
        pass_nE = (len(G.E) == n1*len(G2.E)+n2*len(G1.E) )
    else:  
        pass_nE = (len(G.E) == 2*len(G2.E)*len(G1.E) )
    result  = get_result(pass_nE)
    print("Edge test {0} for {1}. Has {2} edges".format(result,G.name,len(G.E)) )
    return (pass_nV and pass_nE)

tests={'empty': 10, 'complete':20,'cycle':6, 'star':16, 'wheel': 20, 'bipartite':[20,30], 'multipartite':[3,4,5,6] }

for name, pars in tests.items():
  passed = general_test(pars,name)
  if(passed):
      print('{:} test passed'.format(name))
  else:
      print('{:} test failed'.format(name))
  

W10=gbuild.Wheel(10)
W10.draw()

op_tests={'complement': test_complement(W10), 
          'join': test_join(gbuild.Empty(3),gbuild.Empty(4)),
          'cartesian product': test_product(gbuild.Complete(3),gbuild.Cycle(4),'x'),
          'tensor product': test_product(gbuild.Complete(3),gbuild.Cycle(4),'*')  }

for name, test in op_tests.items():
  passed = test
  if(passed):
      print('{:} test passed'.format(name))
  else:
      print('{:} test failed'.format(name))

