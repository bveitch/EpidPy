#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 08:46:30 2020

description : Convenience functions for building Network sir models for testing  
author      : bveitch
version     : 1.0
project     : EpiPy (epidemic modelling in python)
"""

import numpy as np
import random
import math
from scipy.special import gamma as Gamma
from plottingutils import plot_beta_gamma

beta0=0.14
gamma0=0.057
delta0=0.008
diffusivity=0.01
  
def sir_mod(icases):
    return [beta0,gamma0,diffusivity,icases]
            
def seir_mod(icases):
    return [beta0,delta0,gamma0,diffusivity,icases]

def sirs_mod(icases):
    A=delta0
    return [beta0,gamma0,A,diffusivity,icases]


def init_netmod(sir_type,icases):
     models = {
            'sir'        : sir_mod,
            'seir'       : seir_mod,
            'sirs'       : sirs_mod
        }
     mod = models.get(sir_type, lambda: "Invalid modelling type")
     return mod(icases)

   