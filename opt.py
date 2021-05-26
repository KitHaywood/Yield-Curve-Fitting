############################
#     OPTIMIZER MODULE     #
#       August 2014        #
############################

import numpy as np
import pandas as pd
from datetime import datetime
from math import exp
from scipy import interpolate, stats, optimize
from scipy.optimize import minimize, basinhopping
import pylab
import matplotlib.pyplot as plt
from numpy.polynomial.laguerre import lagval
import pdb
# import numbapro
from multiprocessing import Pool
# import y2p # yield to price
#from numbapro import autojit

radiandegs = 57.295775

def laguerre(degree, phi, m):
    """ naked (non integrated) laguerre, suitable for modelling instantaneous forwards"""
    degree = degree - 2
    if degree < 0:
        return np.ones(m.shape[0])
    else:
        coef = np.zeros(degree + 1)      
        coef[degree] = 1
        return -np.exp(-phi * m) * lagval((2 * phi * m), coef)


def laguerreIntegral(degree, phi, m):
    """ does what it says on the tin. Integrated laguerre. Corresponds to zero yields where
        laguerre plain corresponds to instantaneous forwards """
    Laguerredict = {1: lambda phi, m: np.ones(m.shape[0]), \
                    2: lambda phi, m: ((1 / (phi * m)) * (np.exp(- phi * m) - 1)), \
                    3: lambda phi, m: (-(1 / (phi * m)) * (2 * phi * m * np.exp(- phi * m) \
                        + np.exp(- phi * m) - 1)), \
                    4: lambda phi, m: ((1 / (phi * m)) * (2 * ((phi * m) ** 2) * \
                        np.exp(- phi * m) + np.exp(- phi * m) - 1)), \
                    5: lambda phi, m: (np.exp(-m * phi) *(-2 * m * phi * (m * phi * \
                        (2 * m * phi - 3) + 3) + 3 * np.exp(m * phi) - 3))/(3 * m * phi), \
                    6: lambda phi, m: (np.exp(-m * phi) * (2 * (m**2) * (phi**2) * \
                        (m * phi * (m * phi - 4) + 6) - 3 * np.exp(m * phi) + 3)) / (3 * m * phi), \
                    7: lambda phi, m: (np.exp(-m * phi) * (-2 * m * phi * \
                        (m * phi * (m * phi * (m * phi * (2 * m * phi - 15) + 40) - 30) + 15) + 15 * \
                        np.exp(m * phi) - 15)) / (15 * m * phi), \
                    8: lambda phi, m: (np.exp(-m * phi) * (2 * (m**2) * (phi**2) * \
                    
                        (m * phi * (m * phi * (2 * m * phi * (m * phi - 12) + 105) - 180) + 135) - 45 * \
                        np.exp(m * phi) + 45)) / (45 * m * phi)}
    return(Laguerredict[degree](phi, m))

