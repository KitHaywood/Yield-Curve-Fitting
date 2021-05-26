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


def fwdYields(coeffs, linphi, m = range(1, 31)):
    """ returns the Laguerre forward yields given coefficients and phi for give maturities """
    phi = 1.0 / (linphi ** 2) # delinearize phi (linear input for optimization purposes)
    zeroyields = np.zeros(len(m)) # the discounts
    for x in range(len(coeffs)):
        zeroyields = zeroyields + coeffs[x] * laguerre(x + 1, phi, m)
    return zeroyields # discount function


def zeroYields(coeffs, linphi, m = np.arange(1, 31)):
    """ returns the Laguerre zero yields given coefficients and phi for give maturities """
    phi = 1.0 / (linphi ** 2) # delinearize phi (linear input linphi for brute opt purposes)
    zeroyields = np.zeros(len(m)) # the discounts
    for x in range(len(coeffs)):
        zeroyields = zeroyields + coeffs[x] * laguerreIntegral(x + 1, phi, m)
    return zeroyields # discount function


def zeroYieldsRotation(coeffs, linphi, m = np.arange(1, 31), rotators = None):
    """ returns the Laguerre zero yields given coefficients and phi for give maturities
        then rotates them if rotation angles for each factor are provided """
    phi = 1.0 / (linphi ** 2) # delinearize phi (linear input linphi for brute opt purposes)
    zeroyields = np.zeros(len(m)) # the discounts
    for x in range(len(coeffs)):
        laguerreFactor = coeffs[x] * laguerreIntegral(x + 1, phi, m)
        if rotators is not None:
            rotationmatrix = np.array([np.cos(rotators[x]), -np.sin(rotators[x]), 
                                 np.sin(rotators[x]), np.cos(rotators[x])]).reshape(2, 2)
            rotated = np.dot(rotationmatrix, np.array([m, laguerreFactor]))
            laguerreFactor = rotated[1]
        zeroyields = zeroyields + laguerreFactor
    return zeroyields # discount function


def discountFactors(m, zeroyields):
    """ This will do have terms of up to len(coeffs) LaguerreInegral polynomials 
    and return the discount factors """
    factors = np.exp(-m * zeroyields) # discount function
    factors[m == 0] = 1 # handle the divide by zero case in the Laguerredict
    return(factors)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~NONLINEAR~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def f5(z, *params):
    """ function for optimizing laguerre with fixed phi (linearised) """
    mats, cfs, weights = params
    linphi = z[0]
    coeffs = z[1:]
    discfcts = discountFactors(mats, zeroYields(coeffs, linphi, mats))
    return np.sum(weights * np.abs(np.dot(discfcts, cfs) ** 2))


def mp_worker5(coeffs, cfobj, optmethod, basinhop):
    if cfobj["cfmat"] != -1: # if we have a good cashflows matrix then proceed
        cf = cfobj["cfmat"]["cashflows1"]
        date = cfobj["date"]
        mats = np.array(cf.index.tolist())
        durations = np.dot(mats, cf.values) / np.sum(cf.values[1:, :], 0)
        durationweights = 1 / durations
        durationweights = durationweights / np.mean(durationweights) # average 1
        params = (mats, cf.values, durationweights)
        try:
            if basinhop:
                opt_result = basinhopping(f5, coeffs, \
                        minimizer_kwargs = {"args": params, "method": optmethod, \
                        "options": {"disp": False}}, niter = 1000)
            else:
                opt_result = minimize(f5, coeffs, args = params, \
                        method = optmethod, options = {"disp": False})
        except:
            print("error on opt_result", cfobj["date"])
            return
        funmin = opt_result.fun
        bestphi = opt_result.x[0]
        bestresult = opt_result.x[1:]
        yhatdiscfacts = discountFactors(mats[1:], zeroYields(coeffs, linph, mats[1:]))

        p = -cf.ix[0, :] # original prices
        phat = np.dot(yhatdiscfacts, cf.drop(cf.index[0])) # optimized prices
        # now yields
        y = cf.apply(y2p.p2y, maturities = mats)
        cfnew = cf.copy()
        cfnew.ix[0, :] = -phat # change prices
        yhat = cfnew.apply(y2p.p2y, maturities = mats)
        # and the maturities
        bondmats = cfobj["cfmat"]["maturities"].apply(max)
        returner = pd.DataFrame({"mats": bondmats, "durations": durations, \
                "y": y, "yhat": yhat, "p": p, \
                "phat": phat}).sort("mats")
    else:
        print("no cashflows matrix for this day")
        bestresult = []
    return {"date": date, "best_result": bestresult, "best_phi": bestphi,
            "funmin": funmin, "yields": returner}
    
def LaguerreOpt5(cashflowslist, numlaguerres, optmethod, basinhop):
    """ performs an optimization after getting an rvOptPrep object """
    coeffs = np.array([5] + [0] * numlaguerres) # initialize phi and betas
    mapper = [(coeffs, x, optmethod, basinhop) for x in cashflowslist]
    print("Starting pool")
    p = Pool(8)
    result = p.map(mp_worker5, mapper)
    p.close()
    p.join()
    return result

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~NONLINEAR END~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LINEAR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
def f4(z, *params):
    """ function for optimizing laguerre with fixed phi (linearised) """
    mats, cfs, weights, linphi = params
    discfcts = discountFactors(mats, zeroYields(z, linphi, mats))
    return np.sum(weights * np.abs(np.dot(discfcts, cfs) ** 2))


def mp_worker(coeffs, phirange, phistep, cfobj, optmethod):
    if cfobj["cfmat"] != -1: # if we have a good cashflows matrix then proceed
        cf = cfobj["cfmat"]["cashflows1"]
        date = cfobj["date"]
        mats = np.array(cf.index.tolist())
        durations = np.dot(mats, cf.values) / np.sum(cf.values[1:, :], 0)
        durationweights = 1 / durations
        durationweights = durationweights / np.mean(durationweights) # average 1
        # now get initial f4 function value
        bestresult = coeffs
        bestphi = phirange[0]
        paramsinit = (mats, cf.values, durationweights, bestphi)
        try:
            funmin = f4(bestresult, *paramsinit)
        except:
            print("error intializing function minimum")
            return([])
        # start looping through phis
        phis = np.arange(phirange[0], phirange[1], phistep)
        for phi in phis:
            params = (mats, cf.values, durationweights, phi)
            try:
                opt_result = minimize(f4, coeffs, args = params, 
                    method = optmethod, options = {"disp": False})
            except:
                opt_result = []
            if opt_result:   
                if opt_result.fun < funmin:   # testif better than previous
                    funmin = opt_result.fun
                    bestphi = phi
                    bestresult = opt_result.x
        yhatdiscfacts= discountFactors(mats[1:], zeroYields(bestresult, bestphi, mats[1:]))
        p = -cf.ix[0, :] # original prices
        phat = np.dot(yhatdiscfacts, cf.drop(cf.index[0])) # optimized prices
        # now yields
        y = cf.apply(y2p.p2y, maturities = mats)
        cfnew = cf.copy()
        cfnew.ix[0, :] = -phat # change prices
        yhat = cfnew.apply(y2p.p2y, maturities = mats)
        # and the maturities
        bondmats = cfobj["cfmat"]["maturities"].apply(max)
        returner = pd.DataFrame({"mats": bondmats, "durations": durations, \
                "y": y, "yhat": yhat, "p": p, \
                "phat": phat}).sort("mats")
    else:
        print("no cashflows matrix for this day")
        bestresult = []
        funmin = []
        returner = []
    return {"date": date, "best_result": bestresult, "best_phi": bestphi,
            "funmin": funmin, "yields": returner}

def LaguerreOpt(cashflowslist, numlaguerres, phirange, phistep, optmethod):
    """ performs an optimization after getting an rvOptPrep object """
    coeffs = np.zeros(numlaguerres) # number of lagpolys
    mapper = [(coeffs, phirange, phistep, x, optmethod) for x in cashflowslist]
    print("Starting pool")
    p = Pool(8)
    result = p.map(mp_worker, mapper)
    p.close()
    p.join()
    return result

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ AFFINE ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
canstop = True

def stopstop():
    global canstop
    canstop = not canstop

def f6(z, *params):
    """ function for optimizing laguerre with fixed phi linearised) with rotation"""
    mats, cfs, weights, linphi = params
    angle = z[0]
    coeffs = z[1:]
    zyields = zeroYields(coeffs, linphi, mats)
    rotator = np.array([np.cos(angle), -np.sin(angle), 
                         np.sin(angle), np.cos(angle)]).reshape(2, 2)
    rotated = np.dot(rotator, np.array([mats, zyields]))
    matsrot = rotated[0]
    zyieldsrot = rotated[1]
    discfcts = np.exp(-mats * zyieldsrot) # discount function
    discfcts[mats == 0] = 1
    discfcts[np.isnan(mats)] = 1
    returner =  np.sum(weights * np.abs(np.dot(discfcts, cfs) ** 2))
    return returner


def mp_worker6(coeffs, phirange, phistep, cfobj, optmethod):
    if cfobj["cfmat"] != -1: # if we have a good cashflows matrix then proceed
        cf = cfobj["cfmat"]["cashflows1"]
        date = cfobj["date"]
        mats = np.array(cf.index.tolist())
        durations = np.dot(mats, cf.values) / np.sum(cf.values[1:, :], 0)
        durationweights = 1 / durations
        durationweights = durationweights / np.mean(durationweights) # average 1
        # now get initial f4 function value
        bestangle = 0
        bestresult = np.array([bestangle] + coeffs.tolist())
        bestphi = phirange[0]
        paramsinit = (mats, cf.values, durationweights, bestphi)
        try:
            funmin = f6(bestresult, *paramsinit)
        except:
            print("error intializing function minimum")
            return([])
        # start looping through phis
        phis = np.arange(phirange[0], phirange[1], phistep)
        for phi in phis:
            params = (mats, cf.values, durationweights, phi)
            anglecoeffs = np.array([bestangle] + coeffs.tolist())
            try:
                opt_result = minimize(f6, anglecoeffs, args = params, 
                    method = optmethod, options = {"disp": False})
            except:
                opt_result = []
            if opt_result:   
                if opt_result.fun < funmin:   # testif better than previous
                    funmin = opt_result.fun
                    bestphi = phi
                    bestresult = opt_result.x[1:]
                    bestangle = opt_result.x[0]
        yhatzeroyields = zeroYields(bestresult, bestphi, mats[1:])
        rotator = np.array([np.cos(bestangle), -np.sin(bestangle), 
                             np.sin(bestangle), np.cos(bestangle)]).reshape(2, 2)
        rotated = np.dot(rotator, np.array([mats[1:], yhatzeroyields]))
        matsrot = rotated[0]
        yhatzeroyieldsrot = rotated[1] # in case cashflows is today rotation is nan
        matsrot[np.isnan(matsrot)] = 0
        yhatzeroyieldsrot[np.isnan(yhatzeroyieldsrot)] = 1
        # now angle back to degrees
        yhatdiscfacts = np.exp(-matsrot * yhatzeroyieldsrot) # check
        p = -cf.iloc[0, :] # original prices
        phat = np.dot(yhatdiscfacts, cf.iloc[1:, :]) # optimized prices
        # now yields
        y = cf.apply(y2p.p2y, maturities = mats)
        cfnew = cf.copy()
        cfnew.ix[0, :] = -phat # change prices
        yhat = cfnew.apply(y2p.p2y, maturities = mats)
        # and the maturities
        bondmats = cfobj["cfmat"]["maturities"].apply(max)
        returner = pd.DataFrame({"mats": bondmats, "durations": durations, \
                "y": y, "yhat": yhat, "p": p, \
                "phat": phat}).sort("mats")
    else:
        print("no cashflows matrix for this day")
        bestresult = []
        funmin = []
        returner = []
    return {"date": date, "best_result": bestresult, "best_phi": bestphi,
            "bestangle": bestangle, "funmin": funmin, "yields": returner}


def AffineOpt(cashflowslist, numlaguerres, phirange, phistep, optmethod):
    """ performs an optimization after getting an rvOptPrep object """
    coeffs = np.zeros(numlaguerres) # number of lagpolys
    mapper = [(coeffs, phirange, phistep, x, optmethod) for x in cashflowslist]
    print("Starting pool")
    p = Pool(8)
    result = p.map(mp_worker6, mapper)
    p.close()
    p.join()
    return result

#sssssssssssssssssssssssssssssssssssssssssssssssss SVENSSON START ssssssssssssssssssssssssssssssssssssssssssssssssssssssss

def nss1(tau1, tau2, m):
    if m.shape == ():
        return 1.0/100.0
    else:
        un = np.zeros(m.shape)
        un.fill(1.0/100)
        return un

def nss2(tau1, tau2, m):
    return((1.0-np.exp(-m/tau1))/(m/tau1))/100

def nss3(tau1, tau2, m):
    return (((1.0-np.exp(-m/tau1))/(m/tau1))-np.exp(-m/tau1))/100

def nss4(tau1, tau2, m):
    return((1.0-np.exp(-m/tau2))/(m/tau2)-np.exp(-m/tau2))/100

def svZeroYields(coeffs, lintau1, lintau2, m = range(1, 31)):
    """ returns the Laguerre zero yields given coefficients and tau for give maturities """
    tau1 = lintau1 
    tau2 = lintau2 
    zeroyields = coeffs[0] * nss1(tau1, tau2, m) + \
                 coeffs[1] * nss2(tau1, tau2, m) + \
                 coeffs[2] * nss3(tau1, tau2, m) + \
                 coeffs[3] * nss4(tau1, tau2, m)
    return(zeroyields)

def svDiscountFactors(coeffs, lintau1, lintau2, m):
    """ This will do have terms of up to len(coeffs) LaguerreInegral polynomials 
    and return the discount factors """
    factors = np.exp(-m * svZeroYields(coeffs, lintau1, lintau2, m)) # discount function
    factors[m == 0] = 1 # handle the divide by zero case in the Laguerredict
    return(factors)

def svf4(z, *params):
    """ function for optimizing laguerre with fixed phi (linearised) """
    mats, cfs, weights, lintau1, lintau2  = params
    discfcts = svDiscountFactors(z, lintau1, lintau2, mats)
    return np.sum(weights * np.abs(np.dot(discfcts, cfs) ** 2))

def svmp_worker(coeffs, cfobj, ttdist, optmethod):
    if cfobj["cfmat"] != -1: # if we have a good cashflows matrix then proceed
        cf = cfobj["cfmat"]["cashflows1"]
        date = cfobj["date"]
        mats = np.array(cf.index.tolist())
        durations = np.dot(mats, cf.values) / np.sum(cf.values[1:, :], 0)
        durationweights = 1 / durations
        durationweights = durationweights / np.mean(durationweights) # average 1
        # now get initial f4 function value
        besttau1 = 1
        besttau2 = besttau1 + ttdist
        bestresult = coeffs
        paramsinit = (mats, cf.values, durationweights, 1, 1+ttdist)
        try:
            funmin = svf4(bestresult, *paramsinit)
        except:
            print("error intializing function minimum")
            return([])
        # start looping through phis
        taus1 = np.arange(1, 20, 0.5)
        taus2 = taus1 + ttdist
        for tau1 in taus1:
            for tau2 in taus2:
                params = (mats, cf.values, durationweights, tau1, tau2)
                try:
                    opt_result = minimize(svf4, coeffs, args = params, 
                        method = optmethod, options = {"disp": False})
                except:
                    print("bad result", tau1, tau2)
                    opt_result = []
                if opt_result:   
                    if opt_result.fun < funmin:   # testif better than previous
                        funmin = opt_result.fun
                        besttau1 = tau1
                        besttau2 = tau2
                        bestresult = opt_result.x
        yhatdiscfacts= svDiscountFactors(bestresult, besttau1, besttau2, mats[1:])
        p = -cf.iloc[0, :] # original prices
        phat = np.dot(yhatdiscfacts, cf.iloc[1:, :]) # optimized prices
        # now yields
        y = cf.apply(y2p.p2y, maturities = mats)
        cfnew = cf.copy()
        cfnew.ix[0, :] = -phat # change prices
        yhat = cfnew.apply(y2p.p2y, maturities = mats)
        # and the maturities
        bondmats = cfobj["cfmat"]["maturities"].apply(max)
        returner = pd.DataFrame({"mats": bondmats, "durations": durations, \
                "y": y, "yhat": yhat, "p": p, \
                "phat": phat}).sort("mats")
    else:
        print("no cashflows matrix for this day")
        bestresult = []
        funmin = []
        returner = []
    return {"date": date, "best_result": bestresult, "besttau1": besttau1, \
            "besttau2": besttau2, "funmin": funmin, "yields": returner}

def SvenssonOpt(cashflowslist, ttdist, optmethod):
    """ performs an optimization after getting an rvOptPrep object """
    ttdist = 5
    coeffs = np.array([0, 0, 0, 0])
    mapper = [(coeffs, x, ttdist, optmethod) for x in cashflowslist]
    print("Starting Svensson pool")
    p = Pool(4)
    result = p.map(svmp_worker, mapper)
    p.close()
    p.join()
    return result

#sssssssssssssssssssssssssssssssssssssss SVENSSON END ssssssssssssssssssssssssssssssssssssssssssssssssssss
def sl1(phi, m):
    if m.shape == ():
        return 1.0
    else:
        un = np.zeros(m.shape)
        un.fill(1.0)
        return un

def sl2(phi, m):
    return ((1 / (phi * m)) * (np.exp(- phi * m) - 1))

def sl3(phi, m):
    return (-(1 / (phi * m)) * (2 * phi * m * np.exp(- phi * m) + np.exp(- phi * m) - 1))

def sl4(phi, m):
    return ((1 / (phi * m)) * (2 * ((phi * m) ** 2) * np.exp(- phi * m) + np.exp(- phi * m) - 1))

def sl5(phi, m):
    return (np.exp(-m* phi) *(-2* m* phi *(m* phi* (2* m* phi-3)+3)+3* np.exp(m* phi)-3))/(3* m* phi)

def sl6(phi,  m):
    return (np.exp(-m* phi)* (2* (m**2)* (phi**2)* (m* phi* (m* phi-4)+6)-3 *np.exp(m* phi)+3))/(3*m* phi)

def sl7(phi,  m):
    return (np.exp(-m* phi) *(-2 *m* phi *(m* phi* (m* phi* (m* phi* (2* m* phi-15)+40)-30)+15)+15* np.exp(m* phi)-15))/(15 *m *phi)

def sl8(phi,  m):
    return (np.exp(-m* phi)* (2* (m**2)* (phi**2)* (m* phi* (m* phi* (2* m* phi* (m* phi-12)+105)-180)+135)-45* np.exp(m* phi)+45))/(45* m* phi)

############################# Svenssons ############################

