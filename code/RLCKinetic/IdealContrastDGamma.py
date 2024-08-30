import numpy as np
from math import sqrt
from numba import njit, prange


@njit("c16(f8,f8,f8,f8,f8,f8,f8,f8,f8)", fastmath=True)
def dGamma(z0, cp, ron, li, rhoi, lambt, lamb, pi, cc):
    rho = rhoi*(1+pi)/(rhoi + pi)
    ri = (pi/(pi+1)*ron)
    wr = 1/sqrt(lambt*li*(cp + cc))
    zi = 1/( 1/(1j*wr*li) + 1j*wr*cp + 1/ri ) + 1/(1j*wr*cc)
    zf = 1/( 1/(1j*wr*(lamb*li)) + 1j*wr*cp + 1/(rho*ri) ) + 1/(1j*wr*cc)
    return ((zf-z0)/(zf+z0)) - ((zi-z0)/(zi+z0))


@njit("f8[:](f8,f8,f8,f8,f8,f8[:],u8,f8,f8[:],f8)", parallel=True, fastmath=True)
def maxdGammaOfLambda(z0, cp, ron, li, rhoi, lamb, tunning, pi, cc, wContrl):
    maxdGammaOfLambdaV = np.empty((len(lamb)), dtype=np.float64)
    dGammaOfLambdaAndCcV = np.empty((len(cc)), dtype=np.float64)
    if tunning == 0:
        for lambdaI in prange(len(lamb)):
            lambt = 1
            for ccI in prange(len(cc)):
                dGammaOfLambdaAndCcV[ccI] = abs(dGamma(z0, cp, ron, li, rhoi, lambt, lamb[lambdaI], pi, cc[ccI]))
            maxdGammaOfLambdaV[lambdaI] = np.max(dGammaOfLambdaAndCcV)
    elif tunning == 1:
        for lambdaI in prange(len(lamb)):
            # lambt = (4*lamb[lambdaI])/(wContrl*(sqrt(lamb[lambdaI]) - 1) + 1 + sqrt(lamb[lambdaI]))**2
            lambt = sqrt(lamb[lambdaI])**(1-wContrl)
            for ccI in prange(len(cc)):
                dGammaOfLambdaAndCcV[ccI] = abs(dGamma(z0, cp, ron, li, rhoi, lambt, lamb[lambdaI], pi, cc[ccI]))
            maxdGammaOfLambdaV[lambdaI] = np.max(dGammaOfLambdaAndCcV)
    elif tunning == 2:
        for lambdaI in prange(len(lamb)):
            lambt = lamb[lambdaI]
            for ccI in prange(len(cc)):
                dGammaOfLambdaAndCcV[ccI] = abs(dGamma(z0, cp, ron, li, rhoi, lambt, lamb[lambdaI], pi, cc[ccI]))
            maxdGammaOfLambdaV[lambdaI] = np.max(dGammaOfLambdaAndCcV)
    return maxdGammaOfLambdaV


@njit("f8[:](f8,f8,f8,f8,f8,f8,u8,f8[:],f8[:],f8)", parallel=True, fastmath=True)
def maxdGammaOfPi(z0, cp, ron, li, rhoi, lamb, tunning, pi, cc, wContrl):
    maxdGammaOfPiV = np.empty((len(pi)), dtype=np.float64)
    dGammaOfPiAndCcV = np.empty((len(cc)), dtype=np.float64)
    if tunning == 0:
        lambt = 1
        for piI in prange(len(pi)):
            for ccI in prange(len(cc)):
                dGammaOfPiAndCcV[ccI] = abs(dGamma(z0, cp, ron, li, rhoi, lambt, lamb, pi[piI], cc[ccI]))
            maxdGammaOfPiV[piI] = np.max(dGammaOfPiAndCcV)
    elif tunning == 1:
        # lambt = (4*lamb)/(wContrl*(sqrt(lamb) - 1) + 1 + sqrt(lamb))**2
        lambt = sqrt(lamb)**(1-wContrl)
        for piI in prange(len(pi)):
            for ccI in prange(len(cc)):
                dGammaOfPiAndCcV[ccI] = abs(dGamma(z0, cp, ron, li, rhoi, lambt, lamb, pi[piI], cc[ccI]))
            maxdGammaOfPiV[piI] = np.max(dGammaOfPiAndCcV)
    elif tunning == 2:
        lambt = lamb
        for piI in prange(len(pi)):
            for ccI in prange(len(cc)):
                dGammaOfPiAndCcV[ccI] = abs(dGamma(z0, cp, ron, li, rhoi, lambt, lamb, pi[piI], cc[ccI]))
            maxdGammaOfPiV[piI] = np.max(dGammaOfPiAndCcV)
    return maxdGammaOfPiV


@njit("f8[:](f8,f8,f8,f8,f8,f8,u8,f8,f8[:],f8)", parallel=True, fastmath=True)
def dGammaOfCc(z0, cp, ron, li, rhoi, lamb, tunning, pi, cc, wContrl):
    dGammaOfCcV = np.empty((len(cc)), dtype=np.float64)
    if tunning == 0:
        lambt = 1
        for ccI in prange(len(cc)):
            dGammaOfCcV[ccI] = abs(dGamma(z0, cp, ron, li, rhoi, lambt, lamb, pi, cc[ccI]))
    elif tunning == 1:
        # lambt = (4*lamb)/(wContrl*(sqrt(lamb) - 1) + 1 + sqrt(lamb))**2
        lambt = sqrt(lamb)**(1-wContrl)
        for ccI in prange(len(cc)):
            dGammaOfCcV[ccI] = abs(dGamma(z0, cp, ron, li, rhoi, lambt, lamb, pi, cc[ccI]))
    elif tunning == 2:
        lambt = lamb
        for ccI in prange(len(cc)):
            dGammaOfCcV[ccI] = abs(dGamma(z0, cp, ron, li, rhoi, lambt, lamb, pi, cc[ccI]))
    return dGammaOfCcV


@njit("c16[:](f8,f8,f8,f8,f8,f8[:],u8,f8,f8[:],f8)", parallel=True, fastmath=True)
def maxComplexdGammaOfLambda(z0, cp, ron, li, rhoi, lamb, tunning, pi, cc, wContrl):
    maxdGammaOfLambdaV = np.empty((len(lamb)), dtype=np.complex128)
    dGammaOfLambdaAndCcV = np.empty((len(cc)), dtype=np.float64)
    if tunning == 0:
        for lambdaI in prange(len(lamb)):
            lambt = 1
            for ccI in prange(len(cc)):
                dGammaOfLambdaAndCcV[ccI] = abs(dGamma(z0, cp, ron, li, rhoi, lambt, lamb[lambdaI], pi, cc[ccI]))
            maxdGammaOfLambdaV[lambdaI] = dGamma(z0, cp, ron, li, rhoi, lambt, lamb[lambdaI], pi, cc[np.argmax(dGammaOfLambdaAndCcV)])
    elif tunning == 1:
        for lambdaI in prange(len(lamb)):
            # lambt = (4*lamb[lambdaI])/(wContrl*(sqrt(lamb[lambdaI]) - 1) + 1 + sqrt(lamb[lambdaI]))**2
            lambt = sqrt(lamb[lambdaI])**(1-wContrl)
            for ccI in prange(len(cc)):
                dGammaOfLambdaAndCcV[ccI] = abs(dGamma(z0, cp, ron, li, rhoi, lambt, lamb[lambdaI], pi, cc[ccI]))
            maxdGammaOfLambdaV[lambdaI] = dGamma(z0, cp, ron, li, rhoi, lambt, lamb[lambdaI], pi, cc[np.argmax(dGammaOfLambdaAndCcV)])
    elif tunning == 2:
        for lambdaI in prange(len(lamb)):
            lambt = lamb[lambdaI]
            for ccI in prange(len(cc)):
                dGammaOfLambdaAndCcV[ccI] = abs(dGamma(z0, cp, ron, li, rhoi, lambt, lamb[lambdaI], pi, cc[ccI]))
            maxdGammaOfLambdaV[lambdaI] = dGamma(z0, cp, ron, li, rhoi, lambt, lamb[lambdaI], pi, cc[np.argmax(dGammaOfLambdaAndCcV)])
    return maxdGammaOfLambdaV


@njit("c16[:](f8,f8,f8,f8,f8,f8,u8,f8[:],f8[:],f8)", parallel=True, fastmath=True)
def maxComplexdGammaOfPi(z0, cp, ron, li, rhoi, lamb, tunning, pi, cc, wContrl):
    maxdGammaOfPiV = np.empty((len(pi)), dtype=np.complex128)
    dGammaOfPiAndCcV = np.empty((len(cc)), dtype=np.float64)
    if tunning == 0:
        lambt = 1
        for piI in prange(len(pi)):
            for ccI in prange(len(cc)):
                dGammaOfPiAndCcV[ccI] = abs(dGamma(z0, cp, ron, li, rhoi, lambt, lamb, pi[piI], cc[ccI]))
            maxdGammaOfPiV[piI] = dGamma(z0, cp, ron, li, rhoi, lambt, lamb, pi[piI], cc[np.argmax(dGammaOfPiAndCcV)])
    elif tunning == 1:
        # lambt = (4*lamb)/(wContrl*(sqrt(lamb) - 1) + 1 + sqrt(lamb))**2
        lambt = sqrt(lamb)**(1-wContrl)
        for piI in prange(len(pi)):
            for ccI in prange(len(cc)):
                dGammaOfPiAndCcV[ccI] = abs(dGamma(z0, cp, ron, li, rhoi, lambt, lamb, pi[piI], cc[ccI]))
            maxdGammaOfPiV[piI] = dGamma(z0, cp, ron, li, rhoi, lambt, lamb, pi[piI], cc[np.argmax(dGammaOfPiAndCcV)])
    elif tunning == 2:
        lambt = lamb
        for piI in prange(len(pi)):
            for ccI in prange(len(cc)):
                dGammaOfPiAndCcV[ccI] = abs(dGamma(z0, cp, ron, li, rhoi, lambt, lamb, pi[piI], cc[ccI]))
            maxdGammaOfPiV[piI] = dGamma(z0, cp, ron, li, rhoi, lambt, lamb, pi[piI], cc[np.argmax(dGammaOfPiAndCcV)])
    return maxdGammaOfPiV


@njit("c16[:](f8,f8,f8,f8,f8,f8,u8,f8,f8[:],f8)", parallel=True, fastmath=True)
def complexdGammaOfCc(z0, cp, ron, li, rhoi, lamb, tunning, pi, cc, wContrl):
    dGammaOfCcV = np.empty((len(cc)), dtype=np.complex128)
    if tunning == 0:
        lambt = 1
        for ccI in prange(len(cc)):
            dGammaOfCcV[ccI] = dGamma(z0, cp, ron, li, rhoi, lambt, lamb, pi, cc[ccI])
    elif tunning == 1:
        # lambt = (4*lamb)/(wContrl*(sqrt(lamb) - 1) + 1 + sqrt(lamb))**2
        lambt = sqrt(lamb)**(1-wContrl)
        for ccI in prange(len(cc)):
            dGammaOfCcV[ccI] = dGamma(z0, cp, ron, li, rhoi, lambt, lamb, pi, cc[ccI])
    elif tunning == 2:
        lambt = lamb
        for ccI in prange(len(cc)):
            dGammaOfCcV[ccI] = dGamma(z0, cp, ron, li, rhoi, lambt, lamb, pi, cc[ccI])
    return dGammaOfCcV
