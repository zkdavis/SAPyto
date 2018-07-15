import numpy as np
import numpy.ma as ma
import scipy.optimize as op
import emcee
import corner
import matplotlib.pyplot as pl
from SAPyto.pwlFuncs import PwlInteg as pwl
from SAPyto.magnetobrem import mbs

# TODO read flux density
# TODO integrate flux density
# phFlux = ( integral of fnu / nu ) / hPlanck


def Fph(nu_min, nu_max, freqs, Fnu):
    '''Calculate the photon flux for the frequency band [nu_min, nu_max] from
    a given flux density.
    Input:
        nu_min, nu_max: scalars
        freqs: array
        Fnu: array
    Output:
        Photon flux: scalar
        photon flux spectral indices: array
    '''
    if nu_min < freqs[0] or nu_max > freqs[-1]:
        return print('Error: nu_min and nu_max outside frequencies array')

    fmskd = ma.masked_outside(freqs, nu_min, nu_max)
    nus = fmskd.compressed()
    flux = Fnu[~fmskd.mask] / nus
    num_nus = len(nus)

    integral = 0.0
    ss = []
    for i in range(num_nus - 1):
        if flux[i] > 1e-100 and flux[i + 1] > 1e-100:
            s = np.log(flux[i + 1] / flux[i]) / np.log(nus[i + 1] / nus[i])
            ss.append(s)
            integral += flux[i] * np.power(nus[i], 1.0 - s) * pwl.P(nus[i + 1] / nus[i], s)
    return integral / mbs.hPlanck, np.asarray(ss)


def photon_index(freqs, indices):
    def f(x, a, b, c): return a * b**2 + c
    popt, pcov = op.curve_fit(f, freqs, indices)
    return f(freqs, *popt)
