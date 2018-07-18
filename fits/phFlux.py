import numpy as np
import numpy.ma as ma
import scipy.optimize as op
# import emcee
# import corner
# import matplotlib.pyplot as pl
import SAPyto.pwlFuncs as pwlf
import SAPyto.magnetobrem as mbs


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

    nu_mskd = ma.masked_outside(freqs, nu_min, nu_max)
    nus = nu_mskd.compressed()
    flux = Fnu[~nu_mskd.mask] / nus
    num_nus = len(nus)

    integral = 0.0
    pwli = pwlf.PwlInteg()
    for i in range(num_nus - 1):
        if (flux[i] > 1e-100) & (flux[i + 1] > 1e-100):
            s = -np.log(flux[i + 1] / flux[i]) / np.log(nus[i + 1] / nus[i])
            integral += flux[i] * np.power(nus[i], 1.0 - s) * pwli.P(nus[i + 1] / nus[i], s)
    return nus[:-1], flux[:-1], integral / mbs.hPlanck


def photon_index(freqs, fluxes):
    def f(x, a, b): return a * x + b
    popt, pcov = op.curve_fit(f, np.log10(freqs), np.log10(fluxes))
    return np.power(10.0, f(np.log10(freqs), *popt)), popt, pcov
