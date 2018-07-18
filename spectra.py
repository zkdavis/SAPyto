import os
import numpy as np
import numpy.ma as ma
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import matplotlib.colors as col
# import extractor as extr
import SAPyto.misc as misc
import SAPyto.pwlFuncs as pwlf

# TODO produce spectra from nearest t
# TODO produce spectra from integrating light curve over time


def conv2Jy(flux):
    '''Convert flux density (in egs cm^{-2} s^{-1} Hz^{-1}) to janskys.
    '''
    return flux * 1e23


#
#  #      #  ####  #    # #####  ####  #    # #####  #    # ######  ####
#  #      # #    # #    #   #   #    # #    # #    # #    # #      #
#  #      # #      ######   #   #      #    # #    # #    # #####   ####
#  #      # #  ### #    #   #   #      #    # #####  #    # #           #
#  #      # #    # #    #   #   #    # #    # #   #   #  #  #      #    #
#  ###### #  ####  #    #   #    ####   ####  #    #   ##   ######  ####
#


class LightCurves:

    def __init__(self):
        pass

    def nearest(self, nu_in, nus, flux):
        '''This function returns the light curve of the frequency nearest to
        the frequency given: nu_in.
        '''
        i_nu, nu = misc.find_nearest(nus, nu_in)
        print("Nearest frequency: {0} Hz".format(nu))
        return flux[:, i_nu]

    def pwl_interp(self, nu_in, nus, numt, flux):
        '''This function returns an interpolated light curve
        '''
        nu_pos, nu = misc.find_nearest(nus, nu_in)
        lc = np.zeros(numt)
        for i in range(numt):
            if (flux[i, nu_pos] > 1e-100) & (flux[i, nu_pos + 1] > 1e-100):
                s = np.log(flux[i, nu_pos + 1] / flux[i, nu_pos]) / np.log(nus[nu_pos + 1] / nus[nu_pos])
            lc[i] = flux[:, nu_pos] * (nu_in / nus[nu_pos])**s
        return lc

    def integ(self, nu_min, nu_max, numt, freqs, flux):
        '''This function returns the integrated light curve in the given frequency band [nu_min, nu_max]
        '''

        if nu_min < freqs[0]:
            print('nu_min =', nu_min, '\nminimum frequency in array =', freqs[0])
            return np.zeros(numt)
        if nu_max > freqs[-1]:
            print('nu_max =', nu_max, '\nmaximum frequency in array=', freqs[0])
            return np.zeros(numt)

        if nu_max <= nu_min:
            lc = self.pwl_interp(nu_min, freqs, numt, flux)
            return lc

        nu_mskd = ma.masked_outside(freqs, nu_min, nu_max)
        nus = nu_mskd.compressed()
        num_freqs = len(nus)
        Fnu = flux[:, ~nu_mskd.mask]

        pwli = pwlf.PwlInteg()
        lc = np.zeros(numt)

        if nu_min < nus[0]:
            Ftmp = self.pwl_interp(nu_min, freqs, numt, flux)
            for i in range(numt):
                if (Ftmp[i] > 1e-100) & (Fnu[i, 0] > 1e-100):
                    s = -np.log(Fnu[i, 0] / Ftmp[i]) / np.log(nus[0] / nu_min)
                    lc += Ftmp[i] * nu_min * pwli.P(nus[0] / nu_min, s)

        if nu_max > nus[-1]:
            Ftmp = self.pwl_interp(nu_max, freqs, numt, flux)
            for i in range(numt):
                if (Ftmp[i] > 1e-100) & (Fnu[i, -1] > 1e-100):
                    s = -np.log(Ftmp[i] / Fnu[i, -1]) / np.log(nu_max / nus[-1])
                    lc += Fnu[i, -1] * nus[-1] * pwli.P(nu_max / nus[-1], s)

        for i in range(numt):
            for j in range(num_freqs - 1):
                if Fnu[i, j] > 1e-100 and Fnu[i, j + 1] > 1e-100:
                    s = -np.log(Fnu[i, j + 1] / Fnu[i, j]) / np.log(nus[j + 1] / nus[j])
                    lc += Fnu[i, j] * nus[j] * pwli.P(nus[j + 1] / nus[j], s)

        return lc
#
#   ####  #####  ######  ####  ##### #####    ##
#  #      #    # #      #    #   #   #    #  #  #
#   ####  #    # #####  #        #   #    # #    #
#       # #####  #      #        #   #####  ######
#  #    # #      #      #    #   #   #   #  #    #
#   ####  #      ######  ####    #   #    # #    #
#


class spectrum:

    def __init__(self):
        pass
