import numpy as np
import numpy.ma as ma
import SAPyto.misc as misc
import SAPyto.pwlFuncs as pwlf


def conv2Jy(flux):
    '''Convert flux density (in egs cm^{-2} s^{-1} Hz^{-1}) to janskys.
    '''
    return flux * 1e23


def flux_dens(Inu, dL, z, D, R):
    '''Calculates the flux density using the formula in Zheng & Zheng, 2011, ApJ, 728, 105.
    '''
    return (np.pi * ((R**2) * Inu.T) * D**3 * (1.0 + z) / dL**2).T


#
#  #      #  ####  #    # #####  ####  #    # #####  #    # ######  ####
#  #      # #    # #    #   #   #    # #    # #    # #    # #      #
#  #      # #      ######   #   #      #    # #    # #    # #####   ####
#  #      # #  ### #    #   #   #      #    # #####  #    # #           #
#  #      # #    # #    #   #   #    # #    # #   #   #  #  #      #    #
#  ###### #  ####  #    #   #    ####   ####  #    #   ##   ######  ####


class LightCurves:
    def __init__(self):
        pass

    def nearest(self, nu_in, nus, flux):
        '''This function returns the light curve of the frequency nearest to
        the frequency given: nu_in.
        '''
        nu_pos, nu = misc.find_nearest(nus, nu_in)
        print("Nearest frequency: {0} Hz".format(misc.sci_notation(nu)))
        return flux[:, nu_pos]

    def pwl_interp(self, nu_in, numt, nus, flux):
        '''This function returns a power-law interpolated light curve
        '''
        nu_pos, nu = misc.find_nearest(nus, nu_in)
        lc = np.zeros(numt)
        if len(nus) > 1:
            if nu > nu_in:
                nu_pos += 1
            if nus[nu_pos] >= nus[-1]:
                nu_pos -= 1
            for i in range(numt):
                if (flux[i, nu_pos] > 1e-100) & (flux[i, nu_pos + 1] > 1e-100):
                    s = np.log(flux[i, nu_pos + 1] / flux[i, nu_pos]) / np.log(nus[nu_pos + 1] / nus[nu_pos])
                    lc[i] = flux[i, nu_pos] * (nu_in / nus[nu_pos])**s
        else:
            lc = self.nearest(nu, nus, flux)
        return lc

    def integ(self, nu_min, nu_max, numt, freqs, flux):
        '''This function returns the integrated light curve in the given frequency band [nu_min, nu_max]
        '''

        if nu_min < freqs[0]:
            print('nu_min =', nu_min, '\nminimum frequency in array =', freqs[0])
            return np.zeros(numt)
        if nu_max > freqs[-1]:
            print('nu_max =', nu_max, '\nmaximum frequency in array=', freqs[-1])
            return np.zeros(numt)

        if nu_max <= nu_min:
            return self.pwl_interp(nu_min, numt, freqs, flux)

        nu_mskd = ma.masked_outside(freqs, nu_min, nu_max)
        nus = nu_mskd.compressed()
        num_freqs = len(nus)
        Fnu = flux[:, ~nu_mskd.mask] / nus

        pwli = pwlf.PwlInteg()
        lc = np.zeros(numt)

        if nu_min < nus[0]:
            Ftmp = self.pwl_interp(nu_min, numt, nus, Fnu)
            for i in range(numt):
                if (Ftmp[i] > 1e-100) & (Fnu[i, 0] > 1e-100):
                    s = -np.log(Fnu[i, 0] / Ftmp[i]) / np.log(nus[0] / nu_min)
                    lc[i] += Ftmp[i] * nu_min * pwli.P(nus[0] / nu_min, s)
                    # lc[i] += Ftmp[i] * pwli.P(nus[0] / nu_min, s)

        if nu_max > nus[-1]:
            Ftmp = self.pwl_interp(nu_max, numt, nus, Fnu)
            for i in range(numt):
                if (Ftmp[i] > 1e-100) & (Fnu[i, -1] > 1e-100):
                    s = -np.log(Ftmp[i] / Fnu[i, -1]) / np.log(nu_max / nus[-1])
                    lc[i] += Fnu[i, -1] * nus[-1] * pwli.P(nu_max / nus[-1], s)
                    # lc[i] += Fnu[i, -1] * pwli.P(nu_max / nus[-1], s)

        for i in range(numt):
            for j in range(num_freqs - 1):
                if (Fnu[i, j] > 1e-100) & (Fnu[i, j + 1] > 1e-100):
                    s = -np.log(Fnu[i, j + 1] / Fnu[i, j]) / np.log(nus[j + 1] / nus[j])
                    lc[i] += Fnu[i, j] * nus[j] * pwli.P(nus[j + 1] / nus[j], s)
                    # lc[i] += Fnu[i, j] * pwli.P(nus[j + 1] / nus[j], s)

        return lc


#   ####  #####  ######  ####  ##### #####    ##
#  #      #    # #      #    #   #   #    #  #  #
#   ####  #    # #####  #        #   #    # #    #
#       # #####  #      #        #   #####  ######
#  #    # #      #      #    #   #   #   #  #    #
#   ####  #      ######  ####    #   #    # #    #


class spectrum:
    def __init__(self):
        pass

    def nearest(self, t_in, times, flux):
        '''This function returns the spectrum at th nearest time to
        the given one: t_in.
        '''
        t_pos, t = misc.find_nearest(times, t_in)
        print("Nearest time: {0} s".format(misc.sci_notation(t)))
        return flux[t_pos, :]

    def pwl_interp(self, t_in, numf, times, flux):
        '''This function returns a power-law interpolated spectrum at the given time: t_in
        '''
        t_pos, t = misc.find_nearest(times, t_in)
        if t > t_in:
            t_pos += 1
        if times[t_pos] >= times[-1]:
            t_pos -= 1
        spec = np.zeros(numf)
        for j in range(numf):
            if (flux[t_pos, j] > 1e-100) & (flux[t_pos + 1, j] > 1e-100):
                s = np.log(flux[t_pos + 1, j] / flux[t_pos, j]) / np.log(times[t_pos + 1] / times[t_pos])
                spec[j] = flux[t_pos, j] * (t_in / times[t_pos])**s
        return spec

    def integ(self, t_min, t_max, numf, times, flux, ret_tmasked=False):
        '''This function returns the integrated spectrum during the period [t_min, t_max]
        '''

        if t_min < times[0]:
            print('t_min =', t_min, '\nminimum time in array =', times[0])
            return np.zeros(numf)
        if t_max > times[-1]:
            print('t_max =', t_max, '\nmaximum time in array=', times[-1])
            return np.zeros(numf)

        if t_max <= t_min:
            return self.pwl_interp(t_min, numf, times, flux)

        if (t_min == times[0]) & (t_max == times[-1]):
            tt = times
            Fnu = flux
        else:
            t_mskd = ma.masked_outside(times, t_min, t_max)
            tt = t_mskd.compressed()
            Fnu = flux[~t_mskd.mask, :]
        num_times = len(tt)

        pwli = pwlf.PwlInteg()
        spec = np.zeros(numf)

        if t_min < tt[0]:
            Ftmp = self.pwl_interp(t_min, numf, tt, Fnu)
            for j in range(numf):
                if (Ftmp[j] > 1e-100) & (Fnu[0, j] > 1e-100):
                    s = -np.log(Fnu[0, j] / Ftmp[j]) / np.log(tt[0] / t_min)
                    spec[j] += Ftmp[j] * t_min * pwli.P(tt[0] / t_min, s)

        if t_max > times[-1]:
            Ftmp = self.pwl_interp(t_max, numf, tt, Fnu)
            for j in range(numf):
                if (Ftmp[j] > 1e-100) & (Fnu[-1, j] > 1e-100):
                    s = -np.log(Ftmp[j] / Fnu[-1, j]) / np.log(t_max / tt[-1])
                    spec[j] += Fnu[-1, j] * tt[-1] * pwli.P(t_max / tt[-1], s)

        for j in range(numf):
            for i in range(num_times - 1):
                if (Fnu[i, j] > 1e-100) & (Fnu[i + 1, j] > 1e-100):
                    s = -np.log(Fnu[i + 1, j] / Fnu[i, j]) / np.log(tt[i + 1] / tt[i])
                    spec[j] += Fnu[i, j] * tt[i] * pwli.P(tt[i + 1] / tt[i], s)

        if ret_tmasked:
            return spec, tt
        else:
            return spec

    def averaged(self, t_min, t_max, numf, times, flux):
        '''This function returns the averaged spectrum over the period [t_min, t_max]
        '''
        # print(t_min, t_max, numf, times, flux)
        # print(self.integ(t_min, t_max, numf, times, flux, ret_tmasked=True))
        spec, tt = self.integ(t_min, t_max, numf, times, flux, ret_tmasked=True)
        tott = np.sum(tt[1:] - tt[:-1])
        return spec / tott
