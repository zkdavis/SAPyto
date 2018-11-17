import numpy as np
import numpy.ma as ma
import scipy.integrate as sci_integ
import SAPyto.misc as misc
import SAPyto.pwlFuncs as pwlf
import SAPyto.magnetobrem as mbs
import SAPyto.SRtoolkit as srtool


def conv2Jy(flux):
    '''Convert flux density (in egs cm^{-2} s^{-1} Hz^{-1}) to janskys.
    '''
    return flux * 1e23


def Hz2eV(nu):
    '''Convert frequency in hertz to energy in electronvolt
    '''
    return nu * 4.135667662e-15


def eV2Hz(nu):
    '''Convert energy in electronvolt to frequency in hertz
    '''
    return nu * 2.4179937422321953e14


def Hz2m(nu):
    '''Convert frequency in hertz to wavelength in meters
    '''
    return mbs.cLight * 1e-2 / nu


def m2Hz(lamb):
    '''Convert frequency in hertz to wavelength in meters
    '''
    return mbs.cLight * 1e-2 / lamb


def specEnergyFlux(Inu, dL, z, D, R):
    '''Calculates the spectral energy flux of a sphere.
    '''
    return 2.0 * np.pi * R**2 * Inu * D**3 * (1.0 + z) / dL**2


def EnergyFlux(nuInu, dL, D, R):
    '''Calculates the energy flux of a sphere.
    '''
    return 2.0 * np.pi * R**2 * nuInu * D**4 / dL**2


def sec2dy(time):
    '''Convert time in seconds to days
    '''
    return time / 8.64e4


#
#  ###   ##                                    ##
#   #   #   #####          ####  #####   ####    #
#   #  #      #           #    # #    # #         #
#   #  #      #           #    # #####   ####     #
#   #  #      #           #    # #    #      #    #
#   #   #     #           #    # #    # #    #   #
#  ###   ##   #            ####  #####   ####  ##
#                 #######
#
def Itobs(t, nu, jnut, sen_lum, R, muc, Gbulk, muo, z, D):
    pwl = pwlf.PwlInteg()
    Itobs = np.zeros_like(jnut)
    i_edge = np.argmin(np.abs(2.0 * R * muc - sen_lum))
    if (sen_lum[i_edge] > 2.0 * R * muc):
        i_edge = i_edge - 1

    for j in range(nu.size):
        for i in range(t.size):

            if (i <= i_edge):
                i_start = 0
            else:
                i_start = i - i_edge

            for ii in range(i_start, i):
                tob_min = srtool.t_com(t[i], z, Gbulk, muo, x=t[ii - 1] * mbs.cLight * muc)
                tob_max = srtool.t_com(t[i], z, Gbulk, muo, x=t[ii] * mbs.cLight * muc)

                if ii == 0:
                    Itobs[i, j] = np.abs(tob_max - tob_min) * jnut[0, j]
                else:
                    if (jnut[ii, j] > 1e-100) & (jnut[ii - 1, j] > 1e-100):
                        sind = -np.log(jnut[ii, j] / jnut[ii - 1, j]) / np.log(tob_max / tob_min)
                        if (sind < -8.0):
                            sind = -8.0
                        if (sind > 8.0):
                            sind = 8.0
                        Itobs[i, j] = Itobs[i, j] + jnut[ii - 1, j] * tob_min * pwl.P(tob_max / tob_min, sind, 1e-6) / (Gbulk * muc * (muo - srtool.speed(Gbulk)) * D)
    return Itobs

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
        return flux[:, nu_pos] / nu[nu_pos]

    def pwl_interp(self, nu_in, t, nus, flux):
        '''This function returns a power-law interpolated light curve
        '''
        nu_pos, nu = misc.find_nearest(nus, nu_in)
        lc = np.zeros_like(t)
        flux /= nus
        if len(nus) > 1:
            if nu > nu_in:
                nu_pos += 1
            if nus[nu_pos] >= nus[-1]:
                nu_pos -= 1
            for i in range(t.size):
                if (flux[i, nu_pos] > 1e-100) & (flux[i, nu_pos + 1] > 1e-100):
                    s = -np.log(flux[i, nu_pos + 1] / flux[i, nu_pos]) / np.log(nus[nu_pos + 1] / nus[nu_pos])
                    if s > 8.0:
                        s = 8.0
                    if s < -8.0:
                        s = -8.0
                    lc[i] = flux[i, nu_pos] * (nu_in / nus[nu_pos])**s
        else:
            lc = self.nearest(nu, nus, flux)
        return lc

    def integ(self, nu_min, nu_max, t, freqs, flux):
        '''This function returns the integrated light curve in the given frequency band[nu_min, nu_max]
        '''
        licur = np.zeros_like(t)
        if nu_min < freqs[0]:
            print('nu_min =', nu_min, '\nminimum frequency in array =', freqs[0])
            return licur
        if nu_max > freqs[-1]:
            print('nu_max =', nu_max, '\nmaximum frequency in array=', freqs[-1])
            return licur

        if nu_max == nu_min:
            for i in range(t.size):
                # licur[i] = np.interp(nu_max, freqs, flux[i, :])
                licur[i] = np.interp(np.log(nu_max), np.log(freqs), flux[i, :])
        else:
            nu_mskd = ma.masked_outside(freqs, nu_min, nu_max)
            nus = nu_mskd.compressed()
            Fnu = flux[:, ~nu_mskd.mask] / nus
            for i in range(t.size):
                # NOTE: The integral is logarithmic, therfore the nus multiplying Fnu
                licur[i] = sci_integ.simps(nus * Fnu[i, :], x=np.log(nus))

        return licur


#
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

    def pwl_interp(self, t_in, nu, times, flux):
        '''This function returns a power-law interpolated spectrum at the given time: t_in
        '''
        t_pos, t = misc.find_nearest(times, t_in)
        if t > t_in:
            t_pos += 1
        if times[t_pos] >= times[-1]:
            t_pos -= 1
        spec = np.zeros_like(nu)
        numf = nu.size
        for j in range(numf):
            if (flux[t_pos, j] > 1e-100) & (flux[t_pos + 1, j] > 1e-100):
                s = -np.log(flux[t_pos + 1, j] / flux[t_pos, j]) / np.log(times[t_pos + 1] / times[t_pos])
                if s > 8.0:
                    s = 8.0
                if s < -8.0:
                    s = -8.0
                spec[j] = flux[t_pos, j] * (t_in / times[t_pos])**s
        return spec

    def integ(self, t_min, t_max, nu, times, flux, ret_tmasked=False):
        '''This function returns the integrated spectrum during the period[t_min, t_max]
        '''

        if t_min < times[0]:
            print('t_min =', t_min, '\nminimum time in array =', times[0])
            if ret_tmasked:
                return np.zeros_like(nu), times
            else:
                return np.zeros_like(nu)

        if t_max > times[-1]:
            print('t_max =', t_max, '\nmaximum time in array=', times[-1])
            if ret_tmasked:
                return np.zeros_like(nu), times
            else:
                return np.zeros_like(nu)

        if (t_min == times[0]) & (t_max == times[-1]):
            tt = times
            Fnu = flux
        else:
            t_mskd = ma.masked_outside(times, t_min, t_max)
            tt = t_mskd.compressed()
            Fnu = flux[~t_mskd.mask, :]

        spec = np.zeros_like(nu)

        for j in range(nu.size):
            if t_max == t_min:
                # spec[j] = np.interp(t_max, times, flux[:, j])
                spec[j] = np.exp(np.interp(np.log(t_max), np.log(times), np.log(flux[:, j])))
                # tt = np.asarray([t_max])
            else:
                spec[j] = sci_integ.simps(tt * Fnu[:, j], x=np.log(tt))

        if ret_tmasked:
            return spec, tt
        else:
            return spec

    def averaged(self, t_min, t_max, nu, times, flux):
        '''This function returns the averaged spectrum over the period[t_min, t_max]
        '''
        spec, tt = self.integ(t_min, t_max, nu, times, flux, ret_tmasked=True)
        tott = np.sum(tt[1:] - tt[:-1])
        return spec / tott


#
#   #####                                           ######
#  #     #  ####  #    # #####  #####  ####  #    # #     #  ####  #    #
#  #       #    # ##  ## #    #   #   #    # ##   # #     # #    # ##  ##
#  #       #    # # ## # #    #   #   #    # # #  # #     # #    # # ## #
#  #       #    # #    # #####    #   #    # #  # # #     # #    # #    #
#  #     # #    # #    # #        #   #    # #   ## #     # #    # #    #
#   #####   ####  #    # #        #    ####  #    # ######   ####  #    #
def ComptonDom(nus, Fsyn, Fic, t_min, t_max, times):
    spec = spectrum()
    # pwli = pwlf.PwlInteg()
    Nf = nus.size

    # NOTE  synchrotron spectrum and peak
    synint = spec.integ(t_min, t_max, Nf, times, Fsyn)
    syn_pos = synint.argmax()
    syn_peak = synint[syn_pos]
    nu_syn = nus[syn_pos]

    # syntot = 0.0
    # for j in range(Nf - 1):
    #     if (synint[j] > 1e-100) & (synint[j + 1] > 1e-100):
    #         s = -np.log(synint[j + 1] / synint[j]) / np.log(nus[j + 1] / nus[j])
    #         syntot += synint[j] * nus[j] * pwli.P(nus[j + 1] / nus[j], s)

    # NOTE  IC spectrum and peak
    ICint = spec.integ(t_min, t_max, Nf, times, Fic)
    IC_pos = ICint.argmax()
    IC_peak = ICint[IC_pos]
    nu_IC = nus[IC_pos]

    # ICtot = 0.0
    # for j in range(Nf - 1):
    #     if (ICint[j] > 1e-100) & (ICint[j + 1] > 1e-100):
    #         s = -np.log(ICint[j + 1] / ICint[j]) / np.log(nus[j + 1] / nus[j])
    #         ICtot += ICint[j] * nus[j] * pwli.P(nus[j + 1] / nus[j], s)

    A_C = IC_peak / syn_peak
    return nu_syn, nu_IC, A_C
