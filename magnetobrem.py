import scipy.special as scisp
import numpy as np
import scipy.integrate as integrate
from astropy import constants as const
from SAPyto import misc
import extractor.fromHDF5 as extr
import SAPyto.SRtoolkit as SR

cLight = const.c.cgs.value
eCharge = const.e.gauss.value
hPlanck = const.h.cgs.value
me = const.m_e.cgs.value
mp = const.m_p.cgs.value
sigmaT = const.sigma_T.cgs.value
nuconst = eCharge / (2.0 * np.pi * me * cLight)
halfpi = 0.5 * np.pi
twopi = 2.0 * np.pi


class mbs:

    def __init__(self, **kwargs):
        self.Zq = 1.0
        self.mq = me
        self.__dict__.update(kwargs)

    def nu_g(self, B):
        '''Cyclotron frequency
              nu_g = Z e B / (2 pi m_q c)
        Dafault values: Z = 1.0, m_q = m_e
        '''
        return eCharge * self.Zq * B / (twopi * self.mq * cLight)

    def nu_B(self, B, g):
        '''Gyrofrequency'''
        return eCharge * self.Zq * B / (twopi * g * self.mq * cLight)

    def nu_c(self, B, g, alpha=halfpi):
        '''Synchrotron critical frequency'''
        return 1.5 * self.nu_g(B) * np.sin(alpha) * g**2

    def nu_c_iso(self, B, g):
        '''Synchrotron critical frequency'''
        return 1.5 * self.nu_g(B) * g**2

    def chi(self, nu, B):
        '''Harmonic frequency'''
        return nu / self.nu_g(B)

    def Psyn_iso(gamma, B):
        '''Total synchrotron radiated power for an isotropic distribution of
        velocities. Formula given in Rybicki & Lightman (1985), eq. (6.7b):

            P = (4 / 3) sigma_T c beta^2 gamma^2 (B^2 / 8 pi)
        '''
        return 4.0 * sigmaT * cLight * SR.speed2(gamma) * gamma**2 * B**2 / (24.0 * np.pi)

    def Fsync(self, Xc, asym_low=False, asym_high=False):
        '''Synchrotron function'''
        if asym_low:
            return 4.0 * np.pi * np.power(0.5 * Xc, 1.0 / 3.0) / (np.sqrt(3.0) * scisp.gamma(1.0 / 3.0))
        elif asym_high:
            return np.sqrt(np.pi * Xc * 0.5) * np.exp(-Xc)
        else:
            return np.asarray(
                [x * integrate.quad(lambda y: scisp.kv(5.0 / 3.0, y), x, np.inf)[0] for x in Xc])

    def Rsync(self, Xc, asym_low=False, asym_high=False):
        ''' R(x) in Crusius & Schlickeiser (1986)
        '''
        if asym_low:
            return 1.8084180211028021 * np.power(Xc, 1.0 / 3.0)
        if asym_high:
            return 0.5 * np.pi * (1.0 - 11.0 / (18.0 * Xc)) * np.exp(-Xc)
        return 0.5 * np.pi * Xc * (misc.whittW(0.0, 4.0 / 3.0, Xc)
                                   * misc.whittW(0.0, 1.0 / 3.0, Xc)
                                   - misc.whittW(0.5, 5.0 / 6.0, Xc)
                                   * misc.whittW(-0.5, 5.0 / 6.0, Xc))

    def SL07(self, Xc):
        '''Approximation in Schlickeiser & Lerche (2007)'''
        return 1.5 * np.power(Xc, -2.0/3.0) / (0.869 + np.power(Xc, 1.0/3.0) * np.exp(Xc))

    def FDB08fit(self, Xc):
        '''Fit by Finke, Dermer & Boettcher (2008)'''

        def A(x):
            return np.power(10.0,
                            - 0.3577524
                            - 0.8369539 * np.log10(x)
                            - 1.1449608 * np.power(np.log10(x), 2)
                            - 0.6813728 * np.power(np.log10(x), 3)
                            - 0.2275474 * np.power(np.log10(x), 4)
                            - 0.0319673 * np.power(np.log10(x), 5))

        def B(x):
            return np.power(10.0,
                            - 0.3584249
                            - 0.7965204 * np.log10(x)
                            - 1.6113032 * np.power(np.log10(x), 2)
                            + 0.2605521 * np.power(np.log10(x), 3)
                            - 1.6979017 * np.power(np.log10(x), 4)
                            + 0.0329550 * np.power(np.log10(x), 5))

        return np.piecewise(Xc,
                            [Xc < 0.01,
                             (Xc >= 0.01) & (Xc < 1.0),
                             (Xc >= 1.0) & (Xc < 10.0),
                             Xc >= 10.0],
                            [lambda x: self.Rsync(x, asym_low=True),
                             lambda x: A(x),
                             lambda x: B(x),
                             lambda x: self.Rsync(x, asym_high=True)])

    #
    #  ######  #     #    #
    #  #     # ##   ##   # #
    #  #     # # # # #  #   #
    #  ######  #  #  # #     #
    #  #   #   #     # #######
    #  #    #  #     # #     #
    #  #     # #     # #     #
    def RMAfit(self, Xc, g):
        '''Fit by Rueda-Becerril (2017)'''
        # xg3 = Xc * g**3

        def A(x):
            return np.exp(- 0.7871626401625178
                          - 0.7050933708504841 * np.log(x)
                          - 0.3553186929561062 * np.power(np.log(x), 2)
                          - 0.0650331246186839 * np.power(np.log(x), 3)
                          - 0.0060901233982264 * np.power(np.log(x), 4)
                          - 0.0002276461663805 * np.power(np.log(x), 5))

        def B(x):
            return np.exp(- 0.823645515457065
                          - 0.831668613094906 * np.log(x)
                          - 0.525630345887699 * np.power(np.log(x), 2)
                          - 0.220393146971054 * np.power(np.log(x), 3)
                          + 0.016691795295125 * np.power(np.log(x), 4)
                          - 0.028650695862678 * np.power(np.log(x), 5))

        def theFit(x):
            c1 = 3.218090050062573e-4
            c2 = 0.650532122717873
            c3 = 15.57990468980456
            return np.piecewise(x,
                                [x < c1,
                                 (x >= c1) & (x <= c2),
                                 (x > c2) & (x < c3),
                                 x >= c3],
                                [lambda x:self.Rsync(x, asym_low=True),
                                 lambda x: A(x),
                                 lambda x: B(x),
                                 lambda x: self.Rsync(x, asym_high=True)])
        return np.piecewise(Xc,
                            [Xc * g**3 < 0.53, Xc * g**3 >= 0.53],
                            [0.0, lambda x: theFit(x)])

    def RMA(self, Xc, g):
        return np.piecewise(Xc,
                            [Xc * g**3 <= 0.53, Xc * g**3 > 0.53],
                            [0.0, lambda x: x * self.SL07(x)])


#
#                #######
#   ####  #####     #      ##   #####  #      ######
#  #      #    #    #     #  #  #    # #      #
#   ####  #    #    #    #    # #####  #      #####
#       # #####     #    ###### #    # #      #
#  #    # #         #    #    # #    # #      #
#   ####  #         #    #    # #####  ###### ######
class spTable(object):

    def __init__(self, tabname='spTable.h5'):
        self.Nc = extr.hdf5ExtractScalar(tabname, 'num_chi', group='Params')
        self.Ng = extr.hdf5ExtractScalar(tabname, 'num_gam', group='Params')
        self.chi_min, self.chi_max = extr.hdf5ExtractScalar(tabname, ['chi_min', 'chi_max'], group='Params')
        self.Gmin, self.Gmax = extr.hdf5ExtractScalar(tabname, ['g_min', 'g_max'], group='Params')

        self.RPcoefs, self.log_chi, self.log_xi_min = extr.hdf5Extract1D(tabname, ['RadPow_Coefs', 'chi', 'xi_min'])

        self.RadPower, self.log_xi, self.log_gamma = extr.hdf5Extract2D(tabname, ['RadPower', 'xi', 'gamma'])

        self.xi_max = 0.0
        self.dchi = 1.0 / (self.log_chi[1:] - self.log_chi[:-1])

    def I1_interp(self, chi=None, gamma=None):
        if chi is None:
            lchi = self.log_chi
            Nchi = self.Nc - 1
        else:
            lchi = np.log(chi)
            Nchi = chi.size
        if gamma is None:
            lgamma = np.linspace(np.log(self.Gmin), np.log(self.Gmax), num=self.Ng)
            Ngam = self.Ng - 1
        else:
            lgamma = np.log(gamma)
            Ngam = gamma.size
        # self.chiL = np.empty(numg*numc)
        # gammaL = np.empty(numg*numc)
        interp = []
        for lg in lgamma:
            for lc in lchi:
                i = np.min([self.Nc - 2, np.argmin(np.abs(lc - self.log_chi))])
                if lc < self.log_chi[i]:
                    i = np.max([0, i - 1])
                if lc > self.log_chi[i + 1]:
                    i = np.min([self.Nc - 1, i + 1])
                # print(i)
                u = (lc - self.log_chi[i]) * self.dchi[i]
                u1 = 1.0 - u
                if lg < self.log_xi_min[i]:
                    if lg < self.log_xi_min[i + 1]:
                        interp.append(np.log(1e-200))
                    else:
                        coefs = self.RPcoefs[(i + 1) * self.Ng:(i + 2) * self.Ng]
                        val = misc.chebev(lg, coefs, self.log_xi_min[i + 1], 0.0)
                        interp.append(u * val)
                else:
                    if lg < self.log_xi_min[i + 1]:
                        coefs = self.RPcoefs[i * self.Ng:(i + 1) * self.Ng]
                        val = misc.chebev(lg, coefs, self.log_xi_min[i], 0.0)
                        interp.append(u1 * val)
                    else:
                        coefs = self.RPcoefs[(i + 1) * self.Ng:(i + 2) * self.Ng]
                        val = misc.chebev(lg, coefs, self.log_xi_min[i + 1], 0.0)
                        coefs = self.RPcoefs[i * self.Ng:(i + 1) * self.Ng]
                        valp = misc.chebev(lg, coefs, self.log_xi_min[i], 0.0)
                        interp.append(u1 * valp + u * val)
        return np.reshape(np.asarray(interp), (Ngam, Nchi))


#
#                  #######
#  #####  #  ####     #      ##   #####  #      ######
#  #    # # #         #     #  #  #    # #      #
#  #    # #  ####     #    #    # #####  #      #####
#  #    # #      #    #    ###### #    # #      #
#  #    # # #    #    #    #    # #    # #      #
#  #####  #  ####     #    #    # #####  ###### ######
class disTable(object):

    def __init__(self, tabname='disTable.h5', absor=True, RMA=True):
        self.Nx, self.Ng, self.Nq = extr.hdf5ExtractScalar(tabname, ['num_chi', 'num_gam', 'num_q'], group='Params')
        self.globGmin, self.globGmax = extr.hdf5ExtractScalar(tabname, ['g_min', 'g_max'], group='Params')

        self.log_chi, self.q, self.xi_min = extr.hdf5Extract1D(tabname, ['chi', 'q', 'xi_min'])
        self.jTable = extr.hdf5Extract1D(tabname, 'disTable')
        if absor:
            self.aTable = extr.hdf5Extract1D(tabname, 'adisTable')
        if RMA:
            self.jRMATable = extr.hdf5Extract1D(tabname, 'RMATable')
        if absor and RMA:
            self.aRMATable = extr.hdf5Extract1D(tabname, 'aRMATable')
