import scipy.special as scisp
import numpy as np
import scipy.integrate as integrate
from astropy import constants as const
from SAPyto import misc

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

    def __init__(self, Zq=1.0, mq=me):
        self.Zq = Zq
        self.mq = mq

    def nu_g(self, B):
        '''Cyclotron frequency'''
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

    def Fsync(self, nu, B, g, asym_low=False, asym_high=False):
        '''Synchrotron function'''
        xx = nu / self.nu_c(B, g)
        if asym_low:
            return 4.0 * np.pi * np.power(0.5 * xx, 1.0 / 3.0) / (np.sqrt(3.0) * scisp.gamma(1.0 / 3.0))
        elif asym_high:
            return np.sqrt(np.pi * xx * 0.5) * np.exp(-xx)
        else:
            return np.asarray(
                [x * integrate.quad(lambda y: scisp.kv(5.0 / 3.0, y), x, np.inf)[0]
                 for x in xx])

    def Rsync(self, nu, B, g, asym_low=False, asym_high=False):
        ''' R(x) in Crusius & Schlickeiser (1986)
        '''
        x = nu / self.nu_c_iso(B, g)

        if asym_low:
            return 1.80842 * np.power(x, 1.0 / 3.0)
        if asym_high:
            return 0.5 * np.pi * (1.0 - 11.0 / (18.0 * x)) * np.exp(-x)

        return 0.5 * np.pi * x * (misc.whittW(0.0, 4.0/3.0, x) *
                                  misc.whittW(0.0, 1.0/3.0, x) -
                                  misc.whittW(0.5, 5.0/6.0, x) *
                                  misc.whittW(-0.5, 5.0/6.0, x))

    def SL07(self, nu, B, g):
        '''Approximation in Schlickeiser & Lerche (2007)'''
        x = nu / self.nu_c_iso(B, g)
        return 1.5 * x * np.power(x, -2.0/3.0) / (0.869 + np.power(x, 1.0/3.0) * np.exp(x))

    def FDB08fit(self, nu, B, g):
        '''Fit by Finke, Dermer & Boettcher (2008)'''

        nuc = self.nu_c_iso(B, g)
        x = nu / nuc
        low = {'B': B, 'g': g, 'asym_low': True}
        high = {'B': B, 'g': g, 'asym_high': True}

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

        return np.piecewise(x,
                            [x < 0.01,
                             (x >= 0.01) & (x < 1.0),
                             (x >= 1.0) & (x < 10.0),
                             x >= 10.0],
                            [lambda x: self.Rsync(x * nuc, **low),
                             lambda x: A(x),
                             lambda x: B(x),
                             lambda x: self.Rsync(x * nuc, **high)])

    #  ######  #     #    #
    #  #     # ##   ##   # #
    #  #     # # # # #  #   #
    #  ######  #  #  # #     #
    #  #   #   #     # #######
    #  #    #  #     # #     #
    #  #     # #     # #     #

    def RMAfit(self, nu, B, g):
        '''Fit by Rueda-Becerril (2017)'''

        x = nu / self.nu_c_iso(B, g)
        xg3 = x * g**3

        c1 = 3.218090050062573e-4
        c2 = 0.650532122717873
        c3 = 15.57990468980456

        def A(x):
            return np.power(10.0,
                            - 0.7871626401625178
                            - 0.7050933708504841 * np.log10(x)
                            - 0.3553186929561062 * np.power(np.log10(x), 2)
                            - 0.0650331246186839 * np.power(np.log10(x), 3)
                            - 0.0060901233982264 * np.power(np.log10(x), 4)
                            - 0.0002276461663805 * np.power(np.log10(x), 5)
                            )

        def B(x):
            return np.power(10.0,
                            - 0.823645515457065
                            - 0.831668613094906 * np.log10(x)
                            - 0.525630345887699 * np.power(np.log10(x), 2)
                            - 0.220393146971054 * np.power(np.log10(x), 3)
                            + 0.016691795295125 * np.power(np.log10(x), 4)
                            - 0.028650695862678 * np.power(np.log10(x), 5)
                            )

        def theFit(x):
            return np.piecewise(x,
                                [x < c1,
                                 (x >= c1) & (x <= c2),
                                 (x > c2) & (x < c3),
                                 x >= c3],
                                [lambda x: 1.8084180211028020864 * np.power(x, 1.0 / 3.0),
                                 lambda x: A(x),
                                 lambda x: B(x),
                                 lambda x: np.pi * np.exp(-x) * (1.0 - 11.0 / (18.0 * x))])

        return np.piecewise(xg3,
                            [xg3 <= 0.53, xg3 > 0.53],
                            [0.0, lambda x: theFit(x / g**3)])

    def RMA(self, nu, B, g):
        nuc = self.nu_c_iso(B, g)
        x = nu / nuc
        return np.piecewise(x,
                            [x * g**3 <= 0.53, x * g**3 > 0.53],
                            [0.0, lambda x: x * self.SL07(x * nuc, B, g)])
