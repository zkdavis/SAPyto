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
        return 0.5 * np.pi * Xc * (misc.whittW(0.0, 4.0 / 3.0, Xc) *
                                   misc.whittW(0.0, 1.0 / 3.0, Xc) -
                                   misc.whittW(0.5, 5.0 / 6.0, Xc) *
                                   misc.whittW(-0.5, 5.0 / 6.0, Xc))

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
