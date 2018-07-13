import scipy.special as scisp
import numpy as np
import scipy.integrate as integrate
from astropy import constants as const
import misc


class mbs:

    cLight = const.c.cgs.value
    eCharge = const.e.gauss.value
    hPlanck = const.h.cgs.value
    me = const.m_e.cgs.value
    mp = const.m_p.cgs.value
    sigmaT = const.sigma_T.cgs.value
    nuconst = eCharge / (2.0 * np.pi * me * cLight)

    def __init__(self):
        print('Magneto-Bremstrahlung')

    def nu_g(self, B, Z=1.0, m=me):
        '''Cyclotron frequency'''
        return Z * self.eCharge * B / (2.0 * np.pi * m * self.cLight)

    def nu_B(self, B, g, Z=1.0, m=me):
        '''Gyrofrequency'''
        return Z * self.eCharge * B / (2.0 * np.pi * g * m * self.cLight)

    def nu_c(self, B, g):
        '''Synchrotron critical frequency'''
        return 0.5 * 3.0 * self.nu_B(B, g) * g**3

    def chi(self, nu, B, g):
        '''Harmonic frequency'''
        return nu / self.nu_g(B, g)

    def Fsync(self, xx, asym_low=False, asym_high=False):
        '''Synchrotron function'''
        if asym_low:
            return 4.0 * np.pi * np.power(0.5 * xx, 1.0 / 3.0) / (np.sqrt(3.0) * scisp.gamma(1.0 / 3.0))
        elif asym_high:
            return np.sqrt(np.pi * xx * 0.5) * np.exp(-xx)
        else:
            return np.asarray(
                [x * integrate.quad(lambda y: scisp.kv(5.0 / 3.0, y), x, np.inf)[0]
                 for x in xx])

    def Rsync(self, x, asym_low=False, asym_high=False):
        ''' R(x) in Crusius & Schlickeiser (1986)
        '''
        if asym_low:
            return 1.80842 * np.power(x, 1.0/3.0)
        elif asym_high:
            return 0.5 * np.pi * (1.0 - 11.0 / (18.0 * x)) * np.exp(-x)
        else:
            return 0.5 * np.pi * x * (misc.whittW(0.0, 4.0/3.0, x) *
                                      misc.whittW(0.0, 1.0/3.0, x) -
                                      misc.whittW(0.5, 5.0/6.0, x) *
                                      misc.whittW(-0.5, 5.0/6.0, x))

    def SL07(self, x):
        '''Approximation in Schlickeiser & Lerche (2007)'''
        return 1.5 * x * np.power(x, -2.0/3.0) / (0.869 + np.power(x, 1.0/3.0) * np.exp(x))

    def FDB08fit(self, x):
        '''Fit by Finke, Dermer & Boettcher (2008)'''

        def A(x):
            return np.power(10.0, - 0.35775237
                            - 0.83695385 * np.log10(x)
                            - 1.1449608 * np.power(np.log10(x), 2)
                            - 0.68137283 * np.power(np.log10(x), 3)
                            - 0.22754737 * np.power(np.log10(x), 4)
                            - 0.031967334 * np.power(np.log10(x), 5))

        def B(x):
            return np.power(10.0, - 0.35842494
                            - 0.79652041 * np.log10(x)
                            - 1.6113032 * np.power(np.log10(x), 2)
                            + 0.26055213 * np.power(np.log10(x), 3)
                            - 1.6979017 * np.power(np.log10(x), 4)
                            + 0.032955035 * np.power(np.log10(x), 5))

        return np.piecewise(x,
                            [x < 0.01,
                             x >= 0.01 and x < 1.0,
                             x >= 1.0 and x < 10.0,
                             x > 10.0],
                            [self.Rsync(x, asym_low=True),
                             A(x),
                             B(x),
                             self.Rsync(x, asym_high=True)])

    #  ######  #     #    #
    #  #     # ##   ##   # #
    #  #     # # # # #  #   #
    #  ######  #  #  # #     #
    #  #   #   #     # #######
    #  #    #  #     # #     #
    #  #     # #     # #     #

    def RMAfit(self, c, g):
        '''Fit by Rueda-Becerril (2017)'''

        c1 = 3.2180900500625734e-4
        c2 = 0.650532122717873
        c3 = 15.579904689804556

        def Xc(c, g): return 2.0 * c / (3.0 * np.power(g, 2))

        def A(x):
            return np.power(10.0, - 0.7871626401625178
                            - 0.7050933708504841 * np.log10(x)
                            - 0.35531869295610624 * np.power(np.log10(x), 2)
                            - 0.06503312461868385 * np.power(np.log10(x), 3)
                            - 0.0060901233982264096 * np.power(np.log10(x), 4)
                            - 0.00022764616638053332 * np.power(np.log10(x), 5)
                            )

        def B(x):
            return np.power(10.0, - 0.8236455154570651
                            - 0.831668613094906 * np.log10(x)
                            - 0.525630345887699 * np.power(np.log10(x), 2)
                            - 0.22039314697105414 * np.power(np.log10(x), 3)
                            + 0.01669179529512499 * np.power(np.log10(x), 4)
                            - 0.028650695862677572 * np.power(np.log10(x), 5)
                            )

        def theFit(c, g):
            x = Xc(c, g)
            return np.piecewise(x,
                                [x < c1,
                                 x >= c1 and x <= c2,
                                 x > c2 and x < c3,
                                 x >= c3],
                                [lambda x: 1.8084180211028020864 * np.power(x, 1.0 / 3.0),
                                 lambda x: A(x),
                                 lambda x: B(x),
                                 lambda x: np.pi * np.exp(-x) * (1.0 - 11.0 / (18.0 * x))])

        return np.piecewise(c,
                            [c * g <= 0.8, c * g > 0.8],
                            [0.0, lambda x: theFit(x, g)])

    def RMA(self, c, g):
        def Xc(c, g): return 2.0 * c / (3.0 * np.power(g, 2))
        return np.piecewise(c, [c * g < 0.8, c * g >= 0.8],
                            [0.0, lambda c: Xc(c) * self.SL07(Xc(c))])
