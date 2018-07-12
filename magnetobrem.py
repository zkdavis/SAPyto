import scipy.special as scisp
import numpy as np
import scipy.integrate as integrate
import misc


class magnetobrem:

    def __init__(self):
        print('Magneto-Bremstrahlung')

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

    def FDBfit(self, x):
        '''Fit by Finke, Dermer & Boettcher (2008)'''

        def A(x):
            return np.power(10.0, - 0.35775237
                            - 0.83695385 * np.log10(x)
                            - 1.1449608 * np.power(np.log10(x), 2)
                            - 0.68137283 * np.power(np.log10(x), 3)
                            - 0.22754737 * np.power(np.log10(x), 4)
                            - 0.031967334 * np.power(np.log10(x), 5)
                            )

        def B(x):
            return np.power(10.0, - 0.35842494
                            - 0.79652041 * np.log10(x)
                            - 1.6113032 * np.power(np.log10(x), 2)
                            + 0.26055213 * np.power(np.log10(x), 3)
                            - 1.6979017 * np.power(np.log10(x), 4)
                            + 0.032955035 * np.power(np.log10(x), 5)
                            )

        def theFit(x):
            if x < 0.01:
                return self.Rsync(x, asym_low=True)
            elif x >= 0.01 and x < 1.0:
                return A(x)
            elif x >= 1.0 and x < 10.0:
                return B(x)
            else:
                return self.Rsync(x, asym_high=True)

        if type(x) is float:
            return theFit(x)
        elif type(x) in [list, np.ndarray]:
            return np.asarray([theFit(ex) for ex in x])
        else:
            return("Wrong type for argument x")

    #  ######  #     #    #
    #  #     # ##   ##   # #
    #  #     # # # # #  #   #
    #  ######  #  #  # #     #
    #  #   #   #     # #######
    #  #    #  #     # #     #
    #  #     # #     # #     #

    def RMAfit(self, chi, g):
        '''Fit by Rueda-Becerril (2017)'''

        c1 = 3.2180900500625734e-4
        c2 = 0.650532122717873
        c3 = 15.579904689804556

        x = 2.0 * chi / (3.0 * g**2)

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

        def theFit(x):
            if x < c1:
                return 1.8084180211028020864 * np.power(x, 1.0 / 3.0)
            elif x >= c1 and x <= c2:
                return A(x)
            elif x > c2 and x < c3:
                return B(x)
            else:
                return np.pi * np.exp(-x) * (1.0 - 11.0 / (18.0 * x))

        if type(chi) is float:
            if x < 0.8 / g:
                return 0.0
            else:
                return theFit(x)
        elif type(chi) in [list, np.ndarray]:
            return np.piecewise(x,
                                [chi * g <= 0.8, chi * g > 0.8],
                                [lambda x: 0.0, lambda x: theFit(x)])
        else:
            return("Wrong type for argument x")
