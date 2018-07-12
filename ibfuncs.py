import numpy as np
# import logfuns as lf


class logFuncs:

    def __init__():
        print('log functions')

    def log1(self, x, eps=1e-6):
        def e(x): return 0.5 * (x - 1.0)**2
        return np.piecewise(x, [e(x) > eps, e(x) <= eps],
                            [lambda x: np.log(x), lambda x: x - 1.0])

    def log2(self, x, eps=1e-6):
        def e(x): return np.abs((x - 1.0)**3)
        return np.piecewise(x, [e(x) > eps, e(x) <= eps],
                            [lambda x: np.log(x)**2, lambda x: (x - 1.0)**2])

    def log3(self, x, eps=1e-6):
        def e(x): return np.abs(1.5 * (x - 1.0)**4)
        return np.piecewise(x, [e(x) > eps, e(x) <= eps],
                            [lambda x: np.log(x)**3, lambda x: (x - 1.0)**3])

    def log4(self, x, eps=1e-6):
        def e(x): return np.abs(2.0 * (x - 1.0)**5)
        return np.piecewise(x, [e(x) > eps, e(x) <= eps],
                            [lambda x: np.log(x)**4, lambda x: (x - 1.0)**4])

    def log5(self, x, eps=1e-6):
        def e(x): return np.abs(2.5 * (x - 1.0)**6)
        return np.piecewise(x, [e(x) > eps, e(x) <= eps],
                            [lambda x: np.log(x)**5, lambda x: (x - 1.0)**5])


class pwl_funcs:

    def __init__():
        print('Power-law integration functions')

    def P(self, a, s, eps=1e-6):
        '''
                       a
                      /     -s
            P(a, s) = | dx x
                      /
                      1
        '''
        def e(a, s): return np.log(a)**3 * (s - 1.0)**2 / 6.0
        return np.piecewise(a, [e(a, s) > eps, e(a, s) <= eps],
                            [lambda a: (1.0 - a**(1.0 - s)) / (s - 1.0),
                             lambda a: np.log(a)
                             - 0.5 * np.log(a)**2 * (s - 1.0)])

    def Q(self, a, s, eps=1e-6):
        '''
                       a
                      /     -s
            Q(a, s) = | dx x   log(x)
                      /
                      1
        '''
        def e(a, s): return 0.125 * np.log(a)**4 * (s - 1.0)**2
        return np.piecewise(a, [e(a, s) > eps, e(a, s) <= eps],
                            [lambda a: (1.0 - a**(1.0 - s)
                                        * (1.0 + (s - 1.0) * np.log(a)))
                             / (s - 1.0)**2,
                             lambda a: 0.5 * np.log(a)**2
                             - (1.0 / 3.0) * np.log(a)**3 * (s - 1.0)])

    def Q2(self, a, s, eps=1e-6):
        '''
                        a
                       /     -s    2
            Q (a, s) = | dx x   log (x)
             2         /
                       1
        '''
        def e(a, s): return 0.1 * np.log(a)**5 * (s - 1.0)**2
        return np.piecewise(a, [e(a, s) > eps, e(a, s) <= eps],
                            [lambda a: (2.0 * a**s + a
                                        * ((s - 1.0) * np.log(a)
                                           * (np.log(a) - s * np.log(a) - 2.0)
                                           - 2.0)) / (a**s * (s - 1.0)**3),
                             lambda a: np.log(a)**3 / 3.0 - 0.25 * np.log(a)**2 * (s - 1.0)])
