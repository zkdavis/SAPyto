
#### F(x)
def Fsync(xx):
    import numpy as np
    from scipy.special import kv
    import scipy.integrate as integrate

    res = []
    for x in xx:
        res.append(x * integrate.quad(lambda y: kv(5.0/3.0, y), x, np.inf)[0])
    return np.asarray(res)

def asymFsync_low(x):
    import numpy as np
    from scipy.special import gamma
    return 4.0 * np.pi * np.power(x/2.0, 1.0/3.0) / (np.sqrt(3) * gamma(1.0 / 3.0))
    
def asymFsync_high(x):
    import numpy as np
    return np.sqrt(np.pi * x * 0.5) * np.exp(-x)

#### R(x) in Crusius & Schlickeiser (1986)
def Rsync(x):
    from WhitFuncs import whitw
    import numpy as np
    
    return 0.5 * np.pi * x * (whitw(0.0, 4.0/3.0, x) *
                              whitw(0.0, 1.0/3.0, x) -
                              whitw(0.5, 5.0/6.0, x) *
                              whitw(-0.5, 5.0/6.0, x))

def asymRsync_low(x):
    import numpy as np
    return 1.80842 * np.power(x, 1.0/3.0)

def asymRsync_high(x):
    import numpy as np
    return 0.5 * np.pi * (1.0 - 11.0 / (18.0 * x)) * np.exp(-x)

#### Approximation in Schlickeiser & Lerche (2007)
def SL07(x):
    import numpy as np
    return 1.5 * x * np.power(x, -2.0/3.0) / (0.869 + np.power(x, 1.0/3.0) * np.exp(x))

#### Fit by Finke, Dermer & Boettcher (2008)
def FDBfitA(x):
    import numpy as np
    return np.power(10.0,
                    - 0.35775237 - 0.83695385 * np.log10(x)
                    - 1.1449608 * np.power(np.log10(x), 2)
                    - 0.68137283 * np.power(np.log10(x), 3)
                    - 0.22754737 * np.power(np.log10(x), 4)
                    - 0.031967334 * np.power(np.log10(x), 5))

def FDBfitB(x):
    import numpy as np
    return np.power(10.0,
                    -0.35842494 - 0.79652041 * np.log10(x)
                    - 1.6113032 * np.power(np.log10(x), 2)
                    + 0.26055213 * np.power(np.log10(x), 3)
                    - 1.6979017 * np.power(np.log10(x), 4)
                    + 0.032955035 * np.power(np.log10(x), 5))

def FDBfit(x):
    import numpy as np

    def theFit(x):
        if x < 0.01:
            return asymRsync_low(x)
        elif x >= 0.01 and x < 1.0:
            return A(x)
        elif x >= 1.0 and x < 10.0:
            return B(x)
        else:
            return asymRsync_high(x)
    
    if type(x) is float:
        return theFit(x)
    else:
        sl = []
        for ex in x:
            sl.append(theFit(ex))
        return np.array(sl)
