import scipy.special as scisp
import numpy as np
import scipy.integrate as integrate
from SAPyto import misc
import SAPyto.SRtoolkit as SR
import SAPyto.pwlFuncs as pwlf
import SAPyto.constants as C


class mbs:

    def __init__(self, **kwargs):
        self.Zq = 1.0
        self.mq = C.me
        self.__dict__.update(kwargs)

    def nu_g(self, B):
        '''Cyclotron frequency
              nu_g = Z e B / (2 pi m_q c)
        Dafault values: Z = 1.0, m_q = m_e
        '''
        return C.eCharge * self.Zq * B / (C.twopi * self.mq * C.cLight)

    def nu_B(self, B, g):
        '''Gyrofrequency'''
        return C.eCharge * self.Zq * B / (C.twopi * g * self.mq * C.cLight)

    def nu_c(self, B, g, alpha=C.halfpi):
        '''Synchrotron critical frequency'''
        return 1.5 * self.nu_g(B) * np.sin(alpha) * g**2

    def nu_c_iso(self, B, g):
        '''Synchrotron critical frequency'''
        return 1.5 * self.nu_g(B) * g**2

    def chi(self, nu, B):
        '''Harmonic frequency'''
        return nu / self.nu_g(B)

    def Psyn_iso(self, gamma, B):
        '''Total synchrotron radiated power for an isotropic distribution of
        velocities. Formula given in Rybicki & Lightman (1985), eq. (6.7b):

            P = (4 / 3) sigma_T c beta^2 gamma^2 (B^2 / 8 pi)
        '''
        return 4.0 * C.sigmaT * C.cLight * SR.speed2(gamma) * gamma**2 * B**2 / (24.0 * np.pi)

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
        '''Fit by Rueda-Becerril (2017)

        RMAfit(x) = Rsync(x) = 0.5 pi x CS(x)
        '''
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
                                [lambda x: self.Rsync(x, asym_low=True),
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
#  ###### #    # #  ####   ####  # #    # # ##### #   #
#  #      ##  ## # #      #      # #    # #   #    # #
#  #####  # ## # #  ####   ####  # #    # #   #     #
#  #      #    # #      #      # # #    # #   #     #
#  #      #    # # #    # #    # #  #  #  #   #     #
#  ###### #    # #  ####   ####  #   ##   #   #     #
def j_mb(self, nu, B, n0, gmin, gmax, qind):
    '''Description:
    This function reproduces the MBS emissivity from a power-law distribution.
    '''
    MBS = mbs()

    def f(g, c=1.0, q=2.5):
        Xc = 2.0 * c / (3.0 * g**2)
        return g**(1.0 - q) * MBS.RMAfit(Xc, g)

    lf = pwlf.logFuncs()

    calc_jnu: do k = 1, Ng - 1
      if (nn(k) > 1d-100 .and. nn(k + 1) > 1d-100) then
        qq = -dlog(nn(k + 1) / nn(k)) / dlog(gg(k + 1) / gg(k))
          if (qq > 8d0) qq = 8d0
          if (qq < -8d0) qq = -8d0
          jnu = jnu + j_mb(freq, B, nn(k), gg(k), gg(k + 1), qq, RMA_new)
       end if
    end do calc_jnu
    if (jnu < 1d-200) jnu = 0d0

    jmbc = 0.125 * C.jmbConst
    nuB = C.nuConst * B
    chi = nu / nuB
    lchi = np.log(chi)
    lxi_min = np.log(gmin / self.globGmax)
    lxi_max = np.log(gmax / self.globGmax)

    if jmbtab is None:
        jmbtab = self.jTable

    if lchi < self.log_chi[0]:
        return "j_mb: input nu below table nu_min"

    i = np.argmin(np.abs(lchi - self.log_chi))
    if self.log_xi_min[i] >= lxi_max:
        return 0.0
    elif (self.log_xi_min[i] >= lxi_min) & (self.log_xi_min[i] < lxi_max):
        I3min = np.exp(self.I3_interp(lchi, self.log_xi_min[i], qind, chtab=jmbtab))
        I3max = np.exp(self.I3_interp(lchi, lxi_max, qind, chtab=jmbtab))
    else:
        I3min = np.exp(self.I3_interp(lchi, lxi_min, qind, chtab=jmbtab))
        I3max = np.exp(self.I3_interp(lchi, lxi_max, qind, chtab=jmbtab))

    I3diff = I3min - I3max
    I3rel = np.abs(I3diff) / np.abs(I3min)

    if (I3rel < 1e-3) | (I3diff < 0.0):
        I2 = C.chunche_c100g20 * integrate.romberg(f, gmin, gmax, args=(chi, qind), divmax=12)
    else:
        if 0.5 * lf.log2(self.globGmax, 1e-6) * (qind - 1.0)**2 < 1e-3:
            I2 = (1.0 - lf.log1(self.globGmax, 1e-9) * (qind - 1.0)) * I3diff
        else:
            I2 = self.globGmax**(1.0 - qind) * I3diff

    return jmbc * nuB * n0 * I2 * gmin**qind


#
#    ##   #####   ####   ####  #####  #####  ##### #  ####  #    #
#   #  #  #    # #      #    # #    # #    #   #   # #    # ##   #
#  #    # #####   ####  #    # #    # #    #   #   # #    # # #  #
#  ###### #    #      # #    # #####  #####    #   # #    # #  # #
#  #    # #    # #    # #    # #   #  #        #   # #    # #   ##
#  #    # #####   ####   ####  #    # #        #   #  ####  #    #
def a_mb(self, nu, B, n0, gmin, gmax, qind, ambtab=None):
    '''Description:
    This function reproduces the MBS absorption from a power-law distribution.
    '''
    MBS = mbs()

    def f(g, c=1.0, q=2.5):
        Xc = 2.0 * c / (3.0 * g**2)
        return g**(-q) * MBS.RMAfit(Xc, g) * (q + 1.0 + (g**2 / (g**2 - 1.0)))
    lf = pwlf.logFuncs()

    ambc = 3.90625e-3 * C.ambConst
    nuB = C.nuConst * B
    chi = nu / nuB
    lchi = np.log(chi)
    lxi_min = np.log(gmin / self.globGmax)
    lxi_max = np.log(gmax / self.globGmax)

    if ambtab is None:
        ambtab = self.aTable

    if lchi < self.log_chi[0]:
        return "a_mb: input nu below table nu_min"

    i = np.argmin(np.abs(lchi - self.log_chi))
    if self.log_xi_min[i] >= lxi_max:
        return 0.0
    elif (self.log_xi_min[i] >= lxi_min) & (self.log_xi_min[i] < lxi_max):
        A3min = np.exp(self.I3_interp(lchi, self.log_xi_min[i], qind, chtab=ambtab))
        A3max = np.exp(self.I3_interp(lchi, lxi_max, qind, chtab=ambtab))
    else:
        A3min = np.exp(self.I3_interp(lchi, lxi_min, qind, chtab=ambtab))
        A3max = np.exp(self.I3_interp(lchi, lxi_max, qind, chtab=ambtab))

    A3diff = A3min - A3max
    A3rel = np.abs(A3diff) / np.abs(A3min)

    if (A3rel < 1e-3) | (A3diff < 0.0):
        # A2 = 0.0
        A2 = integrate.romberg(f, gmin, gmax, args=(chi, qind))
    else:
        if 0.5 * lf.log2(self.globGmax, 1e-6) * (qind - 1.0)**2 < 1e-3:
            A2 = (1.0 - lf.log1(self.globGmax, 1e-9) * (qind - 1.0)) * A3diff
        else:
            A2 = self.globGmax**(1.0 - qind) * A3diff

    # A2 = integrate.romberg(f, gmin, gmax, args=(chi, qind), divmax=12)

    return ambc * nuB * n0 * A2 * gmin**qind / nu**2
