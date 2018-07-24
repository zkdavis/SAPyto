import numpy as np

np.logspace


class SPN98(object):
    '''Following This is the setup for the model in Sari, Piran & Narayan, 1998,
    ApJ, 497, L17.
    '''

    def __init__(self, **kwargs):
        # self.td = 1.0
        self.eps_B = 0.5
        self.g2 = 1.0
        self.n1 = 1.0
        self.E52 = 1.0
        self.D28 = 1.0
        # self.nu15 = 1.0
        self.eps_e = 0.6
        self.pind = 2.5
        self.__dict__.update(kwargs)

    #
    #  ###### #####  ######  ####   ####
    #  #      #    # #      #    # #
    #  #####  #    # #####  #    #  ####
    #  #      #####  #      #  # #      #
    #  #      #   #  #      #   #  #    #
    #  #      #    # ######  ### #  ####
    #
    def nuc(self, td, adiab=True):
        if adiab:
            # NOTE: Following Eq. (11)
            return 2.7e12 * (np.power(self.eps_B, -1.5)
                             / (np.sqrt(self.E52 * td) * self.n1))
        else:
            # NOTE: Following Eq. (12)
            return 1.3e13 * (np.power(self.eps_B, -1.5)
                             * np.power(self.g2 / self.E52, 4.0 / 7.0)
                             * np.power(td, -2.0 / 7.0)
                             * np.power(self.n1, -13.0 / 14.0))

    def num(self, td, adiab=True):
        if adiab:
            # NOTE: Following Eq. (11)
            return 5.7e14 * (np.sqrt(self.eps_B)
                             * self.eps_e**2
                             * np.sqrt(self.E52)
                             / np.power(td, 1.5))
        else:
            # NOTE: Following Eq. (12)
            return 1.2e14 * (np.sqrt(self.eps_B)
                             * self.eps_e**2
                             * np.power(self.E52 / self.g2, 4.0 / 7.0)
                             * np.power(self.n1, -1.0 / 14.0)
                             * np.power(td, -12.0 / 7.0))

    def nu0(self, adiab=True):
        if adiab:
            nu0 = 1.8e11 * (np.power(self.eps_B, -2.5)
                            * np.power(self.n1, -1.5)
                            / (self.eps_e * self.E52))
        else:
            nu0 = 8.5e12 * (np.power(self.eps_B, -1.9)
                            * np.power(self.eps_e, -0.4)
                            * np.power(self.n1, -1.1)
                            * np.power(self.g2 / self.E52, 0.8))
        return nu0

    #
    #  ##### # #    # ######  ####
    #    #   # ##  ## #      #
    #    #   # # ## # #####   ####
    #    #   # #    # #           #
    #    #   # #    # #      #    #
    #    #   # #    # ######  ####
    #
    def tc(self, nu15, adiab=True):
        if adiab:
            return 7.3e-6 / (self.eps_B**3
                             * self.E52
                             * self.n1**2
                             * nu15**2)
        else:
            return 2.7e-7 * (np.power(self.eps_B, -0.25 * 21.0)
                             * self.g2**2
                             * np.power(self.n1, -0.25 * 13.0)
                             * np.power(nu15, -0.5 * 7.0)
                             / self.E52**2)

    def tm(self, nu15, adiab=True):
        if adiab:
            return 0.69 * (np.power(self.eps_B, 1.0 / 3.0)
                           * np.power(self.eps_e, 4.0 / 3.0)
                           * np.power(self.E52, 1.0 / 3.0)
                           * np.power(nu15, -2.0 / 3.0))
        else:
            return 0.29 * (np.power(self.eps_B, 7.0 / 24.0)
                           * np.power(self.eps_e, 7.0 / 6.0)
                           * np.power(self.E52, 1.0 / 3.0)
                           * np.power(self.g2, -1.0 / 3.0)
                           * np.power(self.n1, -1.0 / 24.0)
                           * np.power(nu15, -7.0 / 12.0))

    def t0(self, adiab=True):
        if adiab:
            t0 = 210.0 * (self.eps_B**2
                          * self.eps_e**2
                          * self.E52
                          * self.n1)
        else:
            t0 = 4.6 * (np.power(self.eps_B * self.eps_e, 1.4)
                        * np.power(self.E52 / self.g2, 0.8)
                        * np.power(self.n1, 0.6))
        return t0

    #
    #  ###### #      #    # #    # ######  ####
    #  #      #      #    #  #  #  #      #
    #  #####  #      #    #   ##   #####   ####
    #  #      #      #    #   ##   #           #
    #  #      #      #    #  #  #  #      #    #
    #  #      ######  ####  #    # ######  ####
    #
    def Fmax(self, td, adiab=True):
        '''Flux in mili-janskys'''
        if adiab:
            # NOTE: Following Eq. (11)
            return 1.1e5 * (np.sqrt(self.eps_B)
                            * self.E52
                            * np.sqrt(self.n1)
                            / self.D28**2) * np.ones(len(td))
        else:
            # NOTE: Following Eq. (12)
            return 4.5e3 * (np.sqrt(self.eps_B)
                            * np.power(self.E52, 8.0 / 7.0)
                            * np.power(self.n1, 5.0 / 14.0)
                            / (np.power(td, 3.0 / 7.0) * self.D28**2))

    def fast_cooling(self, nu, nuc, num, Fnu_max):
        return np.piecewise(nu,
                            [nuc > nu,
                             (num >= nu) & (nu >= nuc),
                             nu > num],
                            [np.power(nu / nuc, 1.0 / 3.0) * Fnu_max,
                             Fnu_max / np.sqrt(nu / nuc),
                             np.power(nu / num, -0.5 * self.pind) * Fnu_max / np.sqrt(num / nuc)])

    def slow_cooling(self, nu, nuc, num, Fnu_max):
        return np.piecewise(nu,
                            [num > nu,
                             (nuc >= nu) & (nu >= num),
                                nu > nuc],
                            [np.power(nu / nuc, 1.0 / 3.0) * Fnu_max,
                                np.power(nu / nuc, 0.5 * (1.0 - self.pind)) * Fnu_max,
                                np.power(num / nuc, 0.5 * (1.0 - self.pind)) * np.power(nu / nuc, -0.5 * self.pind) * Fnu_max])

    def fluxSPN98(self, nu, t, adiab=True):
        # import magnetobrem as mbs
        # MBS = mbs.mbs()
        #
        # B = np.sqrt(32.0 * np.pi * mbs.mp * eps_B * n) * Gbulk * mbs.cLight
        # gm = eps_e * mbs.mp * Gbulk * (pind - 2.0) / (pind - 1.0) / mbs.me
        # Pnu_max = mbs.me * mbs.cLight**2 * mbs.sigmaT * Gbulk * B / (3.0 * mbs.eCharge)
        # gc = 6.0 * np.pi * mbs.me * mbs.cLight / (mbs.sigmaT * B**2 * Gbulk * t)
        # Ne = 4.0 * np.pi * R**3 * n / 3.0
        #
        # num = MBS.nu_g(B) * Gbulk * gm
        # nuc = MBS.nu_g(B) * Gbulk * gc
        # Fnu_max = Ne * Pnu_max / (4.0 * np.pi * dist**2)

        Nnu = len(nu)
        Nt = len(t)
        nu15 = nu * 1e-15
        td = t / 8.64e4
        flux = np.ndarray((Nt, Nnu))

        nuc = self.nuc(td, adiab=adiab)
        num = self.num(td, adiab=adiab)
        nu0 = self.nu0(adiab=adiab)
        tc = self.tc(nu15, adiab=adiab)
        tm = self.tm(nu15, adiab=adiab)
        t0 = self.t0(adiab=adiab)

        Fnu_max = self.Fmax(td, adiab=adiab)

        for i in range(Nt):
            for j in range(Nnu):
                if td[i] <= t0:
                    flux[i, j] = self.fast_cooling(nu[j], nuc[i], num[i], Fnu_max[i])
                else:
                    flux[i, j] = self.slow_cooling(nu[j], nuc[i], num[i], Fnu_max[i])

        return nuc, num, nu0, tc, tm, t0, flux
