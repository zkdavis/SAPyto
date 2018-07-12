import os
import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as col
import extractor as extr


class SAPyto:
    ''' ***  Spectral Analysis with Python toolkit  ***

    It will be assumed that the input flux is a NxM matrix, whose first entry is corresponds to the time and the second to the frequencies.
    '''
    #
    # NOTE Constants
    cLight = 2.99792458e10
    mp = 1.67262158e-24
    me = 9.10938188e-28
    eCharge = 4.803204e-10
    sigmaT = 6.6524586e-25
    hPlanck = 6.629069e-27
    nuconst = 2.7992491077281560779657886e6  # eCharge / 2 * pi * m_e * cLight

    def __init__(self):
        print('Debes bailar como baila el sapito, dando brinquitos.')

    #
    #  #####  #       ####  ##### ##### # #    #  ####
    #  #    # #      #    #   #     #   # ##   # #    #
    #  #    # #      #    #   #     #   # # #  # #
    #  #####  #      #    #   #     #   # #  # # #  ###
    #  #      #      #    #   #     #   # #   ## #    #
    #  #      ######  ####    #     #   # #    #  ####
    #
    def init_plot(self, style='bmh'):
        plt.style.use(style)

    def print_or_show(self, plt_name, do_print=False, fmt='pdf'):
        '''This function will print the data
        '''

        if do_print:
            fig.savefig(pname + '.' + fmt,
                        format=fmt,
                        dpi=300  # ,
                        # rasterized=True,
                        # transparent=True
                        )
        else:
            fig.suptitle(pname)
            fig.show()

    def convert2Jy(self, flux):
        '''Convert flux density (in egs cm^{-2} s^{-1} Hz^{-1}) janskys.
        '''
        return flux * 1e23

    #
    #  #      #  ####  #    # #####  ####  #    # #####  #    # ######  ####
    #  #      # #    # #    #   #   #    # #    # #    # #    # #      #
    #  #      # #      ######   #   #      #    # #    # #    # #####   ####
    #  #      # #  ### #    #   #   #      #    # #####  #    # #           #
    #  #      # #    # #    #   #   #    # #    # #   #   #  #  #      #    #
    #  ###### #  ####  #    #   #    ####   ####  #    #   ##   ######  ####
    #
    def nearLC(self, nu_in, nus, flux):
        '''This function returns the light curve of the frequency nearest to
        the one given (nu_in).
        '''
        i_nu, nu = find_nearest(nus, nu_in)
        return flux[:, i_nu]

    def bandLC(self, nu_min, nu_max, Nnu, nus, flux):
        try:
            len(nus)
        except TypeError:
            return 'nus must be an array'

        numn = max([nu_min, freqs[0]])
        numx = min([nu_max, freqs[-1]])

        if nu_min < freqs[0]:
            print('nu_min =', nu_min)
            print('minimum frequency in array=', freqs[0])
            return 'nu_min below frequencies array'
        if nu_max > freqs[-1]:
            print('nu_max =', nu_max)
            print('maximum frequency in array=', freqs[0])
            return 'nu_min below frequencies array'

        freqs_masked = ma.masked_outside(freqs, nu_min, nu_max)
        freqs_band = freqs_masked.compressed()
        nu_count = len(freqs_band)

        # for nu in freqs_band:

    #
    #   ####  #####  ######  ####  ##### #####    ##
    #  #      #    # #      #    #   #   #    #  #  #
    #   ####  #    # #####  #        #   #    # #    #
    #       # #####  #      #        #   #####  ######
    #  #    # #      #      #    #   #   #   #  #    #
    #   ####  #      ######  ####    #   #    # #    #
    #
    def spectrum(self, t_min):
