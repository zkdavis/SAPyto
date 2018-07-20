import numpy as np
from scipy.special import hyp1f1, hyperu


def exp10(decimal):
    '''Function that gets the exponent of a double number. It only splits the
    string and returns the exponent as an integer
    '''
    parts = ("%e" % decimal).split('e')
    return int(parts[1])


def fortran_double(number):
    '''Function that returns a floating point (as a string) in the FORTRAN
    notation
    '''
    string = "{:.10e}".format(number)
    parts = string.split('e')
    ss = parts[0].split('.')
    ee = parts[-1]
    s = ss[1]

    while s.endswith('0'):
        s = s[:-1]
    if (len(s) > 0):
        ss = ss[0] + '.' + s
    else:
        ss = ss[0]

    if ee.startswith('-'):
        e = ee.split('-')[-1]
        while e.startswith('0'):
            e = e[1:]
        ee = '-' + e
    else:
        while ee.startswith('+') or ee.startswith('0'):
            ee = ee[1:]
    if (len(ee) == 0):
        ee = '0'

    return "{0}d{1}".format(ss, parts[1])


def sci_notation(n, prec=3, fortran=False):
    """Represent n in scientific notation, with the specified precision.

    >>> sci_notation(1234 * 10**1000)
    '1.234e+1003'
    >>> sci_notation(10**1000 // 2, prec=1)
    '5.0e+999'
    """
    exponent = int(np.log10(n))
    if (n < 1):
        exponent -= 1
    mantissa = n / np.power(10, exponent)
    if fortran:
        return '{0:.{1}f}d{2:+d}'.format(mantissa, prec, exponent)
    else:
        return '{0:.{1}f}e{2:+d}'.format(mantissa, prec, exponent)


def find_nearest(arr, val):
    arr = np.asarray(arr)
    i = (np.abs(arr - val)).argmin()
    return i, arr[i]


#
#  #     #
#  #  #  # # ##### #####   ##   #    # ###### #####
#  #  #  # #   #     #    #  #  #   #  #      #    #
#  #  #  # #   #     #   #    # ####   #####  #    #
#  #  #  # #   #     #   ###### #  #   #      #####
#  #  #  # #   #     #   #    # #   #  #      #   #
#   ## ##  #   #     #   #    # #    # ###### #    #
#


def whittM(k, m, z):
    """Evaluates the Whitaker function M(k, m, z) as defined in Abramowitz &
    Stegun, section 13.1.
    """
    if k is int or m is int:
        return np.exp(-0.5 * z) * np.power(z, 0.5 + float(m)) * hyp1f1(0.5 + float(m - k), 1 + 2 * m, z)
    elif k is int and m is int:
        return np.exp(-0.5 * z) * np.power(z, 0.5 + float(m)) * hyp1f1(0.5 + float(m - k), 1 + 2 * m, z)
    else:
        return np.exp(-0.5 * z) * np.power(z, 0.5 + m) * hyp1f1(0.5 + m - k, 1.0 + 2.0 * m, z)


def whittW(k, m, z):
    """Evaluates the Whitaker function W(k, m, z) as defined in Abramowitz &
    Stegun, section 13.1.
    """
    if k is int or m is int:
        return np.exp(-0.5 * z) * np.power(z, 0.5 + float(m)) * hyperu(0.5 + float(m - k), 1 + 2 * m, z)
    elif k is int and m is int:
        return np.exp(-0.5 * z) * np.power(z, 0.5 + float(m)) * hyperu(0.5 + float(m - k), 1 + 2 * m, z)
    else:
        return np.exp(-0.5 * z) * np.power(z, 0.5 + m) * hyperu(0.5 + m - k, 1.0 + 2.0 * m, z)

#
#  #####  #       ####  ##### ##### # #    #  ####
#  #    # #      #    #   #     #   # ##   # #    #
#  #    # #      #    #   #     #   # # #  # #
#  #####  #      #    #   #     #   # #  # # #  ###
#  #      #      #    #   #     #   # #   ## #    #
#  #      ######  ####    #     #   # #    #  ####
#
# def print_or_show(self, plt_name, do_print=False, fmt='pdf'):
#     '''This function will print the data
#     '''
#
#     if do_print:
#         fig.savefig(pname + '.' + fmt,
#                     format=fmt,
#                     dpi=300  # ,
#                     # rasterized=True,
#                     # transparent=True
#                     )
#     else:
#         fig.suptitle(pname)
#         fig.show()
