import numpy as np
from scipy.special import hyp1f1, hyperu


def exp10(decimal):
    '''Function that gets the exponent of a double number. It only splits the
    string and returns the exponent as an integer
    '''
    parts = ("%e" % decimal).split('e')
    return int(parts[1])


def sci_notation(n, prec=3):
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
