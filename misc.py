# Function that gets the exponent of a double number :: It only splits the
# string and returns the exponent as an integer
def exp10(decimal):
    parts = ("%e" % decimal).split('e')
    return int(parts[1])

def sci_notation(n, prec=3):
    """
    Represent n in scientific notation, with the specified precision.

    >>> sci_notation(1234 * 10**1000)
    '1.234e+1003'
    >>> sci_notation(10**1000 // 2, prec=1)
    '5.0e+999'
    """
    from numpy import log10
    exponent = int(log10(n))
    if (n<1):
        exponent-=1
    mantissa = n / 10**exponent
    return '{0:.{1}f}e{2:+d}'.format(mantissa, prec, exponent)
