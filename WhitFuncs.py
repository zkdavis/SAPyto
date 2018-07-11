from scipy.special import hyp1f1,hyperu
import numpy as np

# Whittaker functions for real arguments

#### Whittaker M(k, m, z)
def whitm(k, m, z):
    """Evaluates the Whitaker function M(k, m, z) as defined in Abramowitz &
    Stegun, section 13.1.

    """
    if k is int or m is int:
        return np.exp(-0.5 * z) * np.power(z, 0.5 + float(m)) * hyp1f1(0.5 + float(m - k), 1 + 2 * m, z)
    elif k is int and m is int:
        return np.exp(-0.5 * z) * np.power(z, 0.5 + float(m)) * hyp1f1(0.5 + float(m - k), 1 + 2 * m, z)
    else:
        return np.exp(-0.5 * z) * np.power(z, 0.5 + m) * hyp1f1(0.5 + m - k, 1.0 + 2.0 * m, z)

#### Whittaker W(k, m, z)
def whitw(k, m, z):
    """Evaluates the Whitaker function W(k, m, z) as defined in Abramowitz &
    Stegun, section 13.1.

    """
    if k is int or m is int:
        return np.exp(-0.5 * z) * np.power(z, 0.5 + float(m)) * hyperu(0.5 + float(m - k), 1 + 2 * m, z)
    elif k is int and m is int:
        return np.exp(-0.5 * z) * np.power(z, 0.5 + float(m)) * hyperu(0.5 + float(m - k), 1 + 2 * m, z)
    else:
        return np.exp(-0.5 * z) * np.power(z, 0.5 + m) * hyperu(0.5 + m - k, 1.0 + 2.0 * m, z)
