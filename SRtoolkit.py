import numpy as np


def beta_vel(gamma):
    '''v / c = sqrt(1 - 1 / gamma^2)
    '''
    return np.sqrt(1.0 - np.power(gamma, -2))


def gamma_lorentz(beta):
    '''Calculating the Lorentz factor:
    gamma = 1 / sqrt(1 - beta^2)
    '''
    return 1.0 / np.sqrt(1.0 - beta**2)


def beta_gamma(gamma):
    '''Calculating the momentum:
    gamma * beta = sqrt(gamma^2 - 1)
    '''
    return np.sqrt(gamma**2 - 1.0)
