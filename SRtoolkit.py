import numpy as np
from astropy import constants as const


def speed(gamma):
    '''v / c = sqrt(1 - 1 / gamma^2)
    '''
    return np.sqrt(1.0 - np.power(gamma, -2))


def speed2(gamma):
    '''(v / c)^2 = 1 - 1 / gamma^2
    '''
    return 1.0 - np.power(gamma, -2)


def LorentzFactor(beta):
    '''Calculating the Lorentz factor:
    gamma = 1 / sqrt(1 - beta^2)
    '''
    return 1.0 / np.sqrt(1.0 - beta**2)


def LorentzFactor2(beta):
    '''Calculating the Lorentz factor:
    gamma^2 = 1 / 1 - beta^2
    '''
    return 1.0 / (1.0 - beta**2)


def momentum(gamma):
    '''Calculating the momentum:
    gamma * beta = sqrt(gamma^2 - 1)
    '''
    return np.sqrt(gamma**2 - 1.0)


def momentum2(gamma):
    '''Calculating the momentum:
    (gamma * beta)^2 = gamma^2 - 1
    '''
    return gamma**2 - 1.0


def Doppler(gamma, view_angle):
    '''Doppler factor
    '''
    return 1.0 / (gamma * (1.0 + speed(gamma) * np.cos(view_angle)))


def nu_obs(nu, z, gamma, view_angle=0.0):
    '''Compute the observed frequency for a given redshift z.
    '''
    D = Doppler(gamma, view_angle)
    return nu * D / (1.0 + z)


def t_obs(t, z, gamma, x=0.0, view_angle=0.0):
    '''From Eq. (2.58) in my thesis.
    '''
    mu_obs = np.cos(view_angle)
    D = Doppler(gamma, view_angle)
    return (1.0 + z) * (t / D + gamma * x * (speed(gamma) - mu_obs) / const.c.cgs.value)
