from astropy import constants as const
import numpy as np

halfpi = 0.5 * np.pi
twopi = 2.0 * np.pi
lzero = np.log(1e-200)
# Physical constants in CGS
cLight = const.c.cgs.value
eCharge = const.e.gauss.value
hPlanck = const.h.cgs.value
me = const.m_e.cgs.value
mp = const.m_p.cgs.value
sigmaT = const.sigma_T.cgs.value
nuConst = eCharge / (twopi * me * cLight)
jmbConst = twopi * eCharge**2 / cLight
ambConst = np.pi * eCharge**2 / (me * cLight)
chunche_c100g100 = 2.2619939050180366385e-6
chunche_c100g20 = 2.1157699720918349273e-1
