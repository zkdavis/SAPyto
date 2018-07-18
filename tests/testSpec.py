# In[]: Loading modules
# import numpy as np
import SAPyto.spectra as spec
import SAPyto.SRtoolkit as SR
import extractor.fromHDF5 as extr
import matplotlib.pyplot as plt


# In[]: Loading data
f = 'HSTOThick-wSSC_FinDif-Ng256Nt300Nf384.h5'
wdir = '/Users/jesus/lab/cSPEV/tests/SSCC_discret/Eq1p30RL79/'

Inu = extr.hdf5Extract2D(wdir + f, 'Inut')
t = extr.hdf5Extract1D(wdir + f, 'time')
numt = extr.hdf5ExtractScalar(wdir + f, 'numdt')
nu = extr.hdf5Extract1D(wdir + f, 'frequency')

# In[]: Building light curves
LC = spec.LightCurves()

D = SR.Doppler(10.0, 5.0)
Fnu = spec.flux_dens(Inu, 4.0793e26, 0.03, D, 1e18)
nu_obs = SR.nu_obs(nu, 0.03)
Fnu_lc = LC.integ(9e11, 2e12, numt, nu_obs, nu_obs * Fnu)
Fnu_lc_mono = LC.integ(1e12, 1e12, numt, nu_obs, nu_obs * Fnu)
Fnu_lc_near = LC.nearest(1e12, nu_obs, nu_obs * Fnu)
Fnu_lc_pwl = LC.pwl_interp(1e12, numt, nu_obs, nu_obs * Fnu)

# In[]: Plotting
fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$t\; \mathrm{[s]}$', fontsize='x-large')
ax.set_ylabel(r'$F_{\nu}\: \mathrm{[erg\; cm^{-2}\; s^{-1}]}$', fontsize='x-large')
ax.plot(t, Fnu_lc)
ax.plot(t, Fnu_lc_mono)
ax.plot(t, Fnu_lc_near)
ax.plot(t, Fnu_lc_pwl, ls='--')
# ax.set_ylim(1e5, 1e17)
# ax.set_xlim(numin, numax)
fig.tight_layout()
fig.show()
