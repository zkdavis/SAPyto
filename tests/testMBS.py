#
# In[]: Preamble
import numpy as np
# import scipy.optimize as op
import SAPyto.magnetobrem as mbs
import matplotlib.pyplot as pl
pl.ion()

nu = np.logspace(5, 10, num=80)


#
# In[]: Fig. 3.15 in Rueda-Becerril (2017)
#
# This bubble reproduces Fig. 3.15 in my PhD thesis (Rueda-Becerril 2017)
MBS = mbs.mbs()

Xc = nu / MBS.nu_c_iso(1.0, 10.0)
pl.xscale('log')
pl.yscale('log')
pl.xlim(1e-4, 1e2)
pl.ylim(1e-3, 2.0)
pl.plot(Xc, MBS.Rsync(Xc), c='k', lw=4.0)
pl.plot(Xc, MBS.RMA(Xc, 10.0), c='b')
pl.plot(Xc, MBS.RMAfit(Xc, 10.0), c='r')
# pl.plot(Xc, MBS.FDB08fit(Xc), c='g')
pl.plot(Xc, MBS.Fsync(Xc))


#
# In[]: Testing spTable
#
# This bubble produces the rdiated power spectrum from a moving electron
spTable = mbs.spTable()


#
# In[]: Testing disTable
#
# This bubble produces the emissivity from a distribution of particles
spTable = mbs.spTable()
