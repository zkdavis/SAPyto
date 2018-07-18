
# In[]: Preamble
import numpy as np
# import scipy.optimize as op
import SAPyto.magnetobrem as mbs
import matplotlib.pyplot as pl
pl.ion()

# In[]: Fig. 3.15 in Rueda-Becerril (2017)
#
# This bubble reproduces Fig. 3.15 in my PhD thesis (Rueda-Becerril 2017)
MBS = mbs.mbs()

nu = np.logspace(5, 10, num=80)
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
