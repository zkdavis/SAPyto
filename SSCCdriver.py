import os
import SAPyto.spectra as spec
from SAPyto.misc import fortran_double
import SAPyto.SRtoolkit as SR
import extractor.fromHDF5 as extr


#  #####    ##   #####    ##   #    #  ####
#  #    #  #  #  #    #  #  #  ##  ## #
#  #    # #    # #    # #    # # ## #  ####
#  #####  ###### #####  ###### #    #      #
#  #      #    # #   #  #    # #    # #    #
#  #      #    # #    # #    # #    #  ####


class SSCC_params(object):

    def params(self):
        self.R = 1e18
        self.dLum = 4.0793e26     # Luminosity distance of Mrk 421
        self.z = 0.03             # Redshift of Mrk 421
        self.gamma_bulk = 10.0
        self.theta_obs = 5.0
        self.B = 1.0
        self.theta_e = 10.0
        self.dtacc = 1e2
        self.g1 = 1e2
        self.g2 = 1e4
        self.numbins = 128
        self.numdt = 300
        self.numdf = 256
        self.wCool = True
        self.wMBSabs = True
        self.wSSCem = True
        self.file_label = 'DriverTest'
        self.ISdir = './'
        self.params_file = 'input.par'
        self.HYB = True
        self.MBS = True
        self.arc = 'i7'
        self.OMP = False
        self.DBG = False

    def __init__(self, **kwargs):
        self.params()
        self.__dict__.update(kwargs)

    def write_params(self):
        with open(self.params_file, 'w') as f:
            print(fortran_double(self.R), file=f)
            print(fortran_double(self.dLum), file=f)
            print(fortran_double(self.z), file=f)
            print(fortran_double(self.gamma_bulk), file=f)
            print(fortran_double(self.theta_obs), file=f)
            print(fortran_double(self.B), file=f)
            print(fortran_double(self.theta_e), file=f)
            print(fortran_double(self.dtacc), file=f)
            print(fortran_double(self.g1), file=f)
            print(fortran_double(self.g2), file=f)
            print(self.numbins, file=f)
            print(self.numdt, file=f)
            print(self.numdf, file=f)
            print(self.file_label, file=f)

    def output_file(self):
        outf = ''
        argv = ''
        if self.HYB:
            outf += 'H'
        else:
            outf += 'P'

        if self.MBS:
            outf += 'M'
        else:
            outf += 'S'

        if self.wCool:
            outf += 'V'
            argv += ' T'
        else:
            outf += 'C'
            argv += ' F'

        if self.wMBSabs:
            outf += 'O'
            argv += ' T'
        else:
            outf += 'T'
            argv += ' F'

        if self.wSSCem:
            outf += 'wSSC'
            argv += ' T'
        else:
            outf += 'oSSC'
            argv += ' F'

        return outf + '-' + self.file_label + '.h5', argv


#  #####  #    # #    #
#  #    # #    # ##   #
#  #    # #    # # #  #
#  #####  #    # #  # #
#  #   #  #    # #   ##
#  #    #  ####  #    #


class runSSCC(object):
    def __init__(self, **kwargs):
        self.par = SSCC_params(**kwargs)
        self.par.write_params()
        self.outfile, self.argv = self.par.output_file()
        self.cwd = os.getcwd()

    def compile(self, **kwargs):
        make = 'make NewSSCC -j4'
        if self.par.arc in ['i7', 'corei7', 'I7', 'COREI7']:
            make += ' COREI7=1'
        else:
            make += ' NATIVE=1'

        if self.par.OMP:
            make += ' OPENMP=1'

        if self.par.DBG:
            make += ' DBG=1'

        if self.par.HYB:
            make += ' HYB=1'

        if self.par.MBS:
            make += ' MBS=1'

        os.chdir(self.par.ISdir)
        print("--> Running Makefile:\n   ", make, "\n")
        os.system(make)
        os.chdir(self.cwd)
        print("\n--> Compilation successful\n")

    def runNewSSCC(self, **kwargs):
        run_cmd = '{0}/NewSSCC {1}{2}'.format(self.par.ISdir, self.par.params_file, self.argv)
        print("\n--> Running:\n  ", run_cmd, "\n")
        os.system(run_cmd)
        print("\n--> NewSSCC ran successfully")

    def cleanup(self, **kwargs):
        os.chdir(self.par.ISdir)
        os.system("make clean")
        os.chdir(self.cwd)


# #   ####  #    # ##### #####  #    # #####
# #  #    # #    #   #   #    # #    #   #
# #  #    # #    #   #   #    # #    #   #
# #  #    # #    #   #   #####  #    #   #
# #  #    # #    #   #   #      #    #   #
# #   ####   ####    #   #       ####    #


def build_LCs(nu_min, nu_max, dset='Inut', only_load=True, inJanskys=False, **kwargs):
    run = runSSCC(**kwargs)
    if not only_load:
        run.cleanup()
        run.compile()
        run.runNewSSCC()
    D = SR.Doppler(run.par.gamma_bulk, run.par.theta_obs)
    nu = extr.hdf5Extract1D(run.outfile, 'frequency')
    t = extr.hdf5Extract1D(run.outfile, 'time')
    nu_obs = SR.nu_obs(nu, run.par.z, run.par.gamma_bulk, run.par.theta_obs)
    t_obs = SR.t_obs(t, run.par.z, run.par.gamma_bulk, view_angle=run.par.theta_obs)
    Inu = extr.hdf5Extract2D(run.outfile, dset)
    Fnu = spec.flux_dens(Inu, run.par.dLum, run.par.z, D, run.par.R)
    LC = spec.LightCurves()

    if inJanskys:
        return t_obs, spec.conv2Jy(LC.integ(nu_min, nu_max, run.par.numdt, nu_obs, Fnu))
    else:
        return t_obs, LC.integ(nu_min, nu_max, run.par.numdt, nu_obs, Fnu)


def build_avSpec(t_min, t_max, dset='Inut', only_load=True, inJanskys=False, **kwargs):
    run = runSSCC(**kwargs)
    if not only_load:
        run.cleanup()
        run.compile()
        run.runNewSSCC()
    D = SR.Doppler(run.par.gamma_bulk, run.par.theta_obs)
    nu = extr.hdf5Extract1D(run.outfile, 'frequency')
    t = extr.hdf5Extract1D(run.outfile, 'time')
    nu_obs = SR.nu_obs(nu, run.par.z, run.par.gamma_bulk, run.par.theta_obs)
    t_obs = SR.t_obs(t, run.par.z, run.par.gamma_bulk, view_angle=run.par.theta_obs)
    Inu = extr.hdf5Extract2D(run.outfile, dset)
    Fnu = spec.flux_dens(Inu, run.par.dLum, run.par.z, D, run.par.R)
    sp = spec.spectrum()

    if inJanskys:
        return nu_obs, spec.conv2Jy(sp.averaged(t_min, t_max, run.par.numdf, t_obs, Fnu))
    else:
        return nu_obs, sp.averaged(t_min, t_max, run.par.numdf, t_obs, Fnu)


def build_SEDs(dset='Inut', only_load=True, inJanskys=False, **kwargs):
    run = runSSCC(**kwargs)
    if not only_load:
        run.cleanup()
        run.compile()
        run.runNewSSCC()
    D = SR.Doppler(run.par.gamma_bulk, run.par.theta_obs)
    nu = extr.hdf5Extract1D(run.outfile, 'frequency')
    t = extr.hdf5Extract1D(run.outfile, 'time')
    nu_obs = SR.nu_obs(nu, run.par.z, run.par.gamma_bulk, run.par.theta_obs)
    t_obs = SR.t_obs(t, run.par.z, run.par.gamma_bulk, view_angle=run.par.theta_obs)
    Inu = extr.hdf5Extract2D(run.outfile, dset)
    Fnu = spec.flux_dens(Inu, run.par.dLum, run.par.z, D, run.par.R)
    sp = spec.spectrum()

    if inJanskys:
        return nu_obs, spec.conv2Jy(sp.averaged(t_obs[0], t_obs[-1], run.par.numdf, t_obs, Fnu))
    else:
        return nu_obs, sp.averaged(t_obs[0], t_obs[-1], run.par.numdf, t_obs, Fnu)
