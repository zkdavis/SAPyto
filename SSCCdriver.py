import os
# import numpy as np
# import SAPyto as sap
import extractor.fromHDF5 as extr


class SSCC_params(object):

    def params(self):
        self.R = 1e18
        self.dLum = 1e26
        self.z = 0.03
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
        self.wCool = 'T'
        self.wMBSabs = 'T'
        self.wSSCem = 'T'
        self.file_tail = 'DriverTest'
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
            print(self.R, file=f)
            print(self.dLum, file=f)
            print(self.z, file=f)
            print(self.gamma_bulk, file=f)
            print(self.theta_obs, file=f)
            print(self.B, file=f)
            print(self.theta_e, file=f)
            print(self.dtacc, file=f)
            print(self.g1, file=f)
            print(self.g2, file=f)
            print(self.numbins, file=f)
            print(self.numdt, file=f)
            print(self.numdf, file=f)
            print(self.file_tail, file=f)


class runSSCC(object):
    def __init__(self, **kwargs):
        self.par = SSCC_params(**kwargs)
        self.par.write_params()

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

        cwd = os.getcwd()
        os.chdir(self.par.ISdir)
        print("Running Makefile with:\n   ", make)
        os.system(make)
        os.chdir(cwd)
        print("\n--> Compilation successful\n")

    def run_NewSSCC(self, **kwargs):
        run_cmd = '{0}NewSSCC {1} {2} {3} {4}'.format(self.par.ISdir, self.par.params_file, self.par.wCool, self.par.wMBSabs, self.par.wSSCem)
        print("--> Running:\n  ", run_cmd)
        os.system(run_cmd)
        print("--> NewSSCC ran successfully")

    def cleanup(self, **kwargs):
        os.system("rm -r NewSSCC")


class outSSCC(object):
    # NOTE manipulate data
    def __init__(self):
        pass
