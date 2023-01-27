# hard-coded paths (sorry)
PyCapse_path = '/global/home/users/nsailer/BOSS_Planck_DMDR/emulators/PyCapse/src/'
PyLimpse_path = '/global/home/users/nsailer/BOSS_Planck_DMDR/emulators/PyLimpse/src/'

# imports
import numpy as np
import sys
from cobaya.theory import Theory 
sys.path.append(PyCapse_path)
sys.path.append(PyLimpse_path)
import PyCapse
import PyLimpse

class classy_emu(Theory):
    """
    An emulator for CLASS, computes: Cltt, Clte, Clee, Clpp, (lin) Pk 
    """
    emu_Cl_path : str
    emu_Pk_path : str
    
    def initialize(self):
        """Sets up the class by loading and compiling the emulators."""
        print('Loading Cl emulator from',self.emu_Cl_path)
        self.Cltt, self.Clee, self.Clte, self.Clpp  = PyCapse.load_emu(self.emu_Cl_path)

        print('Loading Pk emulator from',self.emu_Pk_path)
        self.Pk = PyLimpse.load_emu(self.emu_Pk_path)

        print('Compiling the emulators')
        self.compile_emus()

    def get_requirements(self):
        req = {\
               'A_s': None,\
               'n_s': None,\
               'h': None,\
               'omega_b':None,\
               'omega_cdm':None,\
               'tau_reio':None,\
               'xi_idr':None,\
               'a_dark':None\
              }
        return(req)
    
    def get_can_provide(self):
        """What do we provide: Cls and the (linear) Pk"""
        return ['Cl','Pk']

    def compile_emus(self):
        # to compile, run on a dummy set of parameters
        params = np.array([2.847,1.01,0.727,0.0206,0.149,0.068,0.289,4.])
        PyCapse.compute_Cl(params, self.Cltt)
        print('Compiled Ctt')
        PyCapse.compute_Cl(params, self.Clte)
        print('Compiled Cte')
        PyCapse.compute_Cl(params, self.Clee)
        print('Compiled Cee')
        PyCapse.compute_Cl(params, self.Clpp)
        print('Compiled Cpp')
        PyLimpse.compute_Pk(params, self.Pk)
        print('Compiled Pk')

    def get_params(self):
        pp = self.provider

        A_s = pp.get_param('A_s')
        n_s = pp.get_param('n_s')
        h = pp.get_param('h')
        omega_b = pp.get_param('omega_b')
        omega_cdm = pp.get_param('omega_cdm')
        tau_reio = pp.get_param('tau_reio')
        xi_idr = pp.get_param('xi_idr')
        a_dark = pp.get_param('a_dark')

        return np.array([np.log(A_s*1e10), n_s, h, omega_b, omega_cdm, tau_reio, xi_idr, np.log10(a_dark)])

    def get_Cl(self,units=None):
        params = self.get_params()
        fac = (2.7255e6)**2

        ell = np.linspace(0,2508,2509)
        tt = np.zeros(ell.shape)
        te = np.zeros(ell.shape)
        tb = np.zeros(ell.shape)
        ee = np.zeros(ell.shape)
        eb = np.zeros(ell.shape)
        bb = np.zeros(ell.shape)
        pp = np.zeros(ell.shape)

        # all of the emulators start at ell = 2
        tt_emu = np.asarray(PyCapse.compute_Cl(params, self.Cltt)) * fac
        te_emu = np.asarray(PyCapse.compute_Cl(params, self.Clte)) * fac
        ee_emu = np.asarray(PyCapse.compute_Cl(params, self.Clee)) * fac
        pp_emu = np.asarray(PyCapse.compute_Cl(params, self.Clpp))

        tt[2:2+len(tt_emu)] = tt_emu
        te[2:2+len(te_emu)] = te_emu
        ee[2:2+len(ee_emu)] = ee_emu
        pp[2:2+len(pp_emu)] = pp_emu

        return {'ell':ell,'tt':tt,'te':te,'tb':tb,'ee':ee,'eb':eb,'bb':bb,'pp':pp}
        
    def get_Pk(self):    
        # DONT NEED OPTICAL DEPTH FOR PK
        params_tmp = self.get_params()
        params = np.array(list(params_tmp[:5]) + list(params_tmp[-2:]))
        k = self.Pk.kgrid
        Pk = PyLimpse.compute_Pk(params, self.Pk)
        return {'k':k, 'Pklin':Pk}

    def calculate(self, state, want_derived=True, **params_values_dict):
        """
        Evaluate the Cls and Pk
        """
        state['Cl'] = self.get_Cl()
        state['Pk'] = self.get_Pk()
