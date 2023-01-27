import numpy as np
from classy import Class
import matplotlib.pyplot as plt

bd = '/global/home/users/nsailer/BOSS_Planck_DMDR/emulators/'

import sys
sys.path.append(bd + 'PyLimpse/src/')
sys.path.append(bd + 'PyCapse/src')
import PyLimpse
import PyCapse

pars = [2.10732e-9, 0.96824, 0.677, 0.02247, 0.11923, 0.1, 1e9]

input_test = np.array(pars)
input_test[0] = np.log(input_test[0]*1e10)
input_test[-1] = np.log10(input_test[-1])
#input_test = np.array([6.30, 3.46, 0.76, 0.021, 0.11, 0.3, 6.0])
print(input_test)

Pk_emu = PyLimpse.load_emu(bd + 'emulator_Pk_cb_n4_hubble_v3.bson')

k = np.array(Pk_emu.kgrid)
Pk = np.array(PyLimpse.compute_Pk(input_test, Pk_emu))



A_s, n_s, h, omega_b, omega_cdm, xi_idr, a_dark = pars
tau_reio = 0.06
'''
pkparams = {
        'output': 'tCl pCl lCl mPk',
        'l_max_scalars': 3000,
        'P_k_max_h/Mpc': 20.,
        'lensing': 'yes',
        'A_s': A_s,
        'n_s': n_s,
        'h': h,
        'N_ur': 2.0328,
        'N_ncdm': 1,
        'm_ncdm': '0.06',
        'tau_reio': tau_reio,
        'omega_b': omega_b,
        'omega_cdm': omega_cdm,
        'a_idm_dr': a_dark,
        'xi_idr': xi_idr,
        'nindex_idm_dr': 4,
        'f_idm': 1}
'''

pkparams = {
    "A_s": A_s,
    "n_s"  : n_s,
    "h"  : h,
    "omega_b"  : omega_b,
    "omega_cdm"  : omega_cdm,
    'tau_reio': tau_reio,
    'f_idm': 1.0,
    'xi_idr' : xi_idr,
    'm_idm': 1e9,
    'stat_f_idr': 0.875,
    'a_idm_dr': a_dark,
    'nindex_idm_dr': 4,
    'idr_nature': 'free_streaming',
    'N_ur': 2.0328,
    'N_ncdm' :  1,
    'm_ncdm' : 0.06,
    'output':'tCl,pCl,lCl,mPk',
    'lensing':'yes',
    'P_k_max_h/Mpc': 20.}

pkclass = Class()
pkclass.set(pkparams)
pkclass.compute()

pclass = np.array( [pkclass.pk_cb_lin(kk*h, 0 ) * h**3 for kk in k] )




for i in range(len(pclass)): print(k[i],np.array(Pk)[i]/pclass[i])