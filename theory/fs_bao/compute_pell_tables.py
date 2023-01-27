import numpy as np

from classy import Class
from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD

from time import time

speed_of_light = 2.99792458e5

# k vector to use:
kvec = np.concatenate( ([0.0005,],\
                        np.logspace(np.log10(0.0015),np.log10(0.025),10, endpoint=True),\
                        np.arange(0.03,0.51,0.01)) )


def ref_dist(z):
   # Reference Cosmology:
   Omega_M = 0.31
   fb = 0.1571
   h = 0.6766
   ns = 0.9665

   pkparams = {'A_s': np.exp(3.040)*1e-10,
               'n_s': 0.9665,
               'h': h,
               'N_ur': 3.046,
               'N_ncdm': 0,
               'tau_reio': 0.0568,
               'omega_b': h**2 * fb * Omega_M,
               'omega_cdm': h**2 * (1-fb) * Omega_M}

   pkclass = Class()
   pkclass.set(pkparams)
   pkclass.compute()

   Hz_fid = pkclass.Hubble(z) * speed_of_light / h 
   chiz_fid = pkclass.angular_distance(z) * (1.+z) * h 
   
   return Hz_fid,chiz_fid


def compute_pell_tables(pars, z=0.61, ki=None, pi=None, n_idm_dr=4, fid_dists=None, klin_max=2.):
    
    A_s, n_s, h, omega_b, omega_cdm, tau_reio, xi_idr, a_dark = pars
    
    if fid_dists is None:
      Hzfid, chizfid = ref_dist(z)
    else:
      Hzfid, chizfid = fid_dists

    params = {
        'A_s': A_s,
        'n_s': n_s,
        'h': h,
        'N_ur': 1.0196,
        'N_ncdm': 2,
        'm_ncdm': '0.01,0.05',
        'tau_reio': tau_reio,
        'omega_b': omega_b,
        'omega_cdm': omega_cdm,
        'a_idm_dr': a_dark,
        'xi_idr': xi_idr,
        'nindex_idm_dr': n_idm_dr,
        'f_idm': 1}
    
    # If no (emulated) pi given, then use CLASS
    # otherwise just use CLASS to get "background" quantities 
    if pi is None:
        pert_params = {'output': 'mPk',
                       'P_k_max_h/Mpc': klin_max,
                       'z_pk': '0,1'}
                       
        params = {**pert_params,**params}
   
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()
    
    # Calculate power spectrum
    if pi is None:
        ki = np.logspace(-3.0,np.log10(klin_max),200)
        pi = np.array( [cosmo.pk_cb(k*h, z ) * h**3 for k in ki] )
    else: 
        # rescale from z=0 -> z
        D = cosmo.scale_independent_growth_factor(z)
        pi *= D**2
        
    # Caluclate AP parameters
    Hz = cosmo.Hubble(z) * speed_of_light / h # this H(z) in units km/s/(Mpc/h) = 100 * E(z)
    chiz = cosmo.angular_distance(z) * (1.+z) * h # this is the comoving radius in units of Mpc/h 
    apar, aperp = Hzfid / Hz, chiz / chizfid
    
    # Calculate growth rate
    f   = cosmo.scale_independent_growth_factor_f(z)

    # Now do the RSD
    modPT = LPT_RSD(ki, pi, kIR=0.2,\
                cutoff=1, extrap_min = -4, extrap_max = 3, N = 2000, threads=1, jn=5)
    modPT.make_pltable(f, kv=kvec, apar=apar, aperp=aperp, ngauss=3)

    return kvec, modPT.p0ktable, modPT.p2ktable, modPT.p4ktable    
