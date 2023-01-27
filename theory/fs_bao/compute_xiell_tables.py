import sys
sys.path.append('/global/home/users/nsailer/BOSS_Planck_DMDR/theory/fs_bao/')

import numpy as np

from classy import Class
from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD

from scipy.special import spherical_jn
from scipy.integrate import simps
from scipy.interpolate import interp1d

from velocileptors.Utils.spherical_bessel_transform import SphericalBesselTransform as SBT

from pnw_dst import pnw_dst
from compute_pell_tables import kvec, ref_dist, speed_of_light


kint = np.logspace(-3, 2, 2000)
sphr = SBT(kint,L=5,fourier=True,low_ring=False)

def compute_bao_pkmu(mu_obs, B1, F, klin, plin, pnw, f, apar, aperp, R, sigmas):
    '''
    Helper function to get P(k,mu) post-recon in RecIso.
        
    This is turned into Pkell and then Hankel transformed in the bao_predict funciton.
    '''

        
    sigmadd, sigmass, sigmads_dd, sigmads_ss, sigmads_ds = sigmas
    pw = plin - pnw

    Sk = np.exp(-0.5*(klin*R)**2)
        
    # Our philosophy here is to take kobs = klin
    # Then we predict P(ktrue) on the klin grid, so that the final answer is
    # Pobs(kobs,mu_obs) ~ Ptrue(ktrue, mu_true) = interp( klin, Ptrue(klin, mu_true) )(ktrue)
    # (Up to a normalization that we drop.)
    # Interpolating this way avoids having to interpolate Pw and Pnw separately.
        
    F_AP = apar/aperp
    AP_fac = np.sqrt(1 + mu_obs**2 *(1./F_AP**2 - 1) )
    mu = mu_obs / F_AP / AP_fac
    ktrue = klin/aperp*AP_fac
        
    # First construct P_{dd,ss,ds} individually
    dampfac_dd = np.exp( -0.5 * klin**2 * sigmadd * (1 + f*(2+f)*mu**2) )
    pdd = ( (1 + F*mu**2)*(1-Sk) + B1 )**2 * (dampfac_dd * pw + pnw)
        
    # then Pss
    dampfac_ss = np.exp( -0.5 * klin**2 * sigmass )
    pss = Sk**2 * (dampfac_ss * pw + pnw)
        
    # Finally Pds
    dampfac_ds = np.exp(-0.5 * klin**2 * ( 0.5*sigmads_dd*(1+f*(2+f)*mu**2)\
                                             + 0.5*sigmads_ss \
                                             + (1+f*mu**2)*sigmads_ds) )
    linfac = - Sk * ( (1+F*mu**2)*(1-Sk) + B1 )
    pds = linfac * (dampfac_ds * pw + pnw)
        
    # Sum it all up and interpolate?
    ptrue = pdd + pss - 2*pds
    pmodel = interp1d(klin, ptrue, kind='cubic', fill_value=0,bounds_error=False)(ktrue)
    
    return pmodel

def compute_xiells(rout, B1, F, klin, plin, pnw, f, apar, aperp, R, sigmas):
        

    # Generate the sampling
    ngauss = 4
    nus, ws = np.polynomial.legendre.leggauss(2*ngauss)
    nus_calc = nus[0:ngauss]
        
    L0 = np.polynomial.legendre.Legendre((1))(nus)
    L2 = np.polynomial.legendre.Legendre((0,0,1))(nus)
    
    pknutable = np.zeros((len(nus),len(klin)))
    
    for ii, nu in enumerate(nus_calc):
        pknutable[ii,:] = compute_bao_pkmu(nu, B1, F, klin, plin, pnw, f, apar, aperp, R, sigmas)
 
    pknutable[ngauss:,:] = np.flip(pknutable[0:ngauss],axis=0)
        
    p0 = 0.5 * np.sum((ws*L0)[:,None]*pknutable,axis=0) #+ 1000 * polyval(klin,[m0,m1,m2,m3,m4,m5]) / klin
    p2 = 2.5 * np.sum((ws*L2)[:,None]*pknutable,axis=0) #+ 1000 * polyval(klin,[q0,q1,q2,q3,q4,q5]) / klin

    p0t = interp1d(klin,p0, kind='cubic', bounds_error=False, fill_value=0)(kint)
    p2t = interp1d(klin,p2, kind='cubic', bounds_error=False, fill_value=0)(kint)

    damping = np.exp(-(kint/10)**2)
    rr0, xi0t = sphr.sph(0,p0t * damping)
    rr2, xi2t = sphr.sph(2,p2t * damping); xi2t *= -1

    return interp1d(rr0,xi0t,kind='cubic')(rout), interp1d(rr0,xi2t,kind='cubic')(rout)


def compute_xiell_tables(pars, z=0.61, fid_dists=None, ki=None, pi=None, n_idm_dr=4, R=15., rmin=50, rmax=160, dr=0.1, klin_max=2.):
    
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
    
    # If no emulator given for Pk, then use CLASS
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

    # Do the Zeldovich reconstruction predictions
    knw, pnw = pnw_dst(ki, pi)
    pw = pi - pnw
            
    qbao   = cosmo.rs_drag() * h # want this in Mpc/h units

    j0 = spherical_jn(0,ki*qbao)
    Sk = np.exp(-0.5*(ki*15)**2)

    sigmadd = simps( 2./3 * pi * (1-Sk)**2 * (1-j0), x = ki) / (2*np.pi**2)
    sigmass = simps( 2./3 * pi * (-Sk)**2 * (1-j0), x = ki) / (2*np.pi**2)

    sigmads_dd = simps( 2./3 * pi * (1-Sk)**2, x = ki) / (2*np.pi**2)
    sigmads_ss = simps( 2./3 * pi * (-Sk)**2, x = ki) / (2*np.pi**2)
    sigmads_ds = -simps( 2./3 * pi * (1-Sk)*(-Sk)*j0, x = ki) / (2*np.pi**2) # this minus sign is because we subtract the cross term
    
    sigmas = (sigmadd, sigmass, sigmads_dd, sigmads_ss, sigmads_ds)
    
    # Now make the multipoles!
    klin, plin = ki, pi
    routs = np.arange(rmin, rmax, dr)
    
    # this is 1
    xi0_00,xi2_00  = compute_xiells(routs, 0, 0, klin, plin, pnw, f, apar, aperp, R, sigmas )

    # this is 1 + B1 + B1^2
    # and 1 + 2 B1 + 4 B1^2
    xi0_10,xi2_10  = compute_xiells(routs,1, 0, klin, plin, pnw, f, apar, aperp, R, sigmas )
    xi0_20,xi2_20  = compute_xiells(routs,2, 0, klin, plin, pnw, f, apar, aperp, R, sigmas )
    
    # this is 1 + F + F^2
    # and 1 + 2 F + 4 F^2
    xi0_01,xi2_01  = compute_xiells(routs,0, 1, klin, plin, pnw, f, apar, aperp, R, sigmas )
    xi0_02,xi2_02  = compute_xiells(routs,0, 2, klin, plin, pnw, f, apar, aperp, R, sigmas )
    
    # and 1 + B1 + F + B1^2 + F^2 + BF
    xi0_11,xi2_11 = compute_xiells(routs,1, 1, klin, plin, pnw, f, apar, aperp, R, sigmas )
    
    xi0table, xi2table = np.zeros( (len(routs),6) ), np.zeros( (len(routs),6) )
    
    # Form combinations:
    xi0_B1 = 0.5 * (4 * xi0_10 - xi0_20 - 3*xi0_00)
    xi0_B1sq = xi0_10 - xi0_B1 - xi0_00

    xi0_F = 0.5 * (4 * xi0_01 - xi0_02 - 3*xi0_00)
    xi0_Fsq = xi0_01 - xi0_F - xi0_00

    xi0_BF = xi0_11 - xi0_B1 - xi0_F - xi0_B1sq - xi0_Fsq - xi0_00
    
    xi2_B1 = 0.5 * (4 * xi2_10 - xi2_20 - 3*xi2_00)
    xi2_B1sq = xi2_10 - xi2_B1 - xi2_00

    xi2_F = 0.5 * (4 * xi2_01 - xi2_02 - 3*xi2_00)
    xi2_Fsq = xi2_01 - xi2_F - xi2_00

    xi2_BF = xi2_11 - xi2_B1 - xi2_F - xi2_B1sq - xi2_Fsq - xi2_00
    
    # Load
    xi0table[:,0] = xi0_00
    
    xi0table[:,1] = xi0_B1
    xi0table[:,2] = xi0_F

    xi0table[:,3] = xi0_B1sq
    xi0table[:,4]= xi0_Fsq

    xi0table[:,5] = xi0_BF
    
    xi2table[:,0] = xi2_00
    
    xi2table[:,1] = xi2_B1
    xi2table[:,2] = xi2_F

    xi2table[:,3] = xi2_B1sq
    xi2table[:,4]= xi2_Fsq

    xi2table[:,5] = xi2_BF

    return routs, xi0table, xi2table
    
