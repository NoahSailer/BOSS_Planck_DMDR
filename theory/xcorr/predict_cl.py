#!/usr/bin/env python3
#
# Code to compute angular power spectra using Limber's approximation,
# ignoring higher-order corrections such as curved sky or redshift-space
# distortions (that predominantly affect low ell).
#
import numpy as np

from scipy.integrate   import simps
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from classy import Class
from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD

speed_of_light = 2.99792458e5
kvec = np.concatenate( ([0.0005,],\
                        np.logspace(np.log10(0.0015),np.log10(0.025),10, endpoint=True),\
                        np.arange(0.03,0.51,0.01)) )


class AngularPowerSpectra():
    """Computes angular power spectra using the Limber approximation."""
    def lagrange_spam(self,z):
        """Returns the weights to apply to each z-slice to interpolate to z.
           -- Not currently used, could be used if need P(k,z)."""
        dz = self.zlist[:,None] - self.zlist[None,:]
        singular = (dz == 0)
        dz[singular] = 1.0
        fac = (z - self.zlist) / dz
        fac[singular] = 1.0
        return(fac.prod(axis=-1))
        
        
    def mag_bias_kernel(self,s,Nchi_mag=101):
        """Returns magnification bias kernel if 's' is the slope of
           the number counts dlog10N/dm."""
        zval    = self.zchi(self.chival)
        cmax    = np.max(self.chival) * 1.1
        zupper  = lambda x: np.linspace(x,cmax,Nchi_mag)
        chivalp = np.array(list(map(zupper,self.chival))).transpose()
        zvalp   = self.zchi(chivalp)
        dndz_n  = np.interp(zvalp,self.zz,self.dndz,left=0,right=0)
        Ez      = self.Eofz(zvalp)
        g       = (chivalp-self.chival[np.newaxis,:])/chivalp
        g      *= dndz_n*Ez/2997.925
        g       = self.chival * simps(g,x=chivalp,axis=0)
        mag_kern= 1.5*(self.OmM)/2997.925**2*(1+zval)*g*(5*s-2.)
        return(mag_kern)
        
        
    def shot3to2(self):
        """Returns the conversion from 3D shotnoise to 2D shotnoise power."""
        Cshot = self.fchi**2/self.chival**2
        Cshot = simps(Cshot,x=self.chival)
        return(Cshot)
        
    def __init__(self,pars,dndz,zeff,Nchi=201,Nz=251,n_idm_dr=4,ki=None,pi=None):
        """Set up the class.
            OmM:  The value of Omega_m(z=0) for the cosmology.
            chils:The (comoving) distance to last scattering (in Mpc/h).
            dndz: A numpy array (Nbin,2) containing dN/dz vs. z.
            zeff: The 'effective' redshift for computing P(k)."""
        self.n_idm_dr = n_idm_dr
        cosmo = self.get_cosmo(pars)    
        h = cosmo.pars['h']
        zls = cosmo.get_current_derived_parameters(['z_rec'])['z_rec']
        chils = cosmo.comoving_distance(zls)*h
        # Copy the arguments, setting up the z-range.
        self.Nchi = Nchi
        self.OmM = cosmo.get_current_derived_parameters(['Omega_m'])['Omega_m']
        self.zmin = np.min([0.05,dndz[0,0]])
        self.zmax = dndz[-1,0]
        self.zz   = np.linspace(self.zmin,self.zmax,Nz)
        self.dndz = Spline(dndz[:,0],dndz[:,1],ext=1)(self.zz)
        # Normalize dN/dz.
        self.dndz = self.dndz/simps(self.dndz,x=self.zz)
        # Make LCDM class and spline for E(z).
        zgrid     = np.logspace(0,3.1,128)-1.0
        EE = lambda z: cosmo.Hubble(z) * speed_of_light / h 
        self.Eofz = Spline(zgrid,[EE(zz) for zz in zgrid])
        # Set up the chi(z) array and z(chi) spline.
        self.chiz = np.array([cosmo.comoving_distance(z) for z in self.zz])*h
        self.zchi = Spline(self.chiz,self.zz)
        # Work out W(chi) for the objects whose dNdz is supplied.
        chimin    = np.min(self.chiz)
        chimax    = np.max(self.chiz)
        self.chival= np.linspace(chimin,chimax,Nchi)
        zval      = self.zchi(self.chival)
        self.fchi = Spline(self.zz,self.dndz*self.Eofz(self.zz))(zval)
        self.fchi/= simps(self.fchi,x=self.chival)
        # and W(chi) for the CMB
        self.chistar= chils
        self.fcmb = 1.5*self.OmM*(1.0/2997.925)**2*(1+zval)
        self.fcmb*= self.chival*(self.chistar-self.chival)/self.chistar
        # Set the effective redshift.
        self.zeff = zeff
        # and save linear growth.
        DD = lambda z: cosmo.scale_independent_growth_factor(z)
        self.ld2  = (np.array([DD(zz) for zz in zval])/DD(zeff))**2
        #
        self.preal_tables(pars,z=zeff,ki=ki,pi=pi)
  
        
    def __call__(self,bparsX,smag=0.4,Nell=64,Lmax=1001,khfi=None,phfi=None):
        """Computes C_l^{kg} given the emulator for P_{ij}, the
           cosmological parameters (cpars) plus bias params for cross (bparsX) 
           and the magnification slope
           (smag)."""
        # Set up arrays to hold kernels for C_l.
        ell    = np.logspace(1,np.log10(Lmax),Nell) # More ell's are cheap.
        Ckg = np.zeros( (Nell,self.Nchi) )
        # The magnification bias kernel.
        fmag   = self.mag_bias_kernel(smag)
        # Fit splines to our P(k).  The spline extrapolates as needed.
        kgm,pgm = self.get_spectrum(bparsX)
        Pgm    = Spline(kgm,pgm)
        Pmm    = Spline(khfi,phfi,ext=1) # Extrap. w/ zeros.
        ######################################################################################
        # I'm pretty sure that Pmm is wrong, I'll ask around...
        ######################################################################################
        # Work out the integrands for C_l^{gg} and C_l^{kg}.
        for i,chi in enumerate(self.chival):
            kval     = (ell+0.5)/chi        # The vector of k's.            
            f1f2     = self.fchi[i]*self.fcmb[i]/chi**2 * Pgm(kval)
            m1f2     =      fmag[i]*self.fcmb[i]/chi**2 * Pmm(kval)*self.ld2[i]
            Ckg[:,i] = f1f2 + m1f2
        # and then just integrate them.
        Ckg = simps(Ckg,x=self.chival,axis=-1)
        # Now interpolate onto a regular ell grid.
        lval= np.arange(Lmax)
        Ckg = Spline(ell,Ckg)(lval)
        return lval,Ckg
     
        
    def get_cosmo(self,pars):
        A_s, n_s, h, omega_b, omega_cdm, tau_reio, xi_idr, a_dark = pars
    
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
            'nindex_idm_dr': self.n_idm_dr,
            'f_idm': 1}

        cosmo = Class()
        cosmo.set(params)
        cosmo.compute()
        return cosmo
    
    def preal_tables(self,pars,z=0.61,ki=None,pi=None,klin_max=2.):
        # use velocileptors to get the real-space P(k)
        A_s, n_s, h, omega_b, omega_cdm, tau_reio, xi_idr, a_dark = pars
    
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
            'nindex_idm_dr': self.n_idm_dr,
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
        
        # Now do the RSD
        f = cosmo.scale_independent_growth_factor_f(z)
        modPT = LPT_RSD(ki, pi, kIR=0.2, jn=5,cutoff=1, extrap_min = -4, extrap_max = 3, N = 2000, threads=1)
        modPT.make_ptable(f, 0, kv=kvec)
        self.ptab = modPT.pktables[0]      
     
               
    def get_spectrum(self,bpars):   
        b1,b2,bs,alpha = bpars
        b3,sn = 0,0
        bias_monomials = np.array([1, 0.5*b1, 0,\
                               0.5*b2, 0, 0,\
                               0.5*bs, 0, 0, 0,\
                               0.5*b3, 0])  
        za   = self.ptab[:,-1]
        # the first row is kv, last row is za for countrterm
        res = np.sum(self.ptab[:,1:-1] * bias_monomials,axis=1)\
              + alpha * kvec**2 * za + sn
        return kvec,res
