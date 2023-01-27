import numpy as np
import sys
import os
import yaml
from time import time

from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from cobaya.likelihood import Likelihood

import sys
sys.path.append('/global/home/users/nsailer/BOSS_Planck_DMDR/theory/xcorr/')
from predict_cl import AngularPowerSpectra


class GxKLikelihood(Likelihood):
    # From yaml file.
    basedir: str
    
    # optimize: turn this on when optimizng/running the minimizer so that the Jacobian factor isn't included
    # include_priors: this decides whether the marginalized parameter priors are included (should be yes)
    linear_param_dict_fn: str
    optimize: bool
    include_priors: bool
    
    clsfn:  str
    covfn:  str
    zeff:   list
    suffx:  list
    wlafn:  list
    wlxfn:  list
    amin:   list
    xmin:   list
    amax:   list
    xmax:   list
    dndzs:  list
    
    n_idm_dr: int
    #
    
    def initialize(self):
        """Sets up the class."""       
        # Load the data and invert the covariance matrix.
        self.loadData()
        self.cinv = np.linalg.inv(self.cov)
        
        # Load the linear parameters of the theory model theta_a such that
        # P_th = P_{th,nl} + theta_a P^a for some templates P^a we will compute
        self.linear_param_dict = yaml.load(open(self.basedir+self.linear_param_dict_fn), Loader=yaml.SafeLoader)
        self.linear_param_means = {key: self.linear_param_dict[key]['mean'] for key in self.linear_param_dict.keys()}
        self.linear_param_stds  = np.array([self.linear_param_dict[key]['std'] for key in self.linear_param_dict.keys()])
        self.Nlin = len(self.linear_param_dict) 
        
        self.prev_cosmo_pararms = None
        
    def get_requirements(self):
        """What we require."""
        reqs = {'Pk': None,\
                'A_s': None,\
                'n_s': None,\
                'h': None,\
                'omega_cdm': None,\
                'omega_b':None,\
                'tau_reio':None,\
                'a_dark':None,\
                'xi_idr':None,\
                }
               
        # Build the parameter names we require for each sample.
        for suf in self.suffx:
            for pref in ['b1','b2','bs']:
                reqs[pref+'_'+suf] = None
        return(reqs)
    
    def full_predict(self, thetas=None):
        '''
        Combine observe and predict.
        '''
        # for the first time use
        if self.prev_cosmo_pararms is None:
            self.prev_cosmo_pararms = self.get_params().copy()
            self.make_aps_makers()        

        # if the cosmology has changed, recompute tables
        # and update the cosmo parameters
        if np.any(self.prev_cosmo_pararms != self.get_params()):
            self.make_aps_makers()
            self.prev_cosmo_pararms = self.get_params().copy()

        obs = np.array([],dtype='float')
        
        for i,suf in enumerate(self.suffx):
            # Compute theory prediction
            thy = self.clgk_predict(i, suf,thetas=thetas)
            # then "observe" it, appending the observations to obs.
            obs = np.append(obs,self.observe(thy,self.wla[i],self.wlx[i]))

        return obs
    
    def logp(self,**params_values):
        """Return the log-likelihood."""

        thy_obs_0 = self.full_predict()
        self.Delta = self.dd - thy_obs_0
        
        # Now compute template
        self.templates = []
        for param in self.linear_param_dict.keys():
            thetas = self.linear_param_means.copy()
            thetas[param] += 1.0
            self.templates += [ self.full_predict(thetas=thetas) - thy_obs_0 ]
        
        self.templates = np.array(self.templates)
        
        # Make dot products
        self.Va = np.dot(np.dot(self.templates, self.cinv), self.Delta)
        self.Lab = np.dot(np.dot(self.templates, self.cinv), self.templates.T) + self.include_priors * np.diag(1./self.linear_param_stds**2)
        self.Lab_inv = np.linalg.inv(self.Lab)
        
        # Compute the modified chi2
        lnL  = -0.5 * np.dot(self.Delta,np.dot(self.cinv,self.Delta)) # this is the "bare" lnL
        lnL +=  0.5 * np.dot(self.Va, np.dot(self.Lab_inv, self.Va)) # improvement in chi2 due to changing linear params
        if not self.optimize:
            lnL += - 0.5 * np.log( np.linalg.det(self.Lab) ) + 0.5 * self.Nlin * np.log(2*np.pi) # volume factor from the determinant

        return lnL
    
    def get_best_fit(self):
        '''
        Generate best fits including linear templates.
        '''

        self.thy_nl  = self.dd - self.Delta
        self.bf_thetas = np.einsum('ij,j', np.linalg.inv(self.Lab), self.Va)
        self.thy_lin = np.einsum('i,il', self.bf_thetas, self.templates)
        self.thy = self.thy_nl + self.thy_lin
        return self.thy

    
        #
    def loadData(self):
        """Load the data, covariance and windows from files."""
        dd        = np.loadtxt(self.basedir+self.clsfn)
        self.cov  = np.loadtxt(self.basedir+self.covfn)
        self.wla = []
        for fn in self.wlafn:
            self.wla.append(np.loadtxt(self.basedir+fn))
        self.wlx = []
        for fn in self.wlxfn:
            self.wlx.append(np.loadtxt(self.basedir+fn))
        self.dndz = []
        for fn in self.dndzs:
            self.dndz.append(np.loadtxt(self.basedir+fn))
        # Now pack things and modify the covariance matrix to
        # "drop" some data points.
        Nsamp   = (dd.shape[1]-1)//2
        if Nsamp!=len(self.wla):
            raise RuntimeError("Inconsistent inputs: Nsamp mismatch.")
        if Nsamp!=len(self.wlx):
            raise RuntimeError("Inconsistent inputs: Nsamp mismatch.")
        if Nsamp!=len(self.amin):
            raise RuntimeError("Inconsistent inputs: Nsamp mismatch.")
        if Nsamp!=len(self.xmin):
            raise RuntimeError("Inconsistent inputs: Nsamp mismatch.")
        if Nsamp!=len(self.amax):
            raise RuntimeError("Inconsistent inputs: Nsamp mismatch.")
        if Nsamp!=len(self.xmax):
            raise RuntimeError("Inconsistent inputs: Nsamp mismatch.")
        self.xx = dd[:,0]
        self.dd = dd[:,1:].T.flatten()
        self.input_cov = self.cov.copy()
        for j in range(Nsamp):
            for i in np.nonzero(self.xx>self.amax[j])[0]:           # Auto
                ii = i + (2*j+0)*self.xx.size
                self.cov[ii, :] = 0
                self.cov[ :,ii] = 0
                self.cov[ii,ii] = 1e15
            for i in np.nonzero(self.xx>self.xmax[j])[0]:           # Cross
                ii = i + (2*j+1)*self.xx.size
                self.cov[ii, :] = 0
                self.cov[ :,ii] = 0
                self.cov[ii,ii] = 1e15
            for i in np.nonzero(self.xx<self.amin[j])[0]:           # Auto
                ii = i + (2*j+0)*self.xx.size
                self.cov[ii, :] = 0
                self.cov[ :,ii] = 0
                self.cov[ii,ii] = 1e15
            for i in np.nonzero(self.xx<self.xmin[j])[0]:           # Cross
                ii = i + (2*j+1)*self.xx.size
                self.cov[ii, :] = 0
                self.cov[ :,ii] = 0
                self.cov[ii,ii] = 1e15
        #
        
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

        return np.array([A_s, n_s, h, omega_b, omega_cdm, tau_reio, xi_idr, a_dark])    
    
    
    def make_aps_makers(self):
        """ use APS code to set up integrals"""
        pp   = self.provider
        pars = self.get_params()
        Pkemu = pp.get_result('Pk')
        ki,pi = np.array(Pkemu['k']), np.array(Pkemu['Pklin']) # should really be inputting sqrt(Pcb * Pmm)!!!
        
        self.aps_makers = []
        for i,suf in enumerate(self.suffx):
            # Compute theory prediction
            aps = AngularPowerSpectra(pars,self.dndz[i],self.zeff[i],n_idm_dr=self.n_idm_dr,ki=ki,pi=pi)
            self.aps_makers.append(aps)        
    
    
    def clgk_predict(self, i, suf, thetas=None):
        '''
        Predict Clgk for sample 'suf'.
        '''
        pp  = self.provider
        
        # Extract some common parameters.
        b1  = pp.get_param('b1_'+suf)
        b2  = pp.get_param('b2_'+suf)
        bs  = pp.get_param('bs_'+suf)
        
        # Instead of calling the linear parameters directly we will now analytically marginalize over them
        
        if thetas is None:
            alpX = self.linear_param_means['alpha_x_' + suf]
            smag = self.linear_param_means['smag_' + suf]
        else:
            alpX = thetas['alpha_x_' + suf]
            smag = thetas['smag_' + suf]
            
        bparsX= [b1,b2,bs,alpX]
        ##########################################################################################################################################################################################
        Pkemu = pp.get_result('Pk')
        ki,pi = np.array(Pkemu['k']), np.array(Pkemu['Pklin']) # should really be inputting sqrt(Pcb * Pmm)!!!
        khfi = ki   # should really be using the halofit Pk(z)!!! (recheck this!!!!)
        phfi = pi   # should really be using the halofit Pk(z)!!!
        ell,clgk = self.aps_makers[i](bparsX,smag=smag,khfi=khfi,phfi=phfi,Lmax=1251)
        clgg     = np.zeros_like(clgk)
        ##########################################################################################################################################################################################
                        
        thy = np.array([ell,clgg,clgk]).T
        
        return thy
    
    def observe(self,tt,wla,wlx):
        """Applies the window function and binning matrices."""
        lmax = wla.shape[1]
        ells = np.arange(lmax)
        # Have to stack auto and cross.
        obs1 = np.dot(wla,np.interp(ells,tt[:,0],tt[:,1],right=0))
        obs2 = np.dot(wlx,np.interp(ells,tt[:,0],tt[:,2],right=0))
        obs  = np.concatenate([obs1,obs2])
        return(obs)
        #
