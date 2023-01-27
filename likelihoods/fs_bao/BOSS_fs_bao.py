import numpy as np
import yaml
from time import time

from cobaya.likelihood import Likelihood
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from numpy.polynomial.polynomial import polyval

import sys
sys.path.append('/global/home/users/nsailer/BOSS_Planck_DMDR/theory/fs_bao/')
from compute_pell_tables  import compute_pell_tables
from compute_xiell_tables import compute_xiell_tables


# Class to have a full-shape likelihood for a bunch of pieces of data from both galactic caps in the same z bin
# Currently assumes all data have the same fiducial cosmology etc.
# If not I suggest chaning the theory class so that instead of being labelled by "zstr" it gets labelled by sample name.
# And each sample name indexing the fiducial cosmology numbers (chi, Hz etc) in dictionaries. For another time...

class JointLikelihood(Likelihood):
    
    zfid: float
    
    basedir: str
    
    fs_sample_names: list
    bao_sample_names: list
    
    # optimize: turn this on when optimizng/running the minimizer so that the Jacobian factor isn't included
    # include_priors: this decides whether the marginalized parameter priors are included (should be yes)
    linear_param_dict_fn: str
    optimize: bool
    include_priors: bool
    
    fs_datfns: list
    bao_datfns: list

    covfn: str
    
    fs_kmins: list
    fs_mmaxs: list
    fs_qmaxs: list
    fs_matMfns: list
    fs_matWfns: list
    
    bao_rmaxs: list
    bao_rmins: list
    
    n_idm_dr: int

    def initialize(self):
        """Sets up the class."""      
        # Redshift Label for theory classes
        self.zstr = "%.2f" %(self.zfid)

        # Load the linear parameters of the theory model theta_a such that
        # P_th = P_{th,nl} + theta_a P^a for some templates P^a we will compute
        self.linear_param_dict = yaml.load(open(self.basedir+self.linear_param_dict_fn), Loader=yaml.SafeLoader)
        self.linear_param_means = {key: self.linear_param_dict[key]['mean'] for key in self.linear_param_dict.keys()}
        self.linear_param_stds  = np.array([self.linear_param_dict[key]['std'] for key in self.linear_param_dict.keys()])
        self.Nlin = len(self.linear_param_dict) 
        
        # Binning matrix for correlation function, one for each BAO sample
        self.binmat = dict((name, None) for name in self.bao_sample_names)
        
        self.loadData()
        
        self.prev_cosmo_pararms = None
        #

    def get_requirements(self):
                
        req = {'Pk': None,\
               'A_s': None,\
               'n_s': None,\
               'h': None,\
               'omega_cdm': None,\
               'omega_b':None,\
               'tau_reio':None,\
               'a_dark':None,\
               'xi_idr':None,\
               }
        
        for fs_sample_name in self.fs_sample_names:
            req_bias = { \
                   'b1_' + fs_sample_name: None,\
                   'b2_' + fs_sample_name: None,\
                   'bs_' + fs_sample_name: None,\
                   }
            req = {**req, **req_bias}
        
        for bao_sample_name in self.bao_sample_names:
            req_bao = {\
                   'B1_' + bao_sample_name: None,\
                   'F_' +  bao_sample_name: None,\
                    }
            req = {**req, **req_bao}
            
        return(req)
    
    def full_predict(self, thetas=None):
    
        # for the first time use
        if self.prev_cosmo_pararms is None:
            self.prev_cosmo_pararms = self.get_params().copy()
            self.bao_tables()
            self.fs_tables()

        # if the cosmology has changed, recompute tables
        # and update the cosmo parameters
        if np.any(self.prev_cosmo_pararms != self.get_params()):
            self.fs_tables()
            self.bao_tables()
            self.prev_cosmo_pararms = self.get_params().copy()
    
        thy_obs = []

        if thetas is None:
            thetas = self.linear_param_means
        
        for fs_sample_name in self.fs_sample_names:
            fs_thy  = self.fs_predict(fs_sample_name,thetas=thetas)
            fs_obs  = self.fs_observe(fs_thy, fs_sample_name)
            thy_obs = np.concatenate( (thy_obs,fs_obs) )
        
        for bao_sample_name in self.bao_sample_names:
            bao_thy = self.bao_predict(bao_sample_name,thetas=thetas)
            bao_obs = self.bao_observe(bao_thy,bao_sample_name)
            thy_obs = np.concatenate( (thy_obs, bao_obs) )
            
        return thy_obs
    
    def logp(self,**params_values):
        """Return a log-likelihood."""
        # Compute the theory prediction with lin. params. at prior mean
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
        try:
            self.p0_nl  = self.dd - self.Delta
            self.bf_thetas = np.einsum('ij,j', np.linalg.inv(self.Lab), self.Va)
            self.p0_lin = np.einsum('i,il', self.bf_thetas, self.templates)
            return self.p0_nl + self.p0_lin
        except:
            print("Make sure to first compute the posterior.")
        
    def loadData(self):
        """
        Loads the required data.
        
        Do this in two steps... first load full shape data then xirecon, concatenate after.
        
        The covariance is assumed to already be joint in the concatenated format.
        
        """
        # First load the data
        
        self.kdats = {}
        self.p0dats = {}
        self.p2dats = {}
        self.fitiis = {}
        
        for ii, fs_datfn in enumerate(self.fs_datfns):
            fs_sample_name = self.fs_sample_names[ii]
            fs_dat = np.loadtxt(self.basedir+fs_datfn)
            self.kdats[fs_sample_name] = fs_dat[:,0]
            self.p0dats[fs_sample_name] = fs_dat[:,1]
            self.p2dats[fs_sample_name] = fs_dat[:,2]
            
            # Make a list of indices for the monopole and quadrupole only in Fourier space
            # This is specified to each sample in case the k's are different.
            yeses = self.kdats[fs_sample_name] > 0
            nos   = self.kdats[fs_sample_name] < 0
            self.fitiis[fs_sample_name] = np.concatenate( (yeses, nos, yeses, nos, nos ) )
        
        self.rdats = {}
        self.xi0dats = {}
        self.xi2dats = {}
        
        for ii, bao_datfn in enumerate(self.bao_datfns):
            bao_sample_name = self.bao_sample_names[ii]
            bao_dat = np.loadtxt(self.basedir+bao_datfn)
            self.rdats[bao_sample_name] = bao_dat[:,0]
            self.xi0dats[bao_sample_name] = bao_dat[:,1]
            self.xi2dats[bao_sample_name] = bao_dat[:,2]
        
        # Join the data vectors together
        self.dd = []
        
        for fs_sample_name in self.fs_sample_names:
            self.dd = np.concatenate( (self.dd, self.p0dats[fs_sample_name], self.p2dats[fs_sample_name]) )
            
        for bao_sample_name in self.bao_sample_names:
            self.dd = np.concatenate( (self.dd, self.xi0dats[bao_sample_name], self.xi2dats[bao_sample_name]) )
        
        # Now load the covariance matrix.
        cov = np.loadtxt(self.basedir+self.covfn)
        
        # We're only going to want some of the entries in computing chi^2.
        # this is going to tell us how many indices to skip to get to the nth multipole
        startii = 0
        
        for ss, fs_sample_name in enumerate(self.fs_sample_names):
            
            kcut = (self.kdats[fs_sample_name] > self.fs_mmaxs[ss])\
                          | (self.kdats[fs_sample_name] < self.fs_kmins[ss])
            
            for i in np.nonzero(kcut)[0]:     # FS Monopole.
                ii = i + startii
                cov[ii, :] = 0
                cov[ :,ii] = 0
                cov[ii,ii] = 1e25
            
            startii += self.kdats[fs_sample_name].size
            
            kcut = (self.kdats[fs_sample_name] > self.fs_qmaxs[ss])\
                       | (self.kdats[fs_sample_name] < self.fs_kmins[ss])
            
            for i in np.nonzero(kcut)[0]:       # FS Quadrupole.
                ii = i + startii
                cov[ii, :] = 0
                cov[ :,ii] = 0
                cov[ii,ii] = 1e25
            
            startii += self.kdats[fs_sample_name].size
        
        for ss, bao_sample_name in enumerate(self.bao_sample_names):
            
            rcut = (self.rdats[bao_sample_name] < self.bao_rmins[ss])\
                              | (self.rdats[bao_sample_name] > self.bao_rmaxs[ss])
            
            for i in np.nonzero(rcut)[0]:
                ii = i + startii
                cov[ii,:] = 0
                cov[:,ii] = 0
                cov[ii,ii] = 1e25
                
            startii += self.rdats[bao_sample_name].size
            
            for i in np.nonzero(rcut)[0]:
                ii = i + startii
                cov[ii,:] = 0
                cov[:,ii] = 0
                cov[ii,ii] = 1e25
            
            startii += self.rdats[bao_sample_name].size
        
        
        # Copy it and save the inverse.
        self.cov  = cov
        self.cinv = np.linalg.inv(self.cov)
        #print(self.sample_name, np.diag(self.cinv)[:10])
        
        # Finally load the window function matrix.
        self.matMs = {}
        self.matWs = {}
        for ii, fs_sample_name in enumerate(self.fs_sample_names):
            self.matMs[fs_sample_name] = np.loadtxt(self.basedir+self.fs_matMfns[ii])
            self.matWs[fs_sample_name] = np.loadtxt(self.basedir+self.fs_matWfns[ii])
        
        #    
        
    def combine_bias_terms_pkell(self,bvec, p0ktable, p2ktable, p4ktable):
        '''
        Same as function above but for the multipoles.
        
        Returns k, p0, p2, p4, assuming AP parameters from input p{ell}ktable
        '''
    
        b1,b2,bs,b3,alpha0,alpha2,alpha4,alpha6,sn,sn2,sn4 = bvec

        bias_monomials = np.array([1, b1, b1**2,\
                                   b2, b1*b2, b2**2, bs, b1*bs, b2*bs, bs**2, b3, b1*b3,\
                                   alpha0, alpha2, alpha4,alpha6,sn,sn2,sn4])

        p0 = np.sum(p0ktable * bias_monomials,axis=1)
        p2 = np.sum(p2ktable * bias_monomials,axis=1)
        p4 = np.sum(p4ktable * bias_monomials,axis=1)
        
        return p0, p2, p4
        
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
    
    
    def fs_tables(self):
        """ use velocileptors to get tables """
        pp   = self.provider
        Pkemu = pp.get_result('Pk')
        ki,pi = np.array(Pkemu['k']), np.array(Pkemu['Pklin'])
        pars = self.get_params()
        kv, p0ktable, p2ktable, p4ktable = compute_pell_tables(pars, z=self.zfid, ki=ki, pi=pi, n_idm_dr=self.n_idm_dr)
        self.kv = kv
        self.p0ktable = p0ktable
        self.p2ktable = p2ktable
        self.p4ktable = p4ktable
    
    
    def fs_predict(self, fs_sample_name, thetas=None):
        """Use the PT model to compute P_ell, given biases etc."""
        
        pp   = self.provider
        b1   = pp.get_param('b1_' + fs_sample_name) 
        b2   = pp.get_param('b2_' + fs_sample_name)
        bs   = pp.get_param('bs_' + fs_sample_name)
        
        # Instead of calling the linear parameters directly we will now analytically marginalize over them
        
        if thetas is None:
            alp0 = self.linear_param_means['alpha0_' + fs_sample_name]
            alp2 = self.linear_param_means['alpha2_' + fs_sample_name]
            sn0 = self.linear_param_means['SN0_' + fs_sample_name]
            sn2 = self.linear_param_means['SN2_' + fs_sample_name]
        else:
            alp0 = thetas['alpha0_' + fs_sample_name]
            alp2 = thetas['alpha2_' + fs_sample_name]
            sn0 = thetas['SN0_' + fs_sample_name]
            sn2 = thetas['SN2_' + fs_sample_name]
                    
        bias = [b1, b2, bs, 0.]
        cterm = [alp0,alp2,0,0]
        stoch = [sn0, sn2, 0]
        bvec = bias + cterm + stoch
        
        p0, p2, p4 = self.combine_bias_terms_pkell(bvec, self.p0ktable, self.p2ktable, self.p4ktable)
        
        # Put a point at k=0 to anchor the low-k part of the Spline.
        kv,p0 = np.append([0.,],self.kv),np.append([0.0,],p0)
        p2 = np.append([0.,],p2)
        p4 = np.append([0.0,],p4)
        tt = np.array([kv,p0,p2,p4]).T
       
        return(tt)
        
        
    def bao_tables(self):
        "use velocileptors to get xiell tables" 
        pp   = self.provider
        Pkemu = pp.get_result('Pk')
        ki,pi = np.array(Pkemu['k']), np.array(Pkemu['Pklin'])
        pars = self.get_params()
        rvec, xi0table, xi2table = compute_xiell_tables(pars, z=self.zfid, ki=ki, pi=pi, n_idm_dr=self.n_idm_dr)    
        self.rvec = rvec
        self.xi0table = xi0table
        self.xi2table = xi2table
        
    
    def bao_predict(self, bao_sample_name, thetas=None):
        
        pp   = self.provider
        
        B1  = pp.get_param('B1_' + bao_sample_name)
        F   = pp.get_param('F_' + bao_sample_name)
        
        # Analytically marginalize linear parmaeters so these are obtained differently
        if thetas is None:
            M0, M1 = [self.linear_param_means[param_name + '_' + bao_sample_name] for param_name in ['M0','M1',]]
            Q0, Q1 = [self.linear_param_means[param_name + '_' + bao_sample_name] for param_name in ['Q0','Q1',]]
        else:
            M0, M1, = [thetas[param_name + '_' + bao_sample_name] for param_name in ['M0','M1',]]
            Q0, Q1, = [thetas[param_name + '_' + bao_sample_name] for param_name in ['Q0','Q1',]]
        
        xi0t = self.xi0table[:,0] + B1*self.xi0table[:,1] + F*self.xi0table[:,2] \
             + B1**2 * self.xi0table[:,3] + F**2 * self.xi0table[:,4] + B1*F*self.xi0table[:,5]
        
        xi2t = self.xi2table[:,0] + B1*self.xi2table[:,1] + F*self.xi2table[:,2] \
             + B1**2 * self.xi2table[:,3] + F**2 * self.xi2table[:,4] + B1*F*self.xi2table[:,5]
        
        xi0t += polyval(1/self.rvec,[M0,M1])
        xi2t += polyval(1/self.rvec,[Q0,Q1])
        
        return np.array([self.rvec,xi0t,xi2t]).T
    

        
    def fs_observe(self,tt,fs_sample_name):
        """Apply the window function matrix to get the binned prediction."""
        
        # Have to stack ell=0, 2 & 4 in bins of 0.001h/Mpc from 0-0.4h/Mpc.
        kv  = np.linspace(0.0,0.4,400,endpoint=False) + 0.0005
        thy =                     Spline(tt[:,0],tt[:,1],ext=3)(kv)
        thy = np.concatenate([thy,Spline(tt[:,0],tt[:,2],ext=3)(kv)])
        thy = np.concatenate([thy,Spline(tt[:,0],tt[:,3],ext=3)(kv)])
        
        if np.any(np.isnan(thy)) or np.max(thy) > 1e8:
            print("NaN's encountered.")
        
        # wide angle
        expanded_model = np.matmul(self.matMs[fs_sample_name], thy )
        # Convolve with window (true) −> (conv) see eq. 2.18
        # Multiply by ad-hoc factor
        convolved_model = np.matmul(self.matWs[fs_sample_name], expanded_model )
        
        # keep only the monopole and quadrupole
        convolved_model = convolved_model[self.fitiis[fs_sample_name]]
    
        return convolved_model
    
    def bao_observe(self, tt, bao_sample_name, matrix=True):
        '''
        Bin the BAO results... probabaly should eventually use a matrix.
        '''
        
        rdat = self.rdats[bao_sample_name]
        
        if matrix:
            # If no binning matrix for this sample yet, make it.
            if self.binmat[bao_sample_name] is None:  
                
                dr = rdat[1] - rdat[0]
                
                rth = tt[:,0]
                Nvec = len(rth)

                bin_mat = np.zeros( (len(rdat), Nvec) )

                for ii in range(Nvec):
                    # Define basis vector
                    xivec = np.zeros_like(rth); xivec[ii] = 1
    
                    # Define the spline:
                    thy = Spline(rth, xivec, ext='const')
    
                    # Now compute binned basis vector:
                    tmp = np.zeros_like(rdat)
    
                    for i in range(rdat.size):
                        kl = rdat[i]-dr/2
                        kr = rdat[i]+dr/2

                        ss = np.linspace(kl, kr, 100)
                        p     = thy(ss)
                        tmp[i]= np.trapz(ss**2*p,x=ss)*3/(kr**3-kl**3)
        
                    bin_mat[:,ii] = tmp
                
                self.binmat[bao_sample_name] = np.array(bin_mat)
            
            tmp0 = np.dot(self.binmat[bao_sample_name], tt[:,1])
            tmp2 = np.dot(self.binmat[bao_sample_name], tt[:,2])
        
        else:
            thy0 = Spline(tt[:,0],tt[:,1],ext='extrapolate')
            thy2 = Spline(tt[:,0],tt[:,2],ext='extrapolate')
        
            dr   = rdat[1]- rdat[0]
        
            tmp0 = np.zeros_like(rdat)
            tmp2 = np.zeros_like(rdat)
        
            for i in range(rdat.size):
            
                kl = rdat[i]-dr/2
                kr = rdat[i]+dr/2

                ss = np.linspace(kl, kr, 100)
                p0     = thy0(ss)
                tmp0[i]= np.trapz(ss**2*p0,x=ss)*3/(kr**3-kl**3)
                p2     = thy2(ss)
                tmp2[i]= np.trapz(ss**2*p2,x=ss)*3/(kr**3-kl**3)
        
        return np.concatenate((tmp0,tmp2))
