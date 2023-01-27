# Angular correlation function files.

The angular (cross) correlation functions and their covariances and redshift distributions.
Each sample is described by two digits.  If the first digit is odd it represent the NGC
sample, while if it is even it represents the SGC.  The second digit is the redshift slice,
with "1" representing "z1" with z=0.38 and "3" representing "z3" with z=0.59.  Thus
gal_s11_cls.txt contains the angular power spectra for the NGC z1 sample while
gal_s11_cov.txt contains the corresponding covariance matrix (details given in the headers).

For the first digit, the samples 1 and 2 do not include FKP weights, 3 and 4 contain FKP
weights while 5 and 6 include the FKP weights and the ratio of the lensing kernel and dN/dz.
The results in the paper used s5? and s6?.

Cross-correlations with the Planck kappa map in general use the MV map, but the files labeled
"nosz" use the SZ deprojected map.
