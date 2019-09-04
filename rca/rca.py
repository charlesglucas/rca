from __future__ import absolute_import, print_function
import numpy as np
from scipy.interpolate import Rbf
from modopt.signal.wavelet import get_mr_filters, filter_convolve
from scipy.signal import fftconvolve
from modopt.opt.cost import costObj
from modopt.opt.proximity import Positivity
from modopt.opt.reweight import cwbReweight
from scipy.interpolate import Rbf
import sf_deconvolve
import modopt.opt.algorithms as optimalg
import rca.proxs as rca_prox
import rca.grads as grads
import rca.grads2 as grads2
import rca.utils as utils

def quickload(path):
    """ Load pre-fitted RCA model (saved with :func:`RCA.quicksave`).
    
    Parameters
    ----------
    path: str
        Path to where the fitted RCA model was saved.
    """
    if path[-4:] != '.npy':
        path += '.npy'
    RCA_params, fitted_model = np.load(path, allow_pickle=True)
    loaded_rca = RCA(**RCA_params)
    loaded_rca.stars_pos = fitted_model['stars_pos']
    loaded_rca.gal_pos = fitted_model['gal_pos']
    loaded_rca.weights_stars = fitted_model['weights_stars']
    loaded_rca.S = fitted_model['S']
    loaded_rca.flux_ref = fitted_model['flux_ref']
    loaded_rca.psf_size = fitted_model['psf_size']
    loaded_rca.VT = fitted_model['VT']
    loaded_rca.alpha = fitted_model['alpha']
    loaded_rca.is_fitted = True
    return loaded_rca

class RCA(object):
    """ Resolved Components Analysis.
    
    Parameters
    ----------
    n_comp: int
        Number of components to learn.
    upfact: int
        Upsampling factor. Default is 1 (no superresolution).
    ksig: float
        Value of :math:`k` for the thresholding in Starlet domain (taken to be 
        :math:`k\sigma`, where :math:`\sigma` is the estimated noise standard deviation.)
    n_scales: int
        Number of Starlet scales to use for the sparsity constraint. Default is 3. Unused if
        ``filters`` are provided.
    ksig_init: float
        Similar to ``ksig``, for use when estimating shifts and noise levels, as it might 
        be desirable to have it set higher than ``ksig``. Unused if ``shifts`` are provided 
        when running :func:`RCA.fit`. Default is 5.
    filters: np.ndarray
        Optional filters to the transform domain wherein eigenPSFs are assumed to be sparse;
        convolution by them should amount to applying :math:`\Phi`. Optional; if not provided, the
        Starlet transform with `n_scales` scales will be used.
    verbose: bool or int
        If True, will only output RCA-specific lines to stdout. If verbose is set to 2,
        will run ModOpt's optimization algorithms in verbose mode. 
        
    """
    def __init__(self, n_comp, upfact=1, ksig=3, n_scales=3, ksig_init=5, filters=None, 
                 verbose=2):
        self.n_comp = n_comp
        self.upfact = upfact
        self.ksig = ksig
        self.ksig_init = ksig_init
        
        if filters is None:
            # option strings for mr_transform
            self.opt = ['-t2', '-n{}'.format(n_scales)]
            self.default_filters = True
        else:
            self.Phi_filters = filters
            self.default_filters = False
        self.verbose = verbose
        if self.verbose > 1:
            self.modopt_verb = True
        else:
            self.modopt_verb = False
        self.is_fitted = False
        
    def fit(self, obs_stars, obs_gal, stars_pos, gal_pos, S=None, VT=None, alpha=None,
            shifts=None, sigs=None, psf_size=None, psf_size_type='fwhm',
            flux=None, method=2, nb_iter=2, nb_subiter_S=200, nb_reweight=0, 
            nb_subiter_weights=None, n_eigenvects=5, graph_kwargs={}):
        """ Fits RCA to observed star field.
        
        Parameters
        ----------
        obs_data: np.ndarray
            Observed data.
        obs_pos: np.ndarray
            Corresponding positions.
        obs_weights: np.ndarray
            Corresponding weights. Can be either one per observed star, or contain pixel-wise values. Masks can be
            handled via binary weights. Default is None (in which case no weights are applied). Note if fluxes and
            shifts are not provided, weights will be ignored for their estimation. Noise level estimation only removes 
            bad pixels (with weight strictly equal to 0) and otherwise ignores weights.
        S: np.ndarray
            First guess (or warm start) eigenPSFs :math:`S`. Default is ``None``.
        VT: np.ndarray
            Matrix of concatenated graph Laplacians. Default is ``None``.
        alpha: np.ndarray
            First guess (or warm start) weights :math:`\\alpha`, after factorization by ``VT``. Default is ``None``.
        shifts: np.ndarray
            Corresponding sub-pixel shifts. Default is ``None``; will be estimated from
            observed data if not provided.
        sigs: np.ndarray
            Estimated noise levels. Default is ``None``; will be estimated from data
            if not provided.
        psf_size: float
            Approximate expected PSF size in pixels; will be used for the size of the Gaussian window for centroid estimation.
            ``psf_size_type`` determines the convention used for this size (default is FWHM).
            Ignored if ``shifts`` are provided. Default is Gaussian sigma of 7.5 pixels.
        psf_size_type: str
            Can be any of ``'R2'``, ``'fwhm'`` or ``'sigma'``, for the size defined from quadrupole moments, full width at half maximum
            (e.g. from SExtractor) or 1-sigma width of the best matching 2D Gaussian. Default is ``'fwhm'``.
        flux: np.ndarray
            Flux levels. Default is ``None``; will be estimated from data if not provided.
        method: int
            Method 1 is RCA and method 2 is RCA++. Default is RCA.
        nb_iter: int
            Number of overall iterations (i.e. of alternations) after initialization with RCA. Note the weights do not get updated the last time around, so they actually get ``nb_iter-1`` updates.
            Default is 2.          
        nb_subiter_S: int
            Maximum number of iterations for :math:`S` updates. If ModOpt's optimizers achieve 
            internal convergence, that number may (and often is) not reached. Default is
            200.
        nb_reweight: int 
            Number of reweightings to apply during :math:`S` updates. See equation (33) in RCA paper. 
            Default is 0.
        nb_subiter_weights: int
            Maximum number of iterations for :math:`\\alpha` updates. If ModOpt's optimizers achieve 
            internal convergence, that number may (and often is) not reached. Default is None;
            if not provided, will be set to ``2*nb_subiter_S`` (as it was in RCA v1). 
        n_eigenvects: int
            Maximum number of eigenvectors to consider per :math:`(e,a)` couple. Default is ``None``;
            if not provided, *all* eigenvectors will be considered, which can lead to a poor
            selection of graphs, especially when data is undersampled. Ignored if ``VT`` and
            ``alpha`` are provided.
        graph_kwargs: dictionary
            List of optional kwargs to be passed on to the :func:`utils.GraphBuilder`.
        """
        
        self.obs_gal = np.copy(obs_gal)
        self.obs_stars = np.copy(obs_stars)
        self.shap_stars = self.obs_stars.shape
        self.shap_gal = self.obs_gal.shape
        self.obs_data = np.concatenate((self.obs_stars, self.obs_gal), axis=2)
        self.shap = self.obs_data.shape
        self.im_hr_shape = (self.upfact*self.shap[0],self.upfact*self.shap[1],self.shap_stars[2])
        self.stars_pos = stars_pos
        self.gal_pos = gal_pos        

        if S is None:
            self.S = np.zeros(self.im_hr_shape[:2] + (self.n_comp,))
        else:
            self.S = S
        self.VT = VT
        self.alpha = alpha
        self.shifts = shifts
        if shifts is None:
            self.psf_size = self._set_psf_size(psf_size, psf_size_type)
        self.sigs = sigs
        self.flux = flux
        self.nb_iter = nb_iter
        self.nb_subiter_S = nb_subiter_S
        if nb_subiter_weights is None:
            nb_subiter_weights = 2*nb_subiter_S
        self.nb_subiter_weights = nb_subiter_weights
        self.nb_reweight = nb_reweight
        self.n_eigenvects = n_eigenvects
        self.graph_kwargs = graph_kwargs
            
        if self.verbose:
            print('Running basic initialization tasks...')
        self._initialize()
        if self.verbose:
            print('... Done.')
        if self.VT is None or self.alpha is None:
            if self.verbose:
                print('Constructing graph constraint...')
            self._initialize_graph_constraint()
            if self.verbose:
                print('... Done.')
        else:
            self.weights_stars = self.alpha.dot(self.VT)
        if method==1:
            self._fit()
        else:
            self._fit2()
        self.is_fitted = True
        return self.S, self.weights_stars

    
    def quicksave(self, path):
        """ Save fitted RCA model for later use. Ideally, you would probably want to store the
        whole RCA instance, though this might mean storing a lot of data you are not likely to
        use if you do not alter the fit that was already performed.
        Stored models can be loaded with :func:`rca.quickload`.
        
        Parameters
        ----------
        path: str
            Path to where the fitted RCA model should be saved.
        """
        if not self.is_fitted:
            raise ValueError('RCA instance has not yet been fitted to observations. Please run\
            the fit method.')
        RCA_params = {'n_comp': self.n_comp, 'upfact': self.upfact}
        fitted_model = {'stars_pos': self.stars_pos, 'gal_pos': self.gal_pos, 'weights_stars': self.weights_stars, 'S': self.S, 'flux_ref': self.flux_ref}
        np.save(path+'.npy', [RCA_params,fitted_model])
              
    def estimate_psf(self, test_pos, n_neighbors=15, rbf_function='thin_plate', 
                     apply_degradation=False, shifts=None, flux=None,
                     upfact=None, rca_format=False):
        """ Estimate and return PSF at desired positions.
        
        Parameters
        ----------
        test_pos: np.ndarray
            Positions where the PSF should be estimated. Should be in the same format (units,
            etc.) as the ``obs_pos`` fed to :func:`RCA.fit`.
        n_neighbors: int
            Number of neighbors to use for RBF interpolation. Default is 15.
        rbf_function: str
            Type of RBF kernel to use. Default is ``'thin_plate'``.
        apply_degradation: bool
            Whether PSF model should be degraded (shifted and resampled on coarse grid), 
            for instance for comparison with stars. If True, expects shifts to be provided.
            Default is False.
        shifts: np.ndarray
            Intra-pixel shifts to apply if ``apply_degradation`` is set to True.
        flux: np.ndarray
            Flux levels by which reconstructed PSF will be multiplied if provided. For comparison with 
            stars if ``apply_degradation`` is set to True. 
        upfact: int
            Upsampling factor; default is None, in which case that of the RCA instance will be used.
        rca_format: bool
            If True, returns the PSF model in "rca" format, i.e. with axises
            (n_pixels, n_pixels, n_stars). Otherwise, and by default, return them in
            "regular" format, (n_stars, n_pixels, n_pixels).
        """
        if not self.is_fitted:
            raise ValueError('RCA instance has not yet been fitted to observations. Please run\
            the fit method.')
        if upfact is None:
            upfact = self.upfact

        ntest = test_pos.shape[0]               
        # Interpolate weights
        M = utils.thin_plate_interpolation(test_pos, self.stars_pos)
        test_weights = self.weights_stars.dot(M.T)

        PSFs = self._transform(test_weights)
        if apply_degradation:
            shift_kernels, _ = utils.shift_ker_stack(shifts,self.upfact)
            deg_PSFs = np.array([grads.degradation_op(PSFs[:,:,j], shift_kernels[:,:,j], upfact)
                                 for j in range(ntest)])
            if flux is not None:
                deg_PSFs *= flux.reshape(-1,1,1) / np.median(self.flux[self.obs_stars.shape[2]:])
            if rca_format:
                return utils.rca_format(deg_PSFs)
            else:
                return deg_PSFs
        elif rca_format:
            return PSFs
        else:
            return utils.reg_format(PSFs)

    def validation_stars(self, test_stars, test_pos):
        """ Match PSF model to stars - in flux, shift and pixel sampling - for validation tests.
        Returns both the matched PSFs' stamps and chi-square value.
        
        Parameters
        ----------
        test_stars: np.ndarray
            Star stamps to be used for comparison with the PSF model. Should be in "rca" format, 
            i.e. with axises (n_pixels, n_pixels, n_stars).
        test_pos: np.ndarray
            Their corresponding positions.
        """
        if not self.is_fitted:
            raise ValueError('RCA instance has not yet been fitted to observations. Please run\
            the fit method.')
        cents = []
        for star in utils.reg_format(test_stars):
            cents += [utils.CentroidEstimator(star, sig=self.psf_size)]
        test_shifts = np.array([ce.return_shifts() for ce in cents])
        test_fluxes = utils.flux_estimate_stack(test_stars,rad=4)
        matched_psfs = self.estimate_psf(test_pos, apply_degradation=True, 
                                    shifts=test_shifts, flux=test_fluxes)
        return matched_psfs
        
    def _set_psf_size(self, psf_size, psf_size_type):
        """ Handles different "size" conventions."""
        if psf_size is not None:
            if psf_size_type == 'fwhm':
                return psf_size / (2*np.sqrt(2*np.log(2)))
            elif psf_size_type == 'R2':
                return np.sqrt(psf_size / 2)
            elif psf_size_type == 'sigma':
                return psf_size
            else:
                raise ValueError('psf_size_type should be one of "fwhm", "R2" or "sigma"')
        else:
            print('''WARNING: neither shifts nor an estimated PSF size were provided to RCA;
the shifts will be estimated from the data using the default Gaussian
window of 7.5 pixels.''')
            return 7.5
  
    def _initialize(self):
        """ Initialization tasks related to noise levels, shifts and flux. Note it includes
        renormalizing observed data, so needs to be ran even if all three are provided."""
        if self.default_filters:
            init_filters = get_mr_filters(self.shap_stars[:2], opt=self.opt, coarse=True)
        else:
            init_filters = self.Phi_filters
        # noise levels
        if self.sigs is None:
            transf_data = rca_prox.apply_transform(self.obs_data, init_filters)
            sigmads = np.array([1.4826*utils.mad(fs[0]) for fs in transf_data])
            self.sigs = sigmads / np.linalg.norm(init_filters[0])
        else:
            self.sigs = np.copy(self.sigs)
        self.sig_min = np.min(self.sigs)
        # intra-pixel shifts
        if self.shifts is None:
            thresh_data = np.copy(self.obs_stars)
            cents = []
            for i in range(self.shap_stars[2]):
                # don't allow thresholding to be over 80% of maximum observed pixel
                nsig_shifts = min(self.ksig_init, 0.8*self.obs_stars[:,:,i].max()/self.sigs[i])
                thresh_data[:,:,i] = utils.HardThresholding(thresh_data[:,:,i], nsig_shifts*self.sigs[i])
                cents += [utils.CentroidEstimator(thresh_data[:,:,i], sig=self.psf_size)]
            self.shifts = np.array([ce.return_shifts() for ce in cents])
        self.shift_ker_stack,self.shift_ker_stack_adj = utils.shift_ker_stack(self.shifts,
                                                                              self.upfact)
        # flux levels
        if self.flux is None:
            self.flux = utils.flux_estimate_stack(self.obs_stars, rad=4)
        self.flux_ref = np.median(self.flux)
        # Normalize noise levels observed data
        self.sigs /= self.sig_min
        self.sigs_stars = self.sigs[:self.shap_stars[2]]
        self.obs_data /= self.sigs.reshape(1,1,-1)
        self.obs_stars = self.obs_data[:,:,:self.shap_stars[2]]
        self.obs_gal = self.obs_data[:,:,self.shap_stars[2]:]
    
    def _initialize_graph_constraint(self):
        gber = utils.GraphBuilder(self.obs_stars, self.stars_pos, self.n_comp, 
                                  n_eigenvects=self.n_eigenvects, verbose=self.verbose,
                                  **self.graph_kwargs)
        self.VT, self.alpha, self.distances = gber.VT, gber.alpha, gber.distances
        self.sel_e, self.sel_a = gber.sel_e, gber.sel_a
        self.weights_stars = self.alpha.dot(self.VT)                 

    ## RCA
    def _fit(self):
        weights_stars = self.weights_stars
        comp = self.S
        alpha = self.alpha
        #### Source updates set-up ####
        # initialize dual variable and compute Starlet filters for Condat source updates 
        dual_var = np.zeros((self.im_hr_shape))
        if self.default_filters:
            self.Phi_filters = get_mr_filters(self.im_hr_shape[:2], opt=self.opt, coarse=True)
        rho_phi = np.sqrt(np.sum(np.sum(np.abs(self.Phi_filters),axis=(1,2))**2))
        
        # Set up source updates, starting with the gradient
        source_grad = grads.SourceGrad(self.obs_stars, weights_stars, self.flux, self.sigs_stars, 
                                      self.shift_ker_stack, self.shift_ker_stack_adj, 
                                      self.upfact, self.Phi_filters)

        # sparsity in Starlet domain prox (this is actually assuming synthesis form)
        sparsity_prox = rca_prox.StarletThreshold(0) # we'll update to the actual thresholds later

        # and the linear recombination for the positivity constraint
        lin_recombine = rca_prox.LinRecombine(weights_stars, self.Phi_filters)

        #### Weight updates set-up ####
        # gradient
        weight_grad = grads.CoeffGrad(self.obs_stars, comp, self.VT, self.flux, self.sigs_stars, 
                                      self.shift_ker_stack, self.shift_ker_stack_adj, self.upfact)
        
        # cost function
        weight_cost = costObj([weight_grad], verbose=self.modopt_verb) 
        source_cost = costObj([source_grad], verbose=self.modopt_verb)
        
        # k-thresholding for spatial constraint
        iter_func = lambda x: np.floor(np.sqrt(x))+1
        coeff_prox = rca_prox.KThreshold(iter_func)
        

        for k in range(self.nb_iter):
            " ============================== Sources estimation =============================== "
            # update gradient instance with new weights...
            source_grad.update_A(weights_stars)
            
            # ... update linear recombination weights...
            lin_recombine.update_A(weights_stars)
            
            # ... set optimization parameters...
            beta = source_grad.spec_rad + rho_phi
            #beta = 0.6563937030792308 + rho_phi
            tau = 1./beta
            sigma = 1./lin_recombine.norm * beta/2

            # ... update sparsity prox thresholds...
            thresh = utils.reg_format(utils.acc_sig_maps(self.shap,self.shift_ker_stack_adj,self.sigs_stars,
                                                        self.flux,self.flux_ref,self.upfact,weights_stars,
                                                        sig_data=np.ones((self.shap[2],))*self.sig_min))
            thresholds = self.ksig*np.sqrt(np.array([filter_convolve(Sigma_k**2,self.Phi_filters**2) 
                                              for Sigma_k in thresh]))

            sparsity_prox.update_threshold(tau*thresholds)
            
            # and run source update:
            transf_comp = rca_prox.apply_transform(comp, self.Phi_filters)
            if self.nb_reweight:
                reweighter = cwbReweight(thresholds)
                for _ in range(self.nb_reweight):
                    source_optim = optimalg.Condat(transf_comp, dual_var, source_grad, sparsity_prox,
                                                   Positivity(), linear = lin_recombine, cost=source_cost,
                                                   max_iter=self.nb_subiter_S, tau=tau, sigma=sigma)
                    transf_comp = source_optim.x_final
                    reweighter.reweight(transf_comp)
                    thresholds = reweighter.weights 
            else:
                source_optim = optimalg.Condat(transf_comp, dual_var, source_grad, sparsity_prox,
                                               Positivity(), linear = lin_recombine, cost=source_cost,
                                               max_iter=self.nb_subiter_S, tau=tau, sigma=sigma)
                transf_comp = source_optim.x_final
            comp = utils.rca_format(np.array([filter_convolve(transf_compj, self.Phi_filters, True)
                                    for transf_compj in transf_comp]))
            
            #TODO: replace line below with Fred's component selection (to be extracted from `low_rank_global_src_est_comb`)
            ind_select = range(comp.shape[2])


            " ============================== Weights estimation =============================== "
            if k < self.nb_iter-1: 
                # update sources and reset iteration counter for K-thresholding
                weight_grad.update_S(comp)
                coeff_prox.reset_iter()
                weight_optim = optimalg.ForwardBackward(alpha, weight_grad, coeff_prox, cost=weight_cost,
                                                beta_param=weight_grad.inv_spec_rad, auto_iterate=False)
                weight_optim.iterate(max_iter=self.nb_subiter_weights)
                alpha = weight_optim.x_final
                weights_k = alpha.dot(self.VT)

                # renormalize to break scale invariance
                weight_norms = np.sqrt(np.sum(weights_k**2,axis=1)) 
                comp *= weight_norms
                weights_k /= weight_norms.reshape(-1,1)
                #TODO: replace line below with Fred's component selection 
                ind_select = range(weights_stars.shape[0])
                weights_stars = weights_k[ind_select,:]
                supports = None #TODO
    
        self.weights_stars = weights_stars
        self.S = comp
        self.alpha = alpha
        source_grad.MX(transf_comp)
        self.current_rec = source_grad._current_rec           
     
    ## RCA++
    def _fit2(self):
        weights_stars = self.weights_stars
        comp = self.S
        alpha = self.alpha
        
        #### Source updates set-up ####
        # interpolation matrix
        M = utils.thin_plate_interpolation(self.gal_pos, self.stars_pos)
        
        # Initialize dual variable and compute Starlet filters for Condat source updates 
        dual_var = np.zeros((self.im_hr_shape))
        if self.default_filters:
            self.Phi_filters = get_mr_filters(self.im_hr_shape[:2], opt=self.opt, coarse=True)
        rho_phi = np.sqrt(np.sum(np.sum(np.abs(self.Phi_filters),axis=(1,2))**2))
        

        # options for sf_deconvolve
        opts = vars(sf_deconvolve.get_opts(['-i', '', '-p', '','-o', '','-m', 'sparse', '--no_plots']))
        
        # Initialize estimated galaxies
        est_gal = utils.reg_format(np.zeros(self.shap_gal))

        # Set up source updates, starting with the gradient
        source_grad = grads2.SourceGrad(self.obs_data, weights_stars, M, est_gal, self.flux, self.sigs_stars, self.shift_ker_stack, self.shift_ker_stack_adj, self.upfact, self.Phi_filters)
        
        # sparsity in Starlet domain prox (this is actually assuming synthesis form)
        sparsity_prox = rca_prox.StarletThreshold(0) # we'll update to the actual thresholds later

        # and the linear recombination for the positivity constraint
        lin_recombine = rca_prox.LinRecombine(weights_stars, self.Phi_filters)

        #### Weight updates set-up ####                
        weight_grad = grads2.CoeffGrad(self.obs_data, comp, self.VT, M, est_gal, self.flux, self.sigs_stars, self.shift_ker_stack, self.shift_ker_stack_adj, self.upfact)
        
        # cost function
        weight_cost = costObj([weight_grad], verbose=self.modopt_verb) 
        source_cost = costObj([source_grad], verbose=self.modopt_verb)
        
        # k-thresholding for spatial constraint
        iter_func = lambda x: np.floor(np.sqrt(x))+1
        coeff_prox = rca_prox.KThreshold(iter_func)

        for k in range(self.nb_iter):                    
            " ============================== Sources estimation =============================== "
            #### Eigenpsf update ####
            # update gradient instance with new weights...
            source_grad.update(weights_stars, est_gal)
            
            # ... update linear recombination weights...
            lin_recombine.update_A(weights_stars)
            
            # ... set optimization parameters...
            beta = source_grad.spec_rad + rho_phi
            tau = 1./beta
            sigma = 1./lin_recombine.norm * beta/2

            # ... update sparsity prox thresholds...
            thresh = utils.reg_format(utils.acc_sig_maps(self.shap_stars,self.shift_ker_stack_adj,self.sigs_stars,
                                                       self.flux,self.flux_ref,self.upfact,weights_stars,
                                                   sig_data=np.ones((self.shap_stars[2],))*self.sig_min))
            thresholds = self.ksig*np.sqrt(np.array([filter_convolve(Sigma_k**2,self.Phi_filters**2) 
                                              for Sigma_k in thresh]))

            sparsity_prox.update_threshold(tau*thresholds)
            
            # and run source update:
            transf_comp = utils.apply_transform(comp, self.Phi_filters)
            if self.nb_reweight:
                reweighter = cwbReweight(thresholds)
                for _ in range(self.nb_reweight):
                    source_optim = optimalg.Condat(transf_comp, dual_var, source_grad, sparsity_prox,
                                                   Positivity(), linear = lin_recombine, cost=source_cost,
                                                   max_iter=self.nb_subiter_S, tau=tau, sigma=sigma)
                    transf_comp = source_optim.x_final
                    reweighter.reweight(transf_comp)
                    thresholds = reweighter.weights 
            else:
                source_optim = optimalg.Condat(transf_comp, dual_var, source_grad, sparsity_prox,
                                               Positivity(), linear = lin_recombine, cost=source_cost,
                                               max_iter=self.nb_subiter_S, tau=tau, sigma=sigma)
                transf_comp = source_optim.x_final
            comp = utils.rca_format(np.array([filter_convolve(transf_compj, self.Phi_filters, True)
                                    for transf_compj in transf_comp]))
            
            #TODO: replace line below with Fred's component selection (to be extracted from `low_rank_global_src_est_comb`)
            ind_select = range(comp.shape[2])

            if k < self.nb_iter-1:
                " ============================== Galaxies estimation =============================== "                
                psf = utils.reg_format(comp.dot(weights_stars).dot(M.T))                
                psf_norm = np.sum(psf,(1,2))
                psf /= psf_norm.reshape(-1,1,1)                             
                est_gal, _, _ = sf_deconvolve.run(utils.reg_format(self.obs_gal), psf, **opts)
                est_gal /= psf_norm.reshape(-1,1,1)                   
                
                " ============================== Weights estimation =============================== "
                #### Weight update ####
                # update sources and reset iteration counter for K-thresholding
                weight_grad.update(comp, est_gal)
                coeff_prox.reset_iter()
                weight_optim = optimalg.ForwardBackward(alpha, weight_grad, coeff_prox, cost=weight_cost,
                                                beta_param=weight_grad.inv_spec_rad, auto_iterate=False)
                weight_optim.iterate(max_iter=self.nb_subiter_weights)
                alpha = weight_optim.x_final
                weights_k = alpha.dot(self.VT)

                # renormalize to break scale invariance
                weight_norms = np.sqrt(np.sum(weights_k**2,axis=1)) 
                comp *= weight_norms
                weights_k /= weight_norms.reshape(-1,1)
                #TODO: replace line below with Fred's component selection 
                ind_select = range(weights_stars.shape[0])
                weights_stars = weights_k[ind_select,:]
                supports = None #TODO             
    
        self.weights_stars = weights_stars
        self.S = comp
        self.alpha = alpha
        source_grad.MX(transf_comp)
        self.current_rec = source_grad._current_rec                      
        
    def _transform(self, weights):
        return self.S.dot(weights)