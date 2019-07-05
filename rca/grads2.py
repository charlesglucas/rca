import numpy as np
from modopt.opt.gradient import GradParent, GradBasic
from modopt.math.matrix import PowerMethod
from modopt.signal.wavelet import filter_convolve
import utils
from scipy.signal import fftconvolve
from modopt.math.convolve import convolve

def degradation_op(X, shift_ker, D):
    """ Shift and decimate fine-grid image."""
    return utils.decim(fftconvolve(X,shift_ker,mode='same'),
                       D,av_en=0)

def adjoint_degradation_op(x_i, shift_ker, D):
    """ Apply adjoint of the degradation operator."""
    return fftconvolve(utils.transpose_decim(x_i,D),
                       shift_ker,mode='same')
        
class CoeffGrad(GradParent, PowerMethod):
    """ Gradient class for the coefficient update.
    
    Parameters
    ----------
    data: np.ndarray
        Observed data.
    S: np.ndarray
        Current eigenPSFs :math:`S`.
    VT: np.ndarray
        Matrix of concatenated graph Laplacians.
    flux: np.ndarray
        Per-object flux value.
    sig: np.ndarray
        Noise levels.
    ker: np.ndarray
        Shifting kernels.
    ker_rot: np.ndarray
        Inverted shifting kernels.
    D: float
        Upsampling factor.
    """
    def __init__(self, data, S, VT, M, flux, sig, ker, ker_rot, D, data_type='float'):
        self._grad_data_type = data_type
        self.obs_data = data
        self.op = self.MX 
        self.trans_op = self.MtX 
        self.VT = VT
        self.flux = flux
        self.sig = sig
        self.ker = ker
        self.ker_rot  = ker_rot
        self.D = D
        # initialize Power Method to compute spectral radius
        PowerMethod.__init__(self, self.trans_op_op, 
                        (S.shape[-1],VT.shape[0]), auto_run=False)
        self.update_S(np.copy(S), update_spectral_radius=False)
        
        self._current_rec = None # stores latest application of self.MX

    def update_S(self, new_S, update_spectral_radius=True):
        """ Update current eigenPSFs."""
        self.S = new_S
        # Apply degradation operator to components
        normfacs = self.flux / (np.median(self.flux)*self.sig)
        self.FdS = np.array([[nf * degradation_op(S_j,shift_ker,self.D) 
                              for nf,shift_ker in zip(normfacs, utils.reg_format(self.ker))] 
                              for S_j in utils.reg_format(self.S)])
        self.dS = np.array([utils.decim(S_j,self.D,av_en=0) for S_j in utils.reg_format(self.S)])
        if update_spectral_radius:
            PowerMethod.get_spec_rad(self)
            
    def MX(self, alpha):
        """Apply degradation operator and renormalize.

        Parameters
        ----------
        alpha: np.ndarray
            Current weights (after factorization by :math:`V^\\top`).
        """
        A_stars = alpha.dot(self.VT)
        A_gal = A_stars.dot(M.T)
        dec_rec = np.empty(self.obs_data.shape)
        N_stars, N_gal = A_stars.shape[1], A_gal.shape[1]
        for j in range(N_stars):
            dec_rec[:,:,j] = np.sum(A_stars[:,j].reshape(-1,1,1)*self.FdS[:,j],axis=0)
        for j in range(N_gal):
            dec_rec[:,:,N_stars+j] += np.sum(convolve(A_gal[:,j].reshape(-1,1,1)*self.dS, self.X_gal),axis=0)
        self._current_rec = dec_rec
        return self._current_rec

    def MtX(self, x):
        """Adjoint to degradation operator :func:`MX`.

        Parameters
        ----------
        x : np.ndarray
            Set of finer-grid images.
        """ 
        x = utils.reg_format(x)
        N_stars, N_gal = self.VT.shape[1], self.VT_gal.shape[1]
        STx_stars = np.array([np.sum(FdS_i*x[:N_stars], axis=(1,2)) for FdS_i in self.FdS])
        STx_gal = np.array([np.sum(self.dS*convolve_stack(x[N_stars:],self.X_gal, rot_kernel=True), axis=(1,2))]).T
        STx = STx_stars+STx_gal.dot(M.T)
        return STx.dot(self.VT.T) 
    
    def cost(self, x, y=None, verbose=False):
        """ Compute data fidelity term. ``y`` is unused (it's just so ``modopt.opt.algorithms.Condat`` 
        can feed the dual variable.)
        """
        if isinstance(self._current_rec, type(None)):
            self._current_rec = self.MX(x)
        cost_val = 0.5 * np.linalg.norm(self._current_rec - self.obs_data) ** 2
        return cost_val
                
    def get_grad(self, x):
        """Compute current iteration's gradient.
        """
        self.grad = self.MtX(self.MX(x) - self.obs_data)        
      

class SourceGrad(GradParent, PowerMethod):
    """Gradient class for the eigenPSF update.
    
    Parameters
    ----------
    data: np.ndarray
        Input data array, a array of 2D observed images (i.e. with noise).
    A: np.ndarray
        Current estimation of corresponding coefficients.
    flux: np.ndarray
        Per-object flux value.
    sig: np.ndarray
        Noise levels.
    ker: np.ndarray
        Shifting kernels.
    ker_rot: np.ndarray
        Inverted shifting kernels.
    D: float
        Upsampling factor.
    filters: np.ndarray
        Set of filters.
    """

    def __init__(self, obs_data, A_stars, M, X_gal, flux, sig, ker, ker_rot, D, filters, data_type='float'):
        self._grad_data_type = data_type
        self.obs_data = obs_data
        self.op = self.MX 
        self.trans_op = self.MtX 
        self.A_stars = np.copy(A_stars)
        self.M = np.copy(M)
        self.X_gal = np.copy(X_gal)
        self.flux = flux
        self.sig = sig
        self.ker = ker
        self.ker_rot  = ker_rot
        self.D = D
        self.filters = filters
        # initialize Power Method to compute spectral radius
        hr_shape = np.array(obs_data.shape[:2])*D
        PowerMethod.__init__(self, self.trans_op_op, 
                        (A_stars.shape[0],filters.shape[0])+tuple(hr_shape), auto_run=False)
        
        self._current_rec = None # stores latest application of self.MX
        
    def update(self, new_A_stars, new_est_gal, update_spectral_radius=True):
        """Update current weights.
        """
        self.A_stars = new_A_stars
        self.est_gal = new_est_gal
        if update_spectral_radius:
            PowerMethod.get_spec_rad(self)

    def MX(self, transf_S):
        """Apply degradation operator and renormalize.

        Parameters
        ----------
        transf_S : np.ndarray
            Current eigenPSFs in Starlet space.

        Returns
        -------
        np.ndarray result

        """
        normfacs = self.flux / (np.median(self.flux)*self.sig)
        S = utils.rca_format(np.array([filter_convolve(transf_Sj, self.filters, filter_rot=True)
                             for transf_Sj in transf_S]))
        PSFs = S.dot(self.A_stars).dot(M.T)
        reg_obs_gal = np.copy(self.obs_data[:,:,self.A_stars.shape[1]:])
        reg_obs_gal = utils.reg_format(reg_obs_gal)
        ntest = reg_obs_gal.shape[0]

        cents = []
        for galaxy in reg_obs_gal:
            cents += [utils.CentroidEstimator(galaxy)]
        shifts = np.array([ce.return_shifts() for ce in cents])

        shift_kernels, _ = utils.shift_ker_stack(shifts,self.D)
        SA = np.array([fftconvolve(PSFs[:,:,j],shift_kernels[:,:,j],mode='same') for j in range(ntest)])
        
        dec_rec_stars = np.array([nf * degradation_op(S.dot(A_stars_i),shift_ker,self.D) for nf,A_stars_i,shift_ker in zip(normfacs, self.A_stars.T, utils.reg_format(self.ker))])
        dec_rec_gal = convolve_stack(utils.decim(SA,self.D,av_en=0), self.X_gal, method='astropy')
        dec_rec = np.concatenate((dec_rec_stars, dec_rec_gal), axis=0)
        self._current_rec = utils.rca_format(dec_rec)
        return self._current_rec
        
    def _plot_func(self, im, wind=False, cmap='gist_stern', title=''):
        if not wind:
            plt.imshow(im, cmap=cmap, interpolation='Nearest')
        else:
            vmin, vmax = wind
            plt.imshow(im, cmap=cmap, interpolation='Nearest', vmin=vmin, vmax=vmax)
        if title:
            plt.title(title)
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.show()
        plt.close()

    def MtX(self, x):
        """Adjoint to degradation operator :func:`MX`.

        """
        normfacs = self.flux / (np.median(self.flux)*self.sig)
        x = utils.reg_format(x)
        upsamp_x_stars = np.array([nf * adjoint_degradation_op(x_i,shift_ker,self.D) for nf,x_i,shift_ker in zip(normfacs, x, utils.reg_format(self.ker_rot))])
        upsamp_x_gal = np.array([utils.transpose_decim(x_i,self.D) for x_i in x[upsamp_x_stars.shape[0]:]])
        
        x_gal = upsamp_x_gal
        reg_obs_gal = np.copy(self.obs_data[:,:,self.A_stars.shape[1]:])
        reg_obs_gal = utils.reg_format(reg_obs_gal)
        ntest = reg_obs_gal.shape[0]

        cents = []
        for galaxy in reg_obs_gal:
            cents += [utils.CentroidEstimator(galaxy)]
        shifts = np.array([ce.return_shifts() for ce in cents])

        shift_kernels, shift_kernels_rot = utils.shift_ker_stack(shifts,self.D)
        SA = np.array([fftconvolve(x_gal[j],shift_kernels_rot[:,:,j],mode='same') for j in range(ntest)])
        
        x, upsamp_x_stars = utils.rca_format(x), utils.rca_format(upsamp_x_stars)
        xA = upsamp_x_stars.dot(self.A_stars.T)
        xX = convolve_stack(SA, self.X_gal, rot_kernel=True, method='astropy')
        xX = utils.rca_format(xX)
        xXA_gal = xX.dot(self.A_stars.T).dot(M.T)
        xXA = xA + xXA_gal
        '''
        self._plot_func(SA[0], title='x_gal')
        self._plot_func(upsamp_x_stars[:,:,0], title='upsamp_x_stars')
        self._plot_func(xX[:,:,0], title='xX')
        self._plot_func(xA[:,:,0], title='xA')
        self._plot_func(xXA_gal[:,:,0], title='xXA_gal')
        self._plot_func(xXA[:,:,0], title='xXA')
        '''
        return utils.apply_transform(xXA,self.filters)
    
    def cost(self, x, y=None, verbose=False):
        """ Compute data fidelity term. ``y`` is unused (it's just so 
        ``modopt.opt.algorithms.Condat`` can feed the dual variable.)
        """
        if isinstance(self._current_rec, type(None)):
            self._current_rec = self.MX(x)
        cost_val = 0.5 * np.linalg.norm(self._current_rec - self.obs_data) ** 2
        print cost_val
        '''
        self.cost_val.append(cost_val)
        x = range(len(self.cost_val))
        plt.plot(x, self.cost_val)
        plt.show()
        '''
        return cost_val

    def get_grad(self, x):
        """Compute current iteration's gradient.
        """
        self.grad = self.MtX(self.MX(x) - self.obs_data)