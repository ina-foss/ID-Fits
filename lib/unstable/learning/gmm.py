import numpy as np
from yael import yael
from scipy import stats


class GMM:
    
    def __init__(self, n_components, n_iter=100, n_init=1, n_threads=1):
        self.n_components = n_components
        self.n_iter = n_iter
        self.n_init = n_init
        self.n_threads = n_threads
        self.yael_gmm = None
    
    
    def initYaelGmm(self):
        self.yael_gmm = yael.gmm_t()
        self.yael_gmm.d = self.n_features
        self.yael_gmm.k = self.n_components
        
        self.yael_gmm.mu = yael.numpy_to_fvec_ref(self.means_)
        self.yael_gmm.sigma = yael.numpy_to_fvec(self.covars_)
        self.yael_gmm.w = yael.numpy_to_fvec_ref(self.weights_)
    
    
    def fit(self, X):
        n_samples, self.n_features = X.shape
        
        yael_X = yael.numpy_to_fvec_ref(X)
        
        yael_gmm = yael.gmm_learn(
                self.n_features,
                n_samples,
                self.n_components,
                self.n_iter,
                yael_X,
                self.n_threads,
                0,
                self.n_init,
                yael.GMM_FLAGS_W | yael.GMM_FLAGS_SIGMA | yael.GMM_FLAGS_MU)
        
        self.means_ = yael.fvec_to_numpy(yael_gmm.mu, self.n_components*self.n_features).reshape((self.n_components, self.n_features))
        self.covars_ = yael.fvec_to_numpy(yael_gmm.sigma, self.n_components*self.n_features).reshape((self.n_components, self.n_features))
        self.weights_ = yael.fvec_to_numpy(yael_gmm.w, self.n_components)

        yael.gmm_delete(yael_gmm)
        
    
    def computeResponsabilities(self, X):
        if self.yael_gmm is None:
            self.initYaelGmm()

        if len(X.shape) == 1:
            n_samples = 1
        else:
            n_samples = X.shape[0]
        
        yael_X = yael.numpy_to_fvec_ref(X)
        yael_p = yael.fvec_new_0(self.n_components * n_samples)
        
        yael.gmm_compute_p_thread(n_samples, yael_X, self.yael_gmm, yael_p, yael.GMM_FLAGS_W | yael.GMM_FLAGS_SIGMA | yael.GMM_FLAGS_MU, self.n_threads)
        
        return yael.fvec_to_numpy_acquire(yael_p, n_samples*self.n_components).reshape((n_samples, self.n_components))
