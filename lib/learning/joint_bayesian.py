# ID-Fits
# Copyright (c) 2015 Institut National de l'Audiovisuel, INA, All rights reserved.
# 
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3.0 of the License, or (at your option) any later version.
# 
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public
# License along with this library.


import sys
import numpy as np



class JointBayesian:

    def __init__(self):
        pass
    
    
    def fit(self, X, y):
        self.dtype = X.dtype
        self.n_features = X.shape[1]
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        Xc_i = [Xc[y == k] for k in set(y)]
        x_Xc_sum = [np.sum(xc_i, axis=0) for xc_i in Xc_i]
        
        S_mu, S_eps = self._computeCovarianceMatrices(X, y)
        
        iteration = 1
        converged = False
        error_S_mu = error_S_eps = 1e20
        while not converged:
            F = np.linalg.inv(S_eps)
            G_0 = -S_mu.dot(F)
            mu = []
            eps = []
            for i, Xc in enumerate(Xc_i):
                m = Xc.shape[0]
                G = np.dot(np.linalg.inv(m*S_mu+S_eps), G_0)
                
                mu.append(np.dot(np.dot(S_mu, F+m*G), x_Xc_sum[i]))
                
                C = np.dot(np.dot(S_eps, G), x_Xc_sum[i])
                for x in Xc:
                    eps.append(x + C)
                    
            prev_S_mu = np.copy(S_mu)
            prev_S_eps = np.copy(S_eps)
            S_mu = np.cov(mu, ddof=1, rowvar=0)
            S_eps = np.cov(eps, ddof=1, rowvar=0)
            
            prev_error_S_mu = error_S_mu
            prev_error_S_eps = error_S_eps
            error_S_mu = np.linalg.norm(S_mu-prev_S_mu) 
            error_S_eps = np.linalg.norm(S_eps-prev_S_eps)
            print "Finished iteration #%d with error reduction: %f %f"%(iteration, error_S_mu, error_S_eps)
            sys.stdout.flush()
            if (prev_error_S_mu < error_S_mu and prev_error_S_eps < error_S_eps) or (error_S_mu < 1e-1 and error_S_eps < 1e-1):
                S_mu = prev_S_mu
                S_eps = prev_S_eps
                converged = True
            
            iteration += 1
        
        self.S_mu = S_mu
        self.S_eps = S_eps
        
        F = np.linalg.inv(S_eps)
        self.G = - np.dot(np.linalg.inv(2*S_mu + S_eps), np.dot(S_mu, F))
        self.A = np.linalg.inv(S_mu + S_eps) - (F + self.G)
        u, s, v = np.linalg.svd(-self.G)
        self.U = np.diag(np.sqrt(s)).dot(v)
    
    
    def _computeCovarianceMatrices(self, X, y):
        classes_means = []
        s_w = np.zeros((X.shape[1], X.shape[1]), dtype=self.dtype)
        for k in set(y):
            Xg = X[y == k]
            mean = Xg.mean(axis=0)
            classes_means.append(mean)
            Xgc = Xg - mean
            s_w += np.dot(Xgc.T, Xgc) / (len(Xg) - 1)
        s_w /= len(set(y))
        s_b = np.cov(np.asarray(classes_means), ddof=1, rowvar=0)
        return s_b, s_w
    

    def _randomInitialization(self):
        S_mu = np.zeros((self.n_features, self.n_features), dtype=self.dtype)
        S_eps = np.zeros((self.n_features, self.n_features), dtype=self.dtype)
        for i in range(self.n_features):
            S_mu[i, i:] = np.random.normal(size=(self.n_features-i))
            S_eps[i, i:] = np.random.normal(size=(self.n_features-i))
            S_mu[i:, i] = S_mu[i, i:]
            S_eps[i:, i] = S_eps[i, i:]
        return S_mu, S_eps
            
    
    def transform(self, X):
        n_samples = X.shape[0]
        out = np.empty((n_samples, self.n_features+1), dtype=X.dtype)
        Xc = X - self.mean_
        for i in range(n_samples):
            out[i, 0] = np.dot(Xc[i].T, np.dot(self.A, Xc[i]))
            out[i, 1:] = np.dot(self.U, Xc[i].T)
        return out
        
        
    def mesureDistance(self, x, y):
        xc = x - self.mean_
        yc = y - self.mean_
        return np.dot(xc.T, np.dot(self.A, xc)) + np.dot(yc.T, np.dot(self.A, yc)) - 2 * np.dot(xc.T, np.dot(self.G, yc))



def jointBayesianDistance(x, y):
    return x[0] + y[0] + 2*np.dot(x[1:].T, y[1:])
