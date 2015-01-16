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
import random
import numpy as np
from sklearn.linear_model import SGDClassifier


class DiagonalMahalanobisMetric:
    
    def __init__(self, W=None):
        self.W_ = W
    
    
    def fit(self, X, y, n_samples=-1, n_iter=5):
        if n_samples <= 0:
            n_samples = X.shape[0] * (X.shape[0]-1) / 2
            random_sampling = False
        else:
            random_sampling = True
        
        n_features = X.shape[1]
        
        X2 = np.empty((n_samples, n_features), dtype=np.float64)
        y2 = np.empty((n_samples), dtype=np.int8)
        
        print "Creating X and y vectors..."
        sys.stdout.flush()

        # Keep every samples
        if not random_sampling:
            n = X.shape[0]
            index = 0
            for i in range(n):
                for j in range(i+1, n):
                    X2[index] = (X[i] - X[j]) ** 2

                    if y[i] == y[j]:
                        y2[index] = -1
                    else:
                        y2[index] = 1
                    
                    index += 1
        
        # Keep subset of original samples
        else:
            labels = list(set(y))
            Xk = [X[y == label] for label in labels]
            
            for i in range(n_samples):
                coin = int(2*random.random())

                if coin == 0:
                    y_ = -1
                    found = False
                    while not found:
                        samples = random.choice(Xk)
                        found = samples.shape[0] >= 2
                    x1, x2 = random.sample(samples, 2)
                else:
                    y_ = 1
                    class1, class2 = random.sample(Xk, 2)
                    x1 = random.choice(class1)
                    x2 = random.choice(class2)
                    
                X2[i] = (x1 - x2) ** 2
                y2[i] = y_

        print "Performing SGD..."
        sys.stdout.flush()

        svm = SGDClassifier(loss='hinge', penalty='l2', shuffle=True, class_weight='auto', alpha=0.01, n_iter=n_iter)
        svm.fit(X2, y2)
        print "Finished with score: %f" % svm.score(X2, y2)
        self.W_ = svm.coef_[0]
        self.b_ = svm.intercept_[0]

        
    def mesureDistance(self, x1, x2):
        delta = (x1 - x2) ** 2
        return np.inner(self.W_, delta)
    

    """
    def transform(self, X):
        return np.dot(X, np.diag(self.W_))
    """
