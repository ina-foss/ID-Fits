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
from sklearn.decomposition import PCA


class LargeMarginDimensionReduction:
    
    def __init__(self, n_components, n_iter=int(1e6), learning_rate=0.01, distance=None):
        self.n_components_ = n_components
        self.n_iter_ = n_iter
        self.learning_rate_ = learning_rate
        
        if distance is None:
            self.distance_ = self.mahalanobisDistance
        else:
            self.distance_ = distance
        
        
    def fit(self, X, y):
        W = self.initializationWithPca_(X)

        print 'Initializing b...'
        sys.stdout.flush()
        b = self.initializeB_(X, y, W)

        labels = list(set(y))
        Xk = [X[y == label] for label in labels]
        
        for iteration in xrange(self.n_iter_):
            if (iteration % 100) == 0:
                print 'Starting iteration #%i'%(iteration+1)
                sys.stdout.flush()
            
            coin = int(2*random.random())
            if coin == 0:
                y_ = 1
                samples = random.choice(Xk)
                x1, x2 = random.sample(samples, 2)
            else:
                y_ = -1
                class1, class2 = random.sample(Xk, 2)
                x1 = random.choice(class1)
                x2 = random.choice(class2)
            
            if y_*(b-self.distance_(x1, x2, W)) <= 1:
                W -= self.learning_rate_ * y_ * np.dot(W, np.outer(x1-x2, x1-x2))
                b += self.learning_rate_ * y_ * b
                
        self.W_ = W
        self.b_ = b
        
        
    def initializationWithPca_(self, X):
        print 'Initializing with PCA...'
        sys.stdout.flush()
        
        if len(X) > 1000:
            random_indexes = random.sample(range(len(X)), 1000)
        else:
            random_indexes = range(len(X))
        pca = PCA(n_components=self.n_components_, whiten=True)
        pca.fit(X[random_indexes])
        return np.dot(np.diag(np.power(pca.explained_variance_, -0.5)), pca.components_)


    """
    def initializeB_(self, X, y, W):
        acc = 0
        interval = (0.0, 100.0)
        prev_b = 0.0
        b = 1.0
        
        for _ in range(10):
            for i in range(len(X)):
                for j in range(i, len(X)):
                    if y[i] == y[j]:
                        y_ = 1
                    else:
                        y_ = -1
                
                    if y_*(b-self.distance_(x1, x2, W)) > 1:
                        acc += 1

            if acc > prev_best_acc:
                prev_best_acc = acc

                if prev_b < b:
                    interval[0] = b
                else:
                    interval[1] = b
                
                b = (interval[1] - interval[0]) / 2
    """

    def initializeB_(self, X, y, W, n_rec=5, n_num=10):
        interval = np.linspace(1.0, 100.0, num=n_num)
        
        n_samples = X.shape[0]
        X2 = np.dot(X, W.T)

        for _ in range(n_rec):
            acc_values = []

            for b in interval:
                acc = 0

                for i in range(n_samples):
                    for j in range(i+1, n_samples):
                        if y[i] == y[j]:
                            y_ = 1
                        else:
                            y_ = -1
                        
                        delta = X2[i]-X2[j]
                        if y_*(b-np.inner(delta, delta)) > 1:
                            acc += 1

                acc_values.append(acc)
            
            step = interval[1] - interval[0]
            best_b = interval[np.argmax(acc_values)]
            interval = np.linspace(best_b-step, best_b+step, num=n_num)
            print "Current best b value: %f"%best_b
            sys.stdout.flush()

        return best_b

        

    
    def mahalanobisDistance(self, x1, x2, W):
        delta = np.dot(W, (x1 - x2))
        return np.inner(delta, delta)
    
    
    def transform(self, X):
        return np.dot(X, self.W_.T)
