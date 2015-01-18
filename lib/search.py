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


import numpy as np
from scipy.signal import gaussian



def nnList(descriptor, database, nn, similarity=np.inner):
    best_scores = [(None, -1e3)]*nn
    for test_desc, label in zip(*database):
        score = similarity(test_desc, descriptor)
        if score > best_scores[-1][1]:
            for i in range(len(best_scores)):
                if best_scores[i][1] < score:
                    best_scores.insert(i, (label, score))
                    break
            best_scores.pop()
    return best_scores


def nnSearch(descriptor, database, nn, similarity=np.inner):
    best_scores = nnList(descriptor, database, nn, similarity=similarity)
    nn_scores = {}
    for label, score in best_scores:
        if label not in nn_scores:
            nn_scores[label] = 0
        nn_scores[label] += 1
    nn_scores = zip(nn_scores.keys(), nn_scores.values())

    return sorted(nn_scores, key=lambda x: x[1], reverse=True), best_scores


def nnSumSearch(descriptor, database, nn, similarity=np.inner):
    best_scores = nnList(descriptor, database, nn, similarity=similarity)
    nn_scores = {}
    for label, score in best_scores:
        if label not in nn_scores:
            nn_scores[label] = 0
        nn_scores[label] += score + 140
    nn_scores = zip(nn_scores.keys(), nn_scores.values())

    return sorted(nn_scores, key=lambda x: x[1], reverse=True), best_scores


# sigma = 2.15  <=>  (n > 5, 0.01%)
def nnGaussianKernelSearch(descriptor, database, nn, sigma=2.15):
    best_scores = nnList(descriptor, database, nn)

    kernel = gaussian(2*(nn-1)+1, sigma)[nn-1:]
    number_nn = {}
    nn_scores = {}
    for label, score in best_scores:
        if label not in nn_scores:
            nn_scores[label] = 0
            number_nn[label] = 0
        nn_scores[label] += score * kernel[number_nn[label]]
        number_nn[label] += 1
    nn_scores = zip(nn_scores.keys(), nn_scores.values())

    return sorted(nn_scores, key=lambda x: x[1], reverse=True), best_scores
