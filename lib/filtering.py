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



def median_filtering(x):
    return np.median(x, axis=0)


def mean_filtering(x):
    return np.mean(x, axis=0)



class Filter:

    def __init__(self, method="mean", n=3):
        self.n = n
        self.accumulator = []

        if method == "mean":
            self.filtering_func = mean_filtering
        elif method == "median":
            self.filtering_func = median_filtering
        else:
            raise Exception("Unknown filtering method")


    def filter(self, x):
        self.accumulator.append(x)
        
        if len(self.accumulator) > self.n:
            self.accumulator.pop(0)

        return self.filtering_func(self.accumulator)


    def reset(self):
        self.accumulator = []
