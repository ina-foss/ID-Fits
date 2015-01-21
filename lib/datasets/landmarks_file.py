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



def readPtsLandmarkFile(filename, landmarks_number):
    f = open(filename)

    # Skip first 3 lines
    for i in range(3):
        f.readline()

    # Read landmarks position
    landmarks = np.empty((landmarks_number, 2), dtype=np.float)
    for i in range(landmarks_number):
        landmarks[i] = np.array([float(x) for x in f.readline().split()])
            
    return landmarks
