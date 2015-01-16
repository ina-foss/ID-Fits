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
import pylab
from matplotlib import pyplot as plt



def plotROC(rocs, labels, title=None, show_grid=True):
    prev_figsize = pylab.rcParams['figure.figsize']

    pylab.rcParams['figure.figsize'] = (8.0, 6.0)

    for roc,label in zip(rocs,labels):
        plt.plot(roc[0], roc[1], label=label)
    
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(loc="lower right")

    if title:
        plt.title(title)
    if show_grid:
        plt.grid()

    pylab.rcParams['figure.figsize'] = prev_figsize



