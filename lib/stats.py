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



