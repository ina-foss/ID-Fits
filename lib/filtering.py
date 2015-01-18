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
