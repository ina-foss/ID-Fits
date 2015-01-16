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


import os, math
from matplotlib.pyplot import *
import cv2
import pylab


dataset_path = "/home/tlorieul/Data/lfw/"
sample_directory = os.path.join(dataset_path, "William_Ford_Jr")



def showMosaic(imgs, titles = [], fullscreen = False, ncols = 4, nrows = None, color = "gray", resize=True, interpolation=None):
    if titles == []:
        titles = [None]*len(imgs)

    if resize:
        aspect = None
    else:
        aspect = "equal"

    prev_figsize = pylab.rcParams['figure.figsize']
    width = np.max([img.shape[0] for img in imgs]) / float(80)
    if not nrows:
        nrows = math.ceil(len(imgs) / float(ncols))
    pylab.rcParams['figure.figsize'] = (ncols*width+(ncols-1)*0.25, nrows*5)


    for img, i, img_title in zip(imgs, range(len(imgs)), titles):
        subplot(len(imgs)/ncols+1, ncols, i+1)
        imshow(img, aspect=aspect, interpolation=interpolation)
        if color == "gray":
            gray()
        if img_title:
            title(img_title)
        axis("off")
    
    if fullscreen:
        fig_manager = get_current_fig_manager()
        fig_manager.resize(*fig_manager.window.maxsize())
    show()
    
    pylab.rcParams['figure.figsize'] = prev_figsize


def readImagesInDir(directory, color="rgb"):
    imgs_name = [f for f in os.listdir(directory)]

    if color == "rgb":
        imgs = [imread(os.path.join(directory, img_name)) for img_name in imgs_name]
    elif color == "grayscale":
        imgs = [cv2.imread(os.path.join(directory, img_name), cv2.CV_LOAD_IMAGE_GRAYSCALE) for img_name in imgs_name]
    else:
        throw()

    return imgs_name, imgs


def readDataFromFile(filename, directory = None, dtype = np.float, mmap = False, shape = None, binary = False):
    #return np.loadtxt(os.path.join(directory, filename), dtype=dtype)
    f = os.path.join(directory, filename)
    if binary:
        return np.load(f)
    if mmap and shape != None:
        return np.memmap(f, dtype=dtype, mode="r", shape=shape)
    else:
        return np.fromfile(f, dtype=dtype)


def writeDataToFile(data, filename, directory = None, binary = False):
    if directory:
        if not os.path.exists(directory):
            os.makedirs(directory)
        f = os.path.join(directory, filename)
    else:
        f = filename

    if binary:
        np.save(f, data)
    else:
        np.savetxt(f, data)
        #np.savetxt(f, data, header="%i %i"%(len(data), data[0].shape[0]))
        #np.tofile()

