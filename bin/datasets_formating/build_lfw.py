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


import os
import argparse
import shutil
import cv2
import numpy as np

execfile("fix_imports.py")
import config



def loadImagesDatabase(directory):
    names_file = os.path.join(directory, "lfw-names.txt")
    
    names = {}
    with open(names_file, 'r') as f:
        for line in f.readlines():
            name, number = line.split()
            names[name] = int(number)

    data = []
    global_index = 0
    mapping = {}
    for name in sorted(names):
        for index in range(1, names[name]+1):
            filename = os.path.join(directory, name, "%s_%04d.jpg" % (name, index))
            data.append(cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE))
            if data[-1] is None:
                print filename

            mapping[(name, index)] = global_index
            global_index += 1

    return np.asarray(data), mapping


def saveMapping(filename, mapping):
    f = open(filename, 'w')
    f.write("%i\n" % len(mapping))
    for name, index in sorted(mapping):
        f.write("%s %i %i\n" % (name, index, mapping[(name, index)]))



if __name__ == "__main__":
    
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Format LFW-like datasets before further processing.")
    parser.add_argument("directory", help="LFW, LFW-a, etc. directory")
    parser.add_argument("output_name", help="name to give to the dataset")
    args = parser.parse_args()


    # Load the dataset and format it properly
    data, mapping = loadImagesDatabase(args.directory)


    # Save the result
    if not os.path.exists(config.lfw_path):
        os.makedirs(config.lfw_path)

    filename = os.path.join(config.lfw_path, args.output_name)
    np.save(filename, data)
    print "Data saved to %s" % filename

    filename = os.path.join(config.lfw_path, "mapping.txt")
    saveMapping(filename, mapping)
    print "Mapping saved to %s" % filename

    
    # Copy other useful files
    files = ["pairsDevTest.txt", "pairsDevTrain.txt", "peopleDevTest.txt", "peopleDevTrain.txt", "pairs.txt", "people.txt"]
    for f in files:
        filename = os.path.join(args.directory, f)
        if os.path.exists(filename):
            print "Copying %s" % filename
            shutil.copy(filename, config.lfw_path)
        else:
            print "Warning: %s not found" % filename

