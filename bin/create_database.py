import os
import argparse
import numpy as np

import fix_imports
import config
from datasets import lfw
from descriptors import *
from utils.file_manager import makedirsIfNeeded



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Builds database for IR.")
    parser.add_argument("descriptors", help="descriptor to use")
    args = parser.parse_args()

    filename = os.path.splitext(os.path.basename(args.descriptors))[0]

    mapping = lfw.loadMapping()
    labels = [""]*len(mapping)

    sets = lfw.loadTestSets()
    test_set = []
    for s in sets[:-1]:
        test_set += s
    
    descs = np.load(args.descriptors)[test_set]

    for (name, _), index in mapping.iteritems():
        if index in test_set:
            labels[index] = name
    labels = np.asarray(labels)[test_set]

    output_file = os.path.join(config.databases_path, filename)
    makedirsIfNeeded(output_file)
    np.savez(output_file, descriptors=descs, labels=labels)
    print "Saved to %s.npz" % output_file

