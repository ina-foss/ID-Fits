import os
import time
import argparse
import numpy as np

execfile("fix_imports.py")
import config
from datasets import lfw
from alignment import normalizeData



if __name__ == "__main__":

    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="Normalize the dataset")
    parser.add_argument("data", default="lfw", help="data to normalize")
    parser.add_argument("-m", dest="landmark_detector", default="lbf", choices=["lbf", "csiro"], help="alignment method to use")
    parser.add_argument("-n", dest="landmarks_number", type=int, default=68, choices=[51, 68], help="number of landmarks to consider for LBF method")
    parser.add_argument("-o", dest="output", default=None, help="output file")
    args = parser.parse_args()
    
    
    # Arguments
    landmark_detector = args.landmark_detector
    landmarks_number = args.landmarks_number
    filename = args.output
    if filename is None:
        filename = os.path.join(config.data_path, "lfw", "lfw_normalized_%s" % landmark_detector)
        if landmark_detector == "lbf":
            filename += "_%i_landmarks" % landmarks_number


    # Load data
    print "Loading data..."
    if args.data == "lfw":
        data = lfw.loadData(dataset="lfw", preprocess=False)
    else:
        data_file = args.data
        extension = os.path.splitext(data_file)[1]
        if extension == ".npy":
            data = np.load(data_file)
        elif extension == ".npz":
            npz = np.load(data_file)
            data = npz["images"]
        else:
            raise Exception("Cannot open file %s, unknown extension %s"%(data_file, extension))

    
    # Normalization
    print "Normalizing dataset with %s aligner..."%landmark_detector
    t = time.clock()
    normalized_data = normalizeData(data, landmark_detector_name=landmark_detector, landmarks_number=landmarks_number)
    print "Done in %.2f seconds"%(time.clock()-t)
    

    # Save results
    print "Saving results to %s"%filename
    if args.data != "lfw" and extension == ".npz":
        np.savez(filename, labels=npz["labels"], images=normalized_data)
    else:
        np.save(filename, normalized_data)


