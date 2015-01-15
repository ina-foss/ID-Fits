import os
import argparse
import numpy as np

execfile("fix_imports.py")
import config
from datasets import lfw
from learning.lda import computeLDA
from utils.file_manager import pickleSave



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Computes LDAs")
    parser.add_argument("descriptors_file", help="descriptors on which to compute LDA (e.g. ulbp_pca_not_normalized_lfwa.npy)")
    args = parser.parse_args()
    
    filename = args.descriptors_file
    print "Using %s descriptors"%filename
    if filename.find("_not_normalized_") < 0:
        raise Exception("Need to use a non normalized descriptor")
    
    basename = os.path.basename(os.path.splitext(filename)[0]).replace("_not_normalized_", "_")
    pca_dir = os.path.join(config.models_path, "PCA", basename)
    lda_dir = os.path.join(config.models_path,"LDA", basename)
    print "Using PCA files from %s"%pca_dir
    
    sets = lfw.loadSetsPeople()
    data = np.load(filename)
    ldas = computeLDA(data, sets, dim=50, pca_dir=pca_dir)
    
    if not os.path.exists(lda_dir):
        os.makedirs(lda_dir)
    
    for i, lda in enumerate(ldas):
        set_filename = os.path.join(lda_dir, "set_%d.txt"%(i+1))
        pickleSave(set_filename, lda)
    print "Results saved in directory %s"%lda_dir
