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
import shutil
import argparse
import tempfile
import subprocess

execfile(os.path.join(os.path.dirname(__file__), "fix_imports.py"))
import config



if __name__ == "__main__":

    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="Generate models and descriptors for demos.")
    parser.add_argument("--keep-temp", action="store_true", help="keep the temporary descriptors files")
    args = parser.parse_args()
    

    # Variables
    compute_descriptor_script = os.path.join(config.bin_path, "compute_descriptors.py")
    compute_pca_script = os.path.join(config.bin_path, "compute_pca.py")
    compute_lda_script = os.path.join(config.bin_path, "compute_lda.py")
    compute_jb_script = os.path.join(config.bin_path, "compute_jb.py")
    create_database_script = os.path.join(config.bin_path, "create_database.py")
    
    dataset = "lfw_normalized_lbf_68_landmarks"
    dataset_file = os.path.join(config.lfw_path, "%s.npy" % dataset)
    
    if args.keep_temp:
        training_descriptors_path = config.descriptors_path
    else:
        training_descriptors_path = tempfile.mktemp()

    ulbp_descriptors_file = os.path.join(training_descriptors_path, "ulbp_not_normalized_%s.npy" % dataset)
    ulbp_pca_descriptors_file = os.path.join(training_descriptors_path, "ulbp_pca_not_normalized_%s.npy" % dataset)
    ulbp_wpca_descriptors_file = os.path.join(config.descriptors_path, "ulbp_wpca_%s.npy" % dataset)
    ulbp_pca_lda_descriptors_file = os.path.join(config.descriptors_path, "ulbp_pca_lda_%s.npy" % dataset)
    ulbp_pca_jb_descriptors_file = os.path.join(config.descriptors_path, "ulbp_pca_jb_%s.npy" % dataset)
    pca_file = os.path.join(config.models_path, "PCA_%s.txt" % dataset)
    lda_file = os.path.join(config.models_path, "LDA_%s.txt" % dataset)
    jb_file = os.path.join(config.models_path, "JB_%s.txt" % dataset)
    
    
    # Train models
    print "\n\n------------- Training models -------------\n\n"
    subprocess.check_call(["python2", compute_descriptor_script, "ulbp", dataset_file, ulbp_descriptors_file])
    subprocess.check_call(["python2", compute_pca_script, "-d 200", "-s 2000", "-o %s" % pca_file, ulbp_descriptors_file])
    subprocess.check_call(["python2", compute_descriptor_script, "-p %s" % pca_file, "ulbp_pca", dataset_file, ulbp_pca_descriptors_file])
    subprocess.check_call(["python2", compute_lda_script, "-d 50", "-o %s" % lda_file, ulbp_pca_descriptors_file])
    subprocess.check_call(["python2", compute_jb_script, "-o %s" % jb_file, ulbp_pca_descriptors_file])
    if not args.keep_temp:
        shutil.rmtree(training_descriptors_path)


    # Compute test descriptors
    print "\n\n------------- Computing test descriptors -------------\n\n"
    subprocess.check_call(["python2", compute_descriptor_script, "-n", "-p %s" % pca_file, "ulbp_wpca", dataset_file, ulbp_wpca_descriptors_file])
    subprocess.check_call(["python2", compute_descriptor_script, "-n", "-p %s" % pca_file, "-l %s" % lda_file, "ulbp_pca_lda", dataset_file, ulbp_pca_lda_descriptors_file])
    subprocess.check_call(["python2", compute_descriptor_script, "-p %s" % pca_file, "-j %s" % jb_file, "ulbp_pca_jb", dataset_file, ulbp_pca_jb_descriptors_file])


    # Compute databases for retrieval
    print "\n\n------------- Computing databases -------------\n\n"
    subprocess.check_call(["python2", create_database_script, ulbp_wpca_descriptors_file])
    subprocess.check_call(["python2", create_database_script, ulbp_pca_lda_descriptors_file])
    subprocess.check_call(["python2", create_database_script, ulbp_pca_jb_descriptors_file])


