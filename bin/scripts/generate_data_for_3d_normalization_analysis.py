import os
import argparse
import subprocess

execfile(os.path.join(os.path.dirname(__file__), "fix_imports.py"))
import config



if __name__ == "__main__":

    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="Generate models and descriptors for 3D normalization analysis.")
    

    # Variables
    compute_descriptor_script = os.path.join(config.bin_path, "generate_benchmarks_descriptors.py")
    learn_model_script = os.path.join(config.bin_path, "learn_benchmarks_models.py")
    lfw3d_dataset = "lfw3d"
    lfw3d_dataset_file = os.path.join(config.lfw_path, "%s.npy" % lfw3d_dataset)
    training_descriptors_path = os.path.join(config.lfw_benchmark_path, "training")


    # Train models
    print "\n\n------------- Training models -------------\n\n"

    subprocess.call(["python2", compute_descriptor_script, "lfw" ,"training", "-c", "ulbp", lfw3d_dataset_file])
    subprocess.call(["python2", learn_model_script, "lfw", os.path.join(training_descriptors_path, "ulbp_not_normalized_%s" % lfw3d_dataset), "pca", "200"])
    subprocess.call(["python2", compute_descriptor_script, "lfw" ,"training", "-c", "ulbp_pca", lfw3d_dataset_file])
    subprocess.call(["python2", learn_model_script, "lfw", os.path.join(training_descriptors_path, "ulbp_pca_not_normalized_%s" % lfw3d_dataset), "lda", "50"])
    subprocess.call(["python2", learn_model_script, "lfw", os.path.join(training_descriptors_path, "ulbp_pca_not_normalized_%s" % lfw3d_dataset), "jb", "50"])


    # Compute test descriptors
    print "\n\n------------- Computing test descriptors -------------\n\n"
    subprocess.call(["python2", compute_descriptor_script, "lfw" ,"test", "-cn", "ulbp_wpca", lfw3d_dataset_file])
    subprocess.call(["python2", compute_descriptor_script, "lfw" ,"test", "-cn", "ulbp_pca_lda", lfw3d_dataset_file])
    subprocess.call(["python2", compute_descriptor_script, "lfw" ,"test", "-c", "ulbp_pca_jb", lfw3d_dataset_file])


