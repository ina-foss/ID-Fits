import os
import argparse
import subprocess

execfile(os.path.join(os.path.dirname(__file__), "fix_imports.py"))
import config



if __name__ == "__main__":

    # Parse commandline arguments
    parser = argparse.ArgumentParser(description="Generate models and descriptors for normalization analysis.")
    

    # Variables
    compute_descriptor_script = os.path.join(config.bin_path, "generate_benchmarks_descriptors.py")
    learn_model_script = os.path.join(config.bin_path, "learn_benchmarks_models.py")
    lfw_dataset = "lfw"
    lfw_lbf_51_dataset = "lfw_normalized_lbf_51_landmarks"
    lfw_lbf_68_dataset = "lfw_normalized_lbf_68_landmarks"
    lfw_csiro_dataset = "lfw_normalized_csiro"
    lfw_dataset_file = os.path.join(config.lfw_path, "%s.npy" % lfw_dataset)
    lfw_lbf_51_dataset_file = os.path.join(config.lfw_path, "%s.npy" % lfw_lbf_51_dataset)
    lfw_lbf_68_dataset_file = os.path.join(config.lfw_path, "%s.npy" % lfw_lbf_68_dataset)
    lfw_csiro_dataset_file = os.path.join(config.lfw_path, "%s.npy" % lfw_csiro_dataset)
    training_descriptors_path = os.path.join(config.lfw_benchmark_path, "training")


    # Train models
    print "\n\n------------- Training models -------------\n\n"

    subprocess.call(["python2", compute_descriptor_script, "lfw" ,"training", "ulbp", lfw_dataset_file])
    subprocess.call(["python2", learn_model_script, "lfw", os.path.join(training_descriptors_path, "ulbp_not_normalized_%s" % lfw_dataset), "pca", "200"])

    subprocess.call(["python2", compute_descriptor_script, "lfw" ,"training", "ulbp", lfw_lbf_51_dataset_file])
    subprocess.call(["python2", learn_model_script, "lfw", os.path.join(training_descriptors_path, "ulbp_not_normalized_%s" % lfw_lbf_51_dataset), "pca", "200"])

    subprocess.call(["python2", compute_descriptor_script, "lfw" ,"training", "ulbp", lfw_lbf_68_dataset_file])
    subprocess.call(["python2", learn_model_script, "lfw", os.path.join(training_descriptors_path, "ulbp_not_normalized_%s" % lfw_lbf_68_dataset), "pca", "200"])

    subprocess.call(["python2", compute_descriptor_script, "lfw" ,"training", "ulbp", lfw_csiro_dataset_file])
    subprocess.call(["python2", learn_model_script, "lfw", os.path.join(training_descriptors_path, "ulbp_not_normalized_%s" % lfw_csiro_dataset), "pca", "200"])


    # Compute test descriptors
    print "\n\n------------- Computing test descriptors -------------\n\n"
    subprocess.call(["python2", compute_descriptor_script, "lfw" ,"test", "-n", "ulbp_wpca", lfw_dataset_file])
    subprocess.call(["python2", compute_descriptor_script, "lfw" ,"test", "-n", "ulbp_wpca", lfw_lbf_51_dataset_file])
    subprocess.call(["python2", compute_descriptor_script, "lfw" ,"test", "-n", "ulbp_wpca", lfw_lbf_68_dataset_file])
    subprocess.call(["python2", compute_descriptor_script, "lfw" ,"test", "-n", "ulbp_wpca", lfw_csiro_dataset_file])

