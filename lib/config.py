import os


base_path = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))

bin_path = os.path.join(base_path, "bin")

models_path = os.path.join(base_path, "models")

data_path = os.path.join(base_path, "data")
lfw_path = os.path.join(data_path, "lfw")
bioid_path = os.path.join(data_path, "bioid")
_300w_path = os.path.join(data_path, "300-w")

benchmarks_path = os.path.join(base_path, "benchmarks")
lfw_benchmark_path = os.path.join(benchmarks_path, "lfw")
blufr_benchmark_path = os.path.join(benchmarks_path, "blufr")
_300w_benchmark_path = os.path.join(benchmarks_path, "300w")

descriptors_path = os.path.join(base_path, "descriptors")


