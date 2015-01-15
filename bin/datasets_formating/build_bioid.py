import argparse

import fix_imports
from datasets.bioid import BioId



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Format BioId dataset.")
    parser.add_argument("command", choices=["build", "clean"], help="Command to execute.")
    args = parser.parse_args()

    bioid = BioId()

    if args.command == "build":
        bioid._loadImagesFromScratch()
        bioid._loadGroundTruthFromScratch()

    elif args.command == "clean":
        images = bioid.loadImages(cleaned=False)
        ground_truth = bioid.loadGroundTruth(cleaned=False)
        bioid.clean(images, ground_truth)
