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
