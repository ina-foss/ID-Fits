import os
import argparse
import scipy.io
import cv2
import numpy as np

execfile("fix_imports.py")
from datasets import _300w
from datasets.landmarks_file import readPtsLandmarkFile



def readBoundingBoxesFile(dataset, filename):
    mat = scipy.io.loadmat(filename)
        
    bounding_boxes = {}
    if dataset != "ibug":
        for image in mat["bounding_boxes"][0]: 
            image_name = str(image[0,0][0][0])
            detected_bounding_box = image[0,0][1][0]
            true_bounding_box = image[0,0][2][0]
            bounding_boxes[image_name] = (tuple(detected_bounding_box), tuple(true_bounding_box))

        return bounding_boxes
    else: # Hack for ibug !!! (maybe bounding_box of test images ?)
        for image in mat["bounding_boxes"][0,:135]: 
            image_name = str(image[0,0][0][0])
            detected_bounding_box = image[0,0][1][0]
            true_bounding_box = image[0,0][2][0]
            bounding_boxes[image_name] = (tuple(detected_bounding_box), tuple(true_bounding_box))

        return bounding_boxes


def loadDatasetsFromRawData(directory):
    datasets = {}

    data = [os.path.join(directory, dataset) for dataset in _300w.subdatasets]
    bounding_boxes = [os.path.join(directory, "bounding_boxes",  "bounding_boxes_%s.mat" % dataset.replace('/', '_')) for dataset in _300w.subdatasets]
    for name, path_to_data, bounding_boxes_file in zip(_300w.datasets, data, bounding_boxes):
        images = []
        ground_truth = []
        detected_bounding_boxes = []
        true_bounding_boxes = []

        for image_file, (detected_bounding_box, true_bounding_box) in readBoundingBoxesFile(name, bounding_boxes_file).iteritems():
            images.append(cv2.imread(os.path.join(path_to_data, image_file), cv2.CV_LOAD_IMAGE_GRAYSCALE))
            ground_truth.append(readPtsLandmarkFile(os.path.join(path_to_data, image_file.split(".")[0]+".pts"), _300w.landmarks_number))
            detected_bounding_boxes.append(detected_bounding_box)
            true_bounding_boxes.append(true_bounding_box)

        datasets[name] = (np.array(images), np.array(ground_truth), np.array(detected_bounding_boxes), np.array(true_bounding_boxes))

    return datasets



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Format 300-W dataset.")
    args = parser.parse_args()


    datasets = loadDatasetsFromRawData()

    for name, data in datasets.iteritems():
        np.save(os.path.join("300-w", name.replace("/","_")), data)
