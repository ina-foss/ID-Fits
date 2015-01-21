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
import glob
import cv2
import numpy as np

from tools import readDataFromFile, writeDataToFile
from landmarks_file import readPtsLandmarkFile


data_directory = "/home/tlorieul/Data/"



class BioId:
    def __init__(self):
        self.base_path = os.path.join(data_directory, "bioid")
        self.images_path = os.path.join(self.base_path, "images")
        self.landmarks_path = os.path.join(self.base_path, "landmarks")
        self.landmarks_number = 20

    def loadImages(self, reset=False, cleaned=True):
        # Cases needs to rebuild the dataset in Numpy's file format
        if reset:
            return self._loadImagesFromScratch()
        elif cleaned:
            return readDataFromFile("images_cleaned.npy", directory="bioid", binary="True").astype(np.uint8)
        else:
            return readDataFromFile("images.npy", directory="bioid", binary="True").astype(np.uint8)

    def loadGroundTruth(self, reset=False, cleaned=True):
        if reset:
            return self._loadGroundTruthFromScratch()
        elif cleaned:
            return readDataFromFile("landmarks_cleaned.npy", directory="bioid", binary="True")
        else:
            return readDataFromFile("landmarks.npy", directory="bioid", binary="True")

    def loadBoundingBoxes(self):
        bounding_boxes = readDataFromFile("faces_bounding_box.npy", directory="bioid", binary="True")
        return [(bounding_box[0], bounding_box[1], bounding_box[0]+bounding_box[2], bounding_box[1]+bounding_box[3]) for bounding_box in bounding_boxes]

    def clean(self, images, ground_truth):
        not_detected = []
        detected_faces = []
        face_detector = cv2.CascadeClassifier("models/haarcascade_frontalface_alt.xml")
        for index, img in enumerate(images):
            faces = face_detector.detectMultiScale(img.astype(np.uint8), 1.3, 5)
            if len(faces) == 0 or ground_truth[index, 19, 1] > 250.0:
                not_detected.append(index)
            else:
                detected_faces.append(faces[0])

        detected_faces = np.array(detected_faces)
        images = np.delete(images, not_detected, axis=0)
        ground_truth = np.delete(ground_truth, not_detected, axis=0)

        print "%d images removed because the face could not be detected properly"%len(not_detected)

        np.save("bioid/images_cleaned", images)
        np.save("bioid/landmarks_cleaned", ground_truth)
        np.save("bioid/faces_bounding_box", detected_faces)
        

    def _loadImagesFromScratch(self):
        imgs_array = np.empty((1521, 286, 384))
        for img_file in glob.glob(os.path.join(self.images_path, "*.pgm")):
            index = int(os.path.basename(img_file)[6:10])
            imgs_array[index] = cv2.imread(img_file, cv2.CV_LOAD_IMAGE_GRAYSCALE)

        writeDataToFile(imgs_array, "images", "bioid", binary=True)
        return imgs_array
        
    def _loadGroundTruthFromScratch(self):
        landmarks_array = np.empty((1521, 20, 2))
        for f in os.listdir(self.landmarks_path):
            index = int(os.path.basename(f)[6:10])
            landmarks_array[index] = readPtsLandmarkFile(os.path.join(self.landmarks_path, f), self.landmarks_number)

        writeDataToFile(landmarks_array, "landmarks", "bioid", binary=True)
        return landmarks_array
