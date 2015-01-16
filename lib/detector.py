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


import sys
import os
import numpy as np

from datasets import _300w
from cpp_wrapper.face_detection import FaceDetector



def getFace(faces, ground_truth):

    for face in faces:

        intersection_width = min(face[2], ground_truth[2]) - max(face[0], ground_truth[0])
        intersection_height = min(face[3], ground_truth[3]) - max(face[1], ground_truth[1])

        if intersection_height <= 0 or intersection_width <= 0:
            continue

        intersection_surface = float(intersection_width * intersection_height)
        union_surface = float((face[2]-face[0])*(face[3]-face[1]) + (ground_truth[2]-ground_truth[0])*(ground_truth[3]-ground_truth[1]) - intersection_surface)

        if intersection_surface / union_surface > 0.5:
            return face, intersection_surface / union_surface

        return None, intersection_surface / union_surface

    return None, None



if __name__ == "__main__":

    if sys.argv[1] == "detect":

        detector = FaceDetector()

        #images, landmarks, bounding_boxes = dataset300w.loadCompleteDataset()

        discrete_score = 0.0
        continuous_score = 0.0
        total_images_number = 0


        for dataset in _300w.subdatasets:

            print "Starting detecting faces in %s" % dataset

            images, landmarks, bounding_boxes = _300w.load(dataset.replace('/', '_'))

            dataset_discrete_score = 0.0
            dataset_continuous_score = 0.0

            detected_indexes = []
            detected_faces = []

            for i, img, bounding_box in zip(range(len(images)), images, bounding_boxes):

                total_images_number += 1

                faces = detector.detectFaces(img)
                face, score = getFace(faces, bounding_box)

                if face != None:
                    dataset_discrete_score += 1
                    dataset_continuous_score += score 
                    detected_indexes.append(i)
                    detected_faces.append(face)

                if (i+1) % 50 == 0:
                    print "%s: %0.2f%% completed, current scores: discrete=%0.4f continuous=%0.4f ..." % (dataset.replace("/", " "), (float(i+1) / len(images))*100, dataset_discrete_score/(i+1), dataset_continuous_score/(i+1))

            discrete_score += dataset_discrete_score
            continuous_score += dataset_continuous_score
            np.savez(os.path.join("300-w", dataset.replace("/","_")+"_opencv_detector_bounding_boxes"), indexes=detected_indexes, faces=detected_faces)

            print "%s finished, final scores: discrete=%0.4f continuous=%0.4f \n\n" % (dataset.replace("/"," "), dataset_discrete_score/len(images), dataset_continuous_score/len(images))


        discrete_score /= total_images_number
        continuous_score /= total_images_number

        print "Discrete score: %0.4f" % discrete_score
        print "Continuous score: %0.4f" % continuous_score


    elif sys.argv[1] == "format":

        for dataset in _300w.datasets:

            npzfile = np.load(os.path.join(config._300w_path, +"%s_opencv_detector_bounding_boxes.npz" % dataset.replace('/', '_')))
            indexes = npzfile["indexes"]
            faces = npzfile["faces"]

            images, landmarks, bounding_boxes = _300w.load(dataset.replace('/', '_'))

            images = np.delete(images, filter(lambda x: x not in indexes, range(len(images))), axis=0)
            landmarks = np.delete(landmarks, filter(lambda x: x not in indexes, range(len(landmarks))), axis=0)
            """
            faces[:,2] = faces[:,0] + faces[:,2]
            faces[:,3] = faces[:,1] + faces[:,3]
            """

            np.save(os.path.join(config._300w_path, "%s_opencv_detector" % dataset.replace('/', '_')), (images, landmarks, faces))

