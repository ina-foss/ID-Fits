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
import numpy as np
import config



blufr_file = os.path.join(config.lfw_path, "blufr.npz")



def loadLabels():
    data = np.load(blufr_file)
    reindex = np.empty((data["indexes"].shape[0]), dtype=data["indexes"].dtype)
    for i, index in enumerate(data["indexes"]):
        reindex[index] = i
    return data["labels"][reindex]


def loadTrainingSets():
    data = np.load(blufr_file)
    return [data["indexes"][training_set-1] for training_set in data["train"]]



def loadTestSets():
    data = np.load(blufr_file)
    return [data["indexes"][test_set-1] for test_set in data["test"]]



def globalToLocalIndex(global_index, index):
    local_index = np.empty((global_index.shape[0]), dtype=global_index.dtype)
    for i, j in enumerate(global_index):
        local_index[i] = np.where(index == j)[0]
    return local_index

        

def loadTestDescriptors(descriptor_type):
    blufr = np.load(blufr_file)
    sets_number = len(blufr["test"])

    test_sets = loadTestSets()
    descriptors = []
    gallery_indexes = []
    probe_indexes = []

    for i in range(sets_number):
        descriptors.append(np.load(os.path.join(config.blufr_benchmark_path, "test", descriptor_type, "set_%d.npy" % (i+1))))
        gallery_indexes.append(globalToLocalIndex(blufr["indexes"][blufr["gallery"][i]-1], test_sets[i]))
        probe_indexes.append(globalToLocalIndex(blufr["indexes"][blufr["probe_images"][i]-1], test_sets[i]))

    return descriptors, gallery_indexes, probe_indexes



def computeOpenSetIdentificationStatsForThreshold(descriptors, labels, gallery_indexes, probe_indexes, threshold, similarity_func=np.inner, rank=1):
    di = fa = 0
    g = n = 0

    gallery_labels = labels[gallery_indexes]

    print threshold
    import sys
    sys.stdout.flush()

    for probe in probe_indexes:
        
        if labels[probe] in gallery_labels:
            
            g += 1
            #min_similarities = []
            similarity_matrix = map(lambda x: similarity_func(descriptors[probe], descriptors[x]), gallery_indexes)

            #np.empty((len(gallery_labels)), dtype=np.float32)
            """
            for i, image in enumerate(gallery_indexes):
                #distance_matrix.append((labels[image], distance_func(descriptors[probe], descriptors[image]))
                similarity_matrix[i] = similarity_func(descriptors[probe], descriptors[image])
            """
            """
                if distance <= threshold and (len(min_distances) == 0 or (distance < min_distances[-1][1])):
                    min_distances.append((labels[image], distance))
                    min_distances = sorted(min_distances, key=lambda x: x[1])[:-1]

            if len(min_distances) > 0 and labels[probe] in zip(*min_distances)[0]:
                di += 1
            """
            max_index = np.argmax(similarity_matrix)
            max_similarities = similarity_matrix[max_index]
            if max_similarities >= threshold and labels[probe] == labels[max_index]:
                di += 1
                
        else:
            n += 1
            for image in gallery_indexes:
                similarity = similarity_func(descriptors[probe], descriptors[image])
                if similarity >= threshold:
                    fa += 1
                    break

    return fa / float(n), di / float(g)



def computeOpenSetIdentificationROC(descriptors, labels, gallery, probe_images, similarity_func=np.inner, rank=1):
    thresholds = np.linspace(-0.2, 0.8, 50)
    stats = []

    for threshold in thresholds:
        stats.append(computeOpenSetIdentificationStatsForThreshold(descriptors, labels, gallery, probe_images, threshold, similarity_func, rank))

    return zip(*stats)



def computeOpenSetIdentificationMeanROC(descriptors, labels, sets_gallery, sets_probe_images, similarity_func=np.inner, rank=1):
    fprs = []
    tprs = []
    for gallery, probe_images in zip(sets_gallery, sets_probe_images):
        fpr, tpr = computeOpenSetIdentificationROC(descriptors, labels, gallery, probe_images, similarity_func, rank)
        fprs.append(fpr)
        tprs.append(tpr)

    return np.mean(np.asarray(fprs), axis=0), np.mean(np.asarray(tprs), axis=0)
