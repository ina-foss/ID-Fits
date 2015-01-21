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



LFW, LFW_A, LFW_OWN = ["lfw", "lfwa", "lfwn"]

datasets = {
    LFW: "lfw",
    LFW_A: "lfw2",
    LFW_OWN: "lfwn"
}

lfw_path = os.path.join(config.data_path, "lfw")
pairs_file = os.path.join(lfw_path, "pairs.txt")
people_file = os.path.join(lfw_path, "people.txt")
mapping_file = os.path.join(lfw_path, "mapping.txt")



def loadPeopleFile(filename, mapping):
    f = open(filename, 'r')
    f.readline()
    
    people = []
    for line in f.readlines():
        name, nb = line.split()
        nb = int(nb)
        people += [mapping[(name, index)] for index in range(1, nb+1)]

    return people


def loadPairsFile(filename, mapping):
    f = open(filename, 'r')

    line = f.readline().split()
    if len(line) == 1:
        s = 1
        n = int(line[0])
    else:
        s = int(line[0])
        n = int(line[1])

    pairs = [({}, {}) for i in range(s)]

    for matching_pairs, mismatching_pairs in pairs:
        for i in range(n):
            name, index1, index2 = f.readline().split()
            index1 = int(index1)
            index2 = int(index2)
            if mapping[(name, index1)] not in matching_pairs:
                matching_pairs[mapping[(name, index1)]] = []
            matching_pairs[mapping[(name, index1)]].append(mapping[(name, index2)])

        for i in range(n):
            name1, index1, name2, index2 = f.readline().split()
            index1 = int(index1)
            index2 = int(index2)
            if mapping[(name1, index1)] not in mismatching_pairs:
                mismatching_pairs[mapping[(name1, index1)]] = []
            mismatching_pairs[mapping[(name1, index1)]].append(mapping[(name2, index2)])

    if s > 1:
        return pairs
    else:
        return pairs[0]



def loadData(dataset, preprocess=False):
    filename = os.path.join(lfw_path, dataset+".npy")
    if not os.path.exists(filename):
        raise Exception("Dataset %s unknown"%dataset)
    
    data = np.load(filename)

    if preprocess:
        return preprocessData(data)
    else:
        return data

        

def preprocessData(raw_data):
    return raw_data[:, 49:-49, 84:-84]



def loadSetsGroundTruth():
    mapping = loadMapping()
    return loadPairsFile(pairs_file, mapping)



def loadTestSets():

    mapping = loadMapping()
    sets = []

    with open(people_file, 'r') as f:
        sets_number = int(f.readline())
        
        for _ in range(sets_number):
            
            sets.append([])
            set_size = int(f.readline())

            for _ in range(set_size):
                name, number = f.readline().split()
                for index in range(1, int(number)+1):
                    sets[-1].append(mapping[(name, int(index))])

    return sets



def loadTrainingSets():

    sets = loadTestSets()
    training_sets = []
    
    for i in range(len(sets)):
        training_sets.append([])
        for k in range(len(sets)-1):
            training_sets[-1] += sets[(i+k+1) % len(sets)]

    return training_sets



def loadMapping():
    mapping = dict()
    
    with open(mapping_file, "r") as f:
        f.readline()
        
        for line in f.readlines():
            name, index, global_index = line.split()
            index, global_index = int(index), int(global_index)
            mapping[(name,index)] = global_index

    return mapping



def loadDevData(subset="train", load_pairs=True, filename=None):
    if filename:
        if os.path.exists(filename):
            people_file = filename
            pairs_file = None
        else:
            raise ValueError("Unknown file %s"%filename)
    else:
        if subset == "train":
            people_file = os.path.join(lfw_path, "peopleDevTrain.txt")
            pairs_file = os.path.join(lfw_path, "pairsDevTrain.txt")
        elif subset == "test":
            people_file = os.path.join(lfw_path, "peopleDevTest.txt")
            pairs_file = os.path.join(lfw_path, "pairsDevTest.txt")
        else:
            raise ValueError("Unknown subset value")
    
    mapping = loadMapping()

    if load_pairs and pairs_file:
        return loadPeopleFile(people_file, mapping), loadPairsFile(pairs_file, mapping)
    else:
        return loadPeopleFile(people_file, mapping)



def loadTrainingDataLabels(training_set, min_nb_samples_per_class = 10):
    mapping = loadMapping()
    samples_per_classes = {}
    classes, _ = zip(*mapping)

    classes_count = {}
    for name, index in mapping:
        if mapping[(name, index)] in training_set:
            if name not in classes_count:
                classes_count[name] = 0
            classes_count[name] += 1

    kept_classes = []
    for name, count in classes_count.iteritems():
        if count >= min_nb_samples_per_class:
            kept_classes.append(name)

    classes_id = dict(zip(kept_classes, range(len(kept_classes))))
    descs_indexes = []
    y = []
    for name, index in mapping:
        if name in kept_classes and mapping[(name, index)] in training_set:
            new_index = training_set.index(mapping[(name, index)])
            descs_indexes.append(new_index)
            y.append(classes_id[name])

    return descs_indexes, np.array(y, dtype=np.int)



def reindex(indexes, ground_truth_mapping):
    result_mapping = []
    for mapping in ground_truth_mapping:
        new_mapping = {}
        for k in mapping.keys():
            l = mapping[k]
            new_mapping[indexes.index(k)] = []
            for e in l:
                new_mapping[indexes.index(k)].append(indexes.index(e))
        result_mapping.append(new_mapping)
    return tuple(result_mapping)
