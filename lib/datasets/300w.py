import os
import numpy as np

import config


subdatasets = ["afw", "lfpw/trainset", "helen/trainset", "lfpw/testset", "helen/testset", "ibug"]
landmarks_number = 68



def load(dataset, detector=None, contour_landmarks=True):
    if detector is None:
        data = np.load(os.path.join(config._300w_path, dataset+".npy"))[:3]
    elif detector == "perfect":
        data = np.load(os.path.join(config._300w_path, dataset+".npy"))
        data[2] = np.array(data[3])
        data = np.delete(data, 3, 0)
    else:
        data = np.load(os.path.join(config._300w_path, dataset+"_"+detector+"_detector.npy"))
    data[2] = [tuple(face) for face in data[2]]

    if not contour_landmarks:
        for i, shape in enumerate(data[1]):
            data[1][i] = shape[17:]

    return tuple(data)



def loadTrainingSet(detector=None, contour_landmarks=True):
    datasets = ["afw", "helen_trainset", "lfpw_trainset"]
    data = [load(dataset, detector=detector, contour_landmarks=contour_landmarks) for dataset in datasets]

    return tuple(np.hstack(tuple([dataset[i] for dataset in data])) for i in range(3))



def loadTestSet(subset="full", detector=None, contour_landmarks=True):
    datasets = ["ibug", "helen_testset", "lfpw_testset"]

    if subset == "challenging":
        datasets = datasets[:1]
    elif subset == "common":
        datasets = datasets[1:]
    elif subset != "full":
        raise Exception("Unknown subset")

    data = [load(dataset, detector=detector, contour_landmarks=contour_landmarks) for dataset in datasets]
    return tuple(np.hstack(tuple([dataset[i] for dataset in data])) for i in range(3))



def loadCompleteDataset(detector=None, contour_landmarks=True):
    datasets = [load(dataset_name.replace('/', '_'), detector=detector, contour_landmarks=contour_landmarks) for dataset_name in subdatasets]
    return tuple(np.hstack(tuple([dataset[i] for dataset in datasets])) for i in range(3))
