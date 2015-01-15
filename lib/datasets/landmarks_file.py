import numpy as np



def readPtsLandmarkFile(filename, landmarks_number):
    f = open(filename)

    # Skip first 3 lines
    for i in range(3):
        f.readline()

    # Read landmarks position
    landmarks = np.empty((landmarks_number, 2), dtype=np.float)
    for i in range(landmarks_number):
        landmarks[i] = np.array([float(x) for x in f.readline().split()])
            
    return landmarks
