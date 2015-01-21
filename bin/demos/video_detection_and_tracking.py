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
import time
import argparse
import cv2
import numpy as np
from matplotlib import pyplot

import fix_imports

import config
from database import loadDatabase
from search import *
from filtering import Filter
from face_detector import FaceDetectorAndTracker

from cpp_wrapper.alignment import FaceNormalization



def getLinesFromLandmarks(shape):
    if shape.shape[0] == 51:
        shape = np.vstack((np.zeros((17, 2)), shape))
    
    lines = zip(shape[:17], shape[1:17])
    lines += zip(shape[17:22], shape[18:22]) + zip(shape[22:27], shape[23:27])
    lines += zip(shape[27:31], shape[28:31]) + zip(shape[31:36], shape[32:36])
    lines += zip(shape[36:42], shape[37:42]) + [(shape[41], shape[36])]
    lines += zip(shape[42:48], shape[43:48]) + [(shape[47], shape[42])]
    lines += zip(shape[48:60], shape[49:60]) + [(shape[48], shape[59])]
    lines += zip(shape[60:], shape[61:]) + [(shape[60], shape[67])]
    return lines


def displayShape(img, shape):
    for landmark in shape:
        cv2.circle(img, tuple(landmark.astype(np.int)), 1, (0,255,0), -1)
        
    for line in getLinesFromLandmarks(shape):
        cv2.line(img, tuple(line[0].astype(np.int)), tuple(line[1].astype(np.int)), (0,255,0))



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Detects faces in videos and tracks them.")
    parser.add_argument("input_video_file", help="video to process")
    parser.add_argument("-o", dest="output_file", help="where to write processed file")
    parser.add_argument("-f", dest="filter", action="store_true", help="apply filtering")
    parser.add_argument("-m", dest="max_speed", action="store_true", help="process the video at maximum speed")
    parser.add_argument("-s", dest="silent", action="store_true", help="minimum output")
    args = parser.parse_args()
    
    video_file = args.input_video_file
    video = cv2.VideoCapture(video_file)
    if not video.isOpened():
        raise Exception("Cannot read video %s"%video_file)

    if args.output_file:
        output_file = args.output_file
        video_writer = cv2.VideoWriter(
            output_file,
            int(video.get(6)),
            video.get(5),
            (int(video.get(3)), int(video.get(4)))
        )
        if not video_writer.isOpened():
            raise Exception("Cannot write to file %s"%output_file)
        write_output = True
    else:
        write_output = False
    
    
    detector_and_tracker = FaceDetectorAndTracker()
    filters = []

    if not args.silent:
        cv2.namedWindow("Alignment demo")

    fps = video.get(5)
    print "Video running at %0.2f fps" % fps

    if args.max_speed:
        print "Processing video at maximum speed"
        wait = 1
    else:
        wait = 25

    n = 0
    face_detector_freq = int(fps / 2)
    shapes = []
    t = time.time()
    
    while True:
        video.grab()
        isVideoStillReading, frame = video.retrieve()
        
        if not isVideoStillReading:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if n % face_detector_freq == 0:
            shapes = detector_and_tracker.detect(image)

            if args.filter:
                filters = []
                for i, shape in enumerate(shapes):
                    filters.append(Filter(n=3))
                    shapes[i] = filters[-1].filter(shape)

        elif len(shapes) > 0:
            shapes = detector_and_tracker.track(image, shapes)
            
            if args.filter:
                for i, shape in enumerate(shapes):
                    shapes[i] = filters[i].filter(shape)
                
        else:
            filters = []

        
        if not args.silent or (args.silent and write_output):
            for shape in shapes:
                displayShape(frame, shape)

        if not args.silent:
            cv2.imshow("Alignment demo", frame)
            if cv2.waitKey(wait) >= 0:
                break

        n += 1

        if write_output:
            video_writer.write(frame)


if args.max_speed:
    print "Speed: %0.2f fps" % (n / (time.time()-t))
