// ID-Fits
// Copyright (c) 2015 Institut National de l'Audiovisuel, INA, All rights reserved.
// 
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
// 
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public
// License along with this library.


#ifndef FACE_DETECTION_H
#define FACE_DETECTION_H

#include <opencv2/opencv.hpp>


class FaceDetector
{
public:
    FaceDetector(const std::string& model_file): classifier_(model_file)
    {
    }
    
    virtual ~FaceDetector()
    {
    }
    
    virtual void detectFaces(const cv::Mat& img, std::vector<cv::Rect>& faces)
    {
        classifier_.detectMultiScale(img, faces, 1.3, 5, 0, cv::Size(80,80));
    }
    
protected:
    cv::CascadeClassifier classifier_;
};


class HighRecallFaceDetector: public FaceDetector
{
public:
    HighRecallFaceDetector(const std::string& model_file): FaceDetector(model_file)
    {
    }

    void detectFaces(const cv::Mat& img, std::vector<cv::Rect>& faces)
    {
        classifier_.detectMultiScale(img, faces, 1.1, 1, 0, cv::Size(80,80));
    }
};


#endif
