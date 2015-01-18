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


#ifndef ALIGNMENT_H
#define ALIGNMENT_H

#include <vector>
#include "opencv2/opencv.hpp"
#include <SnoopFaceAlignmentLib/ForestBasedRegression.h>



class LandmarkDetector
{
public:
    virtual ~LandmarkDetector() {}
    virtual void detectLandmarks(const cv::Mat& img, const cv::Rect& bounding_box, cv::Mat& landmarks) const = 0;
    virtual void extractLandmarksForNormalization(const cv::Mat& landmarks, cv::Mat& normalization_landmarks) const = 0;
    virtual const cv::Mat& getReferenceShape() const;
    
protected:
    cv::Mat reference_shape_;
};



#if USE_CSIRO_ALIGNMENT == 1

#include "tracker/FaceTracker.hpp"

using namespace FACETRACKER;

class CSIROLandmarkDetector: public LandmarkDetector
{
public:
    CSIROLandmarkDetector(const std::string& tracker_model, const std::string& tracker_params);
    ~CSIROLandmarkDetector();
    void detectLandmarks(const cv::Mat& img, const cv::Rect& bounding_box, cv::Mat& landmarks) const;
    void extractLandmarksForNormalization(const cv::Mat& landmarks, cv::Mat& normalization_landmarks) const;
    
private:
    FaceTracker *tracker_;
    FaceTrackerParams *tracker_params_;
};

#endif


class LBFLandmarkDetector: public LandmarkDetector
{
public:
    LBFLandmarkDetector();
    virtual ~LBFLandmarkDetector();
    void loadModel(const std::string& filename);
    void detectLandmarks(const cv::Mat& img, const cv::Rect& bounding_box, cv::Mat& landmarks) const;
    void extractLandmarksForNormalization(const cv::Mat& landmarks, cv::Mat& normalization_landmarks) const;
    
protected:
    LocalBinaryFeatureAlignment* landmark_detector_;
};


class FaceNormalization
{
public:
    void setReferenceShape(const cv::Mat& reference_shape);
    float normalize(cv::Mat& img, const cv::Mat& landmarks) const;
    float computeSimilarityTransform(const float* src, float* transformation) const;
    
private:
    cv::Mat reference_shape_;
    int L_;
};

#endif
