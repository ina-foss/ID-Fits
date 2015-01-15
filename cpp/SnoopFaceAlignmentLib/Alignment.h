#ifndef ALIGNMENT_H
#define ALIGNMENT_H

#include <vector>
#include "opencv2/opencv.hpp"
#include "tracker/FaceTracker.hpp"
#include <SnoopFaceAlignmentLib/ForestBasedRegression.h>

using namespace FACETRACKER;


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
    void normalize(cv::Mat& img, const cv::Mat& landmarks) const;
    void computeSimilarityTransform(const double* src, double* transformation) const;
    
private:
    cv::Mat reference_shape_;
    int L_;
};

#endif
