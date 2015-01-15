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
