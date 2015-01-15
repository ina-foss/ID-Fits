#ifndef PIPELINE_H
#define PIPELINE_H

#include <SnoopFaceDescLib/Alignment.h>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;


class Pipeline 
{
public:
    Pipeline();
    
    void cropImage(Mat& mat);
    
    bool checkMatching(const Mat* p_vect1, const Mat* p_vect2, double threshold) const;
    double cosineSimilarity(const Mat* p_hist1, const Mat* p_hist2);
    
    vector<Mat> loadFeatureVectors(const string& filename = "feature_vectors.txt");
    void saveFeatureVectors(const vector<Mat>& vectors, const string& filename = "feature_vectors.txt");
    
    vector<Mat> addTranslationNoise(int N, const Mat* data, double sigma);
    /*void addScaleNoise(Mat* data);
    void addRotationNoise(Mat* data);
    void addNoise(Mat* data);*/
    
private:
    Rect ROI;
};

#endif