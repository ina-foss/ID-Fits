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