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


#include <SnoopFaceDescLib/Pipeline.h>
#include <fstream>
#include "opencv2/imgproc/imgproc.hpp"



Pipeline::Pipeline(): ROI(85-1, 50-1, 80+2, 150+2)
{
}

void Pipeline::cropImage(Mat& img)
{
    Mat image(img, ROI);
    image.copyTo(img);
}


bool Pipeline::checkMatching(const Mat* p_vect1, const Mat* p_vect2, double threshold) const
{
    return (cosineSimilarity(p_vect1, p_vect2) > threshold);
}

double Pipeline::cosineSimilarity(const Mat* p_hist1, const Mat* p_hist2)
{
    return p_hist1->dot(*p_hist2);
}



vector<Mat> Pipeline::loadFeatureVectors(const string& filename)
{
    ifstream fs(filename.c_str());
    istringstream ss;
    string line;
    
    int N, k;
    getline(fs, line);
    ss.str(line);
    ss >> N >> k;
    
    vector<Mat> vectors(N);
    
    int i;
    while(getline(fs, line)) {
        ss.clear();
        ss.str(line);
        ss >> i;
        vectors[i] = Mat(1, k, CV_32F);
        for(int j=0; j<k; j++)
            ss >> vectors[i].at<float>(0,j);
    }
    
    return vectors;
}

void Pipeline::saveFeatureVectors(const vector< Mat >& vectors, const string& filename)
{
    ofstream fs(filename.c_str());
    fs << vectors.size() << '\t' << vectors[0].cols << endl;
    for(unsigned int i=0; i<vectors.size(); i++) {
        fs << i << '\t';
        for (int j = 0; j<vectors[i].cols; j++)
            fs << '\t' << vectors[i].at<float>(0,j);
        fs << endl;
    }
    
    fs.close();
}


vector<Mat> Pipeline::addTranslationNoise(int N, const Mat* data, double sigma)
{
    RNG rng;
    Mat M = Mat::eye(2, 3, CV_32F);
    vector<Mat> out(N);
    
    for(int i=0; i<N; i++) {
        M.at<float>(0,2) = rng.gaussian(sigma);
        M.at<float>(1,2) = rng.gaussian(sigma);
        
        warpAffine(data[i], out[i], M, data[i].size(), INTER_LINEAR + WARP_INVERSE_MAP);
    }
    
    return out;
}
