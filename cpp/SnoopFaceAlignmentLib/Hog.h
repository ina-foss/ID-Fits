#ifndef HOG_H
#define HOG_H

#include "opencv2/opencv.hpp"
#include <cmath>
#include <iostream>

using namespace cv;

class Hog
{
public:
    Hog();
    
    Mat compute(Mat& src) const;
    
private:
    int bins_number, pixel_number, cells_size;
    double epsilon;
    
    Mat kernel_x, kernel_y;
};

#endif