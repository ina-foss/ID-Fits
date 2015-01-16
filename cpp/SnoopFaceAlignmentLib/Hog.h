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