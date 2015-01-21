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


#include <SnoopFaceAlignmentLib/Hog.h>


Hog::Hog(): bins_number(8), pixel_number(32), cells_size(8), epsilon(1e-5)
{
    kernel_x = Mat(1,3,CV_64FC1);
    kernel_x.at<double>(0,0) = -1.0;
    kernel_x.at<double>(0,1) = 0.f;
    kernel_x.at<double>(0,2) = 1.0;
    
    kernel_y = kernel_x.t();
}


Mat Hog::compute(Mat& src) const {
    Mat gradient_x, gradient_y, magnitude, angle;
    
    filter2D(src, gradient_x, CV_64FC1, kernel_x, Point(-1,-1));
    filter2D(src, gradient_y, CV_64FC1, kernel_y, Point(-1,-1));
//     filter2D(src, gradient_x, CV_64FC1, kernel_x, Point(-1,-1), BORDER_DEFAULT | BORDER_ISOLATED);
//     filter2D(src, gradient_y, CV_64FC1, kernel_y, Point(-1,-1), BORDER_DEFAULT | BORDER_ISOLATED);
    cartToPolar(gradient_x, gradient_y, magnitude, angle, true);
    
    Mat descriptor = Mat::zeros(16, bins_number, CV_64FC1);
    for(int i=0; i<pixel_number; i++) {
        for(int j=0; j<pixel_number; j++) {
            int block_x = i/cells_size;
            int block_y = j/cells_size;
            int bin = ((int) ((bins_number * angle.at<double>(i,j)) / 180.0)) % bins_number;
            descriptor.at<double>(block_x * 4 + block_y, bin) += magnitude.at<double>(i,j);
        }
    }

    double n = norm(descriptor);
    return descriptor.reshape(1,1) / sqrt(n*n + epsilon);
}

