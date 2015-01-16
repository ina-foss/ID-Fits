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


#ifndef LDA_H

#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>


class Lda
{
public:
    Lda(const cv::Mat& mean, const cv::Mat& scalings)
    {
	cv::cv2eigen(mean.reshape(1, 1), mean_);
	cv::cv2eigen(scalings, scalings_);
    }

    void project(const cv::Mat& src, cv::Mat& dst) const
    {
	Eigen::Matrix<float, 1, Eigen::Dynamic> e_src, e_dst;
        cv::cv2eigen(src.reshape(1, 1), e_src);
        e_dst = (e_src - mean_)*scalings_;
        cv::eigen2cv(e_dst, dst);
    }

private:
    Eigen::Matrix<float, 1, Eigen::Dynamic> mean_;
    Eigen::MatrixXf scalings_;
};


#endif
