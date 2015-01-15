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
