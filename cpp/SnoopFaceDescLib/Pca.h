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


#ifndef PCA_H
#define PCA_H

#include "opencv2/opencv.hpp"
#include <string>
#include <vector>
#include <stdexcept>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>


class Pca
{
public:
    Pca()
    {
    }
    
    Pca(const std::string filename)
    {
        load(filename);
    }
    
    Pca(const Pca& pca)
    {
        *this = pca;
    }
    
    Pca(const cv::Mat& mean, const cv::Mat& eigenvalues, const cv::Mat& eigenvectors)
    {
        cv::cv2eigen(mean, mean_);
        cv::cv2eigen(eigenvalues.t(), eigenvalues_);
        cv::cv2eigen(eigenvectors.t(), eigenvectors_);
    }
    
    Pca& operator=(const Pca& pca)
    {
        mean_ = pca.mean_;
        eigenvalues_ = pca.eigenvalues_;
        eigenvectors_ = pca.eigenvectors_;

        return *this;
    }
    
    void create(const std::vector<cv::Mat>& data, int dim)
    {
        // Eigen code as slow
        int n = (int) data.size();
        int m = data[0].rows*data[0].cols;
        if(m < dim)
            throw std::runtime_error("the output dimension of PCA should be lower or equal to the input dimension");
        
        mean_ = Eigen::Matrix<float, 1, Eigen::Dynamic>::Zero(m);
        for(int i=0; i<n; i++)
        {
            Eigen::Matrix<float, 1, Eigen::Dynamic> row;
            cv::cv2eigen(data[i].reshape(1,1), row);
            mean_ += row;
        }
        mean_ /= n;
        
        Eigen::MatrixXf X(n, m);
        for(int i=0; i<n; i++)
        {
            Eigen::Matrix<float, 1, Eigen::Dynamic> row;
            cv::cv2eigen(data[i].reshape(1,1), row);
            X.row(i) = row - mean_;
        }
        
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(X, Eigen::ComputeThinV);
        if(dim > 0) {
            eigenvalues_ = svd.singularValues().topRows(dim);
            eigenvectors_ = svd.matrixV().leftCols(dim);
        }
        else {
            eigenvalues_ = svd.singularValues();
            eigenvectors_ = svd.matrixV();
        }
        eigenvalues_ = eigenvalues_.array().square() / ((float) (n-1));
        
        /*
        Eigen::MatrixXf Y = X.transpose() * X / ((float) (n-1));
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf, Eigen::NoQRPreconditioner> decomposition(Y);
        if(dim > 0) {
            eigenvalues_ = decomposition.eigenvalues().bottomRows(dim);
            eigenvectors_ = decomposition.eigenvectors().rightCols(dim);
        }
        else {
            eigenvalues_ = decomposition.eigenvalues();
            eigenvectors_ = decomposition.eigenvectors();
        }
        */
    }
    
    void project(const cv::Mat& src, cv::Mat& dst) const
    {
        Eigen::Matrix<float, 1, Eigen::Dynamic> eigen_src, eigen_dst;
        cv::cv2eigen(src.reshape(1, 1), eigen_src);
        eigen_dst = (eigen_src - mean_)*eigenvectors_;
        cv::eigen2cv(eigen_dst, dst);
    }

    void save(const std::string& filename) const
    {
        cv::FileStorage fs(filename, cv::FileStorage::WRITE);
        
        cv::Mat mean, eigenvalues, eigenvectors;
        cv::eigen2cv(mean_, mean);
        cv::eigen2cv(eigenvalues_, eigenvalues);
        cv::eigen2cv(eigenvectors_, eigenvectors);
        
        fs << "mean" << mean;
        fs << "eigenvalues" << eigenvalues.t();
        fs << "eigenvectors" << eigenvectors.t();
        
       fs.release();
    }

    void load(const std::string& filename)
    {
        cv::FileStorage fs(filename, cv::FileStorage::READ);
        if(!fs.isOpened())
            throw std::runtime_error("Cannot open file : " + filename);
        
        cv::Mat mean, eigenvalues, eigenvectors;
        fs["mean"] >> mean;
        fs["eigenvalues"] >> eigenvalues;
        fs["eigenvectors"] >> eigenvectors;
        
        fs.release();
        
        cv::cv2eigen(mean, mean_);
        cv::cv2eigen(eigenvalues.t(), eigenvalues_);
        cv::cv2eigen(eigenvectors.t(), eigenvectors_);
    }
    
    cv::Mat getEigenvalues() const
    {
        cv::Mat eigenvalues;
        cv::eigen2cv(eigenvalues_, eigenvalues);
        return eigenvalues;
    }
    
    int getDimension() const
    {
        return eigenvalues_.cols();
    }
    
    void reduceDimension(int dim)
    {
        eigenvalues_.conservativeResize(Eigen::NoChange, dim);
        eigenvectors_.conservativeResize(Eigen::NoChange, dim);
    }
    
private:
    Eigen::Matrix<float, 1, Eigen::Dynamic> mean_;
    Eigen::Matrix<float, 1, Eigen::Dynamic> eigenvalues_;
    Eigen::MatrixXf eigenvectors_;
};

#endif
