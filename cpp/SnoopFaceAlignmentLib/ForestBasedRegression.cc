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


#include <SnoopFaceAlignmentLib/ForestBasedRegression.h>
#include <iomanip>
#include <stdexcept>



BinaryTree::BinaryTree(): depth_(0), pos_(0), left_(NULL), right_(NULL)
{
    index1_[0] = index1_[1] = 0;
    index2_[0] = index2_[1] = 0;
}

BinaryTree::BinaryTree(int depth, int pos): depth_(depth), pos_(pos), left_(NULL), right_(NULL)
{
    index1_[0] = index1_[1] = 0;
    index2_[0] = index2_[1] = 0;
}

BinaryTree::BinaryTree(std::istream& fs, unsigned int total_tree_depth)
{
    read_binary(fs, depth_);
    read_binary(fs, pos_);
    read_binary(fs, index1_[0]);
    read_binary(fs, index1_[1]);
    read_binary(fs, index2_[0]);
    read_binary(fs, index2_[1]);
    
    if(depth_ < total_tree_depth) {
        left_ = new BinaryTree(fs, total_tree_depth);
        right_ = new BinaryTree(fs, total_tree_depth);
    }
    else
        left_ = right_ = NULL;
}

BinaryTree::BinaryTree(const BinaryTree& copy)
{
    *this = copy;
}

BinaryTree::~BinaryTree()
{
    if(left_)
        delete left_;
    if(right_)
        delete right_;
}
    
BinaryTree& BinaryTree::operator=(const BinaryTree& other)
{
    depth_ = other.depth_;
    pos_ = other.pos_;

    std::copy(other.index1_, other.index1_+2, index1_);
    std::copy(other.index2_, other.index2_+2, index2_);
    
    if(other.left_)
        left_ = new BinaryTree(*other.left_);
    else
        left_ = NULL;
    
    if(other.right_)
        right_ = new BinaryTree(*other.right_);
    else
        right_ = NULL;
    
    return *this;
}

std::ostream& operator<<(std::ostream& os, const BinaryTree& binary_tree)
{
    write_binary(os, binary_tree.depth_);
    write_binary(os, binary_tree.pos_);
    write_binary(os, binary_tree.index1_[0]);
    write_binary(os, binary_tree.index1_[1]);
    write_binary(os, binary_tree.index2_[0]);
    write_binary(os, binary_tree.index2_[1]);
    
    if(binary_tree.left_ && binary_tree.right_) {
        os << *binary_tree.left_;
        os << *binary_tree.right_;
    }

    return os;
}


std::ostream& operator<<(std::ostream& os, const TreeRegressor& tree_regressor)
{
    os << tree_regressor.tree_;
    return os;
}


std::ostream& operator<<(std::ostream& os, const ForestRegressor& forest_regressor)
{
    for(int n=0; n<forest_regressor.N_; n++)
        os << forest_regressor.forest_[n];
    
    int L = forest_regressor.lookup_table_[0].rows;
    write_binary(os, L);
    
    for(int n=0; n<forest_regressor.N_; n++) {
        for(int i=0; i<forest_regressor.leaves_number_; i++) {
            const float* lookup_table_data = (float*) forest_regressor.lookup_table_[n*forest_regressor.leaves_number_ + i].data;
            
            for(int l=0; l<2*L; l++)
	        write_binary(os, lookup_table_data[l]);
        }
    }
    
    return os;
}

AlignmentMethod::AlignmentMethod()
{
}

AlignmentMethod::AlignmentMethod(int T, int N, int D, int L):
    T_(T), N_(N), D_(D), L_(L), leaves_number_(1<<D),
    is_regressor_loaded_(false),
    mean_shape_(NULL),
    forest_regressors_(NULL)
{
}

AlignmentMethod::AlignmentMethod(const std::string& filename)
{
    std::ifstream fs(filename.c_str(), std::ios::in | std::ios::binary);
    if(!fs.is_open())
        throw std::runtime_error("Cannot open file " + filename);
    
    load(fs);
}

AlignmentMethod::~AlignmentMethod()
{
    if(is_regressor_loaded_ && forest_regressors_) {
        for(unsigned int t=0; t<T_; t++)
            delete[] forest_regressors_[t];
        delete[] forest_regressors_;
        
        delete[] mean_shape_;
    }
}

void AlignmentMethod::setMeanShape(float* mean_shape)
{
    mean_shape_ = mean_shape;
}

void AlignmentMethod::setForestRegressors(ForestRegressor** forest_regressors)
{
    forest_regressors_ = forest_regressors;
}

void AlignmentMethod::saveModel(const std::string& filename) const
{
    std::ofstream fs(filename.c_str(), std::ios::out | std::ios::binary);
    if(!fs.is_open())
        throw std::runtime_error("Cannot open file " + filename);
    
    save(fs);
}

cv::Mat AlignmentMethod::align(const cv::Rect& bounding_box, const cv::Mat& img) const
{
    if(img.channels() != 1)
        throw std::runtime_error("image should be grayscale");
    
    cv::Mat shape(L_, 2, CV_32FC1);
    float* shape_data = (float*) shape.data;
    
    std::copy(mean_shape_, mean_shape_+2*L_, shape_data);
    
    adaptNormalizedShapeToFaceBoundingBox(shape_data, bounding_box);
    float transformation_matrix[4];
    
    for(unsigned int t=0; t<T_; t++) {
        computeTransformation(shape_data, transformation_matrix);
        updateRule(t, shape, img, transformation_matrix);
    }
    
    return shape;
}

void AlignmentMethod::computeTransformation(const float* shape, float* transformation_matrix) const
{
    Eigen::Matrix<float, Eigen::Dynamic, 4> A(2*L_, 4);
    Eigen::VectorXf b(2*L_);
    
    for(unsigned int l=0; l<L_; l++) {
        A(2*l, 0) = shape[2*l];
        A(2*l, 1) = -shape[2*l+1];
        A(2*l, 2) = 1;
        A(2*l, 3) = 0;
        
        A(2*l+1, 0) = shape[2*l+1];
        A(2*l+1, 1) = shape[2*l];
        A(2*l+1, 2) = 0;
        A(2*l+1, 3) = 1;
    
        b(2*l) = mean_shape_[2*l];
        b(2*l+1) = mean_shape_[2*l+1];
    }
    
    Eigen::Matrix<float, Eigen::Dynamic, 4> cov;
    cov.noalias() = A.transpose() * A;
    
    Eigen::Vector4f sol = cov.ldlt().solve(A.transpose() *b);
    
    float scaling_factor = 1.0 / std::sqrt(sol(0)*sol(0) + sol(1)*sol(1));
    transformation_matrix[0] = sol(0) * scaling_factor * scaling_factor;
    transformation_matrix[1] = sol(1) * scaling_factor * scaling_factor;
    transformation_matrix[2] = -transformation_matrix[1];
    transformation_matrix[3] = transformation_matrix[0];
}

void AlignmentMethod::normalizeShapeAccordingToFaceBoundingBox(float* shape, const cv::Rect& bounding_box) const
{
    for(unsigned int l=0; l<L_; l++) {
        shape[2*l] = (shape[2*l] - bounding_box.x) / bounding_box.width;
        shape[2*l+1] = (shape[2*l+1] - bounding_box.y) / bounding_box.height;
    }
}

void AlignmentMethod::adaptNormalizedShapeToFaceBoundingBox(float* shape, const cv::Rect& bounding_box) const
{
    for(unsigned int l=0; l<L_; l++) {
        shape[2*l] = (shape[2*l] * bounding_box.width) + bounding_box.x;
        shape[2*l+1] = (shape[2*l+1] * bounding_box.height) + bounding_box.y;
    }
}

cv::Mat AlignmentMethod::getMeanShape() const
{
    return cv::Mat(L_, 2, CV_32FC1, mean_shape_);
}


void AlignmentMethod::save(std::ostream& fs) const
{
    write_binary(fs, T_);
    write_binary(fs, N_);
    write_binary(fs, D_);
    write_binary(fs, L_);
    
    for(unsigned int l=0; l<2*L_; l++)
        write_binary(fs, mean_shape_[l]);
    
    for(unsigned int t=0; t<T_; t++)
        for(unsigned int l=0; l<L_; l++)
            fs << forest_regressors_[t][l];
}

void AlignmentMethod::load(std::istream& fs)
{
    read_binary(fs, T_);
    read_binary(fs, N_);
    read_binary(fs, D_);
    read_binary(fs, L_);

    leaves_number_ = 1<<D_;
    
    mean_shape_ = new float[2*L_];
    for(unsigned int l=0; l<2*L_; l++)
        read_binary(fs, mean_shape_[l]);
    
    forest_regressors_ = new ForestRegressor*[T_];
    for(unsigned int t=0; t<T_; t++) {
        forest_regressors_[t] = new ForestRegressor[L_];
        for(unsigned int l=0; l<L_; l++)
            forest_regressors_[t][l] = ForestRegressor(fs, N_, D_, l);
    }
    
    is_regressor_loaded_ = true;
}


LocalRegressorAlignment::LocalRegressorAlignment(int T, int N, int D, int L):
    AlignmentMethod(T, N, D, L)
{
}

LocalRegressorAlignment::LocalRegressorAlignment(const std::string& filename):
    AlignmentMethod(filename)
{
}

void LocalRegressorAlignment::updateRule(int t, cv::Mat& shape, const cv::Mat& img, const float* transformation_matrix) const
{
    for(unsigned int l=0; l<L_; l++) {
        float *landmark = shape.ptr<float>(l);
        
        cv::Mat output = cv::Mat::zeros(1, 2, CV_32FC1);
        forest_regressors_[t][l].getOutput(output, landmark, img, transformation_matrix);
        
        const float* output_data = (float*) output.data;
        
        landmark[0] += output_data[0] * transformation_matrix[0] + output_data[1] * transformation_matrix[1];
        landmark[1] += output_data[0] * transformation_matrix[2] + output_data[1] * transformation_matrix[3];
    }
}



LocalBinaryFeatureAlignment::LocalBinaryFeatureAlignment(int T, int N, int D, int L):
    AlignmentMethod(T, N, D, L)
{
}

LocalBinaryFeatureAlignment::LocalBinaryFeatureAlignment(const std::string& filename):
    AlignmentMethod(filename)
{
}

void LocalBinaryFeatureAlignment::updateRule(int t, cv::Mat& shape, const cv::Mat& img, const float* transformation_matrix) const
{
    float* shape_data = (float*) shape.data;
    float prev_shape_data[2*L_];
    std::copy(shape_data, shape_data+2*L_, prev_shape_data);
    
    for(unsigned int l=0; l<L_; l++) {
        cv::Mat output = cv::Mat::zeros(L_, 2, CV_32FC1);
        
        forest_regressors_[t][l].getOutput(output, prev_shape_data+2*l, img, transformation_matrix);
        
        const float* output_data = (float*) output.data;
        for(unsigned int l2=0; l2<L_; l2++) {
            shape_data[2*l2] += output_data[2*l2] * transformation_matrix[0] + output_data[2*l2+1] * transformation_matrix[1];
            shape_data[2*l2+1] += output_data[2*l2] * transformation_matrix[2] + output_data[2*l2+1] * transformation_matrix[3];
        }
        
    }
}

