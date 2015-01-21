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


#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

#include <iterator>
#include <sstream>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Cholesky>
#include <SnoopFaceAlignmentLib/LibLinearWrapper.h>


typedef std::vector<cv::Mat> MatVector;


struct BinaryTree
{
public:
    BinaryTree();
    BinaryTree(int depth, int pos);
    BinaryTree(std::istream& fs, unsigned int total_tree_depth);
    BinaryTree(const BinaryTree& copy);
    ~BinaryTree();
    BinaryTree& operator=(const BinaryTree& other);
    
    unsigned int depth_;
    unsigned int pos_;
    double index1_[2];
    double index2_[2];
    BinaryTree* left_;
    BinaryTree* right_;
};

std::ostream& operator<<(std::ostream& os, const BinaryTree& binary_tree);


class Feature
{
public:
    virtual ~Feature() = 0;
    
    inline static bool performTest(const double* index1, const double* index2, const double *landmark, const cv::Mat& img, const double* transformation_matrix)
    {
        int transformed_index1[2], transformed_index2[2];
        
        performTransformation(transformed_index1, index1, landmark, transformation_matrix);
        performTransformation(transformed_index2, index2, landmark, transformation_matrix);
        
        if(transformed_index1[0] < 0 || transformed_index1[0] >= img.cols
            || transformed_index1[1] < 0 || transformed_index1[1] >= img.rows
            || transformed_index2[0] < 0 || transformed_index2[0] >= img.cols
            || transformed_index2[1] < 0 || transformed_index2[1] >= img.rows
        ) {
            std::srand(std::time(NULL));
            return (std::rand() % 2) == 1;
        }
        
        const uint8_t* img_data = (uint8_t*) img.data;
        const uint8_t pixel1 = img_data[transformed_index1[1]*img.cols + transformed_index1[0]];
        const uint8_t pixel2 = img_data[transformed_index2[1]*img.cols + transformed_index2[0]];
        
        return (pixel1 > pixel2);
    }
    
    inline static void performTransformation(int* transformed_index, const double* index, const double* landmark, const double* transformation_matrix)
    {
        transformed_index[0] = landmark[0] + transformation_matrix[0]*index[0] + transformation_matrix[1]*index[1];
        transformed_index[1] = landmark[1] + transformation_matrix[2]*index[0] + transformation_matrix[3]*index[1];
    }
};


class TreeRegressor
{
public:
    TreeRegressor(const BinaryTree& tree, int depth): depth_(depth), tree_(tree)
    {
    }
    
    BinaryTree& getTree()
    {
        return tree_;
    }
    
    inline unsigned int getBinaryOutputIndex(const double* landmark, const cv::Mat& img, const double* transformation_matrix) const
    {
        const BinaryTree *current_node = &tree_;
        
        for(int i=0; i<depth_; i++) {
            if(Feature::performTest(current_node->index1_, current_node->index2_, landmark, img, transformation_matrix))
                current_node = current_node->left_;
            else
                current_node = current_node->right_;
        }
        
        return current_node->pos_;
    }
    
    friend std::ostream& operator<<(std::ostream& os, const TreeRegressor& tree_regressor);
    
private:
    int depth_;
    BinaryTree tree_;
};

std::ostream& operator<<(std::ostream& os, const TreeRegressor& tree_regressor);


typedef std::vector<Node> SparseVector;
typedef std::vector<BinaryTree* > Forest_t;


class ForestRegressor
{
public:
    ForestRegressor()
    {
    }
    
    ForestRegressor(Forest_t& forest, const std::vector<cv::Mat>& lookup_table, int N, int D, int l):
        N_(N), D_(D), l_(l), leaves_number_(1<<D), lookup_table_(lookup_table)
    {
        forest_.reserve(N);
        
        for(int i=0; i<N; i++)
            forest_.push_back(TreeRegressor(*forest[i], D));
    }
    
    ForestRegressor(std::istream& fs, int N, int D, int l): N_(N), D_(D), l_(l), leaves_number_(1<<D)
    {
        forest_.reserve(N);
        
        for(int n=0; n<N_; n++) 
            forest_.push_back(TreeRegressor(BinaryTree(fs, D), D));
        
        int L;
        fs >> L;
        
        lookup_table_.reserve(N_*leaves_number_);
        for(int n=0; n<N_; n++) {
            for(int i=0; i<leaves_number_; i++) {
                lookup_table_.push_back(cv::Mat(L, 2, CV_64FC1));
                double* lookup_table_data = (double*) lookup_table_[leaves_number_*n + i].data;
                
                for(int l=0; l<2*L; l++)
                    fs >> lookup_table_data[l];
            }
        }
    }
    
    std::vector<cv::Mat>& getLookupTable()
    {
        return lookup_table_;
    }
    
    inline void getOutput(cv::Mat& output, const double *landmark, const cv::Mat& img, const double* transformation_matrix) const
    {
        for(int n=0; n<N_; n++)
            output += lookup_table_[leaves_number_*n + forest_[n].getBinaryOutputIndex(landmark, img, transformation_matrix)];
    }
    
    inline void getLocalBinaryFeature(int* lbf, const double *landmark, const cv::Mat& img, const double* transformation_matrix) const
    {
        for(int n=0; n<N_; n++, lbf++)
            *lbf = leaves_number_*n + forest_[n].getBinaryOutputIndex(landmark, img, transformation_matrix);
    }
    
    inline void getLocalBinaryFeatureVector(SparseVector::iterator lbf, const double *landmark, const cv::Mat& img, const double* transformation_matrix) const
    {
        for(int n=0; n<N_; n++) {
            Node node;
            node.index = leaves_number_*N_*l_ + leaves_number_*n + forest_[n].getBinaryOutputIndex(landmark, img, transformation_matrix) + 1;
            node.value = 1.0;
            *lbf = node;
            lbf++;
        }
    }
    
    friend std::ostream& operator<<(std::ostream& os, const ForestRegressor& forest_regressor);
    
private:
    int N_, D_, l_, leaves_number_;
    std::vector<TreeRegressor> forest_;
    std::vector<cv::Mat> lookup_table_;
};

std::ostream& operator<<(std::ostream& os, const ForestRegressor& forest_regressor);


class AlignmentMethod
{
public:
    AlignmentMethod(int T, int N, int D, int L);
    AlignmentMethod(const std::string& filename);
    virtual ~AlignmentMethod();
    
    void setMeanShape(double* mean_shape);
    void setForestRegressors(ForestRegressor** forest_regressors);
    void saveModel(const std::string& filename) const;
    
    cv::Mat align(const cv::Rect& bounding_box, const cv::Mat& img) const;
    void computeTransformation(const double* shape, double* transformation_matrix) const;
    void normalizeShapeAccordingToFaceBoundingBox(double* shape, const cv::Rect& bounding_box) const;
    void adaptNormalizedShapeToFaceBoundingBox(double* shape, const cv::Rect& bounding_box) const;
    cv::Mat getMeanShape() const;

    virtual void updateRule(int t, cv::Mat& shape, const cv::Mat& img, const double* transformation_matrix) const = 0;
    
protected:
    AlignmentMethod();
    
    virtual void save(std::ostream& fs) const;
    virtual void load(std::istream& fs);
    
protected:
    unsigned int T_, N_, D_, L_, leaves_number_;
    bool is_regressor_loaded_;
    double *mean_shape_;
    ForestRegressor **forest_regressors_;
};


class LocalRegressorAlignment: public AlignmentMethod
{
public:
    LocalRegressorAlignment(int T, int N, int D, int L);
    LocalRegressorAlignment(const std::string& filename);
    
protected:
    void updateRule(int t, cv::Mat& shape, const cv::Mat& img, const double* transformation_matrix) const;
};


class LocalBinaryFeatureAlignment: public AlignmentMethod
{
public:
    LocalBinaryFeatureAlignment(int T, int N, int D, int L);
    LocalBinaryFeatureAlignment(const std::string& filename);
    
protected:
    void updateRule(int t, cv::Mat& shape, const cv::Mat& img, const double* transformation_matrix) const;
};

#endif
