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


#ifndef FOREST_BASED_REGRESSION_TRAINING_H
#define FOREST_BASED_REGRESSION_TRAINING_H

#include <ctime>
#include <iomanip>
#include <cmath>
#include <stack>
//#include <boost/random/mersenne_twister.hpp>
//#include <boost/random/uniform_real_distribution.hpp>
#if defined(__GXX_EXPERIMENTAL_CXX0X) || __cplusplus >= 201103L
    #include <random>
#endif
#include <SnoopFaceAlignmentLib/ForestBasedRegression.h>
#include <SnoopFaceAlignmentLib/LibLinearWrapper.h>


typedef std::pair<float[2], float[2]> Feature_t;
typedef std::vector<Node> SparseVector;
const float PI_2_ = 2*3.141592653589793238463;
enum node_separation_criteria_t {LS, VAR, MEAN_NORM, NORMALIZED_LS, NORMALIZED_VAR, TEST};


class ForestRegressorTraining
{
public:
    ForestRegressorTraining(int N, int D):
        N_(N), D_(D), criteria_(LS), sampled_random_features_number_(500)
    {
    }
    
    void setSampledRandomFeaturesNumber(int sampled_random_features_number)
    {
        sampled_random_features_number_ = sampled_random_features_number;
    }
    
    void setNodeSeparationCriteria(node_separation_criteria_t criteria)
    {
        criteria_ = criteria;
    }
    
    void setTransformations(const std::vector<float*>& transformations)
    {
        transformations_ = transformations;
    }
    
    void train(ForestRegressor& forest_regressor, int l, const MatVector& shape, const MatVector& delta_s, const MatVector& image, float radius = 0.2)
    {
        std::cout << "Training forest for landmark #" << l+1 << " ..." << std::endl;
        
        int n = delta_s.size();
        std::vector<int> all_data;
        all_data.reserve(n);
        for(int i=0; i<n; i++)    
            all_data.push_back(i);
        
        unsigned int P = 3*N_/2;
        Forest_t forest;
        forest.resize(P);
        std::vector<float> training_errors(P);
        
        l_ = l;
        shape_ = &shape;
        shape_error_ = &delta_s;
        image_ = &image;
        radius_ = radius;
        
        int leaves_number_ = 1 << D_;
        std::vector<cv::Mat> lookup_table;
        lookup_table.reserve(P*leaves_number_);
        for(unsigned int i=0; i<P*leaves_number_; i++)
            lookup_table.push_back(cv::Mat::zeros(1, 2, CV_32FC1));
        
        #pragma omp parallel for
        for(unsigned int n=0; n<P; n++) {
            forest[n] = trainTree(n, all_data, lookup_table);
            
            TreeRegressor tree(*forest[n], D_);
            training_errors[n] = 0;
            for(unsigned int i=0; i<delta_s.size(); i++) {
                unsigned int pos = tree.getBinaryOutputIndex(shape[i].ptr<float>(l), image[i], transformations_[i]);
                training_errors[n] += cv::norm(delta_s[i].row(l) - lookup_table[n*leaves_number_ + pos]);
            }
        }
        
        // Keep the N_ best trees out of the P computed
        for(unsigned int n=0; n<N_; n++) {
            float max_error = 0;
            unsigned int max_id = 0;
            for(unsigned int p=0; p<P-n; p++) {
                if(max_error < training_errors[p]) {
                    max_id = p;
                    max_error = training_errors[p];
                }
            }
            training_errors.erase(training_errors.begin()+max_id);
            forest.erase(forest.begin()+max_id);
            lookup_table.erase(lookup_table.begin()+max_id*leaves_number_, lookup_table.begin()+(max_id+1)*leaves_number_);
        }
        
        forest_regressor = ForestRegressor(forest, lookup_table, N_, D_, l);
    }
    
    BinaryTree* trainTree(int tree_pos_in_forest, const std::vector<int>& all_data_partition, std::vector<cv::Mat>& lookup_table)
    {
        Feature_t random_features[sampled_random_features_number_];
        generateRandomFeatures(radius_, random_features);
        
        int n = all_data_partition.size();
        bool **random_features_outputs = new bool*[sampled_random_features_number_];
        for(unsigned int f=0; f<sampled_random_features_number_; f++) {
            random_features_outputs[f] = new bool[n];
            
            Feature_t feature = random_features[f];
            for(int i=0; i<n; i++)
                random_features_outputs[f][i] = Feature::performTest(feature.first, feature.second, (*shape_)[i].ptr<float>(l_), (*image_)[i], transformations_[i]);
        }
        
        //return trainTreeIterative(all_data_partition, random_features_output);
        BinaryTree* tree = trainTreeRecursive(tree_pos_in_forest, 0, 0, all_data_partition, random_features, random_features_outputs, lookup_table);
        
        for(unsigned int f=0; f<sampled_random_features_number_; f++)
            delete[] random_features_outputs[f];
        delete[] random_features_outputs;
        
        return tree;
    }
    
    BinaryTree* trainTreeRecursive(int tree_pos_in_forest, unsigned int depth, unsigned int pos, const std::vector<int>& node_partition, Feature_t* random_features, bool** random_features_outputs, std::vector<cv::Mat>& lookup_table)
    {
        BinaryTree* tree = new BinaryTree();
        tree->depth_ = depth;
        tree->pos_ = pos;
        
        if(depth == D_) {
            int n = node_partition.size();
            
            if(n > 0) {
                float sum[2] = {0.0, 0.0};
                for(unsigned int i=0; i<node_partition.size(); i++) {
                    const float *residual = (*shape_error_)[node_partition[i]].ptr<float>(l_);
                    sum[0] += residual[0];
                    sum[1] += residual[1];
                }
                
                sum[0] /= (node_partition.size() * N_);
                sum[1] /= (node_partition.size() * N_);
                
                float *output = lookup_table[(1<<D_)*tree_pos_in_forest + pos].ptr<float>(0);
                std::copy(sum, sum+2, output);
            }
            
            tree->left_ = NULL;
            tree->right_ = NULL;
        }
        else {
            std::vector<int> left, right;
            Feature_t feature;
            trainNode(node_partition, random_features, random_features_outputs, feature, left, right);
            
            std::copy(feature.first, feature.first+2, tree->index1_);
            std::copy(feature.second, feature.second+2, tree->index2_);
            
            tree->left_ = trainTreeRecursive(tree_pos_in_forest, depth+1, 2*pos, left, random_features, random_features_outputs, lookup_table);
            tree->right_ = trainTreeRecursive(tree_pos_in_forest, depth+1, 2*pos+1, right, random_features, random_features_outputs, lookup_table);
        }

        return tree;
    }
    
    void trainNode(const std::vector<int>& node_partition, Feature_t* random_features, bool** random_features_outputs, Feature_t& final_feature, std::vector<int>& left, std::vector<int>& right)
    {
        int n = node_partition.size();
        int n_left;
        
        float sum_left[2];
        float sum_squarednorm_left;
        float total_sum[2] = {0, 0};
        float total_sum_squarednorm = 0;
        for(int i=0; i<n; i++) {
            const float *residual = (*shape_error_)[i].ptr<float>(l_);
            
            total_sum[0] += residual[0];
            total_sum[1] += residual[1];
            
            total_sum_squarednorm += residual[0]*residual[0] + residual[1]*residual[1];
        }
        
        final_feature.first[0] = final_feature.first[1] = 0;
        final_feature.second[0] = final_feature.second[1] = 0;
        
        float min_value = 1e50, value = 0;
        int final_feature_index = -1;
        
        for(unsigned int i=0; i<sampled_random_features_number_; i++) {
            sum_left[0] = sum_left[1] = 0;
            sum_squarednorm_left = 0.f;
            n_left = 0;
            
            for(std::vector<int>::const_iterator it=node_partition.begin(); it!=node_partition.end(); it++) {
                if(random_features_outputs[i][*it]) {
                    const float *residual = (*shape_error_)[*it].ptr<float>(l_);
                    sum_left[0] += residual[0];
                    sum_left[1] += residual[1];
                    sum_squarednorm_left += residual[0]*residual[0] + residual[1]*residual[1];
                    n_left++;
                }
            }
            
            value = minimizationCriteria(total_sum, total_sum_squarednorm, sum_left, sum_squarednorm_left, n, n_left);
            
            if(min_value > value) {
                min_value = value;
                final_feature = random_features[i];
                final_feature_index = i;
            }
        }
        
        left.clear();
        right.clear();
        left.reserve(n);
        right.reserve(n);

        if(final_feature_index != -1) {
            for(std::vector<int>::const_iterator it=node_partition.begin(); it!=node_partition.end(); it++) {
                if(random_features_outputs[final_feature_index][*it])
                    left.push_back(*it);
                else
                    right.push_back(*it);
            }
        }
        else
            right = node_partition;
    }
    
    float minimizationCriteria(const float* total_sum, float total_sum_squarednorm, const float* sum_left, float sum_squarednorm_left, int n, int n_left) const
    {
        int n_right = n - n_left;
        
        float sum_right[2];
        sum_right[0] = total_sum[0]-sum_left[0];
        sum_right[1] = total_sum[1]-sum_left[1];
        
        const float sum_squarednorm_right = total_sum_squarednorm - sum_squarednorm_left;
        
        const float sum_left_norm = sum_left[0]*sum_left[0]+sum_left[1]*sum_left[1];
        const float sum_right_norm = sum_right[0]*sum_right[0]+sum_right[1]*sum_right[1];

        float var_left = 0, var_right = 0;
        
        switch(criteria_)
        {
            case LS:
                var_left = (n_left == 0) ? 0.f : - sum_left_norm / n_left;
                var_right = (n_right == 0) ? 0.f : - sum_right_norm / n_right;
                break;
            
            case VAR:
                var_left = (n_left < 2) ? 0.f : sum_squarednorm_left / (n_left-1) - sum_left_norm / ((n_left-1) * n_left);
                var_right = (n_right < 2) ? 0.f : sum_squarednorm_right / (n_right-1) - sum_right_norm / ((n_right-1) * n_right);
                break;
            
            case MEAN_NORM:
                var_left = (n_left == 0) ? 0.f : - sum_left_norm / (n_left * n_left);
                var_right = (n_right == 0) ? 0.f : - sum_right_norm / (n_right * n_right);
                break;
            
            case NORMALIZED_LS:
                var_left = (n_left == 0) ? 0.f : sum_squarednorm_left / (sum_left_norm / (n_left*n_left)) - n_left;
                var_right = (n_right == 0) ? 0.f : sum_squarednorm_right / (sum_right_norm / (n_right*n_right)) - n_right;
                break;
            
            case NORMALIZED_VAR:
                var_left = (n_left < 2) ? 0.f : (((sum_squarednorm_left * (n_left * n_left)) / (sum_left_norm * (n_left-1))) - n_left / (n_left-1));
                var_right = (n_right < 2) ? 0.f : (((sum_squarednorm_right * (n_right * n_right)) / (sum_right_norm * (n_right-1))) - n_right / (n_right-1));
                break;
                
            case TEST:
                var_left = (n_left == 0) ? 0.f : ((sum_squarednorm_left / n_left) / sum_left_norm / (n_left * n_left) - 1.0);
                var_right = (n_right == 0) ? 0.f : ((sum_squarednorm_right / n_right) / sum_right_norm / (n_right * n_right) - 1.0);
                break;
        }
        
        return std::max(var_left, var_right);
    }
    
    
#if defined(__GXX_EXPERIMENTAL_CXX0X) || __cplusplus >= 201103L
    void generateRandomFeatures(float radius, Feature_t* random_features)
    {
        //boost::random::uniform_real_distribution<float> coordinate(-radius, radius);
        std::uniform_real_distribution<float> coordinate(-radius, radius);
        
        float x, y;
        Feature_t feature;
       
        for(unsigned int i=0; i<sampled_random_features_number_; i++) {
            do {
                x = coordinate(rng_);
                y = coordinate(rng_);
            } while(x*x+y*y > radius*radius);
            feature.first[0] = x;
            feature.first[1] = y;
           
            do {
                x = coordinate(rng_);
                y = coordinate(rng_);
            } while(x*x+y*y > radius*radius);
            feature.second[0] = x;
            feature.second[1] = y;
            
            random_features[i] = feature;
        }
    }
    
#else
    float randomFloat(float a, float b) {
        float random = ((float) rand()) / (float) RAND_MAX;
        float diff = b - a;
        float r = random * diff;
        return a + r;
    }
    
    void generateRandomFeatures(float radius, Feature_t* random_features)
    {
        float x, y;
        Feature_t feature;
       
        for(unsigned int i=0; i<sampled_random_features_number_; i++) {
            do {
                x = randomFloat(-radius, radius);
                y = randomFloat(-radius, radius);
            } while(x*x+y*y > radius*radius);
            feature.first[0] = x;
            feature.first[1] = y;
           
            do {
                x = randomFloat(-radius, radius);
                y = randomFloat(-radius, radius);
            } while(x*x+y*y > radius*radius);
            feature.second[0] = x;
            feature.second[1] = y;
            
            random_features[i] = feature;
        }
    }
#endif
    
    
private:
    const unsigned int N_, D_;
    node_separation_criteria_t criteria_;
    unsigned int sampled_random_features_number_;
    int l_;
    float radius_;
    std::vector<float*> transformations_;
    const std::vector<cv::Mat> *shape_, *shape_error_, *image_;
    //boost::random::mt19937 rng_;
    //unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
#if defined(__GXX_EXPERIMENTAL_CXX0X) || __cplusplus >= 201103L
    std::mt19937 rng_;
#endif
};


class AlignmentMethodTraining
{
public:
    AlignmentMethodTraining(int R, int T, int N, int D):
        R_(R), T_(T), N_(N), D_(D),
        alignment_method_(NULL),
        forest_regressor_training_(N, D)
    {
    }
    
    virtual ~AlignmentMethodTraining()
    {
        if(alignment_method_)
            delete alignment_method_;
        
        for(unsigned int i=0; i<transformations_.size(); i++)
            delete[] transformations_[i];
    }
    
    virtual void train(const MatVector& s0, const MatVector& s_star, const std::vector<cv::Rect>& bounding_boxes, const MatVector& imgs, int eye_index1, int eye_index2, const std::string& output_filename) = 0;
    
    ForestRegressorTraining& getForestRegressorTraining()
    {
        return forest_regressor_training_;
    }
    
protected:
    void computeTransformations(const MatVector& shape)
    {
        int n = shape.size();
        
        transformations_.clear();
        transformations_.reserve(n);
        
        for(int i=0; i<n; i++) {
            float* transformation = new float[4];
            transformations_.push_back(transformation);
            alignment_method_->computeTransformation((float*) shape[i].data, transformations_[i]);
        }
        
        forest_regressor_training_.setTransformations(transformations_);
    }
    
    void computeNormalizedResiduals(const MatVector& delta_s, MatVector& normalized_delta_s)
    {
        int L = delta_s[0].rows;
        normalized_delta_s.reserve(delta_s.size());
        
        for(unsigned int i=0; i<delta_s.size(); i++) {
            normalized_delta_s.push_back(cv::Mat(L, 2, CV_32FC1));
            
            float inverse_transform[4];
            std::copy(transformations_[i], transformations_[i]+4, inverse_transform);
            
            float s_square = inverse_transform[0]*inverse_transform[0] + inverse_transform[1]*inverse_transform[1];
            
            inverse_transform[0] /= s_square;
            inverse_transform[1] = -inverse_transform[1] / s_square;
            inverse_transform[2] = -inverse_transform[1];
            inverse_transform[3] = inverse_transform[0];
            
            for(int l=0; l<L; l++) {
                const float *residual = delta_s[i].ptr<float>(l);
                float *normalized_residual = normalized_delta_s[i].ptr<float>(l);
                
                normalized_residual[0] = inverse_transform[0] * residual[0] + inverse_transform[1] * residual[1];
                normalized_residual[1] = inverse_transform[2] * residual[0] + inverse_transform[3] * residual[1];
            }
        }
    }
    
    void computeInteroccularDistances(const MatVector& s_star, int eye_index1, int eye_index2, std::vector<float>& distances)
    {
        distances.reserve(s_star.size());
        
        for(unsigned int i=0; i<s_star.size(); i++) {
            const float *eye1 = s_star[i].ptr<float>(eye_index1);
            const float *eye2 = s_star[i].ptr<float>(eye_index2);
            
            distances.push_back(std::sqrt((eye1[0]-eye2[0])*(eye1[0]-eye2[0]) + (eye1[1]-eye2[1])*(eye1[1]-eye2[1])));
        }
    }
    
    void computeMeanShape(float* mean_shape, const MatVector& s_star, const std::vector<cv::Rect>& bounding_boxes) const
    {
        int L = s_star[0].rows;
        int training_set_size = s_star.size();
        cv::Mat mean_shape_mat = cv::Mat::zeros(L, 2, CV_32FC1);
        
        for(int i=0; i<training_set_size; i+=R_) {
            cv::Mat normalized_shape(L, 2, CV_32FC1);
            s_star[i].copyTo(normalized_shape);
            
            alignment_method_->normalizeShapeAccordingToFaceBoundingBox((float*) normalized_shape.data, bounding_boxes[i]);
            mean_shape_mat += normalized_shape;
        }
        
        mean_shape_mat /= (training_set_size / R_);
        
        const float *mean_shape_mat_data = (float*) mean_shape_mat.data;
        std::copy(mean_shape_mat_data, mean_shape_mat_data+2*L, mean_shape);
    }
    
    float computeMeanError(const MatVector& delta_s, const std::vector<float>& inter_occular_distances) const
    {
        int training_set_size = delta_s.size();
        int L = delta_s[0].rows;
        
        float error = 0.0;
        for(int i=0; i<training_set_size; i++) {
            float inter_occular_distance = inter_occular_distances[i];
            
            for(int l=0; l<L; l++) {
                const float *residual = delta_s[i].ptr<float>(l);
                error += std::sqrt(residual[0] * residual[0] + residual[1] * residual[1]) / inter_occular_distance;
            }
        }

        return (error / (training_set_size * L))*100;
    }
    
protected:
    const unsigned int R_, T_, N_, D_;
    AlignmentMethod* alignment_method_;
    ForestRegressorTraining forest_regressor_training_;
    std::vector<float*> transformations_;
};


class LocalRegressorAlignmentTraining: public AlignmentMethodTraining
{
public:
    LocalRegressorAlignmentTraining(int R, int T, int N, int D):
        AlignmentMethodTraining(R, T, N, D)
    {
    }
    
    void train(const MatVector& s0, const MatVector& s_star, const std::vector<cv::Rect>& bounding_boxes, const MatVector& imgs, int eye_index1, int eye_index2, const std::string& output_filename = "local_trees_regression_model.txt")
    {
        std::cout << "Starting local trees regressors training..." << std::endl << std::endl;
        
        int training_set_size = s0.size();
        int L = s0[0].rows;
        
        alignment_method_ = new LocalRegressorAlignment(T_, N_, D_, L);
        
        std::cout << "Computing mean shape..." << std::endl;
        float mean_shape[2*L];
        computeMeanShape(mean_shape, s_star, bounding_boxes);
        alignment_method_->setMeanShape(mean_shape);

        ForestRegressor** forest_regressors = new ForestRegressor*[T_];
        for(unsigned int t=0; t<T_; t++)
            forest_regressors[t] = new ForestRegressor[L];
        alignment_method_->setForestRegressors(forest_regressors);
        
        std::vector<cv::Mat> s;
        s.reserve(s0.size());
        for(unsigned int i=0; i<s0.size(); i++) {
            s.push_back(cv::Mat());
            s0[i].copyTo(s[i]);
        }
        
        std::vector<cv::Mat> delta_s;
        delta_s.reserve(training_set_size);
        for(int i=0; i<training_set_size; i++)
            delta_s.push_back(cv::Mat(L, 2, CV_32FC1));
        
        std::vector<float> inter_occular_distances;
        computeInteroccularDistances(s_star, eye_index1, eye_index2, inter_occular_distances);
        
        std::cout << std::endl << std::fixed << std::setprecision(2);
        
        for(unsigned int t=0; t<T_; t++) {
            float time = std::time(NULL);
            
            std::cout << "Starting iteration #" << t+1 << std::endl;
            
            for(int i=0; i<training_set_size; i++)
                delta_s[i] = s_star[i] - s[i];
            std::cout << "Mean error: " << computeMeanError(delta_s, inter_occular_distances) << "%" << std::endl;
            
            
            std::cout << "Computing transformations..." << std::endl;
            computeTransformations(s);
            std::vector<cv::Mat> normalized_delta_s;
            computeNormalizedResiduals(delta_s, normalized_delta_s);
            
            
            std::cout << "Building forest regressors..." << std::endl;
            for(int l=0; l<L; l++)
                forest_regressor_training_.train(forest_regressors[t][l], l, s, normalized_delta_s, imgs);
            
            
            std::cout << "Updating shapes..." << std::endl;
            for(int i=0; i<training_set_size; i++)
                alignment_method_->updateRule(t, s[i], imgs[i], transformations_[i]);
            
            std::cout << "Finished in " << (std::time(NULL) - time)/60.f << " min" << std::endl << std::endl;
        }
        
        for(int i=0; i<training_set_size; i++)
            delta_s[i] = s_star[i] - s[i];
        std::cout << "Final mean error: " << computeMeanError(delta_s, inter_occular_distances) << "%" << std::endl;
        
        std::cout << "Saving model to " << output_filename << " ..." << std::endl;
        alignment_method_->saveModel(output_filename);
        
        for(unsigned int t=0; t<T_; t++)
            delete[] forest_regressors[t];
        delete[] forest_regressors;
    }
};


class ForestRegressorLibLinearFormat: public ForestRegressor
{
public:
    inline void getLocalBinaryFeatureVector(SparseVector::iterator lbf, const float *landmark, const cv::Mat& img, const float* transformation_matrix) const
    {
        for(int n=0; n<N_; n++) {
            Node node;
            node.index = leaves_number_*N_*l_ + leaves_number_*n + forest_[n].getBinaryOutputIndex(landmark, img, transformation_matrix) + 1;
            node.value = 1.0;
            *lbf = node;
            lbf++;
        }
    }
};


class LocalBinaryFeatureTraining: public AlignmentMethodTraining
{
public:
    LocalBinaryFeatureTraining(int R, int T, int N, int D):
        AlignmentMethodTraining(R, T, N, D)
    {
    }
    
    void train(const MatVector& s0, const MatVector& s_star, const std::vector<cv::Rect>& bounding_boxes, const MatVector& imgs, int eye_index1, int eye_index2, const std::string& output_filename = "lbf_regression_model.txt")
    {
        int training_set_size = s0.size();
        int L = s0[0].rows;
        
        std::vector<cv::Mat> s;
        s.reserve(s0.size());
        for(unsigned int i=0; i<s0.size(); i++) {
            s.push_back(cv::Mat());
            s0[i].copyTo(s[i]);
        }
        
        alignment_method_ = new LocalBinaryFeatureAlignment(T_, N_, D_, L);
        
        float mean_shape[2*L];
        computeMeanShape(mean_shape, s_star, bounding_boxes);
        alignment_method_->setMeanShape(mean_shape);

        ForestRegressor** forest_regressors = new ForestRegressor*[T_];
        for(unsigned int t=0; t<T_; t++)
            forest_regressors[t] = new ForestRegressorLibLinearFormat[L];
        alignment_method_->setForestRegressors(forest_regressors);
        
        std::vector<cv::Mat> delta_s;
        delta_s.reserve(training_set_size);
        for(int i=0; i<training_set_size; i++)
            delta_s.push_back(cv::Mat(L, 2, CV_32FC1));
        
        std::vector<float> inter_occular_distances;
        computeInteroccularDistances(s_star, eye_index1, eye_index2, inter_occular_distances);
        
        std::cout << std::fixed << std::setprecision(2);
        
        for(unsigned int t=0; t<T_; t++) {
            double time = std::time(NULL);
            
            std::cout << "Starting iteration #" << t+1 << std::endl;
            
            for(int i=0; i<training_set_size; i++)
                delta_s[i] = s_star[i] - s[i];
            std::cout << "Mean error: " << computeMeanError(delta_s, inter_occular_distances) << "%" << std::endl;
            
            
            std::cout << "Computing transformations..." << std::endl;
            computeTransformations(s);
            std::vector<cv::Mat> normalized_delta_s;
            computeNormalizedResiduals(delta_s, normalized_delta_s);
            
            
            std::cout << "Building forest regressors..." << std::endl;
            for(int l=0; l<L; l++)
                forest_regressor_training_.train(forest_regressors[t][l], l, s, normalized_delta_s, imgs);
            
            
            std::cout << "Computing global regressor..." << std::endl;
            std::vector<SparseVector> lbfs;
            lbfs.reserve(training_set_size);
            SparseVector vector;
            vector.resize(L*N_);
            
            for(int i=0; i<training_set_size; i++) {
                lbfs.push_back(vector);
                for(int l=0; l<L; l++) {
                    const float *landmark = s[i].ptr<float>(l);
                    ((ForestRegressorLibLinearFormat*) &forest_regressors[t][l])->getLocalBinaryFeatureVector(lbfs[i].begin()+l*N_, landmark, imgs[i], transformations_[i]);
                }
            }
            
            std::pair<std::vector<float>, std::vector<float> > regressor;
            learnGlobalRegressor(lbfs.size(), &lbfs[0], &normalized_delta_s[0], regressor);
            
            std::cout << "Updating forest regressors with global regressor values..." << std::endl;
            updateForestRegressorsWithGlobalRegressor(L, forest_regressors[t], regressor);
            
            
            std::cout << "Updating shapes..." << std::endl;
            for(int i=0; i<training_set_size; i++)
                alignment_method_->updateRule(t, s[i], imgs[i], transformations_[i]);
            
            std::cout << "Finished in " << (std::time(NULL) - time)/60.f << " min" << std::endl << std::endl;
        }
        
        for(int i=0; i<training_set_size; i++)
            delta_s[i] = s_star[i] - s[i];
        std::cout << "Final mean error: " << computeMeanError(delta_s, inter_occular_distances) << "%" << std::endl;
        
        std::cout << "Saving model to " << output_filename << " ..." << std::endl;
        alignment_method_->saveModel(output_filename);
        
        for(unsigned int t=0; t<T_; t++)
            delete[] forest_regressors[t];
        delete[] forest_regressors;
    }

private:
    void updateForestRegressorsWithGlobalRegressor(int L, ForestRegressor *forest_regressors, const std::pair<std::vector<float>, std::vector<float> >& regressor)
    {
        for(int l=0; l<L; l++) {
            std::vector<cv::Mat>& lookup_table = forest_regressors[l].getLookupTable();
            
            for(unsigned int n=0; n<N_; n++) {
                for(int i=0; i<(1<<D_); i++) {
                    cv::Mat global_regressor_value(L, 2, CV_32FC1);
                    
                    for(int l2=0; l2<L; l2++) {
                        float* landmark = global_regressor_value.ptr<float>(l2);
                        landmark[0] = regressor.first[L*N_*(1<<D_)*l2 + l*N_*(1<<D_) + n*(1<<D_) + i];
                        landmark[1] = regressor.second[L*N_*(1<<D_)*l2 + l*N_*(1<<D_) + n*(1<<D_) + i];
                    }
                    
                    global_regressor_value.copyTo(lookup_table[n*(1<<D_) + i]);
                }
            }
        }
    }
    
    void learnGlobalRegressor(int nr_samples, const SparseVector* x, const cv::Mat* y, std::pair<std::vector<float>, std::vector<float> >& output)
    {
        int L = y[0].rows;
        int nr_features = L*N_*(1<<D_);
        
        output.first.clear();
        output.second.clear();
        output.first.reserve(nr_features*L);
        output.second.reserve(nr_features*L);
        
        LibLinearRegression liblinear_regression(nr_samples, nr_features);
        
        liblinear_regression.setProblemParameters(1);
        trainModelForAllLandmarks(liblinear_regression, x, y, output);
    }
    
    void trainModelForAllLandmarks(LibLinearRegression& liblinear_regression, const SparseVector* x, const cv::Mat* y, std::pair<std::vector<float>, std::vector<float> >& output)
    {
        int nr_samples = liblinear_regression.getSamplesNumber();
        int nr_features = liblinear_regression.getFeaturesNumber();
        int L = y[0].rows;
        
        #pragma omp parallel for
        for(int l=0; l<L; l++) {
            Node** x_array = new Node*[nr_samples];
            for(int i=0; i<nr_samples; i++) {
                x_array[i] = new Node[L*N_+1];
                std::copy(x[i].begin(), x[i].end(), x_array[i]);
                
                Node final;
                final.index = -1;
                final.value = 0.0;
                x_array[i][L*N_] = final;
            }
            
            double y0[nr_samples], y1[nr_samples];
            for(int i=0; i<nr_samples; i++) {
                const float *residual = y[i].ptr<float>(l);
                y0[i] = (double) residual[0];
                y1[i] = (double) residual[1];
            }

            liblinear_regression.trainModelWithTrainingData(x_array, y0, &output.first[nr_features*l]);
            liblinear_regression.trainModelWithTrainingData(x_array, y1, &output.second[nr_features*l]);
            
            for(int i=0; i<nr_samples; i++)
                delete[] x_array[i];
            delete[] x_array;
        }
    }
    
};

#endif

