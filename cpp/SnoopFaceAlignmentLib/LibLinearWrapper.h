#ifndef LIBLINEAR_WRAPPER_H
#define LIBLINEAR_WRAPPER_H

#include <iostream>
#include <stdexcept>

namespace liblinear {
    #include <liblinear/linear.h>
}


typedef liblinear::feature_node Node;


class LibLinearRegression
{
public:
    LibLinearRegression(int nr_samples, int nr_features, bool bias = false)
    {
        prob_.l = nr_samples;
        prob_.n = bias ? nr_features+1 : nr_features;
        prob_.bias = bias ? 1.0 : -1.0;
        
        //param_.solver_type = liblinear::L2R_L2LOSS_SVR_DUAL;
        param_.solver_type = liblinear::L2R_L2LOSS_SVR;
        param_.eps = 1e-2;
        param_.C = 1;
        param_.nr_weight = 0;
        param_.weight_label = NULL;
        param_.weight = NULL;
        param_.p = 0;
        
        liblinear::set_print_string_function(&LibLinearRegression::null_print_func);
    }
    
    ~LibLinearRegression()
    {
        liblinear::destroy_param(&param_);
    }
    
    void setProblemParameters(double C)
    {
        param_.C = C;
    }
    
    void setProblemInput(Node** x)
    {
        prob_.x = x;
    }
    
    void setProblemOutput(double* y)
    {
        prob_.y = y;
    }
    
    int getSamplesNumber() const
    {
        return prob_.l;
    }
    
    int getFeaturesNumber() const
    {
        return prob_.n;
    }
    
    void trainModelWithTrainingData(Node** x, double* y, double* model) const
    {
        liblinear::problem prob = prob_;
        prob.x = x;
        prob.y = y;
        
        if(liblinear::check_parameter(&prob, &param_) != NULL) {
            std::cout << liblinear::check_parameter(&prob_, &param_) << std::endl;
            throw std::runtime_error("Wrong set of parameters given to LibLinear");
        }
        
        liblinear::model *m = liblinear::train(&prob, &param_);
        std::copy(m->w, m->w+prob.n, model);
        
        liblinear::free_and_destroy_model(&m);
    }
    
    void trainModel(double* model) const
    {
        if(liblinear::check_parameter(&prob_, &param_) != NULL) {
            std::cout << liblinear::check_parameter(&prob_, &param_) << std::endl;
            throw std::runtime_error("Wrong set of parameters given to LibLinear");
        }
        
        liblinear::model *m = liblinear::train(&prob_, &param_);
        std::copy(m->w, m->w+prob_.n, model);
        
        liblinear::free_and_destroy_model(&m);
    }
    
    void performCrossValidation(int nr_fold, double *target) const
    {
        if(liblinear::check_parameter(&prob_, &param_) != NULL) {
            std::cout << liblinear::check_parameter(&prob_, &param_) << std::endl;
            throw std::runtime_error("Wrong set of parameters given to LibLinear");
        }
        
        liblinear::cross_validation(&prob_, &param_, nr_fold, target);
    }
    
private:
    static void null_print_func(const char* str) {}
    
    liblinear::problem prob_;
    liblinear::parameter param_;
};

#endif