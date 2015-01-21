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


#ifndef HDF5
#define HDF5

#include <vector>
#include <opencv2/opencv.hpp>
#include <H5Cpp.h>
#include <boost/multi_array.hpp>
#include <ctype.h>

using namespace H5;

typedef std::vector<cv::Mat> MatVector;


class HDF5Manager {
public:
    HDF5Manager(const std::string& filename, const std::string& flag)
    {
        int h5_flag;
        if(flag == "r")
            h5_flag = H5F_ACC_RDONLY;
        else if(flag == "w")
            h5_flag = H5F_ACC_TRUNC;
        else
            throw;
        
        file_ = H5File(filename, h5_flag);
    }
    
    void loadMatVector(const std::string& dataset_name, MatVector& dst)
    {
        DataSet dataset = file_.openDataSet(dataset_name);
        DataSpace dataspace = dataset.getSpace();
        
        int ndims = dataspace.getSimpleExtentNdims();
        
        int type = 0;
        if(dataset.getDataType() == PredType::NATIVE_UINT8)
            type = CV_8UC1;
        else if(dataset.getDataType() == PredType::NATIVE_DOUBLE)
            type = CV_64FC1;
        else if(dataset.getDataType() == PredType::NATIVE_FLOAT)
            type = CV_32FC1;
        else {
            std::cout << "unknown type" << std::endl;
            throw;
        }

        if(ndims == 2) {
            hsize_t dims[2];
            dataspace.getSimpleExtentDims(dims);
            boost::multi_array<double, 2> array(boost::extents[dims[0]][dims[1]]);    
            
            dataset.read(array.data(), dataset.getDataType());
            
            dst.clear();
            for(int i=0; i<(int)dims[0]; i++) {
                cv::Mat vec = cv::Mat((int) dims[1], 1, type, array.data() + i*((int)dims[1]));
                dst.push_back(cv::Mat());
                vec.copyTo(dst[i]);
            }
        }
        else if(ndims == 3) {
            hsize_t dims[3];
            dataspace.getSimpleExtentDims(dims);
            
            if(type == CV_8UC1) {
                boost::multi_array<uint8_t, 3> array(boost::extents[dims[0]][dims[1]][dims[2]]);
                
                dataset.read(array.data(), dataset.getDataType());
                
                dst.clear();
                for(int i=0; i<(int)dims[0]; i++) {
                    cv::Mat img = cv::Mat((int) dims[1], (int) dims[2], type, array.data() + i*((int)dims[1]*(int)dims[2]));
                    dst.push_back(cv::Mat());
                    img.copyTo(dst[i]);
                }
            }
            else if(type == CV_32FC1) {
                boost::multi_array<float, 3> array(boost::extents[dims[0]][dims[1]][dims[2]]);
                
                dataset.read(array.data(), dataset.getDataType());
                
                dst.clear();
                for(int i=0; i<(int)dims[0]; i++) {
                    cv::Mat img = cv::Mat((int) dims[1], (int) dims[2], type, array.data() + i*((int)dims[1]*(int)dims[2]));
                    dst.push_back(cv::Mat());
                    img.copyTo(dst[i]);
                }
            }
            else if(type == CV_64FC1) {
                boost::multi_array<double, 3> array(boost::extents[dims[0]][dims[1]][dims[2]]);
                
                dataset.read(array.data(), dataset.getDataType());
                
                dst.clear();
                for(int i=0; i<(int)dims[0]; i++) {
                    cv::Mat img = cv::Mat((int) dims[1], (int) dims[2], type, array.data() + i*((int)dims[1]*(int)dims[2]));
                    dst.push_back(cv::Mat());
                    img.copyTo(dst[i]);
                }
            }
        }
        else
            throw;
    }

    void saveMatVector(const std::string& dataset_name, MatVector& src)
    {
        int ndims = src[0].cols != 1 ? 3 : 2;
        if(ndims == 2) {
            hsize_t dims[2];
            dims[0] = src.size();
            dims[1] = src[0].rows;
            
            DataSpace dataspace(2, dims);
            DataSet dataset = DataSet(file_.createDataSet(dataset_name, PredType::NATIVE_DOUBLE, dataspace));
            
            boost::multi_array<double, 2> array(boost::extents[dims[0]][dims[1]]);    
            
            for(int i=0; i<(int)dims[0]; i++)
                array[i] = boost::multi_array_ref<double, 1>((double*) src[i].data, boost::extents[dims[1]]);
            
            dataset.write(array.data(), PredType::NATIVE_DOUBLE);
        }
        else if(ndims == 3) {
            hsize_t dims[3];
            dims[0] = src.size();
            dims[1] = src[0].rows;
            dims[2] = src[0].cols;
            
            DataSpace dataspace(3, dims);
            DataSet dataset = DataSet(file_.createDataSet(dataset_name, PredType::NATIVE_DOUBLE, dataspace));

            boost::multi_array<double, 3> array(boost::extents[dims[0]][dims[1]][dims[2]]);
            for(int i=0; i<(int)dims[0]; i++)
                array[i] = boost::multi_array_ref<double, 2>((double*) src[i].data, boost::extents[dims[1]][dims[2]]);
            
            dataset.write(array.data(), PredType::NATIVE_DOUBLE);
        }
    }
    
    
private:
    H5File file_;
};

#endif