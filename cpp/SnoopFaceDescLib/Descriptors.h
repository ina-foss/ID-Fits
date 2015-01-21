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


#ifndef DESCRIPTORS_H
#define DESCRIPTORS_H

#include "opencv2/opencv.hpp"
#include <SnoopFaceDescLib/Pca.h>


class BaseDescriptor
{
public:
    virtual ~BaseDescriptor() {}
};


class Descriptor
{
public:
    Descriptor(int cell_size=10, int step=-1): cell_size_(cell_size)
    {
        if(step < 0)
            step_ = cell_size;
        else
            step_ = step;
    }
    
    virtual ~Descriptor() {}
    
    inline void compute(const cv::Mat& src, cv::Mat& dst, bool normalize = true) const
    {
        computeDescriptor(src, dst);
        if(normalize)
            dst /= cv::norm(dst);
    }
    
protected:
    int cell_size_, step_;
    virtual void computeDescriptor(const cv::Mat& src, cv::Mat& dst) const = 0;
};



class LbpDescriptor: public Descriptor
{
public:
    LbpDescriptor(int cell_size=10, int step=-1): Descriptor(cell_size, step)
    {
    }
    
    virtual ~LbpDescriptor() {}
    
    virtual void computeDescriptor(const cv::Mat& src, cv::Mat& dst) const
    {
        cv::Mat lbp;
        computeLbpValues(src, lbp);
        computeHist(lbp, dst, 256);
    }
    
protected:
    void computeLbpValues(const cv::Mat& src, cv::Mat& dst) const
    {
        int32_t i32_v1;
        int32_t i32_v2;
        int32_t i32_d;
        uint32_t ui32_d;
        uint32_t ui32_sb;

        dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
        
        for ( int i=1; i<src.rows-1; i++ ) {
            const uint8_t* a_line_prev = src.ptr<uint8_t> ( i-1 );
            const uint8_t* a_line_curr = src.ptr<uint8_t> ( i+0 );
            const uint8_t* a_line_next = src.ptr<uint8_t> ( i+1 );

            uint8_t* a_line_dst = dst.ptr<uint8_t> ( i );

            for ( int j=1; j<src.cols-1; j++ ) {
                uint8_t center = a_line_curr[j];
                uint32_t code = 0;
                
                i32_v1= ( int32_t ) center;
                
                //code |= (p_src->at<uint8_t>(i-1,j-1) > center) << 7;
                i32_v2= ( int32_t ) a_line_prev[j-1];
                i32_d=i32_v1-i32_v2;
                ui32_d= ( uint32_t ) i32_d;
                ui32_sb= ( ( ui32_d >> 31 ) & 1 );  
                code |= ( ( ui32_sb ) ) << 7;
                
                //code |= (p_src->at<uint8_t>(i-1,j) > center) << 6;
                i32_v2= ( int32_t ) a_line_prev[j];
                i32_d=i32_v1-i32_v2;
                ui32_d= ( uint32_t ) i32_d;
                ui32_sb= ( ( ui32_d >> 31 ) & 1 );
                code |= ( ( ui32_sb ) )  << 6;
                
                //code |= (p_src->at<uint8_t>(i-1,j+1) > center) << 5;
                i32_v2= ( int32_t ) a_line_prev[j+1];
                i32_d=i32_v1-i32_v2;
                ui32_d= ( uint32_t ) i32_d;
                ui32_sb= ( ( ui32_d >> 31 ) & 1 );
                code |= ( ( ui32_sb ) ) << 5;
                
                //code |= (p_src->at<uint8_t>(i,j-1) > center) << 0;
                i32_v2= ( int32_t ) a_line_curr[j-1];
                i32_d=i32_v1-i32_v2;
                ui32_d= ( uint32_t ) i32_d;
                ui32_sb= ( ( ui32_d >> 31 ) & 1 );
                code |= ( ( ui32_sb ) ) << 0;
                
                //code |= (p_src->at<uint8_t>(i,j+1) > center) << 4;
                i32_v2= ( int32_t ) a_line_curr[j+1];
                i32_d=i32_v1-i32_v2;
                ui32_d= ( uint32_t ) i32_d;
                ui32_sb= ( ( ui32_d >> 31 ) & 1 );
                code |= ( ( ui32_sb ) ) << 4;
                
                //code |= (p_src->at<uint8_t>(i+1,j-1) > center) << 1;
                i32_v2= ( int32_t ) a_line_next[j-1];
                i32_d=i32_v1-i32_v2;
                ui32_d= ( uint32_t ) i32_d;
                ui32_sb= ( ( ui32_d >> 31 ) & 1 );
                code |= ( ( ui32_sb ) ) << 1;
                
                //code |= (p_src->at<uint8_t>(i+1,j) > center) << 2;
                i32_v2= ( int32_t ) a_line_next[j];
                i32_d=i32_v1-i32_v2;
                ui32_d= ( uint32_t ) i32_d;
                ui32_sb= ( ( ui32_d >> 31 ) & 1 );
                code |= ( ( ui32_sb ) ) << 2;
                
                //code |= (p_src->at<uint8_t>(i+1,j+1) > center) << 3;
                i32_v2= ( int32_t ) a_line_next[j+1];
                i32_d=i32_v1-i32_v2;
                ui32_d= ( uint32_t ) i32_d;
                ui32_sb= ( ( ui32_d >> 31 ) & 1 );
                code |= ( ( ui32_sb ) ) << 3;
                
                a_line_dst[j]= ( uint8_t ) code;
                //p_dst->at<uint8_t>(i,j) = (uint8_t)code;
            }
        }
    }

    void computeHist(const cv::Mat& src, cv::Mat& hist, int range) const
    {
        int nb_cell_x = ((src.cols-2) - cell_size_) / step_ + 1;
        int nb_cell_y = ((src.rows-2) - cell_size_) / step_ + 1;
        
        hist = cv::Mat::zeros(nb_cell_x*nb_cell_y, range, CV_32FC1);
        float* hist_data = (float*) hist.data;
        
        for(int cell_y=1; cell_y<nb_cell_y*step_+1; cell_y+=step_) {
        
            for(int cell_x=1; cell_x<nb_cell_x*step_+1; cell_x+=step_) {
                
                int block_id = ((cell_y-1)/step_)*nb_cell_x + ((cell_x-1)/step_);
                
                for(int y=cell_y; y<cell_y+cell_size_; y++) {
                    const uint8_t* row = src.ptr<uint8_t>(y);
                    
                    for(int x=cell_x; x<cell_x+cell_size_; x++)
                        hist_data[block_id*range + ((int) row[x])] += 1.0f;
                }
                
            }
            
        }
    }
};



class ULbpDescriptor: public LbpDescriptor
{
public:
    ULbpDescriptor(int cell_size_=10, int step_=-1): LbpDescriptor(cell_size_, step_)
    {
        createULBPLookupTable(8);
    }
    
    virtual ~ULbpDescriptor() {}
    
    virtual void computeDescriptor(const cv::Mat& src, cv::Mat& dst) const
    {
        cv::Mat lbp;
        computeLbpValues(src, lbp);
        computeULbpHist(lbp, dst, 59);
    }
    
protected:
    void computeULbpHist(const cv::Mat& src, cv::Mat& hist, int range)  const
    {
        int nb_cell_x = ((src.cols-2) - cell_size_) / step_ + 1;
        int nb_cell_y = ((src.rows-2) - cell_size_) / step_ + 1;
        
        hist = cv::Mat::zeros(nb_cell_x*nb_cell_y, range, CV_32FC1);
        float* hist_data = (float*) hist.data;
        
        for(int cell_y=1; cell_y<nb_cell_y*step_+1; cell_y+=step_) {

            for(int cell_x=1; cell_x<nb_cell_x*step_+1; cell_x+=step_) {
                
                int block_id = ((cell_y-1)/step_)*nb_cell_x + ((cell_x-1)/step_);
                
                for(int y=cell_y; y<cell_y+cell_size_; y++) {
                    const uint8_t* row = src.ptr<uint8_t>(y);
                    
                    for(int x=cell_x; x<cell_x+cell_size_; x++) {
                        int value = ulbp_lookup_table_[row[x]];
                        hist_data[block_id*range + value] += 1.0f;
                    }
                }
                
            }
            
        }
    }
    
private:
    void createULBPLookupTable(int P)
    {
        int N = (int) pow(2,P);
        ulbp_lookup_table_ = std::vector<int>(N);
        
        int index = 1;
        for(int i=0; i<N; i++) {
            if(!isUniformLBP((uint16_t) i, P))
                ulbp_lookup_table_[i] = 0;
            else {
                ulbp_lookup_table_[i] = index;
                index++;
            }
        }
    }
    
    bool isUniformLBP(uint16_t lbp, int P) const
    {
        int transitions = 0;
        uint16_t prev = lbp & 1;
        
        for(int i=0; i<P; i++) {
            if((lbp & 1) != prev)
                transitions++;
            prev = lbp & 1;
            lbp = lbp >> 1;
        }
        
        return (transitions <= 2);
    }
    
    std::vector<int> ulbp_lookup_table_;
};


class ULbpPCADescriptor: public ULbpDescriptor
{
public:
    ULbpPCADescriptor(const Pca& pca): ULbpDescriptor(), pca_(pca)
    {
    }
    
    virtual ~ULbpPCADescriptor() {}
    
    virtual void computeDescriptor(const cv::Mat& src, cv::Mat& dst) const
    {
        cv::Mat lbp, hist;
        computeLbpValues(src, lbp);
        computeULbpHist(lbp, hist, 59);
        pca_.project(hist, dst);
    }
    
protected:
    Pca pca_;
};



class ULbpWPCADescriptor: public ULbpPCADescriptor
{
public:
    ULbpWPCADescriptor(const Pca& pca): ULbpPCADescriptor(pca)
    {
        cv::Mat eigenvalues = pca.getEigenvalues();
        wpca_ = cv::Mat(eigenvalues.rows, eigenvalues.cols, eigenvalues.type());
	cv::pow(eigenvalues, -0.5, wpca_);
    }
    
    virtual ~ULbpWPCADescriptor() {}
    
    virtual void computeDescriptor(const cv::Mat& src, cv::Mat& dst) const
    {
        cv::Mat lbp, hist, pca;
        computeLbpValues(src, lbp);
        computeULbpHist(lbp, hist, 59);
        pca_.project(hist, pca);
        dst = pca.mul(wpca_);
    }
    
private:
    cv::Mat wpca_;
};


class ULbpPCALDADescriptor: public ULbpPCADescriptor
{
public:
    ULbpPCALDADescriptor(const Pca& pca, const Lda& lda): ULbpPCADescriptor(pca), lda_(lda)
    {
    }
    
    virtual ~ULbpPCALDADescriptor() {}
    
    virtual void computeDescriptor(const cv::Mat& src, cv::Mat& dst) const
    {
        cv::Mat lbp, hist, pca;
        computeLbpValues(src, lbp);
        computeULbpHist(lbp, hist, 59);
        pca_.project(hist, pca);
	lda_.project(pca, dst);
    }
    
protected:
    Lda lda_;
};

/*
class PCAReducedDescriptor
{
public:
    PCAReducedDescriptor(const Descriptor* descriptor, const Pca& pca): descriptor_(descriptor), pca_(pca)
    {
    }
    
    void compute(const cv::Mat& src, cv::Mat& dst, bool normalize = true) const
    {
        descriptor_->compute(src, dst);
        computePCADescriptors(src, dst);
        
        if(normalize)
            dst /= cv::norm(dst);
    }
    
protected:
    void computePCADescriptors(const cv::Mat& src, cv::Mat& dst) const
    {
        int N = src.rows;
        int p = pca_.getDimension();
        dst = cv::Mat(N, p, CV_32FC1);
        
        for(int n=0; n<N; n++)
            pca_.project(src.row(n), dst.row(n));
    }
    
    Descriptor* descriptor_;
    Pca pca_;
};
*/

#endif
