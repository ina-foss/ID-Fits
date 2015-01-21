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


#ifndef STATS_H
#define STATS_H

#include <ostream>
#include <fstream>
#include <map>
#include <vector>
#include <cmath>
#include <iostream>

using namespace std;


class Stats
{
public:
    void clear()
    {
        tp_.clear();
        fp_.clear();
        tn_.clear();
        fn_.clear();
    }
    
    void addValue(double threshold, int tp, int fp, int tn, int fn)
    {
        tp_[threshold] = tp;
        fp_[threshold] = fp;
        tn_[threshold] = tn;
        fn_[threshold] = fn;
    }
    
    double getPrecision(double threshold) {
        return tp_[threshold] / ((double) (tp_[threshold] + fp_[threshold]));
    }
    
    double getRecall(double threshold) {
        return tp_[threshold] / ((double) (tp_[threshold] + fn_[threshold]));
    }
    
    double getFPR(double threshold) {
        return fp_[threshold] / ((double) (fp_[threshold] + tn_[threshold]));
    }
    
    double getFNR(double threshold) {
        return fn_[threshold] / ((double) (fn_[threshold] + tp_[threshold]));
    }
    
    double getAccuracy(double threshold) {
        return (tp_[threshold] + tn_[threshold]) / ((double) (tp_[threshold] + tn_[threshold] + fp_[threshold] + fn_[threshold]));
    }
    
    pair<map<double,double>,map<double,double> > getROC() {
        map<double, double> fpr, tpr;
        double threshold;
        for(map<double,int>::iterator it = tp_.begin(); it != tp_.end(); it++) {
            threshold = it->first;
            fpr[threshold] = getFPR(threshold);
            tpr[threshold] = getRecall(threshold);
        }
        
        return make_pair(fpr,tpr);
    }
    
    void save(const string& filename = "stats.txt")
    {
        ofstream fs(filename.c_str());
        fs << tp_.size() << endl;
        fs << "Threshold" << '\t' << "TP" << '\t' << "FP" << '\t' << "FN" << '\t' << "TN" << endl;
        
        for(map<double,int>::iterator it = tp_.begin(); it != tp_.end(); it++)
            fs << it->first << '\t' << tp_[it->first] << '\t' << fp_[it->first] << '\t' << fn_[it->first] << '\t' << tn_[it->first] << endl;
    }
    
private:
    map<double,int> tp_, fp_, tn_, fn_;
};



class PlotValues
{
public:
    PlotValues(const std::string& sampling_name, const std::string& x_axis_name, const std::string& y_axis_name): x_values_sets(10), y_values_sets(10)
    {
        sampling_name_ = sampling_name;
        x_axis_name_ = x_axis_name;
        y_axis_name_ = y_axis_name;
    }
    
    void addSetValues(int i, pair<map<double,double>,map<double,double> > values)
    {
        x_values_sets[i] = values.first;
        y_values_sets[i] = values.second;
    }
    
    void addSetValues(int i, map<double,double> x_values, map<double,double> y_values)
    {
        x_values_sets[i] = x_values;
        y_values_sets[i] = y_values;
    }

    unsigned int getSamplesNumber()
    {
        return (unsigned int) x_values_sets[0].size();
    }

    ostream& toStream(ostream& os) const
    {
        map<double,double> x_values = averageOverSets(x_values_sets);
        map<double,double> y_values = averageOverSets(y_values_sets);
        
        os << sampling_name_ << '\t' << x_axis_name_ << '\t' << y_axis_name_ << endl;
        for(map<double,double>::const_iterator it = x_values.begin(); it != x_values.end(); it++)
            os << it->first << '\t' << x_values[it->first] << '\t' << y_values[it->first] << endl;
        
        return os;
    }
    
private:
    map<double,double> averageOverSets(vector<map<double,double> > stats) const
    {
        map<double,double> mean;
        
        for(unsigned int i=0; i<stats.size(); i++) {
            for(map<double,double>::const_iterator it = stats[i].begin(); it != stats[i].end(); it++)
                mean[it->first] += it->second;
        }
        
        for(map<double,double>::const_iterator it = mean.begin(); it != mean.end(); it++)
            mean[it->first] /= (double) stats.size();
        
        return mean;
    }
    
    std::string sampling_name_, x_axis_name_, y_axis_name_;
    vector<map<double,double> > x_values_sets, y_values_sets;
};


class ROC: public PlotValues
{
public:
    ROC(): PlotValues("Threshold", "FPR", "TPR") {}
};


ostream& operator<<(std::ostream& os, const PlotValues& plot_values);


#endif
