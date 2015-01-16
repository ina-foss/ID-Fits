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


#ifndef TRAIN_H
#define TRAIN_H

#include <cstdlib>
#include <string>
#include <set>
#include <SnoopFaceDescLib/DatasetLoader.h>
#include <SnoopFaceDescLib/Pipeline.h>
#include <SnoopFaceDescLib/Stats.h>


using namespace std;


class Train
{
public:
    void thresholdSampling(double step_size, double start = 0.f, double end = 1.f);
    
    double computeEERThreshold(Stats& stats);
    double computeThresholdForFixedRecall(Stats& stats, double wanted_recall);
    
    double computeStatsFromDescriptorsWithThreshold(Stats& stats, const Pipeline& p, const vector<Mat>& descriptors, const vector<list<int> >& matchs, const vector<list<int> >& mismatchs, double threshold);
    Stats computeStatsFromDescriptors(const Pipeline& p, const vector<Mat>& descriptors, const vector<list<int> >& matchs, const vector<list<int> >& mismatchs);

   
    template<typename T>
    map<double,T> computeMeanOverSets(vector<map<double,T> > stats);
    
    int N;
    set<double> thresholds;
};


template<typename T>
map<double,T> Train::computeMeanOverSets(vector<map<double, T> > stats)
{
    map<double,T> mean;
    
    for(int i=0; i<stats.size(); i++) {
        for(typename map<double,T>::iterator it = stats[i].begin(); it != stats[i].end(); it++)
            mean[it->first] += it->second;
    }
    
    for(typename map<double,T>::iterator it = mean.begin(); it != mean.end(); it++)
        mean[it->first] /= (double) stats.size();
    
    return mean;
}

#endif