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


#include <SnoopFaceDescLib/Train.h>
#include <fstream>
#include <list>

using namespace std;


double Train::computeEERThreshold(Stats& stats)
{
    double diff = 2.0;
    double eer_threshold = 0.f;
    
    double fpr, fnr;
    for(set<double>::iterator it = thresholds.begin(); it != thresholds.end(); it++) {
        fpr = stats.getFPR(*it);
        fnr = stats.getFNR(*it);
        if(abs(fpr - fnr) < diff) {
            diff = abs(fpr - fnr);
            eer_threshold = *it;
        }
    }
    
    return eer_threshold;
}

double Train::computeThresholdForFixedRecall(Stats& stats, double wanted_recall)
{
    double diff = 2.0;
    double recall_threshold = 0.f;
    
    double recall;
    for(set<double>::iterator it = thresholds.begin(); it != thresholds.end(); it++) {
        recall = stats.getRecall(*it);
        if(abs(wanted_recall - recall) < diff) {
            diff = abs(wanted_recall - recall);
            recall_threshold = *it;
        }
    }
    
    return recall_threshold;
}


double Train::computeStatsFromDescriptorsWithThreshold(Stats& stats, const Pipeline& p, const vector<Mat>& descriptors, const vector<list<int> >& matchs, const vector<list<int> >& mismatchs, double threshold)
{
    int tp = 0, fp = 0, tn = 0, fn = 0;
    
    for(unsigned int i=0; i<descriptors.size(); i++) {
        for(list<int>::const_iterator it = matchs[i].begin(); it != matchs[i].end(); it++) {
            if(p.checkMatching(&descriptors[i], &descriptors[*it], threshold))
                tp++;
            else
                fn++;
        }
        for(list<int>::const_iterator it = mismatchs[i].begin(); it != mismatchs[i].end(); it++) {
            if(!p.checkMatching(&descriptors[i], &descriptors[*it], threshold))
                tn++;
            else
                fp++;
        }
    }
    
    stats.addValue(threshold, tp, fp, tn, fn);
    return stats.getAccuracy(threshold);
}


Stats Train::computeStatsFromDescriptors(const Pipeline& p, const vector<Mat>& descriptors, const vector<list<int> >& matchs, const vector<list<int> >& mismatchs)
{
    Stats stats;
    
    for(set<double>::iterator it = thresholds.begin(); it != thresholds.end(); it++)
        computeStatsFromDescriptorsWithThreshold(stats, p, descriptors, matchs, mismatchs, *it);
    
    return stats;
}


void Train::thresholdSampling(double step_size, double start, double end)
{
    thresholds.clear();
    
    int P = (int) ((end - start)/step_size);
    for(int i=0; i<=P; i++)
        thresholds.insert(start + step_size*i);
}



