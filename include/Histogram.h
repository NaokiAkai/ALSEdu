/****************************************************************************
 * ALSEdu: Advanced 2D Localization Systems for educational use
 * Copyright (C) 2019-2021 Naoki Akai
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * @author Naoki Akai
 ****************************************************************************/

#ifndef __HISTOGRAM_H__
#define __HISTOGRAM_H__

#include <iostream>
#include <vector>
#include <algorithm>

namespace als {

class Histogram {
private:
    double binWidth_;
    double minValue_, maxValue_;
    int histogramSize_, valueNum_;
    std::vector<int> histogram_;
    std::vector<double> probability_;

    inline int val2bin(double val) {
        int bin = (int)((val - minValue_) / binWidth_);
        if (bin >= histogramSize_)
            bin = -1;
        return bin;
    }

    inline double bin2val(int bin) {
        return (double)bin * binWidth_ + minValue_;
    }

    void buildHistogram(std::vector<double> values) {
        histogramSize_ = (int)((maxValue_ - minValue_) / binWidth_);

        histogram_.resize(histogramSize_, 0);
        valueNum_ = 0;
        for (size_t i = 0; i < values.size(); i++) {
            int bin = val2bin(values[i]);
            if (bin >= 0) {
                histogram_[bin]++;
                valueNum_++;
            }
        }

        probability_.resize(histogramSize_, 0.0);
        for (size_t i = 0; i < histogram_.size(); i++)
            probability_[i] = (double)histogram_[i] / (double)valueNum_;
    }

public:
    Histogram(void) {}

    Histogram(std::vector<double> values, double binWidth):
        binWidth_(binWidth)
    {
        minValue_ = *std::min_element(values.begin(), values.end());
        maxValue_ = *std::max_element(values.begin(), values.end());
        buildHistogram(values);
    }

    Histogram(std::vector<double> values, double binWidth, double minValue, double maxValue):
        binWidth_(binWidth),
        minValue_(minValue),
        maxValue_(maxValue)
    {
        buildHistogram(values);
    }

    inline double getProbability(double value) {
        if (value >= maxValue_)
            value -= binWidth_;
        int bin = val2bin(value);
        if (bin >= 0)
            return probability_[bin];
        else
            return -1.0;
    }

    void printHistogram(void) {
        for (size_t i = 0; i < histogram_.size(); i++)
            printf("bin = %d: val = %lf, histogram[%d] = %d, probability[%d] = %lf\n", 
                (int)i, bin2val(i), (int)i, histogram_[i], (int)i, probability_[i]);
    }

}; // class Histogram

} // namespace als

#endif // __HISTOGRAM_H__
