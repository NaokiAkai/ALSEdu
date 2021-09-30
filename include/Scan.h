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

#ifndef __SCAN_H__
#define __SCAN_H__

#include <vector>

namespace als {

class Scan {
private:
    // スキャンのパラメータ
    double angleMin_, angleMax_, angleIncrement_;
    double rangeMin_, rangeMax_;
    // 距離データ
    std::vector<double> ranges_;
    // 距離データの数
    int scanNum_;

    void resizeRanges(void) {
        scanNum_ = (int)((angleMax_ - angleMin_) / angleIncrement_) + 1;
        ranges_.resize(scanNum_, 0.0);
    }

public:
    Scan(void):
        angleMin_(-135.0 * M_PI / 180.0),
        angleMax_(135.0 * M_PI / 180.0),
        angleIncrement_(2.0 * M_PI / 180.0),
        rangeMin_(0.05),
        rangeMax_(10.0)
    {
        resizeRanges();
    }

    void setScanParams(double angleMin, double angleMax, double angleIncrement, double rangeMin, double rangeMax) {
        angleMin_ = angleMin;
        angleMax_ = angleMax;
        angleIncrement_ = angleIncrement;
        rangeMin_ = rangeMin;
        rangeMax_ = rangeMax;
        resizeRanges();
    }

    inline void setRange(int i, double r) { ranges_[i] = r; }
    inline void setRanges(std::vector<double> ranges) { ranges_ = ranges; }

    inline int getScanNum(void) { return scanNum_; }
    inline double getAngleMin(void) { return angleMin_; }
    inline double getAngleMax(void) { return angleMax_; }
    inline double getAngleIncrement(void) { return angleIncrement_; }
    inline double getRangeMin(void) { return rangeMin_; }
    inline double getRangeMax(void) { return rangeMax_; }
    inline double getRange(int i) { return ranges_[i]; }
    inline std::vector<double> getRanges(void) { return ranges_; }
}; // class Scan

}; // namespace als

#endif // __SCAN_H__
