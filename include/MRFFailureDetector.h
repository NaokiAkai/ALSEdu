/****************************************************************************
 * ALSEdu: Advanced 2D Localization Systems for educational use
 * Copyright (C) 2019-2021 NAoki Akai
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at https://mozilla.org/MPL/2.0/.
 *
 * @author Naoki Akai
 ****************************************************************************/

#ifndef __MRF_FAILURE_DETECTOR_H__
#define __MRF_FAILURE_DETECTOR_H__

#include <cmath>
#include <vector>

namespace als {

enum MeasurementClass {
    ALIGNED = 0,
    MISALIGNED = 1,
    UNKNOWN = 2
};

class MRFFD {
private:
    // 考慮する残差の最大値
    double maxResidualError_;

    // 尤度分布のパラメータ
    double NDMean_, NDVar_, NDnormConst_, EDLambda_;

    // パラメータ
    // 位置推定正誤判断を行うための最小の残差の数
    int minValidResidualErrorNum_;
    // ループ有り信念伝播により確率更新を行う最大の回数
    int maxLPBComputationNum_;
    // 位置推定に失敗している確率を求めるサンプリングの回数
    int samplingNum_;
    // 残差の解像度
    double residualErrorReso_;
    // 位置推定に失敗していると判断するミスマッチ，未知観測率の閾値
    double misalignmentRatioThreshold_, unknownRatioThreshold_;
    // 隠れ変数間の遷移行列
    std::vector<double> transitionProbMat_;

    // 結果
    // 各クラスの属性（0: マッチ，1: ミスマッチ，2: 未知障害物）
    std::vector<std::vector<double> > measurementClassProbabilities_;
    // 位置推定に失敗している確率
    double failureProbability_;

    // スキャン点がマッチしている場合の尤度の計算
    inline double calculateNormalDistribution(double e) {
        // 残差は正の範囲でしか定義されていないので正規分布が2倍されていることに注意
        return (0.95 * (2.0 * NDnormConst_ * exp(-((e - NDMean_) * (e - NDMean_)) / (2.0 * NDVar_))) + 0.05 * (1.0 / maxResidualError_)) * residualErrorReso_;
    }

    // スキャン点がミスマッチしている場合の尤度の計算
    inline double calculateExponentialDistribution(double e) {
        return (0.95 * (1.0 / (1.0 - exp(-EDLambda_ * maxResidualError_))) * EDLambda_ * exp(-EDLambda_ * e) + 0.05 * (1.0 / maxResidualError_)) * residualErrorReso_;
    }

    // スキャン点が未知障害物である場合の尤度の計算
    inline double calculateUniformDistribution(void) {
        return (1.0 / maxResidualError_) * residualErrorReso_;
    }

    // ベクトルの和を計算
    inline double getSumOfVecotr(std::vector<double> vector) {
        double sum = 0.0;
        for (int i = 0; i < (int)vector.size(); i++)
            sum += vector[i];
        return sum;
    }

    // ベクトル同士のアダマール積を計算
    inline std::vector<double> getHadamardProduct(std::vector<double> vector1, std::vector<double> vector2) {
        for (int i = 0; i < (int)vector1.size(); i++)
            vector1[i] *= vector2[i];
        return vector1;
    }

    // ベクトルの正規化
    inline std::vector<double> normalizeVector(std::vector<double> vector) {
        double sum = getSumOfVecotr(vector);
        for (int i = 0; i < (int)vector.size(); i++)
            vector[i] /= sum;
        return vector;
    }

    // ベクトル間のユーグリッドノルムを計算
    inline double getEuclideanNormOfDiffVectors(std::vector<double> vector1, std::vector<double> vector2) {
        double sum = 0.0;
        for (int i = 0; i < (int)vector1.size(); i++) {
            double diff = vector1[i] - vector2[i];
            sum += diff * diff;
        }
        return sqrt(sum);
    }

    // 確率変数間を伝播するメッセージの計算
    inline std::vector<double> calculateTransitionMessage(std::vector<double> probs) {
        std::vector<double> message(3);
        std::vector<double> tm = transitionProbMat_;
        message[ALIGNED] = tm[ALIGNED] * probs[ALIGNED] + tm[MISALIGNED] * probs[MISALIGNED] + tm[UNKNOWN] * probs[UNKNOWN];
        message[MISALIGNED] = tm[ALIGNED + 3] * probs[ALIGNED] + tm[MISALIGNED + 3] * probs[MISALIGNED] + tm[UNKNOWN + 3] * probs[UNKNOWN];
        message[UNKNOWN] = tm[ALIGNED + 6] * probs[ALIGNED] + tm[MISALIGNED + 6] * probs[MISALIGNED] + tm[UNKNOWN + 6] * probs[UNKNOWN];
        return message;
    }

    // 有効でない残差を除外
    std::vector<double> getValidResidualErrors(std::vector<double> residualErrors) {
        std::vector<double> validResidualErrors;
        for (int i = 0; i < (int)residualErrors.size(); i++) {
            if (0.0 <= residualErrors[i] && residualErrors[i] <= maxResidualError_)
                validResidualErrors.push_back(residualErrors[i]);
        }
        return validResidualErrors;
    }

    // 残差に対する尤度ベクトルを計算
    std::vector<std::vector<double> > getLikelihoodVectors(std::vector<double> validResidualErrors) {
        std::vector<std::vector<double> > likelihoodVectors((int)validResidualErrors.size());
        double pud = calculateUniformDistribution();
        for (int i = 0; i < (int)likelihoodVectors.size(); i++) {
            likelihoodVectors[i].resize(3);
            likelihoodVectors[i][ALIGNED] = calculateNormalDistribution(validResidualErrors[i]);
            likelihoodVectors[i][MISALIGNED] = calculateExponentialDistribution(validResidualErrors[i]);
            likelihoodVectors[i][UNKNOWN] = pud;
            likelihoodVectors[i] = normalizeVector(likelihoodVectors[i]);
        }
        return likelihoodVectors;
    }

    // 隠れ変数の事後分布を推定
    std::vector<std::vector<double> > estimateMeasurementClassProbabilities(std::vector<std::vector<double>> likelihoodVectors) {
        // 全ての潜在変数が繋がっている変数からのメッセージを受け取ることで，周辺事後分布を初期化する
        std::vector<std::vector<double> > measurementClassProbabilities = likelihoodVectors;
        for (int i = 0; i < (int)measurementClassProbabilities.size(); i++) {
            for (int j = 0; j < (int)measurementClassProbabilities.size(); j++) {
                if (i == j)
                    continue;
                std::vector<double> message = calculateTransitionMessage(likelihoodVectors[j]);
                measurementClassProbabilities[i] = getHadamardProduct(measurementClassProbabilities[i], message);
                measurementClassProbabilities[i] = normalizeVector(measurementClassProbabilities[i]);
            }
            measurementClassProbabilities[i] = normalizeVector(measurementClassProbabilities[i]);
        }

        // ループ有り信念伝播の実行
        double variation = 0.0;
        int idx1 = rand() % (int)measurementClassProbabilities.size();
        std::vector<double> message(3);
        message = likelihoodVectors[idx1];
        int checkStep = maxLPBComputationNum_ / 20;
        for (int i = 0; i < maxLPBComputationNum_; i++) {
            // 伝播する次の変数のインデックス
            int idx2 = rand() % (int)measurementClassProbabilities.size();
            int cnt = 0;
            for (;;) {
                if (idx2 != idx1)
                    break;
                idx2 = rand() % (int)measurementClassProbabilities.size();
                cnt++;
                if (cnt >= 10)
                    break;
            }
            message = calculateTransitionMessage(message);
            message = getHadamardProduct(likelihoodVectors[idx2], message);
            std::vector<double> measurementClassProbabilitiesPrev = measurementClassProbabilities[idx2];
            measurementClassProbabilities[idx2] = getHadamardProduct(measurementClassProbabilities[idx2], message);
            measurementClassProbabilities[idx2] = normalizeVector(measurementClassProbabilities[idx2]);
            double diffNorm = getEuclideanNormOfDiffVectors(measurementClassProbabilities[idx2], measurementClassProbabilitiesPrev);
            variation += diffNorm;
            if (i >= checkStep && i % checkStep == 0 && variation < 10e-6)
                break;
            else if (i >= checkStep && i % checkStep == 0)
                variation = 0.0;
            message = measurementClassProbabilities[idx2];
            idx1 = idx2;
        }
        return measurementClassProbabilities;
    }

    // 位置推定に失敗している確率の計算（厳密計算は困難なのでサンプリングで近似計算）
    double predictFailureProbabilityBySampling(std::vector<std::vector<double>> measurementClassProbabilities) {
        int failureCnt = 0;
        for (int i = 0; i < samplingNum_; i++) {
            int misalignedNum = 0, validMeasurementNum = 0;
            int measurementNum = (int)measurementClassProbabilities.size();
            for (int j = 0; j < measurementNum; j++) {
                double darts = (double)rand() / ((double)RAND_MAX + 1.0);
                // validProb は未知障害物観測でない確率に相当
                double validProb = measurementClassProbabilities[j][ALIGNED] + measurementClassProbabilities[j][MISALIGNED];
                if (darts > validProb)
                    continue;
                validMeasurementNum++;
                // 正対応（ALIGNED）より確率が高い場合は誤対応（MISALIGNED）となる
                if (darts > measurementClassProbabilities[j][ALIGNED])
                    misalignedNum++;
            }
            double misalignmentRatio = (double)misalignedNum / (double)validMeasurementNum;
            double unknownRatio = (double)(measurementNum - validMeasurementNum) / (double)measurementNum;
            // ミスマッチ，未知観測率のどちらかが閾値を超えた場合を推定に失敗した状況としてカウント
            if (misalignmentRatio >= misalignmentRatioThreshold_ || unknownRatio >= unknownRatioThreshold_)
                failureCnt++;
        }
        failureProbability_ = (double)failureCnt / (double)samplingNum_;
        return failureProbability_;
    }

    // 各スキャン点に対する事後分布をセット
    void setAllMeasurementClassProbabilities(std::vector<double> residualErrors, std::vector<std::vector<double> > measurementClassProbabilities) {
        measurementClassProbabilities_.resize((int)residualErrors.size());
        int idx = 0, size = (int)measurementClassProbabilities_.size();
        for (int i = 0; i < size; i++) {
            measurementClassProbabilities_[i].resize(3);
            if (0.0 <= residualErrors[i] && residualErrors[i] <= maxResidualError_) {
                measurementClassProbabilities_[i] = measurementClassProbabilities[idx];
                idx++;
            } else {
                measurementClassProbabilities_[i][ALIGNED] = 0.00005;
                measurementClassProbabilities_[i][MISALIGNED] = 0.00005;
                measurementClassProbabilities_[i][UNKNOWN] = 0.9999;
            }
        }
    }

public:
    MRFFD():
        NDMean_(0.0),
        NDVar_(0.04),
        EDLambda_(2.0),
        maxResidualError_(1.0),
        residualErrorReso_(0.01),
        minValidResidualErrorNum_(10),
        maxLPBComputationNum_(1000),
        samplingNum_(1000),
        misalignmentRatioThreshold_(0.1),
        unknownRatioThreshold_(0.7)
    {
        NDnormConst_ = 1.0 / sqrt(2.0 * M_PI * NDVar_);

        transitionProbMat_.resize(9);
        transitionProbMat_[0] = 0.8,       transitionProbMat_[1] = 0.0,       transitionProbMat_[2] = 0.2;
        transitionProbMat_[3] = 0.0,       transitionProbMat_[4] = 0.8,       transitionProbMat_[5] = 0.2;
        transitionProbMat_[6] = 1.0 / 3.0, transitionProbMat_[7] = 1.0 / 3.0, transitionProbMat_[8] = 1.0 / 3.0;
    }

    ~MRFFD() {};

    inline void setMaxResidualError(double maxResidualError) { maxResidualError_ = maxResidualError; }
    inline void setNDMean(double NDMean) { NDMean_ = NDMean; }
    inline void setNDVariance(double NDVar) { NDVar_ = NDVar, NDnormConst_ = 1.0 / sqrt(2.0 * M_PI * NDVar_); }
    inline void setEDLambda(double EDLambda) { EDLambda_ = EDLambda; }
    inline void setResidualErrorReso(double residualErrorReso) { residualErrorReso_ = residualErrorReso; }
    inline void setMinValidResidualErrorNum(int minValidResidualErrorNum) { minValidResidualErrorNum_ = minValidResidualErrorNum; }
    inline void setMaxLPBComputationNum(int maxLPBComputationNum) { maxLPBComputationNum_ = maxLPBComputationNum; }
    inline void setSamplingNum(int samplingNum) { samplingNum_ = samplingNum; }
    inline void setMisalignmentRatioThreshold(double misalignmentRatioThreshold) { misalignmentRatioThreshold_ = misalignmentRatioThreshold; }
    inline void setUnknownRatioThreshold(double unknownRatioThreshold) { unknownRatioThreshold_ = unknownRatioThreshold; }
    inline void setTransitionProbMat(std::vector<double> transitionProbMat) { transitionProbMat_ = transitionProbMat; }

    inline double getFailureProbability(void) { return failureProbability_; }
    inline double getMeasurementClassProbabilities(int errorIndex, int measurementClass) { return measurementClassProbabilities_[errorIndex][measurementClass]; }
    inline std::vector<double> getMeasurementClassProbabilities(int errorIndex) { return measurementClassProbabilities_[errorIndex]; }

    // 各残差のクラスを取得
    std::vector<int> getResidualErrorClasses(void) {
        std::vector<int> residualErrorClasses;
        for (int i = 0; i < (int)measurementClassProbabilities_.size(); i++) {
            double alignedProb = measurementClassProbabilities_[i][ALIGNED];
            double misalignedProb = measurementClassProbabilities_[i][MISALIGNED];
            double unknownProb = measurementClassProbabilities_[i][UNKNOWN];
            if (alignedProb > misalignedProb && alignedProb > unknownProb)
                residualErrorClasses.push_back(ALIGNED);
            else if (misalignedProb > alignedProb && misalignedProb > unknownProb)
                residualErrorClasses.push_back(MISALIGNED);
            else
                residualErrorClasses.push_back(UNKNOWN);
        }
        return residualErrorClasses;
    }

    // 位置推定に失敗している確率を予測
    void predictFailureProbability(std::vector<double> residualErrors) {
        std::vector<double> validResidualErrors = getValidResidualErrors(residualErrors);
        if ((int)validResidualErrors.size() < minValidResidualErrorNum_) {
            std::cerr << "WARNING: Number of validResidualErrors is less than the expected threshold number." <<
                " The threshold is " << minValidResidualErrorNum_ <<
                ", but the number of validResidualErrors " << (int)validResidualErrors.size() << "." << std::endl;
            failureProbability_ = -1.0;
            return;
        }
        std::vector<std::vector<double> > likelihoodVectors = getLikelihoodVectors(validResidualErrors);
        std::vector<std::vector<double> > measurementClassProbabilities = estimateMeasurementClassProbabilities(likelihoodVectors);
        setAllMeasurementClassProbabilities(residualErrors, measurementClassProbabilities);
        predictFailureProbabilityBySampling(measurementClassProbabilities);
    }

    void printFailureProbability(void) {
        std::cout << "Failure probability = " << failureProbability_ * 100.0 << std::endl;
    }
}; // class MRFFD

} // namespace als

#endif // __MRF_FAILURE_DETECTOR_H__
