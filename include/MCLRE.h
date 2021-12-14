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

#ifndef __MCLRE_H__
#define __MCLRE_H__

#include <vector>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <Pose.h>
#include <Particle.h>
#include <Scan.h>
#include <MCL.h>
#include <MAEClassifier.h>
#include <AdaBoostClassifier.h>
#include <MLPClassifier.h>

namespace als {

class MCLRE : public MCL {
private:
    // 正誤判断識別器のタイプ
    // 0: Real AdaBoost, 1: Multilayer Perceptron (MLP), 2: Root Mean Square (RMS)
    int classifierType_;
    // 各パーティクルに対する判断モデルで計算された尤度
    std::vector<double> decisionLikelihoods_;
    // 各パーティクルが持つ信頼度
    std::vector<double> reliabilities_;

    // 信頼度減衰モデルのパラメータ
    double relTransParam1_, relTransParam2_;

    AdaBoostClassifier boostClassifier_;
    std::vector<int> boostPredicts_;

    MLPClassifier mlpClassifier_;
    std::vector<double> mlpPredicts_;

    MAEClassifier maeClassifier_;
    std::vector<double> maes_;

public:
    MCLRE(std::string mapDir, int particleNum):
        MCL(mapDir, particleNum),
        classifierType_(0),
        relTransParam1_(0.0),
        relTransParam2_(0.0)
    {
        decisionLikelihoods_.resize(getParticleNum());
        reliabilities_.resize(getParticleNum(), 0.5);
    };

    ~MCLRE() {};

    inline void useAdaBoostClassifier(void) {
        classifierType_ = 0;
        boostClassifier_.readClassifierParams();
        boostPredicts_.resize(getParticleNum());
    }

    inline void useMLPClassifier(void) {
        classifierType_ = 1;
        mlpClassifier_.readClassifierParams();
        mlpPredicts_.resize(getParticleNum());
    }

    inline void useMAEClassifier(void) {
        classifierType_ = 2,
        maeClassifier_.readClassifierParams();
        maes_.resize(getParticleNum());
    }

    inline double getReliability(void) { return reliabilities_[getMaxLikelihoodParticleIdx()]; }

    // 動作モデルによるパーティクルの更新と信頼度遷移モデルによる信頼度の更新
    void updateParticlesAndReliability(double deltaDist, double deltaYaw) {
        double odomNoise1, odomNoise2, odomNoise3, odomNoise4;
        getOdomNoises(&odomNoise1, &odomNoise2, &odomNoise3, &odomNoise4);
        double dd2 = deltaDist * deltaDist;
        double dy2 = deltaYaw * deltaYaw;
        for (size_t i = 0; i < getParticleNum(); i++) {
            double dd = deltaDist + randNormal(
                odomNoise1 * dd2 + odomNoise2 * dy2);
            double dy = deltaYaw + randNormal(
                odomNoise3 * dd2 + odomNoise4 * dy2);
            Pose pose = getParticlePose(i);
            double yaw = pose.getYaw();
            double x = pose.getX() + dd * cos(yaw);
            double y = pose.getY() + dd * sin(yaw);
            yaw += dy;
            setParticlePose(i, Pose(x, y, yaw));
            double decayRate = relTransParam1_ * dd * dd
                                 + relTransParam2_ * dy * dy;
            if (decayRate > 0.99999)
                decayRate = 0.99999;
            double rel = (1.0 - decayRate) * reliabilities_[i];
            reliabilities_[i] = rel;
        }
    }

    // 判断モデルを計算し，その結果をパーティクルの尤度に反映させる
    void calculateDecisionModel(Scan scan) {
        setTotalLikelihood(0.0);
        double maxLikelihood = 0.0;
        int maxIdx = 0;
        for (size_t i = 0; i < getParticleNum(); i++) {
            std::vector<double> residualErrors = getResidualErrors(getParticlePose(i), scan);
            // 以下の判断モデルで尤度計算を行うと同時に，信頼度も更新する
            if (classifierType_ == 0) {
                int predict = boostClassifier_.predict(residualErrors);
                boostPredicts_[i] = predict;
                decisionLikelihoods_[i] = boostClassifier_.calculateDecisionModel(predict, &reliabilities_[i]);
            } else if (classifierType_ == 1) {
                double predict = mlpClassifier_.predict(residualErrors);
                mlpPredicts_[i] = predict;
                decisionLikelihoods_[i] = mlpClassifier_.calculateDecisionModel(predict, &reliabilities_[i]);
            } else {
                double mae = maeClassifier_.getMAE(residualErrors);
                maes_[i] = mae;
                decisionLikelihoods_[i] = maeClassifier_.calculateDecisionModel(mae, &reliabilities_[i]);
            }
            // 判断モデルと観測モデルの積がパーティクルの尤度となる
            double likelihood = decisionLikelihoods_[i] * getMeasurementLikelihood(i);
            if (i == 0) {
                maxLikelihood = likelihood;
                maxIdx = 0;
            } else if (maxLikelihood < likelihood) {
                maxLikelihood = likelihood;
                maxIdx = i;
            }
            addTotalLikelihood(likelihood);
            setParticleW(i, likelihood);
        }
        setAverageLikelihood(getTotalLikelihood() / (double)getParticleNum());
        setMaxLikelihoodParticleIdx(maxIdx);

        double sum = 0.0;
        double totalLikelihood = getTotalLikelihood();
        for (size_t i = 0; i < getParticleNum(); i++) {
            double w = getParticleW(i) / totalLikelihood;
            setParticleW(i, w);
            sum += w * w;
        }
        setEffectiveSampleSize(1.0 / sum);
    }

    void resampleParticlesAndReliability(void) {
        int particleNum = getParticleNum();
        double threshold = (double)particleNum * getResampleThreshold();
        if (getEffectiveSampleSize() > threshold)
            return;

        // wBufferを用いてリサンプリングを行う
        std::vector<double> wBuffer(particleNum);
        wBuffer[0] = getParticleW(0);
        for (int i = 1; i < particleNum; i++)
            wBuffer[i] = getParticleW(i) + wBuffer[i - 1];

        std::vector<Particle> tmpParticles = getParticles();
        std::vector<double> tmpReliabilities = reliabilities_;
        double wo = 1.0 / (double)particleNum;
        for (int i = 0; i < particleNum; i++) {
            double darts = (double)rand() / ((double)RAND_MAX + 1.0);
            for (int j = 0; j < particleNum; j++) {
                if (darts < wBuffer[j]) {
                    setParticlePose(i, tmpParticles[j].getPose());
                    reliabilities_[i] = tmpReliabilities[j];
                    setParticleW(i, wo);
                    break;
                }
            }
        }
    }

    // 推定された信頼度と最尤パーティクルに対する判断値を端末に表示
    void printReliability(void) {
        std::cout << "Reliability = " << reliabilities_[getMaxLikelihoodParticleIdx()] * 100.0 << " [%]" << std::endl;
        if (classifierType_ == 0)
            std::cout << "AdaBoost predict = " << boostPredicts_[getMaxLikelihoodParticleIdx()] << std::endl;
        else if (classifierType_ == 1)
            std::cout << "MLP predict = " << mlpPredicts_[getMaxLikelihoodParticleIdx()] * 100.0 << " [%]" << std::endl;
        else
            std::cout << "MAE = " << maes_[getMaxLikelihoodParticleIdx()] << " [m] (threshold = " << maeClassifier_.getFailureThreshold() << " [m])" << std::endl;
    }

    void writeMCLREResults(Pose gtRobotPose) {
        static unsigned int cnt = 0;
        static FILE *fp;
        if (fp == NULL) {
            fp= fopen("/tmp/mclre_results.txt", "w");
            fprintf(fp, "# cnt gtx gty gtyaw mclx mcly mclyaw position_error angle_error decision reliability\n");
        }
        Pose mclPose = getMCLPose();
        double dx = fabs(gtRobotPose.getX() - mclPose.getX());
        double dy = fabs(gtRobotPose.getY() - mclPose.getY());
        double positionError;
        if (dx >= dy)
            positionError = dx;
        else
            positionError = dy;
        double dyaw = gtRobotPose.getYaw() - mclPose.getYaw();
        while (dyaw > M_PI)
            dyaw -= 2.0 * M_PI;
        while (dyaw < -M_PI)
            dyaw += 2.0 * M_PI;
        double angleError = fabs(dyaw) * 180.0 / M_PI;
        int idx = getMaxLikelihoodParticleIdx();
        double reliability = reliabilities_[idx];
        if (classifierType_ == 0)
            fprintf(fp, "%d %lf %lf %lf %lf %lf %lf %lf %lf %d %lf\n",
                cnt, gtRobotPose.getX(), gtRobotPose.getY(), gtRobotPose.getYaw(),
                mclPose.getX(), mclPose.getY(), mclPose.getYaw(),
                positionError, angleError, boostPredicts_[idx], reliability);
        else if (classifierType_ == 1)
            fprintf(fp, "%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
                cnt, gtRobotPose.getX(), gtRobotPose.getY(), gtRobotPose.getYaw(),
                mclPose.getX(), mclPose.getY(), mclPose.getYaw(),
                positionError, angleError, mlpPredicts_[idx], reliability);
        else
            fprintf(fp, "%d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
                cnt, gtRobotPose.getX(), gtRobotPose.getY(), gtRobotPose.getYaw(),
                mclPose.getX(), mclPose.getY(), mclPose.getYaw(),
                positionError, angleError, maes_[idx], reliability);
        cnt++;
        printf("cnt = %d, time = %lf\n", cnt, (double)cnt * 0.1);
    }

}; // class MCLRE

} // namespace als

#endif // __MCLRE_H__
