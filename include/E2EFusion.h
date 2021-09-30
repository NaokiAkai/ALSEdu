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

#ifndef __E2E_FUSION_H__
#define __E2E_FUSION_H__

#include <vector>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>
#include <Pose.h>
#include <Particle.h>
#include <Scan.h>
#include <MCL.h>

namespace als {

class E2EFusion : public MCL {
private:
    // E2E で生成されるパーティクルの数
    int e2eParticleNum_;
    // E2E で生成されたパーティクル群
    std::vector<Particle> e2eParticles_;
    // 予測分布をパーティクル群で近似する際の混合ガウス分布の分散
    double sigmaX_, sigmaY_, sigmaYaw_;
    // 分布の線形結合のパラメータ（dUniform_ が0だと遠距離の誘拐に対応不可）
    double dGMM_, dUniform_;
    // MCL とE2E の融合により推定された位置
    Pose fusedPose_;
    // E2E 自己位置推定の結果
    Pose e2ePose_;

public:
    E2EFusion(std::string mapDir, int particleNum, int e2eParticleNum):
        MCL(mapDir, particleNum),
        fusedPose_(0.0, 0.0, 0.0),
        e2eParticleNum_(e2eParticleNum),
        sigmaX_(0.2),
        sigmaY_(0.2),
        sigmaYaw_(3.0 * M_PI / 180.0),
        dGMM_(0.9),
        dUniform_(0.1)
    {
        e2eParticles_.resize(e2eParticleNum_);
    };

    ~E2EFusion() {};

    inline void setPose(double x, double y, double yaw) { fusedPose_.setPose(x, y, yaw); }
    inline void setPose(Pose pose) { fusedPose_ = pose; }

    inline std::vector<Particle> getE2EParticles(void) { return e2eParticles_; }

    // 真値にノイズを加えた位置を中心とした一様分布からパーティクル群を生成
    void generateE2EParticlesFromUniformDistribution(Pose gtRobotPose, Pose gtNoise, Pose uniformRange) {
        double x = gtRobotPose.getX() + randNormal(gtNoise.getX());
        double y = gtRobotPose.getY() + randNormal(gtNoise.getY());
        double yaw = gtRobotPose.getYaw() + randNormal(gtNoise.getYaw());
        Pose mean(x, y, yaw);

        double minX = mean.getX() - uniformRange.getX();
        double maxX = mean.getX() + uniformRange.getX();
        double minY = mean.getY() - uniformRange.getY();
        double maxY = mean.getY() + uniformRange.getY();
        double minYaw = mean.getYaw() - uniformRange.getYaw();
        double maxYaw = mean.getYaw() + uniformRange.getYaw();
        for (size_t i = 0; i < e2eParticles_.size(); i++) {
            double x = (maxX - minX) * (double)rand() / ((double)RAND_MAX + 1.0) + minX;
            double y = (maxY - minY) * (double)rand() / ((double)RAND_MAX + 1.0) + minY;
            double yaw = (maxYaw - minYaw) * (double)rand() / ((double)RAND_MAX + 1.0) + minYaw;
            e2eParticles_[i].setPose(x, y, yaw);
        }
    }

    // 真値にノイズを加えた位置を平均とした正規分布からパーティクル群を生成
    void generateE2EParticlesFromNormalDistribution(Pose gtRobotPose, Pose gtNoise, Pose normVar) {
        double x = gtRobotPose.getX() + randNormal(gtNoise.getX());
        double y = gtRobotPose.getY() + randNormal(gtNoise.getY());
        double yaw = gtRobotPose.getYaw() + randNormal(gtNoise.getYaw());
        Pose mean(x, y, yaw);

        for (size_t i = 0; i < e2eParticles_.size(); i++) {
            double x = mean.getX() + randNormal(normVar.getX());
            double y = mean.getY() + randNormal(normVar.getY());
            double yaw = mean.getYaw() + randNormal(normVar.getYaw());
            e2eParticles_[i].setPose(x, y, yaw);
        }
    }

    // サンプリングされたパーティクル群の平均位置をE2E 自己位置推定で推定された位置とする
    void estimateE2EPose(void) {
        double tmpYaw = e2ePose_.getYaw();
        double x = 0.0, y = 0.0, yaw = 0.0;
        double w = 1.0 / (double)e2eParticles_.size();
        for (size_t i = 0; i < e2eParticles_.size(); i++) {
            x += e2eParticles_[i].getX() * w;
            y += e2eParticles_[i].getY() * w;
            double dyaw = tmpYaw - e2eParticles_[i].getYaw();
            while (dyaw < -M_PI)
                dyaw += 2.0 * M_PI;
            while (dyaw > M_PI)
                dyaw -= 2.0 * M_PI;
            yaw += dyaw * w;
        }
        yaw = tmpYaw - yaw;
        e2ePose_.setPose(x, y, yaw);
    }

    // MCL とE2E の融合を実行
    void executeE2EFusion(Scan scan) {
        // 予測分布から生成されたパーティクル群の尤度を計算（観測モデルのみ利用）
        std::vector<Particle> particles = getParticles();
        std::vector<double> modelLikelihoods = calculateMeasurementModelForGivenParticles(particles, scan);
        int totalParticleNum = particles.size() + e2eParticles_.size();
        double totalLikelihood = 0;
        for (size_t i = 0; i < particles.size(); i++)
            totalLikelihood += modelLikelihoods[i];

        // E2E で生成されたパーティクル群の尤度を計算（観測モデルと予測分布を利用）
        std::vector<double> e2eMeasurementLikelihoods = calculateMeasurementModelForGivenParticles(e2eParticles_, scan);

        // 予測分布を近似するパーティクル群で混合ガウスモデルを定めて予測分布を近似
        // 一様分布は適当な小さな値としてしまう
        double pUniform_ = 10e-6;
        double varX = sigmaX_ * sigmaX_;
        double varY = sigmaY_ * sigmaY_;
        double varYaw = sigmaYaw_ * sigmaYaw_;
        double normConst = 1.0 / sqrt(2.0 * M_PI * (varX + varY + varYaw));
        double mapResolution = getMapResolution();
        double angleResolution = 2.0 * M_PI / 360.0;
        std::vector<double> e2eLikelihoods;
        for (size_t i = 0; i < e2eParticles_.size(); i++) {
            // 混合ガウス分布の計算
            double gmmVal = 0.0;
            for (size_t j = 0; j < particles.size(); j++) {
                double dx = e2eParticles_[i].getX() - particles[j].getX();
                double dy = e2eParticles_[i].getY() - particles[j].getY();
                double dyaw = e2eParticles_[i].getYaw() - particles[j].getYaw();
                while (dyaw < -M_PI)
                    dyaw += 2.0 * M_PI;
                while (dyaw > M_PI)
                    dyaw -= 2.0 * M_PI;
                gmmVal += normConst * exp(-((dx * dx) / (2.0 * varX) + (dy * dy) / (2.0 * varY) + (dyaw * dyaw) / (2.0 * varYaw)));
            }
            double pGMM = gmmVal * mapResolution * mapResolution * angleResolution / (double)particles.size();
            double predLikelihood = dGMM_ * pGMM + dUniform_ * pUniform_;
            double likelihood = predLikelihood * e2eMeasurementLikelihoods[i];
            e2eLikelihoods.push_back(likelihood);
            totalLikelihood += likelihood;
        }
        double averageLikelihood = totalLikelihood / (double)totalParticleNum;
        setTotalLikelihood(totalLikelihood);
        setAverageLikelihood(averageLikelihood);

        // パーティクル群の尤度の正規化と有効サンプル数の計算
        double sum = 0.0;
        for (size_t i = 0; i < particles.size(); i++) {
            double w = modelLikelihoods[i] / totalLikelihood;
            particles[i].setW(w);
            sum += w * w;
        }
        for (size_t i = 0; i < e2eParticles_.size(); i++) {
            double w = e2eLikelihoods[i] / totalLikelihood;
            e2eParticles_[i].setW(w);
            sum += w * w;
        }
        double effectiveSampleSize = 1.0 / sum;
        setEffectiveSampleSize(effectiveSampleSize);

        // 位置の推定
        double tmpYaw = fusedPose_.getYaw();
        double x = 0.0, y = 0.0, yaw = 0.0;
        for (size_t i = 0; i < particles.size(); i++) {
            double w = particles[i].getW();
            x += particles[i].getX() * w;
            y += particles[i].getY() * w;
            double dyaw = tmpYaw - particles[i].getYaw();
            while (dyaw < -M_PI)
                dyaw += 2.0 * M_PI;
            while (dyaw > M_PI)
                dyaw -= 2.0 * M_PI;
            yaw += dyaw * w;
        }
        for (size_t i = 0; i < e2eParticles_.size(); i++) {
            double w = e2eParticles_[i].getW();
            x += e2eParticles_[i].getX() * w;
            y += e2eParticles_[i].getY() * w;
            double dyaw = tmpYaw - e2eParticles_[i].getYaw();
            while (dyaw < -M_PI)
                dyaw += 2.0 * M_PI;
            while (dyaw > M_PI)
                dyaw -= 2.0 * M_PI;
            yaw += dyaw * w;
        }
        yaw = tmpYaw - yaw;
        fusedPose_.setPose(x, y, yaw);
        setMCLPose(fusedPose_);

        // リサンプリング
        double resampleThreshold = getResampleThreshold();
        if (effectiveSampleSize < (double)totalParticleNum * resampleThreshold) {
            std::vector<double> wBuffer(totalParticleNum);
            wBuffer[0] = particles[0].getW();
            for (size_t i = 1; i < particles.size(); i++)
                wBuffer[i] = particles[i].getW() + wBuffer[i - 1];
            for (size_t i = 0; i < e2eParticles_.size(); i++) {
                int idx = i + particles.size();
                wBuffer[idx] = e2eParticles_[i].getW() + wBuffer[idx - 1];
            }

            // 予測分布から生成されたパーティクル群と同じ数のパーティクルをリサンプリングする
            std::vector<Particle> resampledParticles;
            for (size_t i = 0; i < particles.size(); i++) {
                double darts = (double)rand() / ((double)RAND_MAX + 1.0);
                bool resampled = false;
                for (size_t j = 0; j < particles.size(); j++) {
                    if (darts < wBuffer[j]) {
                        Particle particle(particles[j].getPose(), particles[j].getW());
                        resampledParticles.push_back(particle);
                        resampled = true;
                        break;
                    }
                }
                if (!resampled) {
                    for (size_t j = 0; j < e2eParticles_.size(); j++) {
                        if (darts < wBuffer[j + particles.size()]) {
                            Particle particle(e2eParticles_[j].getPose(), e2eParticles_[j].getW());
                            resampledParticles.push_back(particle);
                            break;
                        }
                    }
                }
            }
            // 予測分布から生成されたパーティクル群をリサンプリングされたパーティクル群で上書きする
            particles = resampledParticles;
        }
        setParticles(particles);

        // 最大尤度のパーティクルのインデックスを計算
        int maxLikelihoodParticleIdx;
        double maxLikelihood, wo = 1.0 / (double)totalParticleNum;
        for (size_t i = 0; i < particles.size(); i++) {
            if (i == 0) {
                maxLikelihoodParticleIdx = 0;
                maxLikelihood = particles[0].getW();
            } else if (maxLikelihood < particles[i].getW()) {
                maxLikelihoodParticleIdx = i;
                maxLikelihood = particles[i].getW();
            }
            particles[i].setW(wo);
        }
        setMaxLikelihoodParticleIdx(maxLikelihoodParticleIdx);
    }

    // 自己位置推定に関する軌跡を保存
    void writeTrajectories(Pose gtPose) {
        static FILE *fp;
        static unsigned int cnt = 0;
        if (fp == NULL) {
            fp = fopen("/tmp/e2e_fusion_trajectories.txt", "w");
            fprintf(fp, "# gt.x gt.y gt.yaw fused.x fused.y fused.yaw e2e.x e2e.y e2e.yaw\n");
        }
        fprintf(fp, "%d %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
            cnt, gtPose.getX(), gtPose.getY(), gtPose.getYaw(),
            fusedPose_.getX(), fusedPose_.getY(), fusedPose_.getYaw(),
            e2ePose_.getX(), e2ePose_.getY(), e2ePose_.getYaw());
        cnt++;
    }
}; // class E2EFusion

} // namespace als

#endif // __E2E_FUSION_H__