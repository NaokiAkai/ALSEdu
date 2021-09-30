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

#ifndef __MCL_H__
#define __MCL_H__

#include <vector>
#include <yaml-cpp/yaml.h>
// 尤度場モデルを計算するために必要な距離場の計算にも利用
#include <opencv2/opencv.hpp>
#include <Pose.h>
#include <Particle.h>
#include <Scan.h>

namespace als {

class MCL {
private:
    // マップに関するパラメータ
    std::string mapDir_;
    double mapResolution_;
    int mapWidth_, mapHeight_;
    std::vector<double> mapOrigin_;
    cv::Mat mapImg_;

    // 尤度場モデルを計算するための距離場
    cv::Mat distField_;

    // パーティクル数
    int particleNum_;
    // 最大尤度を持つパーティクルのインデックス
    int maxLikelihoodParticleIdx_;
    // パーティクル群
    std::vector<Particle> particles_;
    // MCLにより推定された位置
    Pose mclPose_;

    // 観測モデルで計算された各パーティクルの尤度
    std::vector<double> measurementLikelihoods_;

    // ビームモデルを用いた棄却法を持ちるかどうか
    bool useScanRejection_;

    // 各スキャン点が未知障害物である確率
    // 動的障害物の棄却使用時，まやはクラス条件付き観測モデル使用時のみ利用可能
    std::vector<double> unknownScanProbs_;

    // 動作モデルによりパーティクル群を更新させる際に使われるパラメータ
    double odomNoise1_, odomNoise2_, odomNoise3_, odomNoise4_;

    // 観測モデルのタイプ
    // 0: ビームモデル，1: 尤度場モデル，2: クラス条件付き観測モデル
    int measurementModel_;
    // 尤度計算の際に間引くスキャン数
    int scanStep_;

    // 観測モデルの線型結合のパラメータ
    double zHit_, zShort_, zMax_, zRand_;
    // 観測モデルのハイパーパラメータ
    double beamSigma_, beamLambda_, lfmSigma_, unknownLambda_;

    // パーティクル群の尤度の総和
    double totalLikelihood_;
    // パーティクル群の尤度の平均値
    double averageLikelihood_;
    // 有効サンプル数
    double effectiveSampleSize_;
    // リサンプリングの閾値
    double resampleThreshold_;

    inline void xy2uv(double x, double y, int *u, int *v) {
        *u = (int)((x - mapOrigin_[0]) / mapResolution_);
        *v = mapHeight_ - 1 - (int)((y - mapOrigin_[1]) / mapResolution_);
    }

    inline void uv2xy(int u, int v, double *x, double *y) {
        *x = (double)u * mapResolution_ + mapOrigin_[0];
        *y = -(double)(v - mapHeight_ + 1) * mapResolution_ + mapOrigin_[1];
    }

    bool readMap(void) {
        std::string yamlFile = mapDir_ + "mcl_map.yaml";
        YAML::Node lconf = YAML::LoadFile(yamlFile);
        mapResolution_ = lconf["resolution"].as<float>();
        mapOrigin_ = lconf["origin"].as<std::vector<double> >();

        std::string imgFile = mapDir_ + "mcl_map.pgm";
        mapImg_ = cv::imread(imgFile, 0);
        mapWidth_ = mapImg_.cols;
        mapHeight_ = mapImg_.rows;

        // 距離場の構築（OpenCVの関数を利用）
        cv::Mat mapImg = mapImg_.clone();
        for (int v = 0; v < mapHeight_; v++) {
            for (int u = 0; u < mapWidth_; u++) {
                uchar val = mapImg.at<uchar>(v, u);
                if (val == 0)
                    mapImg.at<uchar>(v, u) = 0;
                else
                    mapImg.at<uchar>(v, u) = 1;
            }
        }
        cv::Mat distField(mapHeight_, mapWidth_, CV_32FC1);
        cv::distanceTransform(mapImg, distField, cv::DIST_L2, 5);
        for (int v = 0; v < mapHeight_; v++) {
            for (int u = 0; u < mapWidth_; u++) {
                float d = distField.at<float>(v, u) * (float)mapResolution_;
                distField.at<float>(v, u) = d;
            }
        }
        distField_ = distField.clone();
        return true;
    }

    void rejectScan(Scan scan) {
        unknownScanProbs_.resize(scan.getScanNum(), 0.0);
        double zHit, zShort, zRand;
        if (measurementModel_ == 0)
            zHit = zHit_, zShort = zShort_, zRand = zRand_;
        else
            zHit = 0.7, zShort = 0.2, zRand = 0.05;

        double var = beamSigma_ * beamSigma_;
        double normConst = 1.0 / sqrt(2.0 * M_PI * var);
        double pMax = 1.0 / mapResolution_;
        double pRand = 1.0 / (scan.getRangeMax() / mapResolution_);
        for (size_t i = 0; i < scan.getScanNum(); i += scanStep_) {
            double range = scan.getRange(i);
            if (range < scan.getRangeMin() || scan.getRangeMax() < range) {
                unknownScanProbs_[i] = 1.0;
                continue;
            }
            double wBeam = 0.0, wShort = 0.0;
            for (size_t j = 0; j < particles_.size(); j++) {
                Pose pose = particles_[j].getPose();
                double a = scan.getAngleMin() + (double)i * scan.getAngleIncrement() + pose.getYaw();
                double x = pose.getX();
                double y = pose.getY();
                double dx = mapResolution_ * cos(a);
                double dy = mapResolution_ * sin(a);
                double expRange = -1.0;
                for (double r = 0.0; r <= scan.getRangeMax(); r += mapResolution_) {
                    int u, v;
                    xy2uv(x, y, &u, &v);
                    if (0 <= u && u < mapWidth_ && 0 <= v && v < mapHeight_) {
                        if (mapImg_.at<uchar>(v, u) == 0) {
                            expRange = r;
                            break;
                        }
                    } else {
                        break;
                    }
                    x += dx;
                    y += dy;
                }
                if (expRange < 0.0)
                    expRange = scan.getRangeMax();

                double dr = range - expRange;
                double pHit = normConst * exp(-(dr * dr) / (2.0 * var)) * mapResolution_;
                double p = zHit * pHit + zRand * pRand;
                if (range <= expRange) {
                    double shortConst = 1.0 / (1.0 - exp(-beamLambda_ * expRange));
                    double pShort = shortConst * beamLambda_ * exp(-beamLambda_ * range) * mapResolution_;
                    p += zShort * pShort;
                    wShort += log(pShort);
                } else {
                    wShort += log(10e-6);
                }
                if (p > 1.0)
                    p = 1.0;
                wBeam += log(p);
            }
            double p = exp(wShort) / exp(wBeam);
            if (p > 1.0)
                p = 1.0;
            else if (std::isnan(p) && std::isinf(p))
                p = 0.0;
            unknownScanProbs_[i] = p;
        }
    }

    // ビームモデルの計算
    double calculateBeamModel(Pose pose, Scan scan) {
        double var = beamSigma_ * beamSigma_;
        double normConst = 1.0 / sqrt(2.0 * M_PI * var);
        double pMax = 1.0 / mapResolution_;
        double pRand = 1.0 / (scan.getRangeMax() / mapResolution_);
        double w = 0.0;
        for (size_t i = 0; i < scan.getScanNum(); i += scanStep_) {
            if (useScanRejection_ && unknownScanProbs_[i] > 0.9)
                continue;
            double range = scan.getRange(i);
            if (range < scan.getRangeMin() || scan.getRangeMax() < range) {
                w += log(zMax_ * pMax + zRand_ * pRand);
                continue;
            }
            double a = scan.getAngleMin() + (double)i * scan.getAngleIncrement() + pose.getYaw();
            double x = pose.getX();
            double y = pose.getY();
            double dx = mapResolution_ * cos(a);
            double dy = mapResolution_ * sin(a);
            double expRange = -1.0;
            for (double r = 0.0; r <= range; r += mapResolution_) {
                int u, v;
                xy2uv(x, y, &u, &v);
                if (0 <= u && u < mapWidth_ && 0 <= v && v < mapHeight_) {
                    if (mapImg_.at<uchar>(v, u) == 0) {
                        expRange = r;
                        break;
                    }
                } else {
                    break;
                }
                x += dx;
                y += dy;
            }
            if (expRange < 0.0)
                expRange = scan.getRangeMax();
            double dr = range - expRange;
            double pHit = normConst * exp(-(dr * dr) / (2.0 * var)) * mapResolution_;
            double p = zHit_ * pHit + zRand_ * pRand;
            if (range <= expRange) {
                double shortConst = 1.0 / (1.0 - exp(-beamLambda_ * expRange));
                double pShort = shortConst * beamLambda_ * exp(-beamLambda_ * range) * mapResolution_;
                p += zShort_ * pShort;
            }
            if (p > 1.0)
                p = 1.0;
            w += log(p);
        }
        return exp(w);
    }

    // 尤度場モデルの計算
    double calculateLikelihoodFieldModel(Pose pose, Scan scan) {
        double var = lfmSigma_ * lfmSigma_;
        double normConst = 1.0 / sqrt(2.0 * M_PI * var);
        double pMax = 1.0 / mapResolution_;
        double pRand = 1.0 / (scan.getRangeMax() / mapResolution_);
        double w = 0.0;
        for (size_t i = 0; i < scan.getScanNum(); i += scanStep_) {
            if (useScanRejection_ && unknownScanProbs_[i] > 0.9)
                continue;
            double r = scan.getRange(i);
            if (r < scan.getRangeMin() || scan.getRangeMax() < r) {
                w += log(zMax_ * pMax + zRand_ * pRand);
                continue;
            }
            double a = scan.getAngleMin()
                + (double)i * scan.getAngleIncrement() + pose.getYaw();
            double x = r * cos(a) + pose.getX();
            double y = r * sin(a) + pose.getY();
            int u, v;
            xy2uv(x, y, &u, &v);
            if (0 <= u && u < mapWidth_ && 0 <= v && v < mapHeight_) {
                double d = (double)distField_.at<float>(v, u);
                double pHit = normConst
                    * exp(-(d * d) / (2.0 * var)) * mapResolution_;
                double p = zHit_ * pHit + zRand_ * pRand;
                if (p > 1.0)
                    p = 1.0;
                w += log(p);
            } else {
                w += log(zRand_ * pRand);
            }
        }
        return exp(w);
    }

    // クラス条件付き観測モデルの計算
    double calculateClassConditionalMeasurementModel(Pose pose, Scan scan) {
        double var = lfmSigma_ * lfmSigma_;
        double normConst = 1.0 / sqrt(2.0 * M_PI * var);
        double rangeMax = scan.getRangeMax();
        double unknownConst = 1.0 / (1.0 - exp(-unknownLambda_ * rangeMax));
        double pMax = 1.0 / mapResolution_;
        double pRand = 1.0 / (scan.getRangeMax() / mapResolution_);
        double pKnownPrior = 0.5;
        double pUnknownPrior = 1.0 - pKnownPrior;
        double w = 0.0;
        for (size_t i = 0; i < scan.getScanNum(); i += scanStep_) {
            // スキャン点が地図に存在するかしないかの確率
            // この和がパーティクルの尤度となる
            double pKnown, pUnknown;
            double r = scan.getRange(i);
            if (r < scan.getRangeMin() || scan.getRangeMax() < r) {
                pKnown = (zMax_ * pMax + zRand_ * pRand) * pKnownPrior;
                pUnknown = (unknownConst * unknownLambda_
                           * exp(-unknownLambda_ * scan.getRangeMax())
                           * mapResolution_) * pUnknownPrior;
            } else {
                double a = scan.getAngleMin() + (double)i
                           * scan.getAngleIncrement() + pose.getYaw();
                double x = r * cos(a) + pose.getX();
                double y = r * sin(a) + pose.getY();
                int u, v;
                xy2uv(x, y, &u, &v);
                if (0 <= u && u < mapWidth_ && 0 <= v && v < mapHeight_) {
                    double d = (double)distField_.at<float>(v, u);
                    double pHit = normConst * exp(-(d * d) / (2.0 * var))
                                  * mapResolution_;
                    pKnown = (zHit_ * pHit + zRand_ * pRand) * pKnownPrior;
                } else {
                    pKnown = (zRand_ * pRand) * pKnownPrior;
                }
                pUnknown = (unknownConst * unknownLambda_
                           * exp(-unknownLambda_ * r) * mapResolution_)
                           * pUnknownPrior;
            }
            double p = pKnown + pUnknown;
            if (p > 1.0)
                p = 1.0;
            w += log(p);
        }
        return exp(w);
    }

    // 各スキャン点が未知障害物である確率を求める（クラス条件付き観測モデルの利用時）
    void calculateUnknownScanProbs(Pose pose, Scan scan) {
        unknownScanProbs_.resize(scan.getScanNum(), 0.0);
        double var = lfmSigma_ * lfmSigma_;
        double normConst = 1.0 / sqrt(2.0 * M_PI * var);
        double rangeMax = scan.getRangeMax();
        double unknownConst = 1.0 / (1.0 - exp(-unknownLambda_ * rangeMax));
        double pMax = 1.0 / mapResolution_;
        double pRand = 1.0 / (scan.getRangeMax() / mapResolution_);
        double pKnownPrior = 0.5;
        double pUnknownPrior = 1.0 - pKnownPrior;
        for (size_t i = 0; i < scan.getScanNum(); i++) {
            double pKnown, pUnknown;
            double r = scan.getRange(i);
            if (r < scan.getRangeMin() || scan.getRangeMax() < r) {
                pKnown = (zMax_ * pMax + zRand_ * pRand) * pKnownPrior;
                pUnknown = (unknownConst * unknownLambda_
                           * exp(-unknownLambda_ * scan.getRangeMax())
                           * mapResolution_) * pUnknownPrior;
            } else {
                double a = scan.getAngleMin() + (double)i
                           * scan.getAngleIncrement() + pose.getYaw();
                double x = r * cos(a) + pose.getX();
                double y = r * sin(a) + pose.getY();
                int u, v;
                xy2uv(x, y, &u, &v);
                if (0 <= u && u < mapWidth_ && 0 <= v && v < mapHeight_) {
                    double d = (double)distField_.at<float>(v, u);
                    double pHit = normConst * exp(-(d * d) / (2.0 * var))
                                  * mapResolution_;
                    pKnown = (zHit_ * pHit + zRand_ * pRand) * pKnownPrior;
                } else {
                    pKnown = (zRand_ * pRand) * pKnownPrior;
                }
                pUnknown = (unknownConst * unknownLambda_
                           * exp(-unknownLambda_ * r) * mapResolution_)
                           * pUnknownPrior;
            }
            unknownScanProbs_[i] = pUnknown / (pKnown + pUnknown);
        }
    }

public:
    MCL(std::string mapDir, int particleNum):
        mapDir_(mapDir),
        mclPose_(0.0, 0.0, 0.0),
        particleNum_(particleNum),
        odomNoise1_(4.0),
        odomNoise2_(1.0),
        odomNoise3_(1.0),
        odomNoise4_(4.0),
        useScanRejection_(false),
        measurementModel_(0),
        scanStep_(5),
        zHit_(0.9),
        zShort_(0.2),
        zMax_(0.05),
        zRand_(0.05),
        beamSigma_(0.1),
        beamLambda_(0.2),
        lfmSigma_(0.1),
        unknownLambda_(0.1),
        resampleThreshold_(0.5)
    {
        if (!readMap()) {
            std::cerr << "ERROR: MCL map could not be loaded in MCL()." << std::endl;
            exit(1);
        }

        particles_.resize(particleNum_);
        measurementLikelihoods_.resize(particleNum_);
    }

    ~MCL() {};

    inline void setMCLPose(Pose pose) { mclPose_ = pose; }
    inline void setUseScanRejection(bool useScanRejection) { useScanRejection_ = useScanRejection; }
    inline void useBeamModel(void) { measurementModel_ = 0, zHit_ = 0.7, zShort_ = 0.2, zMax_ = 0.05, zRand_ = 0.05; }
    inline void useLikelihoodFieldModel(void) { measurementModel_ = 1, zHit_ = 0.9, zMax_ = 0.05, zRand_ = 0.05; }
    inline void useClassConditionalMeasurementModel(void) { measurementModel_ = 2, zHit_ = 0.9, zMax_ = 0.05, zRand_ = 0.05, useScanRejection_ = false; }
    inline void setParticlePose(int i, Pose pose) { particles_[i].setPose(pose); }
    inline void setParticleW(int i, double w) { particles_[i].setW(w); }
    inline void setTotalLikelihood(double val) { totalLikelihood_ = val; }
    inline void addTotalLikelihood(double val) { totalLikelihood_ += val; }
    inline void setAverageLikelihood(double val) { averageLikelihood_ = val; }
    inline void setEffectiveSampleSize(double val) { effectiveSampleSize_ = val; }
    inline void setMaxLikelihoodParticleIdx(int idx) { maxLikelihoodParticleIdx_ = idx; }
    inline void setParticles(std::vector<Particle> particles) { particles_ = particles; }

    inline double randNormal(double n) { return (n * sqrt(-2.0 * log((double)rand() / RAND_MAX)) * cos(2.0 * M_PI * rand() / RAND_MAX)); }
    inline int getMapWidth(void) { return mapWidth_; }
    inline int getMapHeight(void) { return mapHeight_; }
    inline double getMapResolution(void) { return mapResolution_; }
    inline std::vector<double> getMapOrigin(void) { return mapOrigin_; }
    inline Pose getMCLPose(void) { return mclPose_; }
    inline Pose getParticlePose(int i) { return particles_[i].getPose(); }
    inline int getParticleNum(void) { return particleNum_; }
    inline std::vector<Particle> getParticles(void) { return particles_; }
    inline double getMeasurementLikelihood(int i) { return measurementLikelihoods_[i]; }
    inline double getParticleW(int i) { return particles_[i].getW(); }
    inline double getTotalLikelihood(void) { return totalLikelihood_; }
    inline double getAverageLikelihood(void) { return averageLikelihood_; }
    inline int getMaxLikelihoodParticleIdx(void) { return maxLikelihoodParticleIdx_; }
    inline double getResampleThreshold(void) { return resampleThreshold_; }
    inline double getEffectiveSampleSize(void) { return effectiveSampleSize_; }
    inline void getOdomNoises(double *on1, double *on2, double *on3, double *on4) { *on1 = odomNoise1_, *on2 = odomNoise2_, *on3 = odomNoise3_, *on4 = odomNoise4_; }

    // パーティクル群の分布を現在の推定位置を中心にリセット
    void resetParticlesDistribution(Pose noise) {
        double wo = 1.0 / (double)particles_.size();
        for (size_t i = 0; i < particles_.size(); i++) {
            double x = mclPose_.getX() + randNormal(noise.getX());
            double y = mclPose_.getY() + randNormal(noise.getY());
            double yaw = mclPose_.getYaw() + randNormal(noise.getYaw());
            particles_[i].setPose(x, y, yaw);
            particles_[i].setW(wo);
        }
    }

    // 動作モデルに基づいてパーティクル群を更新
    void updateParticles(double deltaDist, double deltaYaw) {
        double dd2 = deltaDist * deltaDist;
        double dy2 = deltaYaw * deltaYaw;
        for (size_t i = 0; i < particles_.size(); i++) {
            double dd = deltaDist + randNormal(
                odomNoise1_ * dd2 + odomNoise2_ * dy2);
            double dy = deltaYaw + randNormal(
                odomNoise3_ * dd2 + odomNoise4_ * dy2);
            double yaw = particles_[i].getYaw();
            double x = particles_[i].getX() + dd * cos(yaw);
            double y = particles_[i].getY() + dd * sin(yaw);
            yaw += dy;
            particles_[i].setPose(x, y, yaw);
        }
    }

    // 観測モデルを用いて尤度を計算
    void calculateMeasurementModel(Scan scan) {
        // ビームモデルを用いたスキャンの棄却を行う
        // 但しクラス条件付き観測モデル使用時は行わない
        if (useScanRejection_ && measurementModel_ != 2)
            rejectScan(scan);

        // 最大尤度となるパーティクルのインデックスも取得
        totalLikelihood_ = 0.0;
        double maxLikelihood = 0.0;
        for (size_t i = 0; i < particles_.size(); i++) {
            double likelihood = 0.0;
            if (measurementModel_ == 0)
                likelihood = calculateBeamModel(
                    particles_[i].getPose(), scan);
            else if (measurementModel_ == 1)
                likelihood = calculateLikelihoodFieldModel(
                    particles_[i].getPose(), scan);
            else
                likelihood = calculateClassConditionalMeasurementModel(
                    particles_[i].getPose(), scan);
            if (i == 0) {
                maxLikelihood = likelihood;
                maxLikelihoodParticleIdx_ = 0;
            } else if (maxLikelihood < likelihood) {
                maxLikelihood = likelihood;
                maxLikelihoodParticleIdx_ = i;
            }
            measurementLikelihoods_[i] = likelihood;
            totalLikelihood_ += likelihood;
        }
        averageLikelihood_ = totalLikelihood_ / (double)particles_.size();

        // クラス条件付き観測モデル利用時は各スキャン点の未知障害物の確率も計算
        if (measurementModel_ == 2)
            calculateUnknownScanProbs(
                particles_[maxLikelihoodParticleIdx_].getPose(), scan);

        // 尤度の正規化（重みの計算）と有効サンプル数の計算
        double sum = 0.0;
        for (size_t i = 0; i < particles_.size(); i++) {
            double w = measurementLikelihoods_[i] / totalLikelihood_;
            particles_[i].setW(w);
            sum += w * w;
        }
        effectiveSampleSize_ = 1.0 / sum;
    }

    // パーティクル群の重み付き平均を推定位置とする
    void estimatePose(void) {
        double tmpYaw = mclPose_.getYaw();
        double x = 0.0, y = 0.0, yaw = 0.0;
        for (size_t i = 0; i < particles_.size(); i++) {
            double w = particles_[i].getW();
            x += particles_[i].getX() * w;
            y += particles_[i].getY() * w;
            double dyaw = tmpYaw - particles_[i].getYaw();
            while (dyaw < -M_PI)
                dyaw += 2.0 * M_PI;
            while (dyaw > M_PI)
                dyaw -= 2.0 * M_PI;
            yaw += dyaw * w;
        }
        yaw = tmpYaw - yaw;
        mclPose_.setPose(x, y, yaw);
    }

    // パーティクル群のリサンプリング
    void resampleParticles(void) {
        // 有効サンプル数が閾値以上の場合はリサンプリングしない
        double threshold = (double)particles_.size() * resampleThreshold_;
        if (effectiveSampleSize_ > threshold)
            return;

        // パーティクルの重みの和を格納する
        std::vector<double> wBuffer((int)particles_.size());
        wBuffer[0] = particles_[0].getW();
        for (size_t i = 1; i < particles_.size(); i++)
            wBuffer[i] = particles_[i].getW() + wBuffer[i - 1];

        std::vector<Particle> tmpParticles = particles_;
        double wo = 1.0 / (double)particles_.size();
        for (size_t i = 0; i < particles_.size(); i++) {
            double darts = (double)rand() / ((double)RAND_MAX + 1.0);
            for (size_t j = 0; j < particles_.size(); j++) {
                if (darts < wBuffer[j]) {
                    particles_[i].setPose(tmpParticles[j].getPose());
                    particles_[i].setW(wo);
                    break;
                }
            }
        }
    }

    // 与えられた姿勢poseを基に残差を計算
    // 返されるベクトルのサイズはスキャン点の数と同じ
    // 残差は基本正の値であり，有効でない点に対応する残差を表現するために-1.0を代入している
    std::vector<double> getResidualErrors(Pose pose, Scan scan) {
        std::vector<double> residualErrors;
        for (size_t i = 0; i < scan.getScanNum(); i++) {
            double range = scan.getRange(i);
            if (range < scan.getRangeMin() || scan.getRangeMax() < range) {
                residualErrors.push_back(-1.0);
                continue;
            }
            double angle = scan.getAngleMin() + (double)i * scan.getAngleIncrement() + pose.getYaw();
            double x = pose.getX() + range * cos(angle);
            double y = pose.getY() + range * sin(angle);
            int u, v;
            xy2uv(x, y, &u, &v);
            if (0 <= u && u < mapWidth_ && 0 <= v && v < mapHeight_)
                residualErrors.push_back((double)distField_.at<float>(v, u));
            else
                residualErrors.push_back(-1.0);
        }
        return residualErrors;
    }

    // 与えられたパーティクル群particlesに対して観測モデルを用いた尤度計算を行う
    std::vector<double> calculateMeasurementModelForGivenParticles(std::vector<Particle> particles, Scan scan) {
        std::vector<double> likelihoods;
        for (size_t i = 0; i < particles.size(); i++) {
            double likelihood = 0.0;
            if (measurementModel_ == 0)
                likelihood = calculateBeamModel(particles[i].getPose(), scan);
            else if (measurementModel_ == 1)
                likelihood = calculateLikelihoodFieldModel(particles[i].getPose(), scan);
            else
                likelihood = calculateClassConditionalMeasurementModel(particles[i].getPose(), scan);
            likelihoods.push_back(likelihood);
        }
        return likelihoods;
    }

    // MCLによる推定位置を端末に表示
    void printMCLPose(void) {
        std::cout << "MCL Pose: x = " << mclPose_.getX() << " [m], y = " << mclPose_.getY() << " [m], yaw = " << mclPose_.getYaw() * 180.0 / M_PI << " [deg]" << std::endl;
    }

    // MCLにより推定されたパラメータを端末に表示
    void printEvaluationParameters(void) {
        std::cout << "Likelihood: total = " << totalLikelihood_ << ", average = " << averageLikelihood_ << ", ess = " << effectiveSampleSize_ << std::endl;
    }

    // MCLによる推定位置をファイルに記録
    void writeMCLTrajectory(void) {
        static FILE *fp;
        if (fp == NULL)
            fp = fopen("/tmp/mcl_trajectory.txt", "w");
        fprintf(fp, "%lf %lf %lf\n", mclPose_.getX(), mclPose_.getY(), mclPose_.getYaw());
    }

    // MCLの結果をgnuplotで表示
    void plotMCLWorld(double plotRange, Scan scan) {
        static FILE *gp;
        FILE *fp;
        if (gp == NULL) {
            gp = popen("gnuplot -persist", "w");
            fprintf(gp, "set colors classic\n");
            fprintf(gp, "unset key\n");
            fprintf(gp, "set grid\n");
            fprintf(gp, "set size ratio 1 1\n");
            fprintf(gp, "set xlabel \"%s\"\n", "X [m]");
            fprintf(gp, "set ylabel \"%s\"\n", "Y [m]");
            fprintf(gp, "set tics font \"Arial, 14\"\n");
            fprintf(gp, "set xlabel font \"Arial, 14\"\n");
            fprintf(gp, "set ylabel font \"Arial, 14\"\n");

            fp = fopen("/tmp/mcl_map_points.txt", "w");
            for (int u = 0; u < mapWidth_; u++) {
                for (int v = 0; v < mapHeight_; v++) {
                    if (mapImg_.at<uchar>(v, u) == 0) {
                        double x, y;
                        uv2xy(u, v, &x, &y);
                        fprintf(fp, "%lf %lf\n", x, y);
                    }
                }
            }
            fclose(fp);
        }

        fprintf(gp, "set xrange [ %lf : %lf ]\n", mclPose_.getX() - plotRange, mclPose_.getX() + plotRange);
        fprintf(gp, "set yrange [ %lf : %lf ]\n", mclPose_.getY() - plotRange, mclPose_.getY() + plotRange);

        double axesLength = 2.0;
        double x1 = mclPose_.getX() + axesLength * cos(mclPose_.getYaw());
        double y1 = mclPose_.getY() + axesLength * sin(mclPose_.getYaw());
        double x2 = mclPose_.getX() + axesLength * cos(mclPose_.getYaw() + M_PI / 2.0);
        double y2 = mclPose_.getY() + axesLength * sin(mclPose_.getYaw() + M_PI / 2.0);
        fp = fopen("/tmp/mcl_pose1.txt", "w");
        fprintf(fp, "%lf %lf\n", mclPose_.getX(), mclPose_.getY());
        fprintf(fp, "%lf %lf\n", x1, y1);
        fclose(fp);
        fp = fopen("/tmp/mcl_pose2.txt", "w");
        fprintf(fp, "%lf %lf\n", mclPose_.getX(), mclPose_.getY());
        fprintf(fp, "%lf %lf\n", x2, y2);
        fclose(fp);

        // パーティクル群の書き出し
        fp = fopen("/tmp/mcl_particles.txt", "w");
        for (size_t i = 0; i < particles_.size(); i++) {
            double axesLength = 1.0;
            double x = particles_[i].getX() + axesLength * cos(particles_[i].getYaw());
            double y = particles_[i].getY() + axesLength * sin(particles_[i].getYaw());
            fprintf(fp, "%lf %lf\n", particles_[i].getX(), particles_[i].getY());
            fprintf(fp, "%lf %lf\n\n", x, y);
        }
        fclose(fp);

        fp = fopen("/tmp/mcl_scan_lines.txt", "w");
        for (size_t i = 0; i < scan.getScanNum(); i++) {
            double angle = scan.getAngleMin() + (double)i * scan.getAngleIncrement() + mclPose_.getYaw();
            double x = mclPose_.getX() + scan.getRange(i) * cos(angle);
            double y = mclPose_.getY() + scan.getRange(i) * sin(angle);
            fprintf(fp, "%lf %lf\n", mclPose_.getX(), mclPose_.getY());
            fprintf(fp, "%lf %lf\n\n", x, y);
        }
        fclose(fp);

        // クラス条件付き観測モデル利用時は未知障害物と判断されたスキャン点を書き出し
        if (useScanRejection_ && measurementModel_ != 2) {
            fp = fopen("/tmp/unknown_scan_points.txt", "w");
            for (size_t i = 0; i < scan.getScanNum(); i += scanStep_) {
                double range = scan.getRange(i);
                if (scan.getRangeMin() < range && range < scan.getRangeMax() && unknownScanProbs_[i] >= 0.9) {
                    double angle = scan.getAngleMin() + (double)i * scan.getAngleIncrement() + particles_[maxLikelihoodParticleIdx_].getYaw();
                    double x = particles_[maxLikelihoodParticleIdx_].getX() + range * cos(angle);
                    double y = particles_[maxLikelihoodParticleIdx_].getY() + range * sin(angle);
                    fprintf(fp, "%lf %lf\n", x, y);
                }
            }
            fclose(fp);
        }

        // クラス条件付き観測モデル利用時は未知障害物と判断されたスキャン点を書き出し
        if (measurementModel_ == 2) {
            fp = fopen("/tmp/unknown_scan_points.txt", "w");
            for (size_t i = 0; i < scan.getScanNum(); i++) {
                double range = scan.getRange(i);
                if (scan.getRangeMin() < range && range < scan.getRangeMax() && unknownScanProbs_[i] >= 0.9) {
                    double angle = scan.getAngleMin() + (double)i * scan.getAngleIncrement() + particles_[maxLikelihoodParticleIdx_].getYaw();
                    double x = particles_[maxLikelihoodParticleIdx_].getX() + range * cos(angle);
                    double y = particles_[maxLikelihoodParticleIdx_].getY() + range * sin(angle);
                    fprintf(fp, "%lf %lf\n", x, y);
                }
            }
            fclose(fp);
        }

        // 表示
        if (!useScanRejection_ && measurementModel_ != 2) {
            fprintf(gp, "plot \"%s\" with points pointtype 5 pointsize 0.1 lt -1, \
                \"%s\" w l lt 3 lw 0.5, \"%s\" w l lt 4 lw 1, \
                \"%s\" w l lt 2 lw 3, \"%s\" u 1:2 w l lt 1 lw 3\n", 
                "/tmp/mcl_map_points.txt",
                "/tmp/mcl_scan_lines.txt",
                "/tmp/mcl_particles.txt",
                "/tmp/mcl_pose2.txt",
                "/tmp/mcl_pose1.txt");
        } else {
            fprintf(gp, "plot \"%s\" with points pointtype 5 pointsize 0.1 lt -1, \
                \"%s\" w l lt 3 lw 0.5, \"%s\" w l lt 4 lw 1, \
                \"%s\" w l lt 2 lw 3, \"%s\" u 1:2 w l lt 1 lw 3, \"%s\" lt 22 lw 2\n", 
                "/tmp/mcl_map_points.txt",
                "/tmp/mcl_scan_lines.txt",
                "/tmp/mcl_particles.txt",
                "/tmp/mcl_pose2.txt",
                "/tmp/mcl_pose1.txt",
                "/tmp/unknown_scan_points.txt");
        }
        fflush(gp);
    }

    // MCLとE2Eを融合した結果の表示
    void plotE2EFusionWorld(double plotRange, Scan scan, std::vector<Particle> e2eParticles) {
        static FILE *gp;
        FILE *fp;
        if (gp == NULL) {
            gp = popen("gnuplot -persist", "w");
            fprintf(gp, "set colors classic\n");
            fprintf(gp, "unset key\n");
            fprintf(gp, "set grid\n");
            fprintf(gp, "set size ratio 1 1\n");

            fp = fopen("/tmp/mcl_map_points.txt", "w");
            for (int u = 0; u < mapWidth_; u++) {
                for (int v = 0; v < mapHeight_; v++) {
                    if (mapImg_.at<uchar>(v, u) == 0) {
                        double x, y;
                        uv2xy(u, v, &x, &y);
                        fprintf(fp, "%lf %lf\n", x, y);
                    }
                }
            }
            fclose(fp);
        }

        fprintf(gp, "set xrange [ %lf : %lf ]\n", mclPose_.getX() - plotRange, mclPose_.getX() + plotRange);
        fprintf(gp, "set yrange [ %lf : %lf ]\n", mclPose_.getY() - plotRange, mclPose_.getY() + plotRange);

        double axesLength = 2.0;
        double x1 = mclPose_.getX() + axesLength * cos(mclPose_.getYaw());
        double y1 = mclPose_.getY() + axesLength * sin(mclPose_.getYaw());
        double x2 = mclPose_.getX() + axesLength * cos(mclPose_.getYaw() + M_PI / 2.0);
        double y2 = mclPose_.getY() + axesLength * sin(mclPose_.getYaw() + M_PI / 2.0);
        fp = fopen("/tmp/mcl_pose1.txt", "w");
        fprintf(fp, "%lf %lf\n", mclPose_.getX(), mclPose_.getY());
        fprintf(fp, "%lf %lf\n", x1, y1);
        fclose(fp);
        fp = fopen("/tmp/mcl_pose2.txt", "w");
        fprintf(fp, "%lf %lf\n", mclPose_.getX(), mclPose_.getY());
        fprintf(fp, "%lf %lf\n", x2, y2);
        fclose(fp);

        fp = fopen("/tmp/mcl_particles.txt", "w");
        for (size_t i = 0; i < particles_.size(); i++) {
            double axesLength = 1.0;
            double x = particles_[i].getX() + axesLength * cos(particles_[i].getYaw());
            double y = particles_[i].getY() + axesLength * sin(particles_[i].getYaw());
            fprintf(fp, "%lf %lf\n", particles_[i].getX(), particles_[i].getY());
            fprintf(fp, "%lf %lf\n\n", x, y);
        }
        fclose(fp);

        // E2Eで生成されたパーティクル群の書き出し
        fp = fopen("/tmp/e2e_particles.txt", "w");
        for (size_t i = 0; i < e2eParticles.size(); i++) {
            double axesLength = 1.0;
            double x = e2eParticles[i].getX() + axesLength * cos(e2eParticles[i].getYaw());
            double y = e2eParticles[i].getY() + axesLength * sin(e2eParticles[i].getYaw());
            fprintf(fp, "%lf %lf\n", e2eParticles[i].getX(), e2eParticles[i].getY());
            fprintf(fp, "%lf %lf\n\n", x, y);
        }
        fclose(fp);

        fp = fopen("/tmp/mcl_scan_lines.txt", "w");
        for (size_t i = 0; i < scan.getScanNum(); i++) {
            double angle = scan.getAngleMin() + (double)i * scan.getAngleIncrement() + mclPose_.getYaw();
            double x = mclPose_.getX() + scan.getRange(i) * cos(angle);
            double y = mclPose_.getY() + scan.getRange(i) * sin(angle);
            fprintf(fp, "%lf %lf\n", mclPose_.getX(), mclPose_.getY());
            fprintf(fp, "%lf %lf\n\n", x, y);
        }
        fclose(fp);

        if (measurementModel_ == 2) {
            calculateUnknownScanProbs(particles_[maxLikelihoodParticleIdx_].getPose(), scan);
            fp = fopen("/tmp/unknown_scan_points.txt", "w");
            for (size_t i = 0; i < scan.getScanNum(); i++) {
                double range = scan.getRange(i);
                if (scan.getRangeMin() < range && range < scan.getRangeMax() && unknownScanProbs_[i] >= 0.9) {
                    double angle = scan.getAngleMin() + (double)i * scan.getAngleIncrement() + particles_[maxLikelihoodParticleIdx_].getYaw();
                    double x = particles_[maxLikelihoodParticleIdx_].getX() + range * cos(angle);
                    double y = particles_[maxLikelihoodParticleIdx_].getY() + range * sin(angle);
                    fprintf(fp, "%lf %lf\n", x, y);
                }
            }
            fclose(fp);
        }

        // 表示
        if (measurementModel_ != 2) {
            fprintf(gp, "plot \"%s\" with points pointtype 5 pointsize 0.1 lt -1, \
                \"%s\" w l lt 3 lw 0.5, \"%s\" w l lt 4 lw 1, \"%s\" w l lt 8 lw 1, \
                \"%s\" w l lt 2 lw 3, \"%s\" u 1:2 w l lt 1 lw 3\n", 
                "/tmp/mcl_map_points.txt",
                "/tmp/mcl_scan_lines.txt",
                "/tmp/mcl_particles.txt",
                "/tmp/e2e_particles.txt",
                "/tmp/mcl_pose2.txt",
                "/tmp/mcl_pose1.txt");
        } else {
            fprintf(gp, "plot \"%s\" with points pointtype 5 pointsize 0.1 lt -1, \
                \"%s\" w l lt 3 lw 0.5, \"%s\" w l lt 4 lw 1, \"%s\" w l lt 8 lw 1, \
                \"%s\" w l lt 2 lw 3, \"%s\" u 1:2 w l lt 1 lw 3, \"%s\" lt 22 lw 2\n", 
                "/tmp/mcl_map_points.txt",
                "/tmp/mcl_scan_lines.txt",
                "/tmp/mcl_particles.txt",
                "/tmp/e2e_particles.txt",
                "/tmp/mcl_pose2.txt",
                "/tmp/mcl_pose1.txt",
                "/tmp/unknown_scan_points.txt");
        }
        fflush(gp);
    }

    void plotFailureDetectionWorld(double plotRange, Scan scan, std::vector<int> residualErrorClasses) {
        static FILE *gp;
        FILE *fp;
        if (gp == NULL) {
            gp = popen("gnuplot -persist", "w");
            fprintf(gp, "set colors classic\n");
            fprintf(gp, "unset key\n");
            fprintf(gp, "set grid\n");
            fprintf(gp, "set size ratio 1 1\n");

            fp = fopen("/tmp/mcl_map_points.txt", "w");
            for (int u = 0; u < mapWidth_; u++) {
                for (int v = 0; v < mapHeight_; v++) {
                    if (mapImg_.at<uchar>(v, u) == 0) {
                        double x, y;
                        uv2xy(u, v, &x, &y);
                        fprintf(fp, "%lf %lf\n", x, y);
                    }
                }
            }
            fclose(fp);
        }

        fprintf(gp, "set xrange [ %lf : %lf ]\n", mclPose_.getX() - plotRange, mclPose_.getX() + plotRange);
        fprintf(gp, "set yrange [ %lf : %lf ]\n", mclPose_.getY() - plotRange, mclPose_.getY() + plotRange);

        double axesLength = 2.0;
        double x1 = mclPose_.getX() + axesLength * cos(mclPose_.getYaw());
        double y1 = mclPose_.getY() + axesLength * sin(mclPose_.getYaw());
        double x2 = mclPose_.getX() + axesLength * cos(mclPose_.getYaw() + M_PI / 2.0);
        double y2 = mclPose_.getY() + axesLength * sin(mclPose_.getYaw() + M_PI / 2.0);
        fp = fopen("/tmp/mcl_pose1.txt", "w");
        fprintf(fp, "%lf %lf\n", mclPose_.getX(), mclPose_.getY());
        fprintf(fp, "%lf %lf\n", x1, y1);
        fclose(fp);
        fp = fopen("/tmp/mcl_pose2.txt", "w");
        fprintf(fp, "%lf %lf\n", mclPose_.getX(), mclPose_.getY());
        fprintf(fp, "%lf %lf\n", x2, y2);
        fclose(fp);

        fp = fopen("/tmp/mcl_particles.txt", "w");
        for (size_t i = 0; i < particles_.size(); i++) {
            double axesLength = 1.0;
            double x = particles_[i].getX() + axesLength * cos(particles_[i].getYaw());
            double y = particles_[i].getY() + axesLength * sin(particles_[i].getYaw());
            fprintf(fp, "%lf %lf\n", particles_[i].getX(), particles_[i].getY());
            fprintf(fp, "%lf %lf\n\n", x, y);
        }
        fclose(fp);

        fp = fopen("/tmp/mcl_scan_lines.txt", "w");
        for (size_t i = 0; i < scan.getScanNum(); i++) {
            double angle = scan.getAngleMin() + (double)i * scan.getAngleIncrement() + mclPose_.getYaw();
            double x = mclPose_.getX() + scan.getRange(i) * cos(angle);
            double y = mclPose_.getY() + scan.getRange(i) * sin(angle);
            fprintf(fp, "%lf %lf\n", mclPose_.getX(), mclPose_.getY());
            fprintf(fp, "%lf %lf\n\n", x, y);
        }
        fclose(fp);

        // マッチしていると判断されたスキャン点の書き出し
        fp = fopen("/tmp/fd_aligned_points.txt", "w");
        for (size_t i = 0; i < scan.getScanNum(); i++) {
            double range = scan.getRange(i);
            if (scan.getRangeMin() < range && range < scan.getRangeMax() && residualErrorClasses[i] == 0) {
                double angle = scan.getAngleMin() + (double)i * scan.getAngleIncrement() + mclPose_.getYaw();
                double x = mclPose_.getX() + range * cos(angle);
                double y = mclPose_.getY() + range * sin(angle);
                fprintf(fp, "%lf %lf\n", x, y);
            }
        }
        fclose(fp);

        // ミスマッチしていると判断されたスキャン点の書き出し
        fp = fopen("/tmp/fd_misaligned_points.txt", "w");
        for (size_t i = 0; i < scan.getScanNum(); i++) {
            double range = scan.getRange(i);
            if (scan.getRangeMin() < range && range < scan.getRangeMax() && residualErrorClasses[i] == 1) {
                double angle = scan.getAngleMin() + (double)i * scan.getAngleIncrement() + mclPose_.getYaw();
                double x = mclPose_.getX() + range * cos(angle);
                double y = mclPose_.getY() + range * sin(angle);
                fprintf(fp, "%lf %lf\n", x, y);
            }
        }
        fclose(fp);

        // 未知障害物と判断されたスキャン点の書き出し
        fp = fopen("/tmp/fd_unknown_points.txt", "w");
        for (size_t i = 0; i < scan.getScanNum(); i++) {
            double range = scan.getRange(i);
            if (scan.getRangeMin() < range && range < scan.getRangeMax() && residualErrorClasses[i] == 2) {
                double angle = scan.getAngleMin() + (double)i * scan.getAngleIncrement() + mclPose_.getYaw();
                double x = mclPose_.getX() + range * cos(angle);
                double y = mclPose_.getY() + range * sin(angle);
                fprintf(fp, "%lf %lf\n", x, y);
            }
        }
        fclose(fp);

        // 表示
        fprintf(gp, "plot \"%s\" with points pointtype 5 pointsize 0.1 lt -1, \
            \"%s\" w l lt 3 lw 0.5, \"%s\" w l lt 4 lw 1, \"%s\" w l lt 2 lw 3, \"%s\" u 1:2 w l lt 1 lw 3, \
            \"%s\" lt 1 pt 13, \"%s\" lt 3 pt 13, \"%s\" lt 2 pt 13\n", 
            "/tmp/mcl_map_points.txt",
            "/tmp/mcl_scan_lines.txt",
            "/tmp/mcl_particles.txt",
            "/tmp/mcl_pose2.txt",
            "/tmp/mcl_pose1.txt",
            "/tmp/fd_aligned_points.txt",
            "/tmp/fd_misaligned_points.txt",
            "/tmp/fd_unknown_points.txt");
        fflush(gp);
    }

}; // class MCL

} // namespace als

#endif // __MCL_H__
