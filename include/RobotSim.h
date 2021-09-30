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

#ifndef __ROBOT_SIM_H__
#define __ROBOT_SIM_H__

#include <iostream>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>    // マップのパラメータが記述されたファイル（yaml）を読み込むために使用
#include <opencv2/opencv.hpp> // マップ（画像）を扱うためにOpenCVを利用
#include <Pose.h>
#include <Scan.h>

namespace als {

class RobotSim {
private:
    // マップに関する値
    std::string mapDir_;
    double mapResolution_;
    int mapWidth_, mapHeight_;
    std::vector<double> mapOrigin_;
    cv::Mat mapImg_;

    double simulationHz_; // シミュレーションの更新周期

    Pose gtRobotPose_; // シミュレートされたロボットの位置（真値）
    Pose odomPose_;    // オドメトリで推定された位置

    double gtLinearVel_, gtAngularVel_; // 並進・角速度の真値
    double linearVel_, angularVel_;     // オドメトリ計算様に真にノイズを加えた速度

    Scan gtScan_; // シミュレートされたスキャンデータ（真値）
    Scan scan_;   // 真値のスキャンデータにノイズを加えたデータ

    // シミュレーション用のノイズ
    double odomNoise1_, odomNoise2_, odomNoise3_, odomNoise4_, scanRangeNoise_;

    // 自己位置推定に失敗しているかどうか判断する閾値
    double failureThresholdPosition_, failureThresholdAngle_;

    // 自己位置推定に成功・失敗している状態の姿勢
    Pose successPose_; // gtRobotPose_から上の閾値を超えないノイズを加えた姿勢
    Pose failurePose_; // gtRobotPose_から上の閾値を超えるノイズを加えた姿勢

    bool killFlag_;      // ウインドウ上で「q」か「Ecs」でフラグが立ってプログラムが終了
    bool printKeyValue_; // OpenCVを介して受け取ったkeyの値を端末に表示するかを切り替えるフラグ

    // xy座標（マップ）をuv座標（画像）に変換
    // xy座標とuv座標で，y軸とv軸の生の方向が異なることに注意
    inline void xy2uv(double x, double y, int *u, int *v) {
        *u = (int)((x - mapOrigin_[0]) / mapResolution_);
        *v = mapHeight_ - 1 - (int)((y - mapOrigin_[1]) / mapResolution_);
    }

    // uv座標をxy座標に変換
    inline void uv2xy(int u, int v, double *x, double *y) {
        *x = (double)u * mapResolution_ + mapOrigin_[0];
        *y = -(double)(v - mapHeight_ + 1) * mapResolution_ + mapOrigin_[1];
    }

    // マップの読み込み（TODO: 例外処理は？）
    bool readMap(void) {
        // yamlファイルからマップのパラメータを読み込む
        std::string yamlFile = mapDir_ + "ogm.yaml";
        YAML::Node lconf = YAML::LoadFile(yamlFile);
        mapResolution_ = lconf["resolution"].as<float>();
        mapOrigin_ = lconf["origin"].as<std::vector<double> >();

        // マップの画像を読み込む
        std::string imgFile = mapDir_ + "ogm.pgm";
        mapImg_ = cv::imread(imgFile, 0);
        mapWidth_ = mapImg_.cols;
        mapHeight_ = mapImg_.rows;
        return true;
    }

    // ロボット位置とオドメトリ位置の更新
    void updatePoses(void) {
        // ロボット位置の更新
        double deltaTime = 1.0 / simulationHz_;
        double yaw = gtRobotPose_.getYaw();
        double x = gtRobotPose_.getX() + gtLinearVel_ * deltaTime * cos(yaw); 
        double y = gtRobotPose_.getY() + gtLinearVel_ * deltaTime * sin(yaw);
        yaw += gtAngularVel_ * deltaTime;
        gtRobotPose_.setPose(x, y, yaw);

        // オドメトリ更新用の速度を乱数を用いて決める（速度が小さい場合は強制的に0とする）
        double lv2 = gtLinearVel_ * gtLinearVel_;
        double av2 = gtAngularVel_ * gtAngularVel_;
        if (lv2 < 0.001 && av2 < 0.001) {
            linearVel_ = 0.0;
            angularVel_ = 0.0;
        } else {
            linearVel_ = gtLinearVel_ + randNormal(odomNoise1_ * lv2 + odomNoise2_ * av2);
            angularVel_ = gtAngularVel_ + randNormal(odomNoise3_ * lv2 + odomNoise4_ * av2);
        }

        // オドメトリ位置の更新
        yaw = odomPose_.getYaw();
        x = odomPose_.getX() + linearVel_ * deltaTime * cos(yaw); 
        y = odomPose_.getY() + linearVel_ * deltaTime * sin(yaw);
        yaw += angularVel_ * deltaTime;
        odomPose_.setPose(x, y, yaw);
    }

    // スキャンデータのシミュレーション
    void scanWorld(void) {
        for (size_t i = 0; i < gtScan_.getScanNum(); i++) {
            double angle = gtScan_.getAngleMin() + (double)i * gtScan_.getAngleIncrement() + gtRobotPose_.getYaw();
            double x = gtRobotPose_.getX();
            double y = gtRobotPose_.getY();
            double dx = mapResolution_ * cos(angle);
            double dy = mapResolution_ * sin(angle);
            double range = 0.0;
            // 地図上でレイキャストしてスキャンデータをシミュレートする
            for (double r = 0.0; r < gtScan_.getRangeMax(); r += mapResolution_) {
                int u, v;
                xy2uv(x, y, &u, &v);
                if (0 <= u && u < mapWidth_ && 0 <= v && v < mapHeight_) {
                    uchar val = mapImg_.at<uchar>(v, u);
                    if (val == 0) { // 地図上の障害物にヒット
                        range = r;
                        break;
                    }
                } else { // スキャンデータが地図の外に存在
                    break;
                }
                x += dx;
                y += dy;
            }
            gtScan_.setRange(i, range); // スキャンの真値
            if (range == 0.0) {
                scan_.setRange(i, 0.0);
            } else {
                // 真値にノイズを加えた値を実際のスキャンデータとする
                double r = range + randNormal(scanRangeNoise_);
                if (r <= scan_.getRangeMin() || scan_.getRangeMax() <= r)
                    r = 0.0;
                scan_.setRange(i, r);
            }
        }
    }

public:
    RobotSim(std::string mapDir, double simulationHz):
        mapDir_(mapDir),
        simulationHz_(simulationHz),
        odomNoise1_(0.4),
        odomNoise2_(0.2),
        odomNoise3_(0.2),
        odomNoise4_(0.4),
        scanRangeNoise_(0.1),
        failureThresholdPosition_(0.3),
        failureThresholdAngle_(3.0 * M_PI / 180.0),
        killFlag_(false),
        printKeyValue_(false)
    {
        if (!readMap()) {
            std::cerr << "ERROR: A simulation map could not be loaded in readMap()." << std::endl;
            exit(1);
        }

        srand((unsigned int)time(NULL));

        gtRobotPose_.setPose(0.0, 0.0, 0.0);
        odomPose_.setPose(0.0, 0.0, 0.0);
        gtLinearVel_ = gtAngularVel_ = linearVel_ = angularVel_ = 0.0;
    }

    inline void setRobotPose(Pose robotPose) { gtRobotPose_ = odomPose_ = robotPose; }
    inline void setPrintKeyValue(bool printKeyValue) { printKeyValue_ = printKeyValue; }

    inline double randNormal(double n) { return (n * sqrt(-2.0 * log((double)rand() / RAND_MAX)) * cos(2.0 * M_PI * rand() / RAND_MAX)); }
    inline int generateRandomValue(int min, int max) { return min + (int)(rand() * ((double)(max - min) + 1.0) / (1.0 + RAND_MAX)); }
    inline Pose getGTRobotPose(void) { return gtRobotPose_; }
    inline Pose getOdomPose(void) { return odomPose_; }
    inline Pose getSuccessPose(void) { return successPose_; }
    inline Pose getFailurePose(void) { return failurePose_; }
    inline void getVelocities(double *linearVel, double *angularVel) { *linearVel = linearVel_, *angularVel = angularVel_; }
    inline Scan getGTScan(void) { return gtScan_; }
    inline Scan getScan(void) { return scan_; }
    inline bool getKillFlag(void) { return killFlag_; }

    // キー入力に合わせてシミュレション用のパラメータを変更
    // [↑]，[↓]: 並進速度の増減
    // [←]，[→]: 角速度の増減
    // [ ]: 並進・角速度を0に
    // [r]: ロボット位置を原点に戻す
    // [j]: ロボットの位置を現在の位置からランダムに動かす（誘拐ロボット問題用）
    void keyboardOperation(int key) {
        if (key == 'q' || key == 27) {
            killFlag_ = true;
            std::cout << "The robot simulator will be going to shutdown." << std::endl;
            int retVal = system("killall -9 gnuplot");
        } else if (key == 'r') {
            gtRobotPose_.setPose(0.0, 0.0, 0.0);
            odomPose_.setPose(0.0, 0.0, 0.0);
            gtLinearVel_ = gtAngularVel_ = 0.0;
        } else if (key == 82) {
            gtLinearVel_ += 0.2;
        } else if (key == 84) {
            gtLinearVel_ -= 0.2;
        } else if (key == 81) {
            gtAngularVel_ += 0.1;
        } else if (key == 83) {
            gtAngularVel_ -= 0.1;
        } else if (key == ' ') {
            gtLinearVel_ = gtAngularVel_ = 0.0;
        } else if (key == 'j') {
            double x = gtRobotPose_.getX() + randNormal(0.5);
            double y = gtRobotPose_.getY() + randNormal(0.5);
            double yaw = gtRobotPose_.getYaw() + randNormal(5.0 * M_PI / 180.0);
            gtRobotPose_.setPose(x, y, yaw);
        } else if (key == 's') {
            makeSuccessPoseRandomly();
            gtRobotPose_ = successPose_;
            odomPose_ = successPose_;
        } else if (key == 'f') {
            makeFailurePoseRandomly();
            gtRobotPose_ = failurePose_;
            odomPose_ = failurePose_;
        }

        if (printKeyValue_ && key >= 0) {
            std::cout << "key = " << key << std::endl;
        }
    }

    // シミュレーションの更新
    void updateSimulation(void) {
        updatePoses();
        scanWorld();
    }

    // ロボット位置（真値）を端末で表示
    void printRobotPose(void) {
        std::cout << "RobotSim Pose: x = " << gtRobotPose_.getX() << " [m], y = " << gtRobotPose_.getY() << " [m], yaw = " << gtRobotPose_.getYaw() * 180.0 / M_PI << " [deg]" << std::endl;
    }

    // ロボット位置（真値）をファイルに記録
    void writeRobotTrajectory(void) {
        static FILE *fp;
        if (fp == NULL)
            fp = fopen("/tmp/robot_trajectory.txt", "w");
        fprintf(fp, "%lf %lf %lf\n", gtRobotPose_.getX(), gtRobotPose_.getY(), gtRobotPose_.getYaw());
    }

    // オドメトリ位置をファイルに記録
    void writeOdometryTrajectory(void) {
        static FILE *fp;
        if (fp == NULL)
            fp = fopen("/tmp/odometry_trajectory.txt", "w");
        fprintf(fp, "%lf %lf %lf\n", odomPose_.getX(), odomPose_.getY(), odomPose_.getYaw());
    }

    // シミュレーション環境をgnuplotで表示
    void plotSimulationWorld(double plotRange, bool plotOdomPose, bool plotGTScan) {
        static FILE *gp;
        FILE *fp;
        // 1回目はgnuplotの設定とマップのプロットを行う
        if (gp == NULL) {
            // gnuplotの設定
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

            // マップの書き出し
            fp = fopen("/tmp/map_points.txt", "w");
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

        // ロボット位置を図の中心とする
        fprintf(gp, "set xrange [ %lf : %lf ]\n", gtRobotPose_.getX() - plotRange, gtRobotPose_.getX() + plotRange);
        fprintf(gp, "set yrange [ %lf : %lf ]\n", gtRobotPose_.getY() - plotRange, gtRobotPose_.getY() + plotRange);

        // ロボット位置の書き出し
        double axesLength = 2.0;
        double x1 = gtRobotPose_.getX() + axesLength * cos(gtRobotPose_.getYaw());
        double y1 = gtRobotPose_.getY() + axesLength * sin(gtRobotPose_.getYaw());
        double x2 = gtRobotPose_.getX() + axesLength * cos(gtRobotPose_.getYaw() + M_PI / 2.0);
        double y2 = gtRobotPose_.getY() + axesLength * sin(gtRobotPose_.getYaw() + M_PI / 2.0);
        fp = fopen("/tmp/sim_robot_pose1.txt", "w");
        fprintf(fp, "%lf %lf\n", gtRobotPose_.getX(), gtRobotPose_.getY());
        fprintf(fp, "%lf %lf\n", x1, y1);
        fclose(fp);
        fp = fopen("/tmp/sim_robot_pose2.txt", "w");
        fprintf(fp, "%lf %lf\n", gtRobotPose_.getX(), gtRobotPose_.getY());
        fprintf(fp, "%lf %lf\n", x2, y2);
        fclose(fp);

        // オドメトリ位置の書き出し
        if (plotOdomPose) {
            double x1 = odomPose_.getX() + axesLength * cos(odomPose_.getYaw());
            double y1 = odomPose_.getY() + axesLength * sin(odomPose_.getYaw());
            double x2 = odomPose_.getX() + axesLength * cos(odomPose_.getYaw() + M_PI / 2.0);
            double y2 = odomPose_.getY() + axesLength * sin(odomPose_.getYaw() + M_PI / 2.0);
            fp = fopen("/tmp/sim_odom_pose1.txt", "w");
            fprintf(fp, "%lf %lf\n", odomPose_.getX(), odomPose_.getY());
            fprintf(fp, "%lf %lf\n", x1, y1);
            fclose(fp);
            fp = fopen("/tmp/sim_odom_pose2.txt", "w");
            fprintf(fp, "%lf %lf\n", odomPose_.getX(), odomPose_.getY());
            fprintf(fp, "%lf %lf\n", x2, y2);
            fclose(fp);
        }

        // スキャンデータの書き出し
        fp = fopen("/tmp/scan_lines.txt", "w");
        if (plotGTScan) {
            for (size_t i = 0; i < gtScan_.getScanNum(); i++) {
                double angle = gtScan_.getAngleMin() + (double)i * gtScan_.getAngleIncrement() + gtRobotPose_.getYaw();
                double x = gtRobotPose_.getX() + gtScan_.getRange(i) * cos(angle);
                double y = gtRobotPose_.getY() + gtScan_.getRange(i) * sin(angle);
                fprintf(fp, "%lf %lf\n", gtRobotPose_.getX(), gtRobotPose_.getY());
                fprintf(fp, "%lf %lf\n\n", x, y);
            }
        } else {
            for (size_t i = 0; i < scan_.getScanNum(); i++) {
                double angle = scan_.getAngleMin() + (double)i * scan_.getAngleIncrement() + gtRobotPose_.getYaw();
                double x = gtRobotPose_.getX() + scan_.getRange(i) * cos(angle);
                double y = gtRobotPose_.getY() + scan_.getRange(i) * sin(angle);
                fprintf(fp, "%lf %lf\n", gtRobotPose_.getX(), gtRobotPose_.getY());
                fprintf(fp, "%lf %lf\n\n", x, y);
            }
        }
        fclose(fp);

        // プロット
        if (plotOdomPose) {
            fprintf(gp, "plot \"%s\" with points pointtype 5 pointsize 0.1 lt -1, \
                \"%s\" w l lt 3 lw 0.5, \"%s\" w l lt 2 lw 3, \"%s\" u 1:2 w l lt 1 lw 3, \
                \"%s\" w l lt 2 lw 3, \"%s\" u 1:2 w l lt 1 lw 3\n", 
                "/tmp/map_points.txt",
                "/tmp/scan_lines.txt",
                "/tmp/sim_odom_pose2.txt",
                "/tmp/sim_odom_pose1.txt",
                "/tmp/sim_robot_pose2.txt",
                "/tmp/sim_robot_pose1.txt");
        } else {
            fprintf(gp, "plot \"%s\" with points pointtype 5 pointsize 0.1 lt -1, \
                \"%s\" w l lt 3 lw 0.5, \"%s\" w l lt 2 lw 3, \"%s\" u 1:2 w l lt 1 lw 3\n", 
                "/tmp/map_points.txt",
                "/tmp/scan_lines.txt",
                "/tmp/sim_robot_pose2.txt",
                "/tmp/sim_robot_pose1.txt");
        }
        fflush(gp);
    }

    void setRobotPoseRandomly(void) {
        for (;;) {
            // nic1f
            // int u = generateRandomValue(850, 1550);
            // int v = generateRandomValue(750, 1050);
            // garage
            // int u = generateRandomValue(900, 1450);
            // int v = generateRandomValue(650, 1400);
            int u = generateRandomValue(850, 1550);
            int v = generateRandomValue(650, 1400);
            bool isValidPosition = true;
            for (size_t uu = u - 5; uu < u + 5; uu++) {
                for (size_t vv = v - 5; vv < v + 5; vv++) {
                    if (uu < 0 || mapWidth_ <= uu || vv < 0 || mapHeight_ <= vv) {
                        isValidPosition = false;
                        break;
                    }
                    if (mapImg_.at<uchar>(vv, uu) < 250) {
                        isValidPosition = false;
                        break;
                    }
                }
                if (!isValidPosition)
                    break;
            }
            if (isValidPosition) {
                double x, y;
                uv2xy(u, v, &x, &y);
                double yaw = (double)generateRandomValue(-31415, 31415) / 10000.0;
                setRobotPose(Pose(x, y, yaw));
                break;
            }
        }
    }

    void makeSuccessPoseRandomly(void) {
        int positionRange = (int)(failureThresholdPosition_ * 1000.0);
        int angleRange = (int)(failureThresholdAngle_ * 1000.0);
        double x = gtRobotPose_.getX() + (double)generateRandomValue(-positionRange + 1, positionRange - 1) / 1000.0;
        double y = gtRobotPose_.getY() + (double)generateRandomValue(-positionRange + 1, positionRange - 1) / 1000.0;
        double yaw = gtRobotPose_.getYaw() + (double)generateRandomValue(-angleRange + 1, angleRange - 1) / 1000.0;
        successPose_.setPose(x, y, yaw);
    }

    void makeFailurePoseRandomly(void) {
        int positionRange = (int)(failureThresholdPosition_ * 1000.0);
        int angleRange = (int)(failureThresholdAngle_ * 1000.0);
        for (;;) {
            double dx = 2.0 * (double)generateRandomValue(-positionRange, positionRange) / 1000.0;
            double dy = 2.0 * (double)generateRandomValue(-positionRange, positionRange) / 1000.0;
            double dyaw = 2.0 * (double)generateRandomValue(-angleRange, angleRange) / 1000.0;
            if (fabs(dx) >= failureThresholdPosition_ || fabs(dy) >= failureThresholdPosition_ || fabs(dyaw) >= failureThresholdAngle_) {
                double x = gtRobotPose_.getX() + dx;
                double y = gtRobotPose_.getY() + dy;
                double yaw = gtRobotPose_.getYaw() + dyaw;
                failurePose_.setPose(x, y, yaw);
                break;
            }
        }
    }

}; // class RobotSim

} // namespace als

#endif // __ROBOT_SIM_H__
