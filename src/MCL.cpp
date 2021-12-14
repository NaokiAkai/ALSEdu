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

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <RobotSim.h>
#include <MCL.h>

int main(int argc, char **argv) {
    if (argv[1] == NULL) {
        std::cerr << "ERROR: argv[1] must be map directory, \
            e.g., ../maps/nic1f/" << std::endl;
        exit(1);
    }

    double plotRange = 10.0;
    bool plotOdomPose = true;
    bool plotGTScan = true;
    double simulationHz = 10.0;

    als::RobotSim robotSim(argv[1], simulationHz);
    cv::namedWindow("Keyboard Interface Window", cv::WINDOW_NORMAL);

    // パーティクル数
    int particleNum = 100;
    // 初期のパーティクル群の分布
    als::Pose initialNoise(0.5, 0.5, 3.0 * M_PI / 180.0);
    // MCL 用のクラス
    als::MCL mcl(argv[1], particleNum);
    // MCL の初期位置を設定
    mcl.setMCLPose(robotSim.getGTRobotPose());
    // パーティクルを設定された位置の辺りにばらまく
    mcl.resetParticlesDistribution(initialNoise);

    // ビームモデルを用いたスキャンの棄却を行うかの設定
    mcl.setUseScanRejection(false);

    // 使用する観測モデルの設定
    // mcl.useBeamModel();
    mcl.useLikelihoodFieldModel();
    // mcl.useClassConditionalMeasurementModel();

    double usleepTime = (1.0 / simulationHz) * 10e5;
    while (!robotSim.getKillFlag()) {
        int key = cv::waitKey(200);
        robotSim.keyboardOperation(key);
        robotSim.updateSimulation();
        robotSim.writeRobotTrajectory();
        robotSim.writeOdometryTrajectory();
        robotSim.plotSimulationWorld(plotRange, plotOdomPose, plotGTScan);

        // 並進・角速度とスキャンデータをシミュレータから取得
        double linearVel, angularVel;
        robotSim.getVelocities(&linearVel, &angularVel);
        als::Scan scan = robotSim.getScan();

        // 速度を移動量に変換（MCL でsimulationHz を使用しないため）
        double deltaDist = linearVel * (1.0 / simulationHz);
        double deltaYaw = angularVel * (1.0 / simulationHz);

        // MCL の実行
        // 移動量に基づくパーティクル群の更新
        mcl.updateParticles(deltaDist, deltaYaw);
        // 観測モデルによる尤度計算
        mcl.calculateMeasurementModel(scan);
        // ロボット位置の推定
        mcl.estimatePose();
        // パーティクル群のリサンプリング
        mcl.resampleParticles();
        // MCL により推定された値を端末に表示
        mcl.printMCLPose();
        // MCL 中に推定されたパラメータを端末に表示
        mcl.printEvaluationParameters();
        // MCL による推定位置をファイルに記録
        mcl.writeMCLTrajectory();
        // gnuplot でMCL の結果を表示
        mcl.plotMCLWorld(plotRange, scan);

        usleep(usleepTime);
    }
    return 0;
}
