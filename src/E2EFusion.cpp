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

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <RobotSim.h>
#include <MCL.h>
#include <E2EFusion.h>

int main(int argc, char **argv) {
    if (argv[1] == NULL) {
        std::cerr << "ERROR: argv[1] must be map directory, e.g., ../maps/nic1f/" << std::endl;
        exit(1);
    }

    double plotRange = 10.0;
    bool plotOdomPose = true;
    bool plotGTScan = true;
    double simulationHz = 10.0;

    als::RobotSim robotSim(argv[1], simulationHz);
    cv::namedWindow("Keyboard Interface Window", cv::WINDOW_NORMAL);

    // 深層学習を用いずにランダムサンプリングでEnd-to-End 位置推定を再現
    // 真値にノイズを加えた位置を中心にサンプリングを行う
    // E2E 位置推定で生成されるパーティクルの数
    int e2eParticleNum = 50;
    // 真値に対して加えるノイズ
    als::Pose e2eGTNoise(1.0, 1.0, 1.0 * M_PI / 180.0);
    // パーティクルを生成する一様分布の範囲
    als::Pose e2eUniformRange(10.0, 10.0, 10.0 * M_PI / 180.0);
    // 正規分布から生成する場合に使用される分散
    als::Pose e2eNormVar(1.0, 1.0, 1.0 * M_PI / 180.0);

    int particleNum = 100;
    als::E2EFusion mcl(argv[1], particleNum, e2eParticleNum);
    mcl.setMCLPose(robotSim.getGTRobotPose());
    mcl.resetParticlesDistribution(als::Pose(0.5, 0.5, 3.0 * M_PI / 180.0));

    // 観測モデルとして尤度場モデルを使用
    mcl.useLikelihoodFieldModel();

    double usleepTime = (1.0 / simulationHz) * 10e5;
    while (!robotSim.getKillFlag()) {
        int key = cv::waitKey(200);
        robotSim.keyboardOperation(key);
        robotSim.updateSimulation();
        robotSim.plotSimulationWorld(plotRange, plotOdomPose, plotGTScan);

        als::Pose gtRobotPose = robotSim.getGTRobotPose();
        double linearVel, angularVel;
        robotSim.getVelocities(&linearVel, &angularVel);
        als::Scan scan = robotSim.getScan();

        double deltaDist = linearVel * (1.0 / simulationHz);
        double deltaYaw = angularVel * (1.0 / simulationHz);

        // MCL とE2E を融合する位置推定の実行
        mcl.updateParticles(deltaDist, deltaYaw);
        // 真値にノイズを加えた位置を中心とする一様分布から生成
        // mcl.generateE2EParticlesFromUniformDistribution(gtRobotPose, e2eGTNoise, e2eUniformRange);
        // 真値にノイズを加えた位置を平均とする正規分布から生成
        mcl.generateE2EParticlesFromNormalDistribution(gtRobotPose, e2eGTNoise, e2eNormVar);
        mcl.estimateE2EPose();
        mcl.executeE2EFusion(scan);
        mcl.printMCLPose();
        mcl.printEvaluationParameters();
        mcl.plotE2EFusionWorld(plotRange, scan, mcl.getE2EParticles());
        mcl.writeTrajectories(robotSim.getGTRobotPose());

        usleep(usleepTime);
    }

    return 0;
}
