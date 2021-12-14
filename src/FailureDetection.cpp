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
#include <MRFFailureDetector.h>

int main(int argc, char **argv) {
    if (argv[1] == NULL) {
        std::cerr << "ERROR: argv[1] must be map directory, \
            e.g., ../maps/nic1f/" << std::endl;
        exit(1);
    }

    double plotRange = 10.0;
    bool plotOdomPose = true;
    bool plotGTScan = false;
    double simulationHz = 10.0;

    als::RobotSim robotSim(argv[1], simulationHz);
    cv::namedWindow("Keyboard Interface Window", cv::WINDOW_NORMAL);

    // MCL を自己位置推定モジュールとして利用
    int particleNum = 100;
    als::MCL mcl(argv[1], particleNum);
    mcl.setMCLPose(robotSim.getGTRobotPose());
    mcl.resetParticlesDistribution(als::Pose(0.5, 0.5, 3.0 * M_PI / 180.0));
    mcl.useLikelihoodFieldModel();

    // 失敗検出器の宣言
    als::MRFFD failureDetector;

    double usleepTime = (1.0 / simulationHz) * 10e5;
    while (!robotSim.getKillFlag()) {
        int key = cv::waitKey(200);
        robotSim.keyboardOperation(key);
        robotSim.updateSimulation();
        robotSim.plotSimulationWorld(plotRange, plotOdomPose, plotGTScan);

        double linearVel, angularVel;
        robotSim.getVelocities(&linearVel, &angularVel);
        als::Scan scan = robotSim.getScan();

        double deltaDist = linearVel * (1.0 / simulationHz);
        double deltaYaw = angularVel * (1.0 / simulationHz);

        // MCL の実行
        mcl.updateParticles(deltaDist, deltaYaw);
        mcl.calculateMeasurementModel(scan);
        mcl.estimatePose();
        mcl.resampleParticles();

        // MCL による推定結果に対する推定正誤判断の実行
        // MCL による推定位置を取得
        als::Pose mclPose = mcl.getMCLPose();
        // MCL による推定位置からの残差を計算
        std::vector<double> residualErrors = mcl.getResidualErrors(mclPose, scan);
        // 位置推定に失敗している確率を予測
        failureDetector.predictFailureProbability(residualErrors);
        // 失敗確率を端末に表示
        failureDetector.printFailureProbability();
        // 各スキャン点のクラスを取得
        std::vector<int> residualErrorClasses = failureDetector.getResidualErrorClasses();
        // gnuplot で結果を表示
        mcl.plotFailureDetectionWorld(plotRange, scan, residualErrorClasses);

        usleep(usleepTime);
    }
    return 0;
}
