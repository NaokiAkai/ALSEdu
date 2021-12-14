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
#include <MCLRE.h>

int main(int argc, char **argv) {
    if (argv[1] == NULL) {
        std::cerr << "ERROR: argv[1] must be map directory, \
            e.g., ../maps/nic1f/" << std::endl;
        exit(1);
    }

    double plotRange = 10.0;
    bool plotOdomPose = false;
    bool plotGTScan = true;
    double simulationHz = 10.0;

    als::RobotSim robotSim(argv[1], simulationHz);
    cv::namedWindow("Keyboard Interface Window", cv::WINDOW_NORMAL);

    int particleNum = 100;
    als::MCLRE mcl(argv[1], particleNum);
    mcl.setMCLPose(robotSim.getGTRobotPose());
    mcl.resetParticlesDistribution(als::Pose(0.5, 0.5, 3.0 * M_PI / 180.0));

    mcl.setUseScanRejection(false);

    // mcl.useBeamModel();
    mcl.useLikelihoodFieldModel();
    // mcl.useClassConditionalMeasurementModel();

    mcl.useAdaBoostClassifier();
    // mcl.useMLPClassifier();
    // mcl.useMAEClassifier();

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

        // 信頼度付きMCL の実行
        mcl.updateParticlesAndReliability(deltaDist, deltaYaw);
        mcl.calculateMeasurementModel(scan);
        mcl.calculateDecisionModel(scan);
        mcl.estimatePose();
        mcl.resampleParticlesAndReliability();
        mcl.printMCLPose();
        mcl.printEvaluationParameters();
        mcl.printReliability();
        mcl.writeMCLREResults(gtRobotPose);
        mcl.plotMCLWorld(plotRange, scan);

        usleep(usleepTime);
    }
    return 0;
}
