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

#include <iostream>
#include <RobotSim.h>
#include <MCL.h>

void generateREsDataset(als::RobotSim robotSim, als::MCL mcl, int dataNum, std::string saveDir) {
    std::string gtPosesFileName = saveDir + "gt_poses.txt";
    std::string scansFileName = saveDir + "scans.txt";
    std::string successPosesFileName = saveDir + "success_poses.txt";
    std::string failurePosesFileName = saveDir + "failure_poses.txt";
    std::string successResidualErrorsFileName = saveDir + "success_residual_errors.txt";
    std::string failureResidualErrorsFileName = saveDir + "failure_residual_errors.txt";

    FILE *fpGTPoses = fopen(gtPosesFileName.c_str(), "w");
    FILE *fpScans = fopen(scansFileName.c_str(), "w");
    FILE *fpSuccessPoses = fopen(successPosesFileName.c_str(), "w");
    FILE *fpFailurePoses = fopen(failurePosesFileName.c_str(), "w");
    FILE *fpSuccessResidualErrors = fopen(successResidualErrorsFileName.c_str(), "w");
    FILE *fpFailureResidualErrors = fopen(failureResidualErrorsFileName.c_str(), "w");

    for (size_t i = 0; i < dataNum; i++) {
        robotSim.setRobotPoseRandomly();
        robotSim.updateSimulation();
        robotSim.makeSuccessPoseRandomly();
        robotSim.makeFailurePoseRandomly();

        als::Pose gtPose = robotSim.getGTRobotPose();
        als::Pose successPose = robotSim.getSuccessPose();
        als::Pose failurePose = robotSim.getFailurePose();

        fprintf(fpGTPoses, "%lf %lf %lf\n", gtPose.getX(), gtPose.getY(), gtPose.getYaw());
        fprintf(fpSuccessPoses, "%lf %lf %lf\n", successPose.getX(), successPose.getY(), successPose.getYaw());
        fprintf(fpFailurePoses, "%lf %lf %lf\n", failurePose.getX(), failurePose.getY(), failurePose.getYaw());

        als::Scan scan = robotSim.getScan();
        std::vector<double> successResidualErrors = mcl.getResidualErrors(successPose, scan);
        std::vector<double> failureResidualErrors = mcl.getResidualErrors(failurePose, scan);
        for (size_t j = 0; j < scan.getScanNum(); j++) {
            fprintf(fpScans, "%lf", scan.getRange(j));
            fprintf(fpSuccessResidualErrors, "%lf", successResidualErrors[j]);
            fprintf(fpFailureResidualErrors, "%lf", failureResidualErrors[j]);
            if (j != scan.getScanNum() - 1) {
                fprintf(fpScans, " ");
                fprintf(fpSuccessResidualErrors, " ");
                fprintf(fpFailureResidualErrors, " ");
            }
        }
        fprintf(fpScans, "\n");
        fprintf(fpSuccessResidualErrors, "\n");
        fprintf(fpFailureResidualErrors, "\n");
    }

    fclose(fpGTPoses);
    fclose(fpScans);
    fclose(fpSuccessPoses);
    fclose(fpFailurePoses);
    fclose(fpSuccessResidualErrors);
    fclose(fpFailureResidualErrors);
}

int main(int argc, char **argv) {
    if (argv[1] == NULL) {
        std::cerr << "ERROR: argv[1] must be map directory, \
            e.g., ../maps/nic1f/" << std::endl;
        exit(1);
    }
    if (argv[2] == NULL) {
        std::cerr << "ERROR: argv[2] must be dataset directory, \
            e.g., ../datasets/residual_errors/nic1f/" << std::endl;
        exit(1);
    }

    int trainDataNum = 1000;
    int testDataNum = 1000;

    als::RobotSim robotSim(argv[1], 10.0);
    als::MCL mcl(argv[1], 0);

    generateREsDataset(robotSim, mcl, trainDataNum, (std::string)argv[2] + "train/");
    generateREsDataset(robotSim, mcl, testDataNum, (std::string)argv[2] + "test/");

    return 0;
}
