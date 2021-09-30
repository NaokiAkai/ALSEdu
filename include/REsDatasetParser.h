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

#ifndef __RES_DATASET_PARSER_H__
#define __RES_DATASET_PARSER_H__

#include <iostream>
#include <vector>
#include <Pose.h>

namespace als {

class REsDatasetParser {
private:
    std::string datasetDir_;
    std::string trainDatasetDir_, testDatasetDir_;

    std::vector<Pose> readPoses(std::string fileName) {
        FILE *fp = fopen(fileName.c_str(), "r");
        if (fp == NULL) {
            fprintf(stderr, "cannot open %s\n", fileName.c_str());
            exit(1);
        }

        double x, y, yaw;
        std::vector<Pose> poses;
        while (fscanf(fp, "%lf %lf %lf", &x, &y, &yaw) != EOF) {
            Pose pose(x, y, yaw);
            poses.push_back(pose);
        }
        fclose(fp);

        return poses;
    }

    std::vector<std::vector<double> > readResidualErrors(std::string fileName) {
        FILE *fp = fopen(fileName.c_str(), "r");
        if (fp == NULL) {
            fprintf(stderr, "cannot open %s\n", fileName.c_str());
            exit(1);
        }

        char str;
        int scanNum = 0;
        while (fscanf(fp, "%c", &str) != EOF) {
            if (str == ' ')
                scanNum++;
            if (str == '\n') {
                scanNum++;
                break;
            }
        }
        fclose(fp);

        fp = fopen(fileName.c_str(), "r");
        double err;
        std::vector<double> errors;
        std::vector<std::vector<double> > residualErrors;
        while (fscanf(fp, "%lf", &err) != EOF) {
            errors.push_back(err);
            if (errors.size() == scanNum) {
                residualErrors.push_back(errors);
                errors.clear();
            }
        }
        fclose(fp);

        return residualErrors;
    }

public:
    REsDatasetParser(std::string datasetDir):
        datasetDir_(datasetDir)
    {
        trainDatasetDir_ = datasetDir_ + "train/";
        testDatasetDir_ = datasetDir_ + "test/";
    }

    std::vector<Pose> getTrainGTPoses(void) {
        std::string fileName = trainDatasetDir_ + "gt_poses.txt";
        return readPoses(fileName);
    }

    std::vector<Pose> getTrainSuccessPoses(void) {
        std::string fileName = trainDatasetDir_ + "success_poses.txt";
        return readPoses(fileName);
    }

    std::vector<Pose> getTrainFailurePoses(void) {
        std::string fileName = trainDatasetDir_ + "failure_poses.txt";
        return readPoses(fileName);
    }

    std::vector<std::vector<double> > getTrainSuccessResidualErrors(void) {
        std::string fileName = trainDatasetDir_ + "success_residual_errors.txt";
        return readResidualErrors(fileName);
    }

    std::vector<std::vector<double> > getTrainFailureResidualErrors(void) {
        std::string fileName = trainDatasetDir_ + "failure_residual_errors.txt";
        return readResidualErrors(fileName);
    }

    std::vector<Pose> getTestGTPoses(void) {
        std::string fileName = testDatasetDir_ + "gt_poses.txt";
        return readPoses(fileName);
    }

    std::vector<Pose> getTestSuccessPoses(void) {
        std::string fileName = testDatasetDir_ + "success_poses.txt";
        return readPoses(fileName);
    }

    std::vector<Pose> getTestFailurePoses(void) {
        std::string fileName = testDatasetDir_ + "failure_poses.txt";
        return readPoses(fileName);
    }

    std::vector<std::vector<double> > getTestSuccessResidualErrors(void) {
        std::string fileName = testDatasetDir_ + "success_residual_errors.txt";
        return readResidualErrors(fileName);
    }

    std::vector<std::vector<double> > getTestFailureResidualErrors(void) {
        std::string fileName = testDatasetDir_ + "failure_residual_errors.txt";
        return readResidualErrors(fileName);
    }

}; // REsDatasetParser

} // namespace als

#endif // __RES_DATASET_PARSER_H__