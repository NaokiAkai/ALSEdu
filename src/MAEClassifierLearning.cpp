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
#include <REsDatasetParser.h>
#include <MAEClassifier.h>

int main(int argc, char **argv) {
    if (argv[1] == NULL) {
        std::cerr << "ERROR: argv[1] must be dataset directory, \
            e.g., ../datasets/residual_errors/nic1f/" << std::endl;
        exit(1);
    }

    double maxResidualError = 1.0;

    als::REsDatasetParser parser(argv[1]);
    std::vector<std::vector<double> > trainSuccessResidualErrors = parser.getTrainSuccessResidualErrors();
    std::vector<std::vector<double> > trainFailureResidualErrors = parser.getTrainFailureResidualErrors();
    std::vector<std::vector<double> > testSuccessResidualErrors = parser.getTestSuccessResidualErrors();
    std::vector<std::vector<double> > testFailureResidualErrors = parser.getTestFailureResidualErrors();

    als::MAEClassifier classifier;
    classifier.setMaxResidualError(maxResidualError);
    classifier.learnThreshold(trainSuccessResidualErrors, trainFailureResidualErrors);
    classifier.writeClassifierParams(testSuccessResidualErrors, testFailureResidualErrors);
    classifier.writeDecisionLikelihoods();

    return 0;
}
