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

#ifndef __MAE_CLASSIFIER_H__
#define __MAE_CLASSIFIER_H__

#include <yaml-cpp/yaml.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <list>
#include <numeric>
#include <Histogram.h>

namespace als {

class MAEClassifier {
private:
    std::string classifiersDir_;
    double maxResidualError_;

    double failureThreshold_;
    double successMAEMean_, successMAEStd_;
    double failureMAEMean_, failureMAEStd_;

    int truePositiveNum_;
    int falsePositiveNum_;
    int trueNegativeNum_;
    int falseNegativeNum_;

    double maeHistogramBinWidth_;
    Histogram positiveMAEHistogram_, negativeMAEHistogram_;
    Histogram truePositiveMAEHistogram_, trueNegativeMAEHistogram_;
    Histogram falsePositiveMAEHistogram_, falseNegativeMAEHistogram_;

    template <template<class T, class Allocator = std::allocator<T> > class Container> double getMean(Container<double> &x) {
        return std::accumulate(x.begin(), x.end(), 0.0) / x.size();
    }

    template <template<class T, class Allocator = std::allocator<T> > class Container> double getVar(Container<double> &x) {
        double size = x.size();
        double mean = getMean(x);
        return (std::inner_product(x.begin(), x.end(), x.begin(), 0.0) - mean * mean * size) / (size - 1.0);
    }

    template <template<class T, class Allocator = std::allocator<T> > class Container> double getStd(Container<double> &x) {
        return std::sqrt(getVar(x));
    }

    void getStatisticalParamsOfMAE(std::vector<std::vector<double> > residualErrors, double *mean, double *std) {
        std::vector<double> maes;
        for (size_t i = 0; i < residualErrors.size(); i++) {
            double mae = getMAE(residualErrors[i]);
            maes.push_back(mae);
        }
        *mean = getMean(maes);
        *std = getStd(maes);
    }

    std::vector<double> readMAEs(std::string filePath) {
        FILE *fp = fopen(filePath.c_str(), "r");
        if (fp == NULL) {
            fprintf(stderr, "cannot open %s\n", filePath.c_str());
            exit(1);
        }
        double mae;
        std::vector<double> maes;
        while (fscanf(fp, "%lf", &mae) != EOF)
            maes.push_back(mae);
        fclose(fp);
        return maes;
    }

public:
    MAEClassifier(void):
        classifiersDir_("../classifiers/MAE/"),
        maxResidualError_(1.0),
        maeHistogramBinWidth_(0.01) {}

    inline void setMaxResidualError(double maxResidualError) { maxResidualError_ = maxResidualError_; }
    inline void setMAEHistogramBinWidth(double maeHistogramBinWidth) { maeHistogramBinWidth_ = maeHistogramBinWidth; }

    inline double getFailureThreshold(void) { return failureThreshold_; }

    double getMAE(std::vector<double> residualErrors) {
        double sum = 0.0;
        int num = 0;
        for (size_t i = 0; i < residualErrors.size(); i++) {
            if (0.0 <= residualErrors[i] && residualErrors[i] <= maxResidualError_) {
                sum += residualErrors[i];
                num++;
            }
        }
        if (num == 0)
            return 0.0;
        else
            return sum / (double)num;
    }

    void learnThreshold(std::vector<std::vector<double> > trainSuccessResidualErrors, std::vector<std::vector<double> > trainFailureResidualErrors) {
        getStatisticalParamsOfMAE(trainSuccessResidualErrors, &successMAEMean_, &successMAEStd_);
        getStatisticalParamsOfMAE(trainFailureResidualErrors, &failureMAEMean_, &failureMAEStd_);
        failureThreshold_ = (successMAEMean_ + failureMAEMean_) / 2.0;
    }

    void writeClassifierParams(std::vector<std::vector<double> > testSuccessResidualErrors, std::vector<std::vector<double> > testFailureResidualErrors) {
        std::string positiveMAEsFileName = classifiersDir_ + "positive_maes.txt";
        std::string negativeMAEsFileName = classifiersDir_ + "negative_maes.txt";
        std::string truePositiveMAEsFileName = classifiersDir_ + "true_positive_maes.txt";
        std::string trueNegativeMAEsFileName = classifiersDir_ + "true_negative_maes.txt";
        std::string falsePositiveMAEsFileName = classifiersDir_ + "false_positive_maes.txt";
        std::string falseNegativeMAEsFileName = classifiersDir_ + "false_negative_maes.txt";
        FILE *fpPositive = fopen(positiveMAEsFileName.c_str(), "w");
        FILE *fpNegative = fopen(negativeMAEsFileName.c_str(), "w");
        FILE *fpTruePositive = fopen(truePositiveMAEsFileName.c_str(), "w");
        FILE *fpTrueNegative = fopen(trueNegativeMAEsFileName.c_str(), "w");
        FILE *fpFalsePositive = fopen(falsePositiveMAEsFileName.c_str(), "w");
        FILE *fpFalseNegative = fopen(falseNegativeMAEsFileName.c_str(), "w");

        truePositiveNum_ = falseNegativeNum_ = falsePositiveNum_ = trueNegativeNum_ = 0;
        for (size_t i = 0; i < testSuccessResidualErrors.size(); i++) {
            double mae = getMAE(testSuccessResidualErrors[i]);
            fprintf(fpPositive, "%lf\n", mae);
            if (mae <= failureThreshold_) {
                truePositiveNum_++;
                fprintf(fpTruePositive, "%lf\n", mae);
            } else {
                falseNegativeNum_++;
                fprintf(fpFalseNegative, "%lf\n", mae);
            }
        }
        for (size_t i = 0; i < testFailureResidualErrors.size(); i++) {
            double mae = getMAE(testFailureResidualErrors[i]);
            fprintf(fpNegative, "%lf\n", mae);
            if (mae <= failureThreshold_) {
                falsePositiveNum_++;
                fprintf(fpFalsePositive, "%lf\n", mae);
            } else {
                trueNegativeNum_++;
                fprintf(fpTrueNegative, "%lf\n", mae);
            }
        }
        fclose(fpPositive);
        fclose(fpNegative);
        fclose(fpTruePositive);
        fclose(fpTrueNegative);
        fclose(fpFalsePositive);
        fclose(fpFalseNegative);

        printf("truePositiveNum = %d\n", truePositiveNum_);
        printf("falseNegativeNum = %d\n", falseNegativeNum_);
        printf("trueNegativeNum = %d\n", trueNegativeNum_);
        printf("falsePositiveNum = %d\n", falsePositiveNum_);

        std::string yamlFile = classifiersDir_ + "classifier.yaml";
        FILE *fpYaml = fopen(yamlFile.c_str(), "w");
        fprintf(fpYaml, "maxResidualError: %lf\n", maxResidualError_);
        fprintf(fpYaml, "failureThreshold: %lf\n", failureThreshold_);
        fprintf(fpYaml, "successMAEMean: %lf\n", successMAEMean_);
        fprintf(fpYaml, "successMAEStd: %lf\n", successMAEStd_);
        fprintf(fpYaml, "failureMAEMean: %lf\n", failureMAEMean_);
        fprintf(fpYaml, "failureMAEStd: %lf\n", failureMAEStd_);
        fprintf(fpYaml, "truePositiveNum: %d\n", truePositiveNum_);
        fprintf(fpYaml, "falseNegativeNum: %d\n", falseNegativeNum_);
        fprintf(fpYaml, "trueNegativeNum: %d\n", trueNegativeNum_);
        fprintf(fpYaml, "falsePositiveNum: %d\n", falsePositiveNum_);
        fprintf(fpYaml, "positiveMAEsFileName: positive_maes.txt\n");
        fprintf(fpYaml, "negativeMAEsFileName: negative_maes.txt\n");
        fprintf(fpYaml, "truePositiveMAEsFileName: true_positive_maes.txt\n");
        fprintf(fpYaml, "trueNegativeMAEsFileName: true_negative_maes.txt\n");
        fprintf(fpYaml, "falsePositiveMAEsFileName: false_positive_maes.txt\n");
        fprintf(fpYaml, "falseNegativeMAEsFileName: false_negative_maes.txt\n");
        fclose(fpYaml);
        printf("yaml file for the MAE classifier was saved at %s\n", yamlFile.c_str());
    }

    void readClassifierParams(void) {
        std::string yamlFile = classifiersDir_ + "classifier.yaml";
        YAML::Node lconf = YAML::LoadFile(yamlFile);

        maxResidualError_ = lconf["maxResidualError"].as<double>();
        failureThreshold_ = lconf["failureThreshold"].as<double>();

        std::string positiveMAEsFilePath = classifiersDir_ + lconf["positiveMAEsFileName"].as<std::string>();
        std::string negativeMAEsFilePath = classifiersDir_ + lconf["negativeMAEsFileName"].as<std::string>();
        positiveMAEHistogram_ = Histogram(readMAEs(positiveMAEsFilePath), maeHistogramBinWidth_);
        negativeMAEHistogram_ = Histogram(readMAEs(negativeMAEsFilePath), maeHistogramBinWidth_);

        std::string truePositiveMAEsFilePath = classifiersDir_ + lconf["truePositiveMAEsFileName"].as<std::string>();
        std::string trueNegativeMAEsFilePath = classifiersDir_ + lconf["trueNegativeMAEsFileName"].as<std::string>();
        std::string falsePositiveMAEsFilePath = classifiersDir_ + lconf["falsePositiveMAEsFileName"].as<std::string>();
        std::string falseNegativeMAEsFilePath = classifiersDir_ + lconf["falseNegativeMAEsFileName"].as<std::string>();
        truePositiveMAEHistogram_ = Histogram(readMAEs(truePositiveMAEsFilePath), maeHistogramBinWidth_);
        trueNegativeMAEHistogram_ = Histogram(readMAEs(trueNegativeMAEsFilePath), maeHistogramBinWidth_);
        falsePositiveMAEHistogram_ = Histogram(readMAEs(falsePositiveMAEsFilePath), maeHistogramBinWidth_);
        falseNegativeMAEHistogram_ = Histogram(readMAEs(falseNegativeMAEsFilePath), maeHistogramBinWidth_);
    }

    double calculateDecisionModel(double mae, double *reliability) {
        double pSuccess, pFailure;
        if (mae < failureThreshold_) {
            pSuccess = truePositiveMAEHistogram_.getProbability(mae);
            pFailure = falsePositiveMAEHistogram_.getProbability(mae);
        } else {
            pSuccess = falseNegativeMAEHistogram_.getProbability(mae);
            pFailure = trueNegativeMAEHistogram_.getProbability(mae);
        }
        if (pSuccess < 10.0e-6)
            pSuccess = 10.0e-6;
        if (pFailure < 10.0e-6)
            pFailure = 10.0e-6;
        double rel = pSuccess * *reliability;
        double relInv = pFailure * (1.0 - *reliability);
        double p = rel + relInv;
        if (p > 1.0)
            p = 1.0;
        *reliability = rel / (rel + relInv);
        if (*reliability > 0.9999)
            *reliability = 0.9999;
        if (*reliability < 0.0001)
            *reliability = 0.0001;
        return p;
    }

    void writeDecisionLikelihoods(void) {
        readClassifierParams();
        std::string fileName = classifiersDir_ + "mae_decision_likelihoods.txt";
        FILE *fp = fopen(fileName.c_str(), "w");
        for (double rel = 0.0; rel <= 1.0 + 0.05; rel += 0.05) {
            for (double mae = 0.0; mae <= 0.7 + maeHistogramBinWidth_; mae += maeHistogramBinWidth_) {
                double pSuccessPositive = truePositiveMAEHistogram_.getProbability(mae);
                double pFailurePositive = falsePositiveMAEHistogram_.getProbability(mae);
                if (pSuccessPositive < 10.0e-6)
                    pSuccessPositive = 10.0e-6;
                if (pFailurePositive < 10.0e-6)
                    pFailurePositive = 10.0e-6;
                double relPositive = pSuccessPositive * rel;
                double relInvPositive = pFailurePositive * (1.0 - rel);
                double pPositive = relPositive + relInvPositive;

                double pSuccessNegative = falseNegativeMAEHistogram_.getProbability(mae);
                double pFailureNegative = trueNegativeMAEHistogram_.getProbability(mae);
                if (pSuccessNegative < 10.0e-6)
                    pSuccessNegative = 10.0e-6;
                if (pFailureNegative < 10.0e-6)
                    pFailureNegative = 10.0e-6;
                double relNegative = pSuccessNegative * rel;
                double relInvNegative = pFailureNegative * (1.0 - rel);
                double pNegative = relNegative + relInvNegative;

                fprintf(fp, "%lf %lf %lf %lf\n", rel, mae, pPositive, pNegative);
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
    }
}; // class MAEClassifier

} // namespace als

#endif // __MAE_CLASSIFIER_H__
