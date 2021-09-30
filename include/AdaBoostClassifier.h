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

#ifndef __ADABOOST_CLASSIFIER_H__
#define __ADABOOST_CLASSIFIER_H__

#include <yaml-cpp/yaml.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

namespace als {

class AdaBoostClassifier {
private:
    std::string classifiersDir_;
    cv::Ptr<cv::ml::Boost> boost_;
    double maxResidualError_;

    int weakCount_;
    double weightTrimRate_;

    int truePositiveNum_;
    int falsePositiveNum_;
    int trueNegativeNum_;
    int falseNegativeNum_;
    double truePositive_;
    double falsePositive_;
    double trueNegative_;
    double falseNegative_;

public:
    AdaBoostClassifier(void):
        classifiersDir_("../classifiers/AdaBoost/"),
        maxResidualError_(1.0),
        weakCount_(128),
        weightTrimRate_(0.95)
    {
        boost_ = cv::ml::Boost::create();
        boost_->setBoostType(cv::ml::Boost::REAL);
        boost_->setWeakCount(weakCount_);
        boost_->setWeightTrimRate(weightTrimRate_);
    }

    inline void setMaxResidualError(double maxResidualError) { maxResidualError_ = maxResidualError; }

    inline void setWeakCount(int weakCount) {
        weakCount_ = weakCount;
        boost_->setWeakCount(weakCount_);
    }

    inline void setWeightTrimRate(double weightTrimRate) {
        weightTrimRate_ = weightTrimRate;
        boost_->setWeightTrimRate(weightTrimRate_);
    }

    void trainAdaBoost(std::vector<std::vector<double> > trainSuccessResidualErrors, std::vector<std::vector<double> > trainFailureResidualErrors) {
        int positiveDataNum = (int)trainSuccessResidualErrors.size();
        int negativeDataNum = (int)trainFailureResidualErrors.size();
        int dataNum = positiveDataNum + negativeDataNum;
        int residualErrorsNum = (int)trainSuccessResidualErrors[0].size();

        cv::Mat inputsMat(dataNum, residualErrorsNum, CV_32F);
        cv::Mat outputsMat(dataNum, 1, CV_32F);
        for (size_t i = 0; i < positiveDataNum; i++) {
            for (size_t j = 0; j < residualErrorsNum; j++) {
                double error = trainSuccessResidualErrors[i][j];
                if (error < 0.0 || maxResidualError_ < error)
                    error = 0.0;
                inputsMat.at<float>(i, j) = (float)error;
            }
            outputsMat.at<float>(i, 0) = 1.0f;
        }
        for (size_t i = 0; i < negativeDataNum; i++) {
            for (size_t j = 0; j < residualErrorsNum; j++) {
                double error = trainFailureResidualErrors[i][j];
                if (error < 0.0 || maxResidualError_ < error)
                    error = 0.0;
                inputsMat.at<float>(i + positiveDataNum, j) = (float)error;
            }
            outputsMat.at<float>(i + positiveDataNum, 0) = -1.0f;
        }

        cv::InputArray inputs = inputsMat;
        cv::InputArray outputs = outputsMat;
        cv::Ptr<cv::ml::TrainData> data = cv::ml::TrainData::create(inputs, cv::ml::ROW_SAMPLE, outputs);
        boost_->train(data, cv::ml::ROW_SAMPLE);
    }

    void writeClassifierParams(std::vector<std::vector<double> > testSuccessResidualErrors, std::vector<std::vector<double> > testFailureResidualErrors) {
        int positiveDataNum = (int)testSuccessResidualErrors.size();
        int negativeDataNum = (int)testFailureResidualErrors.size();
        int dataNum = positiveDataNum + negativeDataNum;
        int residualErrorsNum = (int)testSuccessResidualErrors[0].size();

        cv::Mat inputsMat(dataNum, residualErrorsNum, CV_32F);
        for (size_t i = 0; i < positiveDataNum; i++) {
            for (size_t j = 0; j < residualErrorsNum; j++) {
                double error = testSuccessResidualErrors[i][j];
                if (error < 0.0 || maxResidualError_ < error)
                    error = 0.0;
                inputsMat.at<float>(i, j) = (float)error;
            }
        }
        for (size_t i = 0; i < negativeDataNum; i++) {
            for (size_t j = 0; j < residualErrorsNum; j++) {
                double error = testFailureResidualErrors[i][j];
                if (error < 0.0 || maxResidualError_ < error)
                    error = 0.0;
                inputsMat.at<float>(i + positiveDataNum, j) = (float)error;
            }
        }

        cv::InputArray inputs = inputsMat;
        cv:: Mat outputs;
        boost_->predict(inputs, outputs, cv::ml::COL_SAMPLE);

        std::string xmlFileName = classifiersDir_ + "adaboost.xml";
        boost_->save(xmlFileName);

        truePositiveNum_ = falseNegativeNum_ = trueNegativeNum_ = falsePositiveNum_ = 0;
        for (size_t i = 0; i < positiveDataNum; i++) {
            if (abs(outputs.at<float>(i, 0)) == 1)
                truePositiveNum_++;
            else
                falseNegativeNum_++;
        }
        for (size_t i = 0; i < negativeDataNum; i++) {
            if (abs(outputs.at<float>(i + positiveDataNum, 0)) == 0)
                trueNegativeNum_++;
            else
                falsePositiveNum_++;
        }

        printf("truePositiveNum = %d\n", truePositiveNum_);
        printf("falseNegativeNum = %d\n", falseNegativeNum_);
        printf("trueNegativeNum = %d\n", trueNegativeNum_);
        printf("falsePositiveNum = %d\n", falsePositiveNum_);

        std::string yamlFile = classifiersDir_ + "classifier.yaml";
        FILE *fpYaml = fopen(yamlFile.c_str(), "w");
        fprintf(fpYaml, "xmlFileName: adaboost.xml\n");
        fprintf(fpYaml, "maxResidualError: %lf\n", maxResidualError_);
        fprintf(fpYaml, "truePositiveNum: %d\n", truePositiveNum_);
        fprintf(fpYaml, "falseNegativeNum: %d\n", falseNegativeNum_);
        fprintf(fpYaml, "trueNegativeNum: %d\n", trueNegativeNum_);
        fprintf(fpYaml, "falsePositiveNum: %d\n", falsePositiveNum_);
        fclose(fpYaml);
    }

    void readClassifierParams(void) {
        std::string yamlFile = classifiersDir_ + "classifier.yaml";
        YAML::Node lconf = YAML::LoadFile(yamlFile);

        std::string xmlFileName = classifiersDir_ + lconf["xmlFileName"].as<std::string>();
        boost_ = cv::ml::StatModel::load<cv::ml::Boost>(xmlFileName);

        maxResidualError_ = lconf["maxResidualError"].as<double>();

        truePositiveNum_ = lconf["truePositiveNum"].as<int>();
        falseNegativeNum_ = lconf["falseNegativeNum"].as<int>();
        trueNegativeNum_ = lconf["trueNegativeNum"].as<int>();
        falsePositiveNum_ = lconf["falsePositiveNum"].as<int>();

        int totalSampleNum = truePositiveNum_ + falseNegativeNum_ + falsePositiveNum_ + trueNegativeNum_;
        truePositive_ = (double)truePositiveNum_ / (double)totalSampleNum;
        falsePositive_ = (double)falsePositiveNum_ / (double)totalSampleNum;
        trueNegative_ = (double)trueNegativeNum_ / (double)totalSampleNum;
        falseNegative_ = (double)falseNegativeNum_ / (double)totalSampleNum;
    }

    double predict(std::vector<double> residualErrors) {
        int residualErrorsNum = (int)residualErrors.size();
        cv::Mat inputMat(1, residualErrorsNum, CV_32F);
        for (size_t i = 0; i < residualErrorsNum; i++) {
            double error = residualErrors[i];
            if (error < 0.0 || maxResidualError_ < error)
                error = 0.0;
            inputMat.at<float>(0, i) = (float)error;
        }

        cv::InputArray input = inputMat;
        cv::Mat output;
        boost_->predict(input, output, cv::ml::COL_SAMPLE);
        return output.at<float>(0, 0);
    }

    double calculateDecisionModel(int predict, double *reliability) {
        double pSuccess, pFailure;
        if (predict == 1) {
            pSuccess = truePositive_;
            pFailure = falsePositive_;
        } else {
            pSuccess = falseNegative_;
            pFailure = trueNegative_;
        }
        double rel = pSuccess * *reliability;
        double relInv = pFailure * (1.0 - *reliability);
        double p = rel + relInv;
        if (p > 1.0)
            p = 1.0;
        *reliability = rel / (rel + relInv);
        if (*reliability > 0.99999)
            *reliability = 0.99999;
        if (*reliability < 0.00001)
            *reliability = 0.00001;
        return p;
    }

    void writeDecisionLikelihoods(void) {
        readClassifierParams();
        std::string fileName = classifiersDir_ + "adaboost_decision_likelihoods.txt";
        FILE *fp = fopen(fileName.c_str(), "w");
        for (double rel = 0.0; rel <= 1.0 + 0.05; rel += 0.05) {
            double pSuccessPositive = truePositive_;
            double pFailurePositive = falsePositive_;
            double relPositive = pSuccessPositive * rel;
            double relInvPositive = pFailurePositive * (1.0 - rel);
            double pPositive = relPositive + relInvPositive;

            double pSuccessNegative = falseNegative_;
            double pFailureNegative = trueNegative_;
            double relNegative = pSuccessNegative * rel;
            double relInvNegative = pFailureNegative * (1.0 - rel);
            double pNegative = relNegative + relInvNegative;

            fprintf(fp, "%lf %lf %lf\n", rel, pPositive, pNegative);
        }
        fclose(fp);
    }

}; // class AdaBoostClassifier

} // namespace als

#endif // __ADABOOST_CLASSIFIER_H__
