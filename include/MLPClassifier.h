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

#ifndef __MLP_CLASSIFIER_H__
#define __MLP_CLASSIFIER_H__

#include <yaml-cpp/yaml.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <Histogram.h>

namespace als {

class MLPClassifier {
private:
    std::string classifiersDir_;
    cv::Ptr<cv::ml::ANN_MLP> mlp_;
    double maxResidualError_;
    float positiveTargetValue_, negativeTargetValue_;

    double predictHistogramBinWidth_;
    Histogram positivePredictsHistogram_, negativePredictsHistogram_;

    inline double sigmoid(float pred) {
        return 1.0 / (1.0 + exp(-pred));
    }

    std::vector<double> readPredicts(std::string filePath) {
        FILE *fp = fopen(filePath.c_str(), "r");
        if (fp == NULL) {
            fprintf(stderr, "cannot open %s\n", filePath.c_str());
            exit(1);
        }
        double predict;
        std::vector<double> predicts;
        while (fscanf(fp, "%lf", &predict) != EOF)
            predicts.push_back(predict);
        fclose(fp);
        return predicts;
    }

public:
    MLPClassifier(void):
        classifiersDir_("../classifiers/MLP/"),
        maxResidualError_(1.0),
        positiveTargetValue_(5.0f),
        negativeTargetValue_(-5.0f),
        predictHistogramBinWidth_(0.05)
    {
        mlp_ = cv::ml::ANN_MLP::create();
    }

    inline void setMaxResidualError(double maxResidualError) { maxResidualError_ = maxResidualError; }
    inline void setPreictdHistogramBinWidth(double predictHistogramBinWidth) { predictHistogramBinWidth_ = predictHistogramBinWidth; }

    void trainMLP(std::vector<std::vector<double> > trainSuccessResidualErrors, std::vector<std::vector<double> > trainFailureResidualErrors) {
        int positiveDataNum = (int)trainSuccessResidualErrors.size();
        int negativeDataNum = (int)trainFailureResidualErrors.size();
        int dataNum = positiveDataNum + negativeDataNum;
        int residualErrorsNum = (int)trainSuccessResidualErrors[0].size();

        cv::Mat_<int> layers(3, 1);
        layers(0) = residualErrorsNum;
        layers(1) = (int)(residualErrorsNum / 2.0f);
        layers(2) = 1;
        mlp_->setLayerSizes(layers);
        mlp_->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM);
        mlp_->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 20000, 0.0001));
        mlp_->setTrainMethod(cv::ml::ANN_MLP::BACKPROP, 0.0001);

        cv::Mat inputsMat(dataNum, residualErrorsNum, CV_32F);
        cv::Mat outputsMat(dataNum, 1, CV_32F);
        for (size_t i = 0; i < positiveDataNum; i++) {
            for (size_t j = 0; j < residualErrorsNum; j++) {
                double error = trainSuccessResidualErrors[i][j];
                if (error < 0.0 || maxResidualError_ < error)
                    error = 0.0;
                inputsMat.at<float>(i, j) = (float)error;
            }
            outputsMat.at<float>(i, 0) = positiveTargetValue_;
        }
        for (size_t i = 0; i < negativeDataNum; i++) {
            for (size_t j = 0; j < residualErrorsNum; j++) {
                double error = trainFailureResidualErrors[i][j];
                if (error < 0.0 || maxResidualError_ < error)
                    error = 0.0;
                inputsMat.at<float>(i + positiveDataNum, j) = (float)error;
            }
            outputsMat.at<float>(i + positiveDataNum, 0) = negativeTargetValue_;
        }

        cv::InputArray inputs = inputsMat;
        cv::InputArray outputs = outputsMat;
        cv::Ptr<cv::ml::TrainData> data = cv::ml::TrainData::create(inputs, cv::ml::ROW_SAMPLE, outputs);
        mlp_->train(data, cv::ml::ROW_SAMPLE);
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
        mlp_->predict(inputs, outputs, cv::ml::ROW_SAMPLE);

        std::string xmlFileName = classifiersDir_ + "mlp.xml";
        mlp_->save(xmlFileName);

        std::string positivePredictsFileName = classifiersDir_ + "positive_predicts.txt";
        std::string negativePredictsFileName = classifiersDir_ + "negative_predicts.txt";
        FILE *fpPositivePredicts = fopen(positivePredictsFileName.c_str(), "w");
        FILE *fpNegativePredicts = fopen(negativePredictsFileName.c_str(), "w");
        for (size_t i = 0; i < positiveDataNum; i++)
            fprintf(fpPositivePredicts, "%lf\n", sigmoid(outputs.at<float>(i, 0)));
        for (size_t i = 0; i < negativeDataNum; i++)
            fprintf(fpNegativePredicts, "%lf\n", sigmoid(outputs.at<float>(i + positiveDataNum, 0)));
        fclose(fpPositivePredicts);
        fclose(fpNegativePredicts);

        std::string yamlFile = classifiersDir_ + "classifier.yaml";
        FILE *fpYaml = fopen(yamlFile.c_str(), "w");
        fprintf(fpYaml, "xmlFileName: mlp.xml\n");
        fprintf(fpYaml, "maxResidualError: %lf\n", maxResidualError_);
        fprintf(fpYaml, "positivePredictsFileName: positive_predicts.txt\n");
        fprintf(fpYaml, "negativePredictsFileName: negative_predicts.txt\n");
        fclose(fpYaml);
    }

    void readClassifierParams(void) {
        std::string yamlFile = classifiersDir_ + "classifier.yaml";
        YAML::Node lconf = YAML::LoadFile(yamlFile);

        std::string xmlFileName = classifiersDir_ + lconf["xmlFileName"].as<std::string>();
        mlp_ = cv::ml::ANN_MLP::load(xmlFileName);

        maxResidualError_ = lconf["maxResidualError"].as<double>();

        std::string positivePredictsFileName = classifiersDir_ + lconf["positivePredictsFileName"].as<std::string>();
        std::string negativePredictsFileName = classifiersDir_ + lconf["negativePredictsFileName"].as<std::string>();
        positivePredictsHistogram_ = Histogram(readPredicts(positivePredictsFileName), predictHistogramBinWidth_, 0.0, 1.0);
        negativePredictsHistogram_ = Histogram(readPredicts(negativePredictsFileName), predictHistogramBinWidth_, 0.0, 1.0);
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
        mlp_->predict(input, output);
        double predict = sigmoid(output.at<float>(0, 0));
        return predict;
    }

    double calculateDecisionModel(double predict, double *reliability) {
        double pSuccess = positivePredictsHistogram_.getProbability(predict);
        double pFailure = negativePredictsHistogram_.getProbability(predict);
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
        std::string fileName = classifiersDir_ + "mlp_decision_likelihoods.txt";
        FILE *fp = fopen(fileName.c_str(), "w");
        for (double rel = 0.0; rel <= 1.0 + 0.05; rel += 0.05) {
            for (double pred = 0.0; pred <= 1.0 + 0.05; pred += 0.05) {
                double pSuccess = positivePredictsHistogram_.getProbability(pred);
                double pFailure = negativePredictsHistogram_.getProbability(pred);
                double relLike = pSuccess * rel;
                double relLikeInv = pFailure * (1.0 - rel);
                double p = relLike + relLikeInv;
                fprintf(fp, "%lf %lf %lf\n", rel, pred, p);
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
    }
}; // class MLPClassifier

} // namespace als

#endif // __MLP_CLASSIFIER_H__