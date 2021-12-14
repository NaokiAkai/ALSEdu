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

int main(int argc, char **argv) {
    // プログラム起動時の引数にマップを格納したディレクトリを指定
    if (argv[1] == NULL) {
        std::cerr << "ERROR: argv[1] must be map directory, \
            e.g., ../maps/nic1f/" << std::endl;
        exit(1);
    }

    // gnuplot で描画する範囲を指定
    double plotRange = 10.0;
    // オドメトリ（誤差修正されていない位置）をプロット
    bool plotOdomPose = true;
    // false だとノイズの加算されたスキャンデータがプロットされる
    bool plotGTScan = true;
    // シミュレーションの更新の速さ
    double simulationHz = 10.0;

    // シミュレーション用のクラス
    als::RobotSim robotSim(argv[1], simulationHz);
    // 受け取ったkey の値を端末に表示
    robotSim.setPrintKeyValue(true);

    // キー入力を受け付けるためにOpenCV のウインドウを用いる
    cv::namedWindow("Keyboard Interface Window", cv::WINDOW_NORMAL);

    double usleepTime = (1.0 / simulationHz) * 10e5;
    while (!robotSim.getKillFlag()) {
        // キー入力の受付
        int key = cv::waitKey(200);
        // キー入力に従ってシミュレーション用のパラメータを操作
        robotSim.keyboardOperation(key);
        // シミュレーションの更新
        robotSim.updateSimulation();
        // ロボットの位置（真値）を端末に表示
        robotSim.printRobotPose();
        // gnuplot で表示
        robotSim.plotSimulationWorld(plotRange, plotOdomPose, plotGTScan);
        usleep(usleepTime);
    }
    return 0;
}
