# ALSEdu

## ALSEduについて

ALSEduは、開発者が研究してきた自己位置推定システム（Advanced Localization System: ALS）の学習用のパッケージです。なお、ALSEduの詳細を解説した書籍が[1]になります（21年9月30日現在で、タイトル仮の出版予定です）。

ALSEduには、パーティクルフィルタに基づく自己位置推定法であるモンテカルロ自己位置推定（Monte Carlo localization: MCL）が実装されています。また、MCLをベースとして、開発者が[2, 3, 4, 5]の論文で発表した手法が実装されています。詳細は[1]の書籍を参考にしてください。





## 開発言語・動作環境

C++を用いて開発されています。Ubuntu 18.04と20.04で動作することを確認していますが、その他のLinux環境でも動作すると思います。但し、WindowsやMacの環境で利用するためには、適宜修正が必要になると思います（Linux以外の環境で動作するかは確認していません）。





## インストール

C++の開発環境を構築した後に、CMake、yaml-cpp、gnuplotおよびOpenCVをインストールしてください（書籍内では、Versionは4.2.0のOpenCVを使用しています）。これらのインストール方法は、ここでは割愛します。

ALSEduは以下の様にインストール、コンパイルしてください。

~~~
$ git clone https://github.com/NaokiAkai/ALSEdu.git
$ cd ALSEdu
$ mkdir build
$ cd build
$ cmake ..
$ make
~~~

コンパイル後は、例えば以下のコマンドでMCLを実行できます。

~~~
$ ./MCL ../maps/nic1f/
~~~

実行すると、2つのgnuplotの画面と、1つのキー入力受付ウインドウが表示されます。キー入力受付ウインドウをアクィティブにした状態で矢印キーを押すと、ロボットが移動します。





## 参考文献

[1] 赤井直紀．"LiDARを用いた高度自己位置推定システム：移動ロボットのための自己位置推定の高性能化とその実装例（仮）"．コロナ社（出版予定）．

[2] Naoki Akai, Luis Yoichi Morales, and Hiroshi Murase. "Mobile robot localization considering class of sensor observations," In *Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, pp. 3159-3166, 2018. 

[3] Naoki Akai, Luis Yoichi Morales, Hiroshi Murase. "Simultaneous pose and reliability estimation using convolutional neural network and Rao-Blackwellized particle filter," *Advanced Robotics*, vol. 32, no. 17, pp. 930-944, 2018. 

[4] Naoki Akai, Luis Yoichi Morales, Takatsugu Hirayama, and Hiroshi Murase. "Misalignment recognition using Markov random fields with fully connected latent variables for detecting localization failures," *IEEE Robotics and Automation Letters*, vol. 4, no. 4, pp. 3955-3962, 2019.

[5] Naoki Akai, Takatsugu Hirayama, and Hiroshi Murase. "Hybrid localization using model- and learning-based methods: Fusion of Monte Carlo and E2E localizations via importance sampling," In *Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)*, pp. 6469-6475, 2020.





## ライセンス

Mozilla Public License Version 2.0
