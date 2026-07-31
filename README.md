# CPP-Transformer

## 概要

c++で作成したTransformerです<br>
Windowsでしか動かない可能性があります<br>
Linuxではセーブ、ロードがうまくいきませんでした<br>
Datasets are not included in this repository.<br>
Please download them from their original sources.<br>

## コンパイル方法

g++ -std=c++20 -I. -Iexternal *.cpp nn/ops/*.cpp -o main<br>
g++ -std=c++20 -march=native -I. -Iexternal *.cpp nn/ops/*.cpp -o main<br>
g++ -std=c++20 -Ofast -DNDEBUG -march=native -I. -Iexternal *.cpp nn/ops/*.cpp -o main<br>

## 実行方法

main

## 使用しているライブラリ
nlohmann jsonを使いました<br>
https://github.com/nlohmann/json<br>

## License

This project is licensed under the MIT License.
See LICENSE for details.

## Datasets

This project was trained using the following datasets.

- MNIST
  https://www.kaggle.com/datasets/hojjatk/mnist-dataset
  License: (Kaggleページのライセンスに従います)

- WikiText-103
  https://huggingface.co/datasets/Salesforce/wikitext
  License: CC BY-SA / GFDL (see dataset page)

- Databricks Dolly 15k
  https://huggingface.co/datasets/databricks/databricks-dolly-15k
  License: CC BY-SA 3.0

## Pretrained Models

Pretrained weights were trained using the datasets listed above.<br>
Please check the license of each dataset before redistributing the weights.

## Disclaimer

This software is provided "as is" without warranty of any kind.

## 性能 MNIST分類
学習率0.01
80%以上で正解

identity
学習率0.01
time: 3461ms
loss:nan
correct:0

学習率0.001
time: 3839ms
loss:0.326524
correct:0.78725

ReLU
time: 3425ms
loss:0.110122
correct:0.929617

LeakyReLU alpha = 0.01
time: 4477ms
loss:0.116964
correct:0.92315

LeakyReLU alpha = 0.1
time: 3730ms
loss:0.116562
correct:0.9222

SiLU
time: 5445ms
loss:0.107087
correct:0.928567

GELU
time: 4429ms
loss:0.0993964
correct:0.9358

square 学習率0.000001
time: 3373ms
loss:2.20839
correct:0.0130333

cube 学習率0.00000000001
time: 4444ms
loss:nan
correct:0.0130167

exp 学習率0.00000000001
time: 4356ms
loss:nan
correct:0

abs 学習率0.001
time: 6447ms
loss:0.14536
correct:0.889283

Straight_Through_Estimator 最大値を正解　80%以上の自信0
time: 3932ms
loss:1.98987
correct:0.286567