# CPP-Transformer

## 概要

c++で作成したTransformerです<br>
Windowsでしか動かない可能性があります<br>
Linuxではセーブ、ロードがうまくいきませんでした<br>
Datasets are not included in this repository.<br>
Please download them from their original sources.<br>

## コンパイル方法

g++ -std=c++20 -I. -Iexternal *.cpp nn/ops/*.cpp data/*.cpp -o main<br>
g++ -std=c++20 -march=native -I. -Iexternal *.cpp nn/ops/*.cpp data/*.cpp -o main<br>
g++ -std=c++20 -Ofast -DNDEBUG -march=native -I. -Iexternal *.cpp nn/ops/*.cpp data/*.cpp -o main<br>

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