#pragma once

#include <string>
#include <random>
#include "nlohmann/json.hpp"
#include "nn/tensor/Tensor.hpp"

namespace cobalt_715::nn::layer{

//層の基底クラス
//全結合やCNNを同じTrainerで扱えるようにしている
struct ILayer{
  //順伝播
  //前層の出力を受け取る
  //training=falseなら学習用にデータを保存しなくてもいい
  virtual const tensor::Tensor& forward(const tensor::Tensor& input,bool training=true) = 0;

  //逆伝播
  //次の層の勾配を受け取る
  virtual const tensor::Tensor& backward(const tensor::Tensor& grad_output) = 0;

  //更新
  //学習率、バッチサイズを受け取る
  virtual void step(double lr,int batch_size=64) = 0;

  //勾配をリセットする
  virtual void zero_grad() = 0;

  //層の種類を返す。適切にオーバーライドすること
  virtual std::string get_type() const = 0;

  //文字列にしたいとき使う
  virtual std::string to_string() const{
    return get_type() + "::to_string() is undef";
  }

  //json形式で保存するとき使う
  virtual nlohmann::ordered_json to_json() const = 0;

  //ランダム初期化する
  virtual void random_init(std::mt19937 &gen) = 0;

  virtual ~ILayer(){}
};

}//namespace cobalt_715::nn::layer