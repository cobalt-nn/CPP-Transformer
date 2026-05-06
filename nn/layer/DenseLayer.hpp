#pragma once

#include <string>
#include <random>
#include "ILayer.hpp"
#include "nlohmann/json.hpp"
#include "nn/tensor/Tensor.hpp"
#include "nn/tensor/MatrixView.hpp"
#include "nn/ops/Activation.hpp"
#include "nn/ops/Acts.hpp"

namespace cobalt_715::nn::layer{

struct DenseLayer : ILayer{
  const tensor::Tensor *input_ptr_;//逆伝播で必要なため
  tensor::Tensor W_,b_;//重み、バイアス
  tensor::Tensor z_,a_;//活性化前、活性化後
  tensor::Tensor dW_,db_;//重みの微分、バイアスの微分
  tensor::Tensor delta_,grad_;//この総出の微分、次の層に渡す勾配

  //全結合層
  const tensor::Tensor& forward(const tensor::Tensor& input,bool training=true) override{
    if(input.shape().size() > 1) throw;//行列までのみ
    input_ptr_ = &input;

    //tensor::MatrixView 

    
    
    return a_;
  }

  //逆伝播
  //次の層の勾配を受け取る
  const tensor::Tensor& backward(const tensor::Tensor& grad_output) override{
    return a_;
  }

  //更新
  //学習率、バッチサイズを受け取る
  void step(double lr,int batch_size=64){
  }

  //勾配をリセットする
  void zero_grad(){
  }

  //層の種類を返す。適切にオーバーライドすること
  std::string get_type() const override{
    return "Dense";
  }

  //文字列にしたいとき使う
  std::string to_string() const{
    return get_type() + "::to_string() is undef";
  }

  //json形式で保存するとき使う
  nlohmann::ordered_json to_json(){
    return nlohmann::ordered_json();
  }

  //ランダム初期化する
  void random_init(std::mt19937 &gen) override{
  }
};

}//namespace cobalt_715::nn::layer