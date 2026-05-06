#pragma once

#include <iostream>
#include <string>
#include <random>
#include <cstdint>
#include <stdexcept>
#include "ILayer.hpp"
#include "nlohmann/json.hpp"
#include "nn/tensor/Tensor.hpp"
#include "nn/tensor/MatrixView.hpp"
#include "nn/ops/Activation.hpp"
#include "nn/ops/Acts.hpp"

namespace cobalt_715::nn::layer{

struct DenseLayer : ILayer{
  DenseLayer(int64_t in,int64_t out)
    : W_({in,out}),
      b_({1,out}),
      z_({0}),
      a_({0}),
      dW_({0}),
      db_({0}),
      delta_({0}),
      grad_({0}){

    float *wd = W_.data();

    for(int i = 0;i < W_.numel();i++){
      wd[i] = static_cast<float>(i);
    }

    float *bd = b_.data();

    for(int i = 0;i < b_.numel();i++){
      bd[i] = static_cast<float>(i);
    }
  }

  const tensor::Tensor *input_ptr_;//逆伝播で必要なため
  tensor::Tensor W_,b_;//重み、バイアス
  tensor::Tensor z_,a_;//活性化前、活性化後
  tensor::Tensor dW_,db_;//重みの微分、バイアスの微分
  tensor::Tensor delta_,grad_;//この総出の微分、次の層に渡す勾配

  const ops::Activation *act = &ops::activations::LeakyReLU;//活性化関数とその微分。デフォルトではLeakyReLU

  //全結合層
  const tensor::Tensor& forward(const tensor::Tensor& input,bool training=true) override{
    if(input.shape().size() > 2) throw std::runtime_error("DenseLayer: input must be 1D");//行列までのみ
    input_ptr_ = &input;

    if(input.shape() != z_.shape()) z_ = tensor::Tensor({input.shape()[0],W_.shape()[1]});
    if(input.shape() != a_.shape()) a_ = tensor::Tensor({input.shape()[0],W_.shape()[1]});

    tensor::MatrixView z_view = z_.as_matrix_view({});

    tensor::MatrixView::matmul(input.as_matrix_view({}),W_.as_matrix_view({}),z_view);

    add_bias_activation();
    
    return a_;
  }

  void add_bias_activation(){
    float *zd = z_.data();
    float *ad = a_.data();
    const float *bd = b_.data();

    int64_t rows = z_.shape()[0];
    int64_t cols = z_.shape()[1];

    for(size_t row = 0;row < rows;row++){
      for(size_t col = 0;col < cols;col++){
        zd[row * cols + col] += bd[col];
        ad[row * cols + col] = act->act_(zd[row * cols + col]);
      }
    }
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
    std::string s;
    s += "activation " + act->name;
    s += "\nW\n";
    s += W_.to_string() + "\nb\n";
    s += b_.to_string();
    return s;
  }

  //json形式で保存するとき使う
  nlohmann::ordered_json to_json() const{
    return nlohmann::ordered_json();
  }

  //ランダム初期化する
  void random_init(std::mt19937 &gen) override{
  }
};

}//namespace cobalt_715::nn::layer