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
      z_({1,1}),
      a_({1,1}),
      dW_({in,out}),
      db_({1,out}),
      delta_({1,1}),
      grad_({1,1}){}

  const tensor::Tensor *input_ptr_;//逆伝播で必要なため
  tensor::Tensor W_,b_;//重み、バイアス
  tensor::Tensor z_,a_;//活性化前、活性化後
  tensor::Tensor dW_,db_;//重みの微分、バイアスの微分
  tensor::Tensor delta_,grad_;//この総出の微分、次の層に渡す勾配

  const ops::Activation *act_ = &ops::activations::LeakyReLU;//活性化関数とその微分。デフォルトではLeakyReLU

  //全結合層
  const tensor::Tensor& forward(const tensor::Tensor& input,bool training=true) override{
    if(input.shape().size() > 2) throw std::runtime_error("DenseLayer: input must be 1D");//行列までのみ
    input_ptr_ = &input;

    //サイズが違うときだけ再確保
    if(input.shape()[0] != z_.shape()[0] || W_.shape()[1] != z_.shape()[1]) z_ = tensor::Tensor({input.shape()[0],W_.shape()[1]});
    if(input.shape()[0] != a_.shape()[0] || W_.shape()[1] != a_.shape()[1]) a_ = tensor::Tensor({input.shape()[0],W_.shape()[1]});

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
        ad[row * cols + col] = act_->act_(zd[row * cols + col]);
      }
    }
  }

  //逆伝播
  //次の層の勾配を受け取る
  const tensor::Tensor& backward(const tensor::Tensor& grad_output) override{
    delta_hadamard_add_db(grad_output);

    if(grad_.shape() != input_ptr_->shape()) grad_ = tensor::Tensor(input_ptr_->shape());

    const tensor::ConstMatrixView input_view = input_ptr_->as_matrix_view({});
    const tensor::MatrixView W_view = W_.as_matrix_view({});
    const tensor::MatrixView delta_view = delta_.as_matrix_view({});

    tensor::MatrixView dW_view = dW_.as_matrix_view({});
    tensor::MatrixView grad_view = grad_.as_matrix_view({});

    tensor::MatrixView::matmul(input_view.t(),delta_view,dW_view);

    tensor::MatrixView::matmul(delta_view,W_view.t(),grad_view);

    return grad_;
  }

  void delta_hadamard_add_db(const tensor::Tensor& grad_output){
    if(delta_.shape()[1] != grad_output.shape()[1] || delta_.shape()[0] != grad_output.shape()[0]){
      delta_ = tensor::Tensor({grad_output.shape()[0],grad_output.shape()[1]});
    }

    float *dd = delta_.data();
    float *dbd = db_.data();
    const float *gd = grad_output.data();
    const float *zd = z_.data();
    const float *ad = a_.data();

    const int64_t rows = delta_.shape()[0];
    const int64_t cols = delta_.shape()[1];

    for(int64_t row = 0;row < rows;row++){
      const int64_t front = row * cols;
      for(int64_t col = 0;col < cols;col++){
        const int64_t index = front + col;
        dd[index] = gd[index] * act_->d_act_(zd[index],ad[index]);

        if(row == 0){
          dbd[col] = dd[index];
        }else{
          dbd[col] += dd[index];
        }
      }
    }
  }

  //更新
  //学習率、バッチサイズを受け取る
  void step(float lr,int batch_size=64){
    dW_.scale_(lr);
    W_ -= dW_;

    db_.scale_(lr);
    b_ -= db_;
  }

  //勾配をリセットする
  void zero_grad(){
    float *dWd = dW_.data();

    for(int64_t i = 0;i < dW_.numel();i++){
      dWd[i] = 0;
    }

    float *dbd = db_.data();

    for(int64_t i = 0;i < db_.numel();i++){
      dbd[i] = 0;
    }
  }

  //層の種類を返す。適切にオーバーライドすること
  std::string get_type() const override{
    return "Dense";
  }

  //文字列にしたいとき使う
  std::string to_string() const{
    std::string s;
    s += "activation " + act_->name;
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
    float limit = sqrt(2.0f / (W_.shape()[0] + W_.shape()[1]));
    std::uniform_real_distribution<float> dist(-limit,limit);

    float *Wd = W_.data();

    for(int64_t i = 0;i < W_.numel();i++){
      Wd[i] = dist(gen);
    }

    float *bd = b_.data();

    for(int64_t i = 0;i < b_.numel();i++){
      bd[i] = dist(gen);
    }
  }
};

}//namespace cobalt_715::nn::layer