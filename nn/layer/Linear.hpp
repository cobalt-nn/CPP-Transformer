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

namespace cobalt_715::nn::layer{

//モデル保持と微分を楽にするために行列積単体での層
struct Linear : ILayer{
  Linear(int64_t in,int64_t out)
    : W_({in,out}),
      output_({1,1}),
      dW_({in,out}),
      grad_({1,1}){}

  const tensor::Tensor *input_ptr_;//逆伝播で必要なため
  tensor::Tensor W_;//重み
  tensor::Tensor output_;//出力
  tensor::Tensor dW_;//重みの微分、バイアスの微分
  tensor::Tensor grad_;//次の層に渡す勾配

  const tensor::Tensor& forward(const tensor::Tensor& input,bool training=true) override{
    if(input.shape().size() > 2) throw std::runtime_error("DenseLayer: input must be 2D");//行列までのみ
    input_ptr_ = &input;

    //サイズが違うときだけ再確保
    if(input.shape()[0] != output_.shape()[0] || W_.shape()[1] != output_.shape()[1]) output_ = tensor::Tensor({input.shape()[0],W_.shape()[1]});

    tensor::MatrixView output_view = output_.as_matrix_view({});

    tensor::MatrixView::matmul(input.as_matrix_view({}),W_.as_matrix_view({}),output_view);
    
    return output_;
  }

  //逆伝播
  //次の層の勾配を受け取る
  const tensor::Tensor& backward(const tensor::Tensor& grad_output) override{
    if(grad_.shape() != input_ptr_->shape()) grad_ = tensor::Tensor(input_ptr_->shape());

    const tensor::ConstMatrixView input_view = input_ptr_->as_matrix_view({});
    const tensor::ConstMatrixView W_view = W_.as_matrix_view({});
    const tensor::ConstMatrixView grad_output_view = grad_output.as_matrix_view({});

    tensor::MatrixView dW_view = dW_.as_matrix_view({});
    tensor::MatrixView grad_view = grad_.as_matrix_view({});

    tensor::MatrixView::matmul_add(input_view.t(),grad_output_view,dW_view);

    tensor::MatrixView::matmul(grad_output_view,W_view.t(),grad_view);

    return grad_;
  }

  //更新
  //学習率、バッチサイズを受け取る
  void step(float lr,int batch_size=64){
    dW_.scale_(lr);
    W_ -= dW_;
  }

  //勾配をリセットする
  void zero_grad(){
    float *dWd = dW_.data();

    std::fill(dWd,dWd + dW_.numel(),0.0f);
  }

  //層の種類を返す。適切にオーバーライドすること
  std::string get_type() const override{
    return "Linear";
  }

  //文字列にしたいとき使う
  std::string to_string() const{
    std::string s;
    s += "\nW\n";
    s += W_.to_string();
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
  }
};

}//namespace cobalt_715::nn::layer