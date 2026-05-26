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
  Linear(int64_t in,int64_t out,bool bias=false)
    : bias_(bias),
      W_({in,out}),
      b_({1,1}),
      output_({1,1}),
      dW_({in,out}),
      db_({1,1}),
      grad_({1,1}){

    if(bias){
      b_ = tensor::Tensor({1,out});
      db_ = tensor::Tensor({1,out});
    }
  }

  const bool bias_;

  const tensor::Tensor *input_ptr_;//逆伝播で必要なため
  tensor::Tensor W_,b_;//重み、バイアス
  tensor::Tensor output_;//出力
  tensor::Tensor dW_,db_;//重みの微分、バイアスの微分
  tensor::Tensor grad_;//次の層に渡す勾配

  const tensor::Tensor& forward(const tensor::Tensor& input,bool training=true) override{
    input_ptr_ = &input;

    //サイズが違うときだけ再確保
    if(input.rank() != output_.rank() || !std::equal(input.shape().begin(),input.shape().end() - 1,output_.shape().begin()) || W_.dim(W_.rank() - 1) != output_.dim(output_.rank() - 1)){
      std::vector<int64_t> output_shape = input.shape();

      output_shape.back() = W_.dim(W_.rank() - 1);

      output_ = tensor::Tensor(output_shape);
    }

    tensor::MatrixView output_view = output_.flatten_matrix_view();

    tensor::MatrixView::matmul(input.flatten_matrix_view(),W_.as_matrix_view({}),output_view);

    if(bias_){
      add_bias();
    }
    
    return output_;
  }

  void add_bias(){
    float *od = output_.data();
    const float *bd = b_.data();

    int64_t rows = output_.shape()[0];
    int64_t cols = output_.shape()[1];

    for(size_t row = 0;row < rows;row++){
      for(size_t col = 0;col < cols;col++){
        od[row * cols + col] += bd[col];
      }
    }
  }

  //逆伝播
  //次の層の勾配を受け取る
  const tensor::Tensor& backward(const tensor::Tensor& grad_output) override{
    if(grad_.shape() != input_ptr_->shape()) grad_ = tensor::Tensor(input_ptr_->shape());

    if(bias_){
      add_db(grad_output);
    }

    const tensor::ConstMatrixView input_view = input_ptr_->flatten_matrix_view();
    const tensor::ConstMatrixView W_view = W_.as_matrix_view({});
    const tensor::ConstMatrixView grad_output_view = grad_output.flatten_matrix_view();

    tensor::MatrixView dW_view = dW_.as_matrix_view({});
    tensor::MatrixView grad_view = grad_.flatten_matrix_view();

    tensor::MatrixView::matmul_add(input_view.t(),grad_output_view,dW_view);

    tensor::MatrixView::matmul(grad_output_view,W_view.t(),grad_view);

    return grad_;
  }

  void add_db(const tensor::Tensor& grad_output){
    float *dbd = db_.data();
    const float *gd = grad_output.data();

    const int64_t rows = grad_output.shape()[0];
    const int64_t cols = grad_output.shape()[1];

    for(int64_t row = 0;row < rows;row++){
      const int64_t front = row * cols;
      for(int64_t col = 0;col < cols;col++){
        const int64_t index = front + col;
        dbd[col] += gd[index];
      }
    }
  }

  //更新
  //学習率、バッチサイズを受け取る
  void step(float lr,int batch_size=64){
    dW_.scale_(lr);
    W_ -= dW_;

    if(bias_){
      db_.scale_(lr);
      b_ -= db_;
    }
  }

  //勾配をリセットする
  void zero_grad(){
    float *dWd = dW_.data();

    std::fill(dWd,dWd + dW_.numel(),0.0f);

    float *dbd = db_.data();

    std::fill(dbd,dbd + db_.numel(),0.0f);
  }

  //層の種類を返す。適切にオーバーライドすること
  std::string get_type() const override{
    return "Linear";
  }

  //文字列にしたいとき使う
  //文字列にしたいとき使う
  std::string to_string() const{
    std::string s;
    s += "W\n";
    s += W_.to_string() + "\n";
    if(bias_){
      s += "b\n" + b_.to_string();
    }
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