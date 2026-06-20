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
#include "nn/io/BinaryIO.hpp"

namespace cobalt_715::nn::layer{

//全結合層
struct Dense : ILayer{
  Dense(int64_t in,int64_t out)
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
  tensor::Tensor delta_,grad_;//この層での微分、次の層に渡す勾配

  const ops::Activation *act_ = &ops::activations::LeakyReLU;//活性化関数とその微分。デフォルトではLeakyReLU

  const tensor::Tensor& forward(const tensor::Tensor& input,bool training=true) override{
    input_ptr_ = &input;

    //サイズが違うときだけ再確保
    if(input.rank() != z_.rank() || !std::equal(input.shape().begin(),input.shape().end() - 1,z_.shape().begin()) || W_.dim(W_.rank() - 1) != z_.dim(z_.rank() - 1)){
      std::vector<int64_t> output_shape = input.shape();

      output_shape.back() = W_.dim(W_.rank() - 1);

      z_ = tensor::Tensor(output_shape);
      a_ = tensor::Tensor(output_shape);
    }

    tensor::MatrixView z_view = z_.flatten_matrix_view();

    tensor::MatrixView::matmul(input.flatten_matrix_view(),W_.as_matrix_view({}),z_view);

    add_bias_activation();
    
    return a_;
  }

  void add_bias_activation(){
    float *zd = z_.data();
    float *ad = a_.data();
    const float *bd = b_.data();

    int64_t rows = z_.numel() / z_.dim(z_.rank() - 1);
    int64_t cols = z_.dim(z_.rank() - 1);

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

    const tensor::ConstMatrixView input_view = input_ptr_->flatten_matrix_view();
    const tensor::MatrixView W_view = W_.as_matrix_view({});
    const tensor::MatrixView delta_view = delta_.flatten_matrix_view();

    tensor::MatrixView dW_view = dW_.as_matrix_view({});
    tensor::MatrixView grad_view = grad_.flatten_matrix_view();

    tensor::MatrixView::matmul_add(input_view.t(),delta_view,dW_view);

    tensor::MatrixView::matmul(delta_view,W_view.t(),grad_view);

    return grad_;
  }

  void delta_hadamard_add_db(const tensor::Tensor& grad_output){
    if(delta_.shape() != grad_output.shape()){
      delta_ = tensor::Tensor(grad_output.shape());
    }

    float *dd = delta_.data();
    float *dbd = db_.data();
    const float *gd = grad_output.data();
    const float *zd = z_.data();
    const float *ad = a_.data();

    const int64_t rows = delta_.numel() / delta_.dim(delta_.rank() - 1);
    const int64_t cols = delta_.dim(delta_.rank() - 1);

    for(int64_t row = 0;row < rows;row++){
      const int64_t front = row * cols;
      for(int64_t col = 0;col < cols;col++){
        const int64_t index = front + col;
        dd[index] = gd[index] * act_->d_act_(zd[index],ad[index]);

        dbd[col] += dd[index];
      }
    }
  }

  //更新
  //学習率、バッチサイズを受け取る
  void step(float lr,int batch_size=64) override{
    dW_.scale_(lr);
    W_ -= dW_;

    db_.scale_(lr);
    b_ -= db_;
  }

  //勾配をリセットする
  void zero_grad() override{
    float *dWd = dW_.data();

    std::fill(dWd,dWd + dW_.numel(),0.0f);

    float *dbd = db_.data();

    std::fill(dbd,dbd + db_.numel(),0.0f);
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

  nlohmann::ordered_json to_json() const override{
    nlohmann::ordered_json j;

    j["layer_type"] = get_type();
    j["in"] = W_.dim(0);
    j["out"] = W_.dim(1);
    j["activation"] = act_->name;

    return j;
  }

  void save(std::ostream &os) const override{
    io::save(os,W_.data(),W_.numel());
    io::save(os,b_.data(),b_.numel());
  }

  void load(const nlohmann::ordered_json &json,std::istream &is) override{
    if(json.at("layer_type") != get_type()){
      throw std::runtime_error("Dense::load type mismatch");
    }

    io::load(is,W_.data(),W_.numel());
    io::load(is,b_.data(),b_.numel());
  }

  //ランダム初期化する
  void random_init(std::mt19937 &gen) override{
    float limit = sqrt(6.0f / (W_.shape()[0] + W_.shape()[1]));
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