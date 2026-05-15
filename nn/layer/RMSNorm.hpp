#pragma once

#include <string>
#include <random>
#include <cstdint>
#include <stdexcept>
#include "nlohmann/json.hpp"
#include "nn/tensor/Tensor.hpp"
#include "nn/layer/ILayer.hpp"

namespace cobalt_715::nn::layer{

//RMSNorm
struct RMSNorm : ILayer{
  RMSNorm(size_t size) :
    gamma_(size),
    d_gamma_(size),
    rms_(0),
    output_({0}),
    delta_(0),
    grad_({0}){}

  const tensor::Tensor *input_;

  const float epsilon_ = 1e-6f;//小さな定数
  std::vector<float> gamma_;//拡大項
  std::vector<float> d_gamma_;//gamma_の微分

  std::vector<double> rms_;//rms

  tensor::Tensor output_;//この層の出力

  std::vector<double> delta_;//grad_を求めるための一時的なもの
  tensor::Tensor grad_;//次の層に渡す勾配

  const tensor::Tensor& forward(const tensor::Tensor& input,bool training=true) override{
    input_ = &input;

    const int64_t cols = input.shape()[input.shape().size() - 1];
    const int64_t big_rows = input.numel() / cols;

    if(cols != gamma_.size()){
      throw std::runtime_error(
        "RMSNorm: gamma size (" +
        std::to_string(gamma_.size()) +
        ") != cols (" +
        std::to_string(cols) + ")"
      );
    }

    set_rms(big_rows,cols);

    if(output_.shape() != input.shape()) output_ = tensor::Tensor(input.shape());

    const float *id = input.data();
    float *od = output_.data();

    for(int64_t i = 0;i < big_rows;i++){
      for(int64_t j = 0;j < cols;j++){
        od[i * cols + j] = static_cast<float>(gamma_[j] * id[i * cols + j] / rms_[i]);
      }
    }

    return output_;
  }

  inline void set_rms(const int64_t big_rows,const int64_t cols){
    if(big_rows != rms_.size()) rms_ = std::vector<double>(big_rows);

    const float *id = input_->data();

    for(int64_t i = 0;i < big_rows;i++){
      double sum = 0.0f;
      for(int64_t j = 0;j < cols;j++){
        double d = id[i * cols + j];
        sum += d * d;
      }
      rms_[i] = std::sqrt(sum / cols + epsilon_);
    }
  }

  const tensor::Tensor& backward(const tensor::Tensor& grad_output) override{
    if(grad_output.shape() != output_.shape()){
      throw std::invalid_argument(
        "RMSNorm::backward: output gradient shape mismatch"
      );
    }

    if(grad_output.shape() != grad_.shape()) grad_ = tensor::Tensor(grad_output.shape());

    if(delta_.size() != rms_.size()){
      delta_ = std::vector<double>(rms_.size());
    }else{
      std::fill(delta_.begin(),delta_.end(),0.0);
    }

    const int64_t cols = input_->shape()[input_->shape().size() - 1];
    const int64_t big_rows = input_->numel() / cols;

    const float *id = input_->data();
    const float *god = grad_output.data();
    float* gd = grad_.data();

    for(int64_t i = 0;i < big_rows;i++){
      for(int64_t j = 0;j < cols;j++){
        d_gamma_[j] += id[i * cols + j] * god[i * cols + j] / rms_[i];
        delta_[i] += gamma_[j] * id[i * cols + j] * god[i * cols + j];
      }
    }

    for(int64_t i = 0;i < big_rows;i++){
      for(int64_t j = 0;j < cols;j++){
        gd[i * cols + j] = static_cast<float>((gamma_[j] * god[i * cols + j] / rms_[i]) - (delta_[i] * id[i * cols + j] / (cols * rms_[i] * rms_[i] * rms_[i])));
      }
    }

    return grad_;
  }

  //更新
  //学習率、バッチサイズを受け取る
  void step(float lr,int batch_size=64) override{
    for(size_t i = 0;i < gamma_.size();i++){
      gamma_[i] -= d_gamma_[i] * lr;
    }
  }

  //勾配をリセットする
  void zero_grad() override{
    std::fill(d_gamma_.begin(),d_gamma_.end(),0.0f);
  }

  //層の種類を返す。適切にオーバーライドすること
  std::string get_type() const override{
    return "RMSNorm";
  }

  //文字列にしたいとき使う
  std::string to_string() const{
    return get_type() + "::to_string() is undef";
  }

  //json形式で保存するとき使う
  nlohmann::ordered_json to_json() const override{
    return nlohmann::ordered_json();
  }

  //ランダム初期化する
  void random_init(std::mt19937 &gen) override{
    std::fill(gamma_.begin(),gamma_.end(),1.0f);//ランダム初期化といったな。あれは嘘だ
  }
};

}//namespace cobalt_715::nn::layer