#pragma once

//#include <iostream>
#include <string>
#include <random>
#include "nlohmann/json.hpp"
#include "nn/tensor/Tensor.hpp"

namespace cobalt_715::nn::layer{

struct ReZero : ILayer{
  ReZero(int64_t in,int64_t layer_out,std::unique_ptr<ILayer> layer)
    : layer_(std::move(layer)),
      WO_(layer_out,in),
      output_({1}),
      grad_({1}){}

  std::unique_ptr<ILayer> layer_;

  Linear WO_;

  const tensor::Tensor *WO_output_ptr_;

  tensor::Tensor output_;
  tensor::Tensor grad_;

  float alpha_ = 0.0f;
  float d_alpha_ = 0.0f;

  const tensor::Tensor& forward(const tensor::Tensor& input,bool training=true) override{
    WO_output_ptr_ = &WO_.forward(layer_->forward(input,training),training);

    if(output_.shape() != input.shape()){
      output_ = tensor::Tensor(input.shape());
    }else{
      std::fill(output_.data(),output_.data() + output_.numel(),0.0f);
    }

    tensor::Tensor::scale(*WO_output_ptr_,alpha_,output_);

    output_ += input;

    return output_;
  }

  const tensor::Tensor& backward(const tensor::Tensor& grad_output) override{
    if(grad_.shape() != grad_output.shape()){
      grad_ = tensor::Tensor(grad_output.shape());
    }else{
      std::fill(grad_.data(),grad_.data() + grad_.numel(),0.0f);
    }

    tensor::Tensor::scale(layer_->backward(WO_.backward(grad_output)),alpha_,grad_);

    for(size_t i = 0;i < grad_output.numel();i++){
      grad_.data()[i] += grad_output.data()[i];
    }

    for(size_t i = 0;i < grad_output.numel();i++){
      d_alpha_ += WO_output_ptr_->data()[i] * grad_output.data()[i];
    }

    return grad_;
  }

  void step(float lr,int batch_size=64) override{
    WO_.step(lr,batch_size);
    layer_->step(lr,batch_size);

    alpha_ -= d_alpha_ * lr;

    //std::cout << "alpha:" << alpha_ << std::endl;
  }

  void zero_grad() override{
    WO_.zero_grad();
    layer_->zero_grad();

    d_alpha_ = 0.0f;
  }

  void reset(){
    layer_->reset();
  }

  std::string get_type() const override{
    return "ReZero";
  }

  std::string to_string() const override{
    std::string s = get_type() + "\n" + WO_.to_string() + "\n";
    s += layer_->to_string();
    return s;
  }

  //json形式で保存するとき使う
  nlohmann::ordered_json to_json() const override{
    return  nlohmann::ordered_json();
  }

  //ランダム初期化する
  void random_init(std::mt19937 &gen) override{
    WO_.random_init(gen);
    layer_->random_init(gen);
  }
};

}//namespace cobalt_715::nn::layer