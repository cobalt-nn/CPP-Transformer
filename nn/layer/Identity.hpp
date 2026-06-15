#pragma once

#include <string>
#include <random>
#include "nlohmann/json.hpp"
#include "nn/tensor/Tensor.hpp"

namespace cobalt_715::nn::layer{

struct Identity : ILayer{
  const tensor::Tensor& forward(const tensor::Tensor& input,bool training=true) override{
    return input;
  }

  const tensor::Tensor& backward(const tensor::Tensor& grad_output) override{
    return grad_output;
  }

  void step(float lr,int batch_size=64) override{}

  void zero_grad() override{}

  std::string get_type() const override{
    return "Identity";
  }

  nlohmann::ordered_json to_json() const override{
    return nlohmann::ordered_json();
  }

  void random_init(std::mt19937 &gen) override{}
};

}//namespace cobalt_715::nn::layer