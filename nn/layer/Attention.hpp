#pragma once

#include <string>
#include <random>
#include "nlohmann/json.hpp"
#include "nn/tensor/Tensor.hpp"

namespace cobalt_715::nn::layer{

//self attention
struct Attention : ILayer{
  const tensor::Tensor& forward(const tensor::Tensor& input,bool training=true) = 0;

  const tensor::Tensor& backward(const tensor::Tensor& grad_output) = 0;

  void step(float lr,int batch_size=64) = 0;

  void zero_grad() = 0;

  std::string get_type() const = 0;

  std::string to_string() const{
    return get_type() + "::to_string() is undef";
  }

  nlohmann::ordered_json to_json() const = 0;

  void random_init(std::mt19937 &gen) = 0;
};

}//namespace cobalt_715::nn::layer