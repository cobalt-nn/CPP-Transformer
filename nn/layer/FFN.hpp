#pragma once

#include <string>
#include <random>
#include <cstdint>
#include "ILayer.hpp"
#include "Dense.hpp"
#include "Linear.hpp"
#include "nlohmann/json.hpp"
#include "nn/tensor/Tensor.hpp"

namespace cobalt_715::nn::layer{

struct FFN : ILayer{
  FFN(int64_t in)
    : dense_(in,in * 4),
      linear_(in * 4,in){}

  FFN(int64_t in,int64_t out1,int64_t out2)
    : dense_(in,out1),
      linear_(out1,out2,true){}

  Dense dense_;
  Linear linear_;

  const tensor::Tensor& forward(const tensor::Tensor& input,bool training=true) override{
    return linear_.forward(dense_.forward(input,training),training);
  }

  const tensor::Tensor& backward(const tensor::Tensor& grad_output) override{
    return dense_.backward(linear_.backward(grad_output));
  }

  void step(float lr,int batch_size=64) override{
    dense_.step(lr,batch_size);
    linear_.step(lr,batch_size);
  }

  void zero_grad() override{
    dense_.zero_grad();
    linear_.zero_grad();
  }

  std::string get_type() const override{
    return "FFN";
  }

   std::string to_string() const override{
    return get_type() + "::to_string() is undef";
  }

  nlohmann::ordered_json to_json() const override{
    return nlohmann::ordered_json();
  }

  void random_init(std::mt19937 &gen) override{
    dense_.random_init(gen);
    linear_.random_init(gen);
  }
};

}//namespace cobalt_715::nn::layer