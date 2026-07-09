#pragma once

#include <iostream>
#include <string>
#include <random>
#include <cstdint>
#include "ILayer.hpp"
#include "Dense.hpp"
#include "Linear.hpp"
#include "nn/ops/Activation.hpp"
#include "nlohmann/json.hpp"
#include "nn/tensor/Tensor.hpp"
#include "nn/io/BinaryIO.hpp"

namespace cobalt_715::nn::layer{

struct FFN : ILayer{
  FFN(int64_t in,const ops::Activation &act = ops::activations::LeakyReLU)
    : in_size_(in),
      out1_size_(in * 4),
      out2_size_(in),
      dense_(in,in * 4,act),
      linear_(in * 4,in,true){}

  FFN(int64_t in,int64_t out1,int64_t out2,const ops::Activation &act = ops::activations::LeakyReLU)
    : in_size_(in),
      out1_size_(out1),
      out2_size_(out2),
      dense_(in,out1,act),
      linear_(out1,out2,true){}

  const int64_t in_size_;
  const int64_t out1_size_;
  const int64_t out2_size_;

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
    nlohmann::ordered_json j;

    j["layer_type"] = get_type();
    j["in"] = in_size_;
    j["out1"] = out1_size_;
    j["out2"] = out2_size_;
    j["dense"] = dense_.to_json();
    j["linear"] = linear_.to_json();

    return j;
  }

  void save(std::ostream &os) const override{
    dense_.save(os);
    linear_.save(os);
  }

  void load(const nlohmann::ordered_json &json,std::istream &is) override{
    if(json.at("layer_type") != get_type()){
      throw std::runtime_error("FFN::load type mismatch");
    }

    dense_.load(json.at("dense"),is);
    linear_.load(json.at("linear"),is);
  }

  void random_init(std::mt19937 &gen) override{
    dense_.random_init(gen);
    linear_.random_init(gen);
  }
};

}//namespace cobalt_715::nn::layer