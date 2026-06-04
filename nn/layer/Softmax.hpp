#pragma once

#include <string>
#include <random>
#include "ILayer.hpp"
#include "nlohmann/json.hpp"
#include "nn/tensor/Tensor.hpp"

namespace cobalt_715::nn::layer{

struct Softmax : ILayer{
  tensor::Tensor output_ = tensor::Tensor({1});

  const tensor::Tensor& forward(const tensor::Tensor& input,bool training=true) override{
    if(output_.shape() != input.shape()) output_ = tensor::Tensor(input.shape());

    const int64_t cols = input.dim(input.rank() - 1);
    const int64_t big_rows = input.numel() / cols;

    for(int64_t row = 0;row < big_rows;row++){
      //最大値を求める
      const float max = *std::max_element(&input.data()[row * cols],&input.data()[(row + 1) * cols]);

      //合計値を求める
      double sum = 0;
      for(int64_t col = 0;col < cols;col++){
        sum += std::exp(input.data()[row * cols + col] - max);
      }

      for(int64_t col = 0;col < cols;col++){
        output_.data()[row * cols + col] = static_cast<float>(std::exp(input.data()[row * cols + col] - max) / sum);
      }
    }

    return output_;
  }

  //とりあえずこの層が最終総 && 損失関数をCross Entropy想定なのでそのまま返す
  const tensor::Tensor& backward(const tensor::Tensor& grad_output) override{
    return grad_output;
  }

  void step(float lr,int batch_size=64) override{
  }

  void zero_grad() override{
  }

  //層の種類を返す。適切にオーバーライドすること
  virtual std::string get_type() const override{
    return "Softmax";
  }

  std::string to_string() const{
    return get_type() + "::to_string() is undef";
  }

  nlohmann::ordered_json to_json() const override{
    return nlohmann::ordered_json();
  }

  void random_init(std::mt19937 &gen) override{
  }
};

}//namespace cobalt_715::nn::layer