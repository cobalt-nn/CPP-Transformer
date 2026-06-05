#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cstdint>
#include <unordered_map>
#include "nn/tensor/Tensor.hpp"

using namespace cobalt_715::nn::tensor;

struct TokenSystem{
  TokenSystem(int64_t in)
    : in_(in),
      voc_({static_cast<int64_t>(itos_.size()),in}){

    for(int64_t i = 0;i < itos_.size();i++){
      stoi_[itos_[i]] = i;
    }

    /*for(int64_t i = 0;i < voc_.numel();i++){
      voc_.data()[i] = i;
    }

    std::cout << voc_.to_string() << std::endl;*/
  }

  Tensor forward(const std::string &str){
    Tensor output({1,static_cast<int64_t>(str.size()),in_});

    place_ = std::vector<int64_t>(str.size());

    for(int64_t i = 0;i < str.size();i++){
      place_[i] = stoi_.at(str[i]);

      for(int64_t j = 0;j < in_;j++){
        output.at({0,i,j}) = voc_.at({place_[i],j});
      }
    }

    /*for(int i = 0;i < output.dim(1);i++){
      for(int j = 0;j < output.dim(2);j++){
        output.at({0,i,j}) += 0.05f * (i - output.dim(1) / 2);
      }
    }*/

    return output;
  }

  void step(const Tensor &grad,const float lr){
    if(place_.size() != grad.dim(1) || in_ != grad.dim(2)) throw std::runtime_error("qawsedrftgyhujikolp");

    for(int64_t i = 0;i < grad.dim(1);i++){
      for(int64_t j = 0;j < grad.dim(2);j++){
        voc_.at({place_[i],j}) -= grad.at({0,i,j}) * lr;
      }
    }
  }

  int64_t char_to_index(const char c){
    return stoi_.at(c);
  }

  char index_to_char(const int64_t i){
    return itos_[i];
  }

  void random_init(std::mt19937 &gen){
    float limit = sqrt(6.0f / (voc_.shape()[0] + voc_.shape()[1]));
    std::uniform_real_distribution<float> dist(-limit,limit);

    float *Wd = voc_.data();

    for(int64_t i = 0;i < voc_.numel();i++){
      Wd[i] = dist(gen);
    }
  }

  size_t to_size() const{
    return itos_.size();
  }

private:
  const int64_t in_;
  const std::string itos_ = "0123456789+-=e";
  std::unordered_map<char,int64_t> stoi_;
  Tensor voc_;

  std::vector<int64_t> place_;
};