#pragma once

#include <iostream>
#include <vector>
#include <cstdint>
#include <random>
#include "nn/tensor/Tensor.hpp"
#include "nn/io/BinaryIO.hpp"
#include "nlohmann/json.hpp"

namespace cobalt_715::nn::language{

struct Embedding{
  Embedding(int64_t token_len,int64_t dim)
  : W_({token_len,dim}),
    dW_({token_len,dim}),
    out_({1,1,1}){

    /*for(int64_t i = 0;i < W_.numel();i++){
      W_.data()[i] = i;
    }

    std::cout << W_.to_string() << std::endl;*/
  }

  Embedding(const nlohmann::ordered_json &j) : Embedding(j["token_len"],j["dim"]){}

  const tensor::Tensor& forward(const std::vector<std::vector<int64_t>> ids,bool training=true){
    for(const std::vector<int64_t> &v:ids){
      if(ids.at(0).size() != v.size()) throw std::runtime_error("Embedding:forward");
    }

    ids_ = ids;

    if(out_.dim(0) != ids_.size() || out_.dim(1) != ids_.at(0).size() || out_.dim(2) != W_.dim(1)){
      out_ = tensor::Tensor({static_cast<int64_t>(ids_.size()),static_cast<int64_t>(ids_.at(0).size()),W_.dim(1)});
    }

    for(int64_t i = 0;i < out_.dim(0);i++){
      for(int64_t j = 0;j < out_.dim(1);j++){
        for(int64_t k = 0;k < out_.dim(2);k++){
          out_.at({i,j,k}) = W_.at({ids_.at(i).at(j),k});
        }
      }
    }

    return out_;
  }

  void backward(const tensor::Tensor& grad_output){
    for(int64_t i = 0;i < out_.dim(0);i++){
      for(int64_t j = 0;j < out_.dim(1);j++){
        for(int64_t k = 0;k < out_.dim(2);k++){
          dW_.at({ids_.at(i).at(j),k}) += grad_output.at({i,j,k});
        }
      }
    }
  }

  void step(float lr,int batch_size=64){
    dW_.scale_(lr);
    W_ -= dW_;
  }

  void zero_grad(){
    float *dWd = dW_.data();

    std::fill(dWd,dWd + dW_.numel(),0.0f);
  }

  void random_init(std::mt19937 &gen){
    std::uniform_real_distribution<float> dist(0,0.02f);

    float *Wd = W_.data();

    for(int64_t i = 0;i < W_.numel();i++){
      Wd[i] = dist(gen);
    }
  }

  nlohmann::ordered_json to_json() const{
    nlohmann::ordered_json j;

    j["token_len"] = W_.dim(0);
    j["dim"] = W_.dim(1);

    return j;
  }

  void save(std::ostream &os) const{
    io::save(os,W_.data(),W_.numel());
  }

  void load(std::istream &is){
    io::load(is,W_.data(),W_.numel());
  }

  tensor::Tensor W_;
  tensor::Tensor dW_;
  tensor::Tensor out_;

  std::vector<std::vector<int64_t>> ids_;
};

}//namespace cobalt_715::nn::language