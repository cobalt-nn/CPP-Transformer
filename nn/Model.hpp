#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include "nlohmann/json.hpp"
#include "tensor/Tensor.hpp"
#include "layer/ILayer.hpp"
#include "layer/Dense.hpp"
#include "layer/RMSNorm.hpp"
#include "layer/Attention.hpp"
#include "ops/Activation.hpp"
#include "ops/Acts.hpp"

namespace cobalt_715::nn{

struct Model : layer::ILayer{
private:
  std::vector<std::unique_ptr<layer::ILayer>> layers_;

public:
  Model& add(std::unique_ptr<ILayer> l){
    layers_.push_back(std::move(l));

    return *this;
  }

  template<typename T,typename... Args>
  Model& add(Args&&... args){
    layers_.push_back(std::make_unique<T>(std::forward<Args>(args)...));

    return *this;
  }

  //順伝播
  //前層の出力を受け取る
  //training=falseなら学習用にデータを保存しなくてもいい
  const tensor::Tensor& forward(const tensor::Tensor& input,bool training=true) override{
    const tensor::Tensor *out = &input;

    for(auto &l:layers_){
      out = &l->forward(*out,training);
    }

    return *out;
  }

  //逆伝播
  //次の層の勾配を受け取る
  const tensor::Tensor& backward(const tensor::Tensor& grad_output) override{
    const tensor::Tensor *out = &grad_output;

    for(int64_t i = layers_.size() - 1;i >= 0;i--){
      auto &l = layers_[i];

      out = &l->backward(*out);
    }

    return *out;
  }

  //更新
  //学習率、バッチサイズを受け取る
  void step(float lr,int batch_size=64) override{
    for(auto &l:layers_){
      l->step(lr,batch_size);
    }
  }

  //勾配をリセットする
  void zero_grad() override{
    for(auto &l:layers_){
      l->zero_grad();
    }
  }

  //何かリセットしたいとき呼び出す
  void reset() override{
    for(auto &l:layers_){
      l->reset();
    }
  }

  //層の種類を返す。適切にオーバーライドすること
  std::string get_type() const override{
    return "Model";
  }

  //文字列にしたいとき使う
  std::string to_string() const{
    std::string s;

    for(auto &l:layers_){
      s += l->to_string() + "\n";
    }

    return s;
  }

  //json形式で保存するとき使う
  nlohmann::ordered_json to_json() const override{
    return nlohmann::ordered_json();
  }

  //ランダム初期化する
  void random_init(std::mt19937 &gen) override{
    for(auto &l:layers_){
      l->random_init(gen);
    }
  }
};

}//namespace cobalt_715::nn