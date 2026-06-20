#pragma once

#include <iostream>
#include <fstream>
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
  Model& add(std::unique_ptr<layer::ILayer> l){
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
  std::string to_string() const override{
    std::string s;

    for(auto &l:layers_){
      s += l->to_string() + "\n";
    }

    return s;
  }

  //json
  nlohmann::ordered_json to_json() const override{
    nlohmann::ordered_json j;

    j["magic_number"] = "CBLM";
    j["version"] = 1;

    nlohmann::ordered_json j_array = nlohmann::ordered_json::array();

    for(auto &l:layers_){
      j_array.push_back(l->to_json());
    }

    j["layers"] = j_array;

    return j;
  }

  //セーブ
  void save(std::ostream &os) const override{
    for(auto &l:layers_){
      l->save(os);
    }
  }

  //ロード
  void load(const nlohmann::ordered_json &json,std::istream &is) override{
    if(json.at("magic_number") != "CBLM"){
      throw std::runtime_error("Model::load magic_number mismatch");
    }

    if(json.at("version").get<int>() != 1){
      throw std::runtime_error("Model::load: unsupported version");
    }

    for(int64_t i = 0;i < layers_.size();i++){
      layers_.at(i)->load(json.at("layers").at(i),is);
    }
  }

  void save_all(const std::string &path_json,const std::string &path_bin){
    {
      std::ofstream ofs_json(path_json);

      if(!ofs_json) throw std::runtime_error("Model::save_all json open failed");

      ofs_json << to_json().dump(2);

      if(!ofs_json) throw std::runtime_error("Model::save_all json write failed");

      ofs_json.close();
    }

    {
      std::ofstream ofs_bin(path_bin,std::ios::binary);

      if(!ofs_bin) throw std::runtime_error("Model::save_all bin open failed");

      save(ofs_bin);

      if(!ofs_bin) throw std::runtime_error("Model::save_all bin write failed");

      ofs_bin.close();
    }
  }

  void load_all(const std::string &path_json,const std::string &path_bin){
    std::ifstream ifs_json(path_json);

    if(!ifs_json) throw std::runtime_error("Model::load_all json open failed");

    nlohmann::ordered_json json;
    ifs_json >> json;

    std::ifstream ifs_bin(path_bin,std::ios::binary);

    if(!ifs_bin) throw std::runtime_error("Model::load_all bin open failed");

    load(json,ifs_bin);
  }

  //ランダム初期化する
  void random_init(std::mt19937 &gen) override{
    for(auto &l:layers_){
      l->random_init(gen);
    }
  }
};

}//namespace cobalt_715::nn