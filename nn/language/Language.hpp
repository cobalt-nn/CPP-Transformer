#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <optional>
#include <cstddef>
#include <random>
#include "EnglishTokenizer.hpp"
#include "Vocabulary.hpp"
#include "Embedding.hpp"
#include "SpecialToken.hpp"
#include "Tokens.hpp"
#include "nn/tensor/Tensor.hpp"
#include "nlohmann/json.hpp"

namespace cobalt_715::nn::language{

struct Language{
  EnglishTokenizer et;
  Vocabulary voc;
  std::optional<Embedding> emb = std::nullopt;

  //formatされたtokenize
  //トークン数を指定できる
  Tokens format(const std::string_view text,const int64_t max_len) const{
    return et.format(text,max_len,&voc.stoi_);
  }

  //stringをtokenに分解
  Tokens tokenize(const std::string_view text) const{
    return et.tokenize(text,&voc.stoi_);
  }

  //tokenizeした結果をstringに戻す
  std::string detokenize(const Tokens &tokens) const{
    return et.detokenize(tokens);
  }

  std::string detokenize(const std::string &text){
    return et.detokenize(text);
  }

  //id[]をTokensに変換する
  Tokens itos(const std::vector<int64_t> &ids) const{
    return voc.itos(ids);
  }

  //Tokensからid[]に変換する
  std::vector<int64_t> stoi(const Tokens &ts) const{
    return voc.stoi(ts);
  }

  Tokens make_target(Tokens tokens){
    tokens.v_.erase(tokens.v_.begin());

    tokens.v_.push_back(token::PAD);

    return tokens;
  }

  //仮　まじで
  tensor::Tensor make_grad(int i,tensor::Tensor output,const std::vector<std::string> &texts,float &loss,float &conf,const int64_t s = 0,const int64_t e = INT64_MAX){
    std::vector<Tokens> tokens;

    for(const std::string &s:texts){
      tokens.push_back(format(s,output.dim(1)));
    }

    return make_grad(output,tokens,loss,conf,s,e);
  }

  //仮　まじで
  tensor::Tensor make_grad(tensor::Tensor output,const std::vector<Tokens> &texts,float &loss,float &conf,const int64_t s = 0,const int64_t e = INT64_MAX){
    std::vector<std::vector<int64_t>> id;
    loss = 0.0f;
    conf = 0.0f;

    for(const Tokens &s:texts){
      id.push_back(stoi(make_target(s)));
    }

    const int64_t pad_id = stoi(std::vector<std::string>{token::PAD}).at(0);

    for(int64_t batch = 0;batch < output.dim(0);batch++){
      float los = 0.0f;
      float con = 0.0f;

      int64_t count = 0;

      for(int64_t row = 0;row < output.dim(1);row++){
        if(pad_id == id.at(batch).at(row)){
          float *ptr = &output.at({batch,row,0});

          std::fill(ptr,ptr + output.dim(2),0.0f);
        }else{
          if(s <= row && row < e){
            los -= std::log(output.at({batch,row,id.at(batch).at(row)}));
            con += output.at({batch,row,id.at(batch).at(row)});
            count++;
          }

          output.at({batch,row,id.at(batch).at(row)}) -= 1.0f;
        }
      }

      if(count != 0){
        loss += los / count;
        conf += con / count;
      }
    }

    loss /= output.dim(0);
    conf /= output.dim(0);

    return output;
  }

  //tokenize前のものを受け取る
  const tensor::Tensor& forward(const std::vector<std::string> &texts,const int64_t max_len,bool training=true){
    std::vector<std::vector<std::string>> ts;

    for(const std::string &s:texts){
      ts.push_back(et.format(s,max_len).v_);
    }

    std::vector<std::vector<int64_t>> ids;

    for(const std::vector<std::string> &s:ts){
      ids.push_back(voc.stoi(s));
    }

    if(!emb) throw std::runtime_error("Language::forward emb is nullopt");

    return emb.value().forward(ids,training);
  }

  //tokenize後のものを受け取る
  const tensor::Tensor& forward(const std::vector<Tokens> &texts,const int64_t max_len,bool training=true){
    if(!emb) throw std::runtime_error("Language::forward emb is nullopt");

    std::vector<std::vector<int64_t>> ids;

    const size_t size = texts.at(0).v_.size();

    for(const Tokens &tokens:texts){
      if(tokens.v_.size() != size) throw std::runtime_error("Language::backward");

      ids.push_back(voc.stoi(tokens.v_));
    }

    return emb.value().forward(ids,training);
  }

  void backward(const tensor::Tensor& grad_output){
    if(!emb) throw std::runtime_error("Language::backward emb is nullopt");

    emb.value().backward(grad_output);
  }

  void step(float lr,int batch_size=64){
    if(!emb) throw std::runtime_error("Language::step emb is nullopt");

    emb.value().step(lr,batch_size);
  }

  void zero_grad(){
    if(!emb) throw std::runtime_error("Language::zero_grad emb is nullopt");

    emb.value().zero_grad();
  }

  //トークン登録
  void add(const std::string_view s){
    voc.add(et.tokenize(s));
  }

  //Embedding初期化。トークン数が分からないため
  void build(const int64_t token_len,const int64_t dim){
    emb = Embedding(token_len,dim);
  }

  //語彙数
  int64_t size() const{
    return voc.size();
  }

  std::vector<Tokens> argmax(const tensor::Tensor &t){
    std::vector<Tokens> ts;

    for(const std::vector<int64_t> v:voc.argmax(t)){
      ts.push_back(Tokens(itos(v)));
    }

    return ts;
  }

  std::vector<Tokens> sample(const tensor::Tensor &t,std::mt19937 &gen){
    std::vector<Tokens> ts;

    for(const std::vector<int64_t> v:voc.sample(t,gen)){
      ts.push_back(Tokens(itos(v)));
    }

    return ts;
  }

  void random_init(std::mt19937 &gen){
    if(!emb) throw std::runtime_error("Language::random_init ");

    emb.value().random_init(gen);
  }

  nlohmann::ordered_json to_json() const{
    nlohmann::ordered_json j;

    if(!emb) throw std::runtime_error("Language::to_json ");

    j["Embedding"] = emb.value().to_json();
    j["Vocabulary"] = voc.to_json();

    return j;
  }

  void load_json(const nlohmann::ordered_json &j){
    emb = Embedding(j["Embedding"]);

    voc.load_json(j["Vocabulary"]);
  }

  void save_all(const std::string &path_json,const std::string &path_bin) const{
    {
      std::ofstream ofs_json(path_json);

      if(!ofs_json) throw std::runtime_error("Language::save_all json open failed");

      ofs_json << to_json().dump(2);

      if(!ofs_json) throw std::runtime_error("Language::save_all json write failed");

      ofs_json.close();
    }

    if(!emb) throw std::runtime_error("Language::save_all ");

    {
      std::ofstream ofs_bin(path_bin,std::ios::binary);

      if(!ofs_bin) throw std::runtime_error("Language::save_all bin open failed");

      emb.value().save(ofs_bin);

      if(!ofs_bin) throw std::runtime_error("Language::save_all bin write failed");

      ofs_bin.close();
    }
  }

  void load_all(const std::string &path_json,const std::string &path_bin){
    std::ifstream ifs_json(path_json);

    if(!ifs_json) throw std::runtime_error("Model::load_all json open failed");

    nlohmann::ordered_json json;
    ifs_json >> json;

    load_json(json);

    std::ifstream ifs_bin(path_bin,std::ios::binary);

    if(!ifs_bin) throw std::runtime_error("Model::load_all bin open failed");

    emb.value().load(ifs_bin);
  }
};

}//namespace cobalt_715::nn::language