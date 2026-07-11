#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <random>
#include <unordered_map>
#include "SpecialToken.hpp"
#include "Tokens.hpp"
#include "EnglishTokenizer.hpp"
#include "nn/tensor/Tensor.hpp"
#include "nlohmann/json.hpp"

namespace cobalt_715::nn::language{

struct Vocabulary{
  Vocabulary(){
    add(EnglishTokenizer::symbol_);
    add(EnglishTokenizer::prefix_);
    add(EnglishTokenizer::suffix_);

    std::string ch = "aa";

    for(char &i = ch[0];i <= 'z';i++){
      ch[1] = 'a';
      for(char &j = ch[1];j <= 'z';j++){
        if(!stoi_.contains(ch)){
          add(std::vector<std::string>{ch});
        }
      }
    }
  }

  //語彙数
  int64_t size() const{
    return static_cast<int64_t>(itos_.size());
  }

  std::string to_string() const{
    std::string s;

    for(const std::string &str:itos_){
      s += str + "\n";
    }

    return s;
  }

  void add(const Tokens &tokens){
    for(const std::string &s:tokens.v_){
      if(stoi_.contains(s)){
        continue;
      }
      stoi_[s] = static_cast<int64_t>(itos_.size());
      itos_.push_back(s);
    }
  }

  //id[]をTokensに変換する
  Tokens itos(const std::vector<int64_t> &ids) const{
    std::vector<std::string> tokens;
    tokens.reserve(ids.size());

    for(int64_t id:ids){
      tokens.push_back(itos_.at(id));
    }

    return Tokens(tokens);
  }

  //Tokensからid[]に変換する
  std::vector<int64_t> stoi(const Tokens &ts) const{
    const std::vector<std::string> &tokens = ts.v_;

    std::vector<int64_t> ids;
    ids.reserve(tokens.size());

    for(const std::string &s:tokens){
      const auto it = stoi_.find(s);
      if(it == stoi_.end()){
        ids.push_back(stoi_.at(token::UNK));
      }else{
        ids.push_back(it->second);
      }
    }

    return ids;
  }

  std::vector<std::vector<int64_t>> argmax(const tensor::Tensor &t){
    if(t.rank() != 3) throw std::runtime_error("Vocabulary::argmax");

    std::vector<std::vector<int64_t>> idss;

    for(int64_t batch = 0;batch < t.dim(0);batch++){
      std::vector<int64_t> ids;
      for(int64_t row = 0;row < t.dim(1);row++){
        const auto max = std::max_element(&t.data()[batch * t.stride()[0] + row * t.stride()[1]],&t.data()[batch * t.stride()[0] + (row + 1) * t.stride()[1]]);
        ids.push_back(max - &t.data()[batch * t.stride()[0] + row * t.stride()[1]]);
      }
      idss.push_back(ids);
    }

    return idss;
  }

  std::vector<std::vector<int64_t>> sample(const tensor::Tensor &t,std::mt19937 &gen){
    if(t.rank() != 3) throw std::runtime_error("Vocabulary::sample");

    std::uniform_real_distribution<double> dist(0.0,1.0);

    std::vector<std::vector<int64_t>> idss;

    for(int64_t batch = 0;batch < t.dim(0);batch++){
      std::vector<int64_t> ids;
      for(int64_t row = 0;row < t.dim(1);row++){
        double r = dist(gen);
        double sum = 0.0;
        int64_t selected = t.dim(2) - 1;
        for(int64_t col = 0;col < t.dim(2);col++){
          sum += t.at({batch,row,col});
          if(sum >= r){
            selected = col;
            break;
          }
        }
        ids.push_back(selected);
      }
      idss.push_back(ids);
    }

    return idss;
  }

  //語彙をjsonにする
  nlohmann::ordered_json to_json() const{
    nlohmann::ordered_json j;

    /*for(int32_t k = 0;k < 200000;k++){
      size_t en = itos_.size() - 1;

      std::string ch = "aa";

      for(char &i = ch[0];i <= 'z';i++){
        ch[1] = 'a';
        for(char &j = ch[1];j <= 'z';j++){
          if(!stoi_.contains(ch)){
            itos_[en] = ch;
            en--;
          }
        }
      }
    }*/

    j["itos"] = itos_;

    return j;
  }

  void load_json(const nlohmann::ordered_json j){
    itos_.clear();
    stoi_.clear();

    add(j["itos"].get<std::vector<std::string>>());
  }

  std::vector<std::string> itos_;
  std::unordered_map<std::string,int64_t> stoi_;
};

}//namespace cobalt_715::nn::language