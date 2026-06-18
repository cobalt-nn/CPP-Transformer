#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>
#include "SpecialToken.hpp"

namespace cobalt_715::nn{

struct Vocabulary{
  Vocabulary(){
    add(token::stokens);
  }

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

  void add(const std::vector<std::string> &tokens){
    for(const std::string &s:tokens){
      if(stoi_.contains(s)){
        continue;
      }
      stoi_[s] = static_cast<int64_t>(itos_.size());
      itos_.push_back(s);
    }
  }

  std::vector<std::string> itos(const std::vector<int64_t> &ids) const{
    std::vector<std::string> tokens;
    tokens.reserve(ids.size());

    for(int64_t id:ids){
      tokens.push_back(itos_.at(id));
    }

    return tokens;
  }

  std::vector<int64_t> stoi(const std::vector<std::string> &tokens) const{
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

private:
  std::vector<std::string> itos_;
  std::unordered_map<std::string,int64_t> stoi_;
};

}//namespace cobalt_715::nn