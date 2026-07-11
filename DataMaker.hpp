#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <random>
#include "nn/language/Language.hpp"
#include "nn/language/Tokens.hpp"

#include "nlohmann/json.hpp"

using namespace cobalt_715::nn;

struct DataMaker{
  DataMaker(const int64_t max_len,
            const language::Language &lang,
            const std::string wikitext = "C:\\Users\\hiros\\Desktop\\Python\\wikitext-103.jsonl",
            const std::string dolly = "C:\\Users\\hiros\\Desktop\\Python\\dolly-15k.jsonl")
    : max_len_(max_len),
      lang_(lang),
      wikitext_(wikitext),
      dolly_(dolly){}

  language::Tokens wikitext(){
    while(wiki_arr_.size() < max_len_){
      std::string text;

      if(!std::getline(wikitext_,text)){
        wikitext_.clear();
        wikitext_.seekg(0);
        continue;
      } 

      nlohmann::ordered_json j = nlohmann::ordered_json::parse(text);

      text = language::token::BOS + j["text"].get<std::string>() + language::token::EOS;

      for(const std::string &s:lang_.tokenize(text).v_){
        wiki_arr_.push_back(s);
      }
    }

    std::vector<std::string> re;

    for(size_t i = 0;i < max_len_;i++){
      re.push_back(wiki_arr_.at(i));
    }

    wiki_arr_.erase(wiki_arr_.begin(),wiki_arr_.begin() + max_len_ - 16);//わずかに重ねる

    return language::Tokens(re);
  }

  language::Tokens dolly(){
    while(dolly_arr_.size() < max_len_){
      std::string text;

      if(!std::getline(dolly_,text)){
        dolly_.clear();
        dolly_.seekg(0);
        continue;
      } 

      nlohmann::ordered_json j = nlohmann::ordered_json::parse(text);

      text = language::token::BOS +
             language::token::USER +
             j["instruction"].get<std::string>() +
             j["context"].get<std::string>() +
             language::token::ASSISTANT +
             j["response"].get<std::string>() +
             language::token::EOS;

      for(const std::string &s:lang_.tokenize(text).v_){
        dolly_arr_.push_back(s);
      }
    }

    std::vector<std::string> re;

    for(size_t i = 0;i < max_len_;i++){
      re.push_back(dolly_arr_.at(i));
    }

    dolly_arr_.erase(dolly_arr_.begin(),dolly_arr_.begin() + max_len_ - 16);//わずかに重ねる

    return language::Tokens(re);
  }

private:
  const int64_t max_len_;
  const language::Language &lang_;
  std::ifstream wikitext_;
  std::ifstream dolly_;
  std::vector<std::string> wiki_arr_;
  std::vector<std::string> dolly_arr_;
};