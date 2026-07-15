#pragma once

#include <iostream>
#include <cstdint>
#include <vector>
#include <string>
#include <random>
#include "Language.hpp"
#include "nn/Model.hpp"

namespace cobalt_715::nn::language{

struct TextGenerator{
  TextGenerator(Language &lang,nn::Model &model)
    : lang_(lang),
      model_(model){}

  Language &lang_;
  nn::Model &model_;

  std::string gen(const std::string &s,std::mt19937 &r_gen){
    std::string input = token::BOS +
                        token::USER +
                        s +
                        token::ASSISTANT;

    const tensor::Tensor *out = &model_.forward(lang_.forward({lang_.tokenize(input)},false),false);

    Tokens out_tokens = lang_.sample(*out,r_gen).at(0);

    std::string out_str = out_tokens.v_.at(out_tokens.v_.size() - 1);

    std::vector<std::string> return_strs;

    int64_t count = 0;

    while(out_str != token::EOS && out_str != token::PAD){
      std::cout << lang_.detokenize(out_str);

      return_strs.push_back(out_str);

      out = &model_.forward(lang_.forward({lang_.tokenize(out_str)},false),false);

      out_str = lang_.sample(*out,r_gen).at(0).v_.at(0);

      count++;
    }

    std::cout << "<count:" << count << ">" << std::endl;

    return lang_.detokenize(return_strs);
  }

  //KV cacheクリア
  void reset(){
    model_.reset();
  }
};

}//namespace cobalt_715::nn::language