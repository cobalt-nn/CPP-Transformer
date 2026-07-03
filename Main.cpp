#include <iostream>
#include <memory>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>
#include <chrono>
#include <ctime>
#include <cstdint>
#include <cmath>
#include <cstddef>
#include "nn/ops/vec.hpp"
#include "nn/ops/vec_cpu.hpp"
#include "nn/tensor/Tensor.hpp"
#include "nn/tensor/TensorView.hpp"
#include "nn/tensor/MatrixView.hpp"
#include "nn/tensor/ConstMatrixView.hpp"

#include "nn/layer/ILayer.hpp"
#include "nn/layer/Dense.hpp"
#include "nn/layer/Linear.hpp"
#include "nn/layer/RMSNorm.hpp"
#include "nn/layer/Attention.hpp"
#include "nn/layer/Softmax.hpp"
#include "nn/layer/ReZero.hpp"
#include "nn/layer/FFN.hpp"
#include "nn/ops/Activation.hpp"
#include "nn/layer/Identity.hpp"
#include "nn/ops/Acts.hpp"
#include "nn/ops/GEMM.hpp"

#include "nn/io/BinaryIO.hpp"

#include "nn/Model.hpp"
#include "nn/EnglishTokenizer.hpp"
#include "nn/Vocabulary.hpp"
#include "nn/Embedding.hpp"
#include "nn/SpecialToken.hpp"

#include "data/MNISTLoader.hpp"

#include "nlohmann/json.hpp"

using namespace cobalt_715::nn;

  static const std::vector<std::string> q = {
    "What food do you ",
    "What language do you ",
    "What animal do you ",
    "What NN do you "
  };

  static const std::vector<std::string> llh = {
    "like",
    "love",
    "have"
  };

  static const std::vector<std::string> food = {
    "apple",
    "banana",
    "orange",
    "pineapple"
  };

  static const std::vector<std::string> lang = {
    "Java",
    "Rust",
    "c",
    "c++"
  };

  static const std::vector<std::string> animal = {
    "dogs",
    "cats",
    "mice",
    "birds"
  };

  static const std::vector<std::string> NN = {
    "Dense",
    "CNN",
    "RNN",
    "Attention"
  };

std::string make_str4(std::mt19937 &gen){
  size_t ty1 = gen() % 4;
  size_t ty2 = gen() % 4;
  size_t ty3 = gen() % 4;

  std::string s;

  if(ty1 == 0){
    s = food.at(ty2) + " and " + food.at(ty3);
  }else if(ty1 == 1){
    s = lang.at(ty2) + " and " + lang.at(ty3);
  }else if(ty1 == 2){
    s = animal.at(ty2) + " and " + animal.at(ty3);
  }else if(ty1 == 3){
    s = NN.at(ty2) + " and " + NN.at(ty3);
  }

  return s + " " +  token::ASSISTANT + " " + s;
}

std::string make_str3(std::mt19937 &gen){
  std::string s;

  size_t ty1 = gen() % 4;
  size_t ty2 = gen() % 4;
  size_t ty3 = gen() % 4;
  size_t ty4 = gen() % 4;

  if(ty1 == 0){
    s += food.at(ty2) + " is food and ";
  }else if(ty1 == 1){
    s += lang.at(ty2) + " is language and ";
  }else if(ty1 == 2){
    s += animal.at(ty2) + " is animal and ";
  }else if(ty1 == 3){
    s += NN.at(ty2) + " is NN and ";
  }

  if(ty3 == 0){
    s += food.at(ty4) + " is food.";
  }else if(ty3 == 1){
    s += lang.at(ty4) + " is language.";
  }else if(ty3 == 2){
    s += animal.at(ty4) + " is animal.";
  }else if(ty3 == 3){
    s += NN.at(ty4) + " is NN.";
  }

  return s;
}

std::string make_str2(std::mt19937 &gen){
  std::string s = "What is ";

  size_t ty1 = gen() % 4;

  size_t ty2 = gen() % 4;

  std::string value;

  if(ty1 == 0){
    ty2 = gen() % 3;
    value = food.at(ty2);
    s += value;
  }else if(ty1 == 1){
    value = lang.at(ty2);
    s += value;
  }else if(ty1 == 2){
    value = animal.at(ty2);
    s += value;
  }else if(ty1 == 3){
    value = NN.at(ty2);
    s += value;
  }

  value[0] = std::toupper(value[0]);

  s += "?" + token::ASSISTANT + value + " is ";

  if(ty1 == 0){
    s += "food";
  }else if(ty1 == 1){
    s += "language";
  }else if(ty1 == 2){
    s += "animal";
  }else if(ty1 == 3){
    s += "NN";
  }

  s += ".";

  return s;
}

std::string make_str1(std::mt19937 &gen){
  std::string s;

  size_t ty1 = 0;

  size_t ty2 = 0;

  size_t ty3 = 0;

  do{
    ty1 = gen() % 4;

    ty2 = gen() % 3;

    ty3 = gen() % 4;
  }while(!(ty1 != 3 || ty2 != 1));

  s += q.at(ty1) + llh.at(ty2) + "?" + token::ASSISTANT;
  s += "I " + llh.at(ty2) + " ";
  if(ty1 == 0){
    s += food.at(ty3);
  }else if(ty1 == 1){
    s += lang.at(ty3);
  }else if(ty1 == 2){
    s += animal.at(ty3);
  }else if(ty1 == 3){
    s += NN.at(ty3);
  }
  if(llh.at(ty2) == "love"){
    s += " the best";
  }
  s += ".";

  return s;
}

std::string make_str(std::mt19937 &gen){
  size_t ty = gen() % 4;

  if(ty == 0){
    return make_str1(gen);
  }else if(ty == 1){
    return make_str2(gen);
  }else if(ty == 2){
    return make_str3(gen);
  }

  return make_str4(gen);
}

int main(){
  std::mt19937 gen(0);

  for(int i = 0;i < 0;i++){
    std::cout << make_str(gen) << std::endl;
  }

  //return 0;

  EnglishTokenizer et;

  /*for(const std::string &s:et.format(case1 + case1_1,32)){
    std::cout << s << std::endl;
  }

  std::cout << et.detokenize(et.format(case1 + case1_1,32)) << std::endl;

  return 0;*/

  Vocabulary voc;
  voc.add(et.tokenize("attention is all you need like love have What do you like? Its the best and food language animal NN"));
  voc.add(et.tokenize(make_str1(gen)));
  voc.add(et.tokenize(make_str2(gen)));
  voc.add(et.tokenize(make_str3(gen)));

  voc.add(et.tokenize(food.at(0)));
  voc.add(et.tokenize(food.at(1)));
  voc.add(et.tokenize(food.at(2)));
  voc.add(et.tokenize(food.at(3)));

  voc.add(et.tokenize(lang.at(0)));
  voc.add(et.tokenize(lang.at(1)));
  voc.add(et.tokenize(lang.at(2)));
  voc.add(et.tokenize(lang.at(3)));

  voc.add(et.tokenize(animal.at(0)));
  voc.add(et.tokenize(animal.at(1)));
  voc.add(et.tokenize(animal.at(2)));
  voc.add(et.tokenize(animal.at(3)));

  voc.add(et.tokenize(NN.at(0)));
  voc.add(et.tokenize(NN.at(1)));
  voc.add(et.tokenize(NN.at(2)));
  voc.add(et.tokenize(NN.at(3)));

  //std::cout << voc.to_string() << std::endl;

  std::vector<std::vector<int64_t>> ids;

  //std::cout << target.to_string() << std::endl;

  Embedding em(voc.size(),32);

  const int64_t cache_len = 64;

  Model m;

  m.add<layer::RMSNorm>(32)
   .add<layer::ReZero>(32,128,std::make_unique<layer::Attention>(32,4,32,32,cache_len,true))
   .add<layer::RMSNorm>(32)
   .add<layer::ReZero>(std::make_unique<layer::FFN>(32))
   .add<layer::RMSNorm>(32)
   .add<layer::ReZero>(32,128,std::make_unique<layer::Attention>(32,4,32,32,cache_len,true))
   .add<layer::RMSNorm>(32)
   .add<layer::ReZero>(std::make_unique<layer::FFN>(32))
   .add<layer::RMSNorm>(32)
   .add<layer::ReZero>(32,128,std::make_unique<layer::Attention>(32,4,32,32,cache_len,true))
   .add<layer::RMSNorm>(32)
   .add<layer::FFN>(32,32*4,voc.size())
   .add<layer::Softmax>();

  std::ifstream ifs("nn/models/embmodel.bin",std::ios::binary);

  //em.load(ifs);

  //m.load_all("nn/models/model.json","nn/models/model.bin");

  const float lr = 0.001f;

  em.random_init(gen);
  m.random_init(gen);

  auto now1 = std::chrono::steady_clock::now();

  for(int64_t i = 0;i < 100;i++){
    std::cout << "time:" << i << " ----------------------------------------" << std::endl;

    ids = {
      {voc.stoi(et.format(make_str(gen),32))},
      {voc.stoi(et.format(make_str(gen),32))},
      {voc.stoi(et.format(make_str(gen),32))},
      {voc.stoi(et.format(make_str(gen),32))}
    };

    tensor::Tensor target({static_cast<int64_t>(ids.size()),static_cast<int64_t>(ids.at(0).size()),voc.size()});

    for(int64_t i = 0;i < target.dim(0);i++){
      for(int64_t j = 0;j < target.dim(1);j++){
        if(j + 1 == target.dim(1)){
          target.at({i,j,0}) = 1.0f;
        }else{
          target.at({i,j,ids.at(i).at(j + 1)}) = 1.0f;
        }
      }
    }

    const tensor::Tensor &em_out = em.forward(ids);
    const tensor::Tensor &m_out = m.forward(em_out);

    em.backward(m.backward(m_out - target));

    float loss = 0.0f;
    float conf = 0.0f;

    for(int64_t batch = 0;batch < m_out.dim(0);batch++){
      for(int64_t row = 0;row < m_out.dim(1);row++){
        auto max = std::max_element(&m_out.data()[batch * m_out.stride()[0] + row * m_out.stride()[1]],&m_out.data()[batch * m_out.stride()[0] + (row + 1) * m_out.stride()[1]]);
        loss -= log(m_out.at({batch,row,(row + 1 < ids.at(0).size()) ? ids.at(batch).at(row + 1):0}));
        conf += *max;
      }
    }

    for(const auto &ve:voc.argmax(m_out)){
      std::cout << et.detokenize(voc.itos(ve)) << std::endl;
    }

    std::cout << "loss:" << loss / m_out.dim(0) / m_out.dim(1) << std::endl;
    std::cout << "confidence:" << conf / m_out.dim(0) / m_out.dim(1) << std::endl;

    em.step(lr,4);
    m.step(lr,4);

    em.zero_grad();
    m.zero_grad();
  }

  auto now2 = std::chrono::steady_clock::now();

  std::cout << std::chrono::duration<double,std::milli>(now2 - now1).count() << "ms" << std::endl;

  //std::cout << std::filesystem::current_path() << std::endl;

  std::ofstream ofs("nn/models/embmodel.bin",std::ios::binary);

  em.save(ofs);

  m.save_all("nn/models/model.json","nn/models/model.bin");

  while(true){
    std::cout << "--------------------------------------------------" << std::endl;

    std::string text;

    std::getline(std::cin,text);

    std::vector<std::string> tokens = et.format(text,text.size() * 2 + 10,false,false);

    tokens.pop_back();
    tokens.push_back(token::ASSISTANT);

    //std::cout << text << std::endl;

    //for(auto s:tokens) std::cout << "1234567890 " << s << std::endl;

    std::vector<std::vector<int64_t>> id = {voc.stoi(tokens)};

    const tensor::Tensor *out = &m.forward(em.forward(id,false),false);

    //std::cout << out->to_string() << std::endl;

    int64_t len = out->dim(1);

    std::vector<std::string> tok;

    while(len < cache_len){
      const int64_t id1 = voc.sample(*out,gen).at(0).at(out->dim(1) - 1);

      tok.push_back(voc.itos({id1}).at(0));

      if(voc.itos({id1}).at(0) == token::EOS) break;

      out = &m.forward(em.forward({{id1}},false),false);

      len++;
    }

    /*for(const std::string &str:tok){
      std::cout << str << std::endl;
    }*/

    std::cout << et.detokenize(tok) << std::endl;

    m.reset();
  }

  return 0;
}