#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include <ctime>
#include <cmath>
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

#include "nn/Model.hpp"
#include "nn/EnglishTokenizer.hpp"
#include "nn/Vocabulary.hpp"
#include "nn/Embedding.hpp"
#include "nn/SpecialToken.hpp"

#include "data/MNISTLoader.hpp"

using namespace cobalt_715::nn;

int main(){
  const std::string case1 = "What food do you like?" + token::ASSISTANT;
  const std::string case1_1 = "I like apple.";
  const std::string case1_2 = "I like pineapple.";
  const std::string case1_3 = "I like banana.";
  const std::string case1_4 = "I like orange.";

  const std::string case01_1 = "apple is food.";
  const std::string case01_2 = "pineapple is food.";
  const std::string case01_3 = "banana is food.";
  const std::string case01_4 = "orange is food.";

  const std::string case2 = "Its food.";
  const std::string case2_1 = "What is apple?" + token::ASSISTANT;
  const std::string case2_2 = "What is pineapple?" + token::ASSISTANT;
  const std::string case2_3 = "What is banana?" + token::ASSISTANT;
  const std::string case2_4 = "What is orange?" + token::ASSISTANT;

  const std::string case3 = "What language do you like?" + token::ASSISTANT;
  const std::string case3_1 = "I like Java.";
  const std::string case3_2 = "I like Rust.";
  const std::string case3_3 = "I like c.";
  const std::string case3_4 = "I like c++.";

  const std::string case03_1 = "Java is language.";
  const std::string case03_2 = "Rust is language.";
  const std::string case03_3 = "c is language.";
  const std::string case03_4 = "c++ is language.";

  const std::string case4 = "Its language.";
  const std::string case4_1 = "What is Java?" + token::ASSISTANT;
  const std::string case4_2 = "What is Rust?" + token::ASSISTANT;
  const std::string case4_3 = "What is c?" + token::ASSISTANT;
  const std::string case4_4 = "What is c++?" + token::ASSISTANT;

  const std::string case5 = "What animal do you like?" + token::ASSISTANT;
  const std::string case5_1 = "I like dogs.";
  const std::string case5_2 = "I like cats.";
  const std::string case5_3 = "I like birds.";
  const std::string case5_4 = "I like mice.";

  const std::string case05_1 = "dogs is animal.";
  const std::string case05_2 = "cats is animal.";
  const std::string case05_3 = "birds is animal.";
  const std::string case05_4 = "mice is animal.";

  EnglishTokenizer et;

  /*for(const std::string &s:et.format(case03_4,32)){
    std::cout << s << std::endl;
  }

  std::cout << et.detokenize(et.format(case03_4,32)) << std::endl;

  return 0;*/

  Vocabulary voc;

  voc.add(et.tokenize(case1 + case1_1 + case1_1 + case1_3 + case1_4));
  voc.add(et.tokenize(case2 + case2_1 + case2_1 + case2_3 + case2_4));
  voc.add(et.tokenize(case3 + case3_1 + case3_1 + case3_3 + case3_4));
  voc.add(et.tokenize(case4 + case4_1 + case4_1 + case4_3 + case4_4));
  voc.add(et.tokenize(case5 + case5_1 + case5_1 + case5_3 + case5_4));

  voc.add(et.tokenize(case1_1 + case1_1 + case1_3 + case1_4));
  voc.add(et.tokenize(case3_1 + case3_1 + case3_3 + case3_4));
  voc.add(et.tokenize(case5_1 + case5_1 + case5_3 + case5_4));

  //std::cout << voc.to_string() << std::endl;

  std::vector<std::vector<int64_t>> ids = {
    {voc.stoi(et.format(case1 + case1_1,32))},
    {voc.stoi(et.format(case1 + case1_1,32))},
    {voc.stoi(et.format(case1 + case1_1,32))},
    {voc.stoi(et.format(case1 + case1_1,32))}
  };

  std::vector<std::vector<std::vector<int64_t>>> idss = {
    {
      {voc.stoi(et.format(case1 + case1_1,32))},
      {voc.stoi(et.format(case1 + case1_2,32))},
      {voc.stoi(et.format(case1 + case1_3,32))},
      {voc.stoi(et.format(case1 + case1_4,32))}
    },
    {
      {voc.stoi(et.format(case2_1 + case2,32))},
      {voc.stoi(et.format(case2_2 + case2,32))},
      {voc.stoi(et.format(case2_3 + case2,32))},
      {voc.stoi(et.format(case2_4 + case2,32))}
    },
    {
      {voc.stoi(et.format(case3 + case3_1,32))},
      {voc.stoi(et.format(case3 + case3_2,32))},
      {voc.stoi(et.format(case3 + case3_3,32))},
      {voc.stoi(et.format(case3 + case3_4,32))}
    },
    {
      {voc.stoi(et.format(case4_1 + case4,32))},
      {voc.stoi(et.format(case4_2 + case4,32))},
      {voc.stoi(et.format(case4_3 + case4,32))},
      {voc.stoi(et.format(case4_4 + case4,32))}
    },
    {
      {voc.stoi(et.format(case5 + case5_1,32))},
      {voc.stoi(et.format(case5 + case5_2,32))},
      {voc.stoi(et.format(case5 + case5_3,32))},
      {voc.stoi(et.format(case5 + case5_4,32))}
    },
    {
      {voc.stoi(et.format(case01_1,10))},
      {voc.stoi(et.format(case01_2,10))},
      {voc.stoi(et.format(case01_3,10))},
      {voc.stoi(et.format(case01_4,10))}
    },
    {
      {voc.stoi(et.format(case03_1,10))},
      {voc.stoi(et.format(case03_2,10))},
      {voc.stoi(et.format(case03_3,10))},
      {voc.stoi(et.format(case03_4,10))}
    },
    {
      {voc.stoi(et.format(case05_1,10))},
      {voc.stoi(et.format(case05_2,10))},
      {voc.stoi(et.format(case05_3,10))},
      {voc.stoi(et.format(case05_4,10))}
    }
  };

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

  std::mt19937 gen(0);

  em.random_init(gen);
  m.random_init(gen);

  const float lr = 0.001f;

  for(int64_t i = 0;i < 2000;i++){
    const size_t ba = gen() % idss.size();

    tensor::Tensor target({static_cast<int64_t>(idss.at(ba).size()),static_cast<int64_t>(idss.at(ba).at(0).size()),voc.size()});

    for(int64_t i = 0;i < target.dim(0);i++){
      for(int64_t j = 0;j < target.dim(1);j++){
        if(j + 1 == target.dim(1)){
          target.at({i,j,0}) = 1.0f;
        }else{
          target.at({i,j,idss.at(ba).at(i).at(j + 1)}) = 1.0f;
        }
      }
    }

    const tensor::Tensor &em_out = em.forward(idss.at(ba));
    const tensor::Tensor &m_out = m.forward(em_out);

    em.backward(m.backward(m_out - target));

    std::cout << "time:" << i << " ----------------------------------------" << std::endl;

    float loss = 0.0f;
    float conf = 0.0f;

    for(int64_t batch = 0;batch < m_out.dim(0);batch++){
      for(int64_t row = 0;row < m_out.dim(1);row++){
        auto max = std::max_element(&m_out.data()[batch * m_out.stride()[0] + row * m_out.stride()[1]],&m_out.data()[batch * m_out.stride()[0] + (row + 1) * m_out.stride()[1]]);
        loss -= log(m_out.at({batch,row,(row + 1 < idss.at(ba).at(0).size()) ? idss.at(ba).at(batch).at(row + 1):0}));
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

  std::uniform_real_distribution<float> dist(0.0f,1.0f);

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