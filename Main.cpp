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
#include <algorithm>
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
#include "nn/language/EnglishTokenizer.hpp"
#include "nn/language/Vocabulary.hpp"
#include "nn/language/Embedding.hpp"
#include "nn/language/SpecialToken.hpp"
#include "nn/language/Language.hpp"
#include "nn/language/Tokens.hpp"

#include "data/MNISTLoader.hpp"

#include "nlohmann/json.hpp"

using namespace cobalt_715::nn;

std::string make_input(std::mt19937 &gen){
  uint64_t l = gen() % 100;
  uint64_t m = gen() % 90 + 10;
  uint64_t n = gen() % 90 + 10;

  std::string s = "<bos>" +
    //std::to_string(l) + "+" +
    std::to_string(m) + "+" +
    std::to_string(n) + "=" +
    std::to_string(l + m + n) +
    "<eos>";

  return s;


  /*uint64_t n = gen() % 100000000;

  std::string s = std::to_string(n);

  s.erase(0,1);

  std::string s2 = s;

  std::reverse(s2.begin(),s2.end());

  return s + "you" + s2 + language::token::EOS;*/


  /*std::string s;

  std::vector<int32_t> arr;

  for(int64_t i = 0;i < 7;i++){
    uint32_t u = gen() % 10;

    arr.push_back(u);

    s += std::to_string(u);
  }

  s += "you";

  for(int64_t i = 0;i < 7;i++){
    s += std::to_string(std::abs(arr.at(i) - arr.at((i + 1) % 7)));
  }

  s += language::token::EOS;

  return s;*/
}

int main(){
  std::mt19937 gen(0);

  for(int32_t i = 0;i < 100;i++){
    std::cout << make_input(gen) << std::endl;
  }

  language::Language lang;

  lang.add("1234567890 + - * / =Attention is all you need reverse");

  const int64_t dim = 32;
  const int64_t head_num = 4;
  const int64_t cache_len = 16;

  const ops::Activation &act = ops::activations::LeakyReLU;

  lang.build(lang.size(),dim);

  Model m;

  m.add<layer::RMSNorm>(dim)
   .add<layer::ReZero>(dim,dim * head_num,std::make_unique<layer::Attention>(dim,head_num,dim,dim,cache_len,true))
   .add<layer::RMSNorm>(dim)
   .add<layer::ReZero>(std::make_unique<layer::FFN>(dim,act))
   .add<layer::RMSNorm>(dim)
   .add<layer::ReZero>(dim,dim * head_num,std::make_unique<layer::Attention>(dim,head_num,dim,dim,cache_len,true))
   .add<layer::RMSNorm>(dim)
   .add<layer::ReZero>(std::make_unique<layer::FFN>(dim,act))
   .add<layer::RMSNorm>(dim)
   .add<layer::ReZero>(dim,dim * head_num,std::make_unique<layer::Attention>(dim,head_num,dim,dim,cache_len,true))
   .add<layer::RMSNorm>(dim)
   .add<layer::ReZero>(std::make_unique<layer::FFN>(dim,act))
   .add<layer::RMSNorm>(dim)
   .add<layer::ReZero>(dim,dim * head_num,std::make_unique<layer::Attention>(dim,head_num,dim,dim,cache_len,true))
   .add<layer::RMSNorm>(dim)
   .add<layer::ReZero>(std::make_unique<layer::FFN>(dim,act))
   .add<layer::RMSNorm>(dim)
   .add<layer::ReZero>(dim,dim * head_num,std::make_unique<layer::Attention>(dim,head_num,dim,dim,cache_len,true))
   .add<layer::RMSNorm>(dim)
   .add<layer::ReZero>(std::make_unique<layer::FFN>(dim,act))
   .add<layer::RMSNorm>(dim)
   .add<layer::ReZero>(dim,dim * head_num,std::make_unique<layer::Attention>(dim,head_num,dim,dim,cache_len,true))
   .add<layer::RMSNorm>(dim)
   .add<layer::ReZero>(std::make_unique<layer::FFN>(dim,act))
   .add<layer::RMSNorm>(dim)
   .add<layer::ReZero>(dim,dim * head_num,std::make_unique<layer::Attention>(dim,head_num,dim,dim,cache_len,true))
   .add<layer::RMSNorm>(dim)
   .add<layer::ReZero>(std::make_unique<layer::FFN>(dim,act))
   .add<layer::RMSNorm>(dim)
   .add<layer::ReZero>(dim,dim * head_num,std::make_unique<layer::Attention>(dim,head_num,dim,dim,cache_len,true))
   .add<layer::RMSNorm>(dim)
   .add<layer::ReZero>(std::make_unique<layer::FFN>(dim,act))
   .add<layer::Linear>(dim,lang.size())
   .add<layer::Softmax>();

  lang.random_init(gen);
  m.random_init(gen);

  /*std::vector<std::string> input = {
    "<bos>1.I'll be back<eos>",
    "<bos>2.Attention is all you need<eos>",
    "<bos>3.I have a pen.<eos>",
    "<bos>4.I have an apple.<eos>"
  };*/

  const float lr = 0.001;

  float loss499 = 0.0f;
  float loss999 = 0.0f;
  float loss1999 = 0.0f;
  float loss2999 = 0.0f;
  float loss3999 = 0.0f;
  float loss4999 = 0.0f;
  float loss5999 = 0.0f;
  float loss6999 = 0.0f;
  float loss7999 = 0.0f;
  float loss8999 = 0.0f;
  float loss9999 = 0.0f;
  float loss10999 = 0.0f;
  float loss11999 = 0.0f;
  float loss12999 = 0.0f;
  float loss13999 = 0.0f;
  float loss14999 = 0.0f;
  float loss15999 = 0.0f;
  float loss16999 = 0.0f;
  float loss17999 = 0.0f;
  float loss18999 = 0.0f;
  float loss19999 = 0.0f;

  float min = INFINITY;

  for(int64_t i = 0;i < 20000;i++){
    std::vector<std::string> input = {
      make_input(gen),
      make_input(gen),
      make_input(gen),
      make_input(gen),
    };

    const tensor::Tensor &out = m.forward(lang.forward(input,cache_len));

    float loss = 0.0f;
    float conf = 0.0f;

    lang.backward(m.backward(lang.make_grad(out,input,loss,conf,7,10)));

    if(loss < min) min = loss;

    if(i % 100 == 0){
      std::cout << i << "----------------------------------------" << std::endl;

      auto argmax = lang.argmax(out);

      for(int64_t i = 0;i < argmax.size();i++){
        std::cout << input.at(i) << std::endl;
        std::cout << " " << argmax.at(i) << std::endl;
      }

      std::cout << "loss:" << loss << std::endl;
      std::cout << "confidence:" << conf << std::endl;
    }

    if(i == 499) loss499 = loss;
    if(i == 999) loss999 = loss;
    if(i == 1999) loss1999 = loss;
    if(i == 2999) loss2999 = loss;
    if(i == 3999) loss3999 = loss;
    if(i == 4999) loss4999 = loss;
    if(i == 5999) loss5999 = loss;
    if(i == 6999) loss6999 = loss;
    if(i == 7999) loss7999 = loss;
    if(i == 8999) loss8999 = loss;
    if(i == 9999) loss9999 = loss;
    if(i == 10999) loss10999 = loss;
    if(i == 11999) loss11999 = loss;
    if(i == 12999) loss12999 = loss;
    if(i == 13999) loss13999 = loss;
    if(i == 14999) loss14999 = loss;
    if(i == 15999) loss15999 = loss;
    if(i == 16999) loss16999 = loss;
    if(i == 17999) loss17999 = loss;
    if(i == 18999) loss18999 = loss;
    if(i == 19999) loss19999 = loss;

    lang.step(lr);
    m.step(lr);

    lang.zero_grad();
    m.zero_grad();
  }

  std::cout << "499:" << loss499 << std::endl;
  std::cout << "999:" << loss999 << std::endl;
  std::cout << "1999:" << loss1999 << std::endl;
  std::cout << "2999:" << loss2999 << std::endl;
  std::cout << "3999:" << loss3999 << std::endl;
  std::cout << "4999:" << loss4999 << std::endl;
  std::cout << "5999:" << loss5999 << std::endl;
  std::cout << "6999:" << loss6999 << std::endl;
  std::cout << "7999:" << loss7999 << std::endl;
  std::cout << "8999:" << loss8999 << std::endl;
  std::cout << "9999:" << loss9999 << std::endl;
  std::cout << "10999:" << loss10999 << std::endl;
  std::cout << "11999:" << loss11999 << std::endl;
  std::cout << "12999:" << loss12999 << std::endl;
  std::cout << "13999:" << loss13999 << std::endl;
  std::cout << "14999:" << loss14999 << std::endl;
  std::cout << "15999:" << loss15999 << std::endl;
  std::cout << "16999:" << loss16999 << std::endl;
  std::cout << "17999:" << loss17999 << std::endl;
  std::cout << "18999:" << loss18999 << std::endl;
  std::cout << "19999:" << loss19999 << std::endl;

  std::cout << "min:" << min << std::endl;

  return 0;
}