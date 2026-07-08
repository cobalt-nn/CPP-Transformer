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
#include "nn/language/EnglishTokenizer.hpp"
#include "nn/language/Vocabulary.hpp"
#include "nn/language/Embedding.hpp"
#include "nn/language/SpecialToken.hpp"
#include "nn/language/Language.hpp"
#include "nn/language/Tokens.hpp"

#include "data/MNISTLoader.hpp"

#include "nlohmann/json.hpp"

using namespace cobalt_715::nn;

int main(){
  std::mt19937 gen(0);

  language::Language lang;

  lang.add("1234567890 I'll be back Attention Is All You Need I have a pen.I have an apple. NN");

  lang.build(lang.size(),32);

  const int64_t cache_len = 32;

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
   .add<layer::ReZero>(std::make_unique<layer::FFN>(32))
   .add<layer::Linear>(32,lang.size())
   .add<layer::Softmax>();

  lang.random_init(gen);
  m.random_init(gen);

  std::vector<std::string> input = {
    "<bos>1.I'll be back<eos>",
    "<bos>2.Attention is all you need<eos>",
    "<bos>3.I have a pen.<eos>",
    "<bos>4.I have an apple.<eos>"
  };

  const float lr = 0.001;

  float loss499 = 0.0f;
  float loss999 = 0.0f;

  for(int64_t i = 0;i < 1000;i++){
    std::cout << i << "----------------------------------------" << std::endl;

    const tensor::Tensor &out = m.forward(lang.forward(input,cache_len));

    float loss = 0.0f;
    float conf = 0.0f;

    lang.backward(m.backward(lang.make_grad(out,input,loss,conf)));

    for(const std::string_view s:lang.argmax(out)){
      std::cout << s << std::endl;
    }

    if(i == 499) loss499 = loss;
    if(i == 999) loss999 = loss;

    std::cout << "loss:" << loss << std::endl;
    std::cout << "confidence:" << conf << std::endl;

    lang.step(lr);
    m.step(lr);

    lang.zero_grad();
    m.zero_grad();
  }

  std::cout << "499:" << loss499 << std::endl;
  std::cout << "999:" << loss999 << std::endl;

  return 0;
}