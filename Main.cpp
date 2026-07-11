#include <iostream>
#include <memory>
#include <map>
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

#include "nlohmann/json.hpp"

#include "DataMaker.hpp"

using namespace cobalt_715::nn;

int main(){
  std::mt19937 gen(0);

  language::Language lang;

  lang.load_all("nn/models/language.json","nn/models/language.bin");

  const int64_t dim = 128;
  const int64_t head_num = 4;
  const int64_t cache_len = 256;
  const ops::Activation &act = ops::activations::LeakyReLU;

  Model m;

  for(size_t i = 0;i < 16;i++){
    m.add<layer::RMSNorm>(dim)
     .add<layer::ReZero>(dim,dim * head_num,std::make_unique<layer::Attention>(dim,head_num,dim,dim,cache_len,true))
     .add<layer::RMSNorm>(dim)
     .add<layer::ReZero>(std::make_unique<layer::FFN>(dim,act));
  }

  m.add<layer::Linear>(dim,lang.size())
   .add<layer::Softmax>();

  m.load_all("nn/models/model.json","nn/models/model.bin");

  //lang.random_init(gen);
  //m.random_init(gen);

  DataMaker dm(cache_len,lang);

  const float lr = 0.00005f;

  std::vector<float> loss_arr;
  std::vector<float> conf_arr;

  for(int64_t i = 0;i < 2500;i++){
    std::vector<language::Tokens> input = {
      dm.wikitext(),
      dm.wikitext(),
      dm.wikitext(),
      dm.wikitext(),
      dm.wikitext(),
      dm.wikitext(),
      dm.wikitext(),
      dm.wikitext()
    };
  }

  for(int64_t i = 2500;i < 6000;i++){
    std::vector<language::Tokens> input = {
      dm.wikitext(),
      dm.wikitext(),
      dm.wikitext(),
      dm.wikitext(),
      dm.wikitext(),
      dm.wikitext(),
      dm.dolly(),
      dm.dolly()
    };
    const tensor::Tensor &out = m.forward(lang.forward(input,cache_len));

    float loss = 0.0f;
    float conf = 0.0f;

    lang.backward(m.backward(lang.make_grad(out,input,loss,conf)));

    if(i % 10 == 0){

      std::cout << i << "****************************************" << std::endl;

      std::cout << "input:" << lang.detokenize(input.at(0)).substr(0,64) << std::endl;
      std::cout << "----------------------------------------" << std::endl;
      std::cout << "output:" << lang.argmax(out).at(0).substr(0,64) << std::endl;

      std::cout << "++++++++++++++++++++++++++++++++++++++++" << std::endl;

      std::cout << "input:" << lang.detokenize(input.at(7)).substr(0,64) << std::endl;
      std::cout << "----------------------------------------" << std::endl;
      std::cout << "output:" << lang.argmax(out).at(7).substr(0,64) << std::endl;

      std::cout << "loss:" << loss << std::endl;
      std::cout << "confidence:" << conf << std::endl;
    }

    loss_arr.push_back(loss);
    conf_arr.push_back(conf);

    lang.step(lr);
    m.step(lr);

    lang.zero_grad();
    m.zero_grad();

    if(i % 500 == 0){
      m.save_all("nn/models/model.json","nn/models/model.bin");
      lang.save_all("nn/models/language.json","nn/models/language.bin");
    }
  }

  m.save_all("nn/models/model.json","nn/models/model.bin");
  lang.save_all("nn/models/language.json","nn/models/language.bin");

  nlohmann::ordered_json json;

  json["loss"] = loss_arr;
  json["confidence"] = conf_arr;

  std::ofstream loss_conf("nn/models/loss.json");

  loss_conf << json.dump(2);

  return 0;
}