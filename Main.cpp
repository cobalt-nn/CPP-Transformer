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

#include "data/MNISTLoader.hpp"

#include "nlohmann/json.hpp"

using namespace cobalt_715::nn;

int main(){
  std::ifstream ifs("");

  std::string text;

  language::Language lang;

  std::map<std::string,int64_t> count; 

  while(std::getline(ifs,text)){
    for(const std::string s:lang.tokenize(text).v_){
      count[s]++;
    }
  }

  std::vector<std::pair<std::string,int64_t>> count2;

  for(const auto &kv:count){
    count2.push_back(std::pair<std::string,int64_t>(kv.first,kv.second));
  }

  std::sort(count2.begin(),count2.end(),[](auto &a,auto &b){return a.second > b.second;});

  for(const auto &kv:count2){
    lang.add(kv.first);

    if(lang.size() >= 1024 * 4) break;
  }

  lang.build(1024 * 4,128);

  lang.save_all("nn/models/language.json","nn/models/language.bin");

  return 0;
}