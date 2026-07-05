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

#include "data/MNISTLoader.hpp"

#include "nlohmann/json.hpp"

using namespace cobalt_715::nn;

int main(){
  language::Language lang;

  lang.load_all("nn/models/Test.json","nn/models/Test.bin");

  //lang.add("I'll be back.Transformer Attention Is All You Need I have an apple.");

  language::EnglishTokenizer et;

  //lang.build(lang.size(),8);

  std::cout << lang.to_json().dump(2) << std::endl;

  for(auto s:et.format("I'll be back",16)){
    std::cout << s << std::endl;
  }

  std::cout << lang.forward({"I'll be back"},16).to_string() << std::endl;

  lang.save_all("nn/models/Test.json","nn/models/Test.bin");

  return 0;
}