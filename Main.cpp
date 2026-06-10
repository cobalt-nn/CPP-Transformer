#include <iostream>
#include <random>
#include <vector>
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
#include "nn/ops/Acts.hpp"
#include "nn/ops/GEMM.hpp"

#include "nn/Model.hpp"
#include "nn/EnglishVocabulary.hpp"

#include "data/MNISTLoader.hpp"

using namespace cobalt_715::nn;

int main(){
  EnglishVocabulary ev;

  std::string s;

  std::getline(std::cin,s);

  for(const std::string_view str:ev.tokenize(s)){
    std::cout << str << std::endl;
  }

  std::cout << "end" << std::endl;


  return 0;
}