#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include <ctime>
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

#include "data/MNISTLoader.hpp"

using namespace cobalt_715::nn;

int main(){
  /*EnglishTokenizer ev;

  std::string s;

  std::getline(std::cin,s);

  for(const std::string_view str:ev.tokenize(s)){
    std::cout << str << std::endl;
  }

  std::cout << "end" << std::endl;


  return 0;*/

  Model m;

  m.add<layer::RMSNorm>(32)
   .add<layer::ReZero>(32,128,std::make_unique<layer::Attention>(32,4,32,32,64,true))
   .add<layer::RMSNorm>(32)
   .add<layer::ReZero>(std::make_unique<layer::FFN>(32))
   .add<layer::RMSNorm>(32)
   .add<layer::ReZero>(32,128,std::make_unique<layer::Attention>(32,4,32,32,64,true))
   .add<layer::RMSNorm>(32)
   .add<layer::ReZero>(std::make_unique<layer::FFN>(32))
   .add<layer::RMSNorm>(32)
   .add<layer::ReZero>(32,128,std::make_unique<layer::Attention>(32,4,32,32,64,true))
   .add<layer::RMSNorm>(32)
   .add<layer::ReZero>(std::make_unique<layer::FFN>(32))
   .add<layer::RMSNorm>(32)
   .add<layer::ReZero>(32,128,std::make_unique<layer::Attention>(32,4,32,32,64,true))
   .add<layer::RMSNorm>(32)
   .add<layer::ReZero>(std::make_unique<layer::FFN>(32));

  tensor::Tensor input = tensor::Tensor({4,64,32});
  tensor::Tensor output = tensor::Tensor({4,64,32});

  auto start = std::chrono::system_clock::now();

  m.backward(m.forward(input) - output);

  auto end = std::chrono::system_clock::now();

  std::cout << end - start << std::endl;


  auto start1 = std::chrono::system_clock::now();

  m.backward(m.forward(input) - output);

  auto end1 = std::chrono::system_clock::now();

  std::cout << end1 - start1 << std::endl;


  start1 = std::chrono::system_clock::now();

  m.backward(m.forward(input) - output);

  end1 = std::chrono::system_clock::now();

  std::cout << end1 - start1 << std::endl;
}