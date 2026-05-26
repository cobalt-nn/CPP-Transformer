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
#include "nn/ops/Activation.hpp"
#include "nn/ops/Acts.hpp"
#include "nn/ops/GEMM.hpp"

#include "nn/Model.hpp"

#include "data/MNISTLoader.hpp"

using namespace cobalt_715::nn;

int main(){
  std::vector<float> v(128);

  std::mt19937 gen(0);

  std::uniform_real_distribution<float> dist(-1,1);

  for(int i = 0;i < v.size();i++){
    v[i] = dist(gen);
  }

  tensor::Tensor t({8,2,8},v);

  layer::Attention a(8,2,2,4);

  a.random_init(gen);

  std::cout << a.forward(t).to_string() << std::endl;

  return 0;
}