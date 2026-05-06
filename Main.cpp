#include <iostream>
#include <vector>
#include "nn/ops/vec.hpp"
#include "nn/ops/vec_cpu.hpp"
#include "nn/tensor/Tensor.hpp"
#include "nn/tensor/TensorView.hpp"
#include "nn/tensor/MatrixView.hpp"

#include "nn/layer/ILayer.hpp"
#include "nn/layer/DenseLayer.hpp"
#include "nn/ops/Activation.hpp"
#include "nn/ops/Acts.hpp"

using namespace cobalt_715::nn::tensor;

int main(){
  std::vector<float> a(30);
  std::vector<float> b(30);
  std::vector<float> out(100);

  for(int i = 0;i < 30;i++){
    a[i] = i;
    b[i] = -i;
  }

  Tensor t({5,6},b);

  std::cout << t.to_string() << std::endl;

  cobalt_715::nn::layer::DenseLayer layer(6,3);

  std::cout << layer.to_string() << std::endl;

  std::cout << layer.forward(t).to_string() << std::endl;

  return 0;
}