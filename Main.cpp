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
  std::vector<float> a(100);
  std::vector<float> b(100);
  std::vector<float> out(100);

  for(int i = 0;i < 100;i++){
    a[i] = i;
    b[i] = -i;
  }

  Tensor t({10,10},a);

  MatrixView m1 = t.unsafe_matrix_view(10,10,-10,-1,99);
  MatrixView m2 = t.as_matrix_view({});

  std::cout << t.to_string() << std::endl;
  std::cout << m1.to_string() << std::endl;
  std::cout << m2.to_string() << std::endl;

  return 0;
}