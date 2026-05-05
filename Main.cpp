#include <iostream>
#include <vector>
#include "nn/ops/vec.hpp"
#include "nn/ops/vec_cpu.hpp"
#include "nn/tensor/Tensor.hpp"
#include "nn/tensor/TensorView.hpp"
#include "nn/tensor/MatrixView.hpp"

#include "nn/layer/ILayer.hpp"

using namespace cobalt_715::nn::tensor;

int main(){
  std::vector<float> a(100);
  std::vector<float> b(100);
  std::vector<float> out(100);

  for(int i = 0;i < 100;i++){
    a[i] = i;
    b[i] = -i;
  }

  Tensor ta({2,5,10},a);

  MatrixView m1(5,3,10,1,&ta.data()[0]);
  MatrixView m2(3,5,10,1,&ta.data()[5]);

  MatrixView m3(5,5,10,1,&ta.data()[50]);

  MatrixView::matmul(m1,m2,m3);

  std::cout << m1.to_string() << "\n" << m2.to_string() << "\n" << m3.to_string() << std::endl;

  std::cout << m3.t().to_string() << "\n" << ta.to_string() << std::endl;

  return 0;
}