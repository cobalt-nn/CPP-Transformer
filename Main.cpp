#include <iostream>
#include <vector>
#include "nn/linear/vec.hpp"
#include "nn/linear/vec_cpu.hpp"
#include "nn/linear/Tensor.hpp"
#include "nn/linear/TensorView.hpp"
#include "nn/linear/MatrixView.hpp"

using namespace cobalt_715::nn::linear;

int main(){
  std::vector<float> a(100);
  std::vector<float> b(100);
  std::vector<float> out(100);

  for(int i = 0;i < 100;i++){
    a[i] = i;
    b[i] = -i;
  }

  Tensor ta({2,5,10},a);

  MatrixView m(5,5,10,2,&ta.data()[0]);
  MatrixView m2(5,5,10,2,&ta.data()[50]);

  std::cout << m.to_string() << std::endl;
  std::cout << m2.to_string() << std::endl;

  MatrixView::add(m,m,m2);

  std::cout << m.to_string() << std::endl;
  std::cout << m2.to_string() << std::endl;

  std::cout << ta.to_string() << std::endl;

  return 0;
}