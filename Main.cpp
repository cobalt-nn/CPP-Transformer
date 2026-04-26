#include <iostream>
#include <vector>
#include "nn/linear/vec.hpp"
#include "nn/linear/vec_cpu.hpp"
#include "nn/linear/Tensor.hpp"

using namespace cobalt_715::nn::linear;

int main(){
  std::vector<float> a(100);
  std::vector<float> b(100);
  std::vector<float> out(100);

  for(int i = 0;i < 100;i++){
    a[i] = i;
    b[i] = -i;
  }

  Tensor at({2,5,10},a);
  Tensor bt({2,5,10},b);

  Tensor ot({2,5,10});

  //ot = at + bt;
  Tensor::scale(at,2,ot);

  std::cout << ot.to_string() << std::endl;

  return 0;
}