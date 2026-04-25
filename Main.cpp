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

  axpy(a.data(),2.0,b.data(),out.data(),100);

  for(float f:out){
    std::cout << f << " ";
  }

  std::cout.flush();

  return 0;
}