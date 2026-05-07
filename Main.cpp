#include <iostream>
#include <random>
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

using namespace cobalt_715::nn;

int main(){
  std::vector<float> v1 =
    {
       0,0,
       0,1,
       1,0,
       1,1
    };

  tensor::Tensor input({4,2},v1);

  std::vector<float> v2 =
    {
      0,
      1,
      1,
      0
    };

  tensor::Tensor target({4,1},v2);

  layer::DenseLayer l1(2,3);
  layer::DenseLayer l2(3,1);

  std::mt19937 gen(0);

  l1.random_init(gen);
  l2.random_init(gen);

  const float lr = 0.1;

  for(int i = 0;i < 200;i++){
    const tensor::Tensor output = l2.forward(l1.forward(input));

    l1.backward(l2.backward(output - target));

    l1.step(lr,4);
    l2.step(lr,4);

    l1.zero_grad();
    l2.zero_grad();

    std::cout << i << "\n" << output.to_string() << std::endl;
  }

  return 0;
}