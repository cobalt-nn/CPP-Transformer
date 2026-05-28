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

void set(tensor::Tensor &input,tensor::Tensor &output,std::mt19937 &gen){
  std::fill(input.data(),input.data() + input.numel(),0.0f);
  std::fill(output.data(),output.data() + output.numel(),0.0f);

  std::vector<bool> v(8);

  for(int i = 0;i < 8;i++){
    uint32_t j = 0;
    while(true){
      j = gen() % 8;

      if(!v[j]){
        v[j] = true;
        break;
      }
    }

    input.at({0,i,j}) = 0.98f + i * 0.01f;
    output.at({0,i,7 - j}) = 1.0f;
  }
}

int main(){
  tensor::Tensor input({1,8,8});
  tensor::Tensor output({1,8,8});

  std::mt19937 gen(0);

  set(input,output,gen);

  std::cout << input.to_string() << std::endl;
  std::cout << output.to_string() << std::endl;

  set(input,output,gen);

  std::cout << input.to_string() << std::endl;
  std::cout << output.to_string() << std::endl;

  set(input,output,gen);

  std::cout << input.to_string() << std::endl;
  std::cout << output.to_string() << std::endl;

  set(input,output,gen);

  std::cout << input.to_string() << std::endl;
  std::cout << output.to_string() << std::endl;

  return 0;

  Model m;

  m.add<layer::RMSNorm>(8)
   .add<layer::Attention>(8,2,4,4)
   .add<layer::Dense>(8,8)
   .add<layer::RMSNorm>(8)
   .add<layer::Attention>(8,2,4,4)
   .add<layer::Dense>(8,8);

  m.random_init(gen);

  for(int i = 0;i < 512;i++){
    set(input,output,gen);

    auto &out = m.forward(input);

    if(i % 32 == 0){
      std::cout << input.to_string() << std::endl;

      std::cout << out.to_string() << std::endl;
    }

    m.backward(out - output);

    m.step(0.001f,1);

    m.zero_grad();
  }

  return 0;
}