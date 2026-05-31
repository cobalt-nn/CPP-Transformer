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
#include "nn/layer/ReZero.hpp"
#include "nn/layer/FFN.hpp"
#include "nn/ops/Activation.hpp"
#include "nn/ops/Acts.hpp"
#include "nn/ops/GEMM.hpp"

#include "nn/Model.hpp"

#include "data/MNISTLoader.hpp"

using namespace cobalt_715::nn;

void set(tensor::Tensor &input,tensor::Tensor &output,std::mt19937 &gen){
  std::fill(input.data(),input.data() + input.numel(),0.0f);
  std::fill(output.data(),output.data() + output.numel(),0.0f);

  for(int batch = 0;batch < input.dim(0);batch++){
    std::vector<bool> v(input.dim(1));
    for(int i = 0;i < input.dim(1);i++){
      uint32_t j = 0;
      while(true){
        j = gen() % input.dim(1);

        if(!v[j]){
          v[j] = true;
          break;
        }
      }

      input.at({batch,i,j}) = 1.3f;
      output.at({batch,input.dim(1) - 1 - i,j}) = 1.3f;
    }

    for(int i = 0;i < input.dim(1);i++){
      for(int j = 0;j < input.dim(2);j++){
        input.at({batch,i,j}) += 0.1f * (i - input.dim(1) / 2) + 0.03f * (j - input.dim(2) / 2);
        output.at({batch,input.dim(1) - 1 - i,j}) += 0.1f * (i - input.dim(1) / 2) + 0.03f * (j - input.dim(2) / 2);
      }
    }
  }
}

int main(){
  tensor::Tensor input({2,8,8});
  tensor::Tensor output({2,8,8});

  /*tensor::Tensor input({2,3,4});
  tensor::Tensor output({2,3,6});

  for(int i = 0;i < output.numel();i++){
    output.data()[i] = i / 10.0f;
  }

  std::cout << "output" << output.to_string() << std::endl;

  layer::Attention a(4,2,2,3);

  std::cout << a.forward(input).to_string() << std::endl;
  std::cout << a.backward(output).to_string() << std::endl;

  return 0;*/

  std::mt19937 gen(0);

  set(input,output,gen);

  /*std::cout << input.to_string() << std::endl;
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

  return 0;*/

  Model m;

  m.add<layer::RMSNorm>(8)
   .add<layer::ReZero>(8,8,std::make_unique<layer::Attention>(8,2,4,4))
   //.add<layer::Attention>(8,2,4,4);
   .add<layer::RMSNorm>(8)
   .add<layer::ReZero>(8,8,std::make_unique<layer::FFN>(8))
   //.add<layer::FFN>(8)
   .add<layer::RMSNorm>(8)
   .add<layer::ReZero>(8,8,std::make_unique<layer::Attention>(8,2,4,4))
   //.add<layer::Attention>(8,2,4,4);
   .add<layer::RMSNorm>(8)
   .add<layer::ReZero>(8,8,std::make_unique<layer::FFN>(8))
   //.add<layer::FFN>(8)
   ;

  m.random_init(gen);

  for(int i = 0;i < 100000000;i++){
    set(input,output,gen);

    auto &out = m.forward(input);

    if(i % 1024 == 0){
      std::cout << "time: " << i << std::endl;

      std::cout << "input" << input.to_string() << std::endl;

      std::cout << "out" << out.to_string() << std::endl;

      //std::cout << m.to_string() << std::endl;
    }

    m.backward(out - output);

    m.step(0.001f,1);

    m.zero_grad();
  }

  return 0;
}