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
#include "nn/layer/Softmax.hpp"
#include "nn/layer/ReZero.hpp"
#include "nn/layer/FFN.hpp"
#include "nn/ops/Activation.hpp"
#include "nn/ops/Acts.hpp"
#include "nn/ops/GEMM.hpp"

#include "nn/Model.hpp"

#include "data/MNISTLoader.hpp"

#include "TokenSystem.hpp"

using namespace cobalt_715::nn;

static std::mt19937 gen(0);
static int64_t dim = 8;
static TokenSystem ts(dim);

std::string set_str(std::mt19937 &gen){
  std::string s;

  uint64_t n = gen() % 100;
  uint64_t m = gen() % 100;

  s += std::to_string(n) + "+" + std::to_string(m) + "=" + std::to_string(n + m);

  return s;
}

void set_t(const std::string &str,tensor::Tensor &output){
  if(output.dim(1) != str.size() || output.dim(2) != ts.to_size()){
    output = tensor::Tensor({1,static_cast<int64_t>(str.size()),static_cast<int64_t>(ts.to_size())});
  }else{
    std::fill(output.data(),output.data() + output.numel(),0.0f);
  }

  for(int64_t i = 0;i < output.dim(1) - 1;i++){
    output.at({0,i,ts.char_to_index(str[i + 1])}) = 1.0f;
  }

  output.at({0,output.dim(1) - 1,output.dim(2) - 1}) = 1.0f;
}

int main(){
  ts.random_init(gen);

  tensor::Tensor output({1,1,1});

  //int64_t in = static_cast<int64_t>(ts.to_size());

  Model m;

  m.add<layer::RMSNorm>(dim)
   .add<layer::ReZero>(dim,16,std::make_unique<layer::Attention>(8,2,16,8,128,true))
   .add<layer::RMSNorm>(dim)
   .add<layer::FFN>(dim)
   .add<layer::RMSNorm>(dim)
   .add<layer::ReZero>(dim,16,std::make_unique<layer::Attention>(8,2,16,8,128,true))
   .add<layer::RMSNorm>(dim)
   .add<layer::FFN>(dim)
   .add<layer::RMSNorm>(dim)
   .add<layer::ReZero>(dim,16,std::make_unique<layer::Attention>(8,2,16,8,128,true))
   .add<layer::RMSNorm>(dim)
   .add<layer::FFN>(dim,dim * 4,static_cast<int64_t>(ts.to_size()))
   .add<layer::Softmax>()
   ;

  m.random_init(gen);

  for(int64_t i = 0;i < 30000;i++){

    std::string str = set_str(gen);

    set_t(str,output);

    const auto in = ts.forward(str);

    //std::cout in.to_string() << std::endl;
    //std::cout << output.to_string() << std::endl;

    const auto out = m.forward(in);

    const auto out2 = m.backward(out - output);

    ts.step(out2,0.001f);

    m.step(0.0001f,1);

    m.zero_grad();

    //std::cout << "out\n" << out.to_string() << std::endl;

    std::string out_str;

    for(int64_t i = 0;i < out.dim(1);i++){
      int64_t max_index = std::max_element(&out.at({0,i,0}),&out.at({0,i,0}) + out.dim(2)) -  &out.at({0,i,0});
      //std::cout << max_index << std::endl;
      out_str += ts.index_to_char(max_index);
    }

    if(i % 10000 == 0){
      std::cout << "time" << i << std::endl;
      std::cout << "input:" << str << std::endl;
      std::cout << "out:" << out_str << "\n" << std::endl;
    }
  }

  std::cout << "end" << std::endl;

  while(true){
    std::string in_str;
    std::cin >> in_str;

    std::string out_str;

    std::cout << in_str << std::endl;

    int64_t count = 0;

    while(true){
      const auto in = ts.forward(in_str);
      const auto out = m.forward(in,false);

      //std::cout << "in" << in.to_string() << std::endl;
      //std::cout << "out" << out.to_string() << std::endl;

      int64_t i = out.dim(1) - 1;

      int64_t max_index = std::max_element(&out.at({0,i,0}),&out.at({0,i,0}) + out.dim(2)) -  &out.at({0,i,0});
      //std::cout << max_index << std::endl;
      in_str = ts.index_to_char(max_index);
      std::cout << "in_str: " << in_str << std::endl;
      out_str += in_str;
      if(ts.index_to_char(max_index) == 'e'|| count > 32){
        std::cout << "out_str: " << out_str << std::endl;
        break;
      }
      count++;
    }
    m.reset();
  }

  return 0;
}