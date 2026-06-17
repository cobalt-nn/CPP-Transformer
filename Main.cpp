#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include <ctime>
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
#include "nn/layer/Identity.hpp"
#include "nn/ops/Acts.hpp"
#include "nn/ops/GEMM.hpp"

#include "nn/Model.hpp"
#include "nn/EnglishTokenizer.hpp"
#include "nn/Vocabulary.hpp"
#include "nn/Embedding.hpp"

#include "data/MNISTLoader.hpp"

using namespace cobalt_715::nn;

int main(){
  const std::string case1 = "1:I'll be back.";
  const std::string case2 = "2:attention is all you need";
  const std::string case3 = "3:I have a pen.";
  const std::string case4 = "4:I have an apple";

  EnglishTokenizer et;

  /*for(const std::string &s:et.tokenize(case1)){
    std::cout << s << std::endl;
  }

  std::cout << et.detokenize(et.tokenize(case1)) << std::endl;

  return 0;*/

  Vocabulary voc;

  voc.add(et.tokenize(case1 + case2 + case3 + case4));

  //std::cout << voc.to_string() << std::endl;

  std::vector<std::vector<int64_t>> ids = {
    {voc.stoi(et.format(case1,16))},
    {voc.stoi(et.format(case2,16))},
    {voc.stoi(et.format(case3,16))},
    {voc.stoi(et.format(case4,16))},
  };

  tensor::Tensor target({4,16,voc.size()});

  for(int64_t i = 0;i < target.dim(0);i++){
    for(int64_t j = 0;j < target.dim(1);j++){
      if(j + 1 == target.dim(1)){
        target.at({i,j,0}) = 1.0f;
      }else{
        target.at({i,j,ids.at(i).at(j + 1)}) = 1.0f;
      }
    }
  }

  //std::cout << target.to_string() << std::endl;

  Embedding em(voc.size(),32);

  Model m;

  m.add<layer::RMSNorm>(32)
   .add<layer::ReZero>(32,128,std::make_unique<layer::Attention>(32,4,32,32,64,true))
   .add<layer::RMSNorm>(32)
   .add<layer::ReZero>(std::make_unique<layer::FFN>(32))
   .add<layer::RMSNorm>(32)
   .add<layer::ReZero>(32,128,std::make_unique<layer::Attention>(32,4,32,32,64,true))
   .add<layer::RMSNorm>(32)
   .add<layer::ReZero>(std::make_unique<layer::FFN>(32))
   .add<layer::RMSNorm>(32)
   .add<layer::ReZero>(32,128,std::make_unique<layer::Attention>(32,4,32,32,64,true))
   .add<layer::RMSNorm>(32)
   .add<layer::FFN>(32,32*4,voc.size())
   .add<layer::Softmax>();

  std::mt19937 gen(0);

  em.random_init(gen);
  m.random_init(gen);

  const float lr = 0.001f;

  for(int64_t i = 0;i < 1000;i++){
    const tensor::Tensor &em_out = em.forward(ids);
    const tensor::Tensor &m_out = m.forward(em_out);

    em.backward(m.backward(m_out - target));

    std::cout << "time:" << i << " ------------------------------" << std::endl;

    for(int64_t batch = 0;batch < m_out.dim(0);batch++){
      std::vector<int64_t> id;
      for(int64_t row = 0;row < m_out.dim(1);row++){
        id.push_back(std::max_element(&m_out.data()[batch * m_out.stride()[0] + row * m_out.stride()[1]],&m_out.data()[batch * m_out.stride()[0] + (row + 1) * m_out.stride()[1]]) - &m_out.data()[batch * m_out.stride()[0] + row * m_out.stride()[1]]);
      }

      std::cout << et.detokenize(voc.itos(id)) << std::endl;
    }

    em.step(lr,4);
    m.step(lr,4);

    em.zero_grad();
    m.zero_grad();
  }
}