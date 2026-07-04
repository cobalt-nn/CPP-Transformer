#include <iostream>
#include <memory>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>
#include <chrono>
#include <ctime>
#include <cstdint>
#include <cmath>
#include <cstddef>
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

#include "nn/io/BinaryIO.hpp"

#include "nn/Model.hpp"
#include "nn/EnglishTokenizer.hpp"
#include "nn/Vocabulary.hpp"
#include "nn/Embedding.hpp"
#include "nn/SpecialToken.hpp"

#include "data/MNISTLoader.hpp"

#include "nlohmann/json.hpp"

using namespace cobalt_715::nn;

std::vector<float> makeTenVector(int i){
  std::vector<float> v(10);
  v.at(i) = 1;
  return v;
}

int main(){
  Model m;

  m.add<layer::Dense>(784,256)
   .add<layer::Dense>(256,64)
   .add<layer::Dense>(64,10);

  const size_t MNIST_size = 60000;

  const size_t batch_size = 32;

  std::vector<float> images;
  std::vector<float> labels;

  std::vector<tensor::Tensor> input;
  std::vector<tensor::Tensor> target;

  MNISTLoader mnist("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");

  for(size_t i = 0;i < MNIST_size;i += batch_size){
    images.clear();
    labels.clear();
    for(size_t j = i;j < std::min(i + batch_size,MNIST_size);j++){
      for(float f:mnist.getImage(j)){
        images.push_back(f);
      }
      for(float f:makeTenVector(mnist.getLabel(j))){
        labels.push_back(f);
      }
    }
    input.push_back(tensor::Tensor({static_cast<int64_t>(std::min(batch_size,MNIST_size - i)),784},images));
    target.push_back(tensor::Tensor({static_cast<int64_t>(std::min(batch_size,MNIST_size - i)),10},labels));
  }

  std::mt19937 gen(0);

  m.random_init(gen);

  const float lr = 0.01;

  auto start = std::chrono::high_resolution_clock::now();

  for(size_t i = 0;i < input.size();i++){
    const tensor::Tensor &output = m.forward(input[i]);
    m.backward(output - target[i]);

    m.step(lr);

    m.zero_grad();
  }

  auto end = std::chrono::high_resolution_clock::now();

  auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  std::cout << "time: " << time << "ms\n";

  double total_loss = 0.0;
  int correct = 0;
  size_t total = input.size();

  for(size_t i = 0;i < total;i++){
    const tensor::Tensor output = m.forward(input[i]);

    tensor::Tensor t = output - target[i];
    t.hadamard_(t);

    for(float f:t.span()){
      total_loss += f;
    }
  }

  std::cout << total_loss << std::endl;

  return 0;
}