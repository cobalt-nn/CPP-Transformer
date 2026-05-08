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

#include "data/MNISTLoader.hpp"

using namespace cobalt_715::nn;

std::vector<float> makeTenVector(int i){
  std::vector<float> v(10);
  v.at(i) = 1;
  return v;
}

int main(){
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

  layer::DenseLayer l1(784,256);
  layer::DenseLayer l2(256,64);
  layer::DenseLayer l3(64,10);

  std::mt19937 gen(0);

  l1.random_init(gen);
  l2.random_init(gen);
  l3.random_init(gen);

  //std::cout << input[0].to_string() << std::endl;

  const float lr = 0.01;

  auto start = std::chrono::high_resolution_clock::now();

  for(size_t i = 0;i < input.size();i++){
    const tensor::Tensor &output = l3.forward(l2.forward(l1.forward(input[i])));
    l1.backward(l2.backward(l3.backward(output - target[i])));

    //std::cout << output.to_string() << std::endl;

    l1.step(lr,batch_size);
    l2.step(lr,batch_size);
    l3.step(lr,batch_size);

    l1.zero_grad();
    l2.zero_grad();
    l3.zero_grad();
  }

  auto end = std::chrono::high_resolution_clock::now();

  auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  std::cout << "time: " << time << "ms\n";

  double total_loss = 0.0;
  int correct = 0;
  size_t total = input.size();

  for(size_t i = 0;i < total;i++){
    const tensor::Tensor output = l3.forward(l2.forward(l1.forward(input[i])));

    tensor::Tensor t = output - target[i];
    t.hadamard_(t);

    for(float f:t.span()){
      total_loss += f;
    }
  }

  std::cout << total_loss << std::endl;

  return 0;
}