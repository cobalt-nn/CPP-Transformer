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
#include "nn/ops/Activation.hpp"
#include "nn/ops/Acts.hpp"
#include "nn/ops/GEMM.hpp"

#include "nn/io/BinaryIO.hpp"

#include "nn/Model.hpp"

#include "data/MNISTLoader.hpp"

#include "nlohmann/json.hpp"

using namespace cobalt_715::nn;

std::vector<float> makeTenVector(int i){
  std::vector<float> v(10);
  v.at(i) = 1;
  return v;
}

int main(){
  const ops::Activation act = ops::activations::GELU;

  Model m;

  m.add<layer::Dense>(784,256,act)
   .add<layer::Dense>(256,64,act)
   .add<layer::Linear>(64,10,true)
   //.add<layer::Dense>(64,10,act)
   .add<layer::Softmax>();

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

  size_t total = input.size();

  auto start = std::chrono::high_resolution_clock::now();

  for(int j = 0;j < 100;j++){
    for(size_t i = 0;i < total - 300;i++){
      const tensor::Tensor &output = m.forward(input[i]);
      m.backward(output - target[i]);

      m.step(lr);

      m.zero_grad();
    }
  }

  auto end = std::chrono::high_resolution_clock::now();

  auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  std::cout << "time: " << time << "ms\n";

  double total_loss = 0.0;
  double total_conf = 0.0;
  double max_conf = 0.0;
  double min_conf = 0.0;
  int correct = 0;

  bool first = true;

  size_t sample_num = 0;

  for(size_t i = total - 300;i < total;i++){
    const tensor::Tensor &output = m.forward(input[i]);

    sample_num += input[i].dim(0);

    for(int64_t row = 0;row < output.dim(0);row++){
      int max_index = -1;
      int target_i = -1;
      double max_element = 0.0;

      for(int64_t col = 0;col < output.dim(1);col++){
        double out_at = output.at({row,col});
        double tar_at = target[i].at({row,col});

        if(tar_at > 0.99f) target_i = col;

        if(max_element < out_at){// && out_at > 0.8f){
          max_element = out_at;
          max_index = col;
        }

        double sub = out_at - tar_at;

        //double loss = sub * sub;//MSE
        double loss = -std::log(out_at + 1e-8f) * tar_at;//Cross Entropy

        total_loss += loss;
      }

      if(target_i == max_index){
        correct++;
        total_conf += max_element;

        if(first){
          first = false;

          max_conf = min_conf = max_element;
        }

        if(max_conf < max_element) max_conf = max_element;
        if(min_conf > max_element) min_conf = max_element;
      }
    }

    /*tensor::Tensor t = output - target[i];
    t.hadamard_(t);

    for(double f:t.span()){
      total_loss += f;
    }*/
  }

  std::cout << "loss:" << total_loss / sample_num << std::endl;
  std::cout << "conf:" << total_conf / sample_num << std::endl;
  std::cout << "max conf:" << max_conf << std::endl;
  std::cout << "min conf:" << min_conf << std::endl;
  std::cout << "correct:" << correct / static_cast<double>(sample_num) << std::endl;

  return 0;
}