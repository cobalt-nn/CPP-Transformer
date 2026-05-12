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
#include "nn/layer/DenseLayer.hpp"
#include "nn/ops/Activation.hpp"
#include "nn/ops/Acts.hpp"

#include "data/MNISTLoader.hpp"

using namespace cobalt_715::nn;

int main(){
  std::vector<float> v(588);

  for(int i = 0;i < v.size();i++){
    v[i] = i;
  }

  tensor::Tensor t({3,14,14},v);

  tensor::MatrixView in1 = t.unsafe_matrix_view(8,12,14,1,15);
  tensor::MatrixView in2 = t.unsafe_matrix_view(8,12,14,1,211);

  tensor::MatrixView out1 = t.unsafe_matrix_view(8,8,14,1,407);
  tensor::MatrixView out2 = t.unsafe_matrix_view(12,12,14,1,407);

  //std::cout << t.to_string() << "\n" << in1.to_string() << "\n" << in2.to_string() << "\n" << out1.to_string() << "\n" << out2.to_string() << std::endl;

  tensor::MatrixView::matmul(in1.t(),in2,out2);

  std::cout << out2.to_string() << std::endl;

// 12 * 12
//{
//{ 141352. , 141864. , 142376. , 142888. , 143400. , 143912. , 144424. , 144936. , 145448. , 145960. , 146472. , 146984. } ,
//{ 143432. , 143952. , 144472. , 144992. , 145512. , 146032. , 146552. , 147072. , 147592. , 148112. , 148632. , 149152. } ,
//{ 145512. , 146040. , 146568. , 147096. , 147624. , 148152. , 148680. , 149208. , 149736. , 150264. , 150792. , 151320. } ,
//{ 147592. , 148128. , 148664. , 149200. , 149736. , 150272. , 150808. , 151344. , 151880. , 152416. , 152952. , 153488. } ,
//{ 149672. , 150216. , 150760. , 151304. , 151848. , 152392. , 152936. , 153480. , 154024. , 154568. , 155112. , 155656. } ,
//{ 151752. , 152304. , 152856. , 153408. , 153960. , 154512. , 155064. , 155616. , 156168. , 156720. , 157272. , 157824. } ,
//{ 153832. , 154392. , 154952. , 155512. , 156072. , 156632. , 157192. , 157752. , 158312. , 158872. , 159432. , 159992. } ,
//{ 155912. , 156480. , 157048. , 157616. , 158184. , 158752. , 159320. , 159888. , 160456. , 161024. , 161592. , 162160. } ,
//{ 157992. , 158568. , 159144. , 159720. , 160296. , 160872. , 161448. , 162024. , 162600. , 163176. , 163752. , 164328. } ,
//{ 160072. , 160656. , 161240. , 161824. , 162408. , 162992. , 163576. , 164160. , 164744. , 165328. , 165912. , 166496. } ,
//{ 162152. , 162744. , 163336. , 163928. , 164520. , 165112. , 165704. , 166296. , 166888. , 167480. , 168072. , 168664. } ,
//{ 164232. , 164832. , 165432. , 166032. , 166632. , 167232. , 167832. , 168432. , 169032. , 169632. , 170232. , 170832. }
//}

  return 0;
}

/*std::vector<float> makeTenVector(int i){
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
  int64_t correct = 0;
  size_t total = 5;

  for(size_t i = 0;i < total;i++){
    const tensor::Tensor output = l3.forward(l2.forward(l1.forward(input[i])));

    //std::cout << output.to_string() << std::endl;

    tensor::Tensor t = output - target[i];
    t.hadamard_(t);

    for(float f:t.span()){
      total_loss += f;
    }

    for(int64_t row = 0;row < output.shape()[0];row++){
      int64_t max_index = 0;
      float max_element = output.at({row,0});
      for(int64_t col = 1;col < output.shape()[1];col++){
        if(max_element < output.at({row,col})){
          max_index = col;
          max_element = output.at({row,col});
        }
      }
      if(max_index == mnist.getLabel(i * batch_size + row)){
        correct++;
      }
      //std::cout << mnist.getLabel(i * batch_size + row) << std::endl;
    }
  }

  std::cout << total_loss << std::endl;

  std::cout << correct / static_cast<float>(total * batch_size) << std::endl;

  return 0;
}*/