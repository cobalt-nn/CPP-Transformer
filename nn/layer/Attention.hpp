#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <cstdint>
#include <stdexcept>
#include "nlohmann/json.hpp"
#include "nn/tensor/Tensor.hpp"
#include "nn/layer/ILayer.hpp"
#include "nn/layer/Linear.hpp"

namespace cobalt_715::nn::layer{

//self attention
struct Attention : ILayer{
  Attention(int64_t in,int64_t num_heads,int64_t d_qk,int64_t d_v)
    : qkv_linear_(in,num_heads * (d_qk * 2 + d_v)),
      num_heads_(num_heads),
      d_qk_(d_qk),
      d_v_(d_v),
      k_offset_(d_qk * num_heads),
      v_offset_(2 * d_qk * num_heads),
      scores_({1,1,1,1}),
      weights_({1,1,1,1}),
      output_({1,1,1}){}

  const tensor::Tensor *input_ptr_;

  Linear qkv_linear_;//Q,K,Vを計算するアフィン変換層

  const tensor::Tensor *qkv_;//qkv_linear_の出力を持っておく

  const int64_t num_heads_;//ヘッド数
  const int64_t d_qk_;//qとkのheadあたりの列数
  const int64_t d_v_;//vのheadあたりの列数

  const int64_t k_offset_;//d_qk_ * num_heads_
  const int64_t v_offset_;//2 * d_qk_ * num_heads_

  tensor::Tensor scores_;//QKt / √d
  tensor::Tensor weights_;//softmax(scores_)
  tensor::Tensor output_;//weights_ @ V

  std::vector<float> max_weights_;
  std::vector<double> sum_weights_;

  const tensor::Tensor& forward(const tensor::Tensor& input,bool training=true) override{
    if(input.rank() != 3) throw std::runtime_error("Attention: input must be 3D");

    input_ptr_ = &input;

    qkv_ = &qkv_linear_.forward(input,training);

    forward_ensure_shape();

    compute_scores();

    std::cout << "scores" << scores_.to_string() << std::endl;

    compute_weights();

    std::cout << "weights" << weights_.to_string() << std::endl;

    compute_output();

    std::cout << "output" << output_.to_string() << std::endl;

    return output_;
  }

  //QKt / √d
  void compute_scores(){
    const float rec_sqrt_d = 1.0f / std::sqrt(d_qk_);

    std::vector<int64_t> scores_dim = {0,0};

    for(int64_t batch = 0;batch < qkv_->shape()[0];batch++){
      scores_dim[0] = batch;

      size_t index = batch * qkv_->stride()[0];
      for(int64_t head = 0;head < num_heads_;head++){
        scores_dim[1] = head;

        tensor::MatrixView scores_view = scores_.as_matrix_view(scores_dim);
        const tensor::ConstMatrixView q_view = qkv_->unsafe_matrix_view(qkv_->dim(1),d_qk_,qkv_->dim(2),1,index + head * d_qk_);
        const tensor::ConstMatrixView k_view = qkv_->unsafe_matrix_view(qkv_->dim(1),d_qk_,qkv_->dim(2),1,index + head * d_qk_ + k_offset_);

        tensor::MatrixView::matmul_impl(rec_sqrt_d,q_view,k_view.t(),1,scores_view);
      }
    } 
  }

  //softmax(scores_)
  void compute_weights(){
    const float *sd = scores_.data();
    float *mwd = max_weights_.data();

    //最大の要素を求める
    for(int64_t big_row = 0;big_row < scores_.numel() / scores_.dim(2);big_row++){
      mwd[big_row] = *std::max_element(sd + big_row * scores_.dim(2),sd + (big_row + 1) * scores_.dim(2));
    }

    double *swd = sum_weights_.data();

    //exp(j - max)の合計値を求める
    for(int64_t big_row = 0;big_row < scores_.numel() / scores_.dim(2);big_row++){
      swd[big_row] = 0;
      for(int64_t col = 0;col < scores_.dim(2);col++){
        swd[big_row] += std::exp(scores_.data()[big_row * scores_.dim(2) + col] - mwd[big_row]);
      }
    }

    float *wd = weights_.data();

    //exp(j - max)の合計値を求める
    for(int64_t big_row = 0;big_row < scores_.numel() / scores_.dim(2);big_row++){
      for(int64_t col = 0;col < scores_.dim(2);col++){
        wd[big_row * weights_.dim(3) + col] = static_cast<float>(std::exp(sd[big_row * weights_.dim(3) + col] - mwd[big_row]) / swd[big_row]);
      }
    }
  }

  //weights_ @ V
  void compute_output(){
    std::vector<int64_t> weights_dim = {0,0};

    for(int64_t batch = 0;batch < input_ptr_->shape()[0];batch++){
      weights_dim[0] = batch;
      size_t v_index = batch * qkv_->stride()[0];

      for(int64_t head = 0;head < num_heads_;head++){
        weights_dim[1] = head;
        const tensor::ConstMatrixView weights_view = weights_.as_matrix_view(weights_dim);
        const tensor::ConstMatrixView v_view = qkv_->unsafe_matrix_view(qkv_->dim(1),d_v_,qkv_->dim(2),1,v_index + head * d_v_ + v_offset_);
        tensor::MatrixView output_view = output_.unsafe_matrix_view(output_.dim(1),d_v_,output_.dim(2),1,output_.stride()[0] * batch + head * d_v_);

        tensor::MatrixView::matmul(weights_view,v_view,output_view);
      }
    }
  }

  //条件によりcontext_,weights_,output_,sum_weights_,max_weights_再確保
  void forward_ensure_shape(){
    if(scores_.shape()[2] != input_ptr_->shape()[1] || input_ptr_->shape()[0] != scores_.shape()[0] || scores_.shape()[1] != num_heads_ || scores_.shape()[2] != scores_.shape()[3]){
      scores_ = tensor::Tensor({input_ptr_->shape()[0],num_heads_,input_ptr_->shape()[1],input_ptr_->shape()[1]});
      weights_ = tensor::Tensor({input_ptr_->shape()[0],num_heads_,input_ptr_->shape()[1],input_ptr_->shape()[1]});
    }else{
      std::fill(weights_.data(),weights_.data() + weights_.numel(),0.0f);
    }

    if(output_.shape()[1] != input_ptr_->shape()[1] || output_.shape()[0] != input_ptr_->shape()[0] || output_.shape()[2] != num_heads_ * d_v_){
      output_ = tensor::Tensor({input_ptr_->shape()[0],input_ptr_->shape()[1],num_heads_ * d_v_});
    }

    if(sum_weights_.size() != scores_.numel() / scores_.dim(3) || sum_weights_.size() != max_weights_.size()){
      sum_weights_ = std::vector<double>(scores_.numel() / scores_.dim(3));
      max_weights_ = std::vector<float>(scores_.numel() / scores_.dim(3));
    }
  }

  const tensor::Tensor& backward(const tensor::Tensor& grad_output) override{
    return output_;
  }

  void step(float lr,int batch_size=64) override{}

  void zero_grad() override{}

  std::string get_type() const override{
    return "Attention";
  }

  std::string to_string() const{
    return get_type() + "::to_string() is undef";
  }

  nlohmann::ordered_json to_json() const override{
    return nlohmann::ordered_json();
  }

  void random_init(std::mt19937 &gen) override{
    qkv_linear_.random_init(gen);
  }
};

}//namespace cobalt_715::nn::layer