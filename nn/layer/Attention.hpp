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
    : qkv_linear_(in,num_heads * (d_qk * 2 + d_v),true),
      num_heads_(num_heads),
      d_qk_(d_qk),
      d_v_(d_v),
      k_offset_(d_qk * num_heads),
      v_offset_(2 * d_qk * num_heads),
      scores_({1,1,1,1}),
      weights_({1,1,1,1}),
      output_({1,1,1}),
      d_qkv_({1,1,1}),
      d_weights_({1,1,1,1}),
      d_scores_({1,1,1,1}){}

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

    //std::cout << qkv_->to_string() << std::endl;

    const tensor::ConstMatrixView big_q_view = qkv_->unsafe_matrix_view(qkv_->dim(0) * qkv_->dim(1),d_qk_ * num_heads_,qkv_->dim(2),1,0);
    const tensor::ConstMatrixView big_k_view = qkv_->unsafe_matrix_view(qkv_->dim(0) * qkv_->dim(1),d_qk_ * num_heads_,qkv_->dim(2),1,k_offset_);
    const tensor::ConstMatrixView big_v_view = qkv_->unsafe_matrix_view(qkv_->dim(0) * qkv_->dim(1),d_v_ * num_heads_,qkv_->dim(2),1,v_offset_);

    //std::cout << big_q_view.to_string() << std::endl;
    //std::cout << big_k_view.to_string() << std::endl;
    //std::cout << big_v_view.to_string() << std::endl;

    forward_ensure_shape();

    compute_scores(big_q_view,big_k_view);

    //std::cout << "scores" << scores_.to_string() << std::endl;

    compute_weights();

    //weights_ = scores_;

    //std::cout << "weights" << weights_.to_string() << std::endl;

    compute_output(big_v_view);

    //std::cout << "output" << output_.to_string() << std::endl;

    float mx = -1e30f;

    for(int i = 0;i < scores_.numel();i++){
      mx = std::max(mx,std::abs(scores_.data()[i]));
    }

    static int64_t count = 0;
    if(count % 1024 == 0){
      std::cout << "scores_mx" << mx << std::endl;
      //std::cout << "scores" << scores_.to_string() << std::endl;
      //std::cout << "weights" << weights_.to_string() << std::endl;
    }
    count++;

    return output_;
  }

  //QKt / √d
  void compute_scores(const tensor::ConstMatrixView big_q_view,const tensor::ConstMatrixView big_k_view){
    const float rec_sqrt_d = 1.0f / std::sqrt(d_qk_);

    std::vector<int64_t> scores_dim = {0,0};

    for(int64_t batch = 0;batch < input_ptr_->shape()[0];batch++){
      scores_dim[0] = batch;

      //size_t index = batch * qkv_->stride()[0];
      for(int64_t head = 0;head < num_heads_;head++){
        scores_dim[1] = head;

        tensor::MatrixView scores_view = scores_.as_matrix_view(scores_dim);

        const tensor::ConstMatrixView q_view = big_q_view.block(big_q_view.numel() / input_ptr_->dim(0) / big_q_view.cols(),d_qk_,
          batch * big_q_view.numel() / big_q_view.cols() / input_ptr_->dim(0),head * d_qk_);

        const tensor::ConstMatrixView k_view = big_k_view.block(big_k_view.numel() / input_ptr_->dim(0) / big_k_view.cols(),d_qk_,
          batch * big_k_view.numel() / big_k_view.cols() / input_ptr_->dim(0),head * d_qk_);

        //const tensor::ConstMatrixView q_view = qkv_->unsafe_matrix_view(qkv_->dim(1),d_qk_,qkv_->dim(2),1,index + head * d_qk_);
        //const tensor::ConstMatrixView k_view = qkv_->unsafe_matrix_view(qkv_->dim(1),d_qk_,qkv_->dim(2),1,index + head * d_qk_ + k_offset_);

        static int64_t count = 0;
        if(count % 512 == 0){
          //std::cout << "q_view" << q_view.to_string() << std::endl;
          //std::cout << "k_view" << k_view.to_string() << std::endl;
          //std::cout << "scores_view" << scores_view.to_string() << std::endl;
        }
        count++;

        tensor::MatrixView::matmul_impl(rec_sqrt_d,q_view,k_view.t(),0,scores_view);
      }
    }
  }

  //softmax(scores_)
  void compute_weights(){
    const float *sd = scores_.data();
    float *mwd = max_weights_.data();

    //最大の要素を求める
    for(int64_t big_row = 0;big_row < scores_.numel() / scores_.dim(3);big_row++){
      mwd[big_row] = *std::max_element(sd + big_row * scores_.dim(3),sd + (big_row + 1) * scores_.dim(3));
    }

    double *swd = sum_weights_.data();

    //exp(j - max)の合計値を求める
    for(int64_t big_row = 0;big_row < scores_.numel() / scores_.dim(3);big_row++){
      swd[big_row] = 0;
      for(int64_t col = 0;col < scores_.dim(3);col++){
        swd[big_row] += std::exp(scores_.data()[big_row * scores_.dim(3) + col] - mwd[big_row]);
      }
    }

    float *wd = weights_.data();

    //exp(j - max)の合計値を求める
    for(int64_t big_row = 0;big_row < scores_.numel() / scores_.dim(3);big_row++){
      for(int64_t col = 0;col < scores_.dim(3);col++){
        wd[big_row * weights_.dim(3) + col] = static_cast<float>(std::exp(sd[big_row * weights_.dim(3) + col] - mwd[big_row]) / swd[big_row]);
      }
    }
  }

  //weights_ @ V
  void compute_output(const tensor::ConstMatrixView big_v_view){
    std::vector<int64_t> weights_dim = {0,0};

    for(int64_t batch = 0;batch < input_ptr_->shape()[0];batch++){
      weights_dim[0] = batch;

      //size_t v_index = batch * qkv_->stride()[0];

      for(int64_t head = 0;head < num_heads_;head++){
        weights_dim[1] = head;
        const tensor::ConstMatrixView weights_view = weights_.as_matrix_view(weights_dim);

        const tensor::ConstMatrixView v_view = big_v_view.block(big_v_view.numel() / input_ptr_->dim(0) / big_v_view.cols(),d_v_,
          batch * big_v_view.numel() / big_v_view.cols() / input_ptr_->dim(0),head * d_v_);

        tensor::MatrixView output_view = output_.unsafe_matrix_view(output_.dim(1),d_v_,output_.dim(2),1,output_.stride()[0] * batch + head * d_v_);

        tensor::MatrixView::matmul(weights_view,v_view,output_view);

        static int64_t count = 0;
        if(count % 1024 == 0){
          //std::cout << "v_view" << v_view.to_string();
          //std::cout << "weights_view" << weights_view.to_string();
        }
        count++;
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
      std::fill(scores_.data(),scores_.data() + scores_.numel(),0.0f);
    }

    if(output_.shape()[1] != input_ptr_->shape()[1] || output_.shape()[0] != input_ptr_->shape()[0] || output_.shape()[2] != num_heads_ * d_v_){
      output_ = tensor::Tensor({input_ptr_->shape()[0],input_ptr_->shape()[1],num_heads_ * d_v_});
    }else{
      std::fill(output_.data(),output_.data() + output_.numel(),0.0f);
    }

    if(sum_weights_.size() != scores_.numel() / scores_.dim(3) || sum_weights_.size() != max_weights_.size()){
      sum_weights_ = std::vector<double>(scores_.numel() / scores_.dim(3));
      max_weights_ = std::vector<float>(scores_.numel() / scores_.dim(3));
    }else{
      std::fill(sum_weights_.begin(),sum_weights_.end(),0.0f);
      std::fill(max_weights_.begin(),max_weights_.end(),0.0f);
    }
  }

  tensor::Tensor d_qkv_;
  tensor::Tensor d_weights_;
  tensor::Tensor d_scores_;

  std::vector<double> sum_d_weights_;

  const tensor::Tensor& backward(const tensor::Tensor& grad_output) override{
    backward_ensure_shape();

    const tensor::ConstMatrixView big_q_view = qkv_->unsafe_matrix_view(qkv_->dim(0) * qkv_->dim(1),d_qk_ * num_heads_,qkv_->dim(2),1,0);
    const tensor::ConstMatrixView big_k_view = qkv_->unsafe_matrix_view(qkv_->dim(0) * qkv_->dim(1),d_qk_ * num_heads_,qkv_->dim(2),1,k_offset_);
    const tensor::ConstMatrixView big_v_view = qkv_->unsafe_matrix_view(qkv_->dim(0) * qkv_->dim(1),d_v_ * num_heads_,qkv_->dim(2),1,v_offset_);

    tensor::MatrixView big_d_q_view = d_qkv_.unsafe_matrix_view(d_qkv_.dim(0) * d_qkv_.dim(1),d_qk_ * num_heads_,d_qkv_.dim(2),1,0);
    tensor::MatrixView big_d_k_view = d_qkv_.unsafe_matrix_view(d_qkv_.dim(0) * d_qkv_.dim(1),d_qk_ * num_heads_,d_qkv_.dim(2),1,k_offset_);
    tensor::MatrixView big_d_v_view = d_qkv_.unsafe_matrix_view(d_qkv_.dim(0) * d_qkv_.dim(1),d_v_ * num_heads_,d_qkv_.dim(2),1,v_offset_);

    d_compute_output(grad_output,big_v_view,big_d_v_view);

for(int i = 0;i < d_weights_.numel();i++){
  if(!std::isfinite(d_weights_.data()[i])){
    std::cout << "d_weights nan\n";
    std::abort();
  }
}

    d_compute_weights();

    //d_scores_ = d_weights_;

for(int i = 0;i < d_scores_.numel();i++){
  if(!std::isfinite(d_scores_.data()[i])){
    std::cout << "d_scores nan\n";
    std::abort();
  }
}

    d_compute_scores(big_q_view,big_k_view,big_d_q_view,big_d_k_view);

    return qkv_linear_.backward(d_qkv_);
  }

  void d_compute_output(const tensor::Tensor& grad_output,const tensor::ConstMatrixView big_v_view,tensor::MatrixView big_d_v_view){
    std::vector<int64_t> weights_dim = {0,0};

    for(int64_t batch = 0;batch < input_ptr_->shape()[0];batch++){
      weights_dim[0] = batch;

      for(int64_t head = 0;head < num_heads_;head++){
        weights_dim[1] = head;

        const tensor::ConstMatrixView w_view = weights_.as_matrix_view(weights_dim);

        tensor::MatrixView dw_view = d_weights_.as_matrix_view(weights_dim);

        const tensor::ConstMatrixView v_view = big_v_view.block(big_v_view.numel() / input_ptr_->dim(0) / big_v_view.cols(),d_v_,
          batch * big_v_view.numel() / big_v_view.cols() / input_ptr_->dim(0),head * d_v_);

        tensor::MatrixView d_v_view = big_d_v_view.block(big_d_v_view.numel() / input_ptr_->dim(0) / big_d_v_view.cols(),d_v_,
          batch * big_d_v_view.numel() / big_d_v_view.cols() / input_ptr_->dim(0),head * d_v_);

        const tensor::ConstMatrixView grad_output_view = grad_output.unsafe_matrix_view(grad_output.dim(1),d_v_,grad_output.dim(2),1,grad_output.stride()[0] * batch + head * d_v_);

        tensor::MatrixView::matmul(w_view.t(),grad_output_view,d_v_view);

        tensor::MatrixView::matmul(grad_output_view,v_view.t(),dw_view);

        static int64_t count = 0;
        if(count % 1024 == 0){
          //std::cout << "w_view" << w_view.to_string() << std::endl;
          //std::cout << "dw_view" << dw_view.to_string() << std::endl;
          //std::cout << "v_view" << v_view.to_string() << std::endl;
          //std::cout << "d_v_view" << d_v_view.to_string() << std::endl;
          //std::cout << "grad_output_view" << grad_output_view.to_string() << std::endl;
          //std::cout << "grad_output_view" << grad_output_view.to_string() << std::endl;
          //std::cout << "v_view" << v_view.to_string() << std::endl;
        }
        count++;

for(int i = 0;i < dw_view.rows();i++){
  for(int j = 0;j < dw_view.cols();j++){
    float x = dw_view.at(i,j);

    if(std::isnan(x) || std::isinf(x)){
      std::cout << "dw nan\n";

      std::cout << "grad_output_view" << grad_output_view.to_string() << std::endl;
      std::cout << "v_view" << v_view.to_string() << std::endl;

      throw std::runtime_error("nan");
    }
  }
}
      }
    }
  }

  void d_compute_weights(){
    //Σgradik * Zikを計算
    for(int64_t big_row = 0;big_row < d_weights_.numel() / d_weights_.dim(3);big_row++){
      sum_d_weights_[big_row] = 0;
      for(int64_t col = 0;col < d_weights_.dim(3);col++){
        sum_d_weights_[big_row] += static_cast<float>(d_weights_.data()[big_row * d_weights_.dim(3) + col] * weights_.data()[big_row * d_weights_.dim(3) + col]);
      }
    }

    //Zij * (gradij - sum)を計算
    for(int64_t big_row = 0;big_row < d_weights_.numel() / d_weights_.dim(3);big_row++){
      for(int64_t col = 0;col < d_weights_.dim(3);col++){
        d_scores_.data()[big_row * d_weights_.dim(3) + col] = weights_.data()[big_row * d_weights_.dim(3) + col] * (d_weights_.data()[big_row * d_weights_.dim(3) + col] - sum_d_weights_[big_row]);
      }
    }
  }

  void d_compute_scores(const tensor::ConstMatrixView big_q_view,const tensor::ConstMatrixView big_k_view,tensor::MatrixView big_d_q_view,tensor::MatrixView big_d_k_view){
    const float rec_sqrt_d = 1.0f / std::sqrt(d_qk_);

    std::vector<int64_t> d_scores_dim = {0,0};

    for(int64_t batch = 0;batch < input_ptr_->shape()[0];batch++){
      d_scores_dim[0] = batch;

      for(int64_t head = 0;head < num_heads_;head++){
        d_scores_dim[1] = head;

        tensor::MatrixView d_scores_view = d_scores_.as_matrix_view(d_scores_dim);

        const tensor::ConstMatrixView q_view = big_q_view.block(big_q_view.numel() / input_ptr_->dim(0) / big_q_view.cols(),d_qk_,
          batch * big_q_view.numel() / big_q_view.cols() / input_ptr_->dim(0),head * d_qk_);

        const tensor::ConstMatrixView k_view = big_k_view.block(big_k_view.numel() / input_ptr_->dim(0) / big_k_view.cols(),d_qk_,
          batch * big_k_view.numel() / big_k_view.cols() / input_ptr_->dim(0),head * d_qk_);

        tensor::MatrixView d_q_view = big_d_q_view.block(big_d_q_view.numel() / input_ptr_->dim(0) / big_d_q_view.cols(),d_qk_,
          batch * big_d_q_view.numel() / big_d_q_view.cols() / input_ptr_->dim(0),head * d_qk_);

        tensor::MatrixView d_k_view = big_d_k_view.block(big_d_k_view.numel() / input_ptr_->dim(0) / big_d_k_view.cols(),d_qk_,
          batch * big_d_k_view.numel() / big_d_k_view.cols() / input_ptr_->dim(0),head * d_qk_);

          tensor::MatrixView::matmul_impl(rec_sqrt_d,d_scores_view.t(),q_view,0.0f,d_k_view);

          tensor::MatrixView::matmul_impl(rec_sqrt_d,d_scores_view,k_view,0.0f,d_q_view);
      }
    }
  }

  void backward_ensure_shape(){
    if(d_qkv_.shape() != qkv_->shape()){
      d_qkv_ = tensor::Tensor(qkv_->shape());
    }else{
      std::fill(d_qkv_.data(),d_qkv_.data() + d_qkv_.numel(),0.0f);
    }


    if(d_weights_.shape() != weights_.shape()){
      d_weights_ = tensor::Tensor(weights_.shape());
      d_scores_ = tensor::Tensor(weights_.shape());
    }else{
      std::fill(d_weights_.data(),d_weights_.data() + d_weights_.numel(),0.0f);
      std::fill(d_scores_.data(),d_scores_.data() + d_scores_.numel(),0.0f);
    }

    if(sum_d_weights_.size() != d_weights_.numel() / d_weights_.dim(3)){
      sum_d_weights_ = std::vector<double>(d_weights_.numel() / d_weights_.dim(3));
    }else{
      std::fill(sum_d_weights_.begin(),sum_d_weights_.end(),0.0f);
    }
  }

  void step(float lr,int batch_size=64) override{
    qkv_linear_.step(lr,batch_size);
  }

  void zero_grad() override{
    qkv_linear_.zero_grad();
  }

  std::string get_type() const override{
    return "Attention";
  }

  std::string to_string() const{
    std::string s = get_type();
    s += "scores_\n" + scores_.to_string() + "\n";
    s += "weights_\n" + weights_.to_string() + "\n";
    s += "d_scores_\n" + d_scores_.to_string() + "\n";
    s += "d_weights_\n" + d_weights_.to_string() + "\n";
    return s;
  }

  nlohmann::ordered_json to_json() const override{
    return nlohmann::ordered_json();
  }

  void random_init(std::mt19937 &gen) override{
    qkv_linear_.random_init(gen);
  }
};

}//namespace cobalt_715::nn::layer