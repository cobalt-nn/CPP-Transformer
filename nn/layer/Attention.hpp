#pragma once

#include <iostream>
#include <optional>
#include <string>
#include <cstring>
#include <vector>
#include <random>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include "nlohmann/json.hpp"
#include "nn/tensor/Tensor.hpp"
#include "nn/layer/ILayer.hpp"
#include "nn/layer/Linear.hpp"
#include "nn/io/BinaryIO.hpp"

namespace cobalt_715::nn::layer{

//self attention
struct Attention : ILayer{
  Attention(int64_t in,
            int64_t num_heads,
            int64_t d_qk,
            int64_t d_v,
            bool causal_mask_bool = false)

    : in_size_(in),
      qkv_linear_(in,num_heads * (d_qk * 2 + d_v),true),
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
      d_scores_({1,1,1,1}),
      causal_mask_bool_(causal_mask_bool){}

  Attention(int64_t in,
            int64_t num_heads,
            int64_t d_qk,
            int64_t d_v,bool causal_mask_bool,
            int64_t kv_cache_max_len,
            int64_t kv_cache_stride_len = -1)

    : Attention(in,num_heads,d_qk,d_v,causal_mask_bool){

    cache_.emplace(d_qk * num_heads,d_v * num_heads,kv_cache_max_len,kv_cache_stride_len);
  }

  struct KVCache{
    KVCache(int64_t d_k,int64_t d_v,int64_t max_len,int64_t stride_len)//stride_len_の分だけまとめて余裕を開ける
      : K_({max_len,d_k}),
        V_({max_len,d_v}),
        max_len_(max_len),
        stride_len_((stride_len < 0) ? max_len / 4:stride_len){

          if(stride_len > max_len) throw std::runtime_error("KVCache");
        }

    void add(const tensor::ConstMatrixView k_view,const tensor::ConstMatrixView v_view){
      if(k_view.rows() != v_view.rows()) throw std::runtime_error("Attention::KVCache k_view.rows() != v_view.rows()");

      if(k_view.cols() != K_.dim(1)) throw std::runtime_error("Attention::KVCache k_view.cols() mismatch");
      if(v_view.cols() != V_.dim(1)) throw std::runtime_error("Attention::KVCache v_view.cols() mismatch");

      if(current_len_ + k_view.rows() > max_len_){
        if(stride_len_ == 0) throw std::runtime_error("Attention::KVCache KV cache overflow");

        int64_t overflow = current_len_ + k_view.rows() - max_len_;

        current_len_ -= stride_len_ + overflow;

        std::memmove(K_.data(),K_.data() + (stride_len_ + overflow) * K_.dim(1),sizeof(float) * current_len_ * K_.dim(1));
        std::memmove(V_.data(),V_.data() + (stride_len_ + overflow) * V_.dim(1),sizeof(float) * current_len_ * V_.dim(1));
      }

      tensor::MatrixView this_k_view = K_.unsafe_matrix_view(k_view.rows(),k_view.cols(),k_view.cols(),1,current_len_ * k_view.cols());

      //this_k_view += k_view;

      for(int64_t row = 0;row < k_view.rows();row++){
        for(int64_t col = 0;col < k_view.cols();col++){
          this_k_view.at(row,col) = k_view.at(row,col);
        }
      }

      tensor::MatrixView this_v_view = V_.unsafe_matrix_view(v_view.rows(),v_view.cols(),v_view.cols(),1,current_len_ * v_view.cols());

      //this_v_view += v_view;

      for(int64_t row = 0;row < v_view.rows();row++){
        for(int64_t col = 0;col < v_view.cols();col++){
          this_v_view.at(row,col) = v_view.at(row,col);
        }
      }

      //std::cout << "current_len_" << current_len_ << std::endl;
      //std::cout << "cache k" << this_k_view.to_string() << std::endl;
      //std::cout << "cache v" << this_v_view.to_string() << std::endl;

      current_len_ += k_view.rows();
    }

    const tensor::ConstMatrixView get_k_view() const{
      return K_.unsafe_matrix_view(current_len_,K_.dim(1),K_.dim(1),1,0);
    }

    const tensor::ConstMatrixView get_v_view() const{
      return V_.unsafe_matrix_view(current_len_,V_.dim(1),V_.dim(1),1,0);
    }

    int64_t get_current_len() const{
      return current_len_;
    }

    int64_t get_max_len() const{
      return max_len_;
    }

    int64_t get_stride_len() const{
      return stride_len_;
    }

    void reset(){
      current_len_ = 0;
    }

  private:
    tensor::Tensor K_;
    tensor::Tensor V_;
    int64_t current_len_ = 0;
    const int64_t max_len_ = 0;
    const int64_t stride_len_ = 0;
  };

  const int64_t in_size_;

  const bool causal_mask_bool_;

  std::optional<KVCache> cache_ = std::nullopt;//forward()の引数trainingがfalseかつKVCacheが存在するときキャッシュする

  const tensor::Tensor *input_ptr_ = nullptr;

  Linear qkv_linear_;//Q,K,Vを計算するアフィン変換層

  const tensor::Tensor *qkv_;//qkv_linear_の出力を持っておく

  const int64_t num_heads_;//ヘッド数
  const int64_t d_qk_;//qとkのheadあたりの列数
  const int64_t d_v_;//vのheadあたりの列数

  const int64_t q_offset_ = 0;//0
  const int64_t k_offset_;//d_qk_ * num_heads_
  const int64_t v_offset_;//2 * d_qk_ * num_heads_

  tensor::Tensor scores_;//QKt / √d
  tensor::Tensor weights_;//softmax(scores_)
  tensor::Tensor output_;//weights_ @ V

  std::vector<float> max_weights_;
  std::vector<double> sum_weights_;
  std::vector<int64_t> causal_mask_col_ends_;//マスクするときscores_の各行の終端を保持する

  const tensor::Tensor& forward(const tensor::Tensor& input,bool training=true) override{
    if(input.rank() != 3) throw std::runtime_error("Attention: input must be 3D");

    input_ptr_ = &input;

    qkv_ = &qkv_linear_.forward(input,training);

    //std::cout << qkv_->to_string() << std::endl;

    const int64_t big_qkv_rows = qkv_->dim(0) * qkv_->dim(1);//TensorViewが無いため

    tensor::ConstMatrixView big_q_view = qkv_->unsafe_matrix_view(big_qkv_rows,d_qk_ * num_heads_,qkv_->dim(2),1,q_offset_);
    tensor::ConstMatrixView big_k_view = qkv_->unsafe_matrix_view(big_qkv_rows,d_qk_ * num_heads_,qkv_->dim(2),1,k_offset_);
    tensor::ConstMatrixView big_v_view = qkv_->unsafe_matrix_view(big_qkv_rows,d_v_ * num_heads_,qkv_->dim(2),1,v_offset_);

    if(cache_ && !training){
      if(input.dim(0) != 1) throw std::runtime_error("Attention: KV cache is true && training is false batch must be 1");

      cache_.value().add(big_k_view,big_v_view);

      big_k_view = cache_.value().get_k_view();
      big_v_view = cache_.value().get_v_view();
    }

    const int64_t Tq = input.dim(1);//Qの系列長
    const int64_t Tkv = (cache_ && !training) ? cache_.value().get_current_len() : Tq;//K,Vの系列長

    //std::cout << "big_q_view" << big_q_view.to_string() << std::endl;
    //std::cout << "big_k_view" << big_k_view.to_string() << std::endl;
    //std::cout << "big_v_view" << big_v_view.to_string() << std::endl;

    forward_ensure_shape(Tq,Tkv);

    compute_scores(big_q_view,big_k_view,training,Tq,Tkv);

    //std::cout << "scores" << scores_.to_string() << std::endl;

    compute_weights(training);

    //weights_ = scores_;

    //std::cout << "weights" << weights_.to_string() << std::endl;

    compute_output(big_v_view,Tkv);

    //std::cout << "output" << output_.to_string() << std::endl;

    /*float mx = -1e30f;

    for(int i = 0;i < scores_.numel();i++){
      mx = std::max(mx,std::abs(scores_.data()[i]));
    }

    static int64_t count = 0;
    if(count % 1024 == 0){
      //std::cout << "scores_mx" << mx << std::endl;
      //std::cout << "scores" << scores_.to_string() << std::endl;
      //std::cout << "weights" << weights_.to_string() << std::endl;
    }
    count++;*/

    return output_;
  }

  //QKt / √d
  void compute_scores(const tensor::ConstMatrixView big_q_view,const tensor::ConstMatrixView big_k_view,bool training,const int64_t Tq,const int64_t Tkv){
    const float rec_sqrt_d = 1.0f / std::sqrt(d_qk_);

    tensor::Tensor small_q({Tq,d_qk_});
    tensor::Tensor small_k({Tkv,d_qk_});

    tensor::MatrixView q_view = small_q.flatten_matrix_view();
    tensor::MatrixView k_view = small_k.flatten_matrix_view();

    std::vector<int64_t> scores_dim = {0,0};

    for(int64_t batch = 0;batch < input_ptr_->shape()[0];batch++){
      scores_dim[0] = batch;

      //size_t index = batch * qkv_->stride()[0];
      for(int64_t head = 0;head < num_heads_;head++){
        scores_dim[1] = head;

        tensor::MatrixView scores_view = scores_.as_matrix_view(scores_dim);

        const tensor::ConstMatrixView q_view1 = big_q_view.block(Tq,d_qk_,batch * Tq,head * d_qk_);

        const tensor::ConstMatrixView k_view1 = big_k_view.block(Tkv,d_qk_,batch * Tkv,head * d_qk_);

        apply_RoPE(q_view,q_view1,training);
        apply_RoPE(k_view,k_view1,training);

        static int64_t count = 0;
        //if(count % 512 == 0){
          //std::cout << "q_view" << q_view.to_string() << std::endl;
          //std::cout << "k_view" << k_view.to_string() << std::endl;
          //std::cout << "scores_view" << scores_view.to_string() << std::endl;
        //}
        count++;

        //std::cout << "before matmul_impl(rec_sqrt_d,q_view,k_view.t(),0,scores_view)" << std::endl;

        tensor::MatrixView::matmul_impl(rec_sqrt_d,q_view,k_view.t(),0.0f,scores_view);

        //std::cout << "compute_scores end" << std::endl;
      }
    }
  }

  void apply_RoPE(tensor::MatrixView &q,const tensor::ConstMatrixView &base_q,bool training){
    const int32_t base = 10000;

    for(int64_t row = 0;row < q.rows();row++){
      for(int64_t col = 0;col < q.cols();col += 2){
        const int64_t pos = (cache_ && !training) ? cache_.value().get_current_len() - q.rows() + row:row;

        const float theta = pos / std::pow(base,static_cast<float>(col) / q.cols());

        const float x = base_q.at(row,col);
        const float y = (col + 1 < q.cols()) ? base_q.at(row,col + 1):0.0f;

        const float sin_theta = std::sin(theta);
        const float cos_theta = std::cos(theta);

        q.at(row,col) = x * cos_theta - y * sin_theta;

        if(col + 1 < q.cols()) q.at(row,col + 1) = x * sin_theta + y * cos_theta;
      }
    }
  }

  //softmax(scores_)
  void compute_weights(bool training){
    if(causal_mask_bool_){
      for(int row = 0;row < weights_.dim(2);row++){
        if(training){
          causal_mask_col_ends_[row] = row + 1;
        }else{
          causal_mask_col_ends_[row] = cache_.value().get_current_len() - weights_.dim(2) + row + 1;
        }
      }
    }else{
      std::fill(causal_mask_col_ends_.begin(),causal_mask_col_ends_.end(),scores_.dim(3));
    }

    for(int64_t batch = 0;batch < weights_.dim(0);batch++){
      for(int64_t head = 0;head < weights_.dim(1);head++){
        tensor::MatrixView scores_view = scores_.as_matrix_view({batch,head});//行連続
        tensor::MatrixView weights_view = weights_.as_matrix_view({batch,head});//行連続

        //std::cout << "weights_view" << weights_view.to_string() << std::endl;

        //最大の要素を求める
        for(int64_t row = 0;row < weights_.dim(2);row++){
          max_weights_[row] = *std::max_element(&scores_view.at(row,0),&scores_view.at(row,causal_mask_col_ends_[row]));
        }

        //exp(col - max)の合計値を求める
        for(int64_t row = 0;row < weights_.dim(2);row++){
          sum_weights_[row] = 0;
          for(int64_t col = 0;col < causal_mask_col_ends_[row];col++){
            sum_weights_[row] += std::exp(scores_view.at(row,col) - max_weights_[row]);
          }
        }

        //std::cout << "weights_view" << weights_view.to_string() << std::endl;

        //softmaxを求める
        for(int64_t row = 0;row < weights_.dim(2);row++){
          for(int64_t col = 0;col < causal_mask_col_ends_[row];col++){
            weights_view.at(row,col) = static_cast<float>(std::exp(scores_view.at(row,col) - max_weights_[row]) / sum_weights_[row]);
          }
        }

        //std::cout << "weights_view" << weights_view.to_string() << std::endl;
      }
    }
  }

  //weights_ @ V
  void compute_output(const tensor::ConstMatrixView big_v_view,const int64_t Tkv){
    std::vector<int64_t> weights_dim = {0,0};

    for(int64_t batch = 0;batch < input_ptr_->shape()[0];batch++){
      weights_dim[0] = batch;

      //size_t v_index = batch * qkv_->stride()[0];

      for(int64_t head = 0;head < num_heads_;head++){
        weights_dim[1] = head;
        const tensor::ConstMatrixView weights_view = weights_.as_matrix_view(weights_dim);

        const tensor::ConstMatrixView v_view = big_v_view.block(Tkv,d_v_,batch * Tkv,head * d_v_);

        tensor::MatrixView output_view = output_.unsafe_matrix_view(output_.dim(1),d_v_,output_.dim(2),1,output_.stride()[0] * batch + head * d_v_);

        /*static int64_t count = 0;
        if(count % 1024 == 0){
          //std::cout << "v_view" << v_view.to_string();
          //std::cout << "weights_view" << weights_view.to_string();
          //std::cout << "output_view" << output_view.to_string();
        }
        count++;*/

        tensor::MatrixView::matmul(weights_view,v_view,output_view);
      }
    }
  }

  //条件によりcontext_,weights_,output_,sum_weights_,max_weights_再確保
  void forward_ensure_shape(const int64_t Tq,const int64_t Tkv){
    if(Tkv != scores_.shape()[3] || scores_.shape()[2] != Tq || input_ptr_->shape()[0] != scores_.shape()[0] || scores_.shape()[1] != num_heads_){
      scores_ = tensor::Tensor({input_ptr_->shape()[0],num_heads_,Tq,Tkv});
      weights_ = tensor::Tensor({input_ptr_->shape()[0],num_heads_,Tq,Tkv});
    }else{
      std::fill(weights_.data(),weights_.data() + weights_.numel(),0.0f);
    }

    if(output_.shape()[1] != input_ptr_->shape()[1] || output_.shape()[0] != input_ptr_->shape()[0] || output_.shape()[2] != num_heads_ * d_v_){
      output_ = tensor::Tensor({input_ptr_->shape()[0],input_ptr_->shape()[1],num_heads_ * d_v_});
    }

    if(sum_weights_.size() != scores_.dim(2) || sum_weights_.size() != max_weights_.size() || sum_weights_.size() != causal_mask_col_ends_.size()){
      sum_weights_ = std::vector<double>(scores_.dim(2));
      max_weights_ = std::vector<float>(scores_.dim(2));
      causal_mask_col_ends_ = std::vector<int64_t>(scores_.dim(2));
    }
  }

  tensor::Tensor d_qkv_;
  tensor::Tensor d_weights_;
  tensor::Tensor d_scores_;

  std::vector<double> sum_d_weights_;

  const tensor::Tensor& backward(const tensor::Tensor& grad_output) override{
    backward_ensure_shape();

    const int64_t big_qkv_rows = qkv_->dim(0) * qkv_->dim(1);//TensorViewが無いため

    const tensor::ConstMatrixView big_q_view = qkv_->unsafe_matrix_view(big_qkv_rows,d_qk_ * num_heads_,qkv_->dim(2),1,q_offset_);
    const tensor::ConstMatrixView big_k_view = qkv_->unsafe_matrix_view(big_qkv_rows,d_qk_ * num_heads_,qkv_->dim(2),1,k_offset_);
    const tensor::ConstMatrixView big_v_view = qkv_->unsafe_matrix_view(big_qkv_rows,d_v_ * num_heads_,qkv_->dim(2),1,v_offset_);

    tensor::MatrixView big_d_q_view = d_qkv_.unsafe_matrix_view(big_qkv_rows,d_qk_ * num_heads_,d_qkv_.dim(2),1,q_offset_);
    tensor::MatrixView big_d_k_view = d_qkv_.unsafe_matrix_view(big_qkv_rows,d_qk_ * num_heads_,d_qkv_.dim(2),1,k_offset_);
    tensor::MatrixView big_d_v_view = d_qkv_.unsafe_matrix_view(big_qkv_rows,d_v_ * num_heads_,d_qkv_.dim(2),1,v_offset_);

    const int64_t Tqkv = input_ptr_->dim(1);//Q,K,Vの系列長

    d_compute_output(grad_output,big_v_view,big_d_v_view,Tqkv);

/*for(int i = 0;i < d_weights_.numel();i++){
  if(!std::isfinite(d_weights_.data()[i])){
    std::cout << "d_weights nan\n";
    std::abort();
  }
}*/

    d_compute_weights();

    //d_scores_ = d_weights_;

/*for(int i = 0;i < d_scores_.numel();i++){
  if(!std::isfinite(d_scores_.data()[i])){
    std::cout << "d_scores nan\n";
    std::abort();
  }
}*/

    d_compute_scores(big_q_view,big_k_view,big_d_q_view,big_d_k_view,Tqkv);

    return qkv_linear_.backward(d_qkv_);
  }

  void d_compute_output(const tensor::Tensor& grad_output,const tensor::ConstMatrixView big_v_view,tensor::MatrixView big_d_v_view,const int64_t Tqkv){
    std::vector<int64_t> weights_dim = {0,0};

    for(int64_t batch = 0;batch < input_ptr_->shape()[0];batch++){
      weights_dim[0] = batch;

      for(int64_t head = 0;head < num_heads_;head++){
        weights_dim[1] = head;

        const tensor::ConstMatrixView w_view = weights_.as_matrix_view(weights_dim);

        tensor::MatrixView dw_view = d_weights_.as_matrix_view(weights_dim);

        const tensor::ConstMatrixView v_view = big_v_view.block(Tqkv,d_v_,batch * Tqkv,head * d_v_);

        tensor::MatrixView d_v_view = big_d_v_view.block(Tqkv,d_v_,batch * Tqkv,head * d_v_);

        const tensor::ConstMatrixView grad_output_view = grad_output.unsafe_matrix_view(grad_output.dim(1),d_v_,grad_output.dim(2),1,grad_output.stride()[0] * batch + head * d_v_);

        tensor::MatrixView::matmul(w_view.t(),grad_output_view,d_v_view);

        tensor::MatrixView::matmul(grad_output_view,v_view.t(),dw_view);

        /*static int64_t count = 0;
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
}*/
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

  void d_compute_scores(const tensor::ConstMatrixView big_q_view,const tensor::ConstMatrixView big_k_view,tensor::MatrixView big_d_q_view,tensor::MatrixView big_d_k_view,const int64_t Tqkv){
    const float rec_sqrt_d = 1.0f / std::sqrt(d_qk_);

    std::vector<int64_t> d_scores_dim = {0,0};

    tensor::Tensor small_q({Tqkv,d_qk_});
    tensor::Tensor small_k({Tqkv,d_qk_});

    tensor::MatrixView q_view = small_q.flatten_matrix_view();
    tensor::MatrixView k_view = small_k.flatten_matrix_view();

    for(int64_t batch = 0;batch < input_ptr_->shape()[0];batch++){
      d_scores_dim[0] = batch;

      for(int64_t head = 0;head < num_heads_;head++){
        d_scores_dim[1] = head;

        tensor::MatrixView d_scores_view = d_scores_.as_matrix_view(d_scores_dim);

        const tensor::ConstMatrixView q_view1 = big_q_view.block(Tqkv,d_qk_,batch * Tqkv,head * d_qk_);

        const tensor::ConstMatrixView k_view1 = big_k_view.block(Tqkv,d_qk_,batch * Tqkv,head * d_qk_);

        apply_RoPE(q_view,q_view1,true);
        apply_RoPE(k_view,k_view1,true);

        tensor::MatrixView d_q_view = big_d_q_view.block(Tqkv,d_qk_,batch * Tqkv,head * d_qk_);

        tensor::MatrixView d_k_view = big_d_k_view.block(Tqkv,d_qk_,batch * Tqkv,head * d_qk_);

        tensor::MatrixView::matmul_impl(rec_sqrt_d,d_scores_view.t(),q_view,0.0f,d_k_view);

        tensor::MatrixView::matmul_impl(rec_sqrt_d,d_scores_view,k_view,0.0f,d_q_view);

        d_apply_RoPE(d_k_view);
        d_apply_RoPE(d_q_view);
      }
    }
  }

  void d_apply_RoPE(tensor::MatrixView &matrix){
    const int32_t base = 10000;

    for(int64_t row = 0;row < matrix.rows();row++){
      for(int64_t col = 0;col < matrix.cols();col += 2){
        const int64_t pos = row;

        const float theta = -pos / std::pow(base,static_cast<float>(col) / matrix.cols());

        const float x = matrix.at(row,col);
        const float y = (col + 1 < matrix.cols()) ? matrix.at(row,col + 1):0.0f;

        const float sin_theta = std::sin(theta);
        const float cos_theta = std::cos(theta);

        matrix.at(row,col) = x * cos_theta - y * sin_theta;

        if(col + 1 < matrix.cols()) matrix.at(row,col + 1) = x * sin_theta + y * cos_theta;
      }
    }
  }

  void backward_ensure_shape(){
    if(d_qkv_.shape() != qkv_->shape()){
      d_qkv_ = tensor::Tensor(qkv_->shape());
    }

    if(d_weights_.shape() != weights_.shape()){
      d_weights_ = tensor::Tensor(weights_.shape());
      d_scores_ = tensor::Tensor(weights_.shape());
    }

    if(sum_d_weights_.size() != d_weights_.numel() / d_weights_.dim(3)){
      sum_d_weights_ = std::vector<double>(d_weights_.numel() / d_weights_.dim(3));
    }
  }

  void step(float lr,int batch_size=64) override{
    qkv_linear_.step(lr,batch_size);
  }

  void zero_grad() override{
    qkv_linear_.zero_grad();
  }

  void reset() override{
    if(cache_){
      cache_.value().reset();
    }
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
    nlohmann::ordered_json j;

    j["layer_type"] = get_type();
    j["in"] = in_size_;
    j["num_heads"] = num_heads_;
    j["d_qk"] = d_qk_;
    j["d_v"] = d_v_;
    j["kv_cache"] = (cache_) ? "true":"false";

    if(cache_){
      j["kv_cache_max_length"] = cache_.value().get_max_len();
      j["kv_cache_stride_length"] = cache_.value().get_stride_len();
    }

    j["causal_mask"] = (causal_mask_bool_) ? "true":"false";
    j["qkv_linear"] = qkv_linear_.to_json();

    return j;
  }

  void save(std::ostream &os) const override{
    qkv_linear_.save(os);
  }

  void load(const nlohmann::ordered_json &json,std::istream &is) override{
    if(json.at("layer_type") != get_type()){
      throw std::runtime_error("Attention::load type mismatch");
    }

    qkv_linear_.load(json.at("qkv_linear"),is);
  }

  void random_init(std::mt19937 &gen) override{
    qkv_linear_.random_init(gen);
  }
};

}//namespace cobalt_715::nn::layer