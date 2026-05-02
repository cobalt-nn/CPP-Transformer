#pragma once

#include <vector>
#include <span>
#include <cstdint>
#include <stdexcept>
#include <string>
#include "nn/backend/backend.hpp"
#include "vec.hpp"
#include "vec_cpu.hpp"

namespace cobalt_715::nn::linear{

//任意次元テンソル
class Tensor{
public:
  //コンストラクタ
  Tensor(const std::vector<int64_t> &shape) : shape_(shape),stride_(make_stride()){
    int64_t data_size = 1;
    for(int64_t i:shape_){
      data_size *= i;
    }
    data_.resize(data_size);
    check_invariants();
  }

  Tensor(const std::vector<int64_t> &shape,const std::vector<float> &data) : shape_(shape),stride_(make_stride()),data_(data){
    check_invariants();
  }

  inline float& at(const std::vector<int64_t>& a){
    check_index(a);
    //check_invariants();
    int64_t index = 0;
    for(int64_t i = 0;i < a.size();i++){
      index += a[i] * stride_[i];
    }
    return data_[index];
  }

  inline const float& at(const std::vector<int64_t>& a) const{
    check_index(a);
    //check_invariants();
    int64_t index = 0;
    for(int64_t i = 0;i < a.size();i++){
      index += a[i] * stride_[i];
    }
    return data_[index];
  }

  inline const std::vector<int64_t>& shape() const noexcept{
    return shape_;
  }

  inline const std::vector<int64_t>& stride() const noexcept{
    return stride_;
  }

  inline int64_t numel() const noexcept{
    return data_.size();
  }

  //ポインタを返しているため気を付けること
  inline float* data() noexcept{return data_.data();}
  inline const float* data() const noexcept{return data_.data();}

  inline std::span<float> span() noexcept{
    return std::span<float>(data_);
  }

  inline std::span<const float> span() const noexcept{
    return std::span<const float>(data_);
  }

  //out[i] = a[i] + b[i]
  inline static void add(const Tensor &a,const Tensor &b,Tensor &out){
    #ifndef NDEBUG
    if(a.shape() != out.shape() || a.shape() != b.shape()) throw std::invalid_argument("Tensor::add dimension mismatch");
    #endif
    vec::add(a.data(),b.data(),out.data(),a.numel());
  }

  inline Tensor operator+(const Tensor &rhs) const{
    Tensor out(shape_);
    add(*this,rhs,out);
    return out;
  }

  inline Tensor& operator+=(const Tensor &rhs){
    #ifndef NDEBUG
    if(this->shape() != rhs.shape()) throw std::invalid_argument("Tensor::+= dimension mismatch");
    #endif
    vec::add_alias_safe(data(),rhs.data(),data(),numel());
    return *this;
  }

  //out[i] = a[i] - b[i]
  inline static void sub(const Tensor &a,const Tensor &b,Tensor &out){
    #ifndef NDEBUG
    if(a.shape() != out.shape() || a.shape() != b.shape()) throw std::invalid_argument("Tensor::sub dimension mismatch");
    #endif
    vec::sub(a.data(),b.data(),out.data(),a.numel());
  }

  inline Tensor operator-(const Tensor &rhs) const{
    Tensor out(shape_);
    sub(*this,rhs,out);
    return out;
  }

  inline Tensor& operator-=(const Tensor &rhs){
    #ifndef NDEBUG
    if(this->shape() != rhs.shape()) throw std::invalid_argument("Tensor::-= dimension mismatch");
    #endif
    vec::sub_alias_safe(data(),rhs.data(),data(),numel());
    return *this;
  }

  //out[i] = a[i] * b[i]
  inline static void hadamard(const Tensor &a,const Tensor &b,Tensor &out){
    #ifndef NDEBUG
    if(a.shape() != out.shape() || a.shape() != b.shape()) throw std::invalid_argument("Tensor::hadamard dimension mismatch");
    #endif
    vec::mul(a.data(),b.data(),out.data(),a.numel());
  }

  inline void hadamard_(const Tensor &rhs){
    #ifndef NDEBUG
    if(shape_ != rhs.shape()) throw std::invalid_argument("Tensor::hadamard_ dimension mismatch");
    #endif
    vec::mul_alias_safe(data(),rhs.data(),data(),numel());
  }

  //out[i] = a[i] * c
  inline static void scale(const Tensor &a,const float c,Tensor &out){
    #ifndef NDEBUG
    if(a.shape() != out.shape()) throw std::invalid_argument("Tensor::hadamard dimension mismatch");
    #endif
    vec::scale(a.data(),c,out.data(),a.numel());
  }

  inline void scale_(const float c){
    vec::scale_alias_safe(data(),c,data(),numel());
  }

  std::string to_string(const int indent_size = 2) const{
    std::string s = "//shape = ";

    for(int64_t u:shape_) s += std::to_string(u) + " ";

    s += "\n";

    std::vector<int64_t> index(shape_.size(),0);

    int64_t dim = 0;

    to_string_recursive(s,index,dim,indent_size);

    return s;
  }

  void to_string_recursive(std::string &s,std::vector<int64_t> &index,int64_t dim,const int indent_size = 2) const{
    s += std::string(indent_size * dim,' ') + "{";

    for(int64_t i = 0;i < shape_.at(dim);i++){
      index.at(dim) = i;

      if(dim == shape_.size() - 1){
        s += " " + std::to_string(at(index)) + " ";
      }else{
        s += "\n";
        to_string_recursive(s,index,dim + 1,indent_size);
      }

      if(i != shape_.at(dim) - 1) s += ",";
    }

    if(dim == shape_.size() - 1){
      s += "}";
    }else{
      s += "\n" + std::string(indent_size * dim,' ') + "}";
    }
  }

private:
  std::vector<int64_t> shape_;//各次元の要素数
  std::vector<int64_t> stride_;//各次元にジャンプするまでに必要な数
  std::vector<float> data_;//data

  //shape_,stride_,data_の関係が合うかどうかとオーバーフローの確認をする
  void check_invariants() const{
    #ifndef NDEBUG
    //shape_とstride_のサイズが同じかどうか確認する
    if(shape_.size() != stride_.size()) throw std::logic_error("Tensor invariant violation: shape and stride size mismatch");

    //data_のサイズがshape_と合っているかとオーバーフローしていないか確認する
    int64_t data_size = 1;
    for(int64_t i = 0;i < shape_.size();i++){
      int64_t dim = shape_[i];

      //非負のみを許す
      if(dim < 0) throw std::invalid_argument("Tensor shape contains negative dimension");

      //オーバーフローを確認する
      if(dim > INT64_MAX / data_size) throw std::overflow_error("Tensor size overflow");

      data_size *= dim;
    }
    if(data_.size() != data_size) throw std::logic_error("Tensor invariant violation: data size mismatch");

    //stride_がshape_と合っているか確認する
    if(stride_ != make_stride()) throw std::logic_error("Tensor invariant violation: invalid stride");
    #endif
  }

  //indexが正しいか確認する
  void check_index(const std::vector<int64_t>& a) const{
    #ifndef NDEBUG
    if(a.size() != shape_.size()){
      throw std::invalid_argument(
        "Tensor index dimension mismatch: expected " +
        std::to_string(shape_.size()) +
        ", got " +
        std::to_string(a.size())
      );
    }

    for(int64_t i = 0;i < shape_.size();i++){
      if(a[i] >= shape_[i]){
        throw std::out_of_range(
          "Tensor index out of bounds at dim " +
          std::to_string(i) +
          ": index=" + std::to_string(a[i]) +
          ", size=" + std::to_string(shape_[i])
        );
      }
    }
    #endif
  }

  //shape_を元にstride_を作る
  std::vector<int64_t> make_stride() const{
    std::vector<int64_t> str(shape_.size());
    int64_t stride_size = 1;
    for(int64_t i = shape_.size();i-- > 0;){
      str[i] = stride_size;
      stride_size *= shape_[i];
    }
    return str;
  }
};

}//namespace cobalt_715::nn::linear