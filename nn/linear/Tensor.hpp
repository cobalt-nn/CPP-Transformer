#pragma once

#include <vector>
#include <span>
#include <cstdint>
#include <stdexcept>
#include <string>
#include "nn/backend/backend.hpp"
#include "vec.hpp"
#include "vec_cpu.hpp"

//任意次元テンソル
class Tensor{
private:
  std::vector<size_t> shape_;//各次元の要素数
  std::vector<size_t> stride_;//各次元にジャンプするまでに必要な数
  std::vector<float> data_;//data

  //shape_,stride_,data_の関係が合うかどうかとオーバーフローの確認をする
  void check_invariants() const{
    #ifndef NDEBUG
    //shape_とstride_のサイズが同じかどうか確認する
    if(shape_.size() != stride_.size()) throw std::logic_error("Tensor invariant violation: shape and stride size mismatch");

    //data_のサイズがshape_と合っているかとオーバーフローしていないか確認する
    size_t data_size = 1;
    for(size_t i = 0;i < shape_.size();i++){
      size_t dim = shape_[i];

      //0になっていないか確認する
      if(dim == 0) throw std::invalid_argument("Tensor shape contains zero dimension");

      //オーバーフローを確認する
      if(dim > SIZE_MAX / data_size) throw std::overflow_error("Tensor size overflow");

      data_size *= dim;
    }
    if(data_.size() != data_size) throw std::logic_error("Tensor invariant violation: data size mismatch");

    //stride_がshape_と合っているか確認する
    if(stride_ != make_stride()) throw std::logic_error("Tensor invariant violation: invalid stride");
    #endif
  }

  //indexが正しいか確認する
  void check_index(const std::vector<size_t>& a) const{
    #ifndef NDEBUG
    if(a.size() != shape_.size()){
      throw std::invalid_argument(
        "Tensor index dimension mismatch: expected " +
        std::to_string(shape_.size()) +
        ", got " +
        std::to_string(a.size())
      );
    }

    for(size_t i = 0;i < shape_.size();i++){
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
  std::vector<size_t> make_stride() const{
    std::vector<size_t> str(shape_.size());
    size_t stride_size = 1;
    for(size_t i = shape_.size();i-- > 0;){
      str[i] = stride_size;
      stride_size *= shape_[i];
    }
    return str;
  }

public:
  float& at(const std::vector<size_t>& a){
    check_index(a);
    //check_invariants();
    size_t index = 0;
    for(size_t i = 0;i < a.size();i++){
      index += a[i] * stride_[i];
    }
    return data_[index];
  }

  const float& at(const std::vector<size_t>& a) const{
    check_index(a);
    //check_invariants();
    size_t index = 0;
    for(size_t i = 0;i < a.size();i++){
      index += a[i] * stride_[i];
    }
    return data_[index];
  }

  const std::vector<size_t>& shape() const noexcept{
    return shape_;
  }

  const std::vector<size_t>& stride() const noexcept{
    return stride_;
  }

  //ポインタを返しているため気を付けること
  float* data() noexcept{return data_.data();}
  const float* data() const noexcept{return data_.data();}

  std::span<float> span() noexcept{
    return std::span<float>(data_);
  }

  std::span<const float> span() const noexcept{
    return std::span<const float>(data_);
  }
};