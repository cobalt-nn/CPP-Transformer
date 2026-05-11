#pragma once

#include <cstdint>
#include <string>
#include "Layout.hpp"

namespace cobalt_715::nn::tensor{

//Tensorなどから一部を行列として借用する
class ConstMatrixView{
public:
  constexpr ConstMatrixView(int64_t rows,int64_t cols,const float *data) :
    rows_(rows),
    cols_(cols),
    row_stride_(cols),
    col_stride_(1),
    data_(data){

    update_layout();
  }

  constexpr ConstMatrixView(int64_t rows,int64_t cols,int64_t row_stride,int64_t col_stride,const float *data) :
    rows_(rows),
    cols_(cols),
    row_stride_(row_stride),
    col_stride_(col_stride),
    data_(data){

    update_layout();
  }

  inline const float& at(const int64_t row,const int64_t col) const{
    return data_[row * row_stride_ + col * col_stride_];
  }

  inline const int64_t rows() const noexcept{
    return rows_;
  }

  inline const int64_t cols() const noexcept{
    return cols_;
  }

  inline const int64_t row_stride() const noexcept{
    return row_stride_;
  }

  inline const int64_t col_stride() const noexcept{
    return col_stride_;
  }

  inline const Layout layout() const noexcept{
    return type_;
  }

  /*inline const bool is_writable() const noexcept{
    return type_ != Layout::OVERLAPPED;
  }*/

  inline int64_t numel() const noexcept{
    return rows_ * cols_;
  }

  //生ポインタかつ何も保証していないので気を付けること
  //at(0,0)であるということだけ
  inline const float* base_ptr() const noexcept{return data_;}

  //転置viewを返す
  inline constexpr ConstMatrixView transpose() const noexcept{
    return ConstMatrixView(cols_,rows_,col_stride_,row_stride_,data_);
  }

  inline constexpr ConstMatrixView t() const noexcept{
    return ConstMatrixView(cols_,rows_,col_stride_,row_stride_,data_);
  }

  std::string to_string() const{
    std::string s = "// " + std::to_string(rows_) + " * " + std::to_string(cols_);
    s += "\n{\n";
    for(int i = 0;i < rows_;i++){
      s += "{ ";
      for(int j = 0;j < cols_ - 1;j++){
        s += std::to_string(at(i,j)) + " , ";
      }
      s += std::to_string(at(i,cols_ - 1)) + " }";
      if(i != rows_ -1) s += " ,";
      s += "\n";
    }
    s += "}";
    return s;
  }

private:
  int64_t rows_;//行
  int64_t cols_;//列
  int64_t row_stride_;//row_stride
  int64_t col_stride_;//col_stride
  const float *data_;//data
  Layout type_;

  constexpr void update_layout() noexcept{
    if(row_stride_ == 0){
      type_ = Layout::OVERLAPPED;
      return;
    }
    if(row_stride_ == cols_ && col_stride_ == 1 || cols_ == 0){
      type_ = Layout::CONTIGUOUS;
      return;
    }
    if(col_stride_ == 1){
      type_ = Layout::ROW_CONTIGUOUS;
      return;
    }
    if(row_stride_ == 1){
      type_ = Layout::COL_CONTIGUOUS;
      return;
    }
    if(std::abs(row_stride_) >= cols_ * std::abs(col_stride_) && col_stride_ != 0){
      type_ = Layout::STRIDED_SAFE;
      return;
    }
    type_ = Layout::OVERLAPPED;
  }
};

}//namespace cobalt_715::nn::tensor