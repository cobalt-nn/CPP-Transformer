#pragma once

#include <cstdint>
#include <string>

namespace cobalt_715::nn::linear{

//Tensorなどから一部を行列として借用する
class MatrixView{
public:
  enum class LayoutType{
    CONTIGUOUS,//完全連続
    ROW_CONTIGUOUS,//行連続
    STRIDED_SAFE,//strided_safe
    OVERLAPPED//書き込み不可能
  };

  MatrixView(int64_t rows,int64_t cols,float *data) :
    rows_(rows),
    cols_(cols),
    row_stride_(cols),
    col_stride_(1),
    data_(data){

    update_layout();
  }

  MatrixView(int64_t rows,int64_t cols,int64_t row_stride,int64_t col_stride,float *data) :
    rows_(rows),
    cols_(cols),
    row_stride_(row_stride),
    col_stride_(col_stride),
    data_(data){

    update_layout();
  }

  float& at(const int64_t row,const int64_t col){
    return data_[row * row_stride_ + col * col_stride_];
  }

  const float& at(const int64_t row,const int64_t col) const{
    return data_[row * row_stride_ + col * col_stride_];
  }

  const int64_t rows() const{
    return rows_;
  }

  const int64_t cols() const{
    return cols_;
  }

  const int64_t row_stride() const{
    return row_stride_;
  }

  const int64_t col_stride() const{
    return col_stride_;
  }

  const LayoutType layout() const{
    return type_;
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
  float *data_;//data
  LayoutType type_;

  void update_layout(){
    if(row_stride_ == 0){
      type_ = LayoutType::OVERLAPPED;
      return;
    }
    if(row_stride_ == cols_ && col_stride_ == 1 || cols_ == 0){
      type_ = LayoutType::CONTIGUOUS;
      return;
    }
    if(col_stride_ == 1){
      type_ = LayoutType::ROW_CONTIGUOUS;
      return;
    }
    if(std::abs(row_stride_) >= cols_ * std::abs(col_stride_) && col_stride_ != 0){
      type_ = LayoutType::STRIDED_SAFE;
      return;
    }
    type_ = LayoutType::OVERLAPPED;
  }
};

}//namespace cobalt_715::nn::linear