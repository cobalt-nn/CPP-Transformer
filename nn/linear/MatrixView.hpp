#pragma once

#include <cstdint>
#include <string>

namespace cobalt_715::nn::linear{

//Tensorなどから一部を行列として借用する
class MatrixView{
public:
  MatrixView(int64_t rows,int64_t cols,float *data) :
    rows_(rows),
    cols_(cols),
    row_stride_(cols),
    data_(data){}

  MatrixView(int64_t rows,int64_t cols,int64_t row_stride,float *data) :
    rows_(rows),
    cols_(cols),
    row_stride_(row_stride),
    data_(data){}

  float& at(const int64_t row,const int64_t col){
    return data_[row * row_stride_ + col];
  }

  const float& at(const int64_t row,const int64_t col) const{
    return data_[row * row_stride_ + col];
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
  float *data_;//data
};

}//namespace cobalt_715::nn::linear