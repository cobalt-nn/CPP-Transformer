#pragma once

//#include <iostream>
#include <cstdint>
#include <string>
#include "vec.hpp"
#include "vec_cpu.hpp"
#include "ElementRef.hpp"

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

  constexpr MatrixView(int64_t rows,int64_t cols,float *data) :
    rows_(rows),
    cols_(cols),
    row_stride_(cols),
    col_stride_(1),
    data_(data){

    update_layout();
  }

  constexpr MatrixView(int64_t rows,int64_t cols,int64_t row_stride,int64_t col_stride,float *data) :
    rows_(rows),
    cols_(cols),
    row_stride_(row_stride),
    col_stride_(col_stride),
    data_(data){

    update_layout();
  }

  inline detail::ElementRef at(const int64_t row,const int64_t col){
    #ifndef NDEBUG
      return detail::ElementRef(&data_[row * row_stride_ + col * col_stride_],type_ != LayoutType::OVERLAPPED);
    #else
      return detail::ElementRef(&data_[row * row_stride_ + col * col_stride_]);
    #endif
  }

  inline const float at(const int64_t row,const int64_t col) const{
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

  inline const LayoutType layout() const noexcept{
    return type_;
  }

  inline const bool is_writable() const noexcept{
    return type_ != LayoutType::OVERLAPPED;
  }

  inline int64_t numel() const noexcept{
    return rows_ * cols_;
  }

  //生ポインタかつ何も保証していないので気を付けること
  //at(0,0)であるということだけ
  inline float* base_ptr() noexcept{return data_;}
  inline const float* base_ptr() const noexcept{return data_;}

  //行列積
  static void matmul(const MatrixView &a,const MatrixView &b,MatrixView &out){
    #ifndef NDEBUG
      if(a.cols() != b.rows()) throw std::invalid_argument("Matrix::dot dimension mismatch");
      if(out.rows() != a.rows() || out.cols() != b.cols()) throw std::invalid_argument("Matrix::dot_Bt dimension mismatch: out.rows() != a.rows() or out.cols() != b.cols()");
    #endif

    if(!out.is_writable()) throw std::logic_error("MatrixView::matmul Write to overlapped");

    for(int64_t i = 0;i < out.rows();i++){
      for(int64_t j = 0;j < out.cols();j++){
        float sum = 0;
        for(int64_t k = 0;k < a.cols();k++){
          sum += a.at(i,k) * b.at(k,j);
        }
        out.at(i,j) = sum;
      }
    }
  }

  //out(i,j) = a(i,j) + b(i,j)
  inline static void add(const MatrixView &a,const MatrixView &b,MatrixView &out){
    #ifndef NDEBUG
      if(a.rows() != b.rows() || a.cols() != b.cols() || a.rows() != out.rows() || a.cols() != out.cols()) throw std::invalid_argument("MatrixView::add dimension mismatch");
    #endif

    if(a.layout() == LayoutType::CONTIGUOUS && b.layout() == LayoutType::CONTIGUOUS && out.layout() == LayoutType::CONTIGUOUS){
      vec::add(a.base_ptr(),b.base_ptr(),out.base_ptr(),a.numel());

    }else if(a.layout() == LayoutType::ROW_CONTIGUOUS && b.layout() == LayoutType::ROW_CONTIGUOUS && out.layout() == LayoutType::ROW_CONTIGUOUS){
      const float *ad = a.base_ptr();
      const float *bd = b.base_ptr();
      float *od = out.base_ptr();

      const int64_t stride = a.row_stride();

      const int64_t cols = a.cols();

      for(int64_t i = 0;i < a.rows();i++){
        vec::add(ad,bd,od,cols);

        ad += stride;
        bd += stride;
        od += stride;
      }
    }else if(out.is_writable()){
      const float *ad = a.base_ptr();
      const float *bd = b.base_ptr();
      float *od = out.base_ptr();

      const int64_t rows = a.rows();
      const int64_t cols = a.cols();

      const int64_t ars = a.row_stride();
      const int64_t acs = a.col_stride();

      const int64_t brs = b.row_stride();
      const int64_t bcs = b.col_stride();

      const int64_t ors = out.row_stride();
      const int64_t ocs = out.col_stride();

      for(int64_t i = 0;i < rows;i++){
        const float *ac = ad;
        const float *bc = bd;
        float *oc = od;
        for(int64_t j = 0;j < cols;j++){
          *oc = *ac + *bc;
          ac += acs;
          bc += bcs;
          oc += ocs;
        }
        ad += ars;
        bd += brs;
        od += ors;
      }
    }else throw std::logic_error("MatrixView::add Write to overlapped");
  }

  //out(i,j) = a(i,j) - b(i,j)
  inline static void sub(const MatrixView &a,const MatrixView &b,MatrixView &out){
    #ifndef NDEBUG
      if(a.rows() != b.rows() || a.cols() != b.cols() || a.rows() != out.rows() || a.cols() != out.cols()) throw std::invalid_argument("MatrixView::sub dimension mismatch");
    #endif

    if(a.layout() == LayoutType::CONTIGUOUS && b.layout() == LayoutType::CONTIGUOUS && out.layout() == LayoutType::CONTIGUOUS){
      vec::sub(a.base_ptr(),b.base_ptr(),out.base_ptr(),a.numel());

    }else if(a.layout() == LayoutType::ROW_CONTIGUOUS && b.layout() == LayoutType::ROW_CONTIGUOUS && out.layout() == LayoutType::ROW_CONTIGUOUS){
      const float *ad = a.base_ptr();
      const float *bd = b.base_ptr();
      float *od = out.base_ptr();

      const int64_t stride = a.row_stride();

      const int64_t cols = a.cols();

      for(int64_t i = 0;i < a.rows();i++){
        vec::sub(ad,bd,od,cols);

        ad += stride;
        bd += stride;
        od += stride;
      }
    }else if(out.is_writable()){
      const float *ad = a.base_ptr();
      const float *bd = b.base_ptr();
      float *od = out.base_ptr();

      const int64_t rows = a.rows();
      const int64_t cols = a.cols();

      const int64_t ars = a.row_stride();
      const int64_t acs = a.col_stride();

      const int64_t brs = b.row_stride();
      const int64_t bcs = b.col_stride();

      const int64_t ors = out.row_stride();
      const int64_t ocs = out.col_stride();

      for(int64_t i = 0;i < rows;i++){
        const float *ac = ad;
        const float *bc = bd;
        float *oc = od;
        for(int64_t j = 0;j < cols;j++){
          *oc = *ac - *bc;
          ac += acs;
          bc += bcs;
          oc += ocs;
        }
        ad += ars;
        bd += brs;
        od += ors;
      }
    }else throw std::logic_error("MatrixView::sub Write to overlapped");
  }

  //out(i,j) = a(i,j) * b(i,j)
  inline static void hadamard(const MatrixView &a,const MatrixView &b,MatrixView &out){
    #ifndef NDEBUG
      if(a.rows() != b.rows() || a.cols() != b.cols() || a.rows() != out.rows() || a.cols() != out.cols()) throw std::invalid_argument("MatrixView::hadamard dimension mismatch");
    #endif

    if(a.layout() == LayoutType::CONTIGUOUS && b.layout() == LayoutType::CONTIGUOUS && out.layout() == LayoutType::CONTIGUOUS){
      vec::mul(a.base_ptr(),b.base_ptr(),out.base_ptr(),a.numel());

    }else if(a.layout() == LayoutType::ROW_CONTIGUOUS && b.layout() == LayoutType::ROW_CONTIGUOUS && out.layout() == LayoutType::ROW_CONTIGUOUS){
      const float *ad = a.base_ptr();
      const float *bd = b.base_ptr();
      float *od = out.base_ptr();

      const int64_t stride = a.row_stride();

      const int64_t cols = a.cols();

      for(int64_t i = 0;i < a.rows();i++){
        vec::mul(ad,bd,od,cols);

        ad += stride;
        bd += stride;
        od += stride;
      }
    }else if(out.is_writable()){
      const float *ad = a.base_ptr();
      const float *bd = b.base_ptr();
      float *od = out.base_ptr();

      const int64_t rows = a.rows();
      const int64_t cols = a.cols();

      const int64_t ars = a.row_stride();
      const int64_t acs = a.col_stride();

      const int64_t brs = b.row_stride();
      const int64_t bcs = b.col_stride();

      const int64_t ors = out.row_stride();
      const int64_t ocs = out.col_stride();

      for(int64_t i = 0;i < rows;i++){
        const float *ac = ad;
        const float *bc = bd;
        float *oc = od;
        for(int64_t j = 0;j < cols;j++){
          *oc = *ac * *bc;
          ac += acs;
          bc += bcs;
          oc += ocs;
        }
        ad += ars;
        bd += brs;
        od += ors;
      }
    }else throw std::logic_error("MatrixView::hadamard Write to overlapped");
  }

  //out(i,j) = a(i,j) * c
  inline static void scale(const MatrixView &a,const float c,MatrixView &out){
    #ifndef NDEBUG
      if(a.rows() != out.rows() || a.cols() != out.cols()) throw std::invalid_argument("MatrixView::scale dimension mismatch");
    #endif

    if(a.layout() == LayoutType::CONTIGUOUS && out.layout() == LayoutType::CONTIGUOUS){
      vec::scale(a.base_ptr(),c,out.base_ptr(),a.numel());

    }else if(a.layout() == LayoutType::ROW_CONTIGUOUS && out.layout() == LayoutType::ROW_CONTIGUOUS){
      const float *ad = a.base_ptr();
      float *od = out.base_ptr();

      const int64_t stride = a.row_stride();

      const int64_t cols = a.cols();

      for(int64_t i = 0;i < a.rows();i++){
        vec::scale(ad,c,od,cols);

        ad += stride;
        od += stride;
      }
    }else if(out.is_writable()){
      const float *ad = a.base_ptr();
      float *od = out.base_ptr();

      const int64_t rows = a.rows();
      const int64_t cols = a.cols();

      const int64_t ars = a.row_stride();
      const int64_t acs = a.col_stride();

      const int64_t ors = out.row_stride();
      const int64_t ocs = out.col_stride();

      for(int64_t i = 0;i < rows;i++){
        const float *ac = ad;
        float *oc = od;
        for(int64_t j = 0;j < cols;j++){
          *oc = *ac * c;
          ac += acs;
          oc += ocs;
        }
        ad += ars;
        od += ors;
      }
    }else throw std::logic_error("MatrixView::scale Write to overlapped");
  }

  //転置viewを返す
  inline constexpr MatrixView transpose() const noexcept{
    return MatrixView(cols_,rows_,col_stride_,row_stride_,data_);
  }

  inline constexpr MatrixView t() const noexcept{
    return MatrixView(cols_,rows_,col_stride_,row_stride_,data_);
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

  constexpr void update_layout() noexcept{
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