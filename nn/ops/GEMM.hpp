#pragma once

#include <immintrin.h>
#include <cstdint>

//ブロックサイズは32の倍数固定

namespace cobalt_715::nn::tensor{
  struct MatrixView;
  struct ConstMatrixView;
  enum class Layout;
}//namespace cobalt_715::nn::tensor

namespace cobalt_715::nn::ops{

//out = alpha * ab + beta * out
void gemm_impl(const float alpha,const tensor::ConstMatrixView &a,const tensor::ConstMatrixView &b,const float beta,tensor::MatrixView &out);

//out = ab
inline void gemm(const tensor::ConstMatrixView &a,const tensor::ConstMatrixView &b,tensor::MatrixView &out){
  gemm_impl(1.0f,a,b,0.0f,out);
}

//out += ab
inline void gemm_add(const tensor::ConstMatrixView &a,const tensor::ConstMatrixView &b,tensor::MatrixView &out){
  gemm_impl(1.0f,a,b,1.0f,out);
}

}//namespace cobalt_715::nn::ops