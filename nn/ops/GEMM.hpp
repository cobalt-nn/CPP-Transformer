#pragma once

#include <cstdint>

//ブロックサイズは32の倍数固定
//えっほかのサイズもしたいってそれは無理な相談だ諦めるんだな

namespace cobalt_715::nn::tensor{
  struct MatrixView;
  struct ConstMatrixView;
  enum class Layout;
}//namespace cobalt_715::nn::tensor

namespace cobalt_715::nn::ops{

//素朴
template<bool FirstK>
void kernel(const float*__restrict a_pack,const float*__restrict bt_pack,const float alpha,const float beta,tensor::MatrixView &out,const int64_t IB,const int64_t JB,const int64_t KB,const int64_t ii,const int64_t jj);

//レジスタタイル
template<bool FirstK>
void kernel_reg_4_4(const float*__restrict a_pack,const float*__restrict bt_pack,const float alpha,const float beta,tensor::MatrixView &out,const int64_t IB,const int64_t JB,const int64_t KB,const int64_t ii,const int64_t jj);

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

inline void set_pack(const tensor::ConstMatrixView &m,int64_t row,int64_t col,int64_t i_size,int64_t j_size,float*__restrict pack){
  const int64_t rs = m.row_stride();
  const int64_t cs = m.col_stride();

  const float*__restrict md = m.base_ptr() + row * rs + col * cs;

  for(int64_t i = 0;i < i_size;i++){
    float *pack_i = pack;
    const float *mdi = md;

    for(int64_t j = 0;j < j_size;j++){
      *pack_i = *mdi;

      pack_i++;
      mdi += cs;
    }

    pack += j_size;
    md += rs;
  }
}

template<int64_t num>
inline void set_pack_interleave(const tensor::ConstMatrixView &m,int64_t row,int64_t col,int64_t i_size,int64_t j_size,float*__restrict pack){
  const int64_t rs = m.row_stride();
  const int64_t cs = m.col_stride();

  const float*__restrict md = m.base_ptr() + row * rs + col * cs;

  for(int64_t i = 0;i < i_size;i++){
    const float *mdi = md;

    for(int64_t j = 0;j < j_size;j++){
      //(j / num) * num * i_sizeはpackの行の最初を示す
      pack[(j / num) * num * i_size + (j % num) + i * num] = *mdi;
      mdi += cs;
    }

    md += rs;
  }
}

}//namespace cobalt_715::nn::ops