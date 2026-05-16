#include <cstdint>
#include "nn/tensor/MatrixView.hpp"
#include "nn/tensor/ConstMatrixView.hpp"

namespace cobalt_715::nn::ops{

template<bool FirstK>
void kernel(const float*__restrict a_pack,const float*__restrict bt_pack,const float alpha,const float beta,tensor::MatrixView &out,const int64_t IB,const int64_t JB,const int64_t KB,const int64_t ii,const int64_t jj){
  const int64_t out_row_stride = out.row_stride();
  const int64_t out_col_stride = out.col_stride();
  float *od = out.base_ptr();

  for(int64_t i = 0;i < IB;i++){
    for(int64_t j = 0;j < JB;j++){
      float sum = 0;

      for(int64_t k = 0;k < KB;k++){
        sum += a_pack[i * KB + k] * bt_pack[j * KB + k];
      }

      float *odij = &od[(ii + i) * out_row_stride + (jj + j) * out_col_stride];

      if constexpr(FirstK){
        *odij = *odij * beta + sum * alpha;
      }else{
        *odij += sum * alpha;
      }
    }
  }
}

template void kernel<true>(const float*__restrict a_pack,const float*__restrict bt_pack,const float alpha,const float beta,tensor::MatrixView &out,const int64_t IB,const int64_t JB,const int64_t KB,const int64_t ii,const int64_t jj);

template void kernel<false>(const float*__restrict a_pack,const float*__restrict bt_pack,const float alpha,const float beta,tensor::MatrixView &out,const int64_t IB,const int64_t JB,const int64_t KB,const int64_t ii,const int64_t jj);

//レジスタタイル
//bは転置済、aもbもset_pack()が前提だ。俺はみんなを信じているぞキリッ★
template<bool FirstK>
void kernel_reg_4_4(const float*__restrict a_pack,const float*__restrict bt_pack,const float alpha,const float beta,tensor::MatrixView &out,const int64_t IB,const int64_t JB,const int64_t KB,const int64_t ii,const int64_t jj){
  const int64_t out_row_stride = out.row_stride();
  const int64_t out_col_stride = out.col_stride();
  float *od = out.base_ptr();

  for(int64_t i = 0;i < IB;i += 4){
    const float *adptr0 = a_pack + i * KB;
    const float *adptr1 = a_pack + (i + 1) * KB;
    const float *adptr2 = a_pack + (i + 2) * KB;
    const float *adptr3 = a_pack + (i + 3) * KB;

    for(int64_t j = 0;j < JB;j += 4){
      float o00=0,o01=0,o02=0,o03=0;
      float o10=0,o11=0,o12=0,o13=0;
      float o20=0,o21=0,o22=0,o23=0;
      float o30=0,o31=0,o32=0,o33=0;

      const float *btdptr0 = bt_pack + j * KB;
      const float *btdptr1 = bt_pack + (j + 1) * KB;
      const float *btdptr2 = bt_pack + (j + 2) * KB;
      const float *btdptr3 = bt_pack + (j + 3) * KB;

      for(int64_t k = 0;k < KB;k++){
        const float a0 = adptr0[k];
        const float a1 = adptr1[k];
        const float a2 = adptr2[k];
        const float a3 = adptr3[k];

        const float bt0 = btdptr0[k];
        const float bt1 = btdptr1[k];
        const float bt2 = btdptr2[k];
        const float bt3 = btdptr3[k];

        o00 += a0 * bt0;o01 += a0 * bt1;o02 += a0 * bt2;o03 += a0 * bt3;
        o10 += a1 * bt0;o11 += a1 * bt1;o12 += a1 * bt2;o13 += a1 * bt3;
        o20 += a2 * bt0;o21 += a2 * bt1;o22 += a2 * bt2;o23 += a2 * bt3;
        o30 += a3 * bt0;o31 += a3 * bt1;o32 += a3 * bt2;o33 += a3 * bt3;

        //sum += a_pack[i * KB + k] * bt_pack[j * KB + k];
      }

      float *odptr0 = &od[(ii + i) * out_row_stride + (jj + j) * out_col_stride];
      float *odptr1 = &od[(ii + i + 1) * out_row_stride + (jj + j) * out_col_stride];
      float *odptr2 = &od[(ii + i + 2) * out_row_stride + (jj + j) * out_col_stride];
      float *odptr3 = &od[(ii + i + 3) * out_row_stride + (jj + j) * out_col_stride];

      if constexpr(FirstK){
        odptr0[0] = odptr0[0] * beta + o00 * alpha;odptr0[1 * out_col_stride] = odptr0[1 * out_col_stride] * beta + o01 * alpha;
        odptr0[2 * out_col_stride] = odptr0[2 * out_col_stride] * beta + o02 * alpha;odptr0[3 * out_col_stride] = odptr0[3 * out_col_stride] * beta + o03 * alpha;

        odptr1[0] = odptr1[0] * beta + o10 * alpha;odptr1[1 * out_col_stride] = odptr1[1 * out_col_stride] * beta + o11 * alpha;
        odptr1[2 * out_col_stride] = odptr1[2 * out_col_stride] * beta + o12 * alpha;odptr1[3 * out_col_stride] = odptr1[3 * out_col_stride] * beta + o13 * alpha;

        odptr2[0] = odptr2[0] * beta + o20 * alpha;odptr2[1 * out_col_stride] = odptr2[1 * out_col_stride] * beta + o21 * alpha;
        odptr2[2 * out_col_stride] = odptr2[2 * out_col_stride] * beta + o22 * alpha;odptr2[3 * out_col_stride] = odptr2[3 * out_col_stride] * beta + o23 * alpha;

        odptr3[0] = odptr3[0] * beta + o30 * alpha;odptr3[1 * out_col_stride] = odptr3[1 * out_col_stride] * beta + o31 * alpha;
        odptr3[2 * out_col_stride] = odptr3[2 * out_col_stride] * beta + o32 * alpha;odptr3[3 * out_col_stride] = odptr3[3 * out_col_stride] * beta + o33 * alpha;

        //*odij = *odij * beta + sum * alpha;
      }else{
        odptr0[0] += o00 * alpha; odptr0[1 * out_col_stride] += o01 * alpha;
        odptr0[2 * out_col_stride] += o02 * alpha; odptr0[3 * out_col_stride] += o03 * alpha;

        odptr1[0] += o10 * alpha; odptr1[1 * out_col_stride] += o11 * alpha;
        odptr1[2 * out_col_stride] += o12 * alpha; odptr1[3 * out_col_stride] += o13 * alpha;

        odptr2[0] += o20 * alpha; odptr2[1 * out_col_stride] += o21 * alpha;
        odptr2[2 * out_col_stride] += o22 * alpha; odptr2[3 * out_col_stride] += o23 * alpha;

        odptr3[0] += o30 * alpha; odptr3[1 * out_col_stride] += o31 * alpha;
        odptr3[2 * out_col_stride] += o32 * alpha; odptr3[3 * out_col_stride] += o33 * alpha;

        //*odij += sum * alpha;
      }
    }
  }
}

template void kernel_reg_4_4<true>(const float*__restrict a_pack,const float*__restrict bt_pack,const float alpha,const float beta,tensor::MatrixView &out,const int64_t IB,const int64_t JB,const int64_t KB,const int64_t ii,const int64_t jj);

template void kernel_reg_4_4<false>(const float*__restrict a_pack,const float*__restrict bt_pack,const float alpha,const float beta,tensor::MatrixView &out,const int64_t IB,const int64_t JB,const int64_t KB,const int64_t ii,const int64_t jj);

//out = alpha * ab + beta * out
void gemm_impl(const float alpha,const tensor::ConstMatrixView &a,const tensor::ConstMatrixView &b,const float beta,tensor::MatrixView &out){
  //ブロックサイズ
  //kernelがそれを前提にしているため32の倍数のみにする
  const constexpr int64_t IB = 32;
  const constexpr int64_t JB = 32;
  const constexpr int64_t KB = 32;

  const tensor::ConstMatrixView bt = b.t();

  const int64_t orows = out.rows();
  const int64_t ocols = out.cols();
  const int64_t acols = a.cols();

  const int64_t a_row_stride = a.row_stride();
  const int64_t a_col_stride = a.col_stride();

  const int64_t bt_row_stride = bt.row_stride();
  const int64_t bt_col_stride = bt.col_stride();

  const int64_t out_row_stride = out.row_stride();
  const int64_t out_col_stride = out.col_stride();

  /*for(int64_t i = 0;i < out.rows();i++){
    for(int64_t j = 0;j < out.cols();j++){
      for(int64_t k = 0;k < a.cols();k++){
        if(k == 0){
          out.at(i,j) = a.at(i,k) * bt.at(j,k);
        }else{
          out.at(i,j) += a.at(i,k) * bt.at(j,k);
        }
      }
    }
  }

  if(true) return;*/

  const int64_t j_end = (ocols / JB) * JB;
  const int64_t k_end = (acols / KB) * KB;

  alignas(64) float a_pack[IB * KB];
  alignas(64) float bt_pack[JB * KB];

  for(int64_t jj = 0;jj < j_end;jj += JB){//jj + JB <= ocols
    for(int64_t kk = 0;kk < k_end;kk += KB){//kk + KB <= acols
      set_pack(bt,jj,kk,JB,KB,bt_pack);

      for(int64_t ii = 0;ii + IB <= orows;ii += IB){
        set_pack(a,ii,kk,IB,KB,a_pack);
        if(kk == 0){
          kernel_reg_4_4<true>(a_pack,bt_pack,alpha,beta,out,IB,JB,KB,ii,jj);
        }else{
          kernel_reg_4_4<false>(a_pack,bt_pack,alpha,beta,out,IB,JB,KB,ii,jj);
        }
      }
    }
  }

  const float *ad = a.base_ptr();
  const float *btd = bt.base_ptr();
  float *od = out.base_ptr();

  //iの残り
  for(int64_t i = (orows / IB) * IB;i < out.rows();i++){
    for(int64_t j = 0;j < ocols;j++){
      float *odij = &out.at(i,j);
      float sum = 0;
      for(int64_t k = 0;k < acols;k++){
        sum += a.at(i,k) * bt.at(j,k);
      }
      *odij = *odij * beta + sum * alpha;
    }
  }

  //jの残り
  for(int64_t i = 0;i < (orows / IB) * IB;i++){
    for(int64_t j = (ocols / JB) * JB;j < out.cols();j++){
      float *odij = &out.at(i,j);
      float sum = 0;
      for(int64_t k = 0;k < acols;k++){
        sum += a.at(i,k) * bt.at(j,k);
      }
      *odij = *odij * beta + sum * alpha;
    }
  }

  //kの残り
  for(int64_t i = 0;i < (orows / IB) * IB;i++){
    for(int64_t j = 0;j < (ocols / JB) * JB;j++){
      float *odij = &out.at(i,j);
      float sum = 0;
      for(int64_t k = (acols / KB) * KB;k < a.cols();k++){
        sum += a.at(i,k) * bt.at(j,k);
      }
      if(0 == (acols / KB) * KB){
        *odij = *odij * beta + sum * alpha;
      }else{
        *odij += sum * alpha;
      }
    }
  }
}

}//namespace cobalt_715::nn::ops