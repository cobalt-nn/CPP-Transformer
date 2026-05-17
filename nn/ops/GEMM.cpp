#include <immintrin.h>
#include <cstdint>
#include "nn/tensor/MatrixView.hpp"
#include "nn/tensor/ConstMatrixView.hpp"

namespace cobalt_715::nn::ops{

//素朴実装
//bは転置済、aもbもset_pack()が前提だ。俺はみんなを信じているぞキリッ★
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

#if defined(__AVX2__) && defined(__FMA__)

//b転置、aもbもset_pack_interleave<8>前提
template<bool FirstK>
void kernel_avx2(const float*__restrict a_pack,const float*__restrict bt_pack,const float alpha,const float beta,tensor::MatrixView &out,const int64_t IB,const int64_t JB,const int64_t KB,const int64_t ii,const int64_t jj){
  const int64_t out_row_stride = out.row_stride();
  const int64_t out_col_stride = out.col_stride();
  float *od = out.base_ptr();

  __m256 v_alpha = _mm256_set1_ps(alpha);

  for(int64_t i = 0;i < IB;i += 4){
    for(int64_t j = 0;j < JB;j += 4){
      __m256 vc00 = _mm256_setzero_ps();
      __m256 vc01 = _mm256_setzero_ps();
      __m256 vc02 = _mm256_setzero_ps();
      __m256 vc03 = _mm256_setzero_ps();

      __m256 vc10 = _mm256_setzero_ps();
      __m256 vc11 = _mm256_setzero_ps();
      __m256 vc12 = _mm256_setzero_ps();
      __m256 vc13 = _mm256_setzero_ps();

      __m256 vc20 = _mm256_setzero_ps();
      __m256 vc21 = _mm256_setzero_ps();
      __m256 vc22 = _mm256_setzero_ps();
      __m256 vc23 = _mm256_setzero_ps();

      __m256 vc30 = _mm256_setzero_ps();
      __m256 vc31 = _mm256_setzero_ps();
      __m256 vc32 = _mm256_setzero_ps();
      __m256 vc33 = _mm256_setzero_ps();

      for(int64_t k = 0;k < KB;k += 8){
        const float *adptr = &a_pack[((k / 8) * IB  + i) * 8 + (k % 8)];

        __m256 va0 = _mm256_load_ps(adptr);
        __m256 va1 = _mm256_load_ps(adptr + 8);
        __m256 va2 = _mm256_load_ps(adptr + 16);
        __m256 va3 = _mm256_load_ps(adptr + 24);

        const float *btdptr = &bt_pack[((k / 8) * JB  + j) * 8 + (k % 8)];

        __m256 vb0 = _mm256_load_ps(btdptr);
        __m256 vb1 = _mm256_load_ps(btdptr + 8);
        __m256 vb2 = _mm256_load_ps(btdptr + 16);
        __m256 vb3 = _mm256_load_ps(btdptr + 24);

        vc00 = _mm256_fmadd_ps(va0,vb0,vc00);
        vc01 = _mm256_fmadd_ps(va0,vb1,vc01);
        vc02 = _mm256_fmadd_ps(va0,vb2,vc02);
        vc03 = _mm256_fmadd_ps(va0,vb3,vc03);

        vc10 = _mm256_fmadd_ps(va1,vb0,vc10);
        vc11 = _mm256_fmadd_ps(va1,vb1,vc11);
        vc12 = _mm256_fmadd_ps(va1,vb2,vc12);
        vc13 = _mm256_fmadd_ps(va1,vb3,vc13);

        vc20 = _mm256_fmadd_ps(va2,vb0,vc20);
        vc21 = _mm256_fmadd_ps(va2,vb1,vc21);
        vc22 = _mm256_fmadd_ps(va2,vb2,vc22);
        vc23 = _mm256_fmadd_ps(va2,vb3,vc23);

        vc30 = _mm256_fmadd_ps(va3,vb0,vc30);
        vc31 = _mm256_fmadd_ps(va3,vb1,vc31);
        vc32 = _mm256_fmadd_ps(va3,vb2,vc32);
        vc33 = _mm256_fmadd_ps(va3,vb3,vc33);

        //sum += a_pack[i * KB + k] * bt_pack[j * KB + k];
      }

      vc00 = _mm256_mul_ps(vc00,v_alpha);
      vc01 = _mm256_mul_ps(vc01,v_alpha);
      vc02 = _mm256_mul_ps(vc02,v_alpha);
      vc03 = _mm256_mul_ps(vc03,v_alpha);

      vc10 = _mm256_mul_ps(vc10,v_alpha);
      vc11 = _mm256_mul_ps(vc11,v_alpha);
      vc12 = _mm256_mul_ps(vc12,v_alpha);
      vc13 = _mm256_mul_ps(vc13,v_alpha);

      vc20 = _mm256_mul_ps(vc20,v_alpha);
      vc21 = _mm256_mul_ps(vc21,v_alpha);
      vc22 = _mm256_mul_ps(vc22,v_alpha);
      vc23 = _mm256_mul_ps(vc23,v_alpha);

      vc30 = _mm256_mul_ps(vc30,v_alpha);
      vc31 = _mm256_mul_ps(vc31,v_alpha);
      vc32 = _mm256_mul_ps(vc32,v_alpha);
      vc33 = _mm256_mul_ps(vc33,v_alpha);

      float *odptr = &od[(ii + i) * out_row_stride + (jj + j) * out_col_stride];

      if constexpr(FirstK){
        float *odptr_ij = odptr;

        *odptr_ij = *odptr_ij * beta + m256_sum(vc00);
        odptr_ij += out_col_stride;

        *odptr_ij = *odptr_ij * beta + m256_sum(vc01);
        odptr_ij += out_col_stride;

        *odptr_ij = *odptr_ij * beta + m256_sum(vc02);
        odptr_ij += out_col_stride;

        *odptr_ij = *odptr_ij * beta + m256_sum(vc03);
        odptr_ij += out_col_stride;


        odptr_ij = odptr + out_row_stride;

        *odptr_ij = *odptr_ij * beta + m256_sum(vc10);
        odptr_ij += out_col_stride;

        *odptr_ij = *odptr_ij * beta + m256_sum(vc11);
        odptr_ij += out_col_stride;

        *odptr_ij = *odptr_ij * beta + m256_sum(vc12);
        odptr_ij += out_col_stride;

        *odptr_ij = *odptr_ij * beta + m256_sum(vc13);
        odptr_ij += out_col_stride;


        odptr_ij = odptr + out_row_stride * 2;

        *odptr_ij = *odptr_ij * beta + m256_sum(vc20);
        odptr_ij += out_col_stride;

        *odptr_ij = *odptr_ij * beta + m256_sum(vc21);
        odptr_ij += out_col_stride;

        *odptr_ij = *odptr_ij * beta + m256_sum(vc22);
        odptr_ij += out_col_stride;

        *odptr_ij = *odptr_ij * beta + m256_sum(vc23);
        odptr_ij += out_col_stride;


        odptr_ij = odptr + out_row_stride * 3;

        *odptr_ij = *odptr_ij * beta + m256_sum(vc30);
        odptr_ij += out_col_stride;

        *odptr_ij = *odptr_ij * beta + m256_sum(vc31);
        odptr_ij += out_col_stride;

        *odptr_ij = *odptr_ij * beta + m256_sum(vc32);
        odptr_ij += out_col_stride;

        *odptr_ij = *odptr_ij * beta + m256_sum(vc33);
        odptr_ij += out_col_stride;

        //*odij = *odij * beta + sum * alpha;
      }else{
        float *odptr_ij = odptr;

        *odptr_ij += m256_sum(vc00);
        odptr_ij += out_col_stride;

        *odptr_ij += m256_sum(vc01);
        odptr_ij += out_col_stride;

        *odptr_ij += m256_sum(vc02);
        odptr_ij += out_col_stride;

        *odptr_ij += m256_sum(vc03);
        odptr_ij += out_col_stride;


        odptr_ij = odptr + out_row_stride;

        *odptr_ij += m256_sum(vc10);
        odptr_ij += out_col_stride;

        *odptr_ij += m256_sum(vc11);
        odptr_ij += out_col_stride;

        *odptr_ij += m256_sum(vc12);
        odptr_ij += out_col_stride;

        *odptr_ij += m256_sum(vc13);
        odptr_ij += out_col_stride;


        odptr_ij = odptr + out_row_stride * 2;

        *odptr_ij += m256_sum(vc20);
        odptr_ij += out_col_stride;

        *odptr_ij += m256_sum(vc21);
        odptr_ij += out_col_stride;

        *odptr_ij += m256_sum(vc22);
        odptr_ij += out_col_stride;

        *odptr_ij += m256_sum(vc23);
        odptr_ij += out_col_stride;


        odptr_ij = odptr + out_row_stride * 3;

        *odptr_ij += m256_sum(vc30);
        odptr_ij += out_col_stride;

        *odptr_ij += m256_sum(vc31);
        odptr_ij += out_col_stride;

        *odptr_ij += m256_sum(vc32);
        odptr_ij += out_col_stride;

        *odptr_ij += m256_sum(vc33);
        odptr_ij += out_col_stride;

        //*odij += sum * alpha;
      }
    }
  }
}

template void kernel_avx2<true>(const float*__restrict a_pack,const float*__restrict bt_pack,const float alpha,const float beta,tensor::MatrixView &out,const int64_t IB,const int64_t JB,const int64_t KB,const int64_t ii,const int64_t jj);

template void kernel_avx2<false>(const float*__restrict a_pack,const float*__restrict bt_pack,const float alpha,const float beta,tensor::MatrixView &out,const int64_t IB,const int64_t JB,const int64_t KB,const int64_t ii,const int64_t jj);

#endif

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
      set_pack_interleave<8>(bt,jj,kk,JB,KB,bt_pack);

      for(int64_t ii = 0;ii + IB <= orows;ii += IB){
        set_pack_interleave<8>(a,ii,kk,IB,KB,a_pack);
        if(kk == 0){
          kernel_avx2<true>(a_pack,bt_pack,alpha,beta,out,IB,JB,KB,ii,jj);
        }else{
          kernel_avx2<false>(a_pack,bt_pack,alpha,beta,out,IB,JB,KB,ii,jj);
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