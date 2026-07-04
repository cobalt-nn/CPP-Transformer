#include <immintrin.h>
#include <cstdint>
#include "GEMM.hpp"
#include "nn/tensor/MatrixView.hpp"
#include "nn/tensor/ConstMatrixView.hpp"

namespace cobalt_715::nn::ops{

//素朴実装
//bは転置済、aもbもset_pack()が前提だ。俺はみんなを信じているぞキリッ★
void kernel(const float*__restrict a_pack,
            const float*__restrict bt_pack,
            float*__restrict out_pack,
            const int64_t IB,
            const int64_t JB,
            const int64_t KB){

  for(int64_t i = 0;i < IB;i++){
    for(int64_t j = 0;j < JB;j++){
      float sum = 0;
      for(int64_t k = 0;k < KB;k++){
        sum += a_pack[i * KB + k] * bt_pack[j * KB + k];
      }
      out_pack[i * JB + j] = sum;
    }
  }
}

//レジスタブロッキング
//bは転置済、aもbもset_pack()が前提だ。俺はみんなを信じているぞキリッ★
void kernel_reg_4_4(const float*__restrict a_pack,
            const float*__restrict bt_pack,
            float*__restrict out_pack,
            const int64_t IB,
            const int64_t JB,
            const int64_t KB){

  for(int64_t i = 0;i < IB;i += 4){
    const float *adptr0 = a_pack + i * KB;
    const float *adptr1 = a_pack + (i + 1) * KB;
    const float *adptr2 = a_pack + (i + 2) * KB;
    const float *adptr3 = a_pack + (i + 3) * KB;

    float *odptr0 = out_pack + i * JB;
    float *odptr1 = out_pack + (i + 1) * JB;
    float *odptr2 = out_pack + (i + 2) * JB;
    float *odptr3 = out_pack + (i + 3) * JB;

    for(int64_t j = 0;j < JB;j += 4){
      float o00=0,o01=0,o02=0,o03=0;
      float o10=0,o11=0,o12=0,o13=0;
      float o20=0,o21=0,o22=0,o23=0;
      float o30=0,o31=0,o32=0,o33=0;

      const float *btdptr0 = bt_pack + j * KB;
      const float *btdptr1 = bt_pack + (j + 1) * KB;
      const float *btdptr2 = bt_pack + (j + 2) * KB;
      const float *btdptr3 = bt_pack + (j + 3) * KB;

      float *optr0 = odptr0 + j;
      float *optr1 = odptr1 + j;
      float *optr2 = odptr2 + j;
      float *optr3 = odptr3 + j;

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
      }

      optr0[0] = o00; optr0[1] = o01; optr0[2] = o02; optr0[3] = o03;
      optr1[0] = o10; optr1[1] = o11; optr1[2] = o12; optr1[3] = o13;
      optr2[0] = o20; optr2[1] = o21; optr2[2] = o22; optr2[3] = o23;
      optr3[0] = o30; optr3[1] = o31; optr3[2] = o32; optr3[3] = o33;
    }
  }
}

#if defined(__AVX2__) && defined(__FMA__)

//AVX2
//aもbもそのまま、aもbもset_pack()が前提だ。俺はみんなを信じているぞキリッ★
void kernel_avx2(const float*__restrict a_pack,
            const float*__restrict b_pack,
            float*__restrict out_pack,
            const int64_t IB,
            const int64_t JB,
            const int64_t KB){

  /*for(int64_t i = 0;i < IB;i += 1){
    for(int64_t j = 0;j < JB;j += 8){
      __m256 vo0 = _mm256_setzero_ps();

      for(int64_t k = 0;k < KB;k += 1){
        __m256 va0 = _mm256_set1_ps(a_pack[i * KB + k]);

        __m256 vb0 = _mm256_load_ps(&b_pack[k * JB + j]);

        vo0 = _mm256_fmadd_ps(va0,vb0,vo0);
      }

      _mm256_store_ps(&out_pack[i * JB + j],vo0);
    }
  }*/

  /*for(int64_t i = 0;i < IB;i += 8){
    for(int64_t j = 0;j < JB;j += 8){
      __m256 vo0 = _mm256_setzero_ps();
      __m256 vo1 = _mm256_setzero_ps();
      __m256 vo2 = _mm256_setzero_ps();
      __m256 vo3 = _mm256_setzero_ps();
      __m256 vo4 = _mm256_setzero_ps();
      __m256 vo5 = _mm256_setzero_ps();
      __m256 vo6 = _mm256_setzero_ps();
      __m256 vo7 = _mm256_setzero_ps();

      for(int64_t k = 0;k < KB;k += 1){
        __m256 va0 = _mm256_set1_ps(a_pack[i * KB + k]);
        __m256 va1 = _mm256_set1_ps(a_pack[(i + 1) * KB + k]);
        __m256 va2 = _mm256_set1_ps(a_pack[(i + 2) * KB + k]);
        __m256 va3 = _mm256_set1_ps(a_pack[(i + 3) * KB + k]);
        __m256 va4 = _mm256_set1_ps(a_pack[(i + 4) * KB + k]);
        __m256 va5 = _mm256_set1_ps(a_pack[(i + 5) * KB + k]);
        __m256 va6 = _mm256_set1_ps(a_pack[(i + 6) * KB + k]);
        __m256 va7 = _mm256_set1_ps(a_pack[(i + 7) * KB + k]);

        __m256 vb0 = _mm256_load_ps(&b_pack[k * JB + j]);

        vo0 = _mm256_fmadd_ps(va0,vb0,vo0);
        vo1 = _mm256_fmadd_ps(va1,vb0,vo1);
        vo2 = _mm256_fmadd_ps(va2,vb0,vo2);
        vo3 = _mm256_fmadd_ps(va3,vb0,vo3);
        vo4 = _mm256_fmadd_ps(va4,vb0,vo4);
        vo5 = _mm256_fmadd_ps(va5,vb0,vo5);
        vo6 = _mm256_fmadd_ps(va6,vb0,vo6);
        vo7 = _mm256_fmadd_ps(va7,vb0,vo7);
      }

      _mm256_store_ps(&out_pack[i * JB + j],vo0);
      _mm256_store_ps(&out_pack[(i + 1) * JB + j],vo1);
      _mm256_store_ps(&out_pack[(i + 2) * JB + j],vo2);
      _mm256_store_ps(&out_pack[(i + 3) * JB + j],vo3);
      _mm256_store_ps(&out_pack[(i + 4) * JB + j],vo4);
      _mm256_store_ps(&out_pack[(i + 5) * JB + j],vo5);
      _mm256_store_ps(&out_pack[(i + 6) * JB + j],vo6);
      _mm256_store_ps(&out_pack[(i + 7) * JB + j],vo7);
    }
  }*/

  for(int64_t i = 0;i < IB;i += 4){
    for(int64_t j = 0;j < JB;j += 32){
      __m256 vo00 = _mm256_setzero_ps();
      __m256 vo01 = _mm256_setzero_ps();
      __m256 vo02 = _mm256_setzero_ps();
      __m256 vo03 = _mm256_setzero_ps();

      __m256 vo10 = _mm256_setzero_ps();
      __m256 vo11 = _mm256_setzero_ps();
      __m256 vo12 = _mm256_setzero_ps();
      __m256 vo13 = _mm256_setzero_ps();

      __m256 vo20 = _mm256_setzero_ps();
      __m256 vo21 = _mm256_setzero_ps();
      __m256 vo22 = _mm256_setzero_ps();
      __m256 vo23 = _mm256_setzero_ps();

      __m256 vo30 = _mm256_setzero_ps();
      __m256 vo31 = _mm256_setzero_ps();
      __m256 vo32 = _mm256_setzero_ps();
      __m256 vo33 = _mm256_setzero_ps();

      for(int64_t k = 0;k < KB;k += 1){
        __m256 va0 = _mm256_set1_ps(a_pack[i * KB + k]);
        __m256 va1 = _mm256_set1_ps(a_pack[(i + 1) * KB + k]);
        __m256 va2 = _mm256_set1_ps(a_pack[(i + 2) * KB + k]);
        __m256 va3 = _mm256_set1_ps(a_pack[(i + 3) * KB + k]);

        __m256 vb0 = _mm256_load_ps(&b_pack[k * JB + j]);
        __m256 vb1 = _mm256_load_ps(&b_pack[k * JB + j + 8]);
        __m256 vb2 = _mm256_load_ps(&b_pack[k * JB + j + 16]);
        __m256 vb3 = _mm256_load_ps(&b_pack[k * JB + j + 24]);

        vo00 = _mm256_fmadd_ps(va0,vb0,vo00);
        vo01 = _mm256_fmadd_ps(va1,vb0,vo01);
        vo02 = _mm256_fmadd_ps(va2,vb0,vo02);
        vo03 = _mm256_fmadd_ps(va3,vb0,vo03);

        vo10 = _mm256_fmadd_ps(va0,vb1,vo10);
        vo11 = _mm256_fmadd_ps(va1,vb1,vo11);
        vo12 = _mm256_fmadd_ps(va2,vb1,vo12);
        vo13 = _mm256_fmadd_ps(va3,vb1,vo13);

        vo20 = _mm256_fmadd_ps(va0,vb2,vo20);
        vo21 = _mm256_fmadd_ps(va1,vb2,vo21);
        vo22 = _mm256_fmadd_ps(va2,vb2,vo22);
        vo23 = _mm256_fmadd_ps(va3,vb2,vo23);

        vo30 = _mm256_fmadd_ps(va0,vb3,vo30);
        vo31 = _mm256_fmadd_ps(va1,vb3,vo31);
        vo32 = _mm256_fmadd_ps(va2,vb3,vo32);
        vo33 = _mm256_fmadd_ps(va3,vb3,vo33);
      }

      _mm256_store_ps(&out_pack[i * JB + j],vo00);
      _mm256_store_ps(&out_pack[(i + 1) * JB + j],vo01);
      _mm256_store_ps(&out_pack[(i + 2) * JB + j],vo02);
      _mm256_store_ps(&out_pack[(i + 3) * JB + j],vo03);

      _mm256_store_ps(&out_pack[i * JB + j + 8],vo10);
      _mm256_store_ps(&out_pack[(i + 1) * JB + j + 8],vo11);
      _mm256_store_ps(&out_pack[(i + 2) * JB + j + 8],vo12);
      _mm256_store_ps(&out_pack[(i + 3) * JB + j + 8],vo13);

      _mm256_store_ps(&out_pack[i * JB + j + 16],vo20);
      _mm256_store_ps(&out_pack[(i + 1) * JB + j + 16],vo21);
      _mm256_store_ps(&out_pack[(i + 2) * JB + j + 16],vo22);
      _mm256_store_ps(&out_pack[(i + 3) * JB + j + 16],vo23);

      _mm256_store_ps(&out_pack[i * JB + j + 24],vo30);
      _mm256_store_ps(&out_pack[(i + 1) * JB + j + 24],vo31);
      _mm256_store_ps(&out_pack[(i + 2) * JB + j + 24],vo32);
      _mm256_store_ps(&out_pack[(i + 3) * JB + j + 24],vo33);
    }
  }
}

#endif

//パックする
inline void set_pack(const tensor::ConstMatrixView &m,
                     int64_t row,
                     int64_t col,
                     int64_t i_size,
                     int64_t j_size,
                     float*__restrict pack){

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

//outに数値を戻す
template<bool FirstK>
inline void write_out(const float alpha,
                      const float beta,
                      tensor::MatrixView &out,
                      int64_t row,
                      int64_t col,
                      int64_t i_size,
                      int64_t j_size,
                      const float*__restrict pack){

  const int64_t rs = out.row_stride();
  const int64_t cs = out.col_stride();

  float*__restrict od = out.base_ptr() + row * rs + col * cs;

  for(int64_t i = 0;i < i_size;i++){
    const float *pack_i = pack;
    float *odi = od;

    for(int64_t j = 0;j < j_size;j++){
      if constexpr(FirstK){
        *odi = alpha * (*pack_i) + beta * (*odi);
      }else{
        *odi += alpha * (*pack_i);
      }

      pack_i++;
      odi += cs;
    }

    pack += j_size;
    od += rs;
  }
}

//out = alpha * ab + beta * out
void gemm_impl(const float alpha,const tensor::ConstMatrixView &a,const tensor::ConstMatrixView &b,const float beta,tensor::MatrixView &out){
  const tensor::ConstMatrixView bt = b.t();

  //ブロックサイズ
  //kernelがそれを前提にしているため32の倍数のみにする
  const constexpr int64_t IB = 32;
  const constexpr int64_t JB = 32;
  const constexpr int64_t KB = 32;

  const int64_t orows = out.rows();
  const int64_t ocols = out.cols();
  const int64_t acols = a.cols();

  const int64_t a_row_stride = a.row_stride();
  const int64_t a_col_stride = a.col_stride();

  const int64_t bt_row_stride = bt.row_stride();
  const int64_t bt_col_stride = bt.col_stride();

  const int64_t out_row_stride = out.row_stride();
  const int64_t out_col_stride = out.col_stride();

  alignas(64) float a_pack[IB * KB];
  alignas(64) float b_pack[JB * KB];
  alignas(64) float out_pack[IB * JB];

  const int64_t ii_end = (out.rows() / IB) * IB;
  const int64_t jj_end = (out.cols() / JB) * JB;
  const int64_t kk_end = (a.cols() / KB) * KB;

  /*for(int64_t i = 0;i < out.rows();i++){
    for(int64_t j = 0;j < out.cols();j++){
      for(int64_t k = 0;k < a.cols();k++){
        if(k == 0){
          out.at(i,j) = alpha * a.at(i,k) * bt.at(j,k) + beta *  out.at(i,j);
        }else{
          out.at(i,j) += alpha * a.at(i,k) * bt.at(j,k);
        }
      }
    }
  }*/

  for(int64_t kk = 0;kk < kk_end;kk += KB){//kk + KB <= acols
    for(int64_t jj = 0;jj < jj_end;jj += JB){//jj + JB <= ocols
      set_pack(b,kk,jj,KB,JB,b_pack);
      //set_pack(bt,jj,kk,JB,KB,b_pack);

      for(int64_t ii = 0;ii < ii_end;ii += IB){
        set_pack(a,ii,kk,IB,KB,a_pack);

        kernel_avx2(a_pack,b_pack,out_pack,IB,JB,KB);

        if(kk == 0){
          write_out<true>(alpha,beta,out,ii,jj,IB,JB,out_pack);
        }else{
          write_out<false>(alpha,beta,out,ii,jj,IB,JB,out_pack);
        }
      }
    }
  }

  const float *ad = a.base_ptr();
  const float *btd = bt.base_ptr();
  float *od = out.base_ptr();

  //iの残り
  for(int64_t i = (orows / IB) * IB;i < out.rows();i++){
    const int64_t air = i * a_row_stride;

    for(int64_t j = 0;j < ocols;j++){
      float *odij = &out.at(i,j);
      const int64_t btjr = j * bt_row_stride;
      float sum = 0;

      for(int64_t k = 0;k < acols;k++){
        sum += ad[air + k * a_col_stride] * btd[btjr + k * bt_col_stride];
      }
      *odij = *odij * beta + sum * alpha;
    }
  }

  //jの残り
  for(int64_t i = 0;i < (orows / IB) * IB;i++){
    const int64_t air = i * a_row_stride;

    for(int64_t j = (ocols / JB) * JB;j < out.cols();j++){
      float *odij = &out.at(i,j);
      const int64_t btjr = j * bt_row_stride;
      float sum = 0;

      for(int64_t k = 0;k < acols;k++){
        sum += ad[air + k * a_col_stride] * btd[btjr + k * bt_col_stride];
      }
      *odij = *odij * beta + sum * alpha;
    }
  }

  //kの残り
  for(int64_t i = 0;i < (orows / IB) * IB;i++){
    const int64_t air = i * a_row_stride;

    for(int64_t j = 0;j < (ocols / JB) * JB;j++){
      float *odij = &out.at(i,j);
      const int64_t btjr = j * bt_row_stride;
      float sum = 0;

      for(int64_t k = (acols / KB) * KB;k < a.cols();k++){
        sum += ad[air + k * a_col_stride] * btd[btjr + k * bt_col_stride];
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