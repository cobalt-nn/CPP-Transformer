#include <cstdint>
#include "nn/tensor/MatrixView.hpp"
#include "nn/tensor/ConstMatrixView.hpp"

namespace cobalt_715::nn::ops{

template<bool FirstK>
void kernel(const float*__restrict a_pack,const float*__restrict bt_pack,const float alpha,const float beta,tensor::MatrixView &out,const int64_t IB,const int64_t JB,const int64_t KB,const int64_t ii,const int64_t jj){
  for(int64_t i = 0;i < IB;i++){
    for(int64_t j = 0;j < JB;j++){
      float sum = 0;

      for(int64_t k = 0;k < KB;k++){
        sum += a_pack[i * KB + k] * bt_pack[j * KB + k];
      }

      if constexpr(FirstK){
        out.at(ii + i,jj + j) = out.at(ii + i,jj + j) * beta + sum * alpha;
      }else{
        out.at(ii + i,jj + j) += sum * alpha;
      }
    }
  }
}

template void kernel<true>(const float*__restrict a_pack,const float*__restrict bt_pack,const float alpha,const float beta,tensor::MatrixView &out,const int64_t IB,const int64_t JB,const int64_t KB,const int64_t ii,const int64_t jj);

template void kernel<false>(const float*__restrict a_pack,const float*__restrict bt_pack,const float alpha,const float beta,tensor::MatrixView &out,const int64_t IB,const int64_t JB,const int64_t KB,const int64_t ii,const int64_t jj);

//out = alpha * ab + beta * out
void gemm_impl(const float alpha,const tensor::ConstMatrixView &a,const tensor::ConstMatrixView &b,const float beta,tensor::MatrixView &out){
  //ブロックサイズ
  const int64_t IB = 32;
  const int64_t JB = 32;
  const int64_t KB = 32;

  const tensor::ConstMatrixView bt = b.t();

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

  alignas(64) float a_pack[IB * KB];
  alignas(64) float bt_pack[JB * KB];

  for(int64_t jj = 0;jj + JB <= out.cols();jj += JB){
    for(int64_t kk = 0;kk + KB <= a.cols();kk += KB){
      set_pack(bt,jj,kk,JB,KB,bt_pack);

      for(int64_t ii = 0;ii + IB <= out.rows();ii += IB){
        set_pack(a,ii,kk,IB,KB,a_pack);
        if(kk == 0){
          kernel<true>(a_pack,bt_pack,alpha,beta,out,IB,JB,KB,ii,jj);
        }else{
          kernel<false>(a_pack,bt_pack,alpha,beta,out,IB,JB,KB,ii,jj);
        }
      }
    }
  }

  //iの残り
  for(int64_t i = (out.rows() / IB) * IB;i < out.rows();i++){
    for(int64_t j = 0;j < out.cols();j++){
      for(int64_t k = 0;k < a.cols();k++){
        if(k == 0){
          out.at(i,j) = out.at(i,j) * beta + a.at(i,k) * bt.at(j,k) * alpha;
        }else{
          out.at(i,j) += a.at(i,k) * bt.at(j,k) * alpha;
        }
      }
    }
  }

  //jの残り
  for(int64_t i = 0;i < (out.rows() / IB) * IB;i++){
    for(int64_t j = (out.cols() / JB) * JB;j < out.cols();j++){
      for(int64_t k = 0;k < a.cols();k++){
        if(k == 0){
          out.at(i,j) = out.at(i,j) * beta + a.at(i,k) * bt.at(j,k) * alpha;
        }else{
          out.at(i,j) += a.at(i,k) * bt.at(j,k) * alpha;
        }
      }
    }
  }

  //kの残り
  for(int64_t i = 0;i < (out.rows() / IB) * IB;i++){
    for(int64_t j = 0;j < (out.cols() / JB) * JB;j++){
      for(int64_t k = (a.cols() / KB) * KB;k < a.cols();k++){
        if(k == 0){
          out.at(i,j) = out.at(i,j) * beta + a.at(i,k) * bt.at(j,k) * alpha;
        }else{
          out.at(i,j) += a.at(i,k) * bt.at(j,k) * alpha;
        }
      }
    }
  }
}

}//namespace cobalt_715::nn::ops