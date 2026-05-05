#pragma once

#include <immintrin.h>
#include "vec.hpp"
#include "nn/backend/backend.hpp"

namespace cobalt_715::nn::ops::vec{

//out[i] = a[i] + b[i]
template<>
inline void add_alias_safe<backend::CPU>(const float *a,const float *b,float *out,const size_t n) noexcept{
  #ifdef __AVX__
  size_t i = 0;
  for(;i + 8 <= n;i += 8){
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 vo = _mm256_add_ps(va,vb);
    _mm256_storeu_ps(out + i,vo);
  }
  for(;i < n;i++){
    out[i] = a[i] + b[i];
  }
  #else
  for(size_t i = 0;i < n;i++){
    out[i] = a[i] + b[i];
  }
  #endif
}

template<>
inline void add<backend::CPU>(const float*__restrict a,const float*__restrict b,float*__restrict out,const size_t n) noexcept{
  add_alias_safe<backend::CPU>(a,b,out,n);
}

//out[i] = a[i] - b[i]
template<>
inline void sub_alias_safe<backend::CPU>(const float *a,const float *b,float *out,const size_t n) noexcept{
  #ifdef __AVX__
  size_t i = 0;
  for(;i + 8 <= n;i += 8){
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 vo = _mm256_sub_ps(va,vb);
    _mm256_storeu_ps(out + i,vo);
  }
  for(;i < n;i++){
    out[i] = a[i] - b[i];
  }
  #else
  for(size_t i = 0;i < n;i++){
    out[i] = a[i] - b[i];
  }
  #endif
}

template<>
inline void sub<backend::CPU>(const float*__restrict a,const float*__restrict b,float*__restrict out,const size_t n) noexcept{
  sub_alias_safe<backend::CPU>(a,b,out,n);
}

//out[i] = a[i] * b[i]
template<>
inline void mul_alias_safe<backend::CPU>(const float *a,const float *b,float *out,const size_t n) noexcept{
  #ifdef __AVX__
  size_t i = 0;
  for(;i + 8 <= n;i += 8){
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 vo = _mm256_mul_ps(va,vb);
    _mm256_storeu_ps(out + i,vo);
  }
  for(;i < n;i++){
    out[i] = a[i] * b[i];
  }
  #else
  for(size_t i = 0;i < n;i++){
    out[i] = a[i] * b[i];
  }
  #endif
}

template<>
inline void mul<backend::CPU>(const float*__restrict a,const float*__restrict b,float*__restrict out,const size_t n) noexcept{
  mul_alias_safe<backend::CPU>(a,b,out,n);
}

//out[i] = a[i] / b[i]
template<>
inline void div_alias_safe<backend::CPU>(const float *a,const float *b,float *out,const size_t n) noexcept{
  #ifdef __AVX__
  size_t i = 0;
  for(;i + 8 <= n;i += 8){
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 vo = _mm256_div_ps(va,vb);
    _mm256_storeu_ps(out + i,vo);
  }
  for(;i < n;i++){
    out[i] = a[i] / b[i];
  }
  #else
  for(size_t i = 0;i < n;i++){
    out[i] = a[i] / b[i];
  }
  #endif
}

template<>
inline void div<backend::CPU>(const float*__restrict a,const float*__restrict b,float*__restrict out,const size_t n) noexcept{
  div_alias_safe<backend::CPU>(a,b,out,n);
}

//out[i] = a[i] * b[i] + c[i]
template<>
inline void fma_alias_safe<backend::CPU>(const float *a,const float *b,const float *c,float *out,const size_t n) noexcept{
  #ifdef __AVX__
  size_t i = 0;
  for(;i + 8 <= n;i += 8){
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 vc = _mm256_loadu_ps(c + i);
    __m256 vo = _mm256_fmadd_ps(va,vb,vc);
    _mm256_storeu_ps(out + i,vo);
  }
  for(;i < n;i++){
    out[i] = a[i] * b[i] + c[i];
  }
  #else
  for(size_t i = 0;i < n;i++){
    out[i] = a[i] * b[i] + c[i];
  }
  #endif
}

template<>
inline void fma<backend::CPU>(const float*__restrict a,const float*__restrict b,const float*__restrict c,float*__restrict out,const size_t n) noexcept{
  fma_alias_safe<backend::CPU>(a,b,c,out,n);
}

//out[i] = a[i] * c
template<>
inline void scale_alias_safe<backend::CPU>(const float *a,const float c,float *out,const size_t n) noexcept{
  #ifdef __AVX__
  __m256 vc = _mm256_set1_ps(c);
  size_t i = 0;
  for(;i + 8 <= n;i += 8){
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vo = _mm256_mul_ps(va,vc);
    _mm256_storeu_ps(out + i,vo);
  }
  for(;i < n;i++){
    out[i] = a[i] * c;
  }
  #else
  for(size_t i = 0;i < n;i++){
    out[i] = a[i] * c;
  }
  #endif
}

template<>
inline void scale<backend::CPU>(const float*__restrict a,const float c,float*__restrict out,const size_t n) noexcept{
  scale_alias_safe(a,c,out,n);
}

//out[i] = a[i] * c + b[i]
template<>
inline void axpy_alias_safe<backend::CPU>(const float *a,const float c,const float *b,float *out,const size_t n) noexcept{
  #ifdef __AVX__
  __m256 vc = _mm256_set1_ps(c);
  size_t i = 0;
  for(;i + 8 <= n;i += 8){
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 vo = _mm256_fmadd_ps(va,vc,vb);
    _mm256_storeu_ps(out + i,vo);
  }
  for(;i < n;i++){
    out[i] = a[i] * c + b[i];
  }
  #else
  for(size_t i = 0;i < n;i++){
    out[i] = a[i] * c + b[i];
  }
  #endif
}

template<>
inline void axpy<backend::CPU>(const float*__restrict a,const float c,const float*__restrict b,float*__restrict out,const size_t n) noexcept{
  axpy_alias_safe(a,c,b,out,n);
}

}//namespace cobalt_715::nn::ops::vec