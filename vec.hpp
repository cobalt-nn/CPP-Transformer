#pragma once

#include "backend.hpp"

//基本的なベクトル演算を用意している
namespace cobalt_715::nn::linear{

//out = a + b
template<typename BackendT = backend::CPU>
inline void add_safe(const float *a,const float *b,float *out,const size_t n) = delete;

template<typename BackendT = backend::CPU>
inline void add(const float*__restrict a,const float*__restrict b,float*__restrict out,const size_t n) = delete;

//out = a - b
template<typename BackendT = backend::CPU>
inline void sub_safe(const float *a,const float *b,float *out,const size_t n) = delete;

template<typename BackendT = backend::CPU>
inline void sub(const float*__restrict a,const float*__restrict b,float*__restrict out,const size_t n) = delete;

//out = a * b
template<typename BackendT = backend::CPU>
inline void mul_safe(const float *a,const float *b,float *out,const size_t n) = delete;

template<typename BackendT = backend::CPU>
inline void mul(const float*__restrict a,const float*__restrict b,float*__restrict out,const size_t n) = delete;

//out = a / b
template<typename BackendT = backend::CPU>
inline void div_safe(const float *a,const float *b,float *out,const size_t n) = delete;

template<typename BackendT = backend::CPU>
inline void div(const float*__restrict a,const float*__restrict b,float*__restrict out,const size_t n) = delete;

//out = a * b + c
template<typename BackendT = backend::CPU>
inline void fma_safe(const float *a,const float *b,const float *c,float *out,const size_t n) = delete;

template<typename BackendT = backend::CPU>
inline void fma(const float*__restrict a,const float*__restrict b,const float*__restrict c,float*__restrict out,const size_t n) = delete;

//out[i] = a[i] * c
template<typename BackendT = backend::CPU>
inline void scale_safe(const float *a,const float c,float *out,const size_t n) = delete;

template<typename BackendT = backend::CPU>
inline void scale(const float*__restrict a,const float c,float*__restrict out,const size_t n) = delete;

//out[i] = a[i] * c + b[i]
template<typename BackendT = backend::CPU>
inline void axpy_safe(const float *a,const float c,const float *b,float *out,const size_t n) = delete;

template<typename BackendT = backend::CPU>
inline void axpy(const float*__restrict a,const float c,const float*__restrict b,float*__restrict out,const size_t n) = delete;

}//namespace cobalt_715::nn::linear