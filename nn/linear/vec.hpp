#pragma once

#include "nn/backend/backend.hpp"

//基本的なベクトル演算を用意している
namespace cobalt_715::nn::linear::vec{

//out[i] = a[i] + b[i]
template<typename BackendT = backend::CPU>
inline void add_alias_safe(const float *a,const float *b,float *out,const size_t n) noexcept = delete;

template<typename BackendT = backend::CPU>
inline void add(const float*__restrict a,const float*__restrict b,float*__restrict out,const size_t n) noexcept = delete;

//out[i] = a[i] - b[i]
template<typename BackendT = backend::CPU>
inline void sub_alias_safe(const float *a,const float *b,float *out,const size_t n) noexcept = delete;

template<typename BackendT = backend::CPU>
inline void sub(const float*__restrict a,const float*__restrict b,float*__restrict out,const size_t n) noexcept = delete;

//out[i] = a[i] * b[i]
template<typename BackendT = backend::CPU>
inline void mul_alias_safe(const float *a,const float *b,float *out,const size_t n) noexcept = delete;

template<typename BackendT = backend::CPU>
inline void mul(const float*__restrict a,const float*__restrict b,float*__restrict out,const size_t n) noexcept = delete;

//out[i] = a[i] / b[i]
template<typename BackendT = backend::CPU>
inline void div_alias_safe(const float *a,const float *b,float *out,const size_t n) noexcept = delete;

template<typename BackendT = backend::CPU>
inline void div(const float*__restrict a,const float*__restrict b,float*__restrict out,const size_t n) noexcept = delete;

//out[i] = a[i] * b[i] + c[i]
template<typename BackendT = backend::CPU>
inline void fma_alias_safe(const float *a,const float *b,const float *c,float *out,const size_t n) noexcept = delete;

template<typename BackendT = backend::CPU>
inline void fma(const float*__restrict a,const float*__restrict b,const float*__restrict c,float*__restrict out,const size_t n) noexcept = delete;

//out[i] = a[i] * c
template<typename BackendT = backend::CPU>
inline void scale_alias_safe(const float *a,const float c,float *out,const size_t n) noexcept = delete;

template<typename BackendT = backend::CPU>
inline void scale(const float*__restrict a,const float c,float*__restrict out,const size_t n) noexcept = delete;

//out[i] = a[i] * c + b[i]
template<typename BackendT = backend::CPU>
inline void axpy_alias_safe(const float *a,const float c,const float *b,float *out,const size_t n) noexcept = delete;

template<typename BackendT = backend::CPU>
inline void axpy(const float*__restrict a,const float c,const float*__restrict b,float*__restrict out,const size_t n) noexcept = delete;

}//namespace cobalt_715::nn::linear