#pragma once

namespace cobalt_715::nn::tensor::detail{

//OVERLAPPEDの時、代入禁止にするため
class ElementRef{
  float* ptr_;

  #ifndef NDEBUG
    bool write_ok_;
  #endif
public:
  #ifndef NDEBUG
  inline ElementRef(float* ptr,bool b){
    ptr_ = ptr;
    write_ok_ = b;
  }
  #endif

  inline ElementRef(float* ptr){
    ptr_ = ptr;
  }

  inline operator float() const { return *ptr_; }

  inline ElementRef& operator=(float v){
    #ifndef NDEBUG
      if(!write_ok_){
        throw std::logic_error("Write to overlapped ElementRef");
      }
    #endif
    *ptr_ = v;
    return *this;
  }

  inline ElementRef& operator+=(float v){
    #ifndef NDEBUG
      if(!write_ok_){
        throw std::logic_error("Write to overlapped ElementRef");
      }
    #endif
    *ptr_ += v;
    return *this;
  }

  inline ElementRef& operator-=(float v){
    #ifndef NDEBUG
      if(!write_ok_){
        throw std::logic_error("Write to overlapped ElementRef");
      }
    #endif
    *ptr_ -= v;
    return *this;
  }

  inline ElementRef& operator*=(float v){
    #ifndef NDEBUG
      if(!write_ok_){
        throw std::logic_error("Write to overlapped ElementRef");
      }
    #endif
    *ptr_ *= v;
    return *this;
  }

  inline ElementRef& operator/=(float v){
    #ifndef NDEBUG
      if(!write_ok_){
        throw std::logic_error("Write to overlapped ElementRef");
      }
    #endif
    *ptr_ /= v;
    return *this;
  }
};

}//namespace cobalt_715::nn::tensor::detail