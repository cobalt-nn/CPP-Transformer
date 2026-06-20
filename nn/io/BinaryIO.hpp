#pragma once

#include <iostream>
#include <cstdint>
#include <stdexcept>

namespace cobalt_715::nn::io{

template<typename T>
void save(std::ostream &os,const T *data,const uint64_t size){
  if(!os){
    throw std::runtime_error("io::save: stream is not ready before write");
  }

  os.write(reinterpret_cast<const char*>(&size),sizeof(size));

  if(!os){
    throw std::runtime_error("io::save: failed while writing size");
  }

  if(size > 0){
    os.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size * sizeof(T)));
    if(!os){
      throw std::runtime_error("io::save: failed while writing data");
    }
  }
}

template<typename T>
void load(std::istream &is,T *data,const uint64_t size){
  if(!is){
    throw std::runtime_error("io::load: stream is not ready before read");
  }

  uint64_t num = 0;

  is.read(reinterpret_cast<char*>(&num),sizeof(num));

  if(!is){
    throw std::runtime_error("io::load: failed while reading size");
  }

  if(num != size){
    throw std::runtime_error("io::load size mismatch");
  }

  if(size > 0){
    is.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(size * sizeof(T)));
    if(!is){
      throw std::runtime_error("io::load: failed while reading data");
    }
  }
}

}//namespace cobalt_715::nn::io