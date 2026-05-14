#pragma once

#include <iostream>
#include <string>
#include <cmath>

namespace cobalt_715::nn::ops{

//活性化関数とその微分を保持する
struct Activation{
  const std::string name;//活性化関数名
  float (*act_)(float);//活性化関数
  float (*d_act_)(float z,float a);//微分。様々な微分に対応するため微分に必要な情報を活性化前、活性化後の順で受け取る
};

//基本的な活性化関数をまとめている
namespace activations{

inline const Activation Sigmoid{
  "Sigmoid",
  [](float x){
    return 1.0f / (1.0f + std::exp(-x));
  },
  [](float z,float a){
    return a * (1.0f - a);
  }
};

inline const Activation tanh{
  "tanh",
  [](float x){
    return std::tanh(x);
  },
  [](float z,float a){
    return 1.0f - a * a;
  }
};

inline const Activation ReLU{
  "ReLU",
  [](float x){
    return (0.0f < x) ? x:0.0f;
  },
  [](float z,float a){
    return (0.0f < z) ? 1.0f:0.0f;
  }
};

inline const Activation LeakyReLU{
  "LeakyReLU",
  [](float x){
    return (0.0f < x) ? x:0.01f * x;;
  },
  [](float z,float a){
    return (0.0f < z) ? 1.0f:0.01f;
  }
};

inline const Activation Straight_Through_Estimator{
  "Straight_Through_Estimator",
  [](float x){
    return (0.0f < x) ? 1.0f:0.0f;
  },
  [](float z,float a){
    return 1.0f;
  }
};

}//namespace activations

}//namespace cobalt_715::nn::ops