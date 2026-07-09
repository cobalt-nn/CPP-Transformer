#pragma once

#include <iostream>
#include <unordered_map>
#include <string>
#include "Activation.hpp"

namespace cobalt_715::nn::ops{

//Mapを使い登録した活性化関数を取り出せるようにしている
//addAct(Activation act)から任意の関数を登録できる
class Acts{
private:
  inline static std::unordered_map<std::string,Activation> acts;

  //デフォルトでの関数を登録する
  static void init(){
    static bool initialized = [](){
      addAct(activations::identity);
      addAct(activations::Sigmoid);
      addAct(activations::sin);
      addAct(activations::cos);
      addAct(activations::tan);
      addAct(activations::asin);
      addAct(activations::acos);
      addAct(activations::atan);
      addAct(activations::sinh);
      addAct(activations::cosh);
      addAct(activations::tanh);
      addAct(activations::ReLU);
      addAct(activations::LeakyReLU);
      addAct(activations::SiLU);
      addAct(activations::GELU);
      addAct(activations::Straight_Through_Estimator);
      addAct(activations::exp);
      addAct(activations::abs);
      addAct(activations::square);
      addAct(activations::cube);
      return true;
    }();
  }

public:
  //任意の関数を登録できる
  static void addAct(const Activation &act){
    acts.emplace(act.name, act);
  }

  //渡した名前の関数を取得できる
  static const Activation& getAct(const std::string &name){
    init();
    auto it = acts.find(name);
    if(it != acts.end()){
      return it->second;
    }

    std::cout << "The specified activation function does not exist. Returning LeakyReLU" << std::endl;
    return acts["LeakyReLU"];
  }
};

}//namespace cobalt_715::nn::ops