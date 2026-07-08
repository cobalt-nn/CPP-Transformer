#pragma once

#include <vector>
#include <string>
#include <cstdint>

namespace cobalt_715::nn::language{

struct Tokens{
  Tokens(std::vector<std::string> s) : v_(s){}

  std::vector<std::string> v_;
};

}//namespace cobalt_715::nn::language