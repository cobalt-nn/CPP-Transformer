#pragma once

#include <string>

namespace cobalt_715::nn::token{

inline constexpr std::string PAD = "<PAD>";
inline constexpr std::string UNK = "<UNK>";
inline constexpr std::string BOS = "<BOS>";
inline constexpr std::string EOS = "<EOS>";
inline constexpr std::string CAP = "<CAP>";//先頭が大文字かどうか
inline constexpr std::string ALL_CAP = "<ALL_CAP>";//すべて大文字かどうか

}//namespace cobalt_715::nn::token