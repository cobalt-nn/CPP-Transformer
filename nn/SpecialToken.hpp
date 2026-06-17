#pragma once

#include <string>
#include <vector>

namespace cobalt_715::nn::token{

inline constexpr std::string PAD = "<PAD>";
inline constexpr std::string UNK = "<UNK>";
inline constexpr std::string BOS = "<BOS>";
inline constexpr std::string EOS = "<EOS>";
inline constexpr std::string CAP = "<CAP>";//先頭が大文字かどうか
inline constexpr std::string ALL_CAP = "<ALL_CAP>";//すべて大文字かどうか

inline const std::vector<std::string> stokens =
  {
    PAD,
    UNK,
    BOS,
    EOS,
    CAP,
    ALL_CAP
  };

}//namespace cobalt_715::nn::token