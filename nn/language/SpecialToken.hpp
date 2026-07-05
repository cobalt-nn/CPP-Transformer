#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace cobalt_715::nn::language::token{

inline constexpr std::string PAD = "<pad>";
inline constexpr std::string UNK = "<unk>";
inline constexpr std::string BOS = "<bos>";
inline constexpr std::string EOS = "<eos>";
inline constexpr std::string CAP = "<cap>";//先頭が大文字かどうか
inline constexpr std::string ALL_CAP = "<all_cap>";//すべて大文字かどうか
inline constexpr std::string USER = "<user>";
inline constexpr std::string ASSISTANT = "<assistant>";
inline constexpr std::string SYSTEM = "<system>";

inline const std::vector<std::string> stokens =
  {
    PAD,
    UNK,
    BOS,
    EOS,
    CAP,
    ALL_CAP,
    USER,
    ASSISTANT,
    SYSTEM
  };

}//namespace cobalt_715::nn::language::tolen