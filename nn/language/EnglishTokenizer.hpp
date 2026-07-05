#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cctype>
#include <algorithm>
#include <cstddef>
#include "SpecialToken.hpp"

namespace cobalt_715::nn::language{

struct EnglishTokenizer{
  std::vector<std::string> format(const std::string_view text,const int64_t max_len) const{
    std::vector<std::string> tokens = tokenize(text);

    while(tokens.size() < max_len){
      tokens.push_back(token::PAD);
    }

    tokens.resize(max_len);

    return tokens;
  }

  //stringをtokenに分解
  std::vector<std::string> tokenize(const std::string_view text) const{
    std::vector<std::string> tokens;

    size_t be = 0;
    size_t next_be = 0;
    size_t en = 0;

    while(be < text.size()){
      std::string back_sym;

      //symbol_で分割
      for(size_t i = be;i < text.size();i++){
        for(const std::string_view sym:symbol_){
          const std::string_view view = text.substr(i,sym.size());

          if(view == sym){
            en = i;
            next_be = en + sym.size();
            back_sym = std::string(sym);
            //std::cout << "\nen" << en << "\nnext_be" << next_be << "\n" << std::endl;

            goto hell;
          }
        }
        next_be = en = text.size();
      }

      hell:

      std::string base(text.substr(be,en - be));

      std::string_view sv = base;

      {
        bool start = std::isupper(static_cast<unsigned char>(base[0]));
        size_t upper = 0;
        size_t alpha = 0;

        for(char &cr:base){
          if(std::isalpha(static_cast<unsigned char>(cr))){
            alpha++;

            if(std::isupper(static_cast<unsigned char>(cr))){
              upper++;
            }
          }

          cr = std::tolower(static_cast<unsigned char>(cr));
        }

        if(alpha > 0 && upper == alpha && base.size() != 1){
          tokens.push_back(token::ALL_CAP);
        }else if(start){
          tokens.push_back(token::CAP);
        }
      }

      std::string suffix;

      if(sv.size() < 6) goto hell2;

      //prefix
      for(const std::string_view pre:prefix_){
        if(sv.starts_with(pre)){
          tokens.push_back(std::string(pre));
          sv.remove_prefix(pre.size());
          break;
        }
      }

      //suffix
      for(const std::string_view suf:suffix_){
        if(sv.ends_with(suf)){
          suffix = std::string(suf);
          sv.remove_suffix(suf.size());
          break;
        }
      }

      hell2:

      if(!sv.empty()){
        tokens.push_back(std::string(sv));
      }

      if(!suffix.empty()){
        tokens.push_back(std::string(suffix));
      }

      if(!back_sym.empty()){
        bool cap = false;
        for(char &cr:back_sym){
          if(std::isupper(static_cast<unsigned char>(cr))) cap = true;
          cr = std::tolower(static_cast<unsigned char>(cr));
        }

        if(cap) tokens.push_back(token::ALL_CAP);

        tokens.push_back(back_sym);
      }

      be = next_be;
    }

    return tokens;
  }

  //tokenizeした結果をstringに戻す
  std::string detokenize(const std::vector<std::string> &tokens) const{
    std::string text;
    bool cap = false;
    bool all_cap = false;
    bool symbol = false;
    int64_t all_cap_count = 0;

    for(std::string s:tokens){
      symbol = false;

      if(s == token::CAP){
        cap = true;
        continue;
      }else if(s == token::ALL_CAP){
        all_cap = true;
        continue;
      }

      for(const std::string &sym:symbol_){
        if(s == sym){
          symbol = true;
          if(all_cap_count > 0){
            all_cap = false;
          }
          break;
        }
      }

      bool is_special = false;
      for(const std::string &ss:token::stokens){
        if(s == ss) is_special = true;
      }
      if(is_special) continue;

      if(cap){
        s[0] = std::toupper(static_cast<unsigned char>(s[0]));
         cap = false;
      }else if(all_cap){
        std::transform(
          s.begin(), 
          s.end(), 
          s.begin(),
          [](char c) {return std::toupper(static_cast<unsigned char>(c));}
        );

        if(symbol){
          all_cap = false;
          all_cap_count = 0;
        }else{
          all_cap_count++;
        }
      }
      text += s;
    }
    return text;
  }

private:
  inline static const std::vector<std::string> symbol_ = {
    token::PAD,
    token::UNK,
    token::BOS,
    token::EOS,
    token::CAP,
    token::ALL_CAP,
    token::USER,
    token::ASSISTANT,
    token::SYSTEM,
    "'ll",
    "'LL",
    "'re",
    "'RE",
    "'ve",
    "'VE",
    "<<=",
    "===",
    ">>=",
    "n't",
    "N'T",
    "%=",
    "'d",
    "'D",
    "'m",
    "'M",
    "'s",
    "'S",
    "'t",
    "'T",
    "*=",
    "++",
    "+=",
    "--",
    "-=",
    "/=",
    "<<",
    "==",
    ">>",
    "s'",
    "S'",
    " ",
    "!",
    "\"",
    "#",
    "$",
    "%",
    "&",
    "'",
    "(",
    ")",
    "*",
    "+",
    ",",
    "-",
    ".",
    "/",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    ":",
    ";",
    "<",
    "=",
    ">",
    "?",
    "@",
    "[",
    "\\",
    "]",
    "^",
    "_",
    "`",
    "{",
    "|",
    "}",
    "~"
  };

  //prefix
  //https://tanzam-dict.net/ja/en/articles/prefixes-in-english参考
  inline static const std::vector<std::string> prefix_ = {
    "counter",
    "circum",
    "contra",
    "hetero",
    "pseudo",
    "centi",
    "extra",
    "hyper",
    "inter",
    "intra",
    "intro",
    "macro",
    "micro",
    "milli",
    "multi",
    "retro",
    "super",
    "trans",
    "ultra",
    "under",
    "ante",
    "anti",
    "arch",
    "auto",
    "down",
    "ever",
    "fore",
    "hemi",
    "homo",
    "hypo",
    "kilo",
    "mega",
    "meta",
    "mono",
    "over",
    "para",
    "peri",
    "poly",
    "post",
    "quad",
    "semi",
    "ann",
    "com",
    "con",
    "dia",
    "dis",
    "enn",
    "mal",
    "neo",
    "non",
    "out",
    "pan",
    "per",
    "pre",
    "pro",
    "sub",
    "sym",
    "syn",
    "tri",
    "uni",
    "ab",
    "ad",
    "bi",
    "co",
    "de",
    "em",
    "en",
    "ex",
    "il",
    "im",
    "in",
    "ir",
    "ob",
    "re",
    "un",
    "up"
  };

  //suffix
  //https://mage8.com/tango/column8.html参考
  inline static const std::vector<std::string> suffix_ = {
    "fication",
    "ability",
    "ibility",
    "isation",
    "ization",
    "manship",
    "philiac",
    "bility",
    "escent",
    "graphy",
    "handed",
    "person",
    "philia",
    "phobia",
    "selves",
    "sphere",
    "worthy",
    "archy",
    "arian",
    "aster",
    "ation",
    "ative",
    "cracy",
    "craft",
    "drome",
    "esque",
    "graph",
    "ician",
    "iform",
    "itive",
    "itude",
    "lysis",
    "mancy",
    "mania",
    "meter",
    "metry",
    "onomy",
    "osity",
    "pathy",
    "phile",
    "phobe",
    "phone",
    "phony",
    "proof",
    "scape",
    "scope",
    "speak",
    "tious",
    "ulous",
    "wards",
    "able",
    "ably",
    "ance",
    "ancy",
    "arch",
    "cide",
    "cule",
    "ence",
    "ency",
    "eous",
    "erel",
    "esce",
    "ette",
    "fold",
    "form",
    "free",
    "gamy",
    "gate",
    "gram",
    "hood",
    "ible",
    "ibly",
    "iour",
    "itis",
    "less",
    "like",
    "ling",
    "ment",
    "most",
    "ness",
    "nomy",
    "osis",
    "phil",
    "self",
    "ship",
    "some",
    "ster",
    "tion",
    "tude",
    "ular",
    "ward",
    "ways",
    "wide",
    "wise",
    "acy",
    "ade",
    "age",
    "ant",
    "ard",
    "ate",
    "ble",
    "bly",
    "cle",
    "cum",
    "dom",
    "eer",
    "ent",
    "ere",
    "ern",
    "ery",
    "ese",
    "ess",
    "est",
    "eth",
    "fic",
    "ful",
    "gen",
    "gon",
    "ial",
    "ian",
    "ics",
    "ier",
    "ify",
    "ile",
    "ine",
    "ing",
    "ion",
    "ior",
    "ise",
    "ish",
    "ist",
    "ite",
    "ity",
    "ium",
    "ive",
    "ize",
    "let",
    "man",
    "men",
    "nik",
    "ock",
    "oid",
    "ory",
    "ose",
    "our",
    "ous",
    "pie",
    "red",
    "rel",
    "tor",
    "ule",
    "ure",
    "yer",
    "al",
    "an",
    "ar",
    "ce",
    "cy",
    "ed",
    "ee",
    "en",
    "er",
    "es",
    "ey",
    "fy",
    "id",
    "ie",
    "in",
    "le",
    "ly",
    "or",
    "ry",
    "se",
    "th",
    "ty",
    "d",
    "s",
    "y"
  };
};

}//namespace cobalt_715::nn::language