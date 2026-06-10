#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <cctype>
#include <algorithm>

namespace cobalt_715::nn{

struct EnglishVocabulary{
  //stringをtokenに分解
  std::vector<std::string> tokenize(const std::string_view text) const{
    std::vector<std::string> tokens;

    size_t be = 0;
    size_t next_be = 0;
    size_t en = 0;

    while(be < text.size()){
      std::string_view back_sym;

      //symbol_で分割
      for(size_t i = be;i < text.size();i++){
        for(const std::string_view sym:symbol_){
          const std::string_view view = text.substr(i,sym.size());

          if(view == sym){
            en = i;
            next_be = en + sym.size();
            back_sym = sym;
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

        if(alpha > 0 && upper == alpha){
          tokens.push_back(ALL_CAP_);
        }else if(start){
          tokens.push_back(CAP_);
        }
      }

      //prefix
      for(const std::string_view pre:prefix_){
        if(sv.starts_with(pre)){
          tokens.push_back(std::string(pre));
          sv.remove_prefix(pre.size());
          break;
        }
      }

      std::string suffix;

      //suffix
      for(const std::string_view suf:suffix_){
        if(sv.ends_with(suf)){
          suffix = std::string(suf);
          sv.remove_suffix(suf.size());
          break;
        }
      }

      if(!sv.empty()){
        tokens.push_back(std::string(sv));
      }

      if(!suffix.empty()){
        tokens.push_back(std::string(suffix));
      }

      if(!back_sym.empty()){
        tokens.push_back(std::string(back_sym));
      }

      be = next_be;
    }

    return tokens;
  }

private:

  const std::string CAP_ = "<CAP>";//先頭が大文字かどうか
  const std::string ALL_CAP_ = "<ALL_CAP>";//すべて大文字かどうか

  std::vector<std::string> symbol_ = {
    "'ll",
    "'re",
    "'ve",
    "<<=",
    "===",
    ">>=",
    "n't",
    "%=",
    "'d",
    "'m",
    "'s",
    "'t",
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
  std::vector<std::string> prefix_ = {
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
  std::vector<std::string> suffix_ = {
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

}//namespace cobalt_715::nn