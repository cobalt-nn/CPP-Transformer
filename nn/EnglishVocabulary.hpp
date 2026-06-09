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
  EnglishVocabulary(){
    for(std::string s:symbol_){
      prefix_.push_back(s);
      suffix_.push_back(s);
    }
  }

  //stringをtokenに分解
  std::vector<std::string> tokenize(const std::string& text) const{
    std::vector<std::string> tokens;
    std::string s;

    std::stringstream ss(text);

    while(std::getline(ss,s,' ')){
      {
        bool start = std::isupper(static_cast<unsigned char>(s[0]));
        size_t upper = 0;
        size_t alpha = 0;

        for(char &cr:s){
          if(std::isalpha(cr)){
            alpha++;

            if(std::isupper(cr)){
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

      //全て小文字とする
      std::transform(
        s.begin(), 
        s.end(), 
        s.begin(),
        [](unsigned char c){return std::tolower(c);}
      );

      //小さい文字はほぼそのまま追加
      if(s.size() < 6){
        tokens.push_back(s);
        tokens.push_back(" ");
        continue;
      }

      //接頭辞で分割する
      for(const std::string &fix:prefix_){
        if(s.starts_with(fix)){
          tokens.push_back(fix);

          s = s.substr(fix.size());

          break;
        }
      }

      std::string suf;

      //接尾辞で分割する
      for(const std::string &fix:suffix_){
        if(s.ends_with(fix)){
          suf = fix;

          s = s.substr(0,s.size() - fix.size());

          break;
        }
      }

      tokens.push_back(s);

      if(!suf.empty()){
        tokens.push_back(suf);
      }

      tokens.push_back(" ");
    }

    tokens.pop_back();//最後のスペースを削除する
    return tokens;
  }

private:

  const std::string CAP_ = "<CAP>";//先頭が大文字かどうか
  const std::string ALL_CAP_ = "<ALL_CAP>";//すべて大文字かどうか

  std::vector<std::string> symbol_ = {
    "%=",
    "*=",
    "++",
    "+=",
    "--",
    "-=",
    "/=",
    "<<",
    "==",
    ">>",
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
    "up",
    "a"
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
    "'s",
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
    "s'",
    "se",
    "th",
    "ty",
    "d",
    "s",
    "y"
  };
};

}//namespace cobalt_715::nn