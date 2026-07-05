#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <optional>
#include <cstddef>
#include "EnglishTokenizer.hpp"
#include "Vocabulary.hpp"
#include "Embedding.hpp"
#include "SpecialToken.hpp"
#include "nn/tensor/Tensor.hpp"
#include "nlohmann/json.hpp"

namespace cobalt_715::nn::language{

struct Language{
  EnglishTokenizer et;
  Vocabulary voc;
  std::optional<Embedding> emb = std::nullopt;

  const tensor::Tensor& forward(const std::vector<std::string> &tokens,const int64_t max_len,bool training=true){
    std::vector<std::vector<std::string>> ts;

    for(const std::string &s:tokens){
      ts.push_back(et.format(s,max_len));
    }

    return forward(ts,training);
  }

  //tokenize済みのものを受け取る
  const tensor::Tensor& forward(const std::vector<std::vector<std::string>> &tokens,bool training=true){
    std::vector<std::vector<int64_t>> ids;

    for(const std::vector<std::string> &s:tokens){
      ids.push_back(voc.stoi(s));
    }

    if(!emb) throw std::runtime_error("Language::forward emb is nullopt");

    return emb.value().forward(ids,training);
  }

  void backward(const tensor::Tensor& grad_output){
    if(!emb) throw std::runtime_error("Language::backward emb is nullopt");

    emb.value().backward(grad_output);
  }

  //トークン登録
  void add(const std::string &s){
    voc.add(et.tokenize(s));
  }

  void add(const std::vector<std::string> &s){
    voc.add(s);
  }

  //Embedding初期化。トークン数が分からないため
  void build(const int64_t token_len,const int64_t dim){
    emb = Embedding(token_len,dim);
  }

  //語彙数
  int64_t size() const{
    return voc.size();
  }

  nlohmann::ordered_json to_json() const{
    nlohmann::ordered_json j;

    if(!emb) throw std::runtime_error("Language::to_json ");

    j["Embedding"] = emb.value().to_json();
    j["Vocabulary"] = voc.to_json();

    return j;
  }

  void load_json(const nlohmann::ordered_json &j){
    emb = Embedding(j["Embedding"]);

    voc.load_json(j["Vocabulary"]);
  }

  void save_all(const std::string &path_json,const std::string &path_bin){
    {
      std::ofstream ofs_json(path_json);

      if(!ofs_json) throw std::runtime_error("Language::save_all json open failed");

      ofs_json << to_json().dump(2);

      if(!ofs_json) throw std::runtime_error("Language::save_all json write failed");

      ofs_json.close();
    }

    if(!emb) throw std::runtime_error("Language::save_all ");

    {
      std::ofstream ofs_bin(path_bin,std::ios::binary);

      if(!ofs_bin) throw std::runtime_error("Language::save_all bin open failed");

      emb.value().save(ofs_bin);

      if(!ofs_bin) throw std::runtime_error("Language::save_all bin write failed");

      ofs_bin.close();
    }
  }

  void load_all(const std::string &path_json,const std::string &path_bin){
    std::ifstream ifs_json(path_json);

    if(!ifs_json) throw std::runtime_error("Model::load_all json open failed");

    nlohmann::ordered_json json;
    ifs_json >> json;

    load_json(json);

    std::ifstream ifs_bin(path_bin,std::ios::binary);

    if(!ifs_bin) throw std::runtime_error("Model::load_all bin open failed");

    emb.value().load(ifs_bin);
  }
};

}//namespace cobalt_715::nn::language