#pragma once

#include <iostream>
#include <string>
#include <random>
#include "ILayer.hpp"
#include "Linear.hpp"
#include "Identity.hpp"
#include "nlohmann/json.hpp"
#include "nn/tensor/Tensor.hpp"
#include "nn/io/BinaryIO.hpp"

namespace cobalt_715::nn::layer{

struct ReZero : ILayer{
  ReZero(int64_t in,int64_t body_out,std::unique_ptr<ILayer> body)
    : ReZero(std::move(body),std::make_unique<Linear>(body_out,in)){}

  ReZero(std::unique_ptr<ILayer> body)
    : ReZero(std::move(body),std::make_unique<Identity>()){}

  ReZero(std::unique_ptr<ILayer> body,std::unique_ptr<ILayer> projection)
    : body_(std::move(body)),
      projection_(std::move(projection)),
      output_({1}),
      grad_({1}){}

  std::unique_ptr<ILayer> body_;

  std::unique_ptr<ILayer> projection_;

  const tensor::Tensor *pr_output_ptr_;

  tensor::Tensor output_;
  tensor::Tensor grad_;

  float alpha_ = 0.0f;
  float d_alpha_ = 0.0f;

  const tensor::Tensor& forward(const tensor::Tensor& input,bool training=true) override{
    pr_output_ptr_ = &projection_->forward(body_->forward(input,training),training);

    if(output_.shape() != input.shape()){
      output_ = tensor::Tensor(input.shape());
    }else{
      std::fill(output_.data(),output_.data() + output_.numel(),0.0f);
    }

    tensor::Tensor::scale(*pr_output_ptr_,alpha_,output_);

    output_ += input;

    return output_;
  }

  const tensor::Tensor& backward(const tensor::Tensor& grad_output) override{
    if(grad_.shape() != grad_output.shape()){
      grad_ = tensor::Tensor(grad_output.shape());
    }else{
      std::fill(grad_.data(),grad_.data() + grad_.numel(),0.0f);
    }

    tensor::Tensor::scale(grad_output,alpha_,grad_);//grad_ = grad_output * alpha_

    const auto &g = body_->backward(projection_->backward(grad_));

    tensor::Tensor::add(grad_output,g,grad_);//grad_ = grad_output + g

    for(size_t i = 0;i < grad_output.numel();i++){
      d_alpha_ += pr_output_ptr_->data()[i] * grad_output.data()[i];
    }

    return grad_;
  }

  void step(float lr,int batch_size=64) override{
    projection_->step(lr,batch_size);
    body_->step(lr,batch_size);

    alpha_ -= d_alpha_ * lr;

    //std::cout << "alpha:" << alpha_ << std::endl;
  }

  void zero_grad() override{
    projection_->zero_grad();
    body_->zero_grad();

    d_alpha_ = 0.0f;
  }

  void reset() override{
    projection_->reset();
    body_->reset();
  }

  std::string get_type() const override{
    return "ReZero";
  }

  std::string to_string() const override{
    std::string s = get_type() + "\n";
    s += body_->to_string() + "\n";
    s += projection_->to_string();
    return s;
  }

  nlohmann::ordered_json to_json() const override{
    nlohmann::ordered_json j;

    j["layer_type"] = get_type();
    j["body"] = body_->to_json();
    j["projection"] = projection_->to_json();

    return j;
  }

  void save(std::ostream &os) const override{
    body_->save(os);
    projection_->save(os);
    io::save(os,&alpha_,1);
  }

  void load(const nlohmann::ordered_json &json,std::istream &is) override{
    if(json.at("layer_type") != get_type()){
      throw std::runtime_error("ReZero::load type mismatch");
    }

    body_->load(json.at("body"),is);
    projection_->load(json.at("projection"),is);

    io::load(is,&alpha_,1);
  }

  //ランダム初期化する
  void random_init(std::mt19937 &gen) override{
    projection_->random_init(gen);
    body_->random_init(gen);
  }
};

}//namespace cobalt_715::nn::layer