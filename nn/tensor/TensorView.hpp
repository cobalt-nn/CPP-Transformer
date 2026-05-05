#pragma once

#include <vector>
#include <cstdint>

namespace cobalt_715::nn::tensor{

//Tensorの一部分を借用する
class TensorView{
public:

private:
  std::vector<int64_t> shape_;//各次元の要素数
  std::vector<int64_t> stride_;//各次元にジャンプするまでに必要な数
  float *data_;//data
};

}//namespace cobalt_715::nn::tensor