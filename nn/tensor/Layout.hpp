#pragma once

namespace cobalt_715::nn::tensor{

//MatrixViewなどのメモリ配置
enum class Layout{
  CONTIGUOUS,//完全連続
  ROW_CONTIGUOUS,//行連続
  COL_CONTIGUOUS,//列連続
  STRIDED_SAFE,//strided_safe
  OVERLAPPED//書き込み不可能
};

}//namespace cobalt_715::nn::tensor