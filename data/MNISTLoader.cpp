#include "MNISTLoader.hpp"

MNISTLoader::MNISTLoader(const std::string &imagePath, const std::string &labelPath){
  imageFile.open(imagePath, std::ios::binary);
  labelFile.open(labelPath, std::ios::binary);
  if(!imageFile || !labelFile) throw std::runtime_error("ファイルを開けません");

  uint32_t magic = readBigEndianUInt32(imageFile);
  numImages = readBigEndianUInt32(imageFile);
  numRows = readBigEndianUInt32(imageFile);
  numCols = readBigEndianUInt32(imageFile);

  uint32_t labelMagic = readBigEndianUInt32(labelFile);
  uint32_t numLabels = readBigEndianUInt32(labelFile);

  if(numImages != numLabels) throw std::runtime_error("画像枚数とラベル枚数が一致しません");
}

// 指定した index の画像を float[] に変換して返す
std::vector<float> MNISTLoader::getImage(int index) {
  if(index < 0 || index >= (int)numImages) throw std::out_of_range("Indexが範囲外です");

  // 画像位置をシーク
  imageFile.seekg(16 + index*numRows*numCols, std::ios::beg);

  std::vector<uint8_t> temp(numRows*numCols);
  imageFile.read(reinterpret_cast<char*>(temp.data()), temp.size());

  // double に変換して 0~1 に正規化
  std::vector<float> image(numRows*numCols);
  for(size_t i=0;i<temp.size();i++) image[i] = temp[i] / 255.0f;

  return image;
}

 // ラベルを取得
 int MNISTLoader::getLabel(int index) {
  if(index < 0 || index >= (int)numImages) throw std::out_of_range("Indexが範囲外です");

  labelFile.seekg(8 + index, std::ios::beg);
  uint8_t label;
  labelFile.read(reinterpret_cast<char*>(&label), 1);
  return static_cast<int>(label);
}