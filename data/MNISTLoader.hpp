#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <stdexcept>

// BigEndian読み込み
inline uint32_t readBigEndianUInt32(std::ifstream &ifs){
    uint32_t val = 0;
    for(int i=0;i<4;i++){
        val <<= 8;
        val |= ifs.get() & 0xFF;
    }
    return val;
}

// 画像データをメモリにロードして保持するクラス
class MNISTLoader {
private:
    std::ifstream imageFile;
    std::ifstream labelFile;
    uint32_t numImages;
    uint32_t numRows;
    uint32_t numCols;

public:
  MNISTLoader(const std::string &imagePath, const std::string &labelPath){
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
  std::vector<float> getImage(int index){
    if(index < 0 || index >= (int)numImages) throw std::out_of_range("Indexが範囲外です");

    // 画像位置をシーク
    imageFile.seekg(16 + index*numRows*numCols, std::ios::beg);

    std::vector<uint8_t> temp(numRows*numCols);
    imageFile.read(reinterpret_cast<char*>(temp.data()), temp.size());

    // float に変換して 0~1 に正規化
    std::vector<float> image(numRows*numCols);
    for(size_t i=0;i<temp.size();i++) image[i] = temp[i] / 255.0f;

    return image;
  }

  // ラベルを取得
  int getLabel(int index){
    if(index < 0 || index >= (int)numImages) throw std::out_of_range("Indexが範囲外です");

    labelFile.seekg(8 + index, std::ios::beg);
    uint8_t label;
    labelFile.read(reinterpret_cast<char*>(&label), 1);
    return static_cast<int>(label);
  }

  int getNumRows() const { return numRows; }
  int getNumCols() const { return numCols; }
  int getNumImages() const { return numImages; }
};