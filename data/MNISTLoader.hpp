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
  MNISTLoader(const std::string &imagePath, const std::string &labelPath);

  // 指定した index の画像を float[] に変換して返す
  std::vector<float> getImage(int index);

  // ラベルを取得
  int getLabel(int index);

  int getNumRows() const { return numRows; }
  int getNumCols() const { return numCols; }
  int getNumImages() const { return numImages; }
};