/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once

#include <algorithm>  // std::move_backward
#include <chrono>     // std::chrono::system_clock
#include <iostream>
#include <random>  // std::default_random_engine
#include <vector>

#include "utils/base.h" // for unit8, uint16

//  0~shulf_size之间打乱顺序
class Shulffer {
 public:
  Shulffer(int shulf_size)
      : shulf_window_size(shulf_size), shulf_window(shulf_size) {
    for (int i = 0; i < shulf_size; i++) {
      shulf_window[i] = i;
    }
  }
  ~Shulffer() {}

  void reset() {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    std::shuffle(shulf_window.begin(), shulf_window.end(),
            std::default_random_engine(seed));
  }

  int next() {
    if (++cussor == shulf_window_size) cussor = 0;
    return shulf_window[cussor];
  }

 private:
  int cussor;
  int shulf_window_size;
  std::vector<int> shulf_window;
};

// shulf_window_size固定为65535，next()返回0~max_id之间的任意数，max_id最多支持65535
class UshortShulffer {
 public:
  UshortShulffer(uint16 max_id) {
    cussor = 0;
    for (int i = 0; i != total_uint16_counts; i++) {
      shulf_window[i] = i % max_id;
    }
  }
  ~UshortShulffer() {}

  void reset() {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    std::shuffle(shulf_window, shulf_window + total_uint16_counts,
            std::default_random_engine(seed));
  }

  uint16 next() { return shulf_window[cussor++]; }

 private:
  uint16 cussor;
  static const int total_uint16_counts = 0x10000;
  uint16 shulf_window[total_uint16_counts];
};


// shulf_window_size固定为65535，next()返回0~max_id之间的任意数，max_id最多支持255
class MiniShuffer {
 public:
  MiniShuffer(uint8 max_id) {
    cussor = 0;
    for (int i = 0; i != total_uint16_counts; i++) {
      shulf_window[i] = i % max_id;
    }
  }
  ~MiniShuffer() {}

  void reset() {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    std::shuffle(shulf_window, shulf_window + total_uint16_counts,
            std::default_random_engine(seed));
  }

  uint16 next() { return shulf_window[cussor++]; }

 private:
  uint16 cussor;
  static const int total_uint16_counts = 0x10000;
  uint8 shulf_window[total_uint16_counts];
};
