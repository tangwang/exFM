/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once

#include <algorithm>  // std::move_backward
#include <chrono>     // std::chrono::system_clock
#include <iostream>
#include <random>  // std::default_random_engine
#include <vector>

#include "utils/base.h"

//  0~shulf_size之间打乱顺序
class Shulffer {
 public:
  Shulffer(int shulf_size)
      : shulf_window(shulf_size), shulf_window_size(shulf_size) {
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

// shulf_size固定为65535， 0 ~ 65535之间打乱顺序
class UshortShulffer {
 public:
  UshortShulffer() {
    cussor = 0;
    for (uint16 i = 0; i != max_int16_id; i++) {
      shulf_window[i] = i;
    }
  }
  ~UshortShulffer() {}

  void reset() {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    std::shuffle(shulf_window, shulf_window + max_int16_id,
            std::default_random_engine(seed));
  }

  uint16 next() { return shulf_window[cussor++]; }

 private:
  uint16 cussor;
  const static uint16 max_int16_id = 0xffff;
  uint16 shulf_window[max_int16_id];
};
