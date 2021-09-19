/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "train/train_opt.h"
#include "utils/base.h"
#include "utils/utils.h"
#include "synchronize/mutex_adapter.h"


// 通用参数的头部结构
struct FMParamUnit {
  real_t w;
  real_t V[DIM];
  void operator+=(struct FMParamUnit &other) {
    w += other.w;
    for (int i = 0; i < DIM; i++) {
      V[i] += other.V[i];
    }
  }
  void clear() {
    w = 0.0;
    for (int i = 0; i < DIM; i++) {
      V[i] = 0.0;
    }
  }
};

class ParamContainerInterface;

struct ParamContext {
  ParamContext(ParamContainerInterface *_container = NULL, FMParamUnit *_param = NULL, Mutex_t *_mutex = NULL,
               real_t _x = 1.0)
      : container(_container), param(_param), mutex(_mutex), x(_x) {}

  ParamContainerInterface *container;
  FMParamUnit *param;
  Mutex_t *mutex;
  real_t x;
};

class ParamContainerInterface {
 public:
  ParamContainerInterface(feaid_t total_fea_num, int _param_size_of_one_fea)
      : param_size_of_one_fea(_param_size_of_one_fea),
        fea_num(total_fea_num),
        param_base_addr((unsigned char *)malloc((fea_num + 1) * param_size_of_one_fea))
  {
  }

  virtual ~ParamContainerInterface() {
    free(param_base_addr);
  }

  void clear_weights(FMParamUnit * param_addr) {
    param_addr->w = 0.0;
    for (int i=0; i < DIM; i++) {
      param_addr->V[i] = 0.0;
    }
  }

  bool isBadID(feaid_t id) const { return id > fea_num || id < 0; }
  feaid_t getUNKnownID() const { return fea_num; }

  FMParamUnit *get() {
    return (FMParamUnit *)param_base_addr;
  }

  FMParamUnit *get(feaid_t id) {
    if (UNLIKELY(isBadID(id))) {
      id = getUNKnownID();
    }
    return (FMParamUnit *)(param_base_addr + param_size_of_one_fea * id);
  }

  int getParamUnitSize() const {
    return param_size_of_one_fea;
  }

  void cpParam(FMParamUnit * dest, feaid_t fid) {
    const FMParamUnit * param_addr = get(fid);
    cpParam(dest, param_addr);
  }

  void cpParam(FMParamUnit * dest, const FMParamUnit * param_addr) {
    memcpy((void *)dest, (const void *)param_addr, param_size_of_one_fea);
  }

  int load(string path, string model_fmt) {
    feaid_t total_fea_num = fea_num + 1;
    int ret = 0;
    if (model_fmt == "bin") {
      ifstream ifs;
      ifs.open(path, std::ifstream::binary);
      int weight_size = sizeof(FMParamUnit);
      feaid_t i = 0;
      for (; i < total_fea_num; i++) {
        FMParamUnit *p = get(i);
        ifs.read((char *)p, weight_size);
        if (!ifs) {
          std::cerr << "load model faild, size not match: " << path << " fea_id: " << i <<  std::endl;
          return -1;
        }
      }
      if (i != total_fea_num) {
          std::cerr << "load model faild, size not match: " << path << " fea_id: " << i <<  std::endl;
          return -2;
      }
    } else {
      ifstream ifs;
      ifs.open(path);
      string line;
      feaid_t i = 0;
      for (; i < total_fea_num; i++) {
        FMParamUnit *p = get(i);
        if (!std::getline(ifs, line)) {
          std::cerr << "load model faild, size not match: " << path << " fea_id: " << i <<  std::endl;
          return -1;
        }
        real_t * read_end = utils::split_string(line, ' ', (real_t *)p);
        if (read_end - (real_t *)p != 1 + DIM) {
          std::cerr << "load model faild, param size not match in line: " << i << " path: " << path << std::endl;
          return -2;
        }
      }
      if (i != total_fea_num) {
          std::cerr << "load model faild, size not match: " << path << " fea_id: " << i <<  std::endl;
          return -2;
      }
    }
    std::cout << "load model ok: " << path <<  " total_fea_num: " << total_fea_num <<  std::endl;
    return ret;
  }

  int dump(string path, string model_fmt) {
    int ret = 0;
    feaid_t total_fea_num = fea_num + 1;

    if (model_fmt == "bin") {
      ofstream ofs(path, std::ios::out | std::ios::binary);
      if (!ofs) {
        return -1;
      }
      const int weight_size = sizeof(FMParamUnit);
      for (feaid_t i = 0; i < total_fea_num; i++) {
        const FMParamUnit *p = get(i);
        ofs.write((const char *)&p, weight_size);
      }
      ofs.close();
    } else {
      ofstream ofs(path);
      if (!ofs) {
        return -1;
      }
      for (feaid_t i = 0; i < total_fea_num; i++) {
        const FMParamUnit *p = get(i);
        ofs << p->w;
        for (int i = 0; i < DIM; i++) {
          ofs << " " << p->V[i];
        }
        ofs << std::endl;
      }
      ofs.close();
    }
    return ret;
  }

  // // backward  现在update是靠solver中的虚函数实现的，没有用这里的update。
  // // 如果放在这里，则不需要solver的多态，update的多态在这里实现，只需要一个solver的实现就行。但是不方便保存update的一些中间变量。需要一个context。
  // void update_param(feaid_t fid, real_t grad) {
  //   FMParamUnit *backward_param = get(fid);
  //   update_param(backward_param, grad);
  // }
  // virtual void update_param(FMParamUnit *backward_param, real_t grad) = 0;

  const feaid_t fea_num;

  const int param_size_of_one_fea;

  unsigned char * const param_base_addr;

 private:
  // 禁用拷贝
  ParamContainerInterface(const ParamContainerInterface &ohter);
  ParamContainerInterface &operator=(const ParamContainerInterface &that);
};

template <class ParamUnit>
class ParamContainer : public ParamContainerInterface {
 public:
  ParamContainer(feaid_t total_fea_num)
      : ParamContainerInterface(total_fea_num, sizeof(ParamUnit)) {
    for (feaid_t i = 0; i < fea_num + 1; i++) {
      ParamUnit *p = (ParamUnit *)get(i);
      new (p) ParamUnit();
    }
  }
  virtual ~ParamContainer() {}
};
