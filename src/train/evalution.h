/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "utils/base.h"
#include "utils/stopwatch.h"

class Evalution {
 public:
  Evalution() {
    reset();
  }
  ~Evalution() {}

  size_t size() const { return label_prob_list.size(); }

  /*
    label: 1 被当成正样本，其余取值被当成负样本（0和-1均可）。
    prob： 预测的打分
  */
  void add(int label, real_t prob) {
    ++total_samples_processed;

    label_prob_list.push_back(std::make_pair(label, prob));

    int pred = prob > 0.0 ? 1 : 0;
    if (label == 1) {
      pred == 1 ? ++tp : ++fn;
    } else {
      pred == 1 ? ++fp : ++tn;
    }
  }

  /*
  打印当前的统计信息，并将其清理
  */
  void output(const char* name) {
    double recall = double(tp) / (tp + fn);
    double precision = double(tp) / (tp + fp);
    double acc = double(tp + tn) / (tn + fp + fn + tp);

    // 计算AUC
    sort(label_prob_list.begin(), label_prob_list.end(),
         utils::judgeByPairSecond<int, real_t>);

    size_t idx = 0, pos_num = 0, neg_num = 0,
           sum_positive_samples_idx_of_ranked_list = 0;

    for (const auto& p : label_prob_list) {
      ++idx;
      if (p.first == 1) {
        sum_positive_samples_idx_of_ranked_list += idx;
        ++pos_num;
      } else {
        ++neg_num;
      }
    }

    double auc = double(sum_positive_samples_idx_of_ranked_list * 2 -
                        pos_num * (pos_num + 1)) /
                 (2 * pos_num * neg_num);
    double cost_time = stopwatch.get_elapsed_by_seconds();
    printf(
        "%s total_processed %ld (%ld per seconds), AUC %.4f, "
        "confusion_matrix(total|tn,fp,fn,tp): %ld | %ld %ld %ld %ld, acc %.4f, "
        "recall %.4f, "
        "precision %.4f\n",
        name, total_samples_processed,
        (size_t)(total_samples_processed / cost_time), auc, tn + fp + fn + tp,
        tn, fp, fn, tp, acc, recall, precision);

    clear();
  }

  const Evalution& operator+=(const Evalution& rhs) {
    tn += rhs.tn;
    fp += rhs.fp;
    fn += rhs.fn;
    tp += rhs.tp;
    label_prob_list.insert(label_prob_list.begin(), rhs.label_prob_list.begin(),
                           rhs.label_prob_list.end());
    return *this;
  }

  void reset() {
    total_samples_processed = tn = fp = fn = tp = 0;
    stopwatch.start();
  }

 private:
  /*
 清理当前所有的信息
 */
  void clear() {
    tn = fp = fn = tp = 0;
    label_prob_list.clear();
  }

  utils::Stopwatch stopwatch;
  size_t total_samples_processed;

  vector<std::pair<int, real_t> > label_prob_list;

  size_t tn, fp, fn, tp;
};
