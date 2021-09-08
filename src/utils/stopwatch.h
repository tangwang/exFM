/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once

#include <stdint.h>
#include <string>
#include <vector>
#include <string.h>
#include <time.h>
#include <sys/time.h>

namespace utils {

// 计时器
class Stopwatch {
public:
    Stopwatch() {
        _clear();
    }

    // 停止时间间隔测量，并将运行时间重置为零。
    void reset() {
        _clear();
    }

    // 停止时间间隔测量，将运行时间重置为零，然后开始测量运行时间。
    void restart() {
        reset();
        _begin();
    }

    // 开始或继续测量某个时间间隔的运行时间。
    void start() {
        _begin();
    }

    // 停止测量某个时间间隔的运行时间, 并返回计时的us数
    long stop() {
        _elapsed = get_elapsed();
        is_running = false;
        return _elapsed;
    }

    long get_elapsed() {
        long ret = _elapsed;
        if (is_running) {
            gettimeofday(&_stop_time, NULL);
            long elapsed_in_this_round = (_stop_time.tv_sec - _start_time.tv_sec) * 1000000
                    + (_stop_time.tv_usec - _start_time.tv_usec);
            ret += elapsed_in_this_round;
        }
        return ret;
    }

    double get_elapsed_by_seconds() {
        return ((double)get_elapsed()) / 1000000.0;
    }

    double get_elapsed_by_ms() {
        return ((double)get_elapsed()) / 1000.0;
    }

    const struct timeval & get_start_time() const {
        return _start_time;
    }

    const struct timeval & get_stop_time() const {
        return _stop_time;
    }

private:
    void _begin() {
        gettimeofday(&_start_time, NULL);
        is_running = true;
    }

    void _clear() {
        memset(&_start_time, 0, sizeof(struct timeval));
        memset(&_stop_time, 0, sizeof(struct timeval));
        _elapsed = 0;
        is_running = false;
    }

    bool is_running;

    struct timeval _start_time; // 开始计时时间
    struct timeval _stop_time;
    long _elapsed; // 耗时，单位us
};

} // namespace utils

