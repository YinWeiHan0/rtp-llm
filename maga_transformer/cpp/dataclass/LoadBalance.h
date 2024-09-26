#pragma once

#include <cstdint>
#include <mutex>
#include <queue>
#include <vector>
#include "autil/EnvUtil.h"
#include "src/fastertransformer/utils/py_utils/pybind_utils.h"

namespace rtp_llm {

class PIController {
public:
    PIController(float kp = 0.0, float ki = 0.1);

    float getCurrent();

    void addTarget(float target);

    void reset();

private:
    float current_     = 1.0;
    float sum_diffs    = 0;
    float kp_          = 0.0;
    float ki_          = 0.1;
    float lower_limit_ = 1.0;
};

struct StepInfo {
    size_t time_us;
    size_t batch_avg_gen_num;
};

class StepRecorder {
public:
    size_t getStepLatency();

    size_t getStepCount();

    size_t getStepPerMin();

    void addStepCount(size_t step_count);

    void registerStep(size_t step_time_us, size_t batch_avg_gen_num = 1);

    void reset();

    bool empty();

private:
    double getIntervalAvgGenNum() {
        return queue_total_gen_num_ * 1.0 / step_records_.size();
    }

    size_t getIntervalDuration() {
        return std::max((size_t)1, step_records_.back().time_us - step_records_.front().time_us);
    }

    size_t getIntervalPerStepLatency() {
        return getIntervalDuration() * 1.0 / (getIntervalAvgGenNum() * (step_records_.size() - 1));
    }


    // all time is us
    const static size_t STEP_RECORDS_MAX_SIZE;
    const static size_t STEP_RECORDS_TIME_RANGE;

    PIController avg_latency_controller_;
    PIController step_count_controller_;

    std::queue<StepInfo> step_records_;
    size_t              min_step_latency_ = 10 * 1000 * 1000;  // 10s
    size_t              queue_total_gen_num_ = 0;
    
    std::mutex          mutex_;
};

struct LoadBalanceInfo {
    int64_t step_latency_us = 0;
    int64_t iterate_count = 0;
    int64_t step_per_minute = 0;
    int64_t available_kv_cache = 0;
    int64_t total_kv_cache = 0;
};

void registerLoadBalanceInfo(const pybind11::module& m);

}  // namespace rtp_llm
