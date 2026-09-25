/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <acl/acl.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "core/framework/sampling/rejection_sampler.h"

namespace xllm {
namespace {

using OutputPair = std::tuple<torch::Tensor, torch::Tensor>;
using Operation = std::function<OutputPair()>;
using Clock = std::chrono::steady_clock;

constexpr int32_t kWarmup = 100;
constexpr int32_t kGroups = 3;
constexpr int32_t kSamples = 100;

struct Measurement {
  int32_t group;
  int32_t sample;
  int32_t order;
  double event_us;
  double enqueue_us;
  double completed_wall_us;
};

// Use the unchanged prefix implementation from the old Torch verifier, not the
// NPU-dispatched greedy helper, as the complete-interval baseline.
OutputPair old_torch_verify(const torch::Tensor& draft,
                            const torch::Tensor& target,
                            const torch::Tensor& bonus) {
  const torch::Tensor full = torch::cat({target, bonus}, /*dim=*/-1);
  const torch::Tensor accepted_mask =
      RejectionSampler::build_accepted_mask(target.eq(draft));
  return {full, torch::where(accepted_mask, full, torch::full_like(full, -1))};
}

void check_output(const OutputPair& output,
                  const OutputPair& expected_cpu,
                  torch::ScalarType dtype) {
  for (const auto& tensors :
       {std::pair{std::get<0>(output), std::get<0>(expected_cpu)},
        std::pair{std::get<1>(output), std::get<1>(expected_cpu)}}) {
    CHECK(tensors.first.defined());
    CHECK_EQ(tensors.first.scalar_type(), dtype);
    CHECK_EQ(tensors.first.sizes(), tensors.second.sizes());
    CHECK(torch::equal(tensors.first.cpu(), tensors.second.to(dtype)))
        << "F1 benchmark correctness mismatch; timings are invalid";
  }
}

class EventTimer final {
 public:
  explicit EventTimer(aclrtStream stream) : stream_(stream) {
    CHECK_EQ(aclrtCreateEvent(&start_), ACL_SUCCESS);
    CHECK_EQ(aclrtCreateEvent(&end_), ACL_SUCCESS);
    CHECK_EQ(aclrtRecordEvent(start_, stream_), ACL_SUCCESS);
    CHECK_EQ(aclrtRecordEvent(end_, stream_), ACL_SUCCESS);
    CHECK_EQ(aclrtSynchronizeEvent(end_), ACL_SUCCESS);
  }

  ~EventTimer() {
    CHECK_EQ(aclrtDestroyEvent(start_), ACL_SUCCESS);
    CHECK_EQ(aclrtDestroyEvent(end_), ACL_SUCCESS);
  }

  EventTimer(const EventTimer&) = delete;
  EventTimer& operator=(const EventTimer&) = delete;

  std::pair<Measurement, OutputPair> measure(const Operation& operation,
                                             int32_t group,
                                             int32_t sample,
                                             int32_t order) const {
    CHECK_EQ(aclrtSynchronizeStream(stream_), ACL_SUCCESS);
    CHECK_EQ(aclrtRecordEvent(start_, stream_), ACL_SUCCESS);
    const auto begin = Clock::now();
    OutputPair output = operation();
    const auto enqueued = Clock::now();
    CHECK_EQ(aclrtRecordEvent(end_, stream_), ACL_SUCCESS);
    CHECK_EQ(aclrtSynchronizeEvent(end_), ACL_SUCCESS);
    const auto completed = Clock::now();
    float elapsed_ms = 0.0F;
    CHECK_EQ(aclrtEventElapsedTime(&elapsed_ms, start_, end_), ACL_SUCCESS);
    return {
        {group,
         sample,
         order,
         static_cast<double>(elapsed_ms) * 1000.0,
         std::chrono::duration<double, std::micro>(enqueued - begin).count(),
         std::chrono::duration<double, std::micro>(completed - begin).count()},
        std::move(output)};
  }

 private:
  aclrtStream stream_;
  aclrtEvent start_ = nullptr;
  aclrtEvent end_ = nullptr;
};

void log_summary(const char* scope,
                 const char* implementation,
                 int64_t batch,
                 int64_t width,
                 const std::vector<Measurement>& measurements) {
  CHECK_EQ(measurements.size(), kGroups * kSamples);
  const std::array<const char*, 3> names = {
      "event_us", "host_enqueue_us", "completed_wall_us"};
  const std::array<double Measurement::*, 3> fields = {
      &Measurement::event_us,
      &Measurement::enqueue_us,
      &Measurement::completed_wall_us};
  for (size_t metric = 0; metric < fields.size(); ++metric) {
    std::vector<double> sorted;
    sorted.reserve(measurements.size());
    for (const auto& measurement : measurements) {
      sorted.emplace_back(measurement.*fields[metric]);
    }
    std::sort(sorted.begin(), sorted.end());
    const size_t middle = sorted.size() / 2;
    const double median = (sorted[middle - 1] + sorted[middle]) / 2.0;
    const size_t p95_index = (sorted.size() * 95 + 99) / 100 - 1;
    LOG(INFO) << "F1 C++ interval scope=" << scope
              << " implementation=" << implementation << " B=" << batch
              << " K=" << width << " metric=" << names[metric]
              << " n=" << sorted.size() << " median=" << median
              << " p95=" << sorted[p95_index];
  }
}

void measure_pair(const char* scope,
                  int64_t batch,
                  int64_t width,
                  const std::array<Operation, 2>& operations,
                  const std::array<torch::ScalarType, 2>& output_dtypes,
                  const OutputPair& expected_cpu,
                  const std::function<void()>& check_inputs,
                  aclrtStream stream,
                  std::ofstream& csv) {
  const std::array<const char*, 2> names = {"old_torch", "tilelang_aot"};
  for (size_t index = 0; index < operations.size(); ++index) {
    check_output(operations[index](), expected_cpu, output_dtypes[index]);
  }
  for (int32_t warmup = 0; warmup < kWarmup; ++warmup) {
    for (const auto& operation : operations) {
      operation();
    }
  }
  CHECK_EQ(aclrtSynchronizeStream(stream), ACL_SUCCESS);
  check_inputs();
  EventTimer timer(stream);
  std::array<std::vector<Measurement>, 2> measurements;
  for (auto& samples : measurements) {
    samples.reserve(kGroups * kSamples);
  }
  for (int32_t group = 0; group < kGroups; ++group) {
    std::array<OutputPair, 2> last_outputs;
    for (int32_t sample = 0; sample < kSamples; ++sample) {
      const int32_t first = (group + sample) % 2;
      for (int32_t order = 0; order < 2; ++order) {
        const int32_t index = (first + order) % 2;
        auto [measurement, output] =
            timer.measure(operations[index], group, sample, order);
        measurements[index].emplace_back(measurement);
        last_outputs[index] = std::move(output);
      }
    }
    for (size_t index = 0; index < operations.size(); ++index) {
      check_output(last_outputs[index], expected_cpu, output_dtypes[index]);
    }
    check_inputs();
  }
  // Reporting and all CPU reads are outside measured intervals.
  for (size_t index = 0; index < operations.size(); ++index) {
    log_summary(scope, names[index], batch, width, measurements[index]);
    for (const auto& sample : measurements[index]) {
      csv << scope << ',' << names[index] << ',' << batch << ',' << width << ','
          << sample.group << ',' << sample.sample << ',' << sample.order << ','
          << sample.event_us << ',' << sample.enqueue_us << ','
          << sample.completed_wall_us << '\n';
    }
  }
  csv.flush();
  CHECK(csv.good()) << "Cannot write F1 benchmark measurements";
}

// Explicit invocation requires --gtest_also_run_disabled and this exact filter.
// TASK_QUEUE_ENABLE=0 must be set before process startup so raw ACL events
// order both Torch and direct AOT submissions on the same stream. This is a
// labelled microbenchmark configuration, not a measurement of the default
// serving queue.
TEST(GreedyPrefixVerifyNpuBenchmarkTest,
     DISABLED_CompleteIntervalsAndMtpUbCast) {
  const char* queue_mode = std::getenv("TASK_QUEUE_ENABLE");
  CHECK(queue_mode != nullptr && std::string(queue_mode) == "0")
      << "Start this benchmark with TASK_QUEUE_ENABLE=0 for ordered ACL timing";
  const char* blocking = std::getenv("ASCEND_LAUNCH_BLOCKING");
  CHECK(blocking == nullptr || std::string(blocking) == "0")
      << "Unset ASCEND_LAUNCH_BLOCKING for asynchronous interval measurement";
  const char* output_dir = std::getenv("XLLM_F1_BENCHMARK_DIR");
  CHECK(output_dir != nullptr)
      << "Set XLLM_F1_BENCHMARK_DIR to an existing output directory";
  const std::filesystem::path directory(output_dir);
  CHECK(std::filesystem::is_directory(directory));
  const std::filesystem::path csv_path =
      directory / "greedy_prefix_verify_cpp.csv";
  CHECK(!std::filesystem::exists(csv_path))
      << "Refusing to overwrite " << csv_path;
  std::ofstream csv(csv_path);
  CHECK(csv.is_open()) << "Cannot open " << csv_path;
  const char* visible_devices = std::getenv("ASCEND_RT_VISIBLE_DEVICES");
  const char* container_devices = std::getenv("ASCEND_VISIBLE_DEVICES");
  const char* soc_name = aclrtGetSocName();
  CHECK(soc_name != nullptr) << "aclrtGetSocName failed";
  csv << std::setprecision(12) << "# logical_device=npu:0,soc=" << soc_name
      << ",ASCEND_RT_VISIBLE_DEVICES="
      << (visible_devices != nullptr ? visible_devices : "<unset>")
      << ",ASCEND_VISIBLE_DEVICES="
      << (container_devices != nullptr ? container_devices : "<unset>") << '\n'
      << "# TASK_QUEUE_ENABLE=0,warmup=100,groups=3,samples=100,mask=true\n"
      << "# rejection=row%4:first/middle/last/all; vocabulary_size=32768\n"
      << "# event intervals include host submission gaps; not kernel-only\n"
      << "# mtp_ub_cast: original int64 draft and target/bonus views; "
         "candidate converts IDs in kernel UB and returns int32; old Torch "
         "returns int64; no step-major reorder/logprob gather/model execution\n"
      << "scope,implementation,batch,width,group,sample,order,event_us,"
         "host_enqueue_us,completed_wall_us\n";
  LOG(INFO) << "F1 C++ microbenchmark: output allocation included, "
               "TASK_QUEUE_ENABLE=0, logical_device=npu:0, soc="
            << soc_name << ", ASCEND_RT_VISIBLE_DEVICES="
            << (visible_devices != nullptr ? visible_devices : "<unset>")
            << ", ASCEND_VISIBLE_DEVICES="
            << (container_devices != nullptr ? container_devices : "<unset>")
            << ", raw samples=" << csv_path
            << "; interference unreviewed, not Triton/GLM E2E or TPOT evidence";

  const torch::Device device("npu:0");
  const auto options = torch::TensorOptions().dtype(torch::kInt32);
  const aclrtStream stream =
      c10_npu::getCurrentNPUStream(device.index()).stream();
  for (const int64_t batch : {1, 2, 4, 8, 16, 32, 64, 128}) {
    for (const int64_t width : {1, 3}) {
      SCOPED_TRACE(::testing::Message() << "B=" << batch << " K=" << width);
      const torch::Tensor full_cpu =
          (torch::arange(batch * (width + 1), options) + 100)
              .view({batch, width + 1});
      torch::Tensor draft_cpu = full_cpu.slice(1, 0, width).clone();
      torch::Tensor masked_cpu = full_cpu.clone();
      auto draft_values = draft_cpu.accessor<int32_t, 2>();
      auto masked_values = masked_cpu.accessor<int32_t, 2>();
      for (int64_t row = 0; row < batch; ++row) {
        const std::array<int64_t, 4> rejection_columns = {
            0, width / 2, width - 1, width};
        const int64_t first_reject = rejection_columns[row % 4];
        if (first_reject == width) {
          continue;
        }
        draft_values[row][first_reject] += 10000;
        for (int64_t col = first_reject + 1; col <= width; ++col) {
          masked_values[row][col] = -1;
        }
      }
      const OutputPair expected_cpu{full_cpu, masked_cpu};
      const torch::Tensor block32 = full_cpu.to(device).flatten();
      const torch::Tensor draft32 = draft_cpu.to(device);
      const torch::Tensor target32 =
          block32.view({batch, width + 1}).slice(1, 0, width);
      const torch::Tensor bonus32 =
          block32.slice(0, width, block32.numel(), width + 1).view({batch, 1});
      CHECK_EQ(target32.stride(0), width + 1);
      CHECK_EQ(bonus32.stride(0), width + 1);
      CHECK_EQ(bonus32.storage_offset(), width);
      const auto check_int32_inputs = [&]() {
        CHECK(torch::equal(block32.cpu(), full_cpu.flatten()));
        CHECK(torch::equal(draft32.cpu(), draft_cpu));
      };
      const std::array<Operation, 2> int32_operations = {
          [&]() { return old_torch_verify(draft32, target32, bonus32); },
          [&]() {
            return RejectionSampler::greedy_sample_from_token_ids(
                draft32, target32, bonus32, /*mask_out_rejected_tokens=*/true);
          }};
      measure_pair("int32_complete",
                   batch,
                   width,
                   int32_operations,
                   {torch::kInt32, torch::kInt32},
                   expected_cpu,
                   check_int32_inputs,
                   stream,
                   csv);

      const torch::Tensor block64 = block32.to(torch::kInt64);
      const torch::Tensor draft64 = draft32.to(torch::kInt64);
      const auto check_int64_inputs = [&]() {
        CHECK(
            torch::equal(block64.cpu(), full_cpu.flatten().to(torch::kInt64)));
        CHECK(torch::equal(draft64.cpu(), draft_cpu.to(torch::kInt64)));
      };
      const std::array<Operation, 2> mtp_operations = {
          [&]() {
            const torch::Tensor target =
                block64.view({batch, width + 1}).slice(1, 0, width);
            const torch::Tensor bonus =
                block64.slice(0, width, block64.numel(), width + 1)
                    .view({batch, 1});
            return old_torch_verify(draft64, target, bonus);
          },
          [&]() {
            const torch::Tensor target =
                block64.view({batch, width + 1}).slice(1, 0, width);
            const torch::Tensor bonus =
                block64.slice(0, width, block64.numel(), width + 1)
                    .view({batch, 1});
            return RejectionSampler::greedy_sample_from_token_ids(
                draft64,
                target,
                bonus,
                /*mask_out_rejected_tokens=*/true);
          }};
      measure_pair("mtp_ub_cast_old_long_new_int32",
                   batch,
                   width,
                   mtp_operations,
                   {torch::kInt64, torch::kInt32},
                   expected_cpu,
                   check_int64_inputs,
                   stream,
                   csv);
    }
  }
}

}  // namespace
}  // namespace xllm
