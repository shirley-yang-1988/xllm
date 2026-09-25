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

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

#include "acl/acl.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

namespace xllm::kernel::npu::tilelang {
namespace {

using OutputPair = std::tuple<torch::Tensor, torch::Tensor>;

OutputPair torch_reference(const torch::Tensor& draft,
                           const torch::Tensor& target,
                           const torch::Tensor& bonus,
                           bool mask) {
  torch::Tensor full =
      torch::cat({target, bonus}, /*dim=*/-1).to(torch::kInt32);
  if (!mask) {
    return {full, torch::Tensor()};
  }
  const torch::Tensor accepted = target.eq(draft).to(torch::kInt32);
  const torch::Tensor combined = torch::cat(
      {accepted, torch::zeros({draft.size(0), 1}, accepted.options())}, -1);
  const torch::Tensor first_reject =
      (1 - combined).argmax(/*dim=*/1, /*keepdim=*/true);
  const torch::Tensor positions =
      torch::arange(draft.size(1) + 1, accepted.options()).unsqueeze(0);
  return {full,
          torch::where(
              positions <= first_reject, full, torch::full_like(full, -1))};
}

OutputPair semantic_reference(const torch::Tensor& draft,
                              const torch::Tensor& target,
                              const torch::Tensor& bonus,
                              bool mask) {
  const torch::Tensor draft_cpu = draft.cpu().to(torch::kInt64);
  const torch::Tensor target_cpu = target.cpu().to(torch::kInt64);
  const torch::Tensor bonus_cpu = bonus.cpu().to(torch::kInt64);
  torch::Tensor full =
      torch::empty({draft.size(0), draft.size(1) + 1}, torch::kInt32);
  auto full_values = full.accessor<int32_t, 2>();
  const auto draft_values = draft_cpu.accessor<int64_t, 2>();
  const auto target_values = target_cpu.accessor<int64_t, 2>();
  const auto bonus_values = bonus_cpu.accessor<int64_t, 2>();
  std::vector<int64_t> first_rejects;
  first_rejects.reserve(draft.size(0));
  for (int64_t row = 0; row < draft.size(0); ++row) {
    int64_t first_reject = draft.size(1);
    for (int64_t col = 0; col < draft.size(1); ++col) {
      full_values[row][col] = static_cast<int32_t>(target_values[row][col]);
      if (first_reject == draft.size(1) &&
          draft_values[row][col] != target_values[row][col]) {
        first_reject = col;
      }
    }
    full_values[row][draft.size(1)] =
        static_cast<int32_t>(bonus_values[row][0]);
    first_rejects.emplace_back(first_reject);
  }
  if (!mask) {
    return {full, torch::Tensor()};
  }
  torch::Tensor masked = full.clone();
  auto masked_values = masked.accessor<int32_t, 2>();
  for (int64_t row = 0; row < draft.size(0); ++row) {
    for (int64_t col = first_rejects[row] + 1; col <= draft.size(1); ++col) {
      masked_values[row][col] = -1;
    }
  }
  return {full, masked};
}

void expect_matches_references(const torch::Tensor& draft,
                               const torch::Tensor& target,
                               const torch::Tensor& bonus,
                               bool mask) {
  const torch::Tensor draft_before = draft.cpu().clone();
  const torch::Tensor target_before = target.cpu().clone();
  const torch::Tensor bonus_before = bonus.cpu().clone();
  const auto [expected_full, expected_masked] =
      semantic_reference(draft, target, bonus, mask);
  const auto [torch_full, torch_masked] =
      torch_reference(draft, target, bonus, mask);
  const auto [full, masked] = greedy_prefix_verify(draft, target, bonus, mask);

  ASSERT_TRUE(full.defined());
  EXPECT_EQ(full.scalar_type(), torch::kInt32);
  EXPECT_EQ(full.device(), target.device());
  EXPECT_EQ(full.sizes(),
            torch::IntArrayRef({draft.size(0), draft.size(1) + 1}));
  EXPECT_TRUE(full.is_contiguous());
  EXPECT_TRUE(torch::equal(full.cpu(), expected_full));
  EXPECT_TRUE(torch::equal(torch_full.cpu(), expected_full));
  EXPECT_EQ(masked.defined(), mask);
  EXPECT_EQ(torch_masked.defined(), mask);
  if (mask) {
    ASSERT_TRUE(masked.defined());
    EXPECT_EQ(masked.scalar_type(), torch::kInt32);
    EXPECT_EQ(masked.device(), full.device());
    EXPECT_EQ(masked.sizes(), full.sizes());
    EXPECT_TRUE(masked.is_contiguous());
    EXPECT_TRUE(torch::equal(masked.cpu(), expected_masked));
    EXPECT_TRUE(torch::equal(torch_masked.cpu(), expected_masked));
    if (full.numel() != 0) {
      EXPECT_NE(full.data_ptr(), masked.data_ptr());
    }
  }
  EXPECT_TRUE(torch::equal(draft.cpu(), draft_before));
  EXPECT_TRUE(torch::equal(target.cpu(), target_before));
  EXPECT_TRUE(torch::equal(bonus.cpu(), bonus_before));
}

// cc_test installs the shared NPU/Python runtime environment for this binary.
class TileLangGreedyPrefixVerifyTest : public ::testing::Test {
 protected:
  const torch::Device device_{"npu:0"};
  const torch::TensorOptions i32_ =
      torch::TensorOptions().dtype(torch::kInt32).device(device_);
};

TEST_F(TileLangGreedyPrefixVerifyTest,
       PreservesFullOutputAcrossFirstRejection) {
  const torch::Tensor draft = torch::tensor(
      {{9, 22, 33}, {11, 22, 33}, {11, 25, 99}, {11, 25, 33}}, i32_);
  const torch::Tensor target =
      torch::tensor({{11, 25, 33}}, i32_).repeat({4, 1});
  const torch::Tensor bonus = torch::full({4, 1}, 44, i32_);

  for (const bool mask : {false, true}) {
    expect_matches_references(draft, target, bonus, mask);
  }
  const auto [full, masked] =
      greedy_prefix_verify(draft, target, bonus, /*mask=*/true);
  EXPECT_TRUE(torch::equal(
      full.cpu(),
      torch::tensor({{11, 25, 33, 44}}, torch::kInt32).repeat({4, 1})));
  EXPECT_TRUE(torch::equal(masked.cpu(),
                           torch::tensor({{11, -1, -1, -1},
                                          {11, 25, -1, -1},
                                          {11, 25, 33, -1},
                                          {11, 25, 33, 44}},
                                         torch::kInt32)));
}

TEST_F(TileLangGreedyPrefixVerifyTest,
       HandlesEmptyShortAndNonSpecializedWidths) {
  for (const int64_t width :
       {0, 1, 2, 3, 4, 7, 31, 32, 33, 63, 64, 65, 127, 128, 129}) {
    SCOPED_TRACE(::testing::Message() << "K=" << width);
    const torch::Tensor full_cpu =
        (torch::arange(5 * (width + 1), torch::kInt32) + 100)
            .reshape({5, width + 1});
    const torch::Tensor target_cpu = full_cpu.slice(1, 0, width);
    torch::Tensor draft_cpu = target_cpu.clone();
    if (width > 0) {
      auto values = draft_cpu.accessor<int32_t, 2>();
      values[0][0] += int32_t{1} << 24;
      values[1][width / 2] += int32_t{1} << 24;
      values[2][width - 1] += int32_t{1} << 24;
    }
    const torch::Tensor full_device = full_cpu.to(device_);
    const torch::Tensor target = full_device.slice(1, 0, width);
    const torch::Tensor bonus =
        full_device.flatten()
            .slice(0, width, full_device.numel(), width + 1)
            .view({5, 1});
    for (const bool mask : {false, true}) {
      expect_matches_references(draft_cpu.to(device_), target, bonus, mask);
    }
  }
}

TEST_F(TileLangGreedyPrefixVerifyTest, HandlesEmptyAndIrregularBatchSizes) {
  for (const int64_t batch :
       {0, 1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 47, 48, 49, 97}) {
    SCOPED_TRACE(::testing::Message() << "B=" << batch);
    for (const int64_t width : {0, 1, 3}) {
      SCOPED_TRACE(::testing::Message() << "K=" << width);
      const torch::Tensor full_cpu =
          (torch::arange(batch * (width + 1), torch::kInt32) + 10)
              .reshape({batch, width + 1});
      torch::Tensor draft_cpu = full_cpu.slice(1, 0, width).clone();
      auto values = draft_cpu.accessor<int32_t, 2>();
      for (int64_t row = 0; row < batch && width > 0; row += 2) {
        values[row][row % width] += 1;
      }
      const torch::Tensor full_device = full_cpu.to(device_);
      for (const bool mask : {false, true}) {
        expect_matches_references(draft_cpu.to(device_),
                                  full_device.slice(1, 0, width),
                                  full_device.slice(1, width, width + 1),
                                  mask);
      }
    }
  }
}

TEST_F(TileLangGreedyPrefixVerifyTest,
       VerifiesFiveDraftStepsAcrossBatchMatrix) {
  constexpr int64_t kWidth = 5;
  for (const int64_t batch : {1, 2, 4, 8, 16, 24, 32}) {
    for (const int64_t pattern : {0, 1, 2, 3, 4, 5}) {
      SCOPED_TRACE(::testing::Message()
                   << "B=" << batch << ", pattern=" << pattern);
      const torch::Tensor storage_cpu =
          (torch::arange((batch + 2) * (kWidth + 1), torch::kInt64) +
           (int64_t{1} << 24))
              .reshape({batch + 2, kWidth + 1});
      const torch::Tensor full_cpu = storage_cpu.slice(0, 1, batch + 1);
      torch::Tensor draft_cpu = full_cpu.slice(1, 0, kWidth).clone();
      auto draft_values = draft_cpu.accessor<int64_t, 2>();
      for (int64_t row = 0; row < batch; ++row) {
        int64_t first = kWidth;
        switch (pattern) {
          case 0:
            first = 0;
            break;
          case 1:
            first = kWidth / 2;
            break;
          case 2:
            first = kWidth - 1;
            break;
          case 4:
            first = 1;
            break;
          case 5:
            first = row % (kWidth + 1);
            break;
          default:
            break;
        }
        const int64_t end = pattern >= 4 ? std::min(first + 1, kWidth) : kWidth;
        for (int64_t col = first; col < end; ++col) {
          draft_values[row][col] ^= int64_t{1};
        }
      }
      const torch::Tensor storage = storage_cpu.to(device_);
      const torch::Tensor full = storage.slice(0, 1, batch + 1);
      const torch::Tensor draft = draft_cpu.to(device_);
      for (const bool mask : {false, true}) {
        expect_matches_references(draft,
                                  full.slice(1, 0, kWidth),
                                  full.slice(1, kWidth, kWidth + 1),
                                  mask);
      }
      EXPECT_TRUE(torch::equal(storage.cpu(), storage_cpu));
    }
  }
}

TEST_F(TileLangGreedyPrefixVerifyTest, ReadsProductionTargetAndBonusViews) {
  constexpr int64_t kBatch = 7;
  constexpr int64_t kWidth = 3;
  const torch::Tensor full = (torch::arange(kBatch * (kWidth + 1), i32_) + 20)
                                 .view({kBatch, kWidth + 1});
  const torch::Tensor target = full.slice(1, 0, kWidth);
  const torch::Tensor bonus = full.flatten()
                                  .slice(0, kWidth, full.numel(), kWidth + 1)
                                  .view({kBatch, 1});
  torch::Tensor draft = target.clone();
  draft.index_put_({2, 1}, 99);

  ASSERT_EQ(target.stride(0), kWidth + 1);
  ASSERT_EQ(bonus.stride(0), kWidth + 1);
  ASSERT_EQ(bonus.storage_offset(), kWidth);
  ASSERT_FALSE(target.is_contiguous());
  ASSERT_FALSE(bonus.is_contiguous());
  for (const bool mask : {false, true}) {
    expect_matches_references(draft, target, bonus, mask);
  }
}

TEST_F(TileLangGreedyPrefixVerifyTest,
       ReadsIndependentNonUnitStridesAndOffsets) {
  constexpr int64_t kBatch = 5;
  constexpr int64_t kWidth = 7;
  const torch::Tensor full_cpu =
      (torch::arange(kBatch * (kWidth + 1), torch::kInt32) + 10)
          .reshape({kBatch, kWidth + 1});
  torch::Tensor target_storage =
      torch::full({kBatch + 2, 2 * kWidth + 5}, -777, i32_);
  torch::Tensor target =
      target_storage.slice(0, 1, kBatch + 1).slice(1, 3, 3 + 2 * kWidth, 2);
  torch::Tensor bonus = target_storage.slice(0, 1, kBatch + 1)
                            .slice(1, 2 * kWidth + 3, 2 * kWidth + 4);
  torch::Tensor draft_storage =
      torch::full({2 * kWidth + 3, kBatch + 2}, -888, i32_);
  torch::Tensor draft = draft_storage.slice(0, 1, 1 + 2 * kWidth, 2)
                            .slice(1, 1, kBatch + 1)
                            .transpose(0, 1);
  target.copy_(full_cpu.slice(1, 0, kWidth).to(device_));
  bonus.copy_(full_cpu.slice(1, kWidth, kWidth + 1).to(device_));
  draft.copy_(target);
  draft.index_put_({1, 3}, 999);
  const torch::Tensor target_storage_before = target_storage.cpu();
  const torch::Tensor draft_storage_before = draft_storage.cpu();

  ASSERT_EQ(target.stride(1), 2);
  ASSERT_GT(draft.stride(1), 1);
  ASSERT_GT(target.storage_offset(), 0);
  ASSERT_GT(draft.storage_offset(), 0);
  for (const bool mask : {false, true}) {
    expect_matches_references(draft, target, bonus, mask);
  }
  EXPECT_TRUE(torch::equal(target_storage.cpu(), target_storage_before));
  EXPECT_TRUE(torch::equal(draft_storage.cpu(), draft_storage_before));
}

TEST_F(TileLangGreedyPrefixVerifyTest, ReadsStridesLargerThanTheUbTile) {
  constexpr int64_t kBatch = 3;
  constexpr int64_t kWidth = 3;
  constexpr int64_t kColumnStride = 65537;
  constexpr int64_t kRowStride = kWidth * kColumnStride + 5;
  constexpr int64_t kOffset = 3;
  constexpr int64_t kStorageSize = kOffset + kBatch * kRowStride;
  torch::Tensor target_storage = torch::full({kStorageSize}, -777, i32_);
  torch::Tensor draft_storage = torch::full({kStorageSize}, -888, i32_);
  torch::Tensor target = target_storage.as_strided(
      {kBatch, kWidth}, {kRowStride, kColumnStride}, kOffset);
  torch::Tensor draft = draft_storage.as_strided(
      {kBatch, kWidth}, {kRowStride, kColumnStride}, kOffset);
  const torch::Tensor values =
      torch::tensor({{11, 25, 33}}, i32_).expand({kBatch, kWidth});
  target.copy_(values);
  draft.copy_(target);
  draft.index_put_({1, 1}, (int32_t{1} << 24) + 25);
  const torch::Tensor bonus = torch::full({kBatch, 1}, 44, i32_);
  const torch::Tensor target_storage_before = target_storage.cpu();
  const torch::Tensor draft_storage_before = draft_storage.cpu();

  for (const bool mask : {false, true}) {
    expect_matches_references(draft, target, bonus, mask);
  }
  EXPECT_TRUE(torch::equal(target_storage.cpu(), target_storage_before));
  EXPECT_TRUE(torch::equal(draft_storage.cpu(), draft_storage_before));
}

TEST_F(TileLangGreedyPrefixVerifyTest,
       AcceptsZeroStrideAndAliasedReadOnlyInputs) {
  const torch::Tensor scalar = torch::tensor({{int32_t{1} << 30}}, i32_);
  const torch::Tensor draft = scalar.expand({9, 3});
  const torch::Tensor target = scalar.expand({9, 3});
  const torch::Tensor bonus = scalar.expand({9, 1});
  ASSERT_EQ(draft.stride(0), 0);
  ASSERT_EQ(draft.stride(1), 0);
  for (const bool mask : {false, true}) {
    expect_matches_references(draft, target, bonus, mask);
  }

  const torch::Tensor target_row = torch::tensor({{11, 25, 33}}, i32_);
  const torch::Tensor draft_row = torch::tensor({{11, 22, 33}}, i32_);
  expect_matches_references(draft_row.expand({9, 3}),
                            target_row.expand({9, 3}),
                            bonus,
                            /*mask=*/true);
}

TEST_F(TileLangGreedyPrefixVerifyTest, ComparesAndWritesAllInt32BitsExactly) {
  constexpr int32_t kAboveFloat32 = (int32_t{1} << 24) + 1;
  constexpr int32_t kLargest = std::numeric_limits<int32_t>::max();
  const torch::Tensor draft =
      torch::tensor({{kAboveFloat32 - 1, kLargest - 1, kLargest},
                     {kAboveFloat32, kLargest - 2, kLargest},
                     {kAboveFloat32, kLargest - 1, kLargest - 1},
                     {kAboveFloat32, kLargest - 1, kLargest}},
                    i32_);
  const torch::Tensor target =
      torch::tensor({{kAboveFloat32, kLargest - 1, kLargest}}, i32_)
          .repeat({4, 1});
  const torch::Tensor bonus = torch::full({4, 1}, kLargest, i32_);
  for (const bool mask : {false, true}) {
    expect_matches_references(draft, target, bonus, mask);
  }
}

TEST_F(TileLangGreedyPrefixVerifyTest, PreservesNegativeAndExtremeInt32Bits) {
  constexpr int32_t kSmallest = std::numeric_limits<int32_t>::min();
  constexpr int32_t kLargest = std::numeric_limits<int32_t>::max();
  const torch::Tensor target =
      torch::tensor({{kSmallest, -1, kLargest}, {-7, kSmallest + 1, -2}}, i32_);
  const torch::Tensor draft =
      torch::tensor({{kSmallest, -1, kLargest}, {-7, kSmallest + 2, -2}}, i32_);
  const torch::Tensor bonus = torch::tensor({{kSmallest}, {-9}}, i32_);
  for (const bool mask : {false, true}) {
    expect_matches_references(draft, target, bonus, mask);
  }
}

TEST_F(TileLangGreedyPrefixVerifyTest,
       ConvertsCallerDtypesExactlyInStridedViews) {
  constexpr int32_t kAdjacent = (int32_t{1} << 24) + 1;
  constexpr int32_t kSmallest = std::numeric_limits<int32_t>::min();
  constexpr int32_t kLargest = std::numeric_limits<int32_t>::max();
  for (const torch::ScalarType draft_dtype : {torch::kInt32, torch::kInt64}) {
    for (const torch::ScalarType target_dtype :
         {torch::kInt32, torch::kInt64}) {
      for (const torch::ScalarType bonus_dtype :
           {torch::kInt32, torch::kInt64}) {
        const torch::Tensor storage =
            torch::tensor({{kAdjacent, kAdjacent + 1, kLargest, kSmallest},
                           {kSmallest, -1, kLargest, kAdjacent}},
                          i32_.dtype(target_dtype));
        const torch::Tensor target = storage.slice(1, 0, 3);
        const torch::Tensor bonus_storage = storage.to(bonus_dtype);
        const torch::Tensor bonus = bonus_storage.flatten()
                                        .slice(0, 3, storage.numel(), 4)
                                        .view({2, 1});
        const torch::Tensor draft_storage =
            torch::tensor({{kAdjacent, 0, kAdjacent, 0, kLargest, 0},
                           {kSmallest, 0, -1, 0, kLargest, 0}},
                          i32_.dtype(draft_dtype));
        const torch::Tensor draft = draft_storage.slice(1, 0, 6, 2);
        ASSERT_EQ(draft.stride(1), 2);
        ASSERT_EQ(target.stride(0), 4);
        ASSERT_EQ(bonus.stride(0), 4);
        ASSERT_EQ(bonus.storage_offset(), 3);
        for (const bool mask : {false, true}) {
          expect_matches_references(draft, target, bonus, mask);
        }
      }
    }
  }
}

TEST_F(TileLangGreedyPrefixVerifyTest, Int64InputsHandleEmptyAndMultiTileRows) {
  const auto options = i32_.dtype(torch::kInt64);
  for (const int64_t batch : {0, 3}) {
    for (const int64_t width : {0, 1, 65, 129}) {
      const torch::Tensor storage =
          (torch::arange(batch * (width + 1), options) + (int64_t{1} << 24))
              .view({batch, width + 1});
      const torch::Tensor target = storage.slice(1, 0, width);
      const torch::Tensor bonus = storage.slice(1, width, width + 1);
      torch::Tensor draft = target.clone();
      if (batch > 0 && width > 0) {
        draft.index_put_({0, 0}, -7);
        draft.index_put_({1, width - 1}, -9);
      }
      for (const bool mask : {false, true}) {
        expect_matches_references(draft, target, bonus, mask);
      }
    }
  }
  const torch::Tensor value =
      torch::tensor({{std::numeric_limits<int32_t>::max()}}, options);
  for (const bool mask : {false, true}) {
    expect_matches_references(
        value.expand({7, 3}), value.expand({7, 3}), value.expand({7, 1}), mask);
  }
}

TEST_F(TileLangGreedyPrefixVerifyTest, RepeatedCallsDoNotReuseMaskedTail) {
  const torch::Tensor target = torch::arange(33 * 3, i32_).view({33, 3}) + 100;
  const torch::Tensor bonus = torch::arange(33, i32_).view({33, 1}) + 1000;
  for (int64_t iteration = 0; iteration < 8; ++iteration) {
    torch::Tensor draft = target.clone();
    draft.index_put_({iteration, iteration % 3}, -123);
    expect_matches_references(draft, target, bonus, iteration % 2 == 0);
  }
}

TEST_F(TileLangGreedyPrefixVerifyTest, UsesTheInputDevicesCurrentStream) {
  const auto stream =
      c10_npu::getStreamFromPool(/*isHighPriority=*/false, device_.index());
  c10_npu::NPUStreamGuard stream_guard(stream);
  const torch::Tensor target = torch::tensor({{11, 25, 33}}, i32_);
  const torch::Tensor draft = torch::tensor({{11, 22, 33}}, i32_);
  const torch::Tensor bonus = torch::tensor({{44}}, i32_);
  expect_matches_references(draft, target, bonus, /*mask=*/true);
  EXPECT_EQ(aclrtSynchronizeStream(stream.stream()), ACL_SUCCESS);
}

TEST_F(TileLangGreedyPrefixVerifyTest,
       RejectsMetadataOutsideInt32BeforeAllocation) {
  constexpr int64_t kTooLarge =
      static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
  const torch::Tensor scalar = torch::ones({1}, i32_);
  const torch::Tensor ids = scalar.view({1, 1});
  const torch::Tensor huge_stride = scalar.as_strided({1, 1}, {kTooLarge, 1});
  const torch::Tensor huge_batch = scalar.expand({kTooLarge, 1});
  const torch::Tensor huge_output = scalar.expand({kTooLarge / 2, 1});
  const torch::Tensor huge_width = scalar.expand({1, kTooLarge - 1});
  EXPECT_DEATH_IF_SUPPORTED(
      static_cast<void>(greedy_prefix_verify(huge_stride, ids, ids, true)),
      "stride exceeds INT32");
  EXPECT_DEATH_IF_SUPPORTED(static_cast<void>(greedy_prefix_verify(
                                huge_batch, huge_batch, huge_batch, true)),
                            "shape exceeds INT32");
  EXPECT_DEATH_IF_SUPPORTED(static_cast<void>(greedy_prefix_verify(
                                huge_output, huge_output, huge_output, true)),
                            "output element count exceeds INT32");
  EXPECT_DEATH_IF_SUPPORTED(static_cast<void>(greedy_prefix_verify(
                                huge_width, huge_width, ids, true)),
                            "output width exceeds INT32");
}

TEST_F(TileLangGreedyPrefixVerifyTest, RejectsInvalidMetadataBeforeLaunch) {
  const torch::Tensor target = torch::ones({2, 3}, i32_);
  const torch::Tensor bonus = torch::ones({2, 1}, i32_);
  const torch::Tensor mismatched_draft = torch::ones({2, 2}, i32_);
  const torch::Tensor invalid_bonus = torch::ones({2, 2}, i32_);
  const torch::Tensor wrong_dtype = target.to(torch::kFloat32);
  const torch::Tensor wrong_bonus_dtype = bonus.to(torch::kFloat32);
  const torch::Tensor cpu_target = target.cpu();
  const torch::Tensor wrong_rank = target.flatten();
  EXPECT_DEATH_IF_SUPPORTED(
      static_cast<void>(greedy_prefix_verify(target, cpu_target, bonus, true)),
      "Check failed");
  EXPECT_DEATH_IF_SUPPORTED(static_cast<void>(greedy_prefix_verify(
                                wrong_rank, wrong_rank, bonus, true)),
                            "Check failed");
  EXPECT_DEATH_IF_SUPPORTED(static_cast<void>(greedy_prefix_verify(
                                torch::Tensor(), target, bonus, true)),
                            "Check failed");
  EXPECT_DEATH_IF_SUPPORTED(static_cast<void>(greedy_prefix_verify(
                                mismatched_draft, target, bonus, true)),
                            "Check failed");
  EXPECT_DEATH_IF_SUPPORTED(static_cast<void>(greedy_prefix_verify(
                                target, target, invalid_bonus, true)),
                            "Check failed");
  for (const bool mask : {false, true}) {
    EXPECT_DEATH_IF_SUPPORTED(static_cast<void>(greedy_prefix_verify(
                                  wrong_dtype, target, bonus, mask)),
                              "Check failed");
    EXPECT_DEATH_IF_SUPPORTED(static_cast<void>(greedy_prefix_verify(
                                  target, wrong_dtype, bonus, mask)),
                              "Check failed");
    EXPECT_DEATH_IF_SUPPORTED(static_cast<void>(greedy_prefix_verify(
                                  target, target, wrong_bonus_dtype, mask)),
                              "Check failed");
    EXPECT_DEATH_IF_SUPPORTED(
        static_cast<void>(greedy_prefix_verify(
            wrong_dtype, wrong_dtype, wrong_bonus_dtype, mask)),
        "Check failed");
  }
}

}  // namespace
}  // namespace xllm::kernel::npu::tilelang
