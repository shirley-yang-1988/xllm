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

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>

#include "core/framework/sampling/draft_proposal.h"
#include "core/framework/sampling/rejection_sampler.h"
#include "core/framework/sampling/sampler.h"
#include "core/platform/device.h"

namespace xllm {
namespace {

std::tuple<torch::Tensor, torch::Tensor> greedy_reference(
    const torch::Tensor& draft,
    const torch::Tensor& target,
    const torch::Tensor& bonus,
    bool mask) {
  const torch::Tensor full = torch::cat({target, bonus}, /*dim=*/-1);
  if (!mask) {
    return {full, torch::Tensor()};
  }
  const torch::Tensor accepted = target.eq(draft).to(torch::kInt64);
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

std::tuple<torch::Tensor, torch::Tensor> greedy_integer_reference(
    const torch::Tensor& draft,
    const torch::Tensor& target,
    const torch::Tensor& bonus) {
  const int64_t batch = draft.size(0);
  const int64_t width = draft.size(1);
  torch::Tensor full = torch::empty({batch, width + 1}, torch::kInt32);
  torch::Tensor masked = torch::empty_like(full);
  const auto draft_ids = draft.accessor<int64_t, 2>();
  const auto target_ids = target.accessor<int64_t, 2>();
  const auto bonus_ids = bonus.accessor<int64_t, 2>();
  auto full_ids = full.accessor<int32_t, 2>();
  auto masked_ids = masked.accessor<int32_t, 2>();
  for (int64_t row = 0; row < batch; ++row) {
    int64_t first_reject = width;
    for (int64_t column = 0; column < width; ++column) {
      if (draft_ids[row][column] != target_ids[row][column]) {
        first_reject = column;
        break;
      }
    }
    for (int64_t column = 0; column <= width; ++column) {
      const int64_t value =
          column == width ? bonus_ids[row][0] : target_ids[row][column];
      full_ids[row][column] = static_cast<int32_t>(value);
      masked_ids[row][column] =
          column <= first_reject ? static_cast<int32_t>(value) : -1;
    }
  }
  return {full, masked};
}

// cc_test supplies the NPU/Python runtime; do not finalize it between suites.
class GreedyPrefixVerifyNpuTest : public ::testing::Test {
 protected:
  torch::Tensor target_logits() const {
    return torch::tensor({{{-1.0F, 5.0F, 5.0F, 1.0F, 0.0F},
                           {0.0F, 1.0F, 2.0F, 4.0F, 3.0F},
                           {4.0F, 0.0F, 1.0F, 2.0F, 3.0F},
                           {0.0F, 4.0F, 1.0F, 2.0F, 3.0F}},
                          {{0.0F, 1.0F, 2.0F, 4.0F, 3.0F},
                           {0.0F, 1.0F, 2.0F, 3.0F, 4.0F},
                           {0.0F, 4.0F, 1.0F, 2.0F, 3.0F},
                           {4.0F, 0.0F, 1.0F, 2.0F, 3.0F}}},
                         f32_);
  }

  void check_stochastic_forward(bool all_random,
                                bool dense_draft,
                                bool enable_fused_kernel = false) {
    const torch::Tensor logits = target_logits().repeat({2, 1, 1});
    const torch::Tensor target_probs =
        logits.slice(1, 0, 3).softmax(/*dim=*/-1, /*dtype=*/torch::kFloat32);
    const torch::Tensor draft =
        torch::tensor({{1, 2, 0}, {0, 4, 1}, {0, 3, 0}, {3, 2, 1}}, i64_);
    const torch::Tensor bonus = torch::tensor({{4}, {2}, {1}, {0}}, i64_);
    const torch::Tensor do_sample =
        all_random ? torch::ones({4}, i32_.dtype(torch::kBool))
                   : torch::tensor({false, true, false, true},
                                   i32_.dtype(torch::kBool));
    // q=p makes random verification accept every draft, even non-argmax IDs.
    // This deterministically distinguishes the two candidates in mixed mode.
    const DraftProposal proposal(
        draft,
        dense_draft ? std::optional<torch::Tensor>(target_probs.clone())
                    : std::nullopt);
    const RejectionSampler sampler(do_sample,
                                   /*all_random_sample=*/all_random,
                                   /*all_greedy_sample=*/false,
                                   /*logprobs=*/true,
                                   /*max_top_logprobs=*/0,
                                   enable_fused_kernel);
    const Device npu_device(device_);
    for (const bool mask : {false, true}) {
      for (const uint64_t seed : {uint64_t{100}, uint64_t{2026}}) {
        SCOPED_TRACE(::testing::Message()
                     << "all_random=" << all_random << " dense_draft="
                     << dense_draft << " mask=" << mask << " seed=" << seed
                     << " enable_fused_kernel=" << enable_fused_kernel);
        npu_device.set_seed(seed);
        const torch::Tensor uniform_rand =
            torch::rand(draft.sizes(), target_probs.options());
        const auto [random_full, random_masked] =
            RejectionSampler::random_sample(
                proposal, target_probs, uniform_rand, bonus, mask);
        const auto [greedy_full, greedy_masked] =
            greedy_reference(draft.to(torch::kInt32),
                             target_probs.argmax(-1).to(torch::kInt32),
                             bonus.to(torch::kInt32),
                             mask);
        const torch::Tensor expected_full =
            all_random ? random_full
                       : torch::where(
                             do_sample.unsqueeze(-1), random_full, greedy_full);
        torch::Tensor expected_tokens = expected_full;
        if (mask) {
          expected_tokens = all_random ? random_masked
                                       : torch::where(do_sample.unsqueeze(-1),
                                                      random_masked,
                                                      greedy_masked);
        }
        const torch::Tensor expected_cpu = expected_tokens.cpu();
        const torch::Tensor expected_logprobs =
            logits.log_softmax(-1, torch::kFloat32)
                .gather(-1, expected_full.unsqueeze(-1))
                .squeeze(-1)
                .cpu();
        const torch::Tensor next_random_reference =
            torch::rand({17}, f32_).cpu();
        if (dense_draft) {
          EXPECT_TRUE(torch::equal(random_full.cpu(),
                                   torch::cat({draft, bonus}, -1).cpu()));
        }

        npu_device.set_seed(seed);
        const SampleOutput output =
            sampler.forward(proposal, logits, bonus, mask);
        const torch::Tensor next_random_actual = torch::rand({17}, f32_).cpu();
        EXPECT_EQ(output.next_tokens.scalar_type(), torch::kInt64);
        EXPECT_EQ(output.next_tokens.device(), device_);
        EXPECT_TRUE(torch::equal(output.next_tokens.cpu(), expected_cpu));
        EXPECT_TRUE(torch::equal(output.logprobs.cpu(), expected_logprobs));
        EXPECT_TRUE(torch::equal(next_random_actual, next_random_reference))
            << "Verification changed NPU RNG consumption";
        EXPECT_EQ(draft.scalar_type(), torch::kInt64);
        EXPECT_EQ(bonus.scalar_type(), torch::kInt64);
        EXPECT_EQ(do_sample.sizes(), torch::IntArrayRef({4}));
      }
    }
  }

  const torch::Device device_{"npu:0"};
  const torch::TensorOptions i32_ =
      torch::TensorOptions().dtype(torch::kInt32).device(device_);
  const torch::TensorOptions i64_ = i32_.dtype(torch::kInt64);
  const torch::TensorOptions f32_ = i32_.dtype(torch::kFloat32);
};

TEST_F(GreedyPrefixVerifyNpuTest, TokenIdHelperPreservesInt32StridedContract) {
  const torch::Tensor expected_full = torch::tensor(
      {{11, 25, 33, 44}, {11, 25, 33, 45}, {11, 25, 33, 46}}, torch::kInt32);
  const torch::Tensor storage = expected_full.to(device_);
  const torch::Tensor target = storage.slice(1, 0, 3);
  const torch::Tensor bonus =
      storage.flatten().slice(0, 3, storage.numel(), 4).view({3, 1});
  const torch::Tensor draft =
      torch::tensor({{11, 22, 33}, {11, 25, 33}, {9, 25, 33}}, i32_);
  const torch::Tensor expected_masked = torch::tensor(
      {{11, 25, -1, -1}, {11, 25, 33, 45}, {11, -1, -1, -1}}, torch::kInt32);
  ASSERT_EQ(target.stride(0), 4);
  ASSERT_EQ(bonus.stride(0), 4);
  ASSERT_EQ(bonus.storage_offset(), 3);
  for (const bool mask : {false, true}) {
    const auto [full, masked] = RejectionSampler::greedy_sample_from_token_ids(
        draft, target, bonus, mask);
    EXPECT_EQ(full.scalar_type(), torch::kInt32);
    EXPECT_EQ(full.device(), device_);
    EXPECT_TRUE(torch::equal(full.cpu(), expected_full));
    ASSERT_EQ(masked.defined(), mask);
    if (mask) {
      EXPECT_EQ(masked.scalar_type(), torch::kInt32);
      EXPECT_TRUE(torch::equal(masked.cpu(), expected_masked));
    }
    EXPECT_TRUE(torch::equal(storage.cpu(), expected_full));
  }
}

TEST_F(GreedyPrefixVerifyNpuTest, FiveDraftStepsInt64MtpBatchMatrix) {
  constexpr int64_t kDraftWidth = 5;
  constexpr int64_t kOutputWidth = kDraftWidth + 1;
  constexpr int64_t kStorageOffset = 3;
  const std::array<int64_t, 8> values = {16777216,
                                         16777217,
                                         std::numeric_limits<int32_t>::max(),
                                         std::numeric_limits<int32_t>::min(),
                                         -16777216,
                                         -16777217,
                                         0,
                                         1};
  const std::array<const char*, 6> patterns = {
      "first", "middle", "last", "all", "rematch", "mixed"};
  const std::array<int64_t, 5> reject_positions = {0, 2, 4, 5, 2};
  for (const int64_t batch : {1, 2, 4, 8, 16, 24, 32}) {
    for (int64_t pattern = 0; pattern < 6; ++pattern) {
      torch::Tensor storage_cpu = torch::full(
          {kStorageOffset + batch * kOutputWidth + 4}, -777, torch::kInt64);
      torch::Tensor ids_cpu =
          storage_cpu.narrow(0, kStorageOffset, batch * kOutputWidth)
              .view({batch, kOutputWidth});
      torch::Tensor draft_cpu =
          torch::empty({batch, kDraftWidth}, torch::kInt64);
      auto ids = ids_cpu.accessor<int64_t, 2>();
      auto draft_ids = draft_cpu.accessor<int64_t, 2>();
      for (int64_t row = 0; row < batch; ++row) {
        for (int64_t column = 0; column < kOutputWidth; ++column) {
          ids[row][column] =
              values[static_cast<size_t>(row + column) % values.size()];
        }
        const int64_t first_reject =
            pattern == 5 ? row % kOutputWidth
                         : reject_positions[static_cast<size_t>(pattern)];
        for (int64_t column = 0; column < kDraftWidth; ++column) {
          const bool mismatch =
              column == first_reject || (pattern < 3 && column > first_reject);
          draft_ids[row][column] =
              mismatch ? ids[row][column] ^ int64_t { 1 } : ids[row][column];
        }
      }
      const torch::Tensor target_cpu = ids_cpu.slice(1, 0, kDraftWidth);
      const torch::Tensor bonus_cpu =
          ids_cpu.slice(1, kDraftWidth, kOutputWidth);
      const auto [expected_full, expected_masked] =
          greedy_integer_reference(draft_cpu, target_cpu, bonus_cpu);
      const auto [old_full, old_masked] =
          greedy_reference(draft_cpu, target_cpu, bonus_cpu, /*mask=*/true);
      ASSERT_TRUE(torch::equal(expected_full, old_full.to(torch::kInt32)));
      ASSERT_TRUE(torch::equal(expected_masked, old_masked.to(torch::kInt32)));

      const torch::Tensor storage = storage_cpu.to(device_);
      const torch::Tensor ids_view =
          storage.narrow(0, kStorageOffset, batch * kOutputWidth)
              .view({batch, kOutputWidth});
      const torch::Tensor target = ids_view.slice(1, 0, kDraftWidth);
      const torch::Tensor bonus = ids_view.slice(1, kDraftWidth, kOutputWidth);
      const torch::Tensor draft = draft_cpu.to(device_);
      ASSERT_EQ(target.stride(0), kOutputWidth);
      ASSERT_EQ(target.stride(1), 1);
      ASSERT_EQ(target.storage_offset(), kStorageOffset);
      ASSERT_EQ(bonus.stride(0), kOutputWidth);
      ASSERT_EQ(bonus.storage_offset(), kStorageOffset + kDraftWidth);
      for (const bool mask : {false, true}) {
        SCOPED_TRACE(::testing::Message()
                     << "batch=" << batch
                     << " pattern=" << patterns[static_cast<size_t>(pattern)]
                     << " mask=" << mask);
        const auto [full, masked] =
            RejectionSampler::greedy_sample_from_token_ids(
                draft, target, bonus, mask);
        EXPECT_EQ(full.sizes(), torch::IntArrayRef({batch, kOutputWidth}));
        EXPECT_EQ(full.scalar_type(), torch::kInt32);
        EXPECT_EQ(full.device(), device_);
        EXPECT_TRUE(torch::equal(full.cpu(), expected_full));
        ASSERT_EQ(masked.defined(), mask);
        if (mask) {
          EXPECT_EQ(masked.sizes(), full.sizes());
          EXPECT_EQ(masked.scalar_type(), torch::kInt32);
          EXPECT_EQ(masked.device(), device_);
          EXPECT_TRUE(torch::equal(masked.cpu(), expected_masked));
        }
        EXPECT_EQ(draft.scalar_type(), torch::kInt64);
        EXPECT_EQ(target.scalar_type(), torch::kInt64);
        EXPECT_EQ(bonus.scalar_type(), torch::kInt64);
        EXPECT_TRUE(torch::equal(draft.cpu(), draft_cpu));
        EXPECT_TRUE(torch::equal(storage.cpu(), storage_cpu));
      }
    }
  }
}

TEST_F(GreedyPrefixVerifyNpuTest, TokenIdHelperAcceptsCallerIntegerDtypes) {
  const torch::Tensor expected_full =
      torch::tensor({{1, 3, 0, 4}}, torch::kInt32);
  const torch::Tensor expected_masked =
      torch::tensor({{1, 3, -1, -1}}, torch::kInt32);
  for (const torch::ScalarType draft_dtype : {torch::kInt32, torch::kInt64}) {
    for (const torch::ScalarType target_dtype :
         {torch::kInt32, torch::kInt64}) {
      for (const torch::ScalarType bonus_dtype :
           {torch::kInt32, torch::kInt64}) {
        const torch::Tensor draft =
            torch::tensor({{1, 2, 0}}, i32_.dtype(draft_dtype));
        const torch::Tensor target =
            torch::tensor({{1, 3, 0}}, i32_.dtype(target_dtype));
        const torch::Tensor bonus =
            torch::tensor({{4}}, i32_.dtype(bonus_dtype));
        for (const bool mask : {false, true}) {
          const auto [full, masked] =
              RejectionSampler::greedy_sample_from_token_ids(
                  draft, target, bonus, mask);
          EXPECT_EQ(full.scalar_type(), torch::kInt32);
          EXPECT_TRUE(torch::equal(full.cpu(), expected_full));
          ASSERT_EQ(masked.defined(), mask);
          if (mask) {
            EXPECT_EQ(masked.scalar_type(), torch::kInt32);
            EXPECT_TRUE(torch::equal(masked.cpu(), expected_masked));
          }
          EXPECT_EQ(draft.scalar_type(), draft_dtype);
          EXPECT_EQ(target.scalar_type(), target_dtype);
          EXPECT_EQ(bonus.scalar_type(), bonus_dtype);
        }
      }
    }
  }
}

TEST_F(GreedyPrefixVerifyNpuTest, ScoresKeepArgmaxAndConvertOnlyGreedyIds) {
  const torch::Tensor logits = target_logits().slice(1, 0, 3);
  const torch::Tensor expected_full =
      torch::tensor({{1, 3, 0, 4}, {3, 4, 1, 2}}, torch::kInt32);
  const torch::Tensor expected_masked =
      torch::tensor({{1, 3, -1, -1}, {3, 4, 1, 2}}, torch::kInt32);
  for (const torch::ScalarType dtype : {torch::kInt32, torch::kInt64}) {
    const torch::Tensor draft =
        torch::tensor({{1, 2, 0}, {3, 4, 1}}, i32_.dtype(dtype));
    const torch::Tensor bonus = torch::tensor({{4}, {2}}, i32_.dtype(dtype));
    for (const bool probabilities : {false, true}) {
      const torch::Tensor scores =
          probabilities ? logits.softmax(-1, torch::kFloat32) : logits;
      // The global Sampler's argmax still produces Long, including first-ID
      // ties.
      const torch::Tensor argmax = Sampler::greedy_sample(scores);
      EXPECT_EQ(argmax.scalar_type(), torch::kInt64);
      EXPECT_TRUE(torch::equal(argmax.cpu(),
                               expected_full.slice(1, 0, 3).to(torch::kInt64)));
      for (const bool mask : {false, true}) {
        const auto [full, masked] =
            RejectionSampler::greedy_sample(draft, scores, bonus, mask);
        EXPECT_EQ(full.scalar_type(), torch::kInt32);
        EXPECT_EQ(full.device(), device_);
        EXPECT_TRUE(torch::equal(full.cpu(), expected_full));
        ASSERT_EQ(masked.defined(), mask);
        if (mask) {
          EXPECT_EQ(masked.scalar_type(), torch::kInt32);
          EXPECT_TRUE(torch::equal(masked.cpu(), expected_masked));
        }
        EXPECT_EQ(draft.scalar_type(), dtype);
        EXPECT_EQ(bonus.scalar_type(), dtype);
      }
    }
  }
}

TEST_F(GreedyPrefixVerifyNpuTest, GreedyForwardGathersLogprobsFromFullIds) {
  const torch::Tensor logits = target_logits();
  const torch::Tensor do_sample =
      torch::tensor({false, false}, i32_.dtype(torch::kBool));
  const RejectionSampler sampler(do_sample,
                                 /*all_random_sample=*/false,
                                 /*all_greedy_sample=*/true,
                                 /*logprobs=*/true,
                                 /*max_top_logprobs=*/2);
  const torch::Tensor draft = torch::tensor({{1, 2, 0}, {3, 4, 1}}, i64_);
  const torch::Tensor bonus = torch::tensor({{4}, {2}}, i64_);
  const DraftProposal proposal(draft);
  const torch::Tensor full = torch::tensor({{1, 3, 0, 4}, {3, 4, 1, 2}}, i32_);
  const torch::Tensor masked =
      torch::tensor({{1, 3, -1, -1}, {3, 4, 1, 2}}, i32_);
  const torch::Tensor logprobs = logits.log_softmax(-1, torch::kFloat32);
  const torch::Tensor expected_logprobs =
      logprobs.gather(-1, full.to(torch::kInt64).unsqueeze(-1)).squeeze(-1);
  const auto [top_values, top_tokens] =
      logprobs.topk(/*k=*/2, /*dim=*/-1, /*largest=*/true, /*sorted=*/true);
  for (const bool mask : {false, true}) {
    const SampleOutput output = sampler.forward(proposal, logits, bonus, mask);
    EXPECT_EQ(output.next_tokens.scalar_type(), torch::kInt32);
    EXPECT_EQ(output.next_tokens.device(), device_);
    EXPECT_TRUE(
        torch::equal(output.next_tokens.cpu(), (mask ? masked : full).cpu()));
    EXPECT_TRUE(torch::equal(output.logprobs.cpu(), expected_logprobs.cpu()));
    EXPECT_TRUE(torch::equal(output.top_logprobs.cpu(), top_values.cpu()));
    EXPECT_TRUE(torch::equal(output.top_tokens.cpu(), top_tokens.cpu()));
    EXPECT_EQ(output.logprobs.sizes(), torch::IntArrayRef({2, 4}));
    EXPECT_EQ(output.top_tokens.scalar_type(), torch::kInt64);
  }
  EXPECT_EQ(draft.scalar_type(), torch::kInt64);
  EXPECT_EQ(bonus.scalar_type(), torch::kInt64);
}

TEST_F(GreedyPrefixVerifyNpuTest, GreedyHelperAndForwardPreserveRng) {
  const torch::Tensor logits = target_logits();
  const torch::Tensor target = Sampler::greedy_sample(logits.slice(1, 0, 3));
  const torch::Tensor draft = torch::tensor({{1, 2, 0}, {3, 4, 1}}, i64_);
  const torch::Tensor bonus = torch::tensor({{4}, {2}}, i64_);
  const torch::Tensor do_sample = torch::zeros({2}, i32_.dtype(torch::kBool));
  const DraftProposal proposal(draft);
  const Device npu_device(device_);
  for (const bool mask : {false, true}) {
    const auto [expected_full, expected_masked] =
        greedy_reference(draft.cpu(), target.cpu(), bonus.cpu(), mask);
    for (const bool logprobs : {false, true}) {
      const RejectionSampler sampler(do_sample,
                                     /*all_random_sample=*/false,
                                     /*all_greedy_sample=*/true,
                                     logprobs,
                                     /*max_top_logprobs=*/0);
      for (const uint64_t seed : {uint64_t{100}, uint64_t{2026}}) {
        SCOPED_TRACE(::testing::Message() << "mask=" << mask << " logprobs="
                                          << logprobs << " seed=" << seed);
        npu_device.set_seed(seed);
        const torch::Tensor expected_next_random =
            torch::rand({17}, f32_).cpu();
        npu_device.set_seed(seed);
        const auto [full, masked] =
            RejectionSampler::greedy_sample_from_token_ids(
                draft, target, bonus, mask);
        const SampleOutput output =
            sampler.forward(proposal, logits, bonus, mask);
        const torch::Tensor actual_next_random = torch::rand({17}, f32_).cpu();
        EXPECT_TRUE(torch::equal(expected_next_random, actual_next_random));
        EXPECT_TRUE(torch::equal(full.cpu(), expected_full.to(torch::kInt32)));
        ASSERT_EQ(masked.defined(), mask);
        if (mask) {
          EXPECT_TRUE(
              torch::equal(masked.cpu(), expected_masked.to(torch::kInt32)));
        }
        EXPECT_EQ(output.next_tokens.scalar_type(), torch::kInt32);
        EXPECT_TRUE(torch::equal(
            output.next_tokens.cpu(),
            (mask ? expected_masked : expected_full).to(torch::kInt32)));
        EXPECT_EQ(output.logprobs.defined(), logprobs);
      }
    }
  }
}

TEST_F(GreedyPrefixVerifyNpuTest,
       FusedFlagWithLogprobsKeepsUnfusedSelectionAndRng) {
  // NPU has no rejection_sample fused backend. The existing logprobs guard
  // must keep this supported configuration on the original random path.
  for (const bool all_random : {false, true}) {
    for (const bool dense_draft : {false, true}) {
      check_stochastic_forward(all_random,
                               dense_draft,
                               /*enable_fused_kernel=*/true);
    }
  }
}

TEST_F(GreedyPrefixVerifyNpuTest, MixedForwardPreservesSelectionAndRng) {
  for (const bool dense_draft : {false, true}) {
    check_stochastic_forward(/*all_random=*/false, dense_draft);
  }
}

TEST_F(GreedyPrefixVerifyNpuTest, RandomForwardPreservesInt64AndRng) {
  for (const bool dense_draft : {false, true}) {
    check_stochastic_forward(/*all_random=*/true, dense_draft);
  }
}

}  // namespace
}  // namespace xllm
