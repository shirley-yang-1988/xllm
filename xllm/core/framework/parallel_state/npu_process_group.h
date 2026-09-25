/* Copyright 2025-2026 The xLLM Authors.

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

#pragma once

#include <hccl/hccl_types.h>
#include <torch_npu/csrc/core/npu/NPUEvent.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "core/common/global_flags.h"
#include "core/framework/parallel_state/process_group.h"
#include "hccl/hccl.h"

namespace xllm {

class ProcessGroupImpl : public ProcessGroup {
 public:
  // Constructor.
  ProcessGroupImpl(int rank,
                   int world_size,
                   const torch::Device& device,
                   HcclComm comm);

  ProcessGroupImpl(int rank,
                   int world_size,
                   int rank_size,
                   int port,
                   bool trans,
                   const std::string& host,
                   const std::string& group_name,
                   const torch::Device& device);

  ProcessGroupImpl(int32_t global_rank,
                   int32_t local_rank,
                   const std::vector<int32_t>& group_ranks,
                   int32_t world_size,
                   int32_t rank_size,
                   int32_t port,
                   const std::string& host,
                   const std::string& group_name,
                   const torch::Device& device);

  // Destructor.
  ~ProcessGroupImpl() override;

  std::string hccl_comm_name(bool init_comm = true) override;
  HcclComm hccl_comm() override;

 private:
  HcclComm comm_ = nullptr;
  c10_npu::NPUStream comm_stream_;
};

// Issue an in-place SUM all-reduce of `input` over the communicator `comm`
// stands for, submitted on the stream the caller is already on. The
// communicator is the one of the group to reduce over: the Python model
// executor reduces over process groups the Python side created itself, so this
// side cannot look the communicator up and the caller passes the handle. It
// crosses as a plain integer because a torch schema has no pointer type
// ("npu_all_reduce" in npu_ops_library.cpp).
//
// Submitting through torch_npu's ProcessGroupHCCL instead puts the collective
// on the communication stream that group owns and makes the caller's stream
// wait for it, which a captured ACLGraph then carries as a cross-stream edge
// around every collective.  A collective submitted here carries no such edge.
// The communicator must expand on AIV (HCCL_OP_EXPANSION_MODE=AIV): an
// AICPU-expanded collective cannot run on the capture stream and fails the
// capture instead of degrading silently.
void all_reduce_on_current_stream(torch::Tensor& input, int64_t comm);

// TODO: LOG HcclGetErrorString(r)
#if defined(USE_NPU)
#define HCCLCHECK(cmd)                     \
  do {                                     \
    HcclResult r = cmd;                    \
    if (r != HCCL_SUCCESS) {               \
      LOG(FATAL) << "Failed, HCCL error."; \
    }                                      \
  } while (0)
#endif
}  // namespace xllm
