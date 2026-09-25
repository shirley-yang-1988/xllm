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

#pragma once

#include "completion.pb.h"

namespace xllm {

struct SequenceOutput;

namespace api_service {

// Populate a fresh choice from the existing CPU output; finish metadata is
// handled separately so streaming finish chunks never repeat token IDs.
void set_completion_choice(proto::Choice* choice,
                           const SequenceOutput& output,
                           bool return_token_ids);

// Replace the response's choices with a single delta or finish-only choice.
// Return false without changing the response when there is no such chunk.
bool set_completion_delta(proto::CompletionResponse* response,
                          const SequenceOutput& output,
                          bool return_token_ids);
bool set_completion_finish(proto::CompletionResponse* response,
                           const SequenceOutput& output);

}  // namespace api_service
}  // namespace xllm
