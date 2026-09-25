/* Copyright 2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "api_service/completion_response.h"

#include <optional>
#include <vector>

#include "core/framework/request/request_output.h"

namespace xllm {
namespace api_service {
namespace {

void set_logprobs(proto::Choice* choice,
                  const std::optional<std::vector<LogProb>>& logprobs) {
  if (!logprobs.has_value() || logprobs.value().empty()) {
    return;
  }

  auto* proto_logprobs = choice->mutable_logprobs();
  // One entry per generated token, so the three parallel fields would otherwise
  // grow from empty and re-copy themselves O(log n) times per response.
  const int num_logprobs = static_cast<int>(logprobs.value().size());
  proto_logprobs->mutable_tokens()->Reserve(num_logprobs);
  proto_logprobs->mutable_token_ids()->Reserve(num_logprobs);
  proto_logprobs->mutable_token_logprobs()->Reserve(num_logprobs);
  for (const auto& logprob : logprobs.value()) {
    proto_logprobs->add_tokens(logprob.token);
    proto_logprobs->add_token_ids(logprob.token_id);
    proto_logprobs->add_token_logprobs(logprob.logprob);
  }
}

}  // namespace

void set_completion_choice(proto::Choice* choice,
                           const SequenceOutput& output,
                           bool return_token_ids) {
  choice->set_index(output.index);
  choice->set_text(output.text);
  set_logprobs(choice, output.logprobs);
  if (return_token_ids) {
    choice->mutable_token_ids()->Add(output.token_ids.begin(),
                                     output.token_ids.end());
  }
}

bool set_completion_delta(proto::CompletionResponse* response,
                          const SequenceOutput& output,
                          bool return_token_ids) {
  if (output.text.empty() && (!return_token_ids || output.token_ids.empty())) {
    return false;
  }
  response->clear_choices();
  set_completion_choice(response->add_choices(), output, return_token_ids);
  return true;
}

bool set_completion_finish(proto::CompletionResponse* response,
                           const SequenceOutput& output) {
  if (!output.finish_reason.has_value()) {
    return false;
  }
  response->clear_choices();
  auto* choice = response->add_choices();
  choice->set_index(output.index);
  choice->set_text("");
  choice->set_finish_reason(output.finish_reason.value());
  return true;
}

}  // namespace api_service
}  // namespace xllm
