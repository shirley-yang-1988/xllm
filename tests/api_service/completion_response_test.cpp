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

#include "api_service/completion_response.h"

#include <gtest/gtest.h>
#include <json2pb/json_to_pb.h>
#include <json2pb/pb_to_json.h>

#include <cstdint>
#include <limits>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "api_service/completion_json_parser.h"
#include "core/framework/request/request_output.h"
#include "core/framework/request/request_params.h"

namespace xllm {
namespace {

void expect_response_json(const proto::CompletionResponse& response,
                          const nlohmann::json& expected) {
  // Match CompletionCall's JSON options, including omission of empty arrays.
  json2pb::Pb2JsonOptions options;
  options.bytes_to_base64 = false;
  options.jsonify_empty_array = false;
  std::string json_text;
  std::string error;
  ASSERT_TRUE(
      json2pb::ProtoMessageToJson(response, &json_text, options, &error))
      << error;
  EXPECT_EQ(nlohmann::json::parse(json_text), expected);
}

TEST(CompletionResponseTest, ReturnTokenIdsIsOptInWithoutEnablingLogprobs) {
  const proto::CompletionRequest default_request;
  EXPECT_FALSE(default_request.has_return_token_ids());
  EXPECT_FALSE(default_request.return_token_ids());

  for (const bool return_token_ids : {false, true}) {
    const nlohmann::json body = {{"model", "test-model"},
                                 {"prompt", {11, 12}},
                                 {"temperature", 0},
                                 {"return_token_ids", return_token_ids}};
    const auto [status, processed_json] =
        preprocess_completion_prompt(body.dump());
    ASSERT_TRUE(status.ok()) << status.message();
    proto::CompletionRequest request;
    json2pb::Json2PbOptions options;
    std::string error;
    ASSERT_TRUE(
        json2pb::JsonToProtoMessage(processed_json, &request, options, &error))
        << error;
    EXPECT_TRUE(request.has_return_token_ids());
    EXPECT_EQ(request.return_token_ids(), return_token_ids);
    EXPECT_FALSE(request.has_logprobs());
    EXPECT_EQ((std::vector<int32_t>(request.token_ids().begin(),
                                    request.token_ids().end())),
              (std::vector<int32_t>{11, 12}));
    const RequestParams params(request, "", "");
    EXPECT_FALSE(params.logprobs);
    EXPECT_EQ(params.temperature, 0.0f);
  }
}

TEST(CompletionResponseTest, OptOutPreservesTextAndExistingLogprobs) {
  SequenceOutput output;
  output.index = 2;
  output.text = "x";
  output.token_ids = {123};
  LogProb logprob;
  logprob.token = "x";
  logprob.token_id = 123;
  logprob.logprob = -0.5f;
  output.logprobs = std::vector<LogProb>{logprob};
  const nlohmann::json expected = {{"choices",
                                    {{{"index", 2},
                                      {"text", "x"},
                                      {"logprobs",
                                       {{"tokens", {"x"}},
                                        {"token_ids", {123}},
                                        {"token_logprobs", {-0.5}}}}}}}};

  for (const bool explicitly_disabled : {false, true}) {
    proto::CompletionRequest request;
    if (explicitly_disabled) {
      request.set_return_token_ids(false);
    }
    proto::CompletionResponse full_response;
    api_service::set_completion_choice(
        full_response.add_choices(), output, request.return_token_ids());
    expect_response_json(full_response, expected);

    proto::CompletionResponse delta_response;
    ASSERT_TRUE(api_service::set_completion_delta(
        &delta_response, output, request.return_token_ids()));
    expect_response_json(delta_response, expected);
  }
}

TEST(CompletionResponseTest, OptInReturnsFullIdsWithoutLogprobs) {
  SequenceOutput output;
  output.index = 0;
  output.text = "decoded text";
  output.token_ids = {0, 16777217, std::numeric_limits<int32_t>::max()};
  proto::CompletionRequest request;
  request.set_return_token_ids(true);
  proto::CompletionResponse response;
  api_service::set_completion_choice(
      response.add_choices(), output, request.return_token_ids());

  ASSERT_EQ(response.choices_size(), 1);
  EXPECT_FALSE(response.choices(0).has_logprobs());
  expect_response_json(response,
                       {{"choices",
                         {{{"index", 0},
                           {"text", "decoded text"},
                           {"token_ids", {0, 16777217, 2147483647}}}}}});
  EXPECT_EQ(output.token_ids, (std::vector<int32_t>{0, 16777217, 2147483647}));
  EXPECT_FALSE(output.logprobs.has_value());
}

TEST(CompletionResponseTest, EmptyTextIdsAreEmittedOnceBeforeFinish) {
  SequenceOutput output;
  output.index = 3;
  output.token_ids = {41, 42};
  output.finish_reason = "length";
  proto::CompletionResponse response;
  response.set_id("request-1");

  ASSERT_TRUE(api_service::set_completion_delta(
      &response, output, /*return_token_ids=*/true));
  ASSERT_EQ(response.choices_size(), 1);
  EXPECT_FALSE(response.choices(0).has_finish_reason());
  expect_response_json(
      response,
      {{"id", "request-1"},
       {"choices", {{{"index", 3}, {"text", ""}, {"token_ids", {41, 42}}}}}});

  // Reuse the same response exactly as the streaming serializer does.
  ASSERT_TRUE(api_service::set_completion_finish(&response, output));
  ASSERT_EQ(response.choices_size(), 1);
  EXPECT_EQ(response.choices(0).token_ids_size(), 0);
  EXPECT_FALSE(response.choices(0).has_logprobs());
  expect_response_json(
      response,
      {{"id", "request-1"},
       {"choices",
        {{{"index", 3}, {"text", ""}, {"finish_reason", "length"}}}}});
  EXPECT_EQ(output.token_ids, (std::vector<int32_t>{41, 42}));
}

TEST(CompletionResponseTest, OptOutSkipsEmptyTextIdsButKeepsFinish) {
  SequenceOutput output;
  output.index = 0;
  output.token_ids = {41, 42};
  output.finish_reason = "stop";
  proto::CompletionResponse response;

  EXPECT_FALSE(api_service::set_completion_delta(
      &response, output, /*return_token_ids=*/false));
  EXPECT_EQ(response.choices_size(), 0);
  ASSERT_TRUE(api_service::set_completion_finish(&response, output));
  expect_response_json(
      response,
      {{"choices", {{{"index", 0}, {"text", ""}, {"finish_reason", "stop"}}}}});
}

TEST(CompletionResponseTest, TextOnlyAndFinishOnlyChunksDoNotRepeatIds) {
  proto::CompletionResponse response;
  SequenceOutput output;
  output.index = 0;
  output.token_ids = {41};
  ASSERT_TRUE(api_service::set_completion_delta(
      &response, output, /*return_token_ids=*/true));

  output.token_ids.clear();
  output.text = "decoded later";
  ASSERT_TRUE(api_service::set_completion_delta(
      &response, output, /*return_token_ids=*/true));
  expect_response_json(
      response, {{"choices", {{{"index", 0}, {"text", "decoded later"}}}}});

  output.text.clear();
  EXPECT_FALSE(api_service::set_completion_delta(
      &response, output, /*return_token_ids=*/true));
  EXPECT_FALSE(api_service::set_completion_finish(&response, output));
  output.finish_reason = "length";
  ASSERT_TRUE(api_service::set_completion_finish(&response, output));
  expect_response_json(
      response,
      {{"choices",
        {{{"index", 0}, {"text", ""}, {"finish_reason", "length"}}}}});
}

}  // namespace
}  // namespace xllm
