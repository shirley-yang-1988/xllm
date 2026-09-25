include(cc_test)

set(sampler_npu_test_srcs "")
set(sampler_npu_test_deps "")
if(USE_NPU)
  list(APPEND sampler_npu_test_srcs
    greedy_prefix_verify_npu_test.cpp
    greedy_prefix_verify_npu_benchmark_test.cpp
  )
  list(APPEND sampler_npu_test_deps :tilelang_kernels :platform)
endif()

cc_test(
  NAME
    sampler_test
  SRCS
    gumbel_sampling_test.cpp
    rejection_sampler_test.cpp
    "${PROJECT_SOURCE_DIR}/xllm/core/framework/sampling/rejection_sampler.cpp"
    sampling_params_test.cpp
    ${sampler_npu_test_srcs}
  DEPS
    absl::strings
    GTest::gtest_main
    :config
    :sampler
    glog::glog
    :state_dict
    ${sampler_npu_test_deps}
)
target_link_libraries(sampler_test PRIVATE brpc OpenSSL::SSL OpenSSL::Crypto leveldb::leveldb protobuf::libprotobuf)
target_link_libraries(sampler_test
                      PUBLIC
                      Python::Python
                      $<$<BOOL:${USE_NPU}>:ascendcl>
                      $<$<BOOL:${USE_NPU}>:unified_dlog>
                      $<$<BOOL:${USE_NPU}>:hccl>
                      $<$<BOOL:${USE_NPU}>:c_sec>
                      $<$<BOOL:${USE_NPU}>:nnopbase>)
add_dependencies(sampler_test brpc-static)

cc_test(
  NAME
    json_object_grammar_test
  SRCS
    json_object_grammar_test.cpp
  DEPS
    :json_object_grammar
    GTest::gtest_main
)

cc_test(
  NAME
    sampler_filter_mask_test
  SRCS
    sampler_filter_mask_test.cpp
  DEPS
    :platform
    :sampler
    GTest::gtest_main
)
target_link_libraries(sampler_filter_mask_test
                      PUBLIC
                      Python::Python
                      $<$<BOOL:${USE_NPU}>:ascendcl>
                      $<$<BOOL:${USE_NPU}>:unified_dlog>
                      $<$<BOOL:${USE_NPU}>:hccl>
                      $<$<BOOL:${USE_NPU}>:c_sec>
                      $<$<BOOL:${USE_NPU}>:nnopbase>)
