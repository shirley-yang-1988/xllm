include(CMakeParseArguments)

# inspired by https://github.com/abseil/abseil-cpp
# cc_test()
# CMake function to imitate Bazel's cc_test rule.
#
# Parameters:
# NAME: name of target (see Usage below)
# SRCS: List of source files for the binary
# DEPS: List of other libraries to be linked in to the binary targets
# COPTS: List of private compile options
# LINKOPTS: List of link options
# ARGS: Command line arguments to test case
#
# Usage:
# cc_library(
#   NAME
#     awesome
#   HDRS
#     "a.h"
#   SRCS
#     "a.cc"
# )
#
# cc_test(
#   NAME
#     awesome_test
#   SRCS
#     "awesome_test.cc"
#   DEPS
#     :awesome
#     GTest::gmock
# )
#
function(cc_test)
  if(NOT BUILD_TESTING)
    return()
  endif()

  cmake_parse_arguments(
    CC_TEST # prefix
    "" # options
    "NAME;ENVIRONMENT" # one value args
    "SRCS;COPTS;LINKOPTS;DEPS;INCLUDES;ARGS;DATA" # multi value args
    ${ARGN}
  )

  # place test data in build directory
  if(CC_TEST_DATA)
    foreach(data ${CC_TEST_DATA})
      configure_file(${data} ${CMAKE_CURRENT_BINARY_DIR}/${data} COPYONLY)
    endforeach()
  endif()

  set(_CC_TEST_SRCS "")
  set(_CC_TEST_INCLUDE_DIRS ${CC_TEST_INCLUDES})
  list(APPEND _CC_TEST_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})

  # xllm test sources live under tests and often include private headers
  # from the mirrored production source directory.
  if(DEFINED XLLM_TESTS_DIR)
    list(APPEND _CC_TEST_INCLUDE_DIRS
      ${PROJECT_SOURCE_DIR}/xllm
      ${XLLM_TESTS_DIR}
      ${XLLM_TESTS_DIR}/core
    )
  endif()

  foreach(src IN LISTS CC_TEST_SRCS)
    if(IS_ABSOLUTE "${src}")
      list(APPEND _CC_TEST_SRCS "${src}")
      get_filename_component(src_dir "${src}" DIRECTORY)
      list(APPEND _CC_TEST_INCLUDE_DIRS "${src_dir}")
      continue()
    endif()

    set(src_path "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
    if(EXISTS "${src_path}")
      list(APPEND _CC_TEST_SRCS "${src}")
      get_filename_component(src_dir "${src_path}" DIRECTORY)
      list(APPEND _CC_TEST_INCLUDE_DIRS "${src_dir}")
      if(DEFINED XLLM_TESTS_DIR)
        file(RELATIVE_PATH xllm_test_src_dir "${XLLM_TESTS_DIR}" "${src_dir}")
        if(NOT xllm_test_src_dir MATCHES "^\\.\\.")
          list(APPEND _CC_TEST_INCLUDE_DIRS
            "${PROJECT_SOURCE_DIR}/xllm/${xllm_test_src_dir}"
          )
        endif()
      endif()
      continue()
    endif()

    list(APPEND _CC_TEST_SRCS "${src}")
  endforeach()

  list(REMOVE_DUPLICATES _CC_TEST_INCLUDE_DIRS)

  add_executable(${CC_TEST_NAME})
  target_sources(${CC_TEST_NAME} PRIVATE ${_CC_TEST_SRCS})
  target_include_directories(${CC_TEST_NAME}
    PUBLIC 
      "$<BUILD_INTERFACE:${COMMON_INCLUDE_DIRS}>" 
      ${_CC_TEST_INCLUDE_DIRS}
  )

  target_compile_options(${CC_TEST_NAME}
    PRIVATE ${CC_TEST_COPTS}
  )

  target_link_libraries(${CC_TEST_NAME}
    PUBLIC ${CC_TEST_DEPS}
    PRIVATE ${CC_TEST_LINKOPTS}
  )

  if(USE_NPU)
    target_sources(${CC_TEST_NAME} PRIVATE
      "${PROJECT_SOURCE_DIR}/tests/npu_test_environment.cpp"
    )
    set(COMMON_LIBS ascendcl Python::Python torch_npu torch_python)
    target_link_libraries(${CC_TEST_NAME} PRIVATE ${COMMON_LIBS})
  endif()

  add_dependencies(all_tests ${CC_TEST_NAME})

  # third_party targets stay in all_tests, so they are still built, but they are
  # not registered with CTest: registration here executes the test binary, and a
  # third_party harness carries its own main() and its own runtime bootstrap
  # (see third_party/torch_npu_ops/triton_npu/test) that xLLM's test
  # registration must not drive. xLLM owns only its own test surface.
  string(FIND "${CMAKE_CURRENT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/third_party/" _cc_test_third_party_pos)
  if(_cc_test_third_party_pos EQUAL 0)
    message(STATUS "cc_test(${CC_TEST_NAME}): third_party target, built but not registered with CTest")
    return()
  endif()

  # gtest_add_tests() derives the case list by scanning the sources for TEST()
  # declarations, so a declaration its pattern does not match is silently absent
  # from CTest even though the binary contains it. Discover the cases from the
  # built binary instead, so registration cannot drift from the binary.
  # The timeout is generous because these binaries link torch and the NPU
  # runtime, and the default 5 seconds can be exceeded while many targets build
  # in parallel.
  set(_cc_test_properties "")
  if(CC_TEST_ENVIRONMENT)
    list(APPEND _cc_test_properties ENVIRONMENT "${CC_TEST_ENVIRONMENT}")
  endif()

  gtest_discover_tests(
    ${CC_TEST_NAME}
    EXTRA_ARGS ${CC_TEST_ARGS}
    TEST_LIST _cc_test_${CC_TEST_NAME}_tests
    PROPERTIES ${_cc_test_properties}
    DISCOVERY_TIMEOUT 60
  )
  #add_test(NAME ${CC_TEST_NAME} COMMAND ${CC_TEST_NAME} ${CC_TEST_ARGS})
endfunction()
