// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstddef>
#include <ios>
#include <memory>
#include <ostream>
#include <streambuf>
#include <string>
#include <utility>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/frontend/tensorflow_lite/frontend.hpp"
#include "openvino/frontend/tensorflow_lite/graph_iterator.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/c/internal/litert_logging_helper.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_op_code.h"
#include "litert/c/options/litert_intel_openvino_options.h"
#include "litert/cc/internal/litert_extended_model.h"
#include "litert/cc/internal/litert_handle.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_opaque_options.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_intel_openvino_options.h"
#include "litert/vendors/c/litert_compiler_plugin.h"
#include "litert/vendors/intel_openvino/compiler/graph_iterator.h"
#include "litert/vendors/intel_openvino/compiler/openvino_soc_config.h"

namespace {

constexpr char kPluginManufacturer[] = "IntelOpenVINO";

constexpr LiteRtOpCode kSupportedOps[] = {
    kLiteRtOpCodeTflConv2d,
    kLiteRtOpCodeTflDepthwiseConv2d,
    kLiteRtOpCodeTflSplit,
    kLiteRtOpCodeTflFullyConnected,
    kLiteRtOpCodeTflAdd,
    kLiteRtOpCodeTflReshape,
    kLiteRtOpCodeTflMean,
    kLiteRtOpCodeTflResizeBilinear,
    kLiteRtOpCodeTflResizeNearestNeighbor,
    kLiteRtOpCodeTflConcatenation,
    kLiteRtOpCodeTflMaxPool2d,
    kLiteRtOpCodeTflAveragePool2d,
    kLiteRtOpCodeTflMul,
    kLiteRtOpCodeTflTransposeConv,
    kLiteRtOpCodeTflSoftmax,
    kLiteRtOpCodeTflMirrorPad,
    kLiteRtOpCodeTflStridedSlice,
    kLiteRtOpCodeTflDepthToSpace,
    kLiteRtOpCodeTflGather,
    kLiteRtOpCodeTflBatchMatmul,
    kLiteRtOpCodeTflLeakyRelu,
    kLiteRtOpCodeTflPack,
    kLiteRtOpCodeTflCast,
    kLiteRtOpCodeTflDiv,
    kLiteRtOpCodeTflCumsum,
    kLiteRtOpCodeTflSub,
    kLiteRtOpCodeTflGelu,
    kLiteRtOpCodeTflGatherNd,
    kLiteRtOpCodeTflSum,
    kLiteRtOpCodeTflReduceMax,
    kLiteRtOpCodeTflReduceAll,
    kLiteRtOpCodeTflEmbeddingLookup,
    kLiteRtOpCodeTflConv3d,
    kLiteRtOpCodeTflArgMax,
    kLiteRtOpCodeTflOneHot,
    kLiteRtOpCodeTflUnpack,
    // These ops donot call get_attribute
    kLiteRtOpCodeTflDequantize,
    kLiteRtOpCodeTflLogistic,
    kLiteRtOpCodeTflRelu,
    kLiteRtOpCodeTflTanh,
    kLiteRtOpCodeTflPad,
    kLiteRtOpCodeTflTranspose,
    kLiteRtOpCodeTflSlice,
    kLiteRtOpCodeTflQuantize,
    kLiteRtOpCodeTflRange,
    kLiteRtOpCodeTflBroadcastTo,
    kLiteRtOpCodeTflPadv2,
    kLiteRtOpCodeTflEqual,
    kLiteRtOpCodeTflNotEqual,
    kLiteRtOpCodeTflExp,
    kLiteRtOpCodeTflReverseV2,
    kLiteRtOpCodeTflMaximum,
    kLiteRtOpCodeTflLogicalOr,
    kLiteRtOpCodeTflExpandDims,
    kLiteRtOpCodeTflLog,
    kLiteRtOpCodeTflSin,
    kLiteRtOpCodeTflPow,
    kLiteRtOpCodeTflFloorDiv,
    kLiteRtOpCodeTflFloorMod,
    kLiteRtOpCodeTflCos,
    kLiteRtOpCodeTflMinimum,
    kLiteRtOpCodeTflSquaredDifference,
    kLiteRtOpCodeTflRsqrt,
    kLiteRtOpCodeTflAbs,
    kLiteRtOpCodeTflLess,
    kLiteRtOpCodeTflSelect,
    kLiteRtOpCodeTflSelectV2,
    kLiteRtOpCodeTflHardSwish,
    kLiteRtOpCodeTflPrelu,
    kLiteRtOpCodeTflSqrt,
    kLiteRtOpCodeTflGreaterEqual,
    kLiteRtOpCodeTflLessEqual,
    kLiteRtOpCodeTflLogicalAnd,
    kLiteRtOpCodeTflLogicalNot,
    kLiteRtOpCodeTflL2Normalization,
    kLiteRtOpCodeTflGreater,
    kLiteRtOpCodeTflRelu0To1,
    kLiteRtOpCodeTflSquare,
};
// clang format on

class EliminateMatMulFakeQuantize : public ov::pass::MatcherPass {
 public:
  OPENVINO_MATCHER_PASS_RTTI("EliminateMatMulFakeQuantize");
  EliminateMatMulFakeQuantize() {
    namespace pattern = ov::pass::pattern;
    auto matmul_pattern = pattern::wrap_type<ov::op::v0::MatMul>(
        {pattern::any_input(), pattern::any_input()},
        pattern::consumers_count(1));
    auto fq_pattern = pattern::wrap_type<ov::op::v0::FakeQuantize>(
        {matmul_pattern, pattern::any_input(), pattern::any_input(),
         pattern::any_input(), pattern::any_input()});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
      auto pattern_map = m.get_pattern_value_map();
      auto matmul = pattern_map[matmul_pattern];
      auto fq = pattern_map[fq_pattern].get_node_shared_ptr();

      ov::copy_runtime_info(fq, matmul.get_node_shared_ptr());
      ov::replace_node(fq, matmul.get_node_shared_ptr());
      return true;
    };

    auto m = std::make_shared<pattern::Matcher>(fq_pattern,
                                                "EliminateMatMulFakeQuantize");
    register_matcher(m, callback);
  }
};

// Helper functions for attention matmul merging
namespace {

bool SoleConsumerIs(const ov::Output<ov::Node>& output,
                    const ov::Node* candidate_node) {
  auto targets = output.get_target_inputs();
  if (targets.empty()) return false;
  for (const auto& target : targets) {
    if (target.get_node() != candidate_node) return false;
  }
  return true;
}

bool MatMulAttrsOk(const std::shared_ptr<ov::Node>& node) {
  if (auto matmul = std::dynamic_pointer_cast<ov::op::v0::MatMul>(node)) {
    return matmul->get_transpose_b() && !matmul->get_transpose_a();
  }
  return false;
}

ov::Output<ov::Node> GetSliceSource(const ov::Output<ov::Node>& output) {
  if (auto slice = std::dynamic_pointer_cast<ov::op::v8::Slice>(
          output.get_node_shared_ptr())) {
    return slice->input_value(0);
  }
  return {};
}

bool SameOutput(const ov::Output<ov::Node>& a, const ov::Output<ov::Node>& b) {
  return a.get_node() == b.get_node() && a.get_index() == b.get_index();
}

}  // namespace

// Pass 1: Merge QK score matmuls
// Matches: Concat(MatMul(Q, K_cache, trans_b=T), MatMul(Q, K_cur, trans_b=T), axis=last)
// Replaces: MatMul(Q, Concat(K_cache, K_cur, axis=seq_dim_of_K), trans_b=T)
class MergeQKMatMuls : public ov::pass::MatcherPass {
 public:
  OPENVINO_MATCHER_PASS_RTTI("MergeQKMatMuls");
  MergeQKMatMuls() {
    namespace pattern = ov::pass::pattern;

    auto matmul_cache_pattern = pattern::wrap_type<ov::op::v0::MatMul>(
        {pattern::any_input(), pattern::any_input()});
    auto matmul_cur_pattern = pattern::wrap_type<ov::op::v0::MatMul>(
        {pattern::any_input(), pattern::any_input()});
    auto concat_pattern = pattern::wrap_type<ov::op::v0::Concat>(
        {matmul_cache_pattern, matmul_cur_pattern});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
      auto pattern_map = m.get_pattern_value_map();
      auto concat = std::dynamic_pointer_cast<ov::op::v0::Concat>(
          pattern_map[concat_pattern].get_node_shared_ptr());
      if (!concat || concat->get_input_size() != 2) return false;

      auto mm0 = concat->input_value(0).get_node_shared_ptr();
      auto mm1 = concat->input_value(1).get_node_shared_ptr();

      if (!MatMulAttrsOk(mm0) || !MatMulAttrsOk(mm1)) return false;

      // Both MatMuls must share the same Q (input port 0)
      auto q0 = mm0->input_value(0);
      auto q1 = mm1->input_value(0);
      if (!SameOutput(q0, q1)) return false;

      // The Concat axis must be the last dim (scores sequence axis)
      int64_t axis = concat->get_axis();
      auto score_shape = concat->get_output_partial_shape(0);
      if (score_shape.rank().is_static()) {
        int64_t rank = score_shape.rank().get_length();
        if (axis != rank - 1 && axis != -1) return false;
      }

      // Safety: each MatMul output consumed only by this Concat
      if (!SoleConsumerIs(mm0->output(0), concat.get())) return false;
      if (!SoleConsumerIs(mm1->output(0), concat.get())) return false;

      auto k_cache = mm0->input_value(1);
      auto k_cur = mm1->input_value(1);

      // K has shape [..., seq, head_dim], seq is second-to-last axis
      auto k_shape = k_cache.get_partial_shape();
      int64_t k_seq_axis = k_shape.rank().is_static()
          ? k_shape.rank().get_length() - 2 : 2;

      auto k_full = std::make_shared<ov::op::v0::Concat>(
          ov::OutputVector{k_cache, k_cur}, k_seq_axis);
      auto new_mm = std::make_shared<ov::op::v0::MatMul>(
          q0, k_full, false, true);

      k_full->set_friendly_name(mm0->get_friendly_name() + "/k_full_concat");
      new_mm->set_friendly_name(mm0->get_friendly_name() + "/qk_merged");

      ov::copy_runtime_info({mm0, mm1, concat}, {k_full, new_mm});
      ov::replace_node(concat, new_mm);
      return true;
    };

    auto m = std::make_shared<pattern::Matcher>(concat_pattern, "MergeQKMatMuls");
    register_matcher(m, callback);
  }
};

// Pass 2: Merge AV context matmuls
// Matches: Add(MatMul(A_cache, V_cache, trans_b=T), MatMul(A_cur, V_cur, trans_b=T))
// Replaces: MatMul(A_full, V_full, trans_b=T) where A_full and V_full are concatenated
class MergeAVMatMuls : public ov::pass::MatcherPass {
 public:
  OPENVINO_MATCHER_PASS_RTTI("MergeAVMatMuls");
  MergeAVMatMuls() {
    namespace pattern = ov::pass::pattern;

    auto matmul_cache_pattern = pattern::wrap_type<ov::op::v0::MatMul>(
        {pattern::any_input(), pattern::any_input()});
    auto matmul_cur_pattern = pattern::wrap_type<ov::op::v0::MatMul>(
        {pattern::any_input(), pattern::any_input()});
    auto add_pattern = pattern::wrap_type<ov::op::v1::Add>(
        {matmul_cache_pattern, matmul_cur_pattern});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
      auto pattern_map = m.get_pattern_value_map();
      auto add = std::dynamic_pointer_cast<ov::op::v1::Add>(
          pattern_map[add_pattern].get_node_shared_ptr());
      if (!add || add->get_input_size() != 2) return false;

      auto mm0 = add->input_value(0).get_node_shared_ptr();
      auto mm1 = add->input_value(1).get_node_shared_ptr();

      if (!MatMulAttrsOk(mm0) || !MatMulAttrsOk(mm1)) return false;

      // Safety: each MatMul output consumed only by this Add
      if (!SoleConsumerIs(mm0->output(0), add.get())) return false;
      if (!SoleConsumerIs(mm1->output(0), add.get())) return false;

      auto a_cache = mm0->input_value(0);
      auto v_cache = mm0->input_value(1);
      auto a_cur = mm1->input_value(0);
      auto v_cur = mm1->input_value(1);

      // Check output shape compatibility
      auto out_shape0 = mm0->get_output_partial_shape(0);
      auto out_shape1 = mm1->get_output_partial_shape(0);
      if (out_shape0.rank().is_static() && out_shape1.rank().is_static()) {
        if (out_shape0.rank().get_length() != out_shape1.rank().get_length()) {
          return false;
        }
      }

      // Build A_input - optimize if both are slices of same source
      ov::Output<ov::Node> a_input;
      std::shared_ptr<ov::Node> a_concat_node;
      auto a_cache_src = GetSliceSource(a_cache);
      auto a_cur_src = GetSliceSource(a_cur);

      if (a_cache_src.get_node() && a_cur_src.get_node() &&
          SameOutput(a_cache_src, a_cur_src)) {
        // Both slices from same source - use source directly
        a_input = a_cache_src;
      } else {
        // General case - concatenate
        auto a_shape = a_cache.get_partial_shape();
        int64_t a_seq_axis = a_shape.rank().is_static()
            ? a_shape.rank().get_length() - 1 : 3;
        a_concat_node = std::make_shared<ov::op::v0::Concat>(
            ov::OutputVector{a_cache, a_cur}, a_seq_axis);
        a_concat_node->set_friendly_name(mm0->get_friendly_name() + "/a_full_concat");
        a_input = a_concat_node->output(0);
      }

      // Build V_full
      auto v_shape = v_cache.get_partial_shape();
      int64_t v_seq_axis = v_shape.rank().is_static()
          ? v_shape.rank().get_length() - 1 : 3;
      auto v_full = std::make_shared<ov::op::v0::Concat>(
          ov::OutputVector{v_cache, v_cur}, v_seq_axis);
      v_full->set_friendly_name(mm0->get_friendly_name() + "/v_full_concat");

      // Create fused MatMul
      auto new_mm = std::make_shared<ov::op::v0::MatMul>(
          a_input, v_full, false, true);
      new_mm->set_friendly_name(mm0->get_friendly_name() + "/av_merged");

      ov::NodeVector nodes_to_copy = {mm0, mm1, add};
      ov::NodeVector new_nodes = {v_full, new_mm};
      if (a_concat_node) {
        new_nodes.insert(new_nodes.begin(), a_concat_node);
      }
      ov::copy_runtime_info(nodes_to_copy, new_nodes);
      ov::replace_node(add, new_mm);
      return true;
    };

    auto m = std::make_shared<pattern::Matcher>(add_pattern, "MergeAVMatMuls");
    register_matcher(m, callback);
  }
};

// When exporting a model via the OpenVINO NPU plugin, standard string streams
// might encounter a 32-bit std::streamsize limitation on specific platforms,
// which restricts model export capacity. This custom output stream buffer
// bypasses that limitation, enabling support for larger models.
class CustomOStreamBuf : public std::streambuf {
 public:
  CustomOStreamBuf() = default;
  std::string drain_str() { return std::move(target_); }

 protected:
  std::streamsize xsputn(const char* s, std::streamsize n) override {
    target_.append(s, n);
    return n;
  }
  int_type overflow(int_type ch) override {
    if (ch != traits_type::eof()) {
      target_.push_back(static_cast<char>(ch));
      return ch;
    }
    return traits_type::eof();
  }

 private:
  std::string target_;
};
}  // namespace

LiteRtStatus LiteRtGetCompilerPluginVersion(LiteRtApiVersion* api_version) {
  if (api_version == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  api_version->major = LITERT_API_VERSION_MAJOR;
  api_version->minor = LITERT_API_VERSION_MINOR;
  api_version->patch = LITERT_API_VERSION_PATCH;
  return kLiteRtStatusOk;
}

const char* LiteRtGetCompilerPluginSocManufacturer() {
  return kPluginManufacturer;
}

LiteRtStatus LiteRtGetCompilerPluginSupportedHardware(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtHwAccelerators* supported_hardware) {
  if (!compiler_plugin || !supported_hardware) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *supported_hardware = kLiteRtHwAcceleratorNpu;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumCompilerPluginSupportedSocModels(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtParamIndex* num_supported_soc_models) {
  if (compiler_plugin == nullptr || num_supported_soc_models == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_supported_soc_models = litert::openvino::GetNumSocModels();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompilerPluginSupportedSocModel(
    LiteRtCompilerPlugin compiler_plugin, LiteRtParamIndex soc_model_idx,
    const char** soc_model_name) {
  if (compiler_plugin == nullptr ||
      soc_model_idx >= litert::openvino::GetNumSocModels() ||
      soc_model_name == nullptr) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *soc_model_name = litert::openvino::GetSocModelName(soc_model_idx);
  return kLiteRtStatusOk;
}

// Compiled Result Definition
/// \brief Define storage of compiled result object for OV compiler plugin
struct LiteRtCompiledResultT {
  std::vector<std::string> byte_code;
  std::vector<std::string> graph_names;
};

LiteRtStatus LiteRtGetCompiledResultByteCode(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex byte_code_idx,
    const void** byte_code, size_t* byte_code_size) {
  const char* raw_data_ptr = compiled_result->byte_code[byte_code_idx].data();
  *byte_code = static_cast<void*>(const_cast<char*>(raw_data_ptr));
  *byte_code_size = compiled_result->byte_code[byte_code_idx].length();
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetCompiledResultCallInfo(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex call_idx,
    const void** call_info, size_t* call_info_size,
    LiteRtParamIndex* byte_code_idx) {
  if (call_idx >= compiled_result->graph_names.size()) {
    return kLiteRtStatusErrorIndexOOB;
  }

  auto& graph_name = compiled_result->graph_names[call_idx];
  *call_info = graph_name.data();
  *call_info_size = graph_name.size();
  *byte_code_idx = call_idx;

  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtGetNumCompiledResultCalls(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_calls) {
  *num_calls = compiled_result->graph_names.size();
  return kLiteRtStatusOk;
}

void LiteRtDestroyCompiledResult(LiteRtCompiledResult compiled_result) {
  delete compiled_result;
}

LiteRtStatus LiteRtCompiledResultNumByteCodeModules(
    LiteRtCompiledResult compiled_result, LiteRtParamIndex* num_byte_code) {
  if (!compiled_result || !num_byte_code) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  *num_byte_code = compiled_result->byte_code.size();
  return kLiteRtStatusOk;
}

// Plugin Definition
/// \brief Define Compiler plugin APIs
struct LiteRtCompilerPluginT {
  using IntelOpenVinoOptions = ::litert::intel_openvino::IntelOpenVinoOptions;

  LiteRtCompilerPluginT(LiteRtEnvironmentOptions env, LiteRtOptions options) {
    if (options == nullptr) return;
    auto cc_options = litert::Options(options, litert::OwnHandle::kNo);
    auto opaques_status = cc_options.GetOpaqueOptions();
    if (!opaques_status) return;

    auto target_opq_status = litert::FindOpaqueOptions(
        *opaques_status, LrtGetIntelOpenVinoOptionsIdentifier());
    if (target_opq_status) {
      auto payload_status = target_opq_status->GetData<const char>();
      if (payload_status) {
        LrtIntelOpenVinoOptions raw_options = nullptr;
        if (LrtCreateIntelOpenVinoOptionsFromToml(
                payload_status.Value(), &raw_options) == kLiteRtStatusOk) {
          intel_openvino_opts =
              IntelOpenVinoOptions::CreateFromOwnedHandle(raw_options);
        }
      }
    }
  }

  const ::litert::Expected<IntelOpenVinoOptions>& GetIntelOpenVinoOptions()
      const {
    return intel_openvino_opts;
  }

  const ::litert::Expected<litert::OpaqueOptions>& GetOpaqueOptions() const {
    return opq;
  }

 private:
  litert::Expected<litert::Options> compiler_opts =
      litert::Error(kLiteRtStatusErrorInvalidArgument, "Null options");
  litert::Expected<litert::OpaqueOptions> opq =
      litert::Error(kLiteRtStatusErrorInvalidArgument, "Null opaque options");
  litert::Expected<IntelOpenVinoOptions> intel_openvino_opts = litert::Error(
      kLiteRtStatusErrorInvalidArgument, "Null Intel OpenVINO options");
};

LiteRtStatus LiteRtCreateCompilerPlugin(LiteRtCompilerPlugin* compiler_plugin,
                                        LiteRtEnvironmentOptions env,
                                        LiteRtOptions options) {
  LiteRtSetMinLoggerSeverity(LiteRtGetDefaultLogger(), LITERT_INFO);
  LiteRtPropagateMinLoggerSeverity(env);
  auto* plugin = new LiteRtCompilerPluginT(env, options);
  *compiler_plugin = plugin;
  return kLiteRtStatusOk;
}

void LiteRtDestroyCompilerPlugin(LiteRtCompilerPlugin compiler_plugin) {
  delete compiler_plugin;
}

bool IsOpSupported(const ::litert::Op& op) {
  for (const auto& supportedOp : kSupportedOps) {
    if (op.Code() == supportedOp) return true;
  }
  return false;
}

#ifdef __cplusplus
extern "C" {
#endif
LiteRtStatus LiteRtCompilerPluginPartition(LiteRtCompilerPlugin compiler_plugin,
                                           const char* soc_model,
                                           LiteRtSubgraph subgraph,
                                           LiteRtOpList selected_ops) {
  ::litert::Subgraph graph(subgraph);

  // Check if any subgraph input has dims.size() >= 6.
  auto subgraph_inputs = graph.Inputs();
  for (size_t i = 0; i < subgraph_inputs.size(); ++i) {
    auto ranked_type = subgraph_inputs[i].RankedTensorType();
    if (ranked_type.HasValue()) {
      auto dims = ranked_type.Value().Layout().Dimensions();
      if (dims.size() >= 6) {
        LITERT_LOG(LITERT_WARNING,
                   "Model not supported: subgraph input %zu has %zu dimensions "
                   "(>= 6), skipping partitioning.",
                   i, dims.size());
        return kLiteRtStatusErrorUnsupported;
      }
    }
  }

  // TODO(rjasuja): Enhance implementation for Partition() call
  for (const auto& op : graph.Ops()) {
    if (!IsOpSupported(op)) {
      LITERT_LOG(LITERT_INFO, "op type %d is not supported", op.Code());
      continue;
    }
    LITERT_RETURN_IF_ERROR(LiteRtPushOp(selected_ops, op.Get(), 0));
  }

  return kLiteRtStatusOk;
}
#ifdef __cplusplus
} /* end extern "C" */
#endif

LiteRtStatus LiteRtCompilerPluginCompile(
    LiteRtCompilerPlugin compiler_plugin, const char* soc_model,
    LiteRtModel partitions, LiteRtCompiledResult* compiled_result) {
  try {
    auto model = litert::ExtendedModel::CreateFromNonOwnedHandle(partitions);
    const auto num_partitions = model.NumSubgraphs();

    // Configure device and OpenVINO settings from Intel OpenVINO options

    std::string device = "NPU";  // Default device
    ov::AnyMap configs_map;
    bool eliminate_fq = false;
    bool merge_attn_matmuls = false;

    if (compiler_plugin->GetIntelOpenVinoOptions().HasValue()) {
      const auto& intel_opts =
          compiler_plugin->GetIntelOpenVinoOptions().Value();

      // Configure device type
      auto device_type = intel_opts.GetDeviceType();
      switch (device_type) {
        case kLiteRtIntelOpenVinoDeviceTypeCPU:
          device = "CPU";
          break;
        case kLiteRtIntelOpenVinoDeviceTypeGPU:
          device = "GPU";
          break;
        case kLiteRtIntelOpenVinoDeviceTypeNPU:
          device = "NPU";
          break;
        case kLiteRtIntelOpenVinoDeviceTypeAUTO:
          device = "AUTO";
          break;
      }

      LITERT_LOG(LITERT_INFO, "Using Intel OpenVINO device: %s",
                 device.c_str());

      auto performance_mode = intel_opts.GetPerformanceMode();

      // Add custom configuration options
      int num_custom_options = intel_opts.GetNumConfigsMapOptions();
      for (int i = 0; i < num_custom_options; ++i) {
        auto [key, value] = intel_opts.GetConfigsMapOption(i);
        if (!key.empty()) {  // Valid config option
          if (key == "optimize_fq_after_matmul") {
            LITERT_LOG(LITERT_INFO,
                       "Custom config: optimize_fq_after_matmul = %s",
                       value.c_str());
            eliminate_fq = (value == "true");
            continue;  // This is a special case handled separately, so skip adding to configs_map
          }
          if (key == "merge_attn_matmuls") {
            LITERT_LOG(LITERT_INFO,
                       "Custom config: merge_attn_matmuls = %s",
                       value.c_str());
            merge_attn_matmuls = (value == "true");
            continue;  // This is a special case handled separately, so skip adding to configs_map
          }
          configs_map[key] = value;
          LITERT_LOG(LITERT_INFO, "Custom config: %s = %s", key.c_str(),
                     value.c_str());
        }
      }

      // Configure performance mode (can be overridden by custom options)
      switch (performance_mode) {
        case kLiteRtIntelOpenVinoPerformanceModeLatency:
          if (configs_map.find(ov::hint::performance_mode.name()) ==
              configs_map.end()) {
            configs_map[ov::hint::performance_mode.name()] =
                ov::hint::PerformanceMode::LATENCY;
            LITERT_LOG(LITERT_INFO, "Performance mode: LATENCY");
          }
          break;
        case kLiteRtIntelOpenVinoPerformanceModeThroughput:
          if (configs_map.find(ov::hint::performance_mode.name()) ==
              configs_map.end()) {
            configs_map[ov::hint::performance_mode.name()] =
                ov::hint::PerformanceMode::THROUGHPUT;
            LITERT_LOG(LITERT_INFO, "Performance mode: THROUGHPUT");
          }
          break;
        case kLiteRtIntelOpenVinoPerformanceModeCumulativeThroughput:
          if (configs_map.find(ov::hint::performance_mode.name()) ==
              configs_map.end()) {
            configs_map[ov::hint::performance_mode.name()] =
                ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT;
            LITERT_LOG(LITERT_INFO, "Performance mode: CUMULATIVE_THROUGHPUT");
          }
          break;
      }
    } else {
      // Default configuration if no options provided
      configs_map[ov::hint::performance_mode.name()] =
          ov::hint::PerformanceMode::LATENCY;
      LITERT_LOG(LITERT_INFO, "Using default configuration (LATENCY mode)");
    }

    LITERT_RETURN_IF_ERROR(
        litert::openvino::ConfigureCompilationParams(soc_model, configs_map));

    auto result = std::make_unique<LiteRtCompiledResultT>();
    result->byte_code.resize(num_partitions);
    result->graph_names.resize(num_partitions);
    auto tflite_fe =
        std::make_shared<ov::frontend::tensorflow_lite::FrontEnd>();
    LITERT_LOG(LITERT_INFO, "TensorFlow Lite FrontEnd initialized");

    ov::Core core;
    LITERT_LOG(LITERT_INFO, "OpenVINO Core initialized with device: %s",
               device.c_str());
    for (int partition_idx = 0; partition_idx < num_partitions;
         ++partition_idx) {
      auto graph_name = absl::StrFormat("Partition_%d", partition_idx);
      litert::Expected<litert::Subgraph> expected_subgraph =
          model.Subgraph(partition_idx);
      if (expected_subgraph.HasValue()) {
        LITERT_LOG(LITERT_INFO, "Compiling %s with %zu ops",
                   graph_name.c_str(),
                   expected_subgraph.Value().Ops().size());
        std::shared_ptr<ov::frontend::tensorflow_lite::GraphIterator>
            graph_delegate =
                std::make_shared<litert::openvino::GraphIteratorDelegate>(
                    &expected_subgraph.Value());
        LITERT_LOG(LITERT_INFO, "Graph delegate created");
        auto input_model = tflite_fe->load(graph_delegate);
        LITERT_LOG(LITERT_INFO, "Model loaded");
        auto ov_model = tflite_fe->convert(input_model);

        if (eliminate_fq || merge_attn_matmuls) {
          ov::pass::Manager pass_manager;
          if (eliminate_fq) {
            // Eliminate FakeQuantize nodes after MatMul operations.
            pass_manager.register_pass<EliminateMatMulFakeQuantize>();
          }
          if (merge_attn_matmuls) {
            // Merge split attention MatMuls for Gemma4 decode optimization.
            pass_manager.register_pass<MergeQKMatMuls>();
            pass_manager.register_pass<MergeAVMatMuls>();
          }
          pass_manager.run_passes(ov_model);
        }

        // Note: CommonOptimizations (including EliminateDuplicateFakeQuantize)
        // should be applied here before compilation, but the transformation headers
        // are not installed in the OpenVINO runtime package. The NPU plugin does
        // not apply these optimizations by default. To enable FakeQuantize
        // elimination, the NPU plugin would need to be modified or a preprocessing
        // step added.

        // Use device and configs_map from Intel OpenVINO options
        auto compiled_model = core.compile_model(ov_model, device, configs_map);

        CustomOStreamBuf obuf;
        std::ostream oss(&obuf);
        compiled_model.export_model(oss);
        LITERT_LOG(LITERT_INFO, "Model export done");
        result->byte_code[partition_idx] = obuf.drain_str();

        result->graph_names.emplace_back(graph_name);
      } else {
        LITERT_LOG(LITERT_INFO, "Failed to retrieve Subgraph");
        return kLiteRtStatusErrorCompilation;
      }
    }
    *compiled_result = result.release();
    // TODO: Add support for caching
    return kLiteRtStatusOk;
  } catch (const ov::Exception& e) {
    LITERT_LOG(LITERT_ERROR, "Exception in compilation: %s", e.what());
    return kLiteRtStatusErrorCompilation;
  }
}

LiteRtStatus LiteRtCompilerPluginRegisterAllTransformations(
    LiteRtCompilerPlugin compiler_plugin,
    LiteRtTransformation** transformations, LiteRtParamIndex* num_patterns) {
  *num_patterns = 0;
  return kLiteRtStatusOk;
}

LiteRtStatus LiteRtCompilerPluginCheckCompilerCompatibility(
    LiteRtApiVersion api_version, LiteRtCompilerPlugin compiler_plugin,
    LiteRtEnvironmentOptions env, LiteRtOptions options,
    const char* soc_model_name) {
  return kLiteRtStatusOk;
}
