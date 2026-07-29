// Copyright 2026 Google LLC.
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

#include "litert/vendors/intel_openvino/compiler/npu_optimizer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/tensor.hpp"
#include <gtest/gtest.h>

namespace litert {
namespace openvino {
namespace {

// Builds the "split-cache" attention pattern emitted by the LiteRT generative
// converter (gemma4 prefill/decode), matching the layout observed in the
// model:
//   K_cache: [B,H,S_past,D]   K_slice: [B,H,1,D]
//   V_cache: [B,H,D,S_past]   V_slice: [B,H,D,1]   (transposed, adj_y=true)
//   scores = Concat[ MatMul(Q,K_cache,T), MatMul(Q,K_slice,T) ] + mask
//   probs  = Softmax(scores)
//   out    = Add[ MatMul(Slice(probs,past),V_cache,T),
//                 MatMul(Slice(probs,cur),V_slice,T) ]
// |s_past| is the cached sequence length and |s_cur| is the current chunk size
// (the K/V slice length): decode uses s_cur=1, prefill uses s_cur=chunk (e.g.
// 128). The query length equals s_cur in both cases.
std::shared_ptr<ov::Model> BuildSplitCacheAttention(int64_t s_past = 8,
                                                    int64_t s_cur = 1) {
  using ov::op::v0::Concat;
  using ov::op::v0::Constant;
  using ov::op::v0::MatMul;
  using ov::op::v0::Parameter;
  using ov::op::v1::Add;
  using ov::op::v8::Slice;
  using ov::op::v8::Softmax;

  constexpr int64_t kBatch = 1;
  constexpr int64_t kHeads = 8;
  constexpr int64_t kDim = 256;
  const int64_t lq = s_cur;
  const int64_t s_kv = s_past + s_cur;
  const auto f = ov::element::f32;

  auto q = std::make_shared<Parameter>(
      f, ov::Shape{static_cast<size_t>(kBatch), static_cast<size_t>(kHeads),
                   static_cast<size_t>(lq), static_cast<size_t>(kDim)});
  auto k_cache = std::make_shared<Parameter>(
      f, ov::Shape{static_cast<size_t>(kBatch), static_cast<size_t>(kHeads),
                   static_cast<size_t>(s_past), static_cast<size_t>(kDim)});
  auto k_slice = std::make_shared<Parameter>(
      f, ov::Shape{static_cast<size_t>(kBatch), static_cast<size_t>(kHeads),
                   static_cast<size_t>(s_cur), static_cast<size_t>(kDim)});
  auto v_cache = std::make_shared<Parameter>(
      f, ov::Shape{static_cast<size_t>(kBatch), static_cast<size_t>(kHeads),
                   static_cast<size_t>(kDim), static_cast<size_t>(s_past)});
  auto v_slice = std::make_shared<Parameter>(
      f, ov::Shape{static_cast<size_t>(kBatch), static_cast<size_t>(kHeads),
                   static_cast<size_t>(kDim), static_cast<size_t>(s_cur)});
  auto mask = std::make_shared<Parameter>(
      f, ov::Shape{static_cast<size_t>(kBatch), 1, static_cast<size_t>(lq),
                   static_cast<size_t>(s_kv)});

  // scores = Q * K^T (transpose_b), concatenated over the sequence axis.
  auto qk_cache = std::make_shared<MatMul>(q, k_cache, /*transpose_a=*/false,
                                           /*transpose_b=*/true);
  auto qk_slice = std::make_shared<MatMul>(q, k_slice, false, true);
  auto scores = std::make_shared<Concat>(ov::OutputVector{qk_cache, qk_slice},
                                         /*axis=*/-1);
  auto masked = std::make_shared<Add>(scores, mask);
  auto probs = std::make_shared<Softmax>(masked, /*axis=*/-1);

  // Split probs back into past / current, then probs * V^T (transpose_b).
  auto start0 = Constant::create(ov::element::i64, ov::Shape{1}, {0});
  auto stop0 = Constant::create(ov::element::i64, ov::Shape{1}, {s_past});
  auto step = Constant::create(ov::element::i64, ov::Shape{1}, {1});
  auto axis = Constant::create(ov::element::i64, ov::Shape{1}, {-1});
  auto slice_past = std::make_shared<Slice>(probs, start0, stop0, step, axis);

  auto start1 = Constant::create(ov::element::i64, ov::Shape{1}, {s_past});
  auto stop1 = Constant::create(ov::element::i64, ov::Shape{1}, {s_kv});
  auto slice_cur = std::make_shared<Slice>(probs, start1, stop1, step, axis);

  auto pv_cache = std::make_shared<MatMul>(slice_past, v_cache, false, true);
  auto pv_slice = std::make_shared<MatMul>(slice_cur, v_slice, false, true);
  auto out = std::make_shared<Add>(pv_cache, pv_slice);

  auto result = std::make_shared<ov::op::v0::Result>(out);
  return std::make_shared<ov::Model>(
      ov::ResultVector{result},
      ov::ParameterVector{q, k_cache, k_slice, v_cache, v_slice, mask},
      "split_cache_attention");
}

// Builds the "head-folded" split-cache pattern emitted for Gemma global
// (multi-query) layers: the query heads are folded into the sequence axis
// before the QK MatMul so a single-KV-head cache can be matmul'd without an
// explicit broadcast, and the per-query mask is tiled H-fold to match:
//
//   q_folded  = Reshape(Q[1,H,S,D] -> [1,1,H*S,D])
//   K_cache: [1,1,s_past,D]   K_slice: [1,1,S,D]      (single KV head)
//   V_cache: [1,1,D,s_past]   V_slice: [1,1,D,S]      (transposed, adj_y=true)
//   mask    = Concat([base[1,1,S,s_kv]] * H, axis=2)  -> [1,1,H*S,s_kv]
//   scores  = Concat[ MatMul(q_folded,K_cache,T), MatMul(q_folded,K_slice,T) ]
//   probs   = Softmax(scores + mask)
//   out     = Add[ MatMul(Slice(probs,past),V_cache,T),
//                  MatMul(Slice(probs,cur), V_slice,T) ]   -> [1,1,H*S,D]
//
// The folded output [1,1,H*S,D] is, element for element, the head-major
// flattening of a [1,H,S,D] tensor, so it can be compared directly against the
// head-preserved SDPA output.
std::shared_ptr<ov::Model> BuildFoldedMqaAttention(int64_t heads, int64_t s_cur,
                                                   int64_t s_past,
                                                   int64_t dim) {
  using ov::op::v0::Concat;
  using ov::op::v0::Constant;
  using ov::op::v0::MatMul;
  using ov::op::v0::Parameter;
  using ov::op::v1::Add;
  using ov::op::v1::Reshape;
  using ov::op::v8::Slice;
  using ov::op::v8::Softmax;

  const int64_t s_kv = s_past + s_cur;
  const int64_t folded_q = heads * s_cur;
  const auto f = ov::element::f32;
  auto S = [](int64_t v) { return static_cast<size_t>(v); };

  auto q = std::make_shared<Parameter>(
      f, ov::Shape{1, S(heads), S(s_cur), S(dim)});
  auto q_shape = Constant::create<int64_t>(ov::element::i64, ov::Shape{4},
                                           {1, 1, folded_q, dim});
  auto q_folded = std::make_shared<Reshape>(q, q_shape, /*special_zero=*/false);

  auto k_cache =
      std::make_shared<Parameter>(f, ov::Shape{1, 1, S(s_past), S(dim)});
  auto k_slice =
      std::make_shared<Parameter>(f, ov::Shape{1, 1, S(s_cur), S(dim)});
  auto v_cache =
      std::make_shared<Parameter>(f, ov::Shape{1, 1, S(dim), S(s_past)});
  auto v_slice =
      std::make_shared<Parameter>(f, ov::Shape{1, 1, S(dim), S(s_cur)});
  auto mask_base =
      std::make_shared<Parameter>(f, ov::Shape{1, 1, S(s_cur), S(s_kv)});

  // Tile the per-query mask H-fold along the query axis (axis=2).
  ov::OutputVector mask_copies(static_cast<size_t>(heads), mask_base);
  auto mask = std::make_shared<Concat>(mask_copies, /*axis=*/2);

  auto qk_cache = std::make_shared<MatMul>(q_folded, k_cache,
                                           /*transpose_a=*/false,
                                           /*transpose_b=*/true);
  auto qk_slice = std::make_shared<MatMul>(q_folded, k_slice, false, true);
  auto scores = std::make_shared<Concat>(ov::OutputVector{qk_cache, qk_slice},
                                         /*axis=*/-1);
  auto masked = std::make_shared<Add>(scores, mask);
  auto probs = std::make_shared<Softmax>(masked, /*axis=*/-1);

  auto start0 = Constant::create(ov::element::i64, ov::Shape{1}, {0});
  auto stop0 = Constant::create(ov::element::i64, ov::Shape{1}, {s_past});
  auto step = Constant::create(ov::element::i64, ov::Shape{1}, {1});
  auto axis = Constant::create(ov::element::i64, ov::Shape{1}, {-1});
  auto slice_past = std::make_shared<Slice>(probs, start0, stop0, step, axis);

  auto start1 = Constant::create(ov::element::i64, ov::Shape{1}, {s_past});
  auto stop1 = Constant::create(ov::element::i64, ov::Shape{1}, {s_kv});
  auto slice_cur = std::make_shared<Slice>(probs, start1, stop1, step, axis);

  auto pv_cache = std::make_shared<MatMul>(slice_past, v_cache, false, true);
  auto pv_slice = std::make_shared<MatMul>(slice_cur, v_slice, false, true);
  auto out = std::make_shared<Add>(pv_cache, pv_slice);

  auto result = std::make_shared<ov::op::v0::Result>(out);
  return std::make_shared<ov::Model>(
      ov::ResultVector{result},
      ov::ParameterVector{q, k_cache, k_slice, v_cache, v_slice, mask_base},
      "folded_mqa_attention");
}

template <typename T>
size_t CountOps(const std::shared_ptr<ov::Model>& model) {
  size_t n = 0;
  for (const auto& node : model->get_ops()) {
    if (std::dynamic_pointer_cast<T>(node)) {
      ++n;
    }
  }
  return n;
}

std::shared_ptr<ov::op::v13::ScaledDotProductAttention> FindSdpa(
    const std::shared_ptr<ov::Model>& model) {
  for (const auto& node : model->get_ops()) {
    if (auto sdpa =
            std::dynamic_pointer_cast<ov::op::v13::ScaledDotProductAttention>(
                node)) {
      return sdpa;
    }
  }
  return nullptr;
}

// Fills |tensor| with deterministic pseudo-random values in [-1, 1).
void FillRandom(ov::Tensor tensor, uint32_t seed) {
  auto* data = tensor.data<float>();
  // Simple LCG so the test is self-contained and reproducible.
  uint64_t state = seed * 2654435761u + 1u;
  for (size_t i = 0; i < tensor.get_size(); ++i) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    data[i] = static_cast<float>((state >> 40) & 0xFFFF) / 32768.0f - 1.0f;
  }
}

// Negative: when the K_cache (or V_cache) tensor is consumed by something
// outside this attention block — typical of layers that share a single KV
// cache across heads/layers — fusing in place would rewrite what the other
// consumer sees. The pass must skip such blocks.
TEST(FuseSplitAttentionToSDPATest, DoesNotFuseSharedKvCache) {
  auto model = BuildSplitCacheAttention(/*s_past=*/8, /*s_cur=*/8);
  // BuildSplitCacheAttention orders parameters as
  //   {q, k_cache, k_slice, v_cache, v_slice, mask}.
  // Add an extra Result that consumes k_cache directly, so K_cache now feeds
  // two inputs (the qk_cache MatMul and this Result) — HasSingleConsumer
  // returns false and the matcher must bail out.
  auto k_cache = model->get_parameters()[1];
  model->add_results({std::make_shared<ov::op::v0::Result>(k_cache)});

  NpuOptimizer()
      .SetCastIntegerSignToFloat(false)
      .SetFuseSplitAttentionToSDPA(true)
      .Run(model);

  EXPECT_EQ(CountOps<ov::op::v13::ScaledDotProductAttention>(model), 0u);
  EXPECT_EQ(CountOps<ov::op::v0::MatMul>(model), 4u);
}

TEST(FuseSplitAttentionToSDPATest, FusesSplitCachePattern) {
  auto model = BuildSplitCacheAttention(/*s_past=*/8, /*s_cur=*/8);

  // Precondition: 4 MatMuls, 1 Softmax, no SDPA.
  EXPECT_EQ(CountOps<ov::op::v0::MatMul>(model), 4u);
  EXPECT_EQ(CountOps<ov::op::v8::Softmax>(model), 1u);
  EXPECT_EQ(CountOps<ov::op::v13::ScaledDotProductAttention>(model), 0u);

  NpuOptimizer()
      .SetCastIntegerSignToFloat(false)
      .SetFuseSplitAttentionToSDPA(true)
      .Run(model);

  // Postcondition: the four attention MatMuls and the Softmax are gone,
  // replaced by exactly one ScaledDotProductAttention op.
  EXPECT_EQ(CountOps<ov::op::v13::ScaledDotProductAttention>(model), 1u);
  EXPECT_EQ(CountOps<ov::op::v8::Softmax>(model), 0u);
  EXPECT_EQ(CountOps<ov::op::v0::MatMul>(model), 0u);
}

// End-to-end numerical check: the fused SDPA must produce the same output as
// the original split-cache graph for the same inputs. Uses a decode shape
// (s_past=7, s_cur=1) so the V_cache [B,H,D,S] / V_slice [B,H,D,1] transpose
// path is exercised.
TEST(FuseSplitAttentionToSDPATest, NumericallyMatchesSplitCache) {
  auto reference = BuildSplitCacheAttention(/*s_past=*/7, /*s_cur=*/1);
  // Deep-copy before mutating so we can run both versions on identical inputs.
  auto fused = reference->clone();

  NpuOptimizer()
      .SetCastIntegerSignToFloat(false)
      .SetFuseSplitAttentionToSDPA(true)
      .Run(fused);
  ASSERT_NE(FindSdpa(fused), nullptr) << "fusion did not fire";

  ov::Core core;
  auto ref_compiled = core.compile_model(reference, "CPU");
  auto fused_compiled = core.compile_model(fused, "CPU");
  auto ref_req = ref_compiled.create_infer_request();
  auto fused_req = fused_compiled.create_infer_request();

  // Same input tensors fed to both. Inputs are ordered as constructed:
  // {q, k_cache, k_slice, v_cache, v_slice, mask}.
  const size_t num_inputs = reference->inputs().size();
  ASSERT_EQ(num_inputs, fused->inputs().size());
  std::vector<ov::Tensor> inputs;
  for (size_t i = 0; i < num_inputs; ++i) {
    const auto& port = reference->input(i);
    ov::Tensor t(port.get_element_type(), port.get_shape());
    FillRandom(t, static_cast<uint32_t>(i + 1));
    inputs.push_back(t);
    ref_req.set_input_tensor(i, t);
    fused_req.set_input_tensor(i, t);
  }

  ref_req.infer();
  fused_req.infer();

  auto ref_out = ref_req.get_output_tensor(0);
  auto fused_out = fused_req.get_output_tensor(0);
  ASSERT_EQ(ref_out.get_size(), fused_out.get_size());
  const auto* a = ref_out.data<float>();
  const auto* b = fused_out.data<float>();
  float max_abs_diff = 0.0f;
  for (size_t i = 0; i < ref_out.get_size(); ++i) {
    max_abs_diff = std::max(max_abs_diff, std::abs(a[i] - b[i]));
    ASSERT_FALSE(std::isnan(b[i])) << "fused output has NaN at " << i;
  }
  EXPECT_LT(max_abs_diff, 1e-4f)
      << "fused output diverges from split-cache reference";
}

// With preserve_q_heads enabled, fusing the head-folded MQA pattern must undo
// the query fold: the resulting SDPA keeps the query head dimension
// ([1,H,S,D]) and consumes the pre-fold query and per-query (un-tiled) mask.
TEST(FuseSplitAttentionToSDPATest, PreserveQHeadsKeepsHeadDim) {
  constexpr int64_t kHeads = 16;
  constexpr int64_t kSeq = 8;
  constexpr int64_t kPast = 8;
  constexpr int64_t kDim = 32;
  auto model = BuildFoldedMqaAttention(kHeads, kSeq, kPast, kDim);

  // Sanity: the fold Reshape and the H-fold mask Concat are present up front.
  EXPECT_EQ(CountOps<ov::op::v1::Reshape>(model), 1u);
  EXPECT_EQ(CountOps<ov::op::v0::Concat>(model), 2u);  // mask tile + scores

  NpuOptimizer()
      .SetCastIntegerSignToFloat(false)
      .SetFuseSplitAttentionToSDPA(true)
      .SetSdpaPreserveQHeads(true)
      .Run(model);

  auto sdpa = FindSdpa(model);
  ASSERT_NE(sdpa, nullptr) << "fusion did not fire";

  // Query fed to SDPA is the pre-fold [1, H, S, D].
  const auto q_ps = sdpa->get_input_partial_shape(0);
  ASSERT_TRUE(q_ps.rank().is_static() && q_ps.rank().get_length() == 4);
  EXPECT_EQ(q_ps[1].get_length(), kHeads);
  EXPECT_EQ(q_ps[2].get_length(), kSeq);
  EXPECT_EQ(q_ps[3].get_length(), kDim);

  // SDPA output is head-preserved [1, H, S, D].
  const auto out_ps = sdpa->get_output_partial_shape(0);
  ASSERT_TRUE(out_ps.rank().is_static() && out_ps.rank().get_length() == 4);
  EXPECT_EQ(out_ps[1].get_length(), kHeads);
  EXPECT_EQ(out_ps[2].get_length(), kSeq);
  EXPECT_EQ(out_ps[3].get_length(), kDim);

  // The head-fold Reshape and the mask-tile Concat were bypassed: only the
  // K/V concats built by the fusion remain (2), the mask tile is gone.
  EXPECT_EQ(CountOps<ov::op::v13::ScaledDotProductAttention>(model), 1u);
  EXPECT_EQ(CountOps<ov::op::v8::Softmax>(model), 0u);
  EXPECT_EQ(CountOps<ov::op::v0::MatMul>(model), 0u);

  // The fold-bypass is the point of this pass: SDPA's query wires straight to
  // the pre-fold tensor, leaving the fold Reshape with no consumer reachable
  // from the graph outputs. It must therefore be gone (there is no explicit DCE
  // pass in NpuOptimizer::Run, so this verifies the bypass produced a graph
  // whose result-reachable op set has zero Reshapes -- transpose -> SDPA with
  // no head fold/unfold round-trip, matching the layer_005_correct reference).
  EXPECT_EQ(CountOps<ov::op::v1::Reshape>(model), 0u);
}

// Numerical check: the head-preserved SDPA must reproduce the folded reference
// output exactly. The folded output [1,1,H*S,D] is the head-major flattening of
// [1,H,S,D], so the two output buffers compare element-for-element in order.
TEST(FuseSplitAttentionToSDPATest, PreserveQHeadsNumericallyMatchesFolded) {
  constexpr int64_t kHeads = 16;
  constexpr int64_t kSeq = 8;
  constexpr int64_t kPast = 8;
  constexpr int64_t kDim = 32;
  auto reference = BuildFoldedMqaAttention(kHeads, kSeq, kPast, kDim);
  auto fused = reference->clone();

  NpuOptimizer()
      .SetCastIntegerSignToFloat(false)
      .SetFuseSplitAttentionToSDPA(true)
      .SetSdpaPreserveQHeads(true)
      .Run(fused);
  ASSERT_NE(FindSdpa(fused), nullptr) << "fusion did not fire";

  ov::Core core;
  auto ref_compiled = core.compile_model(reference, "CPU");
  auto fused_compiled = core.compile_model(fused, "CPU");
  auto ref_req = ref_compiled.create_infer_request();
  auto fused_req = fused_compiled.create_infer_request();

  // Inputs ordered as constructed: {q, k_cache, k_slice, v_cache, v_slice,
  // mask_base}. Both graphs share the same parameter set.
  const size_t num_inputs = reference->inputs().size();
  ASSERT_EQ(num_inputs, fused->inputs().size());
  for (size_t i = 0; i < num_inputs; ++i) {
    const auto& port = reference->input(i);
    ov::Tensor t(port.get_element_type(), port.get_shape());
    FillRandom(t, static_cast<uint32_t>(i + 1));
    ref_req.set_input_tensor(i, t);
    fused_req.set_input_tensor(i, t);
  }

  ref_req.infer();
  fused_req.infer();

  auto ref_out = ref_req.get_output_tensor(0);   // [1,1,H*S,D]
  auto fused_out = fused_req.get_output_tensor(0);  // [1,H,S,D]
  ASSERT_EQ(ref_out.get_size(), fused_out.get_size());
  const auto* a = ref_out.data<float>();
  const auto* b = fused_out.data<float>();
  float max_abs_diff = 0.0f;
  for (size_t i = 0; i < ref_out.get_size(); ++i) {
    max_abs_diff = std::max(max_abs_diff, std::abs(a[i] - b[i]));
    ASSERT_FALSE(std::isnan(b[i])) << "fused output has NaN at " << i;
  }
  EXPECT_LT(max_abs_diff, 1e-4f)
      << "head-preserved SDPA diverges from folded reference";
}

}  // namespace
}  // namespace openvino
}  // namespace litert
