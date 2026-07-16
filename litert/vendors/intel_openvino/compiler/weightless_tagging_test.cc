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

#include "litert/vendors/intel_openvino/compiler/weightless_tagging.h"

#include <cstddef>
#include <memory>

#include "openvino/core/model.hpp"
#include "openvino/frontend/tensorflow_lite/frontend.hpp"
#include "openvino/op/constant.hpp"
#include <gtest/gtest.h>
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/test/load_test_model.h"
#include "litert/vendors/intel_openvino/compiler/graph_iterator.h"
#include "litert/vendors/intel_openvino/compiler/weight_bank.h"
#include "litert/vendors/intel_openvino/compiler/weightless_caching_attributes.hpp"

namespace litert {
namespace openvino {
namespace {

// Builds a finalized bank from |model| and converts subgraph 0 through the
// same frontend path the plugin uses, returning the OpenVINO model.
std::shared_ptr<ov::Model> BuildBankAndModel(litert::compiler::Model& model,
                                             const LiteRtCompilerContext* ctx,
                                             WeightBank& bank) {
  for (size_t s = 0; s < model.NumSubgraphs(); ++s) {
    auto graph = model.Subgraph(s);
    if (graph.HasValue()) bank.AddSubgraph(graph.Value());
  }
  bank.Finalize();

  auto fe = std::make_shared<ov::frontend::tensorflow_lite::FrontEnd>();
  auto subgraph = model.Subgraph(0);
  std::shared_ptr<ov::frontend::tensorflow_lite::GraphIterator> delegate =
      std::make_shared<litert::openvino::GraphIteratorDelegate>(
          ctx, &subgraph.Value());
  return fe->convert(fe->load(delegate));
}

// Returns the WeightlessCacheAttribute on |node|, or nullptr if absent.
const ov::WeightlessCacheAttribute* FindAttr(
    const std::shared_ptr<ov::Node>& node) {
  auto& rt_info = node->get_rt_info();
  auto it = rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static());
  if (it == rt_info.end()) return nullptr;
  return &it->second.as<ov::WeightlessCacheAttribute>();
}

// Minimum size a weight Constant must exceed to be tagged weightless (mirrors
// kMinWeightlessTagBytes in weightless_tagging.cc; small control constants stay
// baked).
constexpr size_t kMinWeightlessTagBytes = 256;

// Every eligible weight constant (bank-known and above the size threshold) is
// tagged with an attribute whose offset and size match the bank, every tagged
// node's friendly_name resolves in the bank, and the returned count equals the
// number of eligible constants -- so the test is correct regardless of the
// fixture's weight sizes.
TEST(WeightlessTaggingTest, TagsWeightConstants) {
  auto cc_model = testing::LoadTestFileModel("multi_subgraph.tflite");
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  litert::compiler::Model model(ctx, cc_model.Get());
  WeightBank bank;
  auto ov_model = BuildBankAndModel(model, ctx, bank);

  // Count eligible weight constants BEFORE tagging: bank-known and larger than
  // the threshold.
  size_t eligible = 0;
  for (const auto& node : ov_model->get_ops()) {
    auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
    if (!constant) continue;
    if (bank.OffsetOfName(constant->get_friendly_name()).has_value() &&
        constant->get_byte_size() > kMinWeightlessTagBytes) {
      ++eligible;
    }
  }

  const size_t tagged = TagWeightlessConstants(ov_model, bank);
  EXPECT_EQ(tagged, eligible);

  size_t observed_tagged = 0;
  for (const auto& node : ov_model->get_ops()) {
    auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
    if (!constant) continue;
    const auto offset = bank.OffsetOfName(constant->get_friendly_name());
    const auto* attr = FindAttr(node);
    if (offset.has_value() &&
        constant->get_byte_size() > kMinWeightlessTagBytes) {
      // Eligible weights are tagged with their bank offset and native size.
      ASSERT_NE(attr, nullptr);
      EXPECT_EQ(attr->bin_offset, *offset);
      EXPECT_EQ(attr->original_size, constant->get_byte_size());
      ++observed_tagged;
    } else {
      // Sub-threshold control constants and frontend-synthesized constants the
      // bank never saw stay baked (untagged).
      EXPECT_EQ(attr, nullptr);
    }
  }
  EXPECT_EQ(observed_tagged, tagged);
}

// Tagging against an empty bank tags nothing and leaves all constants untouched.
TEST(WeightlessTaggingTest, EmptyBankTagsNothing) {
  auto cc_model = testing::LoadTestFileModel("multi_subgraph.tflite");
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  litert::compiler::Model model(ctx, cc_model.Get());

  // Convert without populating the bank.
  auto fe = std::make_shared<ov::frontend::tensorflow_lite::FrontEnd>();
  auto subgraph = model.Subgraph(0);
  std::shared_ptr<ov::frontend::tensorflow_lite::GraphIterator> delegate =
      std::make_shared<litert::openvino::GraphIteratorDelegate>(
          ctx, &subgraph.Value());
  auto ov_model = fe->convert(fe->load(delegate));

  WeightBank empty_bank;
  empty_bank.Finalize();
  EXPECT_EQ(TagWeightlessConstants(ov_model, empty_bank), 0u);
  for (const auto& node : ov_model->get_ops()) {
    EXPECT_EQ(FindAttr(node), nullptr);
  }
}

}  // namespace
}  // namespace openvino
}  // namespace litert
