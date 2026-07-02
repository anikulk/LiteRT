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

#include "litert/vendors/intel_openvino/compiler/weight_bank.h"

#include <algorithm>
#include <cstddef>
#include <vector>

#include <gtest/gtest.h>
#include "litert/c/internal/litert_compiler_context.h"
#include "litert/compiler/cc/litert_model.h"
#include "litert/test/load_test_model.h"

namespace litert {
namespace openvino {
namespace {

// add_cst.tflite has a single constant operand (a 16-byte add constant), so the
// bank should record exactly that one buffer.
TEST(WeightBankTest, RecordsConstantWeightBuffers) {
  auto cc_model = testing::LoadTestFileModel("add_cst.tflite");
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  litert::compiler::Model model(ctx, cc_model.Get());
  auto graph = model.Subgraph(0);
  ASSERT_TRUE(graph.HasValue());

  WeightBank bank;
  bank.AddSubgraph(graph.Value());

  EXPECT_EQ(bank.NumBuffers(), 1u);
  EXPECT_EQ(bank.TotalBytes(), 16u);
}

// cst_multi_subgraph.tflite has two subgraphs that reference the SAME constant
// buffer. Accumulating both subgraphs must keep a single entry: this is the
// cross-partition sharing case (e.g. prefill + decode sharing a weight).
TEST(WeightBankTest, DeduplicatesBufferSharedAcrossSubgraphs) {
  auto cc_model = testing::LoadTestFileModel("cst_multi_subgraph.tflite");
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  litert::compiler::Model model(ctx, cc_model.Get());
  ASSERT_EQ(model.NumSubgraphs(), 2u);
  auto subgraph_0 = model.Subgraph(0);
  auto subgraph_1 = model.Subgraph(1);
  ASSERT_TRUE(subgraph_0.HasValue());
  ASSERT_TRUE(subgraph_1.HasValue());

  WeightBank bank;
  bank.AddSubgraph(subgraph_0.Value());
  const size_t buffers_after_first = bank.NumBuffers();
  bank.AddSubgraph(subgraph_1.Value());

  // The shared buffer is recorded once, not once per subgraph.
  EXPECT_EQ(buffers_after_first, 1u);
  EXPECT_EQ(bank.NumBuffers(), 1u);
  EXPECT_EQ(bank.TotalBytes(), 16u);
}

// A freshly constructed bank is empty.
TEST(WeightBankTest, EmptyByDefault) {
  WeightBank bank;
  EXPECT_EQ(bank.NumBuffers(), 0u);
  EXPECT_EQ(bank.TotalBytes(), 0u);
}

// Finalize() packs the single buffer at offset 0 and sets the bank size to its
// byte size.
TEST(WeightBankTest, FinalizePacksSingleBufferAtZero) {
  auto cc_model = testing::LoadTestFileModel("add_cst.tflite");
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  litert::compiler::Model model(ctx, cc_model.Get());
  auto graph = model.Subgraph(0);
  ASSERT_TRUE(graph.HasValue());

  WeightBank bank;
  bank.AddSubgraph(graph.Value());
  bank.Finalize();

  EXPECT_EQ(bank.BankSize(), bank.TotalBytes());
  EXPECT_EQ(bank.BankSize(), 16u);
}

// Finalize() lays out multiple buffers contiguously and without overlap: the
// offsets cover [0, BankSize()) exactly once when walked in offset order.
TEST(WeightBankTest, FinalizeLaysOutContiguously) {
  auto cc_model = testing::LoadTestFileModel("multi_subgraph.tflite");
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  litert::compiler::Model model(ctx, cc_model.Get());
  WeightBank bank;
  for (size_t s = 0; s < model.NumSubgraphs(); ++s) {
    auto graph = model.Subgraph(s);
    ASSERT_TRUE(graph.HasValue());
    bank.AddSubgraph(graph.Value());
  }
  bank.Finalize();

  ASSERT_EQ(bank.NumBuffers(), 3u);
  EXPECT_EQ(bank.BankSize(), bank.TotalBytes());
  EXPECT_EQ(bank.BankSize(), 48u);

  // Collect the per-buffer offsets and verify they tile [0, BankSize())
  // exactly once with no gaps or overlap.
  std::vector<size_t> offsets;
  for (size_t s = 0; s < model.NumSubgraphs(); ++s) {
    auto graph = model.Subgraph(s);
    for (const auto& op : graph.Value().Ops()) {
      for (const auto& input : op.Inputs()) {
        if (input.HasWeights()) {
          offsets.push_back(bank.OffsetOf(input.Weights().BufferId()));
        }
      }
    }
  }
  std::sort(offsets.begin(), offsets.end());
  offsets.erase(std::unique(offsets.begin(), offsets.end()), offsets.end());
  ASSERT_EQ(offsets.size(), 3u);
  EXPECT_EQ(offsets[0], 0u);
  EXPECT_EQ(offsets[1], 16u);
  EXPECT_EQ(offsets[2], 32u);
}

// Finalizing twice (after re-adding) yields the same deterministic layout.
TEST(WeightBankTest, FinalizeIsDeterministic) {
  auto cc_model = testing::LoadTestFileModel("multi_subgraph.tflite");
  const LiteRtCompilerContext* ctx = LrtGetCompilerContext();
  litert::compiler::Model model(ctx, cc_model.Get());
  auto load_bank = [&]() {
    WeightBank bank;
    for (size_t s = 0; s < model.NumSubgraphs(); ++s) {
      auto graph = model.Subgraph(s);
      bank.AddSubgraph(graph.Value());
    }
    bank.Finalize();
    return bank;
  };

  WeightBank bank_a = load_bank();
  WeightBank bank_b = load_bank();
  EXPECT_EQ(bank_a.BankSize(), bank_b.BankSize());
}

}  // namespace
}  // namespace openvino
}  // namespace litert
