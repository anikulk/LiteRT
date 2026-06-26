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
#include <cstdint>
#include <vector>

#include "litert/compiler/cc/litert_model.h"

namespace litert::openvino {

void WeightBank::AddSubgraph(const litert::compiler::Subgraph& subgraph) {
  for (const auto& op : subgraph.Ops()) {
    for (const auto& input : op.Inputs()) {
      if (!input.HasWeights()) {
        continue;
      }
      const auto weights = input.Weights();
      // Keyed by BufferId, so a buffer shared by multiple ops/partitions is
      // recorded once. The size is identical for a given id, so re-assignment
      // is harmless.
      buffer_sizes_[weights.BufferId()] = weights.Bytes().size();
    }
  }
}

size_t WeightBank::TotalBytes() const {
  size_t total = 0;
  for (const auto& [buffer_id, size] : buffer_sizes_) {
    total += size;
  }
  return total;
}

void WeightBank::Finalize() {
  // Lay buffers out in ascending BufferId order for a deterministic packing.
  std::vector<int32_t> buffer_ids;
  buffer_ids.reserve(buffer_sizes_.size());
  for (const auto& [buffer_id, size] : buffer_sizes_) {
    buffer_ids.push_back(buffer_id);
  }
  std::sort(buffer_ids.begin(), buffer_ids.end());

  buffer_offsets_.clear();
  size_t offset = 0;
  for (int32_t buffer_id : buffer_ids) {
    buffer_offsets_[buffer_id] = offset;
    offset += buffer_sizes_[buffer_id];
  }
  bank_size_ = offset;
}

size_t WeightBank::OffsetOf(int32_t buffer_id) const {
  auto it = buffer_offsets_.find(buffer_id);
  return it == buffer_offsets_.end() ? 0 : it->second;
}

}  // namespace litert::openvino
