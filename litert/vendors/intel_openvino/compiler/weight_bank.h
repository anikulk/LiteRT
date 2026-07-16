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

#ifndef LITERT_VENDORS_INTEL_OPENVINO_COMPILER_WEIGHT_BANK_H_
#define LITERT_VENDORS_INTEL_OPENVINO_COMPILER_WEIGHT_BANK_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/compiler/cc/litert_model.h"

namespace litert::openvino {

// Backs cross-partition weight sharing: collects the distinct constant-weight
// buffers used across a model's partitions into one deduplicated pool, so the
// shared weights are stored once (referenced by every partition via the
// GlobalGraph const_map / shared buffer pool) instead of duplicated per
// partition.
//
// Buffers are keyed by LiteRt Weights::BufferId(): LiteRt's buffer manager
// assigns the same BufferId to tensors that share storage, so a buffer used by
// both the prefill and decode partitions is recorded exactly once. The bank
// also maps each weight tensor's name to its BufferId so the converted
// OpenVINO weights (whose friendly_name is the LiteRt tensor name) can be
// resolved back to the shared buffer they belong to.
class WeightBank {
 public:
  WeightBank() = default;

  // Records every constant-weight buffer referenced by |subgraph|. Repeated
  // BufferIds (within or across subgraphs) collapse to a single entry. Safe to
  // call once per partition to accumulate the union of all weight buffers.
  void AddSubgraph(const litert::compiler::Subgraph& subgraph);

  // Number of distinct weight buffers recorded so far.
  size_t NumBuffers() const { return buffer_bytes_.size(); }

  // Total bytes across all distinct buffers (i.e. the deduplicated weight size).
  size_t TotalBytes() const;

  // BufferId of the weight tensor named |tensor_name|, or nullopt if unknown.
  // Used to build the GlobalGraph const_map (OV weight -> shared buffer id).
  // Valid any time after AddSubgraph().
  std::optional<int32_t> BufferIdOfName(std::string_view tensor_name) const;

  // The deduplicated buffer pool as (BufferId -> bytes view), for populating the
  // GlobalGraph shared buffer pool. Available after AddSubgraph() calls.
  const std::unordered_map<int32_t, absl::Span<const uint8_t>>& Buffers()
      const {
    return buffer_bytes_;
  }

  // Assigns each recorded buffer a byte offset in a single tightly-packed bank.
  // Offsets are assigned in ascending BufferId order so the layout is
  // deterministic across runs (independent of op/subgraph iteration order).
  // Call once after all subgraphs have been added. Used by the weightless
  // (NPU) arm, which references the packed bank file by offset via weights_path.
  void Finalize();

  // Byte offset of |buffer_id| within the packed bank. Valid only after
  // Finalize(); returns 0 for an unknown id.
  size_t OffsetOf(int32_t buffer_id) const;

  // Byte offset of the weight tensor named |tensor_name| within the packed
  // bank, or nullopt if no recorded weight has that name. Tensors that share a
  // buffer have distinct names but resolve to the same offset. Valid only after
  // Finalize(); used to tag weightless OpenVINO constants by their friendly_name.
  std::optional<size_t> OffsetOfName(std::string_view tensor_name) const;

  // Total packed bank size in bytes (equals TotalBytes() for a tight pack).
  size_t BankSize() const { return bank_size_; }

  // Returns the packed bank: every distinct buffer's bytes copied to its
  // assigned offset, so the result has size BankSize(). Must be called after
  // Finalize().
  std::string SerializeBank() const;

 private:
  // BufferId -> the buffer's bytes (a view into the model's mmapped weights).
  std::unordered_map<int32_t, absl::Span<const uint8_t>> buffer_bytes_;
  // Weight tensor name -> its BufferId. Many names may map to one BufferId
  // (tensors that share storage), which is how shared weights resolve to a
  // single pool buffer.
  std::unordered_map<std::string, int32_t> name_to_buffer_id_;
  // BufferId -> byte offset in the packed bank, populated by Finalize().
  std::unordered_map<int32_t, size_t> buffer_offsets_;
  // Total packed bank size, set by Finalize().
  size_t bank_size_ = 0;
};

}  // namespace litert::openvino

#endif  // LITERT_VENDORS_INTEL_OPENVINO_COMPILER_WEIGHT_BANK_H_
