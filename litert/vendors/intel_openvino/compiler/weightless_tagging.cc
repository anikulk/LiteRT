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
#include <cstdint>
#include <map>
#include <memory>
#include <optional>

#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "litert/vendors/intel_openvino/compiler/weight_bank.h"
#include "litert/vendors/intel_openvino/compiler/weightless_caching_attributes.hpp"

namespace litert::openvino {

// Constants at or below this size are never tagged weightless, even though they
// remain recorded in the shared bank (so every larger weight keeps its bank
// offset). Shape-determining control constants -- Transpose permutations,
// gather/reduce axes, Reshape targets, scalars -- are read by OpenVINO during
// compile-time shape inference, but a weightless-tagged constant carries no data
// at compile time (the bank file is only materialized at runtime by the
// dispatcher), so tagging one yields a garbage permutation and a compile crash
// (e.g. "Permutation AxisVector{0,0,2,0} is not valid"). These control constants
// are bounded by tensor rank (< 6 dims) and thus at most a few dozen bytes, while
// every shareable real weight is far larger, so the threshold separates them
// cleanly. Leaving a sub-threshold constant baked costs only negligible
// duplication; crucially it stays in the bank, so real-weight bank offsets are
// unchanged from the all-buffers layout.
constexpr size_t kMinWeightlessTagBytes = 256;

size_t TagWeightlessConstants(const std::shared_ptr<ov::Model>& model,
                              const WeightBank& bank,
                              std::map<uint32_t, uint32_t>* const_map) {
  size_t tagged = 0;
  uint32_t ordinal = 0;
  for (const auto& node : model->get_ops()) {
    auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
    if (!constant) {
      continue;
    }
    // Stable per-model ordinal over Constant nodes; used as the const_map key
    // (the GlobalGraph reference keys const_map by the op's input index -- we
    // use the constant's ordinal here as the equivalent stable handle).
    const uint32_t this_ordinal = ordinal++;
    // The TFLite frontend names a weight constant after its LiteRt tensor, so
    // the bank can resolve it to a bank offset. Constants the bank never saw
    // (e.g. axis/scalar constants the frontend synthesizes) return nullopt and
    // are left baked.
    const std::optional<size_t> offset =
        bank.OffsetOfName(constant->get_friendly_name());
    if (!offset.has_value()) {
      continue;
    }
    // Small control/shape constants stay baked so compile-time shape inference
    // can read their values (see kMinWeightlessTagBytes). They remain in the
    // bank, preserving every larger weight's offset.
    if (constant->get_byte_size() <= kMinWeightlessTagBytes) {
      continue;
    }
    constant->get_rt_info()[ov::WeightlessCacheAttribute::get_type_info_static()] =
        ov::WeightlessCacheAttribute(constant->get_byte_size(), *offset,
                                     constant->get_element_type());
    if (const_map != nullptr) {
      const std::optional<int32_t> buffer_id =
          bank.BufferIdOfName(constant->get_friendly_name());
      if (buffer_id.has_value()) {
        (*const_map)[this_ordinal] = static_cast<uint32_t>(*buffer_id);
      }
    }
    ++tagged;
  }
  return tagged;
}

}  // namespace litert::openvino
