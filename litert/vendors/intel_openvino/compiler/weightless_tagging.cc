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
#include <optional>

#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "litert/vendors/intel_openvino/compiler/weight_bank.h"
#include "litert/vendors/intel_openvino/compiler/weightless_caching_attributes.hpp"

namespace litert::openvino {

size_t TagWeightlessConstants(const std::shared_ptr<ov::Model>& model,
                              const WeightBank& bank) {
  size_t tagged = 0;
  for (const auto& node : model->get_ops()) {
    auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
    if (!constant) {
      continue;
    }
    // The TFLite frontend names a weight constant after its LiteRt tensor, so
    // the bank can resolve it to a bank offset. Constants the bank never saw
    // (e.g. axis/scalar constants the frontend synthesizes) return nullopt and
    // are left baked.
    const std::optional<size_t> offset =
        bank.OffsetOfName(constant->get_friendly_name());
    if (!offset.has_value()) {
      continue;
    }
    constant->get_rt_info()[ov::WeightlessCacheAttribute::get_type_info_static()] =
        ov::WeightlessCacheAttribute(constant->get_byte_size(), *offset,
                                     constant->get_element_type());
    ++tagged;
  }
  return tagged;
}

}  // namespace litert::openvino
