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

#ifndef LITERT_VENDORS_INTEL_OPENVINO_COMPILER_WEIGHTLESS_TAGGING_H_
#define LITERT_VENDORS_INTEL_OPENVINO_COMPILER_WEIGHTLESS_TAGGING_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>

#include "openvino/core/model.hpp"
#include "litert/vendors/intel_openvino/compiler/weight_bank.h"

namespace litert::openvino {

// Tags each weight constant in |model| with an ov::WeightlessCacheAttribute
// carrying the constant's size and its offset in |bank|, so the OpenVINO
// serializer can emit a weightless blob that references the shared bank instead
// of baking the bytes. A constant is matched to the bank by its friendly_name
// (which the TFLite frontend sets to the LiteRt tensor name); constants the
// bank does not know (e.g. small scalars the frontend synthesizes) are left
// untouched. |bank| must already be Finalize()-d.
//
// If |const_map| is non-null, it is populated with the GlobalGraph-style
// mapping (a per-model constant ordinal -> shared BufferId) for every tagged
// constant, mirroring the upstream reference plugin's const_map. This is
// carried in the container as reference-parity metadata; OpenVINO itself
// resolves the weights via the WeightlessCacheAttribute + weights_path, not the
// const_map.
//
// Returns the number of constants tagged.
size_t TagWeightlessConstants(const std::shared_ptr<ov::Model>& model,
                              const WeightBank& bank,
                              std::map<uint32_t, uint32_t>* const_map = nullptr);

}  // namespace litert::openvino

#endif  // LITERT_VENDORS_INTEL_OPENVINO_COMPILER_WEIGHTLESS_TAGGING_H_
