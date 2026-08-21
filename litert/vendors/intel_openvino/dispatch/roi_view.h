// Copyright (C) 2026 Intel Corporation
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

#ifndef ODML_LITERT_LITERT_VENDORS_INTEL_OPENVINO_DISPATCH_ROI_VIEW_H_
#define ODML_LITERT_LITERT_VENDORS_INTEL_OPENVINO_DISPATCH_ROI_VIEW_H_

#include <cstddef>
#include <exception>

#include "openvino/core/coordinate.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/runtime/tensor.hpp"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/litert_common.h"
#include "litert/cc/litert_expected.h"

namespace litert::openvino {

// Returns a zero-copy ROI sub-view of buffer_tensor sized to port_shape (offset
// 0 on every dim), inheriting the parent's strides for strided KV access. The
// split-context KV-cache primitive: one max-length KV buffer shared by shorter
// signatures. Returns the buffer unchanged on equal shapes; an error (never a
// throw) on rank mismatch, under-sized buffer, or out-of-bounds extent.
inline litert::Expected<ov::Tensor> MakePortView(const ov::Tensor& buffer_tensor,
                                                 const ov::Shape& port_shape) {
  try {
    const ov::Shape& buffer_shape = buffer_tensor.get_shape();
    if (buffer_shape == port_shape) {
      return buffer_tensor;
    }
    if (buffer_shape.size() != port_shape.size()) {
      return litert::Error(
          kLiteRtStatusErrorRuntimeFailure,
          "KV buffer rank does not match graph port rank; cannot create view");
    }
    // The design trims exactly one (cache-length) dim; more than one is
    // unexpected, so warn but still build the view (bounds guard is final).
    size_t differing_dims = 0;
    for (size_t i = 0; i < buffer_shape.size(); ++i) {
      if (buffer_shape[i] < port_shape[i]) {
        return litert::Error(
            kLiteRtStatusErrorRuntimeFailure,
            "KV buffer smaller than graph port; cannot create view");
      }
      if (buffer_shape[i] != port_shape[i]) {
        ++differing_dims;
      }
    }
    if (differing_dims > 1) {
      LITERT_LOG(LITERT_WARNING,
                 "Strided KV sub-view trims %zu dimensions; the split-context "
                 "design expects exactly one",
                 differing_dims);
    }
    ov::Coordinate begin(buffer_shape.size(), 0);
    ov::Coordinate end(port_shape.begin(), port_shape.end());
    // Zero-copy ROI sub-view sharing the parent's memory and strides.
    ov::Tensor view(buffer_tensor, begin, end);

    // Bounds guard: the ROI inherits the parent's byte strides, so an
    // under-allocated parent would let the driver walk off the end at execution
    // (SIGSEGV in libze_intel_npu, never at bind time). Fail loudly here instead.
    const ov::Strides& parent_strides = buffer_tensor.get_strides();  // bytes
    const size_t elem_size = buffer_tensor.get_element_type().size();
    size_t max_byte_offset = 0;
    for (size_t i = 0; i < port_shape.size(); ++i) {
      if (port_shape[i] == 0) continue;
      max_byte_offset += (port_shape[i] - 1) * parent_strides[i];
    }
    const size_t roi_end_byte = max_byte_offset + elem_size;
    const size_t parent_capacity = buffer_tensor.get_byte_size();
    LITERT_LOG(LITERT_VERBOSE,
               "Strided KV sub-view bind: parent data=%p parent_capacity=%zu "
               "roi_end_byte=%zu elem_size=%zu",
               buffer_tensor.data(), parent_capacity, roi_end_byte, elem_size);
    if (roi_end_byte > parent_capacity) {
      return litert::Error(
          kLiteRtStatusErrorRuntimeFailure,
          "Strided KV sub-view extends past the parent allocation; the parent "
          "buffer was allocated smaller than its declared (max-length) shape");
    }
    return view;
  } catch (const std::exception& e) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure, e.what());
  }
}

}  // namespace litert::openvino

#endif  // ODML_LITERT_LITERT_VENDORS_INTEL_OPENVINO_DISPATCH_ROI_VIEW_H_
