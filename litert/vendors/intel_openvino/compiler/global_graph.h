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

#ifndef LITERT_VENDORS_INTEL_OPENVINO_COMPILER_GLOBAL_GRAPH_H_
#define LITERT_VENDORS_INTEL_OPENVINO_COMPILER_GLOBAL_GRAPH_H_

#include <cstdint>
#include <map>
#include <string>

#include "litert/cc/litert_expected.h"

namespace litert::openvino {

// Cross-partition weight sharing container, structured after the upstream
// reference "Example Vendor Plugin" GlobalGraph pattern
// (google-ai-edge/LiteRT commit 20da64b8): instead of serializing each partition
// as an isolated blob, ALL partitions are aggregated into ONE global container
// with a shared buffer pool + per-subgraph topology, and the SAME serialized blob
// is returned for every partition.
//
// Adaptation for OpenVINO: the reference resolves shared constants by attaching
// the pooled buffer into the invocation's input list at runtime. OpenVINO CANNOT
// do that (a compiled OV blob's weights are baked or pulled via weights_path; OV
// constants are not runtime-bindable inputs). So this container carries the same
// {shared buffer pool, per-subgraph payload + const_map} schema, but the actual
// weight resolution still uses OpenVINO's native weightless caching: each
// partition's OV payload is compiled weightless (WeightlessCacheAttribute), and
// at runtime the dispatcher materializes the buffer pool once to a file for
// ov::weights_path. The const_map is retained as reference-parity metadata.
//
// Serialized layout (single blob, little-endian):
//   magic  "OVGLOBAL"                       (8 bytes)
//   uint32 num_buffers
//     repeat: uint32 buffer_id, uint64 size, [size bytes]     (shared pool)
//   uint32 num_subgraphs
//     repeat: uint32 name_len, [name], uint8 device_enum,
//             uint32 const_map_len,
//               repeat: uint32 input_index, uint32 buffer_id  (const_map)
//             uint64 payload_len, [payload bytes]              (weightless OV blob)
class OpenVinoGlobalGraph {
 public:
  // One compiled partition entry in the container.
  struct Subgraph {
    std::string name;                              // e.g. "Partition_0"
    uint8_t device = 0;                            // LiteRtIntelOpenVinoGraphBackend
    std::map<uint32_t, uint32_t> const_map;        // OV input index -> buffer_id
    std::string payload;                           // weightless OV exported blob
  };

  // Shared buffer pool: buffer_id -> raw weight bytes (deduplicated).
  std::map<uint32_t, std::string> buffers;
  // Partition topologies, keyed by graph name (selected at dispatch by
  // function_name / graph order).
  std::map<std::string, Subgraph> subgraphs;

  // Serialize the whole container to one blob (see layout above).
  std::string Serialize() const;

  // Parse a container blob. Returns an error if magic/bounds are invalid.
  static litert::Expected<OpenVinoGlobalGraph> Parse(const uint8_t* data,
                                                     size_t size);

  // Fast check: does |data| begin with the OVGLOBAL magic?
  static bool HasMagic(const uint8_t* data, size_t size);

  // Total bytes across the shared buffer pool (the deduplicated weight size).
  size_t BankBytes() const;
};

}  // namespace litert::openvino

#endif  // LITERT_VENDORS_INTEL_OPENVINO_COMPILER_GLOBAL_GRAPH_H_
