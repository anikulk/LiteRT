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

#include "litert/vendors/intel_openvino/compiler/global_graph.h"

#include <cstdint>
#include <cstring>
#include <string>

#include "litert/cc/litert_expected.h"

namespace litert::openvino {
namespace {

constexpr char kMagic[8] = {'O', 'V', 'G', 'L', 'O', 'B', 'A', 'L'};

void PutU32(std::string& s, uint32_t v) {
  s.append(reinterpret_cast<const char*>(&v), sizeof(v));
}
void PutU64(std::string& s, uint64_t v) {
  s.append(reinterpret_cast<const char*>(&v), sizeof(v));
}

// Bounds-checked little-endian readers over a [data, data+size) cursor.
struct Reader {
  const uint8_t* p;
  const uint8_t* end;
  bool ok = true;
  bool Bytes(void* out, size_t n) {
    if (!ok || static_cast<size_t>(end - p) < n) {
      ok = false;
      return false;
    }
    std::memcpy(out, p, n);
    p += n;
    return true;
  }
  uint32_t U32() {
    uint32_t v = 0;
    Bytes(&v, sizeof(v));
    return v;
  }
  uint64_t U64() {
    uint64_t v = 0;
    Bytes(&v, sizeof(v));
    return v;
  }
  bool Str(std::string& out, size_t n) {
    if (!ok || static_cast<size_t>(end - p) < n) {
      ok = false;
      return false;
    }
    out.assign(reinterpret_cast<const char*>(p), n);
    p += n;
    return true;
  }
};

}  // namespace

bool OpenVinoGlobalGraph::HasMagic(const uint8_t* data, size_t size) {
  return data != nullptr && size >= sizeof(kMagic) &&
         std::memcmp(data, kMagic, sizeof(kMagic)) == 0;
}

size_t OpenVinoGlobalGraph::BankBytes() const {
  size_t total = 0;
  for (const auto& [id, bytes] : buffers) total += bytes.size();
  return total;
}

std::string OpenVinoGlobalGraph::Serialize() const {
  std::string out;
  out.append(kMagic, sizeof(kMagic));
  // shared buffer pool
  PutU32(out, static_cast<uint32_t>(buffers.size()));
  for (const auto& [id, bytes] : buffers) {
    PutU32(out, id);
    PutU64(out, bytes.size());
    out.append(bytes);
  }
  // subgraphs
  PutU32(out, static_cast<uint32_t>(subgraphs.size()));
  for (const auto& [name, sg] : subgraphs) {
    PutU32(out, static_cast<uint32_t>(sg.name.size()));
    out.append(sg.name);
    out.push_back(static_cast<char>(sg.device));
    PutU32(out, static_cast<uint32_t>(sg.const_map.size()));
    for (const auto& [input_index, buffer_id] : sg.const_map) {
      PutU32(out, input_index);
      PutU32(out, buffer_id);
    }
    PutU64(out, sg.payload.size());
    out.append(sg.payload);
  }
  return out;
}

litert::Expected<OpenVinoGlobalGraph> OpenVinoGlobalGraph::Parse(
    const uint8_t* data, size_t size) {
  if (!HasMagic(data, size)) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "OpenVinoGlobalGraph: bad magic");
  }
  Reader r{data + sizeof(kMagic), data + size};
  OpenVinoGlobalGraph g;

  const uint32_t num_buffers = r.U32();
  for (uint32_t i = 0; i < num_buffers && r.ok; ++i) {
    const uint32_t id = r.U32();
    const uint64_t sz = r.U64();
    std::string bytes;
    r.Str(bytes, static_cast<size_t>(sz));
    if (r.ok) g.buffers.emplace(id, std::move(bytes));
  }

  const uint32_t num_subgraphs = r.U32();
  for (uint32_t i = 0; i < num_subgraphs && r.ok; ++i) {
    Subgraph sg;
    const uint32_t name_len = r.U32();
    r.Str(sg.name, name_len);
    uint8_t dev = 0;
    r.Bytes(&dev, 1);
    sg.device = dev;
    const uint32_t cm_len = r.U32();
    for (uint32_t j = 0; j < cm_len && r.ok; ++j) {
      const uint32_t idx = r.U32();
      const uint32_t bid = r.U32();
      sg.const_map.emplace(idx, bid);
    }
    const uint64_t payload_len = r.U64();
    r.Str(sg.payload, static_cast<size_t>(payload_len));
    if (r.ok) g.subgraphs.emplace(sg.name, std::move(sg));
  }

  if (!r.ok) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "OpenVinoGlobalGraph: truncated/corrupt container");
  }
  return g;
}

}  // namespace litert::openvino
