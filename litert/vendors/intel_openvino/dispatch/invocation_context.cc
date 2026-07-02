// Copyright (C) 2025 Intel Corporation
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

#include "litert/vendors/intel_openvino/dispatch/invocation_context.h"

#include <algorithm>
#include <chrono>  // NOLINT
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <ios>
#include <istream>
#include <streambuf>
#include <string>
#include <vector>

#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "openvino/core/any.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"
#include "litert/c/internal/litert_logging.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/c/litert_common.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_tensor_buffer.h"
#include "litert/c/litert_tensor_buffer_requirements.h"
#include "litert/c/litert_tensor_buffer_types.h"
#include "litert/c/options/litert_intel_openvino_options.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/util/tensor_type_util.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/intel_openvino/bytecode_header.h"

namespace {

// This class is copied from the OpenVINO codebase with minor modifications
// for Google C++ Style Guide compliance. It wraps a pre-allocated memory
// buffer to provide a std::streambuf interface, enabling zero-copy stream
// reading.
//
// TODO(b/449624371): Remove SharedStreamBuffer once OpenVINO provides a
// public equivalent.
class SharedStreamBuffer : public std::streambuf {
 public:
  SharedStreamBuffer(const char* data, size_t size)
      : data_(data), size_(size), offset_(0) {}
  explicit SharedStreamBuffer(const void* data, size_t size)
      : SharedStreamBuffer(reinterpret_cast<const char*>(data), size) {}

 protected:
  // override std::streambuf methods
  std::streamsize xsgetn(char* s, std::streamsize count) override {
    auto real_count = std::min<std::streamsize>(size_ - offset_, count);
    std::memcpy(s, data_ + offset_, real_count);
    offset_ += real_count;
    return real_count;
  }

  int_type underflow() override {
    return (size_ == offset_) ? traits_type::eof()
                              : traits_type::to_int_type(*(data_ + offset_));
  }

  int_type uflow() override {
    return (size_ == offset_) ? traits_type::eof()
                              : traits_type::to_int_type(*(data_ + offset_++));
  }

  std::streamsize showmanyc() override { return size_ - offset_; }

  pos_type seekpos(pos_type pos, std::ios_base::openmode which) override {
    return seekoff(pos, std::ios_base::beg, which);
  }

  pos_type seekoff(off_type off, std::ios_base::seekdir dir,
                   std::ios_base::openmode which) override {
    if (which != std::ios_base::in) {
      return pos_type(off_type(-1));
    }

    size_t new_offset;
    switch (dir) {
      case std::ios_base::beg:
        new_offset = off;
        break;
      case std::ios_base::cur:
        new_offset = offset_ + off;
        break;
      case std::ios_base::end:
        new_offset = size_ + off;
        break;
      default:
        return pos_type(off_type(-1));
    }

    // Check bounds
    if (new_offset > size_) {
      return pos_type(off_type(-1));
    }

    offset_ = new_offset;
    return pos_type(offset_);
  }

  // Non-virtual overload with default argument for backward compatibility
  pos_type seekoff(off_type off, std::ios_base::seekdir dir) {
    return seekoff(off, dir, std::ios_base::in);
  }

 private:
  const char* data_;
  const size_t size_;
  size_t offset_;
};

// Frame prepended by the compiler to partition 0's bytecode in embedded
// weight-sharing mode: "WLBANK\0\0" + uint64 bank_size (LE) + [bank bytes].
constexpr char kBankMagic[8] = {'W', 'L', 'B', 'A', 'N', 'K', 0, 0};
constexpr size_t kBankFrameHeader = sizeof(kBankMagic) + sizeof(uint64_t);

// Process-wide path of the shared weights bank once it has been extracted to a
// temp file. The compiler embeds the bank in a single partition, but every
// partition needs weights_path at import; the first partition to see the frame
// writes the temp file and records its path here, and the others reuse it.
absl::Mutex& BankPathMutex() {
  static absl::Mutex* mu = new absl::Mutex();
  return *mu;
}
std::string& SharedBankPath() ABSL_EXCLUSIVE_LOCKS_REQUIRED(BankPathMutex()) {
  static std::string* path = new std::string();
  return *path;
}

// If |bytecode| begins with the WLBANK frame, extracts the bank to a temp .bin
// file exactly once (recording the path for later partitions), advances
// |bytecode|/|size| past the frame to the weightless payload, and returns the
// bank path. Otherwise returns any previously extracted bank path (empty if
// none). OpenVINO's weights_path requires a filesystem path ending in ".bin".
litert::Expected<std::string> MaybeExtractWeightsBank(const uint8_t*& bytecode,
                                                      size_t& size) {
  const bool has_frame =
      size >= kBankFrameHeader &&
      std::memcmp(bytecode, kBankMagic, sizeof(kBankMagic)) == 0;
  if (!has_frame) {
    absl::MutexLock lock(&BankPathMutex());
    return SharedBankPath();  // may be empty (no embedded bank in this model)
  }

  uint64_t bank_size = 0;
  std::memcpy(&bank_size, bytecode + sizeof(kBankMagic), sizeof(bank_size));
  if (kBankFrameHeader + bank_size > size) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "WLBANK frame size exceeds bytecode buffer");
  }
  const uint8_t* bank_data = bytecode + kBankFrameHeader;

  absl::MutexLock lock(&BankPathMutex());
  std::string& shared_path = SharedBankPath();
  if (shared_path.empty()) {
    const char* tmp_dir = std::getenv("TMPDIR");
    if (tmp_dir == nullptr || tmp_dir[0] == '\0') tmp_dir = "/tmp";
    // The device buffer is per-process and identical across partitions, so a
    // fixed name is sufficient; the .bin extension is required by the GPU
    // plugin's weightless importer.
    std::string path =
        std::string(tmp_dir) + "/litert_ov_weights_bank.bin";
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
      return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                           "Cannot open temp weights bank for writing: " + path);
    }
    out.write(reinterpret_cast<const char*>(bank_data),
              static_cast<std::streamsize>(bank_size));
    out.close();
    if (!out) {
      return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                           "Failed writing temp weights bank: " + path);
    }
    shared_path = path;
    LITERT_LOG(LITERT_INFO,
               "Extracted embedded weights bank (%llu bytes) -> %s",
               static_cast<unsigned long long>(bank_size), shared_path.c_str());
  }

  // Advance past the frame to the weightless payload.
  bytecode += kBankFrameHeader + bank_size;
  size -= kBankFrameHeader + bank_size;
  return shared_path;
}

}  // namespace

litert::Expected<LiteRtDispatchInvocationContextT::Ptr>
LiteRtDispatchInvocationContextT::Create(
    LiteRtDispatchDeviceContextT& device_context,
    LiteRtDispatchExecutableType exec_type,
    const LiteRtMemBuffer* exec_bytecode_buffer, const char* function_name,
    int num_inputs, int num_outputs,
    const IntelOpenVinoOptions* intel_openvino_opts) {
  const void* exec_bytecode_ptr =
      static_cast<const uint8_t*>(exec_bytecode_buffer->base_addr) +
      exec_bytecode_buffer->offset;
  auto exec_bytecode_size = exec_bytecode_buffer->size;

  // If the compiler embedded a self-describing header, honor the device
  // recorded there.  Per-partition bytecode produced by the LiteRT OpenVINO
  // compiler plugin always carries this header, so each partition can be
  // dispatched to its own target device (NPU/CPU/GPU) even when the
  // model-wide options request a different default.
  std::string device = "NPU";  // Default device
  LiteRtIntelOpenVinoGraphBackend embedded_graph_backend =
      kLiteRtIntelOpenVinoGraphBackendNPU;
  size_t payload_offset = 0;
  bool device_from_header = litert::openvino::TryParseBytecodeHeader(
      exec_bytecode_ptr, exec_bytecode_size, &embedded_graph_backend,
      &payload_offset);
  if (device_from_header) {
    device = litert::openvino::GraphBackendToString(embedded_graph_backend);
    exec_bytecode_ptr =
        static_cast<const uint8_t*>(exec_bytecode_ptr) + payload_offset;
    exec_bytecode_size -= payload_offset;
    LITERT_LOG(LITERT_INFO, "Dispatch: using device '%s' from bytecode header",
               device.c_str());
  } else {
    LITERT_LOG(LITERT_INFO,
               "Dispatch: no bytecode header found, defaulting to '%s'",
               device.c_str());
  }

  // Validate that the requested device is actually available on this system
  // before setting the device.
  auto core = device_context.getCore();
  if (!core) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Failed to get OpenVINO core from device context");
  }
  {
    const std::vector<std::string>& available_devices =
        OpenVINOSharedCore::GetInstance()->GetAvailableDevices();

    auto matches = [&device](const std::string& name) {
      if (name == device) return true;
      auto dot = name.find('.');
      return dot != std::string::npos && name.substr(0, dot) == device;
    };

    if (std::none_of(available_devices.begin(), available_devices.end(),
                     matches)) {
      std::string available_list;
      for (const auto& d : available_devices) {
        if (!available_list.empty()) available_list += ", ";
        available_list += d;
      }
      LITERT_LOG(LITERT_ERROR,
                 "Dispatch: requested OpenVINO device '%s' is not available. "
                 "Available devices: [%s]",
                 device.c_str(), available_list.c_str());
      return litert::Error(
          kLiteRtStatusErrorRuntimeFailure,
          "Requested OpenVINO device is not available on this system");
    }
  }
  LITERT_LOG(LITERT_INFO, "Using Intel OpenVINO device: %s", device.c_str());

  OpenVINOSharedCore::GetInstance()->SetDevice(device);

  if (!exec_bytecode_ptr || exec_bytecode_size == 0) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Empty bytecode buffer");
  }

  // Embedded weight-sharing: if the compiler prepended the shared weights bank
  // (WLBANK frame, on partition 0), extract it to a temp .bin once and peel the
  // frame off so only the weightless payload is imported. Every partition then
  // imports weightless with weights_path pointing at that shared bank.
  auto bytecode_cursor = static_cast<const uint8_t*>(exec_bytecode_ptr);
  LITERT_ASSIGN_OR_RETURN(
      const std::string weights_bank_path,
      MaybeExtractWeightsBank(bytecode_cursor, exec_bytecode_size));
  exec_bytecode_ptr = bytecode_cursor;

  SharedStreamBuffer membuf(static_cast<const char*>(exec_bytecode_ptr),
                            exec_bytecode_size);
  std::istream model_stream(&membuf);
  if (!model_stream) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure,
                         "Failed to open model bytecode stream");
  }
  ov::CompiledModel compiled_model;
  try {
    if (!weights_bank_path.empty()) {
      LITERT_LOG(LITERT_INFO,
                 "Importing weightless model with shared weights bank '%s'",
                 weights_bank_path.c_str());
      compiled_model = core->import_model(
          model_stream, device,
          {ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE),
           ov::enable_weightless(true), ov::weights_path(weights_bank_path)});
    } else {
      compiled_model = core->import_model(model_stream, device);
    }
  } catch (const std::exception& e) {
    return litert::Error(kLiteRtStatusErrorRuntimeFailure, e.what());
  }

  auto infer_request = compiled_model.create_infer_request();
  LITERT_LOG(LITERT_INFO, "Openvino InvocationContext Initialize SUCCESS");
  // TODO: add support for loading cached model
  return Ptr(new LiteRtDispatchInvocationContextT(infer_request, device_context,
                                                  num_inputs, num_outputs));
}

litert::Expected<LiteRtTensorBufferRequirements>
LiteRtDispatchInvocationContextT::GetTensorBufferRequirements(
    const LiteRtRankedTensorType& tensor_type) {
  LiteRtTensorBufferType supported_tensor_buffer_types[] = {
      kLiteRtTensorBufferTypeOpenVINOTensorBuffer,
      // OpenVINO RemoteTensor doesn't support copy-free AHWB buffer. Until
      // it's supported, we use DMA-BUF.
      kLiteRtTensorBufferTypeDmaBuf,
      kLiteRtTensorBufferTypeAhwb,
  };

  int num_supported_tensor_buffer_types =
      sizeof(supported_tensor_buffer_types) /
      sizeof(supported_tensor_buffer_types[0]);
  auto buffer_size = litert::internal::GetNumPackedBytes(tensor_type);
  if (!buffer_size) {
    return litert::Unexpected(buffer_size.Error());
  }

  LiteRtTensorBufferRequirements requirements;
  auto status =
      device_context_.runtime_context()->create_tensor_buffer_requirements(
          num_supported_tensor_buffer_types, supported_tensor_buffer_types,
          *buffer_size, 0, /*strides=*/nullptr, &requirements);
  if (status != kLiteRtStatusOk)
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                              "Failed to get buffer requirements");

  return requirements;
}

litert::Expected<LiteRtTensorBufferRequirements>
LiteRtDispatchInvocationContextT::GetInputRequirements(
    int input_index, const LiteRtRankedTensorType& tensor_type) {
  return GetTensorBufferRequirements(tensor_type);
}

litert::Expected<LiteRtTensorBufferRequirements>
LiteRtDispatchInvocationContextT::GetOutputRequirements(
    int output_index, const LiteRtRankedTensorType& tensor_type) {
  return GetTensorBufferRequirements(tensor_type);
}

litert::Expected<void> LiteRtDispatchInvocationContextT::AttachInput(
    int graph_input_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  LITERT_ASSIGN_OR_RETURN(ov::Tensor ov_tensor,
                          device_context_.getOVTensor(tensor_buffer_handle));
  // TODO: visit this if need to maintain graph indices for inputs and outputs
  // in dispatch_api
  infer_request_.set_input_tensor(graph_input_index, ov_tensor);
  return {};
}

litert::Expected<void> LiteRtDispatchInvocationContextT::AttachOutput(
    int graph_output_index, LiteRtTensorBufferHandle tensor_buffer_handle) {
  LITERT_ASSIGN_OR_RETURN(ov::Tensor ov_tensor,
                          device_context_.getOVTensor(tensor_buffer_handle));
  // TODO: visit this if need to maintain graph indices for inputs and outputs
  // in dispatch_api
  infer_request_.set_output_tensor(graph_output_index, ov_tensor);
  return {};
}

litert::Expected<void> LiteRtDispatchInvocationContextT::Invoke() {
  infer_request_.start_async();
  if (!infer_request_.wait_for(
          std::chrono::milliseconds(kInferRequestTimeoutMs)))
    return litert::Unexpected(
        kLiteRtStatusErrorRuntimeFailure,
        "Failed to execute inference request due to timeout");
  return {};
}
