// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Standalone de-risk check for the split-context KV strided-ROI design.
//
// Question being answered: does the OpenVINO plugin accept an ov::Tensor that
// is a region-of-interest (ROI) sub-view of a LARGER parent buffer — bound to a
// STATIC-shape compiled input/output port via set_input_tensor /
// set_output_tensor — and read/write it correctly through the ROI's
// (non-contiguous) strides?
//
// This mirrors exactly what dispatch/invocation_context.cc MakePortView does:
//   ov::Tensor(parent, begin, end)  with begin=0 on every dim, end=port_shape.
//
// The parent buffer trims a MIDDLE dimension (seq at dim 2, like the K cache
// layout [1,H,S,D]), so the ROI is genuinely strided — a contiguous-assuming
// plugin would produce wrong results here.
//
// This is a hardware-dependent manual check, not a unit test: it exercises the
// real OpenVINO device plugin (CPU/GPU/NPU). Run it on the target device before
// trusting the split-context KV path there:
//
//   cd $LITERT_DIR
//   bazel-7.4.1-linux-x86_64 run --config=linux \
//     //litert/vendors/intel_openvino/dispatch:roi_view_manual_test -- NPU
//
// The trailing arg is the OpenVINO device (CPU | GPU | NPU); it defaults to CPU.
// Proven PASS on CPU (2026-07-28); rerun on NPU/GPU hardware to confirm there.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "openvino/openvino.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace {

constexpr size_t kH = 2;         // heads
constexpr size_t kSeqBig = 8;    // parent (decode) seq length
constexpr size_t kSeqPort = 5;   // compiled port (prefill) seq length
constexpr size_t kD = 3;         // head dim

// Physical linear offset of logical index (0,h,s,d) inside the BIG buffer.
size_t BigOffset(size_t h, size_t s, size_t d) {
  return ((0 * kH + h) * kSeqBig + s) * kD + d;
}

int Fail(const std::string& msg) {
  std::fprintf(stderr, "FAIL: %s\n", msg.c_str());
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string device = argc > 1 ? argv[1] : "CPU";
  std::fprintf(stderr, "ROI de-risk test on device: %s\n", device.c_str());

  try {
    ov::Core core;

    // Build a tiny static-shape model: y = x * 2, input port [1,H,kSeqPort,D].
    // The compiled port is STATIC at kSeqPort — nothing about the model knows
    // the buffer will actually be a sub-view of a bigger tensor.
    const ov::Shape port_shape{1, kH, kSeqPort, kD};
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                         ov::PartialShape(port_shape));
    auto two = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {2.0f});
    auto mul = std::make_shared<ov::op::v1::Multiply>(param, two);
    auto result = std::make_shared<ov::op::v0::Result>(mul);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                             ov::ParameterVector{param}, "roi_test");

    ov::CompiledModel compiled = core.compile_model(model, device);
    ov::InferRequest req = compiled.create_infer_request();

    // --- Parent (big) input buffer, seq=kSeqBig. Fill with distinct values. ---
    ov::Shape big_in_shape{1, kH, kSeqBig, kD};
    ov::Tensor big_in(ov::element::f32, big_in_shape);
    float* in_data = big_in.data<float>();
    for (size_t h = 0; h < kH; ++h)
      for (size_t s = 0; s < kSeqBig; ++s)
        for (size_t d = 0; d < kD; ++d)
          in_data[BigOffset(h, s, d)] =
              static_cast<float>(h * 100 + s * 10 + d + 1);

    // ROI sub-view sized to the static port: begin=0 everywhere, end=port_shape.
    // Trims dim 2 from kSeqBig to kSeqPort -> strided (non-contiguous) view.
    ov::Coordinate begin(big_in_shape.size(), 0);
    ov::Coordinate end(port_shape.begin(), port_shape.end());
    ov::Tensor in_roi(big_in, begin, end);

    // Sanity: the ROI reports the port shape but is strided vs the parent.
    if (in_roi.get_shape() != port_shape)
      return Fail("input ROI shape != port shape");
    std::fprintf(stderr, "input ROI shape ok; strides bytes:");
    for (auto st : in_roi.get_strides()) std::fprintf(stderr, " %zu", st);
    std::fprintf(stderr, "\n");

    // --- Parent (big) output buffer + ROI, mirroring AttachOutput. ---
    ov::Shape big_out_shape{1, kH, kSeqBig, kD};
    ov::Tensor big_out(ov::element::f32, big_out_shape);
    float* out_data = big_out.data<float>();
    for (size_t i = 0; i < big_out.get_size(); ++i) out_data[i] = -999.0f;
    ov::Tensor out_roi(big_out, begin,
                       ov::Coordinate(port_shape.begin(), port_shape.end()));

    // Bind the strided ROI views to the static ports — the crux of the test.
    req.set_input_tensor(in_roi);
    req.set_output_tensor(out_roi);
    req.infer();

    // Verify: for every logical (h,s<kSeqPort,d), out == 2*in, read via ROI
    // strides from the big buffers. Also verify the trimmed region of the big
    // OUTPUT buffer (s>=kSeqPort) was left untouched (proves strided write, not
    // a contiguous overwrite).
    int mismatches = 0;
    for (size_t h = 0; h < kH; ++h) {
      for (size_t s = 0; s < kSeqBig; ++s) {
        for (size_t d = 0; d < kD; ++d) {
          float got = out_data[BigOffset(h, s, d)];
          if (s < kSeqPort) {
            float want = 2.0f * in_data[BigOffset(h, s, d)];
            if (got != want) {
              if (mismatches < 10)
                std::fprintf(stderr,
                             "  mismatch @ (h=%zu,s=%zu,d=%zu): got %.1f want %.1f\n",
                             h, s, d, got, want);
              ++mismatches;
            }
          } else {
            // Outside the ROI: must remain the -999 sentinel.
            if (got != -999.0f) {
              if (mismatches < 10)
                std::fprintf(stderr,
                             "  ROI over-write @ (h=%zu,s=%zu,d=%zu): got %.1f "
                             "(sentinel clobbered)\n",
                             h, s, d, got);
              ++mismatches;
            }
          }
        }
      }
    }

    if (mismatches != 0)
      return Fail(std::to_string(mismatches) + " element mismatches");

    std::fprintf(stderr,
                 "PASS: strided ROI bound to static port; input read and output "
                 "written correctly through ROI strides; trimmed region "
                 "untouched.\n");
    return 0;
  } catch (const std::exception& e) {
    return Fail(std::string("exception: ") + e.what());
  }
}
