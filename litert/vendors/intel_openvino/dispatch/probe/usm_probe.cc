// Probe: does ov::RemoteContext::create_tensor(USM_HOST_BUFFER) return a
// distinct allocation each call, even for identical size / same shared context?
//
// This mirrors GpuSharedBank::Bind()'s pool allocation (Mode A). If OV pooled
// USM-host buffers by size, two "models" with an identical total pool size
// would alias -> weight-sharing corruption across models in one process.
// Expected result: every base pointer is distinct.

#include <cstdint>
#include <cstdio>
#include <vector>

#include "openvino/openvino.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_gpu/remote_properties.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "openvino/runtime/remote_tensor.hpp"

namespace {

void* AllocPoolAndGetBase(ov::RemoteContext& ctx, size_t total_bytes,
                          ov::RemoteTensor* keep_alive) {
  // Exactly what GpuSharedBank::Bind does for the pool.
  ov::RemoteTensor usm = ctx.create_tensor(
      ov::element::u8, ov::Shape{total_bytes},
      {{ov::intel_gpu::shared_mem_type.name(),
        ov::intel_gpu::SharedMemType::USM_HOST_BUFFER}});
  void* base =
      usm.get_params().at(ov::intel_gpu::mem_handle.name()).as<void*>();
  *keep_alive = std::move(usm);  // hold it so the allocation isn't freed
  return base;
}

}  // namespace

int main() {
  // One Core for the whole process -- the recommended OV pattern, and what the
  // LiteRT dispatch does (singleton).
  ov::Core core;

  const size_t kTotal = 512ull * 1024 * 1024;  // 512 MB, same size every call
  printf("Probe: %zu-byte USM_HOST_BUFFER allocations on a shared GPU context\n",
         kTotal);

  // --- Case 1: same shared RemoteContext, N identical-size allocations ---
  ov::RemoteContext ctx = core.get_default_context("GPU");
  const int kN = 4;
  std::vector<ov::RemoteTensor> hold(kN);
  std::vector<void*> bases(kN);
  for (int i = 0; i < kN; ++i) {
    bases[i] = AllocPoolAndGetBase(ctx, kTotal, &hold[i]);
    printf("  case1 alloc[%d] base = %p\n", i, bases[i]);
  }

  // --- Case 2: two separate get_default_context() calls (simulating two
  // models each doing their own Bind on the process-shared core) ---
  ov::RemoteContext ctx_a = core.get_default_context("GPU");
  ov::RemoteContext ctx_b = core.get_default_context("GPU");
  ov::RemoteTensor hold_a, hold_b;
  void* base_a = AllocPoolAndGetBase(ctx_a, kTotal, &hold_a);
  void* base_b = AllocPoolAndGetBase(ctx_b, kTotal, &hold_b);
  printf("  case2 modelA base = %p\n", base_a);
  printf("  case2 modelB base = %p\n", base_b);

  // --- Verdict ---
  bool all_distinct = true;
  for (int i = 0; i < kN && all_distinct; ++i)
    for (int j = i + 1; j < kN; ++j)
      if (bases[i] == bases[j]) all_distinct = false;
  if (base_a == base_b) all_distinct = false;

  printf("\nVERDICT: %s\n",
         all_distinct
             ? "PASS - every USM_HOST allocation is a distinct base pointer "
               "(no size-keyed pooling; two models cannot alias)."
             : "FAIL - a base pointer was reused across allocations!");
  return all_distinct ? 0 : 1;
}
