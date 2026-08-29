/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/mr/detail/pool_memory_resource_impl.hpp>
#include <rmm/mr/host_writable.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>

#include <cstddef>
#include <optional>

RMM_NAMESPACE_BEGIN
namespace mr {
/**
 * @addtogroup memory_resources
 * @{
 * @file
 */

/**
 * @brief A coalescing best-fit suballocator which uses a pool of memory allocated from
 *        an upstream memory_resource.
 *
 * Allocation and deallocation are thread-safe. Also,
 * this class is compatible with CUDA per-thread default stream.
 *
 * This class is copyable and shares ownership of its internal state, allowing
 * multiple instances to safely reference the same underlying pool.
 */
class RMM_EXPORT pool_memory_resource
  : public cuda::mr::shared_resource<detail::pool_memory_resource_impl> {
  using shared_base = cuda::mr::shared_resource<detail::pool_memory_resource_impl>;

 public:
  /**
   * @brief Enables the `cuda::mr::device_accessible` property
   *
   * This property declares that a `pool_memory_resource` provides device accessible memory
   */
  RMM_CONSTEXPR_FRIEND void get_property(pool_memory_resource const&,
                                         cuda::mr::device_accessible) noexcept
  {
  }

  /**
   * @brief Construct a `pool_memory_resource` and allocate the initial device memory pool using
   * `upstream`.
   *
   * @throws rmm::logic_error if `initial_pool_size` is not aligned to a multiple of 256 bytes.
   * @throws rmm::logic_error if `maximum_pool_size` is neither the default nor aligned to a
   * multiple of 256 bytes.
   *
   * @param upstream The resource from which to allocate blocks for the pool.
   * @param initial_pool_size Minimum size, in bytes, of the initial pool.
   * @param maximum_pool_size Maximum size, in bytes, that the pool can grow to. Defaults to all
   * of the available memory from the upstream resource.
   */
  explicit pool_memory_resource(cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
                                std::size_t initial_pool_size,
                                std::optional<std::size_t> maximum_pool_size = std::nullopt);

  /**
   * @briefreturn{rmm::device_async_resource_ref to the upstream resource}
   */
  [[nodiscard]] device_async_resource_ref get_upstream_resource() const noexcept;

  /**
   * @brief Computes the size of the current pool
   *
   * Includes allocated as well as free memory.
   *
   * @return std::size_t The total size of the currently allocated pool.
   */
  [[nodiscard]] std::size_t pool_size() const noexcept;

  /**
   * @brief Allocates memory of at least `bytes` bytes that the host may write to on return.
   *
   * Unlike `allocate`, the calling thread may write to the returned memory immediately with no
   * further synchronization. The pool may recycle a block whose previous owner still has a copy in
   * flight on the stream it was freed on; this performs the minimal wait for that copy rather than
   * requiring the caller to synchronize the whole stream.
   *
   * Memory returned by this function must be freed with `deallocate_host_writable`.
   *
   * @throws rmm::out_of_memory if the requested allocation could not be fulfilled
   *
   * @param stream The stream in which to order this allocation
   * @param bytes The size in bytes of the allocation
   * @param alignment Unused; alignment is always at least `CUDA_ALLOCATION_ALIGNMENT`
   * @return void* Pointer to memory the host may write to immediately
   */
  [[nodiscard]] void* allocate_host_writable(cuda::stream_ref stream,
                                             std::size_t bytes,
                                             std::size_t alignment = CUDA_ALLOCATION_ALIGNMENT);

  /**
   * @brief Deallocates memory returned by `allocate_host_writable`.
   *
   * @param stream The stream in which to order this deallocation
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation to deallocate
   * @param alignment Unused
   * @param device_exposed Whether the memory was ever used by device work on `stream`. When false,
   * a later host writer needs no wait at all. Defaults to true, which is always safe.
   */
  void deallocate_host_writable(cuda::stream_ref stream,
                                void* ptr,
                                std::size_t bytes,
                                std::size_t alignment = CUDA_ALLOCATION_ALIGNMENT,
                                bool device_exposed   = true) noexcept;

  /**
   * @brief Selects the mechanism used by `allocate_host_writable`.
   *
   * For evaluating the alternatives; a released version would settle on one.
   *
   * @param mode The mechanism to use
   */
  void set_host_write_sync_mode(host_write_sync_mode mode) noexcept;

  /**
   * @brief Suppresses the shared per-stream event record on the host-writable deallocate path.
   *
   * Only for isolating the cost of that record. Unsafe in general: the device-side cross-stream
   * reuse logic relies on the per-stream event being current.
   *
   * @param skip Whether to skip the per-stream event record
   */
  void set_skip_stream_event_record(bool skip) noexcept;

  /**
   * @brief Sets how many events are cycled per stream for per-block completion tracking.
   *
   * @param size The number of events in the per-stream ring
   */
  void set_block_event_ring_size(std::size_t size) noexcept;

  /**
   * @brief Returns counters describing `allocate_host_writable` behavior.
   *
   * @return Counters accumulated since construction or the last reset
   */
  [[nodiscard]] host_writable_stats host_writable_statistics();

  /// Resets the counters returned by `host_writable_statistics`.
  void reset_host_writable_statistics();
};

static_assert(cuda::mr::resource_with<pool_memory_resource, cuda::mr::device_accessible>,
              "pool_memory_resource does not satisfy the cuda::mr::resource concept");

/** @} */  // end of group
}  // namespace mr
RMM_NAMESPACE_END
