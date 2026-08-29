/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/detail/export.hpp>

#include <cstddef>

RMM_NAMESPACE_BEGIN
namespace mr {

/**
 * @brief Mechanism used by `allocate_host_writable` to make a recycled block safe for the host
 * to write to.
 *
 * A pool can hand back a block whose previous owner still has a copy in flight on the stream it
 * was freed on. A host writer must not touch the block until that copy has completed. These modes
 * differ only in how precisely they identify the work that must complete.
 */
enum class host_write_sync_mode {
  /**
   * @brief Synchronize the whole allocating stream.
   *
   * This is what callers must do today with the public `allocate`. Correct, but waits for all work
   * queued on the stream, including work unrelated to the block and queued after it was freed.
   */
  stream_sync,

  /**
   * @brief Wait on the pool's existing per-stream event.
   *
   * The pool already records one event per stream on every deallocation, so this adds no new
   * machinery. The event's position is that of the most recent free on the stream, so this skips
   * any work queued after that free, but not work queued before it.
   */
  stream_event,

  /**
   * @brief Wait on an event recorded for the specific block being recycled.
   *
   * Gives the minimal wait, at the cost of an additional `cudaEventRecord` per deallocation.
   */
  block_event,

  /**
   * @brief Skip the wait entirely for blocks never exposed to the device.
   *
   * A block that came straight from upstream, or whose previous owner declared it host-only via
   * `deallocate_host_writable`, needs no wait at all. Falls back to `block_event` otherwise.
   */
  clean_tracking,
};

/**
 * @brief Counters describing how `allocate_host_writable` behaved, for evaluating the modes.
 */
struct host_writable_stats {
  std::size_t allocations{};          ///< Calls to `allocate_host_writable`.
  std::size_t waits{};                ///< Calls that had to wait on an event or stream.
  std::size_t fast_path{};            ///< Calls that needed no wait at all.
  std::size_t query_short_circuit{};  ///< Waits skipped because the event had already completed.
  std::size_t event_records{};        ///< `cudaEventRecord` calls made on the deallocate path.
  std::size_t events_created{};       ///< CUDA events created for per-block tracking.
};

}  // namespace mr
RMM_NAMESPACE_END
