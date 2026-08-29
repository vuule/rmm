/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/detail/error.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/process_is_exiting.hpp>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <vector>

RMM_NAMESPACE_BEGIN
namespace mr::detail {

/**
 * @brief A recycling pool of CUDA events created with `cudaEventDisableTiming`.
 *
 * Event creation is expensive enough that a per-block event scheme is only viable if events are
 * reused. Callers `acquire()` an event, record it, and `release()` it once they have synchronized
 * on it. Events are never destroyed until the pool is destroyed, so the high-water mark of
 * simultaneously-held events bounds the number of live CUDA events.
 *
 * Not thread safe; callers must hold their own lock.
 */
class event_pool {
 public:
  event_pool() = default;

  ~event_pool()
  {
    if (rmm::process_is_exiting()) { return; }
    for (auto event : all_events_) {
      RMM_ASSERT_CUDA_SUCCESS_SAFE_SHUTDOWN(cudaEventSynchronize(event));
      RMM_ASSERT_CUDA_SUCCESS_SAFE_SHUTDOWN(cudaEventDestroy(event));
    }
  }

  event_pool(event_pool const&)            = delete;
  event_pool& operator=(event_pool const&) = delete;
  event_pool(event_pool&&)                 = delete;
  event_pool& operator=(event_pool&&)      = delete;

  /// Returns an event that is not currently held by any caller.
  [[nodiscard]] cudaEvent_t acquire()
  {
    if (!available_.empty()) {
      auto event = available_.back();
      available_.pop_back();
      return event;
    }
    cudaEvent_t event{};
    RMM_ASSERT_CUDA_SUCCESS(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    all_events_.push_back(event);
    return event;
  }

  /// Returns `event` to the pool for reuse. The caller must have synchronized on it.
  void release(cudaEvent_t event) noexcept
  {
    if (event != nullptr) { available_.push_back(event); }
  }

  /// The number of events this pool has created, i.e. the high-water mark of concurrent use.
  [[nodiscard]] std::size_t events_created() const noexcept { return all_events_.size(); }

 private:
  std::vector<cudaEvent_t> all_events_;  ///< Every event created, for destruction.
  std::vector<cudaEvent_t> available_;   ///< Events not currently held by a caller.
};

}  // namespace mr::detail
RMM_NAMESPACE_END
