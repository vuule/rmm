/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file Measures the cost of obtaining pinned host memory that is safe for the host to write to.
 *
 * libcudf's `rmm_host_allocator::allocate` synchronizes the whole stream on every allocation so
 * that `thrust::host_vector` can initialize the memory. The actual requirement is narrower: the
 * pinned pool can hand back a block whose previous owner still has a copy in flight, and the host
 * must not write to it until that copy completes. This benchmark compares the stream
 * synchronization against the narrower waits implemented by `allocate_host_writable`.
 *
 * Two regimes are reported separately because they answer different questions:
 *
 *  - Steady state: allocate/free in a loop with the stream either idle or carrying a small kernel.
 *    This measures the fixed per-call overhead.
 *  - Queued backlog: a single allocation measured while the stream carries work that the recycled
 *    block does not depend on, enqueued after that block was freed. This is the case the exercise
 *    is about, where the stream synchronization waits for unbounded unrelated work.
 */

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/host_writable.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;
using rmm::mr::host_write_sync_mode;

/// Occupies a stream without relying on wall-clock calibration being exact.
__global__ void spin_kernel(std::int64_t cycles)
{
  auto const start = clock64();
  while (clock64() - start < cycles) {}
}

/// Roughly 1 GHz on the clock64 counter, so cycles are approximately nanoseconds.
constexpr std::int64_t cycles_per_ms = 1'000'000;

double to_us(clock_type::duration duration)
{
  return std::chrono::duration<double, std::micro>(duration).count();
}

/// Returns the median, which is more robust than the mean to the occasional scheduling outlier.
double median(std::vector<double> values)
{
  if (values.empty()) { return 0.0; }
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

/// The strategy used to obtain host-writable memory, including the two reference points.
enum class strategy {
  malloc_floor,        ///< Plain `malloc`, for scale. No pinning, no pool, no synchronization.
  pool_no_wait,        ///< Pool `allocate` with no wait at all. Unsafe; the achievable floor.
  stream_sync,         ///< What libcudf does today: `allocate` plus `cudaStreamSynchronize`.
  stream_event,        ///< Wait on the pool's existing per-stream event.
  block_event,         ///< Wait on an event recorded for the specific block.
  block_event_single,  ///< As `block_event`, without the redundant per-stream record.
  clean_host_only,     ///< Declare blocks host-only on free, so no wait is ever needed.
};

char const* name_of(strategy kind)
{
  switch (kind) {
    case strategy::malloc_floor: return "malloc (reference)";
    case strategy::pool_no_wait: return "pool, no wait (unsafe floor)";
    case strategy::stream_sync: return "allocate + stream sync (today)";
    case strategy::stream_event: return "per-stream event";
    case strategy::block_event: return "per-block event";
    case strategy::block_event_single: return "per-block event, 1 record";
    case strategy::clean_host_only: return "clean tracking, host-only";
  }
  return "?";
}

/// Whether the strategy tracks device exposure and can therefore skip the wait entirely.
bool is_host_only(strategy kind) { return kind == strategy::clean_host_only; }

void configure(rmm::mr::pool_memory_resource& pool, strategy kind)
{
  pool.set_skip_stream_event_record(kind == strategy::block_event_single);
  switch (kind) {
    case strategy::stream_sync:
      pool.set_host_write_sync_mode(host_write_sync_mode::stream_sync);
      break;
    case strategy::stream_event:
      pool.set_host_write_sync_mode(host_write_sync_mode::stream_event);
      break;
    case strategy::block_event:
    case strategy::block_event_single:
      pool.set_host_write_sync_mode(host_write_sync_mode::block_event);
      break;
    case strategy::clean_host_only:
      pool.set_host_write_sync_mode(host_write_sync_mode::clean_tracking);
      break;
    default: break;
  }
}

/// Timings for one configuration, in microseconds.
struct result {
  double allocate{};
  double deallocate{};
};

/**
 * @brief Measures per-call allocate and deallocate cost in a steady-state loop.
 *
 * Each iteration allocates, copies the block to the device (so it is genuinely device-exposed),
 * frees it, and then optionally queues a small kernel. The kernel is queued *after* the free so
 * that the next allocation does not depend on it, which is the ordering that distinguishes a
 * targeted wait from a stream synchronization.
 */
result run_steady_state(rmm::mr::pool_memory_resource& pool,
                        rmm::cuda_stream const& stream,
                        void* device_dst,
                        strategy kind,
                        std::size_t bytes,
                        bool queue_small_kernel,
                        std::size_t iterations)
{
  configure(pool, kind);

  std::vector<double> allocate_times;
  std::vector<double> deallocate_times;
  allocate_times.reserve(iterations);
  deallocate_times.reserve(iterations);

  for (std::size_t i = 0; i < iterations; ++i) {
    void* ptr{};

    auto const alloc_start = clock_type::now();
    switch (kind) {
      case strategy::malloc_floor: ptr = std::malloc(bytes); break;
      case strategy::pool_no_wait:
        ptr = pool.allocate(stream.view(), bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
        break;
      default: ptr = pool.allocate_host_writable(stream.view(), bytes); break;
    }
    auto const alloc_end = clock_type::now();

    // A host writer would touch the memory here; do the same so the comparison is honest.
    std::memset(ptr, 0x5A, bytes);

    if (kind != strategy::malloc_floor) {
      RMM_CUDA_TRY(cudaMemcpyAsync(device_dst, ptr, bytes, cudaMemcpyHostToDevice, stream.value()));
    }

    auto const free_start = clock_type::now();
    switch (kind) {
      case strategy::malloc_floor: std::free(ptr); break;
      case strategy::pool_no_wait:
        pool.deallocate(stream.view(), ptr, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
        break;
      default:
        pool.deallocate_host_writable(
          stream.view(), ptr, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT, !is_host_only(kind));
        break;
    }
    auto const free_end = clock_type::now();

    if (queue_small_kernel && kind != strategy::malloc_floor) {
      spin_kernel<<<1, 1, 0, stream.value()>>>(cycles_per_ms / 100);  // ~10us
    } else if (!queue_small_kernel) {
      // Keep the stream genuinely idle for the next iteration, outside the timed region.
      RMM_CUDA_TRY(cudaStreamSynchronize(stream.value()));
    }

    // Discard the first few iterations, which include pool growth and event creation.
    if (i >= iterations / 4) {
      allocate_times.push_back(to_us(alloc_end - alloc_start));
      deallocate_times.push_back(to_us(free_end - free_start));
    }
  }

  RMM_CUDA_TRY(cudaStreamSynchronize(stream.value()));
  return {median(std::move(allocate_times)), median(std::move(deallocate_times))};
}

/**
 * @brief Measures one allocation while the stream carries a backlog the block does not depend on.
 *
 * Each trial primes a recyclable dirty block, enqueues `backlog_ms` of kernel work *after* that
 * block was freed, and then times a single allocation. A correct implementation only needs to wait
 * for the primed block's copy, which is already at the head of the queue; synchronizing the stream
 * instead waits for the whole backlog.
 */
double run_queued_backlog(rmm::mr::pool_memory_resource& pool,
                          rmm::cuda_stream const& stream,
                          void* device_dst,
                          strategy kind,
                          std::size_t bytes,
                          std::int64_t backlog_ms,
                          std::size_t trials)
{
  configure(pool, kind);

  std::vector<double> times;
  times.reserve(trials);

  for (std::size_t i = 0; i < trials; ++i) {
    // Prime a block that the device has been given, and free it so it can be recycled.
    void* primed{};
    if (kind == strategy::malloc_floor) {
      primed = std::malloc(bytes);
      std::free(primed);
    } else {
      primed = (kind == strategy::pool_no_wait)
                 ? pool.allocate(stream.view(), bytes, rmm::CUDA_ALLOCATION_ALIGNMENT)
                 : pool.allocate_host_writable(stream.view(), bytes);
      std::memset(primed, 0x5A, bytes);
      RMM_CUDA_TRY(
        cudaMemcpyAsync(device_dst, primed, bytes, cudaMemcpyHostToDevice, stream.value()));
      if (kind == strategy::pool_no_wait) {
        pool.deallocate(stream.view(), primed, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
      } else {
        pool.deallocate_host_writable(
          stream.view(), primed, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT, !is_host_only(kind));
      }
    }

    // The backlog is enqueued after the free, so the recycled block does not depend on it.
    if (kind != strategy::malloc_floor) {
      spin_kernel<<<1, 1, 0, stream.value()>>>(backlog_ms * cycles_per_ms);
    }

    void* ptr{};
    auto const start = clock_type::now();
    switch (kind) {
      case strategy::malloc_floor: ptr = std::malloc(bytes); break;
      case strategy::pool_no_wait:
        ptr = pool.allocate(stream.view(), bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
        break;
      default: ptr = pool.allocate_host_writable(stream.view(), bytes); break;
    }
    auto const end = clock_type::now();
    std::memset(ptr, 0xA5, bytes);
    times.push_back(to_us(end - start));

    if (kind == strategy::malloc_floor) {
      std::free(ptr);
    } else {
      RMM_CUDA_TRY(cudaStreamSynchronize(stream.value()));
      pool.deallocate_host_writable(stream.view(), ptr, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
      RMM_CUDA_TRY(cudaStreamSynchronize(stream.value()));
    }
  }

  return median(std::move(times));
}

/**
 * @brief Measures an allocation when the backlog was enqueued *between* two frees.
 *
 * This is the only shape in which a per-block event beats the pool's existing per-stream event.
 * The per-stream event is re-recorded on every free, so after a later free its position is past
 * the backlog, and waiting on it waits for the backlog. An event belonging to the earlier-freed
 * block is still positioned before the backlog.
 *
 * A separator allocation is kept alive between the two blocks so they cannot coalesce; if they
 * did, the merged block would carry only the later event and the distinction would vanish.
 */
double run_interleaved_frees(rmm::mr::pool_memory_resource& pool,
                             rmm::cuda_stream const& stream,
                             void* device_dst,
                             strategy kind,
                             std::size_t bytes,
                             std::int64_t backlog_ms,
                             std::size_t trials)
{
  configure(pool, kind);

  // The block allocated at the end must be a better fit for `bytes` than the later-freed block.
  auto const larger_bytes = bytes * 4;

  std::vector<double> times;
  times.reserve(trials);

  for (std::size_t i = 0; i < trials; ++i) {
    auto* early = pool.allocate_host_writable(stream.view(), bytes);
    auto* sep   = pool.allocate_host_writable(stream.view(), bytes);
    auto* late  = pool.allocate_host_writable(stream.view(), larger_bytes);

    std::memset(early, 0x5A, bytes);
    std::memset(late, 0x5A, larger_bytes);
    RMM_CUDA_TRY(cudaMemcpyAsync(device_dst, early, bytes, cudaMemcpyHostToDevice, stream.value()));
    RMM_CUDA_TRY(
      cudaMemcpyAsync(device_dst, late, larger_bytes, cudaMemcpyHostToDevice, stream.value()));

    pool.deallocate_host_writable(
      stream.view(), early, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT, !is_host_only(kind));

    // Enqueued between the two frees: after `early`'s event, before `late`'s.
    spin_kernel<<<1, 1, 0, stream.value()>>>(backlog_ms * cycles_per_ms);

    pool.deallocate_host_writable(
      stream.view(), late, larger_bytes, rmm::CUDA_ALLOCATION_ALIGNMENT, !is_host_only(kind));

    void* ptr{};
    auto const start = clock_type::now();
    ptr = (kind == strategy::stream_sync) ? pool.allocate_host_writable(stream.view(), bytes)
                                          : pool.allocate_host_writable(stream.view(), bytes);
    auto const end = clock_type::now();
    std::memset(ptr, 0xA5, bytes);
    times.push_back(to_us(end - start));

    RMM_CUDA_TRY(cudaStreamSynchronize(stream.value()));
    pool.deallocate_host_writable(stream.view(), ptr, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
    pool.deallocate_host_writable(stream.view(), sep, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
    RMM_CUDA_TRY(cudaStreamSynchronize(stream.value()));
  }

  return median(std::move(times));
}

std::string format_size(std::size_t bytes)
{
  if (bytes >= (std::size_t{1} << 20)) { return std::to_string(bytes >> 20) + " MiB"; }
  if (bytes >= (std::size_t{1} << 10)) { return std::to_string(bytes >> 10) + " KiB"; }
  return std::to_string(bytes) + " B";
}

}  // namespace

int main()
{
  constexpr std::size_t pool_size    = std::size_t{512} << 20;
  constexpr std::size_t iterations   = 2000;
  constexpr std::size_t heavy_trials = 10;
  constexpr std::int64_t backlog_ms  = 200;

  std::vector<std::size_t> const sizes{512, std::size_t{4} << 20};
  std::vector<strategy> const strategies{strategy::malloc_floor,
                                         strategy::pool_no_wait,
                                         strategy::stream_sync,
                                         strategy::stream_event,
                                         strategy::block_event,
                                         strategy::block_event_single,
                                         strategy::clean_host_only};

  void* device_dst{};
  RMM_CUDA_TRY(cudaMalloc(&device_dst, sizes.back()));

  std::printf("Pinned host allocation, host-writable on return. Median of %zu iterations.\n",
              iterations);
  std::printf("All times in microseconds.\n\n");

  for (auto bytes : sizes) {
    std::printf("=== size %s ===\n", format_size(bytes).c_str());
    std::printf("%-32s %12s %12s %12s %12s\n",
                "strategy",
                "alloc idle",
                "free idle",
                "alloc small",
                "free small");
    for (auto kind : strategies) {
      // A fresh pool per configuration so pool state cannot leak between them.
      rmm::mr::pinned_host_memory_resource upstream{};
      rmm::mr::pool_memory_resource pool{upstream, pool_size};
      rmm::cuda_stream stream{};

      auto const idle  = run_steady_state(pool, stream, device_dst, kind, bytes, false, iterations);
      auto const small = run_steady_state(pool, stream, device_dst, kind, bytes, true, iterations);

      std::printf("%-32s %12.2f %12.2f %12.2f %12.2f\n",
                  name_of(kind),
                  idle.allocate,
                  idle.deallocate,
                  small.allocate,
                  small.deallocate);
    }
    std::printf("\n");
  }

  std::printf("=== queued backlog: %lld ms of kernel work enqueued after the free ===\n",
              static_cast<long long>(backlog_ms));
  std::printf("%-32s %16s %16s\n", "strategy", "alloc 512 B", "alloc 4 MiB");
  for (auto kind : strategies) {
    std::vector<double> per_size;
    for (auto bytes : sizes) {
      rmm::mr::pinned_host_memory_resource upstream{};
      rmm::mr::pool_memory_resource pool{upstream, pool_size};
      rmm::cuda_stream stream{};
      per_size.push_back(
        run_queued_backlog(pool, stream, device_dst, kind, bytes, backlog_ms, heavy_trials));
    }
    std::printf("%-32s %16.2f %16.2f\n", name_of(kind), per_size[0], per_size[1]);
  }
  std::printf("\n");

  std::printf("=== backlog enqueued between two frees (recycles the earlier-freed block) ===\n");
  std::printf("%-32s %16s\n", "strategy", "alloc 512 B");
  for (auto kind : {strategy::stream_sync,
                    strategy::stream_event,
                    strategy::block_event,
                    strategy::block_event_single}) {
    rmm::mr::pinned_host_memory_resource upstream{};
    rmm::mr::pool_memory_resource pool{upstream, pool_size};
    rmm::cuda_stream stream{};
    auto const time =
      run_interleaved_frees(pool, stream, device_dst, kind, 512, backlog_ms, heavy_trials);
    std::printf("%-32s %16.2f\n", name_of(kind), time);
  }
  std::printf("\n");

  // Fast-path hit rate for the device-exposure tracking option, under an alternating workload.
  {
    rmm::mr::pinned_host_memory_resource upstream{};
    rmm::mr::pool_memory_resource pool{upstream, pool_size};
    rmm::cuda_stream stream{};
    pool.set_host_write_sync_mode(host_write_sync_mode::clean_tracking);
    pool.reset_host_writable_statistics();

    constexpr std::size_t bytes = std::size_t{64} << 10;
    for (std::size_t i = 0; i < iterations; ++i) {
      bool const expose = (i % 4 != 0);  // three quarters of buffers are handed to the device
      auto* ptr         = pool.allocate_host_writable(stream.view(), bytes);
      std::memset(ptr, 0x5A, bytes);
      if (expose) {
        RMM_CUDA_TRY(
          cudaMemcpyAsync(device_dst, ptr, bytes, cudaMemcpyHostToDevice, stream.value()));
      }
      pool.deallocate_host_writable(
        stream.view(), ptr, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT, expose);
    }
    RMM_CUDA_TRY(cudaStreamSynchronize(stream.value()));

    auto const stats = pool.host_writable_statistics();
    std::printf("=== clean tracking, 25%% of buffers never exposed to the device ===\n");
    std::printf("allocations           %zu\n", stats.allocations);
    std::printf(
      "no wait needed        %zu (%.1f%%)\n",
      stats.fast_path,
      100.0 * static_cast<double>(stats.fast_path) / static_cast<double>(stats.allocations));
    std::printf("waited                %zu\n", stats.waits);
    std::printf("  already complete    %zu\n", stats.query_short_circuit);
    std::printf("event records on free %zu\n", stats.event_records);
    std::printf("cuda events created   %zu\n", stats.events_created);
  }

  RMM_CUDA_TRY(cudaFree(device_dst));
  return 0;
}
