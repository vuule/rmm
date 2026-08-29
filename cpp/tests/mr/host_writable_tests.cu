/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/host_writable.hpp>
#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

/// Occupies a stream for a while without depending on wall-clock calibration being exact.
__global__ void spin_kernel(std::int64_t cycles)
{
  auto const start = clock64();
  while (clock64() - start < cycles) {}
}

constexpr std::int64_t spin_cycles    = 100'000'000;  // ~100ms at ~1GHz
constexpr std::uint8_t first_pattern  = 0xAB;
constexpr std::uint8_t second_pattern = 0xCD;
constexpr std::size_t initial_pool    = std::size_t{4} << 20;
constexpr std::size_t transfer_bytes  = std::size_t{256} << 10;

class HostWritableTest : public ::testing::Test {
 protected:
  rmm::mr::pinned_host_memory_resource upstream_{};
  rmm::mr::pool_memory_resource pool_{upstream_, initial_pool};
  rmm::cuda_stream stream_{};

  /**
   * @brief Drives the use-after-free hazard and returns what the device actually received.
   *
   * Fills a pinned block from the host, queues a long kernel and then a copy of that block to the
   * device, frees the block, immediately reallocates the same size, and overwrites it from the host
   * with a different pattern. The copy has not run yet when the host overwrites, so unless the
   * reallocation waited for it, the device observes the second pattern.
   *
   * @param host_writable Whether to allocate through `allocate_host_writable` (the prototype) or
   * plain `allocate` (the hazard, with the wait removed).
   * @return The first byte the device received.
   */
  std::uint8_t run_hazard(bool host_writable)
  {
    void* device_dst{};
    RMM_CUDA_TRY(cudaMalloc(&device_dst, transfer_bytes));
    RMM_CUDA_TRY(cudaMemset(device_dst, 0, transfer_bytes));

    auto* first = static_cast<std::uint8_t*>(
      host_writable
        ? pool_.allocate_host_writable(stream_.view(), transfer_bytes)
        : pool_.allocate(stream_.view(), transfer_bytes, rmm::CUDA_ALLOCATION_ALIGNMENT));
    std::memset(first, first_pattern, transfer_bytes);

    // Put the copy behind a long kernel so it is still pending when the host overwrites below.
    spin_kernel<<<1, 1, 0, stream_.value()>>>(spin_cycles);
    RMM_CUDA_TRY(
      cudaMemcpyAsync(device_dst, first, transfer_bytes, cudaMemcpyHostToDevice, stream_.value()));

    if (host_writable) {
      pool_.deallocate_host_writable(stream_.view(), first, transfer_bytes);
    } else {
      pool_.deallocate(stream_.view(), first, transfer_bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
    }

    auto* second = static_cast<std::uint8_t*>(
      host_writable
        ? pool_.allocate_host_writable(stream_.view(), transfer_bytes)
        : pool_.allocate(stream_.view(), transfer_bytes, rmm::CUDA_ALLOCATION_ALIGNMENT));
    // The pool must recycle the same block for this to test anything.
    EXPECT_EQ(second, first);
    std::memset(second, second_pattern, transfer_bytes);

    std::vector<std::uint8_t> received(transfer_bytes, 0);
    RMM_CUDA_TRY(cudaStreamSynchronize(stream_.value()));
    RMM_CUDA_TRY(cudaMemcpy(received.data(), device_dst, transfer_bytes, cudaMemcpyDeviceToHost));

    if (host_writable) {
      pool_.deallocate_host_writable(stream_.view(), second, transfer_bytes);
    } else {
      pool_.deallocate(stream_.view(), second, transfer_bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
    }
    RMM_CUDA_TRY(cudaStreamSynchronize(stream_.value()));
    RMM_CUDA_TRY(cudaFree(device_dst));

    // Every byte is written with one pattern, so the first byte characterizes the result.
    return received[0];
  }
};

/// Confirms the hazard is real: with no wait, the device receives the host's later write.
TEST_F(HostWritableTest, HazardManifestsWithoutWait)
{
  EXPECT_EQ(run_hazard(false), second_pattern);
}

TEST_F(HostWritableTest, StreamSyncClosesHazard)
{
  pool_.set_host_write_sync_mode(rmm::mr::host_write_sync_mode::stream_sync);
  EXPECT_EQ(run_hazard(true), first_pattern);
}

TEST_F(HostWritableTest, StreamEventClosesHazard)
{
  pool_.set_host_write_sync_mode(rmm::mr::host_write_sync_mode::stream_event);
  EXPECT_EQ(run_hazard(true), first_pattern);
}

TEST_F(HostWritableTest, BlockEventClosesHazard)
{
  pool_.set_host_write_sync_mode(rmm::mr::host_write_sync_mode::block_event);
  EXPECT_EQ(run_hazard(true), first_pattern);
}

TEST_F(HostWritableTest, CleanTrackingClosesHazard)
{
  pool_.set_host_write_sync_mode(rmm::mr::host_write_sync_mode::clean_tracking);
  EXPECT_EQ(run_hazard(true), first_pattern);
}

/// A block declared host-only must still be handed back without a wait, and must be intact.
TEST_F(HostWritableTest, CleanTrackingSkipsWaitForHostOnlyBlocks)
{
  pool_.set_host_write_sync_mode(rmm::mr::host_write_sync_mode::clean_tracking);
  pool_.reset_host_writable_statistics();

  constexpr std::size_t iterations = 32;
  for (std::size_t i = 0; i < iterations; ++i) {
    auto* ptr =
      static_cast<std::uint8_t*>(pool_.allocate_host_writable(stream_.view(), transfer_bytes));
    std::memset(ptr, first_pattern, transfer_bytes);
    EXPECT_EQ(ptr[0], first_pattern);
    EXPECT_EQ(ptr[transfer_bytes - 1], first_pattern);
    // Never handed to the device, so the next host writer needs no wait.
    pool_.deallocate_host_writable(
      stream_.view(), ptr, transfer_bytes, rmm::CUDA_ALLOCATION_ALIGNMENT, false);
  }

  auto const stats = pool_.host_writable_statistics();
  EXPECT_EQ(stats.allocations, iterations);
  // The initial pool block is held in the legacy stream's free list, so the very first allocation
  // on this stream is a cross-stream steal and waits on that stream's event. Every subsequent
  // allocation is served by a host-only free on our own stream and needs no wait.
  EXPECT_EQ(stats.waits, 1u);
  EXPECT_EQ(stats.fast_path, iterations - 1);
  // Host-only frees record nothing at all.
  EXPECT_EQ(stats.event_records, 0u);
}

/// Interleaving device-exposed and host-only frees must keep the wait only where it is needed.
TEST_F(HostWritableTest, CleanTrackingWaitsOnlyForExposedBlocks)
{
  pool_.set_host_write_sync_mode(rmm::mr::host_write_sync_mode::clean_tracking);

  void* device_dst{};
  RMM_CUDA_TRY(cudaMalloc(&device_dst, transfer_bytes));

  constexpr std::size_t iterations = 16;
  pool_.reset_host_writable_statistics();
  for (std::size_t i = 0; i < iterations; ++i) {
    bool const expose = (i % 2 == 0);
    auto* ptr =
      static_cast<std::uint8_t*>(pool_.allocate_host_writable(stream_.view(), transfer_bytes));
    std::memset(ptr, static_cast<int>(i), transfer_bytes);
    if (expose) {
      RMM_CUDA_TRY(
        cudaMemcpyAsync(device_dst, ptr, transfer_bytes, cudaMemcpyHostToDevice, stream_.value()));
    }
    pool_.deallocate_host_writable(
      stream_.view(), ptr, transfer_bytes, rmm::CUDA_ALLOCATION_ALIGNMENT, expose);
  }
  RMM_CUDA_TRY(cudaStreamSynchronize(stream_.value()));

  auto const stats = pool_.host_writable_statistics();
  EXPECT_EQ(stats.allocations, iterations);
  EXPECT_EQ(stats.allocations, stats.waits + stats.fast_path);
  // Half the frees were host-only and recorded nothing.
  EXPECT_EQ(stats.event_records, iterations);
  RMM_CUDA_TRY(cudaFree(device_dst));
}

/// The prototype must not corrupt data across many recycles under real device traffic.
TEST_F(HostWritableTest, RepeatedRecycleUnderDeviceTrafficPreservesData)
{
  for (auto mode : {rmm::mr::host_write_sync_mode::stream_event,
                    rmm::mr::host_write_sync_mode::block_event,
                    rmm::mr::host_write_sync_mode::clean_tracking}) {
    pool_.set_host_write_sync_mode(mode);

    void* device_dst{};
    RMM_CUDA_TRY(cudaMalloc(&device_dst, transfer_bytes));

    constexpr std::size_t iterations = 64;
    for (std::size_t i = 0; i < iterations; ++i) {
      auto const pattern = static_cast<std::uint8_t>(i & 0xFF);
      auto* ptr =
        static_cast<std::uint8_t*>(pool_.allocate_host_writable(stream_.view(), transfer_bytes));
      std::memset(ptr, pattern, transfer_bytes);
      RMM_CUDA_TRY(
        cudaMemcpyAsync(device_dst, ptr, transfer_bytes, cudaMemcpyHostToDevice, stream_.value()));

      std::vector<std::uint8_t> received(transfer_bytes, 0);
      RMM_CUDA_TRY(cudaStreamSynchronize(stream_.value()));
      RMM_CUDA_TRY(cudaMemcpy(received.data(), device_dst, transfer_bytes, cudaMemcpyDeviceToHost));
      ASSERT_EQ(received.front(), pattern) << "mode " << static_cast<int>(mode) << " iter " << i;
      ASSERT_EQ(received.back(), pattern) << "mode " << static_cast<int>(mode) << " iter " << i;

      pool_.deallocate_host_writable(stream_.view(), ptr, transfer_bytes);
    }
    RMM_CUDA_TRY(cudaStreamSynchronize(stream_.value()));
    RMM_CUDA_TRY(cudaFree(device_dst));
  }
}

}  // namespace
