/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/detail/export.hpp>
#include <rmm/detail/format.hpp>
#include <rmm/logger.hpp>
#include <rmm/mr/detail/event_pool.hpp>
#include <rmm/mr/host_writable.hpp>
#include <rmm/process_is_exiting.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>
#ifdef RMM_DEBUG_PRINT
#include <iostream>
#endif

RMM_NAMESPACE_BEGIN
namespace mr::detail {

/**
 * @brief A CRTP helper function
 *
 * https://www.fluentcpp.com/2017/05/19/crtp-helper/
 *
 * Does two things:
 * 1. Makes "crtp" explicit in the inheritance structure of a CRTP base class.
 * 2. Avoids having to `static_cast` in a lot of places
 *
 * @tparam T The derived class in a CRTP hierarchy
 */
template <typename T>
struct crtp {
  [[nodiscard]] T& underlying() { return static_cast<T&>(*this); }
  [[nodiscard]] T const& underlying() const { return static_cast<T const&>(*this); }
};

/**
 * @brief Whether a block type can carry a completion event for host-writable allocation.
 *
 * `coalescing_free_list`'s block does; the plain `block_base` used by fixed-size free lists does
 * not, so pools built on it fall back to the per-stream event.
 */
template <typename BlockType>
concept tracks_free_event = requires(BlockType& block, cudaEvent_t event, std::uint64_t seq) {
  block.set_free_event(event, seq);
  { std::as_const(block).free_event() } -> std::convertible_to<cudaEvent_t>;
};

/**
 * @brief Base class for a stream-ordered memory resource
 *
 * This base class uses CRTP (https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern)
 * to provide static polymorphism to enable defining suballocator resources that maintain separate
 * pools per stream. All of the stream-ordering logic is contained in this class, but the logic
 * to determine how memory pools are managed and the type of allocation is implemented in a derived
 * class and in a free list class.
 *
 * For example, a coalescing pool memory resource uses a coalescing_free_list and maintains data
 * structures for allocated blocks and has functions to allocate and free blocks and to expand the
 * pool.
 *
 * Classes derived from stream_ordered_memory_resource must implement the following four methods,
 * documented separately:
 *
 * 1. `std::size_t get_maximum_allocation_size() const`
 * 2. `block_type expand_pool(std::size_t size, free_list& blocks, cuda_stream_view stream)`
 * 3. `split_block allocate_from_block(block_type const& b, std::size_t size)`
 * 4. `block_type free_block(void* ptr, std::size_t size) noexcept`
 */
template <typename PoolResource, typename FreeListType>
class stream_ordered_memory_resource : public crtp<PoolResource> {
 public:
  ~stream_ordered_memory_resource() { release(); }

  stream_ordered_memory_resource()                                                 = default;
  stream_ordered_memory_resource(stream_ordered_memory_resource const&)            = delete;
  stream_ordered_memory_resource(stream_ordered_memory_resource&&)                 = delete;
  stream_ordered_memory_resource& operator=(stream_ordered_memory_resource const&) = delete;
  stream_ordered_memory_resource& operator=(stream_ordered_memory_resource&&)      = delete;

  /**
   * @brief Allocates memory of size at least `bytes` bytes.
   *
   * The returned pointer has at least 256B alignment.
   *
   * @throws `std::bad_alloc` if the requested allocation could not be fulfilled
   *
   * @param stream The stream in which to order this allocation
   * @param bytes The size in bytes of the allocation
   * @param alignment Unused; alignment is always at least `CUDA_ALLOCATION_ALIGNMENT`
   * @return void* Pointer to the newly allocated memory
   */
  [[nodiscard]] void* allocate(cuda::stream_ref stream,
                               std::size_t bytes,
                               std::size_t /*alignment*/)
  {
    auto const strm = cuda_stream_view{stream};

    RMM_LOG_TRACE("[A][stream %s][%zuB]", rmm::detail::format_stream(strm), bytes);

    if (bytes == 0) { return nullptr; }

    lock_guard lock(mtx_);

    auto stream_event = get_event(strm);

    bytes = rmm::align_up(bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
    RMM_EXPECTS(bytes <= this->underlying().get_maximum_allocation_size(),
                std::string("Maximum allocation size exceeded (failed to allocate ") +
                  rmm::detail::format_bytes(bytes) + ")",
                rmm::out_of_memory);
    auto const block = this->underlying().get_block(bytes, stream_event);

    RMM_LOG_TRACE("[A][stream %s][%zuB][%p]",
                  rmm::detail::format_stream(stream_event.stream),
                  bytes,
                  block.pointer());

    log_summary_trace();

    return block.pointer();
  }

  /**
   * @brief Deallocate memory pointed to by `ptr`.
   *
   * @param stream The stream in which to order this deallocation
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation to deallocate
   * @param alignment Unused
   */
  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t /*alignment*/) noexcept
  {
    auto const strm = cuda_stream_view{stream};

    RMM_LOG_TRACE("[D][stream %s][%zuB][%p]", rmm::detail::format_stream(strm), bytes, ptr);

    if (bytes == 0 || ptr == nullptr) { return; }

    lock_guard lock(mtx_);
    auto stream_event = get_event(strm);

    bytes            = rmm::align_up(bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
    auto const block = this->underlying().free_block(ptr, bytes);

    // TODO: cudaEventRecord has significant overhead on deallocations. For the non-PTDS case
    // we may be able to delay recording the event in some situations. But using events rather
    // than streams allows stealing from deleted streams.
    RMM_ASSERT_CUDA_SUCCESS(cudaEventRecord(stream_event.event, strm.value()));

    stream_free_blocks_[stream_event].insert(block);

    log_summary_trace();
  }

  /**
   * @brief Allocates memory of size at least `bytes` synchronously.
   *
   * @throws `std::bad_alloc` if the requested allocation could not be fulfilled
   *
   * @param bytes The size in bytes of the allocation
   * @param alignment Alignment of the allocation
   * @return void* Pointer to the newly allocated memory
   */
  [[nodiscard]] void* allocate_sync(std::size_t bytes,
                                    std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT)
  {
    auto const stream = cuda::stream_ref{cudaStream_t{nullptr}};
    void* ptr         = allocate(stream, bytes, alignment);
    RMM_CUDA_TRY(cudaStreamSynchronize(stream.get()));
    return ptr;
  }

  /**
   * @brief Deallocate memory pointed to by `ptr` synchronously.
   *
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation to deallocate
   * @param alignment Alignment of the allocation
   */
  void deallocate_sync(
    void* ptr,
    std::size_t bytes,
    [[maybe_unused]] std::size_t alignment = rmm::CUDA_ALLOCATION_ALIGNMENT) noexcept
  {
    auto const stream = cuda::stream_ref{cudaStream_t{nullptr}};
    deallocate(stream, ptr, bytes, alignment);
    RMM_ASSERT_CUDA_SUCCESS_SAFE_SHUTDOWN(cudaStreamSynchronize(stream.get()));
  }

  /**
   * @brief Allocates memory of at least `bytes` bytes that the host may write to on return.
   *
   * Equivalent to `allocate`, except that on return it is safe for the calling thread to write to
   * the returned memory without any further synchronization. A pool can recycle a block whose
   * previous owner still has a copy in flight on the stream it was freed on; this entry point
   * performs the minimal wait that guarantees such a copy has completed, rather than requiring the
   * caller to synchronize the whole stream.
   *
   * The wait is performed without holding the pool lock.
   *
   * @throws `std::bad_alloc` if the requested allocation could not be fulfilled
   *
   * @param stream The stream in which to order this allocation
   * @param bytes The size in bytes of the allocation
   * @param alignment Unused; alignment is always at least `CUDA_ALLOCATION_ALIGNMENT`
   * @return void* Pointer to memory the host may write to immediately
   */
  [[nodiscard]] void* allocate_host_writable(cuda::stream_ref stream,
                                             std::size_t bytes,
                                             std::size_t alignment)
  {
    if (bytes == 0) { return nullptr; }

    auto const strm = cuda_stream_view{stream};

    if (host_write_mode_ == host_write_sync_mode::stream_sync) {
      void* ptr = allocate(stream, bytes, alignment);
      {
        lock_guard lock(mtx_);
        ++hw_stats_.allocations;
        ++hw_stats_.waits;
      }
      RMM_CUDA_TRY(cudaStreamSynchronize(strm.value()));
      return ptr;
    }

    void* ptr{};
    cudaEvent_t wait_event{};
    {
      lock_guard lock(mtx_);

      auto const stream_event = get_event(strm);
      auto const aligned      = rmm::align_up(bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
      RMM_EXPECTS(aligned <= this->underlying().get_maximum_allocation_size(),
                  std::string("Maximum allocation size exceeded (failed to allocate ") +
                    rmm::detail::format_bytes(aligned) + ")",
                  rmm::out_of_memory);

      // `get_block` reports through `cross_stream_wait_event_` when it takes a block from another
      // stream, because in that case it only orders our *stream* behind the donor's event.
      cross_stream_wait_event_ = nullptr;
      auto const block         = this->underlying().get_block(aligned, stream_event);
      ptr                      = block.pointer();

      if (cross_stream_wait_event_ != nullptr) {
        wait_event = cross_stream_wait_event_;
      } else if constexpr (tracks_free_event<block_type>) {
        if (per_block_events_enabled()) {
          // Null when the block has never been freed, i.e. it came straight from upstream.
          wait_event = block.free_event();
        } else {
          wait_event = stream_event.event;
        }
      } else {
        // No new machinery: the per-stream event was last recorded at the most recent free on
        // this stream, which is at or after the free of this block.
        wait_event = stream_event.event;
      }

      ++hw_stats_.allocations;
      if (wait_event == nullptr) {
        ++hw_stats_.fast_path;
      } else {
        ++hw_stats_.waits;
      }
    }

    if (wait_event != nullptr) {
      // An event that has already completed still costs a driver round trip to synchronize on.
      // Query first so the common already-idle case stays off that path.
      if (cudaEventQuery(wait_event) == cudaSuccess) {
        lock_guard lock(mtx_);
        ++hw_stats_.query_short_circuit;
      } else {
        RMM_CUDA_TRY(cudaEventSynchronize(wait_event));
      }
    }
    return ptr;
  }

  /**
   * @brief Deallocates memory allocated by `allocate_host_writable`.
   *
   * @param stream The stream in which to order this deallocation
   * @param ptr Pointer to be deallocated
   * @param bytes The size in bytes of the allocation to deallocate
   * @param alignment Unused
   * @param device_exposed Whether this memory was ever used by device work on `stream`, e.g. as
   * the source or destination of a copy. When false, a subsequent host writer needs no wait at
   * all. Defaults to true, which is always safe.
   */
  void deallocate_host_writable(cuda::stream_ref stream,
                                void* ptr,
                                std::size_t bytes,
                                std::size_t /*alignment*/,
                                bool device_exposed = true) noexcept
  {
    if (bytes == 0 || ptr == nullptr) { return; }

    auto const strm = cuda_stream_view{stream};

    lock_guard lock(mtx_);
    auto const stream_event = get_event(strm);

    auto const aligned = rmm::align_up(bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
    auto block         = this->underlying().free_block(ptr, aligned);

    if constexpr (tracks_free_event<block_type>) {
      bool const track_clean = host_write_mode_ == host_write_sync_mode::clean_tracking;
      if (track_clean && !device_exposed) {
        // Nothing was ever queued against this block, so no event is needed and none is recorded.
        // Blocks it coalesces with keep their own events, which is what a later waiter needs.
        block.set_free_event(nullptr, 0);
        stream_free_blocks_[stream_event].insert(block);
        return;
      }
    }

    if (!skip_stream_event_record_) {
      RMM_ASSERT_CUDA_SUCCESS(cudaEventRecord(stream_event.event, strm.value()));
      ++hw_stats_.event_records;
    }

    if constexpr (tracks_free_event<block_type>) {
      if (per_block_events_enabled()) {
        auto const event = next_block_event(stream_event.event);
        RMM_ASSERT_CUDA_SUCCESS(cudaEventRecord(event, strm.value()));
        ++hw_stats_.event_records;
        block.set_free_event(event, ++free_seq_);
      }
    }

    stream_free_blocks_[stream_event].insert(block);
  }

  /// Selects the mechanism used by `allocate_host_writable`. For evaluating the alternatives.
  void set_host_write_sync_mode(host_write_sync_mode mode) noexcept { host_write_mode_ = mode; }

  /**
   * @brief Suppresses the shared per-stream `cudaEventRecord` on the host-writable deallocate path.
   *
   * Only for isolating the cost of that record. Unsafe in general: the device-side cross-stream
   * reuse logic relies on the per-stream event being current.
   */
  void set_skip_stream_event_record(bool skip) noexcept { skip_stream_event_record_ = skip; }

  /// Sets how many events are cycled per stream for per-block tracking.
  void set_block_event_ring_size(std::size_t size) noexcept
  {
    block_event_ring_size_ = std::max<std::size_t>(size, 1);
  }

  /// Returns counters describing `allocate_host_writable` behavior.
  [[nodiscard]] host_writable_stats host_writable_statistics()
  {
    lock_guard lock(mtx_);
    auto stats           = hw_stats_;
    stats.events_created = block_event_pool_.events_created();
    return stats;
  }

  /// Resets the counters returned by `host_writable_statistics`.
  void reset_host_writable_statistics()
  {
    lock_guard lock(mtx_);
    hw_stats_ = host_writable_stats{};
  }

 protected:
  using free_list  = FreeListType;
  using block_type = typename free_list::block_type;
  using lock_guard = std::lock_guard<std::mutex>;

  using stream_id_type = unsigned long long;  ///< Stream identifier returned by cudaStreamGetId
  // Derived classes must implement these four methods

  /*
   * @brief Get the maximum size of a single allocation supported by this suballocator memory
   * resource
   *
   * Default implementation is the maximum `std::size_t` value, but fixed-size allocators will have
   * a lower limit. Override this function in derived classes as necessary.
   *
   * @return std::size_t The maximum size of a single allocation supported by this memory resource
   */
  // std::size_t get_maximum_allocation_size() const

  /*
   * @brief Allocate space (typically from upstream) to supply the suballocation pool and return
   * a sufficiently sized block.
   *
   * This function returns a block because in some suballocators, a single block is allocated
   * from upstream and returned. In other suballocators, many blocks are created from upstream. In
   * the latter case, the function returns one block and inserts all the rest into the free list
   * `blocks`.
   *
   * @param size The minimum size block to return
   * @param blocks The free list into which to optionally insert new blocks
   * @param stream The stream on which the memory is to be used.
   * @return block_type a block of at least `size` bytes
   */
  // block_type expand_pool(std::size_t size, free_list& blocks, cuda_stream_view stream)

  /// Pair representing a block that has been split for allocation
  using split_block = std::pair<block_type, block_type>;

  /*
   * @brief Split block `b` if necessary to return a pointer to memory of `size` bytes.
   *
   * If the block is split, the remainder is returned as the remainder element in the output
   * `split_block`.
   *
   * @param b The block to allocate from.
   * @param size The size in bytes of the requested allocation.
   * @param stream_event The stream and associated event on which the allocation will be used.
   * @return A `split_block` comprising the allocated pointer and any unallocated remainder of the
   * input block.
   */
  // split_block allocate_from_block(block_type const& b, std::size_t size)

  /*
   * @brief Finds, frees and returns the block associated with pointer `ptr`.
   *
   * @param ptr The pointer to the memory to free.
   * @param size The size of the memory to free. Must be equal to the original allocation size.
   * @return The (now freed) block associated with `ptr`. The caller is expected to return the block
   * to the pool.
   */
  // block_type free_block(void* ptr, std::size_t size) noexcept

  /**
   * @brief Returns the block `b` (last used on stream `stream_event`) to the pool.
   *
   * @param block The block to insert into the pool.
   * @param stream The stream on which the memory was last used.
   */
  void insert_block(block_type const& block, cuda_stream_view stream)
  {
    stream_free_blocks_[get_event(stream)].insert(block);
  }

  void insert_blocks(free_list&& blocks, cuda_stream_view stream)
  {
    stream_free_blocks_[get_event(stream)].insert(std::move(blocks));
  }

#ifdef RMM_DEBUG_PRINT
  void print_free_blocks() const
  {
    std::cout << "stream free blocks: ";
    for (auto& free_blocks : stream_free_blocks_) {
      std::cout << "stream: " << free_blocks.first.stream << " event: " << free_blocks.first.event
                << " ";
      free_blocks.second.print();
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
#endif

  /**
   * @brief Get the mutex object
   *
   * @return std::mutex
   */
  std::mutex& get_mutex() { return mtx_; }

  struct stream_event_pair {
    cudaStream_t stream;
    cudaEvent_t event;

    bool operator<(stream_event_pair const& rhs) const { return event < rhs.event; }
  };

 private:
  /**
   * @brief get a unique CUDA event (possibly new) associated with `stream`
   *
   * The event is created on the first call, and it is not recorded. If compiled for per-thread
   * default stream and `stream` is the default stream, the event is created in thread local
   * memory and is unique per CPU thread.
   *
   * @param stream The stream for which to get an event.
   * @return The stream_event for `stream`.
   */
  stream_event_pair get_event(cuda_stream_view stream)
  {
    if (stream.is_per_thread_default()) {
      // Create a thread-local event for each device. These events are
      // deliberately leaked since the destructor needs to call into
      // the CUDA runtime and thread_local destructors (can) run below
      // main: it is undefined behaviour to call into the CUDA
      // runtime below main.
      thread_local std::vector<cudaEvent_t> events_tls(
        static_cast<std::size_t>(rmm::get_num_cuda_devices()));
      auto event = [device_id = this->device_id_]() {
        auto& e = events_tls[static_cast<std::size_t>(device_id.value())];
        if (!e) {
          // These events are deliberately not destructed and therefore live until
          // program exit.
          RMM_ASSERT_CUDA_SUCCESS(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        }
        return e;
      }();
      return stream_event_pair{stream.value(), event};
    }
    // We use cudaStreamLegacy as the event map key for the default stream for consistency between
    // PTDS and non-PTDS mode. In PTDS mode, the cudaStreamLegacy map key will only exist if the
    // user explicitly passes it, so it is used as the default location for the free list
    // at construction. For consistency, the same key is used for null stream free lists in
    // non-PTDS mode.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-cstyle-cast)
    auto* const stream_to_store = stream.is_default() ? cudaStreamLegacy : stream.value();
    stream_id_type stream_id{};
    RMM_ASSERT_CUDA_SUCCESS(cudaStreamGetId(stream_to_store, &stream_id));
    auto const iter = stream_events_.find(stream_id);
    return (iter != stream_events_.end()) ? iter->second : [&]() {
      stream_event_pair stream_event{stream_to_store, nullptr};
      RMM_ASSERT_CUDA_SUCCESS(
        cudaEventCreateWithFlags(&stream_event.event, cudaEventDisableTiming));
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      stream_events_[stream_id] = stream_event;
      return stream_event;
    }();
  }

  /// Whether the current mode tracks a completion event per block rather than per stream.
  [[nodiscard]] bool per_block_events_enabled() const noexcept
  {
    return host_write_mode_ == host_write_sync_mode::block_event ||
           host_write_mode_ == host_write_sync_mode::clean_tracking;
  }

  /**
   * @brief Returns the next event to use for per-block tracking on the stream owning
   * `stream_event`.
   *
   * Events are cycled through a fixed-size ring per stream. Reusing an event re-records it at a
   * later position on the same stream, so a block still referring to it waits for at least its own
   * free position: stale tags over-wait but stay correct.
   */
  cudaEvent_t next_block_event(cudaEvent_t stream_event)
  {
    auto& ring = block_event_rings_[stream_event];
    if (ring.events.size() < block_event_ring_size_) {
      ring.events.push_back(block_event_pool_.acquire());
    }
    auto const event = ring.events[ring.next % ring.events.size()];
    ++ring.next;
    return event;
  }

  /**
   * @brief Splits a block into an allocated block of `size` bytes and a remainder block, and
   * inserts the remainder into a free list.
   *
   * @param block The block to split into allocated and remainder portions.
   * @param size The size of the block to allocate from `b`.
   * @param blocks The `free_list` in which to insert the remainder block.
   * @return The allocated block.
   */
  block_type allocate_and_insert_remainder(block_type block, std::size_t size, free_list& blocks)
  {
    auto const [allocated, remainder] = this->underlying().allocate_from_block(block, size);
    if (remainder.is_valid()) { blocks.insert(remainder); }
    return allocated;
  }

  /**
   * @brief Get an available memory block of at least `size` bytes
   *
   * @param size The number of bytes to allocate
   * @param stream_event The stream and associated event on which the allocation will be used.
   * @return block_type A block of memory of at least `size` bytes
   */
  block_type get_block(std::size_t size, stream_event_pair stream_event)
  {
    // Try to find a satisfactory block in free list for the same stream (no sync required)
    auto iter = stream_free_blocks_.find(stream_event);
    if (iter != stream_free_blocks_.end()) {
      block_type const block = iter->second.get_block(size);
      if (block.is_valid()) { return allocate_and_insert_remainder(block, size, iter->second); }
    }

    free_list& blocks =
      (iter != stream_free_blocks_.end()) ? iter->second : stream_free_blocks_[stream_event];

    // Try to find an existing block in another stream
    {
      block_type const block = get_block_from_other_stream(size, stream_event, blocks, false);
      if (block.is_valid()) { return block; }
    }

    // no large enough blocks available on other streams, so sync and merge until we find one
    {
      block_type const block = get_block_from_other_stream(size, stream_event, blocks, true);
      if (block.is_valid()) { return block; }
    }

    log_summary_trace();

    // no large enough blocks available after merging, so grow the pool
    block_type const block =
      this->underlying().expand_pool(size, blocks, cuda_stream_view{stream_event.stream});

    return allocate_and_insert_remainder(block, size, blocks);
  }

  /**
   * @brief Find a free block of at least `size` bytes in a `free_list` with a different
   * stream/event than `stream_event`.
   *
   * If an appropriate block is found in a free list F associated with event E,
   * `stream_event.stream` will be made to wait on event E.
   *
   * @param size The requested size of the allocation.
   * @param stream_event The stream and associated event on which the allocation is being
   * requested.
   * @return A block with non-null pointer and size >= `size`, or a nullptr block if none is
   *         available in `blocks`.
   */
  block_type get_block_from_other_stream(std::size_t size,
                                         stream_event_pair stream_event,
                                         free_list& blocks,
                                         bool merge_first)
  {
    auto find_block = [&](auto iter) {
      auto other_event   = iter->first.event;
      auto& other_blocks = iter->second;
      if (merge_first) {
        merge_lists(stream_event, blocks, other_event, std::move(other_blocks));

        RMM_LOG_DEBUG("[A][Stream %s][%zuB][Merged stream %s]",
                      rmm::detail::format_stream(stream_event.stream),
                      size,
                      rmm::detail::format_stream(iter->first.stream));

        stream_free_blocks_.erase(iter);

        block_type const block = blocks.get_block(size);  // get the best fit block in merged lists
        if (block.is_valid()) { return allocate_and_insert_remainder(block, size, blocks); }
      } else {
        block_type const block = other_blocks.get_block(size);
        if (block.is_valid()) {
          // Since we found a block associated with a different stream, we have to insert a wait
          // on the stream's associated event into the allocating stream.
          RMM_CUDA_TRY(cudaStreamWaitEvent(stream_event.stream, other_event, 0));
          // That wait orders our stream, not the calling host thread. A host writer must wait on
          // the donor's event itself.
          cross_stream_wait_event_ = other_event;
          return allocate_and_insert_remainder(block, size, other_blocks);
        }
      }
      return block_type{};
    };

    for (auto iter = stream_free_blocks_.begin(), next_iter = iter;
         iter != stream_free_blocks_.end();
         iter = next_iter) {
      ++next_iter;  // Points to element after `iter` to allow erasing `iter` in the loop body

      if (iter->first.event != stream_event.event) {
        block_type const block = find_block(iter);

        if (block.is_valid()) {
          RMM_LOG_DEBUG((merge_first) ? "[A][Stream %s][%zuB][Found after merging stream %s]"
                                      : "[A][Stream %s][%zuB][Taken from stream %s]",
                        rmm::detail::format_stream(stream_event.stream),
                        size,
                        rmm::detail::format_stream(iter->first.stream));
          return block;
        }
      }
    }
    return block_type{};
  }

  void merge_lists(stream_event_pair stream_event,
                   free_list& blocks,
                   cudaEvent_t other_event,
                   free_list&& other_blocks)
  {
    // Since we found a block associated with a different stream, we have to insert a wait
    // on the stream's associated event into the allocating stream.
    RMM_CUDA_TRY(cudaStreamWaitEvent(stream_event.stream, other_event, 0));

    // Merge the two free lists
    blocks.insert(std::move(other_blocks));

    // The merged-in blocks are now keyed by `stream_event.event`, but that event's last
    // recorded position (if it was ever recorded) precedes the wait just enqueued above.
    // Re-record the event so that later consumers of these blocks that synchronize on it
    // (a cross-stream steal or merge, or `release()`) transitively wait on `other_event`,
    // and therefore on any work still in flight on the donor stream.
    RMM_CUDA_TRY(cudaEventRecord(stream_event.event, stream_event.stream));

    // Per-block events are only comparable by sequence number within one stream, and this list now
    // holds blocks freed on two. The record above is ordered after the wait on `other_event`, so
    // re-tagging every block with it restores that invariant without any further synchronization.
    if constexpr (tracks_free_event<block_type>) {
      if (per_block_events_enabled()) {
        auto const seq = ++free_seq_;
        for (auto& blk : blocks) {
          blk.set_free_event(stream_event.event, seq);
        }
      }
    }

    // A host writer taking a block from this list must wait on the re-recorded event.
    cross_stream_wait_event_ = stream_event.event;
  }

  /**
   * @brief Clear free lists and events
   *
   * Note: only called by destructor.
   */
  void release()
  {
    lock_guard lock(mtx_);

    if (rmm::process_is_exiting()) { return; }

    for (auto s_e : stream_events_) {
      RMM_ASSERT_CUDA_SUCCESS_SAFE_SHUTDOWN(cudaEventSynchronize(s_e.second.event));
      RMM_ASSERT_CUDA_SUCCESS_SAFE_SHUTDOWN(cudaEventDestroy(s_e.second.event));
    }

    stream_events_.clear();
    stream_free_blocks_.clear();
  }

  void log_summary_trace()
  {
#if (RMM_LOG_ACTIVE_LEVEL <= RMM_LOG_LEVEL_TRACE)
    std::size_t num_blocks{0};
    std::size_t max_block{0};
    std::size_t free_mem{0};
    std::for_each(stream_free_blocks_.cbegin(),
                  stream_free_blocks_.cend(),
                  [this, &num_blocks, &max_block, &free_mem](auto const& freelist) {
                    num_blocks += freelist.second.size();
                    auto summary = this->underlying().free_list_summary(freelist.second);
                    max_block    = std::max(summary.first, max_block);
                    free_mem += summary.second;
                  });
    RMM_LOG_TRACE("[Summary][Free lists: %zu][Blocks: %zu][Max Block: %zu][Total Free: %zu]",
                  stream_free_blocks_.size(),
                  num_blocks,
                  max_block,
                  free_mem);
#endif
  }

  // map of stream_event_pair --> free_list
  // Event (or associated stream) must be synced before allocating from associated free_list to a
  // different stream
  std::map<stream_event_pair, free_list> stream_free_blocks_;

  // bidirectional mapping between non-default streams and events
  std::unordered_map<stream_id_type, stream_event_pair> stream_events_;

  std::mutex mtx_;  // mutex for thread-safe access

  /// A fixed-size ring of events cycled for per-block completion tracking on one stream.
  struct event_ring {
    std::vector<cudaEvent_t> events;
    std::size_t next{0};
  };

  host_write_sync_mode host_write_mode_{host_write_sync_mode::stream_sync};
  bool skip_stream_event_record_{false};
  std::size_t block_event_ring_size_{64};
  event_pool block_event_pool_;
  std::map<cudaEvent_t, event_ring> block_event_rings_;
  std::uint64_t free_seq_{};
  cudaEvent_t cross_stream_wait_event_{};
  host_writable_stats hw_stats_{};

  rmm::cuda_device_id device_id_{rmm::get_current_cuda_device()};
};  // namespace detail

}  // namespace mr::detail
RMM_NAMESPACE_END
