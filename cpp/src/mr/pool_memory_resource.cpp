/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <rmm/mr/pool_memory_resource.hpp>

#include <cstddef>
#include <memory>
#include <optional>

RMM_NAMESPACE_BEGIN
namespace mr {

pool_memory_resource::pool_memory_resource(
  cuda::mr::any_resource<cuda::mr::device_accessible> upstream,
  std::size_t initial_pool_size,
  std::optional<std::size_t> maximum_pool_size)
  : shared_base(cuda::mr::make_shared_resource<detail::pool_memory_resource_impl>(
      std::move(upstream), initial_pool_size, maximum_pool_size))
{
}

device_async_resource_ref pool_memory_resource::get_upstream_resource() const noexcept
{
  return get().get_upstream_resource();
}

std::size_t pool_memory_resource::pool_size() const noexcept { return get().pool_size(); }

void* pool_memory_resource::allocate_host_writable(cuda::stream_ref stream,
                                                   std::size_t bytes,
                                                   std::size_t alignment)
{
  return get().allocate_host_writable(stream, bytes, alignment);
}

void pool_memory_resource::deallocate_host_writable(cuda::stream_ref stream,
                                                    void* ptr,
                                                    std::size_t bytes,
                                                    std::size_t alignment,
                                                    bool device_exposed) noexcept
{
  get().deallocate_host_writable(stream, ptr, bytes, alignment, device_exposed);
}

void pool_memory_resource::set_host_write_sync_mode(host_write_sync_mode mode) noexcept
{
  get().set_host_write_sync_mode(mode);
}

void pool_memory_resource::set_skip_stream_event_record(bool skip) noexcept
{
  get().set_skip_stream_event_record(skip);
}

void pool_memory_resource::set_block_event_ring_size(std::size_t size) noexcept
{
  get().set_block_event_ring_size(size);
}

host_writable_stats pool_memory_resource::host_writable_statistics()
{
  return get().host_writable_statistics();
}

void pool_memory_resource::reset_host_writable_statistics()
{
  get().reset_host_writable_statistics();
}

}  // namespace mr
RMM_NAMESPACE_END
