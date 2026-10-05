#include "graphics/host_gpu/renderer/gpuProfiler.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <client/TracyProfiler.hpp>
#include <common/TracyQueue.hpp>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <vector>

namespace Libs::Graphics::GpuProfiler {

namespace {

// Tracy's query ids are 16-bit.
constexpr uint32_t kMaxQueries = 65536;

struct Range {
	uint64_t tick  = 0;
	uint64_t begin = 0;
	uint64_t end   = 0;
};

struct State {
	std::once_flag        init;
	bool                  enabled = false;
	vk::Device            device;
	vk::QueryPool         pool;
	uint32_t              count   = 0;
	uint8_t               context = 0;
	PFN_vkGetCalibratedTimestampsKHR calibrated = nullptr;
	// Queries are used in order: [tail, head) are written or pending. The GPU thread advances
	// head; the collector (the scheduler's priority thread) advances tail.
	std::mutex            mutex;
	uint64_t              head       = 0;
	std::atomic<uint64_t> tail       = 0;
	uint64_t              range_tick = UINT64_MAX;
	uint64_t              range_begin = 0;
	std::deque<Range>     ranges;
	// The connection (+1) under which each query's zone event was sent; 0: none.
	std::vector<uint32_t> sent;
	std::vector<uint64_t> results;
};

State& GetState() {
	static State state;
	return state;
}

uint64_t ReadGpuClock(State& state) {
	VkCalibratedTimestampInfoKHR info {};
	info.sType      = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR;
	info.timeDomain = VK_TIME_DOMAIN_DEVICE_KHR;
	uint64_t timestamp = 0;
	uint64_t deviation = 0;
	if (state.calibrated(static_cast<VkDevice>(state.device), 1, &info, &timestamp, &deviation) !=
	    VK_SUCCESS) {
		return 0;
	}
	return timestamp;
}

bool Initialize(State& state, RenderContext& context) {
	const auto& graphics = context.GetGraphics();
	if (const char* value = std::getenv("KYTY_PROFILE_GPU"); value != nullptr && value[0] == '0') {
		return false;
	}
	state.calibrated = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetCalibratedTimestampsKHR != nullptr
	                       ? VULKAN_HPP_DEFAULT_DISPATCHER.vkGetCalibratedTimestampsKHR
	                       : VULKAN_HPP_DEFAULT_DISPATCHER.vkGetCalibratedTimestampsEXT;
	if (!graphics.host_query_reset_enabled || !graphics.calibrated_timestamps_enabled ||
	    state.calibrated == nullptr) {
		LOGF("GpuProfiler: off (host query reset %d, calibrated timestamps %d)\n",
		     graphics.host_query_reset_enabled ? 1 : 0, graphics.calibrated_timestamps_enabled ? 1 : 0);
		return false;
	}
	state.device = graphics.device;
	for (state.count = kMaxQueries; state.count >= 1024; state.count /= 2) {
		vk::QueryPoolCreateInfo info {};
		info.queryType  = vk::QueryType::eTimestamp;
		info.queryCount = state.count;
		if (state.device.createQueryPool(&info, nullptr, &state.pool) == vk::Result::eSuccess) {
			break;
		}
	}
	if (!state.pool) {
		return false;
	}
	state.device.resetQueryPool(state.pool, 0, state.count);
	state.sent.assign(state.count, 0);
	state.results.resize(static_cast<size_t>(state.count) * 2);

	// The GPU clock against Tracy's: one pair read back to back (a few microseconds apart).
	const int64_t  cpu_before = tracy::Profiler::GetTime();
	const uint64_t gpu        = ReadGpuClock(state);
	const int64_t  cpu_after  = tracy::Profiler::GetTime();
	state.context = static_cast<uint8_t>(tracy::GetGpuCtxCounter().fetch_add(1));
	{
		auto* item = tracy::Profiler::QueueSerial();
		tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuNewContext);
		tracy::MemWrite(&item->gpuNewContext.cpuTime, cpu_before + (cpu_after - cpu_before) / 2);
		tracy::MemWrite(&item->gpuNewContext.gpuTime, static_cast<int64_t>(gpu));
		std::memset(&item->gpuNewContext.thread, 0, sizeof(item->gpuNewContext.thread));
		tracy::MemWrite(&item->gpuNewContext.period,
		                graphics.physical_device_properties.limits.timestampPeriod);
		tracy::MemWrite(&item->gpuNewContext.context, state.context);
		tracy::MemWrite(&item->gpuNewContext.flags, tracy::GpuContextFlags(0));
		tracy::MemWrite(&item->gpuNewContext.type, tracy::GpuContextType::Vulkan);
#ifdef TRACY_ON_DEMAND
		tracy::GetProfiler().DeferItem(*item);
#endif
		tracy::Profiler::QueueSerialFinish();
	}
	{
		static constexpr char kName[] = "Host GPU";
		auto*                 name    = static_cast<char*>(tracy::tracy_malloc(sizeof(kName) - 1));
		std::memcpy(name, kName, sizeof(kName) - 1);
		auto* item = tracy::Profiler::QueueSerial();
		tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuContextName);
		tracy::MemWrite(&item->gpuContextNameFat.context, state.context);
		tracy::MemWrite(&item->gpuContextNameFat.ptr, reinterpret_cast<uint64_t>(name));
		tracy::MemWrite(&item->gpuContextNameFat.size, static_cast<uint16_t>(sizeof(kName) - 1));
#ifdef TRACY_ON_DEMAND
		tracy::GetProfiler().DeferItem(*item);
#endif
		tracy::Profiler::QueueSerialFinish();
	}
	LOGF("GpuProfiler: %u timestamp queries, period %.3f ns\n", state.count,
	     graphics.physical_device_properties.limits.timestampPeriod);
	return true;
}

// Sends the GPU times of every query of the ticks up to `tick` (executed by now) and frees them.
void Collect(State& state, uint64_t tick) {
	std::vector<Range> ranges;
	{
		std::lock_guard lock(state.mutex);
		while (!state.ranges.empty() && state.ranges.front().tick <= tick) {
			ranges.push_back(state.ranges.front());
			state.ranges.pop_front();
		}
		if (state.range_tick != UINT64_MAX && state.range_tick <= tick &&
		    state.head != state.range_begin) {
			ranges.push_back({state.range_tick, state.range_begin, state.head});
			state.range_begin = state.head;
		}
	}
	const uint32_t connection = tracy::GetProfiler().ConnectionId() + 1;
	const bool     connected  = tracy::GetProfiler().IsConnected();
	for (const auto& range: ranges) {
		for (uint64_t first = range.begin; first < range.end;) {
			const auto     index = static_cast<uint32_t>(first % state.count);
			const uint32_t size  = static_cast<uint32_t>(
                std::min<uint64_t>(range.end - first, state.count - index));
			const auto result = state.device.getQueryPoolResults(
			    state.pool, index, size, sizeof(uint64_t) * 2 * size, state.results.data(),
			    sizeof(uint64_t) * 2,
			    vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability);
			for (uint32_t i = 0; i < size; i++) {
				auto& sent = state.sent[index + i];
				if (connected && sent == connection &&
				    (result == vk::Result::eSuccess || result == vk::Result::eNotReady) &&
				    state.results[i * 2 + 1] != 0) {
					auto* item = tracy::Profiler::QueueSerial();
					tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuTime);
					tracy::MemWrite(&item->gpuTime.gpuTime, static_cast<int64_t>(state.results[i * 2]));
					tracy::MemWrite(&item->gpuTime.queryId, static_cast<uint16_t>(index + i));
					tracy::MemWrite(&item->gpuTime.context, state.context);
					tracy::Profiler::QueueSerialFinish();
				}
				sent = 0;
			}
			state.device.resetQueryPool(state.pool, index, size);
			first += size;
		}
		state.tail.store(range.end, std::memory_order_release);
	}
}

} // namespace

namespace Detail {

int64_t BeginSlow(RenderContext& context, CommandBuffer& buffer, const char* name, size_t size,
                  uint32_t color) {
	auto& state = GetState();
	std::call_once(state.init, [&] { state.enabled = Initialize(state, context); });
	if (!state.enabled || !tracy::GetProfiler().IsConnected()) {
		return -1;
	}
	auto&          scheduler = context.GetCommandScheduler();
	const uint64_t tick      = scheduler.CurrentTick();
	uint64_t       query     = 0;
	{
		std::lock_guard lock(state.mutex);
		// Room for the zone's two timestamps.
		if (state.head + 2 - state.tail.load(std::memory_order_acquire) > state.count) {
			return -1;
		}
		if (tick != state.range_tick) {
			if (state.range_tick != UINT64_MAX && state.head != state.range_begin) {
				state.ranges.push_back({state.range_tick, state.range_begin, state.head});
			}
			state.range_tick  = tick;
			state.range_begin = state.head;
			scheduler.DeferPriorityOperation([&state, tick] { Collect(state, tick); });
		}
		query = state.head;
		state.head += 2;
	}
	const auto index = static_cast<uint32_t>(query % state.count);
	buffer.Recorder().writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, state.pool, index);
	const auto srcloc = tracy::Profiler::AllocSourceLocation(0, "host GPU", "GpuProfiler", name, size,
	                                                         color);
	auto*      item   = tracy::Profiler::QueueSerial();
	tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuZoneBeginAllocSrcLocSerial);
	tracy::MemWrite(&item->gpuZoneBegin.cpuTime, tracy::Profiler::GetTime());
	tracy::MemWrite(&item->gpuZoneBegin.srcloc, srcloc);
	tracy::MemWrite(&item->gpuZoneBegin.thread, tracy::GetThreadHandle());
	tracy::MemWrite(&item->gpuZoneBegin.queryId, static_cast<uint16_t>(index));
	tracy::MemWrite(&item->gpuZoneBegin.context, state.context);
	tracy::Profiler::QueueSerialFinish();
	state.sent[index] = tracy::GetProfiler().ConnectionId() + 1;
	return static_cast<int64_t>(query);
}

void EndSlow(CommandBuffer& buffer, int64_t token) {
	auto&      state = GetState();
	const auto index = static_cast<uint32_t>((static_cast<uint64_t>(token) + 1) % state.count);
	buffer.Recorder().writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, state.pool, index);
	auto* item = tracy::Profiler::QueueSerial();
	tracy::MemWrite(&item->hdr.type, tracy::QueueType::GpuZoneEndSerial);
	tracy::MemWrite(&item->gpuZoneEnd.cpuTime, tracy::Profiler::GetTime());
	tracy::MemWrite(&item->gpuZoneEnd.thread, tracy::GetThreadHandle());
	tracy::MemWrite(&item->gpuZoneEnd.queryId, static_cast<uint16_t>(index));
	tracy::MemWrite(&item->gpuZoneEnd.context, state.context);
	tracy::Profiler::QueueSerialFinish();
	state.sent[index] = tracy::GetProfiler().ConnectionId() + 1;
}

} // namespace Detail

} // namespace Libs::Graphics::GpuProfiler
