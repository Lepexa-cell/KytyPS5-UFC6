#include "graphics/host_gpu/renderer/commandScheduler.h"

#include "common/assert.h"
#include "common/common.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "common/timer.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/timeline.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <vector>

namespace Libs::Graphics {

static thread_local CommandScheduler* g_deferred_callback_scheduler = nullptr;

namespace {

// Mega-suite fix 5: submit worker (Vulkan driver calls) owns P-Core 1 (0x000C).
// The CommandProcessor translator owns P-Core 0, guest jobs own P-Cores 2..5.
void PinThreadToPerformanceCores() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	Common::PinCurrentThreadToMask(static_cast<uint32_t>(Common::PerfCoreMask::SubmitWorker),
	                               Common::PerfCorePriority::Highest);
#else
	(void)0;
#endif
}

void ReportVulkanFatal(const char* what, vk::Result result, uint64_t tick, uint32_t debug_op,
                       uint64_t debug_submit, uint32_t arg0, uint32_t arg1, uint32_t arg2,
                       uint32_t arg3, uint64_t arg4) {
	LOGF("%s failed: %s (%d), tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	     " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	     what, vk::to_string(result).c_str(), static_cast<int>(result), tick, debug_op,
	     debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::printf("%s failed: %s (%d), tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	            " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	            what, vk::to_string(result).c_str(), static_cast<int>(result), tick, debug_op,
	            debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::fflush(stdout);
}

// KYTY_DRAW_FLUSH_INTERVAL=N overrides CompleteDraw()'s periodic non-blocking flush interval.
// Mid-frame progressive submit: UFC records ~2332 draws into one buffer while the GPU sits idle.
// HTM parallel pass recording: closing and queueing a chunk every 1024 draws (Submit + BeginNext,
// no Wait) lets the GPU execute the first chunk while the CPU records the next. 2048 left larger
// idle bubbles on octagon frames; 32 was ~500 submits/frame and drowned the overlap in
// vkQueueSubmit overhead. 0 disables.
// The flush is deferred while a dynamic-rendering scope is open: splitting a pass forces
// End/BeginRendering load+store traffic (VRAM -> tile cache reload). The pending chunk is
// flushed at the next pass boundary (BeginRendering on a new state / EndRendering) with a
// 4x safety cap so an unbounded single-pass frame cannot grow one buffer forever.
uint32_t DrawFlushInterval() {
	static const uint32_t interval = [] {
		const char* v = std::getenv("KYTY_DRAW_FLUSH_INTERVAL");
		if (v == nullptr) {
			return 2048u;
		}
		return static_cast<uint32_t>(std::strtoul(v, nullptr, 10));
	}();
	return interval;
}

// UFC 5 telemetry: 161 vkQueueSubmit per frame (~every 14 draws) is almost entirely
// RELEASE_MEM fence writes, not the 2048-draw progressive flush. Guest-visible values
// are already written synchronously, so a larger batch does not change correctness.
// A per-frame budget keeps the presented frame at a handful of natural submits
// (end of frame, flip, FlushAndWait drains) instead of one submit per fence.
// KYTY_FRAME_FLUSH_BUDGET overrides the cap; 0 disables the cap.
uint32_t FrameFlushBudget() {
	static const uint32_t budget = [] {
		const char* v = std::getenv("KYTY_FRAME_FLUSH_BUDGET");
		if (v == nullptr) {
			return 8u;
		}
		return static_cast<uint32_t>(std::strtoul(v, nullptr, 10));
	}();
	return budget;
}

} // namespace

CommandScheduler::CommandPool::CommandPool(GraphicContext& graphics, MasterSemaphore& master,
                                            uint32_t family)
    : m_graphics(graphics), m_master(master) {
	EXIT_IF(family == static_cast<uint32_t>(-1));
	vk::CommandPoolCreateInfo create {};
	create.queueFamilyIndex = family;
	create.flags            = vk::CommandPoolCreateFlagBits::eTransient |
	                          vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	const auto result       = graphics.device.createCommandPool(&create, nullptr, &m_pool);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_pool == nullptr);
}

CommandScheduler::CommandPool::~CommandPool() {
	m_graphics.device.destroyCommandPool(m_pool, nullptr);
}

size_t CommandScheduler::CommandPool::Grow() {
	const auto first = m_ticks.size();
	m_ticks.resize(first + GrowStep);
	m_buffers.resize(first + GrowStep);

	vk::CommandBufferAllocateInfo allocate {};
	allocate.commandPool        = m_pool;
	allocate.level              = vk::CommandBufferLevel::ePrimary;
	allocate.commandBufferCount = static_cast<uint32_t>(GrowStep);
	EXIT_IF(m_graphics.device.allocateCommandBuffers(&allocate, m_buffers.data() + first) !=
	        vk::Result::eSuccess);
	return first;
}

vk::CommandBuffer CommandScheduler::CommandPool::Commit() {
	auto       gpu_tick = m_master.KnownGpuTick();
	const auto search   = [this, &gpu_tick](size_t begin, size_t end) -> std::optional<size_t> {
		for (size_t index = begin; index < end; ++index) {
			if (gpu_tick >= m_ticks[index]) {
				m_ticks[index] = m_master.CurrentTick();
				return index;
			}
		}
		return std::nullopt;
	};

	auto found = search(m_hint, m_ticks.size());
	if (!found) {
		m_master.Refresh();
		gpu_tick = m_master.KnownGpuTick();
		found    = search(m_hint, m_ticks.size());
	}
	if (!found) {
		found = search(0, m_hint);
	}
	if (!found) {
		found           = Grow();
		m_ticks[*found] = m_master.CurrentTick();
	}

	m_hint = (*found + 1) % m_ticks.size();
	return m_buffers[*found];
}

bool CommandScheduler::InDeferredOperation() noexcept {
	return g_deferred_callback_scheduler != nullptr;
}

CommandScheduler::CommandScheduler(RenderContext& context, GraphicContext& graphics)
    : m_master(graphics), m_context(context), m_graphics(graphics),
      m_command_pool(graphics, m_master, graphics.queue_family), m_command(*this),
      m_priority_thread([this](std::stop_token stop) { PriorityOperationsThread(stop); }) {}

CommandScheduler::~CommandScheduler() {
	Shutdown();
}

void CommandScheduler::Shutdown() {
	{
		std::unique_lock lock(m_operation_mutex);
		if (m_operation_state == OperationState::Closed) {
			return;
		}
		if (g_deferred_callback_scheduler == this) {
			EXIT_IF(m_operation_state == OperationState::Open);
			// A priority callback cannot join its own runner, while a normal callback can be
			// executing inside the shutdown owner's final PopPendingOperations. The owning
			// thread will finish shutdown after this callback returns.
			return;
		}
		if (m_operation_state == OperationState::Draining) {
			m_operation_available.wait(
			    lock, [this] { return m_operation_state == OperationState::Closed; });
			return;
		}
		m_operation_state = OperationState::Draining;
	}
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	PopPendingOperations();
	DrainPriorityOperations();
	m_priority_thread.request_stop();
	m_operation_available.notify_all();
	if (m_priority_thread.joinable()) {
		m_priority_thread.join();
	}
	// Every tick was waited on above, so the queue thread has nothing left to submit.
	StopSubmitThread();
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(!m_pending_operations.empty() || !m_priority_operations.empty() ||
		        m_priority_active);
		m_operation_state = OperationState::Closed;
	}
	m_operation_available.notify_all();
}

void CommandScheduler::Begin(HW::Context& registers, HW::UserConfig& user_config,
                             HW::Shader& shaders) {
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(m_operation_state != OperationState::Open);
	}
	m_command.Bind(registers, user_config, shaders);

	if (m_command.IsInvalid()) {
		BeginNext();
	}
}

void CommandScheduler::BeginRendering(const RenderState& state) {
	// Any graphics scope makes the open submission mixed: it must stay on graphics.
	m_compute_only = false;
	Current().BeginRendering(state);
}

void CommandScheduler::EndRendering() {
	if (Active() && !m_command.IsInvalid()) {
		Context().GetRenderExecutor().FlushPendingComputeBarrier();
		Current().EndRendering();
	}
}

bool CommandScheduler::IsRendering() const {
	return Active() && !m_command.IsInvalid() && Current().IsRendering();
}

void CommandScheduler::Flush() {
	SubmitInfo submit;
	Flush(submit);
	// Actual submissions only: the graphics command processor records the
	// presented frame. Compute queues and the async submit worker do not.
	if (GuestGpu::IsGpuThread() && Active()) {
		Context().GetGpu().GraphicsProcessor().TelemetryCountFlush();
		m_frame_submits++;
	}
}

bool CommandScheduler::FrameFlushBudgetReached() const {
	const auto budget = FrameFlushBudget();
	return budget != 0u && m_frame_submits >= budget;
}

void CommandScheduler::CompleteReleaseMemWrite() {
	// Tuned for Frostbite UFC 5/6: higher batch size reduces CPU overhead from frequent submissions.
	// 64 was ~one submit per 14 draws (161 flushes/frame). The write is already guest-visible.
	constexpr uint32_t WritesPerSubmission = 512;
	if (++m_recorded_release_mem_writes < WritesPerSubmission || FrameFlushBudgetReached()) {
		return;
	}
	CheckActive();
	Flush();
}

void CommandScheduler::CompleteReleaseMemInterrupt() {
	// Deliberately smaller than CompleteReleaseMemWrite's 512: this event has already been queued
	// for guest delivery once its tick completes (see Sync::TriggerEopEventAtEndOfPipe ->
	// DeferPriorityOperation), and a guest thread may be blocked waiting on it via an event queue.
	// Batching still defers only the vkQueueSubmit -- the event fires once that (now slightly
	// larger) submission's tick completes, same as before, just a handful of RELEASE_MEM events
	// later instead of immediately. 16 was the other half of the 161-flush frame.
	constexpr uint32_t InterruptsPerSubmission = 256;
	if (++m_recorded_release_mem_interrupts < InterruptsPerSubmission ||
	    FrameFlushBudgetReached()) {
		return;
	}
	CheckActive();
	Flush();
}

void CommandScheduler::CompleteDraw() {
	const auto interval = DrawFlushInterval();
	if (interval == 0u || ++m_recorded_draws < interval) {
		return;
	}
	CheckActive();
	// Never split an open dynamic-rendering scope: chunking mid-pass closes and
	// reopens it (EndRendering -> BeginRendering), forcing the GPU to reload
	// color/depth attachments and delaying UI draws past the Flip. Flush only
	// between passes; end-of-frame / flip / FlushAndWait still submit.
	if (IsRendering()) {
		return;
	}
	// Once the frame already has its handful of natural submits, further progressive
	// chunks only add driver overhead. End-of-frame / flip / FlushAndWait still submit.
	if (FrameFlushBudgetReached()) {
		return;
	}
	// Draws leave vkCmdBeginRendering open across calls. End() (via Flush -> Submit) closes
	// that scope before the buffer is queued, and the next draw reopens it on the new buffer.
	// A coalesced compute barrier is recorded by Flush() itself, so it stays in this chunk.
	// Non-blocking: the closed buffer goes to the async submit worker; the CPU does not wait.
	Flush();
}

void CommandScheduler::Flush(SubmitInfo& submit) {
	Context().GetRenderExecutor().FlushPendingComputeBarrier();
	Submit(submit);
	BeginNext();
}

void CommandScheduler::BeginPresentedFrame() {
	// Per-presented-frame batch windows: a new frame starts with a fresh submit budget
	// and empty RELEASE_MEM / draw batch counters, so leftover counts can't force an
	// immediate micro-flush at the start of the frame (161-flush elimination: submits only
	// in handful-sized batches via CompleteReleaseMemWrite/Interrupt/CompleteDraw).
	m_frame_submits                   = 0;
	m_recorded_release_mem_writes     = 0;
	m_recorded_release_mem_interrupts = 0;
	m_recorded_draws                  = 0;
}

void CommandScheduler::FlushAndWait() {
	KYTY_PROFILER_FUNCTION();
	Context().GetRenderExecutor().FlushPendingComputeBarrier();
	const auto tick = Submit();
	m_master.Wait(tick);
	BeginNext();
}

void CommandScheduler::Finish() {
	KYTY_PROFILER_FUNCTION();
	CheckActive();
	Context().GetRenderExecutor().FlushPendingComputeBarrier();
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	BeginNext();
	PopPendingOperations();
}

// How long the GPU thread spends drained on the host GPU, and how often. A drain that
// submits the current tick first (a CPU read of GPU-owned memory mid-frame) is counted apart
// from one that waits on already submitted work.
namespace {
struct WaitStats {
	uint64_t count       = 0;
	uint64_t drain_count = 0;
	double   total_s     = 0.0;
	double   drain_s     = 0.0;
	double   max_s       = 0.0;
};
WaitStats g_wait_stats;

void RecordWait(double seconds, bool drained) {
	if (!GuestGpu::IsGpuThread()) {
		return;
	}
	auto& stats = g_wait_stats;
	stats.count++;
	stats.total_s += seconds;
	stats.max_s = std::max(stats.max_s, seconds);
	if (drained) {
		stats.drain_count++;
		stats.drain_s += seconds;
	}
	if (stats.count % 256 == 0) {
		LOGF("GpuWaits: count=%" PRIu64 " total=%.1fms max=%.1fms drains=%" PRIu64
		     " drain_total=%.1fms\n",
		     stats.count, stats.total_s * 1000.0, stats.max_s * 1000.0, stats.drain_count,
		     stats.drain_s * 1000.0);
		stats = {};
	}
}
} // namespace

void CommandScheduler::Wait(uint64_t tick) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(tick > CurrentTick());
	Common::Timer timer;
	timer.Start();
	if (tick == CurrentTick()) {
		CheckActive();
		// A stream-buffer wrap can wait while a draw is being prepared through a reference to
		// Current(). The wrapper stays stable while its pooled Vulkan buffer is retired. Deferred
		// resources are released only at the next GPU operation boundary.
		{
			KYTY_PROFILER_BLOCK("CommandScheduler::Wait (forced submit-then-wait)");
			const auto submitted_tick = Submit();
			EXIT_IF(submitted_tick != tick);
			Timeline::Mark("drain-begin", tick);
			m_master.Wait(tick);
			Timeline::Mark("drain-end", tick);
			BeginNext();
		}
		RecordWait(timer.GetTimeS(), true);
	} else {
		Timeline::Mark("wait-begin", tick);
		m_master.Wait(tick);
		Timeline::Mark("wait-end", tick);
		RecordWait(timer.GetTimeS(), false);
	}
}

void CommandScheduler::PopPendingOperations() {
	// Hot path (~16k draws/frame): skip the driver timeline query entirely when
	// nothing is queued. KnownGpuTick is a relaxed atomic; Refresh() is the
	// getSemaphoreCounterValue call and stays on the miss path only.
	{
		std::lock_guard lock(m_operation_mutex);
		if (m_pending_operations.empty()) {
			return;
		}
		if (!m_master.IsFree(m_pending_operations.front().tick)) {
			m_master.Refresh();
		}
	}
	for (;;) {
		PendingOperation operation;
		{
			std::lock_guard lock(m_operation_mutex);
			if (m_pending_operations.empty() ||
			    !m_master.IsFree(m_pending_operations.front().tick)) {
				return;
			}
			operation = std::move(m_pending_operations.front());
			m_pending_operations.pop();
		}
		WaitPriorityOperations(operation.tick);
		RunOperation(std::move(operation.callback));
	}
}

void CommandScheduler::DeferOperation(Common::UniqueFunction<void>&& operation) {
	CheckActive();
	EXIT_IF(!operation);
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		m_pending_operations.push({std::move(operation), CurrentTick()});
		return;
	}
	if (g_deferred_callback_scheduler == this) {
		lock.unlock();
		operation();
		return;
	}
	m_operation_available.wait(lock,
	                           [this] { return m_operation_state == OperationState::Closed; });
	lock.unlock();
	operation();
}

void CommandScheduler::DeferPriorityOperation(Common::UniqueFunction<void>&& operation) {
	CheckActive();
	EXIT_IF(!operation);
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		m_priority_operations.push({std::move(operation), CurrentTick()});
		lock.unlock();
		m_operation_available.notify_one();
		return;
	}
	if (g_deferred_callback_scheduler == this) {
		lock.unlock();
		operation();
		return;
	}
	m_operation_available.wait(lock,
	                           [this] { return m_operation_state == OperationState::Closed; });
	lock.unlock();
	operation();
}

void CommandScheduler::PriorityOperationsThread(std::stop_token stop) {
	// Event-release traffic unblocks guest job workers: P-cores at above-normal
	// priority so RELEASE_MEM events never stall behind E-core scheduling.
	KYTY_PROFILER_THREAD("GpuPriorityOps");
	Common::PinCurrentThreadToPerformanceCores(Common::PerfCorePriority::AboveNormal);
	while (!stop.stop_requested()) {
		PendingOperation operation;
		{
			std::unique_lock lock(m_operation_mutex);
			m_operation_available.wait(lock, [this, &stop] {
				return stop.stop_requested() || !m_priority_operations.empty();
			});
			if (stop.stop_requested()) {
				return;
			}
			operation = std::move(m_priority_operations.front());
			m_priority_operations.pop();
			m_priority_active      = true;
			m_priority_active_tick = operation.tick;
		}
		m_master.Wait(operation.tick);
		Timeline::Mark("tick-done", operation.tick);
		if (!stop.stop_requested()) {
			RunOperation(std::move(operation.callback));
		}
		{
			std::lock_guard lock(m_operation_mutex);
			m_priority_active      = false;
			m_priority_active_tick = 0;
		}
		m_operation_available.notify_all();
	}
}

void CommandScheduler::DrainPriorityOperations() {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(
	    lock, [this] { return m_priority_operations.empty() && !m_priority_active; });
}

void CommandScheduler::WaitPriorityOperations(uint64_t tick) {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(lock, [this, tick] {
		const bool active_before_or_at = m_priority_active && m_priority_active_tick <= tick;
		const bool queued_before_or_at =
		    !m_priority_operations.empty() && m_priority_operations.front().tick <= tick;
		return !active_before_or_at && !queued_before_or_at;
	});
}

void CommandScheduler::RunOperation(Common::UniqueFunction<void>&& operation) {
	auto* previous                = g_deferred_callback_scheduler;
	g_deferred_callback_scheduler = this;
	operation();
	g_deferred_callback_scheduler = previous;
}

bool CommandScheduler::IsFree(uint64_t tick) {
	if (m_master.IsFree(tick)) {
		return true;
	}
	m_master.Refresh();
	return m_master.IsFree(tick);
}

void CommandScheduler::CheckActive() const {
	EXIT_IF(!Active());
}

CommandBuffer& CommandScheduler::Current() {
	CheckActive();
	return m_command;
}

const CommandBuffer& CommandScheduler::Current() const {
	CheckActive();
	return m_command;
}

CommandBuffer& CommandScheduler::BeginCommand() {
	EXIT_IF(!m_command.IsInvalid());
	m_command.m_buffer = m_command_pool.Commit();
	m_command.Begin();
	return m_command;
}

uint64_t CommandScheduler::Submit(SubmitInfo submit) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(m_command.IsInvalid());
	EXIT_IF(submit.num_wait_semaphores > SubmitInfo::MaxSemaphores ||
	        submit.num_signal_semaphores >= SubmitInfo::MaxSemaphores);

	m_command.End();
	EXIT_IF(m_graphics.queue == nullptr);

	// Async compute routing: a submission holding only dispatches runs on the dedicated
	// queue. Cross-queue order uses the shared master timeline: a compute submit waits
	// for the last graphics tick, a graphics submit after compute waits for the last
	// compute tick. Same-queue fallback (no dedicated queue) needs no extra edges.
	const bool dedicated_compute =
	    m_graphics.compute_queue != nullptr && m_graphics.compute_queue != m_graphics.queue;
	const bool compute_submit = m_compute_only && dedicated_compute;
	if (compute_submit && m_last_graphics_tick != 0) {
		submit.AddWait(m_master.Handle(), m_last_graphics_tick,
		               vk::PipelineStageFlagBits::eComputeShader);
	} else if (!compute_submit && dedicated_compute && m_last_compute_tick != 0) {
		submit.AddWait(m_master.Handle(), m_last_compute_tick);
	}

	SubmitJob job {
	    .buffer       = m_command.m_buffer,
	    .submit       = submit,
	    .compute_only = compute_submit,
	    .debug_op     = m_command.m_debug_op,
	    .debug_submit = m_command.m_debug_submit_id,
	    .debug_arg0   = m_command.m_debug_arg0,
	    .debug_arg1   = m_command.m_debug_arg1,
	    .debug_arg2   = m_command.m_debug_arg2,
	    .debug_arg3   = m_command.m_debug_arg3,
	    .debug_arg4   = m_command.m_debug_arg4,
	};
	m_command.m_buffer                = nullptr;
	m_recorded_release_mem_writes     = 0;
	m_recorded_release_mem_interrupts = 0;
	m_recorded_draws                  = 0;
	m_compute_only                    = false;

	if (!m_async_submit) {
		QueueSubmit(job);
		if (job.compute_only) {
			m_last_compute_tick = job.tick;
		} else if (dedicated_compute) {
			m_last_graphics_tick = job.tick;
		}
		return job.tick;
	}
	{
		// Ticks are allocated in queue order and the queue thread submits in that order, so the
		// timeline is signaled in order.
		std::lock_guard lock(m_submit_mutex);
		job.tick = m_master.NextTick();
		job.submit.AddSignal(m_master.Handle(), job.tick);
		if (job.compute_only) {
			m_last_compute_tick = job.tick;
		} else if (dedicated_compute) {
			m_last_graphics_tick = job.tick;
		}
		if (m_submit_head >= m_submit_jobs.size()) {
			// Fully drained: restart at 0 and reuse capacity (no free).
			m_submit_jobs.clear();
			m_submit_head = 0;
		} else if (m_submit_head > 256 && m_submit_head * 2 >= m_submit_jobs.size()) {
			// Periodically compact the consumed prefix so the vector does
			// not grow unboundedly when the producer outruns the consumer.
			m_submit_jobs.erase(
			    m_submit_jobs.begin(),
			    m_submit_jobs.begin() +
			        static_cast<std::vector<SubmitJob>::difference_type>(m_submit_head));
			m_submit_head = 0;
		}
		m_submit_jobs.push_back(job);
	}
	m_submit_available.notify_one();
	return job.tick;
}

void CommandScheduler::QueueSubmit(SubmitJob& job) {
	auto&      graphics = m_graphics;
	vk::Result result;
	// Compute-only work targets the dedicated queue (own mutex: two queues can submit in
	// parallel); mixed/graphics work keeps the graphics queue and its existing lock order.
	if (job.compute_only && graphics.compute_queue != nullptr &&
	    graphics.compute_queue != graphics.queue) {
		Common::LockGuard lock(graphics.queue_compute_mutex);
		if (!m_async_submit) {
			job.tick = m_master.NextTick();
			job.submit.AddSignal(m_master.Handle(), job.tick);
		}
		const auto& submit = job.submit;

		vk::TimelineSemaphoreSubmitInfo timeline_info {};
		timeline_info.waitSemaphoreValueCount   = submit.num_wait_semaphores;
		timeline_info.pWaitSemaphoreValues      = submit.wait_ticks.data();
		timeline_info.signalSemaphoreValueCount = submit.num_signal_semaphores;
		timeline_info.pSignalSemaphoreValues    = submit.signal_ticks.data();

		vk::SubmitInfo submit_info {};
		submit_info.pNext                = &timeline_info;
		submit_info.waitSemaphoreCount   = submit.num_wait_semaphores;
		submit_info.pWaitSemaphores      = submit.wait_semaphores.data();
		submit_info.pWaitDstStageMask    = submit.wait_stages.data();
		submit_info.commandBufferCount   = 1;
		submit_info.pCommandBuffers      = &job.buffer;
		submit_info.signalSemaphoreCount = submit.num_signal_semaphores;
		submit_info.pSignalSemaphores    = submit.signal_semaphores.data();

		result = graphics.compute_queue.submit(1, &submit_info, nullptr);
		Timeline::Mark("vk-submit-compute", job.tick);
	} else {
		Common::LockGuard lock(graphics.queue_mutex);
		if (!m_async_submit) {
			job.tick = m_master.NextTick();
			job.submit.AddSignal(m_master.Handle(), job.tick);
		}
		const auto& submit = job.submit;

		vk::TimelineSemaphoreSubmitInfo timeline_info {};
		timeline_info.waitSemaphoreValueCount   = submit.num_wait_semaphores;
		timeline_info.pWaitSemaphoreValues      = submit.wait_ticks.data();
		timeline_info.signalSemaphoreValueCount = submit.num_signal_semaphores;
		timeline_info.pSignalSemaphoreValues    = submit.signal_ticks.data();

		vk::SubmitInfo submit_info {};
		submit_info.pNext                = &timeline_info;
		submit_info.waitSemaphoreCount   = submit.num_wait_semaphores;
		submit_info.pWaitSemaphores      = submit.wait_semaphores.data();
		submit_info.pWaitDstStageMask    = submit.wait_stages.data();
		submit_info.commandBufferCount   = 1;
		submit_info.pCommandBuffers      = &job.buffer;
		submit_info.signalSemaphoreCount = submit.num_signal_semaphores;
		submit_info.pSignalSemaphores    = submit.signal_semaphores.data();

		result = graphics.queue.submit(1, &submit_info, nullptr);
		Timeline::Mark("vk-submit", job.tick);
	}

	if (result == vk::Result::eErrorDeviceLost) {
		DumpDeviceLossDiagnostics(graphics);
	}
	if (result != vk::Result::eSuccess) {
		ReportVulkanFatal("vkQueueSubmit", result, job.tick, job.debug_op, job.debug_submit,
		                  job.debug_arg0, job.debug_arg1, job.debug_arg2, job.debug_arg3,
		                  job.debug_arg4);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandScheduler::EnableAsyncSubmit() {
	// Two-thread model entry point: the owner enables it while constructing,
	// before any other thread can submit. From here on the translator thread only
	// records/closes command buffers and enqueues SubmitJobs; the submit worker
	// thread owns every blocking driver call (vkQueueSubmit, queue-mutex waits).
	EXIT_IF(m_async_submit);
	m_async_submit  = true;
	m_submit_thread = std::jthread([this](std::stop_token stop) { SubmitThread(stop); });
}

void CommandScheduler::SubmitThread(std::stop_token stop) {
	// Driver worker: P-core affinity + highest priority so the 16k-draw UFC 5
	// standup never starves the RTX 4070 while CPU5 keeps translating PM4.
	KYTY_PROFILER_THREAD("GpuQueueSubmit");
	PinThreadToPerformanceCores();
	for (;;) {
		SubmitJob job;
		{
			std::unique_lock lock(m_submit_mutex);
			// A stop request still submits the queued jobs: their ticks may already be waited on.
			// Shutdown() waits every tick before requesting the stop, so the queue must be empty
			// by then -- every barrier, upload and EOP write already recorded still reaches the
			// driver in tick order. The condition-variable notify in StopSubmitThread wakes us.
			if (!m_submit_available.wait(lock, stop, [this] {
				    return m_submit_head < m_submit_jobs.size();
			    })) {
				return;
			}
			job = m_submit_jobs[m_submit_head++];
			if (m_submit_head >= m_submit_jobs.size()) {
				// Fully drained: restart at 0 and reuse capacity (no free).
				m_submit_jobs.clear();
				m_submit_head = 0;
			}
		}
		// Heavy driver call happens here, off the translator thread: queue-locked
		// vkQueueSubmit + timeline signal + present-side semaphore handoff.
		QueueSubmit(job);
	}
}

void CommandScheduler::StopSubmitThread() {
	// Ordered teardown: stop flag + notify wakes the worker, join waits for the in-flight
	// vkQueueSubmit to return, then the empty-queue check proves no tick was dropped.
	if (m_submit_thread.joinable()) {
		m_submit_thread.request_stop();
		m_submit_available.notify_all();
		m_submit_thread.join();
	}
	std::lock_guard lock(m_submit_mutex);
	EXIT_IF(m_submit_head < m_submit_jobs.size());
}

void CommandScheduler::BeginNext() {
	CheckActive();
	BeginCommand();
}

} // namespace Libs::Graphics
