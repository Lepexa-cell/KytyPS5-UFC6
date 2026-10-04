#include "graphics/host_gpu/renderer/sync.h"

#include "common/assert.h"
#include "common/common.h"
#include "common/logging/log.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/presentation/videoOut.h"
#include "kernel/eventQueue.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <atomic>
#include <cstring>
#include <limits>

namespace Libs::Graphics::Sync {

constexpr uint64_t GRAPHICS_REFERENCE_CLOCK_FREQUENCY = 100000000;
// Upper bound of the virtual GPU time a single frame may report: 2 000 000
// ticks = 20.0 ms on the 100 MHz reference clock (an ideal 60 FPS frame is
// 1 666 666 ticks = 16.666 ms).
constexpr uint64_t kMaxVirtualFrameTicks = 2000000;

namespace {

// Virtual GPU clock shared by RELEASE_MEM/COPY_DATA timestamps. g_virtual_clock
// is the last published guest-visible tick; g_frame_base anchors the start of
// the current frame so per-frame growth can be clamped. Both advance only via
// atomic CAS and never move backwards.
std::atomic<uint64_t> g_virtual_clock {0};
std::atomic<uint64_t> g_frame_base {0};
std::atomic<bool>     g_frame_base_set {false};

} // namespace

void NotifyFrameBoundary() {
	// Re-anchor the frame budget to the current virtual time. The next frame may
	// then report at most kMaxVirtualFrameTicks above this point, no matter how
	// long host-side translation of its draws takes. CAS publish keeps the base
	// monotonic if several flip paths race; a lost race simply keeps the older
	// (smaller) base, which only tightens the budget.
	const uint64_t current = g_virtual_clock.load(std::memory_order_acquire);
	uint64_t       prev    = g_frame_base.load(std::memory_order_relaxed);
	while (current > prev) {
		if (g_frame_base.compare_exchange_weak(prev, current, std::memory_order_acq_rel,
		                                       std::memory_order_relaxed)) {
			break;
		}
	}
	g_frame_base_set.store(true, std::memory_order_release);
}

bool ScaleReferenceClock(uint64_t host_ticks, uint64_t host_frequency, uint64_t& value) {
	if (host_frequency == 0) {
		return false;
	}

	const auto     whole_seconds = host_ticks / host_frequency;
	const auto     remainder     = host_ticks % host_frequency;
	constexpr auto MAX_VALUE     = std::numeric_limits<uint64_t>::max();
	if (whole_seconds > MAX_VALUE / GRAPHICS_REFERENCE_CLOCK_FREQUENCY ||
	    remainder > MAX_VALUE / GRAPHICS_REFERENCE_CLOCK_FREQUENCY) {
		return false;
	}

	const auto whole_value      = whole_seconds * GRAPHICS_REFERENCE_CLOCK_FREQUENCY;
	const auto fractional_value = (remainder * GRAPHICS_REFERENCE_CLOCK_FREQUENCY) / host_frequency;
	if (whole_value > MAX_VALUE - fractional_value) {
		return false;
	}
	value = whole_value + fractional_value;
	return true;
}

uint64_t ReadReferenceClock() {
	const auto host_frequency = LibKernel::KernelGetTscFrequency();
	const auto host_ticks     = LibKernel::KernelReadTsc();
	// Anchor the guest-visible clock to the first observed host tick so the first
	// timestamp the game ever sees starts near zero. An absolute host-tick base
	// (time since host boot) would make the first delta look like hours and trip
	// Frostbite DRS into its emergency throttle. Elapsed ticks are scaled to the
	// 100 MHz PS5 graphics reference clock below; both RELEASE_MEM (data_sel == 3,
	// IT_RELEASE_MEM opcode 0x49) and COPY_DATA reference-clock reads
	// (IT_COPY_DATA opcode 0x40, src_sel 9/18) share this generator, and the
	// atomic CAS loop at the end keeps it strictly monotonic.
	// HTM virtualization: the value is additionally clamped to a per-frame budget
	// (see kMaxVirtualFrameTicks and NotifyFrameBoundary below) so CPU-side
	// command translation time never leaks into the guest-visible GPU clock.
	static std::atomic<uint64_t> base_ticks {0};
	static std::atomic<bool>     base_set {false};
	uint64_t base = base_ticks.load(std::memory_order_relaxed);
	if (!base_set.load(std::memory_order_acquire)) {
		uint64_t expected = 0;
		if (base_ticks.compare_exchange_strong(expected, host_ticks, std::memory_order_acq_rel,
		                                         std::memory_order_relaxed)) {
			base = host_ticks;
			base_set.store(true, std::memory_order_release);
		} else {
			base = expected;
		}
	}
	const uint64_t elapsed_ticks = host_ticks >= base ? host_ticks - base : 0;
	uint64_t   value          = 0;
	if (!ScaleReferenceClock(elapsed_ticks, host_frequency, value)) {
		EXIT("cannot scale host clock, ticks=0x%016" PRIx64 " frequency=%" PRIu64 "\n", host_ticks,
		     host_frequency);
	}
	// HTM virtualization for Frostbite DRS: a frame's worth of CPU-side command
	// translation (~80 ms for ~16k draws) must not leak into the guest-visible
	// GPU clock, or the engine concludes the GPU stalled and engages its
	// emergency ~12 FPS divider. Clamp virtual growth to kMaxVirtualFrameTicks
	// per frame (20.0 ms at 100 MHz; ideal 60 FPS is 1 666 666 ticks =
	// 16.666 ms), anchored at the last Flip/NextFrame boundary. Stamps stay
	// strictly monotonic via atomic CAS: once pinned at the cap, later stamps in
	// the same frame reuse cap+1, so 16k draws cannot walk the clock up while
	// the raw host clock sits far above the budget.
	if (!g_frame_base_set.load(std::memory_order_acquire)) {
		// First stamp ever: anchor the frame budget here so even frame 0 (before
		// the first Flip) is clamped. A lost CAS race just adopts the winner's
		// base, which is also clamped below.
		uint64_t expected = 0;
		uint64_t desired  = value;
		if (desired > kMaxVirtualFrameTicks) {
			desired = kMaxVirtualFrameTicks;
		}
		if (g_frame_base.compare_exchange_strong(expected, desired, std::memory_order_acq_rel,
		                                         std::memory_order_relaxed)) {
			g_frame_base_set.store(true, std::memory_order_release);
		}
	}
	if (g_frame_base_set.load(std::memory_order_acquire)) {
		const uint64_t frame_base = g_frame_base.load(std::memory_order_acquire);
		uint64_t       cap        = frame_base + kMaxVirtualFrameTicks;
		if (cap < frame_base) {
			cap = std::numeric_limits<uint64_t>::max();
		}
		if (value > cap) {
			value = cap;
		}
		uint64_t prev = g_virtual_clock.load(std::memory_order_relaxed);
		// Inside one frame the raw host clock can sit far above the cap (e.g.
		// 80 ms of translation vs the 20 ms budget). Once the clock is pinned
		// at the cap, publish cap+1 and stop: further stamps reuse cap+1 via
		// the monotonic branch below, so 16k draws cannot walk the clock up.
		if (prev >= cap) {
			uint64_t pinned = cap + 1;
			if (pinned < cap) {
				pinned = std::numeric_limits<uint64_t>::max();
			}
			while (true) {
				if (pinned <= prev) {
					pinned = prev;
					if (pinned == std::numeric_limits<uint64_t>::max()) {
						return pinned;
					}
				}
				if (g_virtual_clock.compare_exchange_weak(prev, pinned,
				                                          std::memory_order_acq_rel,
				                                          std::memory_order_relaxed)) {
					return pinned;
				}
			}
		}
	}
	uint64_t prev    = g_virtual_clock.load(std::memory_order_relaxed);
	uint64_t current = value;
	// Ensure strictly monotonic output: if the clock read would go backwards or stall,
	// return prev+1 instead so RELEASE_MEM/COPY_DATA emit only ascending timestamps.
	while (true) {
		if (current <= prev) {
			current = prev + 1;
		}
		if (g_virtual_clock.compare_exchange_weak(prev, current, std::memory_order_acq_rel,
		                                          std::memory_order_relaxed)) {
			return current;
		}
	}
}

enum class EndOfPipeWriteSize : uint32_t { Dword = 4, Qword = 8 };
enum class EndOfPipeWriteAction { Write, WriteBack, Interrupt, InterruptWriteBack };

static CommandBufferDebugOp DebugOperation(EndOfPipeWriteAction action) {
	switch (action) {
		case EndOfPipeWriteAction::Write: return CommandBufferDebugOp::EopWrite;
		case EndOfPipeWriteAction::WriteBack:
		case EndOfPipeWriteAction::InterruptWriteBack: return CommandBufferDebugOp::EopWriteBack;
		case EndOfPipeWriteAction::Interrupt: return CommandBufferDebugOp::EopInterrupt;
	}
	EXIT("unsupported end-of-pipe write action\n");
	return CommandBufferDebugOp::Unknown;
}

static bool TriggersInterrupt(EndOfPipeWriteAction action) {
	return action == EndOfPipeWriteAction::Interrupt ||
	       action == EndOfPipeWriteAction::InterruptWriteBack;
}

static void RecordEndOfPipeWrite(uint64_t submit_id, CommandBuffer& buffer, uint64_t destination,
                                 uint64_t value, EndOfPipeWriteSize size,
                                 EndOfPipeWriteAction action, int interrupt_event_id = 0,
                                 uint32_t context_id = 0) {
	EXIT_IF(destination == 0);
	(void)buffer.Handle();

	const auto width      = static_cast<uint32_t>(size);
	const auto value_low  = static_cast<uint32_t>(value);
	const auto value_high = static_cast<uint32_t>(value >> 32u);
	const auto operation  = static_cast<uint32_t>(DebugOperation(action));
	if (TriggersInterrupt(action)) {
		buffer.SetDebugInfo(operation, submit_id, width, context_id, value_low, value_high,
		                    destination);
		TriggerEopEventAtEndOfPipe(buffer, interrupt_event_id, context_id);
	} else {
		buffer.SetDebugInfo(operation, submit_id, width, value_low, value_high, 0, destination);
	}
}

void WriteAtEndOfPipe32(uint64_t submit_id, CommandBuffer& buffer, uint32_t* dst_gpu_addr,
                        uint32_t value) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Dword, EndOfPipeWriteAction::Write);
}

void WriteAtEndOfPipeGds32(uint64_t submit_id, CommandBuffer& buffer, uint32_t* dst_gpu_addr,
                           uint32_t dw_offset, uint32_t dw_num) {
	EXIT_IF(dst_gpu_addr == nullptr);
	(void)buffer.Handle();
	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::EopWrite), submit_id,
	                    dw_offset, dw_num, 0, 0, reinterpret_cast<uint64_t>(dst_gpu_addr));
}

void WriteAtEndOfPipe64(uint64_t submit_id, CommandBuffer& buffer, uint64_t* dst_gpu_addr,
                        uint64_t value) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Qword, EndOfPipeWriteAction::Write);
}

void WriteAtEndOfPipeWithWriteBack64(uint64_t submit_id, CommandBuffer& buffer,
                                     uint64_t* dst_gpu_addr, uint64_t value) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Qword, EndOfPipeWriteAction::WriteBack);
}

void WriteAtEndOfPipeWithWriteBack32(uint64_t submit_id, CommandBuffer& buffer,
                                     uint32_t* dst_gpu_addr, uint32_t value) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Dword, EndOfPipeWriteAction::WriteBack);
}

void WriteAtEndOfPipeWithInterruptWriteBack64(uint64_t submit_id, CommandBuffer& buffer,
                                              uint64_t* dst_gpu_addr, uint64_t value, int event_id,
                                              uint32_t context_id) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Qword, EndOfPipeWriteAction::InterruptWriteBack,
	                     event_id, context_id);
}

void WriteAtEndOfPipeWithInterruptWriteBack32(uint64_t submit_id, CommandBuffer& buffer,
                                              uint32_t* dst_gpu_addr, uint32_t value, int event_id,
                                              uint32_t context_id) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Dword, EndOfPipeWriteAction::InterruptWriteBack,
	                     event_id, context_id);
}

void WriteAtEndOfPipeWithInterrupt64(uint64_t submit_id, CommandBuffer& buffer,
                                     uint64_t* dst_gpu_addr, uint64_t value, int event_id,
                                     uint32_t context_id) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Qword, EndOfPipeWriteAction::Interrupt, event_id,
	                     context_id);
}

void WriteAtEndOfPipeWithInterrupt32(uint64_t submit_id, CommandBuffer& buffer,
                                     uint32_t* dst_gpu_addr, uint32_t value, int event_id,
                                     uint32_t context_id) {
	RecordEndOfPipeWrite(submit_id, buffer, reinterpret_cast<uint64_t>(dst_gpu_addr), value,
	                     EndOfPipeWriteSize::Dword, EndOfPipeWriteAction::Interrupt, event_id,
	                     context_id);
}

uint64_t PrepareVideoOutFlip(CommandBuffer& buffer, int handle, int index, int flip_mode,
                             int64_t flip_arg) {
	for (;;) {
		uint64_t   request_id = 0;
		auto&      video_out  = buffer.GetContext().GetVideoOut();
		const auto result =
		    video_out.SubmitFlipFromGpu(buffer, handle, index, flip_mode, flip_arg, request_id);
		if (result == OK) {
			EXIT_IF(request_id == 0);
			return request_id;
		}
		if (result != VideoOut::VIDEO_OUT_ERROR_FLIP_QUEUE_FULL) {
			EXIT("GPU flip submission failed, result=%d handle=%d index=%d mode=%d arg=%" PRId64
			     "\n",
			     result, handle, index, flip_mode, flip_arg);
		}
		video_out.WaitForSubmitSlot(handle);
	}
}

void WriteAtEndOfPipeWithInterruptWriteBackFlip32(uint64_t submit_id, CommandBuffer& buffer,
                                                  uint32_t* dst_gpu_addr, uint32_t value,
                                                  int handle, int index, int flip_mode,
                                                  int64_t flip_arg, uint64_t request_id,
                                                  int event_id) {
	EXIT_IF(dst_gpu_addr == nullptr);
	(void)buffer.Handle();
	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::EopWriteBackFlip), submit_id,
	                    static_cast<uint32_t>(handle), static_cast<uint32_t>(index),
	                    static_cast<uint32_t>(flip_mode), value, static_cast<uint64_t>(flip_arg));

	auto& renderer  = buffer.GetContext();
	auto& scheduler = renderer.GetCommandScheduler();
	EXIT_IF(!scheduler.Active() || &buffer != &scheduler.Current());
	scheduler.DeferPriorityOperation([&renderer, event_id, request_id] {
		renderer.GetVideoOut().CompleteFlip(request_id);
		renderer.TriggerInterrupt(event_id, 0);
	});
}

void WriteAtEndOfPipeWithFlip32(uint64_t submit_id, CommandBuffer& buffer, uint32_t* dst_gpu_addr,
                                uint32_t value, int handle, int index, int flip_mode,
                                int64_t flip_arg, uint64_t request_id) {
	EXIT_IF(dst_gpu_addr == nullptr);
	(void)buffer.Handle();
	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::EopFlip), submit_id,
	                    static_cast<uint32_t>(handle), static_cast<uint32_t>(index),
	                    static_cast<uint32_t>(flip_mode), value, static_cast<uint64_t>(flip_arg));

	auto& renderer  = buffer.GetContext();
	auto& scheduler = renderer.GetCommandScheduler();
	EXIT_IF(!scheduler.Active() || &buffer != &scheduler.Current());
	scheduler.DeferPriorityOperation(
	    [&renderer, request_id] { renderer.GetVideoOut().CompleteFlip(request_id); });
}

void WriteAtEndOfPipeOnlyFlip(uint64_t submit_id, CommandBuffer& buffer, int handle, int index,
                              int flip_mode, int64_t flip_arg, uint64_t request_id) {
	(void)buffer.Handle();
	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::EopOnlyFlip), submit_id,
	                    static_cast<uint32_t>(handle), static_cast<uint32_t>(index),
	                    static_cast<uint32_t>(flip_mode), 0, static_cast<uint64_t>(flip_arg));

	auto& renderer  = buffer.GetContext();
	auto& scheduler = renderer.GetCommandScheduler();
	EXIT_IF(!scheduler.Active() || &buffer != &scheduler.Current());
	scheduler.DeferPriorityOperation(
	    [&renderer, request_id] { renderer.GetVideoOut().CompleteFlip(request_id); });
}

void TriggerEopEventAtEndOfPipe(CommandBuffer& buffer, int event_id, uint32_t context_id) {
	(void)buffer.Handle();
	auto& renderer  = buffer.GetContext();
	auto& scheduler = renderer.GetCommandScheduler();
	EXIT_IF(!scheduler.Active() || &buffer != &scheduler.Current());
	scheduler.DeferPriorityOperation(
	    [&renderer, event_id, context_id] { renderer.TriggerInterrupt(event_id, context_id); });
}

static void InterruptEventResetFunc(LibKernel::EventQueue::KernelEqueueEvent* event) {
	EXIT_IF(event == nullptr);
	event->triggered    = false;
	event->event.fflags = 0;
	event->event.data   = 0;
}

static void InterruptEventTriggerFunc(LibKernel::EventQueue::KernelEqueueEvent* event,
                                      void*                                     trigger_data) {
	EXIT_IF(event == nullptr);

	auto triggered_event = event->event;
	triggered_event.fflags++;
	triggered_event.data = reinterpret_cast<intptr_t>(trigger_data);
	if (event->triggered) {
		event->pending_events.push_back(triggered_event);
	} else {
		event->event     = triggered_event;
		event->triggered = true;
	}
}

int AddEqEvent(RenderContext& renderer, LibKernel::EventQueue::KernelEqueue eq, int id,
               void* udata) {
	LibKernel::EventQueue::KernelEqueueEvent event;
	event.triggered           = false;
	event.event.ident         = static_cast<uintptr_t>(id);
	event.event.filter        = LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS;
	event.event.udata         = udata;
	event.event.fflags        = 0;
	event.event.data          = id;
	event.filter.reset_func   = InterruptEventResetFunc;
	event.filter.trigger_func = InterruptEventTriggerFunc;

	int result = LibKernel::EventQueue::KernelAddEvent(eq, event);

	if (result == 0) {
		renderer.AddInterruptEq(eq, id);
	}

	return result;
}

int DeleteEqEvent(RenderContext& renderer, LibKernel::EventQueue::KernelEqueue eq, int id) {
	int result = LibKernel::EventQueue::KernelDeleteEvent(
	    eq, static_cast<uintptr_t>(id), LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS);
	if (result == OK || result == LibKernel::KERNEL_ERROR_ENOENT) {
		renderer.DeleteInterruptEq(eq, id);
	}

	return result;
}

void ReadGds(const Buffer& gds, uint32_t* dst, uint32_t dw_offset, uint32_t dw_size) {
	const auto offset = uint64_t {dw_offset} * sizeof(uint32_t);
	const auto size   = uint64_t {dw_size} * sizeof(uint32_t);
	EXIT_IF(dst == nullptr || offset > gds.Size() || size > gds.Size() - offset ||
	        gds.Mapped().empty());
	std::memcpy(dst, gds.Mapped().data() + offset, static_cast<size_t>(size));
}

} // namespace Libs::Graphics::Sync
