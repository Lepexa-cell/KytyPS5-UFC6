#ifndef KYTY_COMMON_PROFILER_H_
#define KYTY_COMMON_PROFILER_H_

#include "common/common.h"

#include <atomic>
#include <cstdint>
#include <optional>
#include <tracy/Tracy.hpp> // IWYU pragma: export

namespace profiler::colors {

inline constexpr uint32_t RedA100        = 0xff8a80;
inline constexpr uint32_t Blue300        = 0x64b5f6;
inline constexpr uint32_t CyanA700       = 0x00b8d4;
inline constexpr uint32_t Green100       = 0xc8e6c9;
inline constexpr uint32_t Green200       = 0xa5d6a7;
inline constexpr uint32_t Green300       = 0x81c784;
inline constexpr uint32_t Green400       = 0x66bb6a;
inline constexpr uint32_t Amber300       = 0xffd54f;
inline constexpr uint32_t DeepOrangeA200 = 0xff6e40;

} // namespace profiler::colors

namespace Profiler {

void SetThreadName(const char* name);
// Ends a frame in Tracy's frame view: one call per presented frame.
void MarkFrame();

void Initialize();
void Shutdown();

// KYTY_PROFILE_ZONES=0: no zones, only Tracy's call-stack sampling and context switches (with an
// elevated process). ~300,000 zones a frame inflated the share of small functions that open one.
inline bool g_zones = true;

// Per-frame counters, plotted under Tracy's timeline at every MarkFrame: an event counter shows
// how many happened since the previous frame, a time counter how long they took (microseconds
// added, plotted in milliseconds), a gauge its current value. Adding is a relaxed atomic add,
// skipped without --profile. Names and plot formats: kCounterInfo in profiler.cpp.
enum class Counter : uint32_t {
	// Host draws and dispatches recorded.
	Draws,
	Dispatches,
	// Program lookups refused (a draw or dispatch is skipped): the shader gave up earlier, is on
	// the skip list, or its resources could not be read.
	SkippedGaveUp,
	SkippedSkipList,
	SkippedResources,
	// Draws skipped while their graphics pipeline compiles in the background.
	SkippedPipelinePending,
	// Shader programs and pipelines.
	ShadersCompiled,
	ShadersGaveUp,
	ShaderCompileTime,
	GraphicsPipelinesBuilt,
	ComputePipelinesBuilt,
	PipelineBuildTime,
	PipelinesPending, // gauge
	// Resource tables (SRT replay traces).
	SrtRefreshes,
	SrtReplayed,
	SrtAbandoned,
	SrtRecorded,
	// The GPU thread waiting for the host GPU, and readbacks of GPU-written memory.
	GpuWaits,
	GpuWaitTime,
	Readbacks,
	ReadbackBytes,
	// The recording thread: command buffers submitted, and drains (a caller waiting for it).
	CommandBuffersSubmitted,
	RecordingDrains,
	// Uploads of guest memory into GPU buffers.
	UploadBytes,
	// Texture cache images created and destroyed.
	ImagesCreated,
	ImagesDestroyed,
	// Bindless texture keys resolved.
	BindlessResolved,
	Total
};

inline bool g_counters = false;

// Set on the GPU thread (Thread_Gpu): GpuWaits and GpuWaitTime count only its waits, not those of
// the threads that wait for ticks in the background (deferred operations).
inline thread_local bool t_gpu_thread = false;

struct alignas(64) CounterSlot {
	std::atomic<int64_t> value {0};
};
inline CounterSlot g_counter_slots[static_cast<uint32_t>(Counter::Total)];

inline void Add(Counter counter, int64_t amount = 1) {
	if (g_counters) {
		g_counter_slots[static_cast<uint32_t>(counter)].value.fetch_add(amount,
		                                                                 std::memory_order_relaxed);
	}
}

// Colours of Messages: a failure, something skipped or slow, and plain progress.
inline constexpr uint32_t MessageFailure = 0xff5252;
inline constexpr uint32_t MessageWarning = 0xffab40;
inline constexpr uint32_t MessageInfo    = 0x80cbc4;

// True while a Tracy server is connected (only then do messages and zone texts cost anything).
bool Connected();

// A message on the calling thread's timeline and in Tracy's message list (formatted only while
// connected).
void Message(uint32_t color, const char* format, ...)
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 2, 3)))
#endif
    ;

// Consecutive zones of one task, such as the phases of a shader compile, each with the same text
// (the shader's stage and hash; Tracy's Find Zone groups by it). Unlike KYTY_PROFILER_BLOCK they
// stay on with KYTY_PROFILE_ZONES=0: they are rare and are what a capture is often taken for.
class Phases {
public:
	Phases() = default;
	~Phases() { End(); }
	KYTY_CLASS_NO_COPY(Phases);

	void SetText(const char* format, ...)
#if defined(__GNUC__) || defined(__clang__)
	    __attribute__((format(printf, 2, 3)))
#endif
	    ;
	// Ends the current phase and begins the next (see KYTY_PROFILER_PHASE).
	void Begin(const tracy::SourceLocationData* location);
	void End() { m_zone.reset(); }
	// Colours the current phase (a failure, for instance).
	void Color(uint32_t color) {
		if (m_zone) {
			m_zone->Color(color);
		}
	}

private:
	std::optional<tracy::ScopedZone> m_zone;
	char                             m_text[96] {};
	size_t                           m_text_size = 0;
};

struct Lifecycle {
	static constexpr const char* name               = "Profiler";
	static constexpr auto        initialize         = Profiler::Initialize;
	static constexpr auto        shutdown           = Profiler::Shutdown;
	static constexpr auto        emergency_shutdown = Profiler::Shutdown;
};

} // namespace Profiler

#define KYTY_PROFILER_CONCAT_IMPL(a, b) a##b
#define KYTY_PROFILER_CONCAT(a, b)      KYTY_PROFILER_CONCAT_IMPL(a, b)
#define KYTY_PROFILER_COLOR_OR_DEFAULT(default_color, ...)                                         \
	KYTY_PROFILER_COLOR_OR_DEFAULT_IMPL(default_color __VA_OPT__(, ) __VA_ARGS__, default_color)
#define KYTY_PROFILER_COLOR_OR_DEFAULT_IMPL(default_color, color, ...) color

#define KYTY_PROFILER_BLOCK(name, ...)                                                             \
	KYTY_PROFILER_BLOCK_IMPL(__LINE__, name __VA_OPT__(, ) __VA_ARGS__)
#define KYTY_PROFILER_BLOCK_IMPL(line, name, ...)                                                  \
	static constexpr tracy::SourceLocationData KYTY_PROFILER_CONCAT(                               \
	    kyty_profiler_source_location_, line) {name, TracyFunction, TracyFile,                     \
	                                           static_cast<uint32_t>(line),                        \
	                                           KYTY_PROFILER_COLOR_OR_DEFAULT(0, __VA_ARGS__)};    \
	tracy::ScopedZone KYTY_PROFILER_CONCAT(kyty_profiler_block_, line)(                           \
	    &KYTY_PROFILER_CONCAT(kyty_profiler_source_location_, line),                              \
	    TRACY_CALLSTACK, tracy::ProfilerAvailable() && Profiler::g_zones)

#define KYTY_PROFILER_FUNCTION(...) KYTY_PROFILER_BLOCK(nullptr __VA_OPT__(, ) __VA_ARGS__)

#define KYTY_PROFILER_THREAD(name) Profiler::SetThreadName(name)

// Begins the next phase of a Profiler::Phases.
#define KYTY_PROFILER_PHASE(phases, name, ...)                                                     \
	do {                                                                                           \
		static constexpr tracy::SourceLocationData kyty_profiler_phase_location {                  \
		    name, TracyFunction, TracyFile, static_cast<uint32_t>(__LINE__),                       \
		    KYTY_PROFILER_COLOR_OR_DEFAULT(0, __VA_ARGS__)};                                       \
		(phases).Begin(&kyty_profiler_phase_location);                                             \
	} while (false)

#endif /* KYTY_COMMON_PROFILER_H_ */
