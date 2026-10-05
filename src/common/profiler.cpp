#include "common/profiler.h"

#include "common/emulatorConfig.h"

#include <common/TracyProtocol.hpp>
#include <common/TracyVersion.hpp>
#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <tracy/Tracy.hpp>

namespace Profiler {

namespace {

enum class CounterKind : uint8_t { Events, Bytes, TimeUs, Gauge };

struct CounterInfo {
	const char* name;
	CounterKind kind;
	uint32_t    color;
};

// Plot names are compared by pointer: each must stay this one string.
constexpr CounterInfo kCounterInfo[] = {
    {"Draws", CounterKind::Events, 0x64b5f6},
    {"Dispatches", CounterKind::Events, 0x4dd0e1},
    {"Skipped: shader gave up", CounterKind::Events, 0xff5252},
    {"Skipped: skip list", CounterKind::Events, 0x9e9e9e},
    {"Skipped: resources unreadable", CounterKind::Events, 0xff8a80},
    {"Skipped: pipeline compiling", CounterKind::Events, 0xffab40},
    {"Shaders compiled", CounterKind::Events, 0x81c784},
    {"Shaders gave up", CounterKind::Events, 0xff5252},
    {"Shader compile ms", CounterKind::TimeUs, 0xa5d6a7},
    {"Graphics pipelines built", CounterKind::Events, 0x66bb6a},
    {"Compute pipelines built", CounterKind::Events, 0x43a047},
    {"Pipeline build ms", CounterKind::TimeUs, 0xc8e6c9},
    {"Pipelines pending", CounterKind::Gauge, 0xffd54f},
    {"SRT refreshes", CounterKind::Events, 0xce93d8},
    {"SRT replayed", CounterKind::Events, 0xba68c8},
    {"SRT abandoned", CounterKind::Events, 0xff6e40},
    {"SRT recorded", CounterKind::Events, 0xf48fb1},
    {"GPU waits", CounterKind::Events, 0xe57373},
    {"GPU wait ms", CounterKind::TimeUs, 0xef5350},
    {"Readbacks", CounterKind::Events, 0xffb74d},
    {"Readback bytes", CounterKind::Bytes, 0xffa726},
    {"Command buffers submitted", CounterKind::Events, 0x90a4ae},
    {"Recording drains", CounterKind::Events, 0xff7043},
    {"Upload bytes", CounterKind::Bytes, 0x4fc3f7},
    {"Images created", CounterKind::Events, 0xaed581},
    {"Images destroyed", CounterKind::Events, 0xdce775},
    {"Bindless keys resolved", CounterKind::Events, 0x7986cb},
};
static_assert(std::size(kCounterInfo) == static_cast<size_t>(Counter::Total));

void ConfigurePlots() {
	for (const auto& info: kCounterInfo) {
		const auto format = info.kind == CounterKind::Bytes ? tracy::PlotFormatType::Memory
		                                                    : tracy::PlotFormatType::Number;
		tracy::Profiler::ConfigurePlot(info.name, format, true, info.kind != CounterKind::TimeUs,
		                               info.color);
	}
}

void PlotCounters() {
	for (size_t index = 0; index < std::size(kCounterInfo); index++) {
		const auto& info  = kCounterInfo[index];
		auto&       value = g_counter_slots[index].value;
		switch (info.kind) {
			case CounterKind::Gauge:
				tracy::Profiler::PlotData(info.name, value.load(std::memory_order_relaxed));
				break;
			case CounterKind::TimeUs:
				tracy::Profiler::PlotData(
				    info.name, static_cast<double>(value.exchange(0, std::memory_order_relaxed)) / 1000.0);
				break;
			default:
				tracy::Profiler::PlotData(info.name, value.exchange(0, std::memory_order_relaxed));
				break;
		}
	}
}

} // namespace

void SetThreadName(const char* name) {
	if (tracy::ProfilerAvailable() && name != nullptr) {
		tracy::SetThreadName(name);
	}
}

void MarkFrame() {
	if (tracy::ProfilerAvailable()) {
		FrameMark;
		if (g_counters) {
			PlotCounters();
		}
	}
}

bool Connected() {
	return tracy::ProfilerAvailable() && tracy::GetProfiler().IsConnected();
}

void Message(uint32_t color, const char* format, ...) {
	if (!Connected()) {
		return;
	}
	char    text[512];
	va_list args;
	va_start(args, format);
	const int size = std::vsnprintf(text, sizeof(text), format, args);
	va_end(args);
	if (size <= 0) {
		return;
	}
	tracy::Profiler::LogString(tracy::MessageSourceType::User, tracy::MessageSeverity::Info, color,
	                           0, std::min(static_cast<size_t>(size), sizeof(text) - 1), text);
}

void Phases::SetText(const char* format, ...) {
	va_list args;
	va_start(args, format);
	const int size = std::vsnprintf(m_text, sizeof(m_text), format, args);
	va_end(args);
	m_text_size = size <= 0 ? 0 : std::min(static_cast<size_t>(size), sizeof(m_text) - 1);
	if (m_zone && m_text_size != 0) {
		m_zone->Text(m_text, m_text_size);
	}
}

void Phases::Begin(const tracy::SourceLocationData* location) {
	m_zone.reset();
	if (!tracy::ProfilerAvailable()) {
		return;
	}
	m_zone.emplace(location, TRACY_CALLSTACK, true);
	if (m_text_size != 0) {
		m_zone->Text(m_text, m_text_size);
	}
}

void Initialize() {
	if (const char* zones = std::getenv("KYTY_PROFILE_ZONES"); zones != nullptr && zones[0] == '0') {
		g_zones = false;
	}
	if (Config::ProfilerEnabled() && !tracy::ProfilerAvailable()) {
		tracy::StartupProfiler();
		TracySetProgramName("KytyPS5");
		ConfigurePlots();
		g_counters = true;
		::printf("Tracy profiler enabled: client %d.%d.%d, protocol %u, "
		         "broadcast %u, connect to 127.0.0.1:8086\n",
		         tracy::Version::Major, tracy::Version::Minor, tracy::Version::Patch,
		         tracy::ProtocolVersion, tracy::BroadcastVersion);
	}
}

void Shutdown() {
	g_counters = false;
	if (tracy::ProfilerAvailable()) {
		tracy::ShutdownProfiler();
	}
}

} // namespace Profiler
