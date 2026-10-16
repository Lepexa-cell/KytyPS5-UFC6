#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMMAND_PROCESSOR_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMMAND_PROCESSOR_H

#include "common/assert.h"
#include "common/platform/sysTimer.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <vector>

namespace Libs::Graphics {

bool TestWaitRegMemValue(uint64_t value, uint64_t ref, uint64_t mask, uint32_t func);

// KYTY_OPT_TELEMETRY_SUBCOUNTERS=0 restores the single State timer. Telemetry-only change:
// all draws take the same code path with or without subcounters.
[[nodiscard]] inline bool TelemetrySubcountersEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_OPT_TELEMETRY_SUBCOUNTERS");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

// KYTY_OPT_UPLOAD_DEDUP=0 disables the RebindBuffers upload dedup (default: enabled).
// Telemetry-only change: no behavioural difference when disabled.
[[nodiscard]] inline bool UploadDedupEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_OPT_UPLOAD_DEDUP");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

// Step 4c: KYTY_OPT_BDA_CLEAN_SKIP=0 restores the legacy BDA walk (default: enabled).
// Telemetry-only change when the hint reports dirty: the full walk still runs.
[[nodiscard]] inline bool BdaCleanSkipEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_OPT_BDA_CLEAN_SKIP");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

// KYTY_OPT_PIPELINE_LAST_HIT=0 restores the full GraphicsPipelineKey hash + map lookup on every
// draw (default: reuse the last pipeline when the compact key matches).
// CPU-only change: MaterializeResources/specialization/invalidation are untouched.
[[nodiscard]] inline bool PipelineLastHitEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_OPT_PIPELINE_LAST_HIT");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

// KYTY_OPT_SUBMIT_TELEMETRY=0 disables the per-submit cause accounting below
// (default: enabled). Diagnostic only: all submits take the same code path either way.
[[nodiscard]] inline bool SubmitTelemetryEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_OPT_SUBMIT_TELEMETRY");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

// SubmitCause lives in render.h so CommandScheduler's Flush()/Submit() signatures can
// name it without an include cycle.

// KYTY_OPT_SYNC_DOWNLOAD_DIAG=1 adds a per-frame sync-download breakdown (default: off).
// Diagnostic only: when off, the sync path adds no extra clock reads. When on, the
// synchronous BufferCache download splits its fence wait into submit+BeginNext CPU /
// host-side tick wait / priority-operations wait / publish, and the BufferFlush
// slice sites count completed vs yielded graphics/compute boundaries plus suspend
// points.
[[nodiscard]] inline bool SyncDownloadDiagEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_OPT_SYNC_DOWNLOAD_DIAG");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

// KYTY_OPT_NARROW_READBACK_BARRIER=1 narrows the pre-copy source barrier to the dirty
// [min,max) span actually copied (default: off = legacy whole-buffer barrier).
// Scope: DownloadBufferCopies runs it for both async and sync downloads; the change
// is barrier geometry only. The span is clamped into the buffer below, but that clamp
// never validates a copy -- an out-of-range copy stays the caller's bug. The post-copy
// download-buffer barrier is untouched, the copy list/order, publication, dirty
// tracking and tick waits are identical.
[[nodiscard]] inline bool NarrowReadbackBarrierEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_OPT_NARROW_READBACK_BARRIER");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

// KYTY_OPT_SUBMIT_MERGE_RELEASE_FLUSH=0 restores the legacy immediate submit after
// every interrupt-path CpOpReleaseMem (default: merged into the CompleteReleaseMemInterrupt
// batching window below). Independent from KYTY_OPT_SUBMIT_TELEMETRY.
[[nodiscard]] inline bool SubmitMergeReleaseFlushEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_OPT_SUBMIT_MERGE_RELEASE_FLUSH");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

enum class Pm4ProcessResult { Complete, Blocked };

// Microsecond frame telemetry (QPC-based). GPU thread only: no atomics by design.
struct FrameTelemetry {
	uint64_t pm4_op_us[256] = {};
	uint64_t draw_record_us = 0;
	uint64_t draw_state_us = 0;
	uint64_t draw_state_prepare_vs_us = 0;
	uint64_t draw_state_prepare_ps_us = 0;
	uint64_t draw_state_key_find_us = 0;
	uint64_t draw_state_materialize_us = 0;
	// Step 4: Mat sub-timers (measured around the MaterializeResources call in
	// pipelineCache.cpp; the SRT layer itself stays telemetry-free).
	//  - mat_refresh: the synchronous SRT refresh (walker + guest memory reads).
	//  - mat_specialize: specialization build + permutation scan that follows.
	uint64_t draw_state_mat_refresh_us = 0;
	uint64_t draw_state_mat_specialize_us = 0;
	uint64_t draw_state_permutation_us = 0;
	uint64_t draw_state_targets_us = 0;
	uint64_t draw_targets_color_us = 0;
	uint64_t draw_targets_depth_us = 0;
	uint64_t draw_targets_cache_lookup_us = 0;
	uint64_t draw_targets_cache_insert_us = 0;
	uint64_t draw_targets_overlap_us = 0;
	uint64_t draw_targets_expand_us = 0;
	uint64_t draw_targets_refresh_us = 0;
	uint64_t draw_targets_clear_us = 0;
	uint64_t draw_targets_clear_tracker_us = 0;
	uint64_t draw_targets_clear_gpu_wait_us = 0;
	uint64_t draw_targets_clear_readback_us = 0;
	uint64_t draw_targets_clear_writeback_us = 0;
	uint64_t readmem_flush_submit_us = 0;
	uint64_t readmem_fence_wait_us = 0;
	uint64_t readmem_copy_readback_us = 0;
	// KYTY_OPT_SYNC_DOWNLOAD_DIAG breakdown of the sync-download covered span only
	// (explicit DownloadBufferCopies<false> calls, never the scheduler's other waits):
	// calls, already-submitted ticks, forced/current-tick drains, and the fence span
	// split into submit+BeginNext CPU / host-side tick wait / priority operations /
	// publish.
	uint32_t sync_dl_calls = 0;
	uint32_t sync_dl_already_submitted = 0;
	uint32_t sync_dl_forced = 0;
	uint64_t sync_dl_submit_us = 0;
	uint64_t sync_dl_master_wait_us = 0;
	uint64_t sync_dl_priority_wait_us = 0;
	uint64_t sync_dl_publish_us = 0;
	// Slice-boundary counters: BufferFlush sites only (not actual vkQueueSubmit calls).
	// Graphics/compute split into completed vs yielded progress, plus suspend points.
	uint32_t slice_graphics_completed = 0;
	uint32_t slice_graphics_yielded = 0;
	uint32_t slice_compute_completed = 0;
	uint32_t slice_compute_yielded = 0;
	uint32_t slice_suspend_points = 0;
	uint32_t draw_targets_clear_wait_count = 0;
	uint64_t draw_targets_bind_us = 0;
	uint64_t draw_bindings_us = 0;
	// Step 4: Bind sub-timers (measured in PrepareBindings/FindBuffers/Rebind*/CommitBindings).
	//  - bind_resolve: PrepareBindings (texture desc compute + sampler lookup).
	//  - bind_buffers: FindBuffers (descriptor decode + buffer cache lookup).
	//  - bind_images: RebindImages (image view lookup).
	//  - bind_commit: CommitBindings (descriptor writes).
	// Step 4b adds: bind_rebind (whole RebindBuffers), bind_storage (its
	// NativeStorageBuffer loop), bind_srt_upload (its flattened-SRT NativeUpload +
	// FillFlatSlotsOnGpu), bind_sd_upload (its shader_data NativeUpload), bind_bda
	// (PrepareBda) and bind_targets (color-target loop) in PrepareGraphicsBindings.
	uint64_t draw_bind_resolve_us = 0;
	uint64_t draw_bind_buffers_us = 0;
	uint64_t draw_bind_images_us = 0;
	uint64_t draw_bind_commit_us = 0;
	// Step 4b: deeper Bind split (measured in PrepareGraphicsBindings/RebindBuffers).
	//  - bind_rebind: whole RebindBuffers per stage.
	//  - bind_storage: NativeStorageBuffer loop inside RebindBuffers.
	//  - bind_srt_upload: NativeUpload flattened_srt + FillFlatSlotsOnGpu (both branches).
	//  - bind_sd_upload: NativeUpload shader_data.
	//  - bind_bda: PrepareBda.
	//  - bind_targets: color-target FindImage/BindRenderTarget loop.
	uint64_t draw_bind_rebind_us = 0;
	uint64_t draw_bind_storage_us = 0;
	uint64_t draw_bind_srt_upload_us = 0;
	uint64_t draw_bind_sd_upload_us = 0;
	uint64_t draw_bind_bda_us = 0;
	uint64_t draw_bind_targets_us = 0;
	// Step 4b: Mat split (measured around coarse phases of MaterializeResources).
	//  - mat_srt_walk: SRT walker RefreshFlatBuffer (descriptor/SRT value graph).
	//  - mat_guest_read: guest-memory reads serving the walk (armed only around the
	//    slow path; the clean-page fast path stays uninstrumented).
	//  - mat_spec_write: snapshot writes + specialization build after the walk.
	uint64_t draw_state_mat_srt_walk_us = 0;
	uint64_t draw_state_mat_guest_read_us = 0;
	uint64_t draw_state_mat_spec_write_us = 0;
	// Step 4: bind-cache (KYTY_OPT_BIND_CACHE) hit/miss counters.
	uint32_t bind_cache_hits = 0;
	uint32_t bind_cache_misses = 0;
	// Step 4c: BDA clean-skip range counters (draw-only via DrawBindTelemetryScope).
	//  - bda_ranges_checked: mapped ranges probed with the CPU-clean hint.
	//  - bda_ranges_skipped: ranges that skipped the region walk (hint clean).
	//  - bda_ranges_walked: ranges that took the legacy region walk.
	uint32_t bda_ranges_checked = 0;
	uint32_t bda_ranges_skipped = 0;
	uint32_t bda_ranges_walked = 0;
	uint64_t draw_vertex_us = 0;
	uint64_t draw_pipe_us = 0;
	uint64_t draw_emit_us = 0;
	uint64_t wait_readmem_us = 0;
	uint64_t wait_flip_us = 0;
	uint64_t wait_gc_us = 0;
	uint32_t gpusync_clear_count = 0;
	uint32_t gpusync_texture_count = 0;
	uint32_t gpusync_buffer_count = 0;
	uint32_t gpusync_other_count = 0;
	// Regression analysis: how much the sync readback downloads beyond its range
	// (ReadbackWindow widens every sync read to ~4 MB) and how many GPU-dirty
	// tracker bytes/pieces its window carries.
	uint64_t clear_read_extra_bytes = 0;
	uint64_t clear_read_requested_bytes = 0;
	uint32_t clear_read_range_count = 0;
	uint32_t clear_gpu_predicate_count = 0;
	uint32_t clear_gpu_predicate_fallback_count = 0;
	uint32_t clear_gpu_predicate_nocand_count = 0;
	// Step 3b: per-reason fallback counters for TryMaterializeColorClearGpu.
	uint32_t clear_fb_no_cond = 0;
	uint32_t clear_fb_resources = 0;
	uint32_t clear_fb_unsupported = 0;
	uint32_t clear_fb_no_cand = 0;
	uint32_t clear_fb_too_many = 0;
	uint32_t clear_fb_meta_range = 0;
	uint32_t clear_fb_depth_msaa = 0;
	uint32_t clear_fb_int_format = 0;
	uint32_t clear_fb_pipeline = 0;
	uint32_t draws = 0;
	uint32_t dispatches = 0;
	uint32_t flushes = 0;
	// Compact per-cause accounting for real vkQueueSubmit calls: count, CPU time
	// (Submit+BeginNext, never the GPU wait), and batch volume (draws+dispatches
	// recorded since the previous submit). Indexed by SubmitCause; the Wait/Finish
	// buckets additionally add submit_wait_* (m_master.Wait() only -- pure GPU
	// idle, measured separately from the CPU submit interval). GPU thread only.
	static constexpr size_t kSubmitCauses = static_cast<size_t>(SubmitCause::Count);
	uint32_t submit_count[kSubmitCauses] = {};
	uint64_t submit_cpu_us[kSubmitCauses] = {};
	uint32_t submit_batch_draws[kSubmitCauses] = {};
	uint32_t submit_batch_dispatches[kSubmitCauses] = {};
	uint32_t submit_wait_count = 0;
	uint64_t submit_wait_us = 0;
	uint32_t draw_state_l1_hits = 0;
	uint32_t draw_state_l1_misses = 0;
	uint32_t color_resolves = 0;
	uint32_t depth_resolves = 0;
	uint32_t cache_lookups = 0;
	uint32_t cache_same_backing_hits = 0;
	uint32_t cache_overlap_hits = 0;
	uint32_t cache_inserts = 0;
	uint32_t cache_expands = 0;
	uint32_t cache_refreshes = 0;
	uint32_t cache_clears = 0;
	uint32_t cache_candidates_total = 0;
	uint32_t cache_candidates_max = 0;
	uint32_t cache_images = 0;
	// Command-recording dedup (RecordedStateCache): vkCmdBindPipeline and dynamic-state
	// commands actually emitted vs skipped as redundant. Compact counters only, no
	// per-command timing. GPU thread only.
	uint32_t record_pipeline_emitted = 0;
	uint32_t record_pipeline_skipped = 0;
	uint32_t record_dynamic_emitted = 0;
	uint32_t record_dynamic_skipped = 0;
};

inline FrameTelemetry& GetFrameTelemetry() {
	static thread_local FrameTelemetry instance;
	return instance;
}

inline uint64_t TelemetryFrequencyHz() {
	static const uint64_t frequency = [] {
		uint64_t value = 0;
		SysQueryPerformanceFrequency(&value);
		return value;
	}();
	return frequency;
}

inline uint64_t TelemetryNowUs() {
	uint64_t counter = 0;
	SysQueryPerformanceCounter(&counter);
	const uint64_t frequency = TelemetryFrequencyHz();
	if (frequency == 0) {
		return 0;
	}
	// 64-bit only: lld-link has no __udivti3 for 128-bit division. The remainder term
	// stays below frequency (~10 MHz), so its scaled value never overflows.
	return (counter / frequency) * 1000000ull + ((counter % frequency) * 1000000ull) / frequency;
}

inline void TelemetryAddPm4Op(uint8_t opcode, uint64_t delta_us) {
	GetFrameTelemetry().pm4_op_us[opcode] += delta_us;
}

inline void TelemetryAddDrawRecord(uint64_t delta_us) {
	GetFrameTelemetry().draw_record_us += delta_us;
}

inline void TelemetryAddDrawState(uint64_t delta_us) {
	GetFrameTelemetry().draw_state_us += delta_us;
}

inline void TelemetryAddDrawStatePrepareVs(uint64_t delta_us) {
	GetFrameTelemetry().draw_state_prepare_vs_us += delta_us;
}

inline void TelemetryAddDrawStatePreparePs(uint64_t delta_us) {
	GetFrameTelemetry().draw_state_prepare_ps_us += delta_us;
}

inline void TelemetryAddDrawStateKeyFind(uint64_t delta_us) {
	GetFrameTelemetry().draw_state_key_find_us += delta_us;
}

inline void TelemetryAddDrawStateMaterialize(uint64_t delta_us) {
	GetFrameTelemetry().draw_state_materialize_us += delta_us;
}

inline void TelemetryAddDrawStateMatRefresh(uint64_t delta_us) {
	GetFrameTelemetry().draw_state_mat_refresh_us += delta_us;
}

inline void TelemetryAddDrawStateMatSpecialize(uint64_t delta_us) {
	GetFrameTelemetry().draw_state_mat_specialize_us += delta_us;
}

inline void TelemetryAddDrawStatePermutation(uint64_t delta_us) {
	GetFrameTelemetry().draw_state_permutation_us += delta_us;
}

inline void TelemetryAddDrawStateTargets(uint64_t delta_us) {
	GetFrameTelemetry().draw_state_targets_us += delta_us;
}

inline void TelemetryAddDrawTargetsColor(uint64_t delta_us) {
	GetFrameTelemetry().draw_targets_color_us += delta_us;
}

inline void TelemetryAddDrawTargetsDepth(uint64_t delta_us) {
	GetFrameTelemetry().draw_targets_depth_us += delta_us;
}

inline void TelemetryAddDrawTargetsCacheLookup(uint64_t delta_us) {
	GetFrameTelemetry().draw_targets_cache_lookup_us += delta_us;
}

inline void TelemetryAddDrawTargetsCacheInsert(uint64_t delta_us) {
	GetFrameTelemetry().draw_targets_cache_insert_us += delta_us;
}

inline void TelemetryAddDrawTargetsOverlap(uint64_t delta_us) {
	GetFrameTelemetry().draw_targets_overlap_us += delta_us;
}

inline void TelemetryAddDrawTargetsExpand(uint64_t delta_us) {
	GetFrameTelemetry().draw_targets_expand_us += delta_us;
}

inline void TelemetryAddDrawTargetsRefresh(uint64_t delta_us) {
	GetFrameTelemetry().draw_targets_refresh_us += delta_us;
}

inline void TelemetryAddDrawTargetsClear(uint64_t delta_us) {
	GetFrameTelemetry().draw_targets_clear_us += delta_us;
}

inline void TelemetryAddDrawTargetsClearTracker(uint64_t delta_us) {
	GetFrameTelemetry().draw_targets_clear_tracker_us += delta_us;
}

inline void TelemetryAddDrawTargetsClearGpuWait(uint64_t delta_us) {
	GetFrameTelemetry().draw_targets_clear_gpu_wait_us += delta_us;
}

inline void TelemetryAddDrawTargetsClearReadback(uint64_t delta_us) {
	GetFrameTelemetry().draw_targets_clear_readback_us += delta_us;
}

inline void TelemetryAddDrawTargetsClearWriteback(uint64_t delta_us) {
	GetFrameTelemetry().draw_targets_clear_writeback_us += delta_us;
}

inline void TelemetryAddReadmemFlushSubmit(uint64_t delta_us) {
	GetFrameTelemetry().readmem_flush_submit_us += delta_us;
}

inline void TelemetryAddReadmemFenceWait(uint64_t delta_us) {
	GetFrameTelemetry().readmem_fence_wait_us += delta_us;
}

inline void TelemetryAddReadmemCopyReadback(uint64_t delta_us) {
	GetFrameTelemetry().readmem_copy_readback_us += delta_us;
}

inline void TelemetryCountSyncDownloadCall(bool forced) {
	if (!SyncDownloadDiagEnabled()) {
		return;
	}
	auto& telemetry = GetFrameTelemetry();
	telemetry.sync_dl_calls++;
	if (forced) {
		telemetry.sync_dl_forced++;
	} else {
		telemetry.sync_dl_already_submitted++;
	}
}

inline void TelemetryAddSyncDownloadSubmit(uint64_t delta_us) {
	if (!SyncDownloadDiagEnabled()) {
		return;
	}
	GetFrameTelemetry().sync_dl_submit_us += delta_us;
}

inline void TelemetryAddSyncDownloadMasterWait(uint64_t delta_us) {
	if (!SyncDownloadDiagEnabled()) {
		return;
	}
	GetFrameTelemetry().sync_dl_master_wait_us += delta_us;
}

inline void TelemetryAddSyncDownloadPriorityWait(uint64_t delta_us) {
	if (!SyncDownloadDiagEnabled()) {
		return;
	}
	GetFrameTelemetry().sync_dl_priority_wait_us += delta_us;
}

inline void TelemetryAddSyncDownloadPublish(uint64_t delta_us) {
	if (!SyncDownloadDiagEnabled()) {
		return;
	}
	GetFrameTelemetry().sync_dl_publish_us += delta_us;
}

inline void TelemetryCountSliceGraphics(bool completed) {
	if (!SyncDownloadDiagEnabled()) {
		return;
	}
	auto& telemetry = GetFrameTelemetry();
	if (completed) {
		telemetry.slice_graphics_completed++;
	} else {
		telemetry.slice_graphics_yielded++;
	}
}

inline void TelemetryCountSliceCompute(bool completed) {
	if (!SyncDownloadDiagEnabled()) {
		return;
	}
	auto& telemetry = GetFrameTelemetry();
	if (completed) {
		telemetry.slice_compute_completed++;
	} else {
		telemetry.slice_compute_yielded++;
	}
}

inline void TelemetryCountSliceSuspendPoint() {
	if (!SyncDownloadDiagEnabled()) {
		return;
	}
	GetFrameTelemetry().slice_suspend_points++;
}

inline void TelemetryCountClearGpuWait() {
	GetFrameTelemetry().draw_targets_clear_wait_count++;
}

inline void TelemetryCountGpusyncClear() {
	GetFrameTelemetry().gpusync_clear_count++;
}

inline void TelemetryCountGpusyncTexture() {
	GetFrameTelemetry().gpusync_texture_count++;
}

inline void TelemetryCountGpusyncBuffer() {
	GetFrameTelemetry().gpusync_buffer_count++;
}

inline void TelemetryCountGpusyncOther() {
	GetFrameTelemetry().gpusync_other_count++;
}

// Regression analysis: a sync readback with a (window_bytes) download window
// carrying (dirty_bytes) GPU-dirty tracker bytes in (ranges) pieces.
inline void TelemetryAddClearReadWindow(uint64_t window_bytes, uint64_t dirty_bytes, uint32_t ranges) {
	auto& telemetry = GetFrameTelemetry();
	telemetry.clear_read_requested_bytes += window_bytes;
	telemetry.clear_read_extra_bytes += dirty_bytes;
	telemetry.clear_read_range_count += ranges;
}

inline void TelemetryCountClearGpuPredicate() {
	GetFrameTelemetry().clear_gpu_predicate_count++;
}

inline void TelemetryCountClearGpuPredicateFallback() {
	GetFrameTelemetry().clear_gpu_predicate_fallback_count++;
}

inline void TelemetryCountClearGpuPredicateNocand() {
	GetFrameTelemetry().clear_gpu_predicate_nocand_count++;
}

inline void TelemetryCountClearFbNoCond() {
	GetFrameTelemetry().clear_fb_no_cond++;
}

inline void TelemetryCountClearFbResources() {
	GetFrameTelemetry().clear_fb_resources++;
}

inline void TelemetryCountClearFbUnsupported() {
	GetFrameTelemetry().clear_fb_unsupported++;
}

inline void TelemetryCountClearFbNoCand() {
	GetFrameTelemetry().clear_fb_no_cand++;
}

inline void TelemetryCountClearFbTooMany() {
	GetFrameTelemetry().clear_fb_too_many++;
}

inline void TelemetryCountClearFbMetaRange() {
	GetFrameTelemetry().clear_fb_meta_range++;
}

inline void TelemetryCountClearFbDepthMsaa() {
	GetFrameTelemetry().clear_fb_depth_msaa++;
}

inline void TelemetryCountClearFbIntFormat() {
	GetFrameTelemetry().clear_fb_int_format++;
}

inline void TelemetryCountClearFbPipeline() {
	GetFrameTelemetry().clear_fb_pipeline++;
}

inline void TelemetryAddDrawTargetsBind(uint64_t delta_us) {
	GetFrameTelemetry().draw_targets_bind_us += delta_us;
}

inline void TelemetryCountDrawStateL1Hit() {
	GetFrameTelemetry().draw_state_l1_hits++;
}

inline void TelemetryCountDrawStateL1Miss() {
	GetFrameTelemetry().draw_state_l1_misses++;
}

inline void TelemetryCountColorResolve() {
	GetFrameTelemetry().color_resolves++;
}

inline void TelemetryCountDepthResolve() {
	GetFrameTelemetry().depth_resolves++;
}

inline void TelemetryCountCacheLookup() {
	GetFrameTelemetry().cache_lookups++;
}

inline void TelemetryCountCacheSameBackingHit() {
	GetFrameTelemetry().cache_same_backing_hits++;
}

inline void TelemetryCountCacheOverlapHit() {
	GetFrameTelemetry().cache_overlap_hits++;
}

inline void TelemetryCountCacheInsert() {
	GetFrameTelemetry().cache_inserts++;
}

inline void TelemetryCountCacheExpand() {
	GetFrameTelemetry().cache_expands++;
}

inline void TelemetryCountCacheRefresh() {
	GetFrameTelemetry().cache_refreshes++;
}

inline void TelemetryCountCacheClear() {
	GetFrameTelemetry().cache_clears++;
}

inline void TelemetryAddCacheCandidates(uint32_t count) {
	auto& telemetry = GetFrameTelemetry();
	telemetry.cache_candidates_total += count;
	telemetry.cache_candidates_max = std::max(telemetry.cache_candidates_max, count);
}

inline void TelemetrySetCacheImages(uint32_t count) {
	GetFrameTelemetry().cache_images = count;
}

inline void TelemetryAddDrawBindings(uint64_t delta_us) {
	GetFrameTelemetry().draw_bindings_us += delta_us;
}

// Step 4: draw-only attribution for Bind sub-timers. PrepareBindings/FindBuffers/Rebind*
// serve compute dispatches too, while draw_bindings_us accumulates draw work only, so the
// sub-timers accumulate only inside a DrawBindTelemetryScope (set around the draw bind block
// in ExecutePreparedDraw). CommitBindings gates on the graphics bind point instead.
inline bool& DrawBindTelemetryScope() {
	static thread_local bool active = false;
	return active;
}

inline void TelemetryAddDrawBindResolve(uint64_t delta_us) {
	if (DrawBindTelemetryScope()) {
		GetFrameTelemetry().draw_bind_resolve_us += delta_us;
	}
}

inline void TelemetryAddDrawBindBuffers(uint64_t delta_us) {
	if (DrawBindTelemetryScope()) {
		GetFrameTelemetry().draw_bind_buffers_us += delta_us;
	}
}

inline void TelemetryAddDrawBindImages(uint64_t delta_us) {
	if (DrawBindTelemetryScope()) {
		GetFrameTelemetry().draw_bind_images_us += delta_us;
	}
}

inline void TelemetryAddDrawBindCommit(uint64_t delta_us) {
	GetFrameTelemetry().draw_bind_commit_us += delta_us;
}

inline void TelemetryAddDrawBindRebind(uint64_t delta_us) {
	if (DrawBindTelemetryScope()) {
		GetFrameTelemetry().draw_bind_rebind_us += delta_us;
	}
}

inline void TelemetryAddDrawBindStorage(uint64_t delta_us) {
	if (DrawBindTelemetryScope()) {
		GetFrameTelemetry().draw_bind_storage_us += delta_us;
	}
}

inline void TelemetryAddDrawBindSrtUpload(uint64_t delta_us) {
	if (DrawBindTelemetryScope()) {
		GetFrameTelemetry().draw_bind_srt_upload_us += delta_us;
	}
}

inline void TelemetryAddDrawBindSdUpload(uint64_t delta_us) {
	if (DrawBindTelemetryScope()) {
		GetFrameTelemetry().draw_bind_sd_upload_us += delta_us;
	}
}

inline void TelemetryAddDrawBindBda(uint64_t delta_us) {
	if (DrawBindTelemetryScope()) {
		GetFrameTelemetry().draw_bind_bda_us += delta_us;
	}
}

inline void TelemetryAddDrawBindTargets(uint64_t delta_us) {
	if (DrawBindTelemetryScope()) {
		GetFrameTelemetry().draw_bind_targets_us += delta_us;
	}
}

inline void TelemetryAddDrawStateMatSrtWalk(uint64_t delta_us) {
	GetFrameTelemetry().draw_state_mat_srt_walk_us += delta_us;
}

inline void TelemetryAddDrawStateMatGuestRead(uint64_t delta_us) {
	GetFrameTelemetry().draw_state_mat_guest_read_us += delta_us;
}

inline void TelemetryAddDrawStateMatSpecWrite(uint64_t delta_us) {
	GetFrameTelemetry().draw_state_mat_spec_write_us += delta_us;
}

inline void TelemetryCountBindCacheHit() {
	GetFrameTelemetry().bind_cache_hits++;
}

inline void TelemetryCountBindCacheMiss() {
	GetFrameTelemetry().bind_cache_misses++;
}

// Step 4c: BDA clean-skip range counters. Draw-only via DrawBindTelemetryScope, kept
// separate from the bind-cache hit/miss counters above.
inline void TelemetryCountBdaRangeChecked() {
	if (DrawBindTelemetryScope()) {
		GetFrameTelemetry().bda_ranges_checked++;
	}
}

inline void TelemetryCountBdaRangeSkipped() {
	if (DrawBindTelemetryScope()) {
		GetFrameTelemetry().bda_ranges_skipped++;
	}
}

inline void TelemetryCountBdaRangeWalked() {
	if (DrawBindTelemetryScope()) {
		GetFrameTelemetry().bda_ranges_walked++;
	}
}

inline void TelemetryAddDrawVertex(uint64_t delta_us) {
	GetFrameTelemetry().draw_vertex_us += delta_us;
}

inline void TelemetryAddDrawPipe(uint64_t delta_us) {
	GetFrameTelemetry().draw_pipe_us += delta_us;
}

inline void TelemetryAddDrawEmit(uint64_t delta_us) {
	GetFrameTelemetry().draw_emit_us += delta_us;
}

inline void TelemetryAddWaitReadMem(uint64_t delta_us) {
	GetFrameTelemetry().wait_readmem_us += delta_us;
}

inline void TelemetryAddWaitFlip(uint64_t delta_us) {
	GetFrameTelemetry().wait_flip_us += delta_us;
}

inline void TelemetryAddWaitGC(uint64_t delta_us) {
	GetFrameTelemetry().wait_gc_us += delta_us;
}

inline void TelemetryCountDraw() {
	GetFrameTelemetry().draws++;
}

inline void TelemetryCountDispatch() {
	GetFrameTelemetry().dispatches++;
}

inline void TelemetryCountFlush() {
	GetFrameTelemetry().flushes++;
}

// Number of SubmitCause buckets; kept next to the recorders so the report loop in
// graphicsRun.cpp cannot drift apart from the FrameTelemetry array size.
inline constexpr size_t SubmitCauseCount() {
	return static_cast<size_t>(SubmitCause::Count);
}

// Records one real vkQueueSubmit: cause bucket, Submit()+BeginNext() CPU time, and the
// batch volume (draws+dispatches recorded since the previous submit, as deltas against
// the frame totals). GPU thread only; the present-thread swapchain scheduler never
// touches FrameTelemetry. TelemetryEndFrame zeroes the frame totals every 60 frames,
// so the scheduler's batch bases must reset with them (see ResetSubmitBatchBase) --
// otherwise the first batch after a reset clamps to zero.
inline void TelemetryCountSubmit(SubmitCause cause, uint64_t cpu_us, uint32_t batch_draws,
                                 uint32_t batch_dispatches) {
	if (!SubmitTelemetryEnabled()) {
		return;
	}
	auto& telemetry = GetFrameTelemetry();
	const size_t index = static_cast<size_t>(cause);
	if (index >= FrameTelemetry::kSubmitCauses) {
		return;
	}
	telemetry.submit_count[index]++;
	telemetry.submit_cpu_us[index] += cpu_us;
	telemetry.submit_batch_draws[index] += batch_draws;
	telemetry.submit_batch_dispatches[index] += batch_dispatches;
}

inline void TelemetryCountSubmitWait(uint64_t wait_us) {
	if (!SubmitTelemetryEnabled()) {
		return;
	}
	auto& telemetry = GetFrameTelemetry();
	telemetry.submit_wait_count++;
	telemetry.submit_wait_us += wait_us;
}

void TelemetryEndFrame();

enum class ContextStateOperation : uint32_t {
	Clear     = 0,
	Push      = 1,
	Pop       = 2,
	PushClear = 3,
};

class Pm4Execution {
public:
	[[nodiscard]] bool MadeProgress() const noexcept { return m_made_progress; }

private:
	friend class CommandProcessor;

	struct BufferCursor {
		std::span<const uint32_t> commands;
		uint32_t                  offset_dw = 0;
	};

	std::vector<BufferCursor> m_buffer_stack;
	std::span<const uint32_t> m_next_buffer;
	bool                      m_chain         = false;
	bool                      m_suspended     = false;
	bool                      m_made_progress = false;
};

class CommandProcessor {
public:
	struct FlipInfo {
		int     handle    = 0;
		int     index     = 0;
		int     flip_mode = 0;
		int64_t flip_arg  = 0;
	};

	CommandProcessor(RenderContext& renderer, int interrupt_event_id)
	    : m_renderer(renderer), m_interrupt_event_id(interrupt_event_id) {}
	~CommandProcessor() = default;

	KYTY_CLASS_NO_COPY(CommandProcessor);

	void Reset();
	void ApplyContextStateOperation(ContextStateOperation operation);

	void            BufferInit();
	void            BufferFlush(SubmitCause cause = SubmitCause::SliceFlush);
	void            BufferWait();
	// See CommandScheduler::CompleteReleaseMemWrite -- only safe for a RELEASE_MEM that already
	// wrote its guest-visible value synchronously and scheduled no interrupt callback.
	void            CompleteReleaseMemWrite();
	// See CommandScheduler::CompleteReleaseMemInterrupt -- for a RELEASE_MEM that DOES request a
	// guest interrupt/event; uses a smaller batch bound as a hedge against delaying it.
	// merged=true when the legacy immediate BufferFlush was folded into this window under
	// KYTY_OPT_SUBMIT_MERGE_RELEASE_FLUSH (telemetry cause ReleaseFlush), false for the
	// pre-existing batched path (cause ReleaseIrq).
	void            CompleteReleaseMemInterrupt(bool merged = false);
	// See CommandScheduler::CompleteDraw -- called after every DrawIndex/DrawAuto.
	void            CompleteDraw();
	HW::Context&    GetCtx() { return m_ctx; }
	HW::UserConfig& GetUcfg() { return m_ucfg; }
	HW::Shader&     GetShCtx() { return m_sh_ctx; }

	void SetIndexType(uint32_t index_type_and_size);
	void SetIndexBaseAddress(uint64_t index_base_addr);
	void SetIndexBufferSize(uint32_t index_buffer_size);
	void SetDrawIndirectArgsBaseAddress(uint64_t draw_indirect_args_base_addr);
	void SetDispatchIndirectArgsBaseAddress(uint64_t dispatch_indirect_args_base_addr);
	[[nodiscard]] uint64_t GetDispatchIndirectArgsBaseAddress() const {
		return m_dispatch_indirect_args_base_addr;
	}
	void SetNumInstances(uint32_t num_instances);
	void DrawIndex(DrawIndexArgs args);
	void DrawIndexOffset(uint32_t index_offset, uint32_t index_count);
	void DrawIndexAuto(DrawAutoArgs args);
	void DrawIndirect(uint32_t data_offset, uint32_t draw_initiator, bool indexed);
	void DrawIndirectMulti(uint32_t data_offset, uint32_t max_count_or_count,
	                       const volatile uint32_t* count_addr, uint32_t stride_in_bytes,
	                       uint32_t draw_initiator, bool indexed, uint32_t draw_index_register);
	void WriteAtEndOfPipe32(uint32_t cache_policy, uint32_t event_write_dest,
	                        uint32_t eop_event_type, uint32_t cache_action, uint32_t event_index,
	                        uint32_t event_write_source, void* dst_gpu_addr, uint32_t value,
	                        uint32_t interrupt_selector, uint32_t interrupt_context_id = 0);
	void WriteAtEndOfPipe64(uint32_t cache_policy, uint32_t event_write_dest,
	                        uint32_t eop_event_type, uint32_t cache_action, uint32_t event_index,
	                        uint32_t event_write_source, void* dst_gpu_addr, uint64_t value,
	                        uint32_t interrupt_selector, uint32_t interrupt_context_id = 0);
	void Flip();
	void Flip(void* dst_gpu_addr, uint32_t value);
	void FlipWithInterrupt(uint32_t eop_event_type, uint32_t cache_action, void* dst_gpu_addr,
	                       uint32_t value);
	void PrepareCpuFlip(uint64_t request_id);
	void SynchronizeGpu();
	void EmitGlobalBarrier();
	void TriggerEopEventAtEndOfPipe(uint32_t interrupt_context_id);
	void DispatchDirect(uint32_t thread_group_x, uint32_t thread_group_y, uint32_t thread_group_z,
	                    uint32_t mode);
	void DispatchIndirect(uint64_t args_addr, uint32_t mode);
	void WaitFlipDone(uint32_t video_out_handle, uint32_t display_buffer_index);
	void TriggerEvent(uint32_t event_type, uint32_t event_index, uint64_t event_address = 0);

	void SetUserDataMarker(HW::UserSgprType type) { m_user_data_marker = type; }
	[[nodiscard]] HW::UserSgprType GetUserDataMarker() const { return m_user_data_marker; }

	void ResetDeCe();
	void SetCeComplete(bool complete) { m_ce_complete = complete; }
	void WaitCe();
	void WaitDeDiff(uint32_t diff);
	void WaitForRewind(bool valid);
	void IncrementDe();
	void IncrementCe();

	void WriteConstRam(uint32_t offset, const uint32_t* src, uint32_t dw_num);
	void DumpConstRam(uint32_t* dst, uint32_t offset, uint32_t dw_num);

	template <typename T>
	void WaitRegMem(uint32_t func, const T* addr, T ref, T mask, uint32_t poll, uint32_t wait_op);
	// A label as the command processor sees it: end-of-pipe writes count once recorded, even
	// before the CPU can see them (GuestGpu::LabelsAtCompletion).
	template <typename T>
	[[nodiscard]] T ReadLabel(const volatile T* addr) const;
	void WriteData(uint32_t* dst, const uint32_t* src, uint32_t dw_num, uint32_t write_control);
	void WriteReferenceClock(uint64_t dst_address, uint32_t num_bytes);
	void DmaData(uint8_t engine, uint8_t dst_sel, uint8_t dst_cache_policy,
	             uint64_t dst_address_or_offset, uint8_t src_sel, uint8_t src_cache_policy,
	             uint64_t src_address_or_offset_or_immediate, uint32_t num_bytes,
	             uint8_t wait_for_previous, uint8_t write_confirm, uint8_t block_engine);
	void SetPredication(uint32_t condition, uint32_t op, uint32_t wait_op,
	                    const volatile void* address, uint32_t count_in_dwords);
	[[nodiscard]] bool ShouldSkipPredicatedPackets() const { return m_predicate_skip; }

	Pm4ProcessResult Process(Pm4Execution& execution, std::span<const uint32_t> commands);
	void             ProcessIndirectBuffer(std::span<const uint32_t> commands, bool chain);

	void SetFlip(const FlipInfo& flip) { m_flip = flip; }

	[[nodiscard]] uint64_t GetSubmitId() const { return m_submit_id; }
	void                   SetSubmitId(uint64_t submit_id) { m_submit_id = submit_id; }
	[[nodiscard]] bool     IsAsyncComputeQueue() const { return m_interrupt_event_id >= 0x20; }
	// Public for TelemetryEndFrame's per-submit batch-base reset (GPU thread only).
	CommandScheduler&   GetScheduler() const { return m_renderer.GetCommandScheduler(); }

private:
	template <typename T>
	void WriteAtEndOfPipe(uint32_t cache_policy, uint32_t event_write_dest, uint32_t eop_event_type,
	                      uint32_t cache_action, uint32_t event_index, uint32_t event_write_source,
	                      void* dst_gpu_addr, T value, uint32_t interrupt_selector,
	                      uint32_t interrupt_context_id);
	void ProcessPm4(Pm4Execution& execution);
	void SuspendPm4();
	CommandBuffer&      CurrentBuffer() { return GetScheduler().Current(); }

	RenderContext&   m_renderer;
	HW::Context      m_ctx;
	HW::Context      m_saved_ctx;
	bool             m_context_state_pushed = false;
	HW::UserConfig   m_ucfg;
	HW::Shader       m_sh_ctx;
	HW::UserSgprType m_user_data_marker                 = HW::UserSgprType::Unknown;
	uint32_t         m_index_type_and_size              = 0;
	uint32_t         m_index_buffer_size                = 0;
	uint64_t         m_index_base_addr                  = 0;
	uint64_t         m_draw_indirect_args_base_addr     = 0;
	uint64_t         m_dispatch_indirect_args_base_addr = 0;
	// Persistent draw state: indirect draws update it for subsequent draws.
	uint32_t m_num_instances = 1;
	// The DrawIndirectArgs whose instance count NUM_INSTANCES holds after a GPU-read indirect draw;
	// read on the CPU only if a later draw takes its instance count from there.
	uint64_t m_num_instances_args = 0;
	[[nodiscard]] uint32_t NumInstances();

	uint32_t m_de_count    = 0;
	uint32_t m_ce_count    = 0;
	bool     m_ce_complete = false;

	uint32_t m_const_ram[0x3000] = {0};

	FlipInfo  m_flip;
	const int m_interrupt_event_id;
	uint64_t  m_submit_id                   = 0;
	uint64_t  m_synthetic_occlusion_counter = 0;
	bool      m_predicate_skip              = false;
};

} // namespace Libs::Graphics

#endif // GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMMAND_PROCESSOR_H
