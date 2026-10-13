#include "graphics/host_gpu/renderer/cache/textureCache.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/image/tiler.h"
#include "gpu_tiler_shaders/clear_predicate_check_spv.h"
#include "gpu_tiler_shaders/clear_predicate_fill_vert_spv.h"
#include "gpu_tiler_shaders/clear_predicate_fill_frag_spv.h"
#include "gpu_tiler_shaders/clear_predicate_fill_frag_uint_spv.h"
#include "gpu_tiler_shaders/clear_predicate_fill_frag_sint_spv.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/commandRecorder.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <tuple>
#include <vulkan/vulkan_format_traits.hpp>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

constexpr uint64_t NumFramesBeforeRemoval = 32;
// Overlapping aliases (transient render targets sharing memory) stay registered until unused for
// this many presented frames as well; see AliasAgeByFrames.
constexpr uint64_t AliasFramesBeforeRemoval = 4;

// An overlapped alias was freed once unused for NumFramesBeforeRemoval scheduler ticks. A title
// submits hundreds of times a frame, so that is usually within the frame that used it: the
// aliases of a transient image pool (Wolverine: ~4 a frame) were destroyed and recreated from
// guest memory every frame. They now also have to be unused for AliasFramesBeforeRemoval
// presented frames, and live aliases follow a single-owner rule (CommitGpuWrite).
// KYTY_IMAGE_ALIAS_AGE=0 (or =ticks) restores the tick rule alone.
[[nodiscard]] bool AliasAgeByFrames() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_IMAGE_ALIAS_AGE");
		return value == nullptr ||
		       (std::strcmp(value, "0") != 0 && std::strcmp(value, "ticks") != 0);
	}();
	return enabled;
}

// The collector writes a GPU-written image back to guest memory before it frees it (under
// memory pressure). Whether the image is CPU-dirty is checked when the copy is recorded, but the
// bytes land only once the GPU is done with that tick, and the image is untracked by then: a
// guest that reused the memory in between (a retired render target becoming a CPU-written vertex
// table, as in Jetsku's 919e712a) had it overwritten with stale pixels. The guest bytes are hashed
// when the copy is recorded and again before it is published; if they changed, the write-back
// is dropped. KYTY_EVICT_WRITEBACK_GUARD=0 publishes unconditionally again.
[[nodiscard]] bool EvictWriteBackGuard() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_EVICT_WRITEBACK_GUARD");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// KYTY_LOG_CLEAR_WAIT=1 logs one line per color-clear GPU wait (measurements only).
[[nodiscard]] bool ClearWaitLogEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_LOG_CLEAR_WAIT");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

// KYTY_EXPERIMENT_SKIP_CLEAR_WAIT=1 skips the synchronous readback in MaterializeColorClear.
// Experimental: measures the FPS ceiling without the GPU wait. Clear artifacts expected.
[[nodiscard]] bool SkipClearWaitEnabled() {
	const auto* value = std::getenv("KYTY_EXPERIMENT_SKIP_CLEAR_WAIT");
	return value != nullptr && value[0] == '1';
}

// Step 3/3b UNSAFE (default off): KYTY_OPT_CLEAR_GPU_PREDICATE=1 enables the predicated
// GPU color-clear path. Even step 3 (07b1899) flickers and loses the device on fight
// entry: the synchronous ReadMemory in MaterializeColorClear downloads a ~4 MB window
// (~10 MB dirty/frame) of GPU-written data into guest memory and clears tracker state,
// and other readers depend on those side effects. Any path that skips it crashes.
// Keep the code for now; the safe default is the synchronous CPU readback path.
[[nodiscard]] bool ClearGpuPredicateEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_OPT_CLEAR_GPU_PREDICATE");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

// Step 3b: KYTY_OPT_CLEAR_SKIP_NOCAND=1 skips the readback for images whose metadata
// codes never decode. Default off: the sync readback's side effects (window download +
// per-page tracker cleanup) turned out to be load-bearing (fighter-select Device lost).
[[nodiscard]] bool ClearSkipNocandEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_OPT_CLEAR_SKIP_NOCAND");
		// Default off: the sync readback's side effects (window download +
		// per-page tracker cleanup) are load-bearing, see step 3b analysis.
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

// KYTY_OPT_CLEAR_SCRATCH=0 restores the per-slice std::vector path in MaterializeColorClear
// (default: enabled). The scratch path removes per-slice heap work from the uniformity check
// without touching the synchronous ReadMemory wait above it: chunked guest-backing reads with
// early exit on the first non-uniform byte, and chunked 0xFF write-back. Same bytes read and
// written, same ClearImage/InvalidateMemory order, same EXITs on unreadable backing.
[[nodiscard]] bool ClearScratchEnabled() {
	static const bool enabled = [] {
		const auto* value = std::getenv("KYTY_OPT_CLEAR_SCRATCH");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

constexpr uint64_t kClearScratchChunk = 64 * 1024;

[[nodiscard]] std::vector<uint8_t> &ClearScratchChunkBuffer() {
	static thread_local std::vector<uint8_t> chunk(static_cast<size_t>(kClearScratchChunk));
	return chunk;
}

// True when every byte of [address, address + size) equals code. A mixed pattern (an ordinary
// draw's compression state) exits after the first differing chunk instead of copying and
// scanning the whole slice.
[[nodiscard]] bool IsClearSliceUniform(uint64_t address, uint64_t size, uint8_t code) {
	auto& chunk = ClearScratchChunkBuffer();
	for (uint64_t done = 0; done < size;) {
		const auto part = static_cast<size_t>(std::min<uint64_t>(size - done, kClearScratchChunk));
		if (!LibKernel::Memory::TryReadBacking(address + done, chunk.data(), part)) {
			EXIT("TextureCache: failed to read color metadata slice\n");
		}
		for (size_t i = 0; i < part; i++) {
			if (chunk[i] != code) {
				return false;
			}
		}
		done += part;
	}
	return true;
}

// Writes size 0xFF bytes at address in chunks (post-clear key expansion). The caller keeps the
// single InvalidateMemory before this loop, as in the vector path.
void WriteClearSliceFF(uint64_t address, uint64_t size) {
	auto& chunk = ClearScratchChunkBuffer();
	std::fill(chunk.begin(), chunk.end(), uint8_t {0xff});
	for (uint64_t done = 0; done < size;) {
		const auto part = static_cast<size_t>(std::min<uint64_t>(size - done, kClearScratchChunk));
		LibKernel::Memory::WriteBacking(address + done, chunk.data(), part);
		done += part;
	}
}

// Hash of the guest backing bytes of a range, read in chunks; false if a chunk is unreadable.
[[nodiscard]] bool HashGuestBacking(uint64_t address, uint64_t size, uint64_t& hash) {
	constexpr uint64_t   Chunk = 64 * 1024;
	std::vector<uint8_t> bytes(static_cast<size_t>(std::min(size, Chunk)));
	uint64_t             seed = 0;
	for (uint64_t done = 0; done < size;) {
		const auto part = std::min(size - done, Chunk);
		if (!LibKernel::Memory::TryReadBacking(address + done, bytes.data(), part)) {
			return false;
		}
		seed = XXH3_64bits_withSeed(bytes.data(), static_cast<size_t>(part), seed);
		done += part;
	}
	hash = seed;
	return true;
}

// Step 3b regression fix: predicated clears cover integer targets with a matching
// fragment output type (uvec4/ivec4). DecodeColorClear never accepts DCC fast-clear
// codes on integer formats, only the register-clear path does (checked via
// DecodePackedColorClear, which handles eR32Uint/eR32Sint).
[[nodiscard]] TextureCache::ClearFillClass ClassifyClearFill(vk::Format format) {
	switch (format) {
		case vk::Format::eR8Uint:
		case vk::Format::eR8G8Uint:
		case vk::Format::eR8G8B8A8Uint:
		case vk::Format::eB8G8R8A8Uint:
		case vk::Format::eR16Uint:
		case vk::Format::eR16G16Uint:
		case vk::Format::eR16G16B16A16Uint:
		case vk::Format::eR32Uint:
		case vk::Format::eR32G32Uint:
		case vk::Format::eR32G32B32A32Uint:
		case vk::Format::eA2B10G10R10UintPack32:
		case vk::Format::eA2R10G10B10UintPack32:
			return TextureCache::ClearFillClass::Uint;
		case vk::Format::eR8Sint:
		case vk::Format::eR8G8Sint:
		case vk::Format::eR8G8B8A8Sint:
		case vk::Format::eB8G8R8A8Sint:
		case vk::Format::eR16Sint:
		case vk::Format::eR16G16Sint:
		case vk::Format::eR16G16B16A16Sint:
		case vk::Format::eR32Sint:
		case vk::Format::eR32G32Sint:
		case vk::Format::eR32G32B32A32Sint: return TextureCache::ClearFillClass::Sint;
		default: return TextureCache::ClearFillClass::Float;
	}
}

// Step 3b: fallback reasons live in textureCache.h (TextureCache::ClearFbReason);
// the counters print in [FRAME_TELEMETRY_CLEARFB].

[[nodiscard]] bool DecodeColorClear(const TextureCache::ImageDesc& desc, uint8_t code,
                                  vk::ClearColorValue& clear) {
	const auto& metadata = desc.info.metadata;
	const auto  format   = desc.view_info.format;
	const bool  cmask    = metadata.kind == ImageMetadataKind::Cmask;
	if (cmask ? code == 0 : code == 0x20) {
		// Register clears belong to the color buffer; the texture pipe cannot decode them.
		return desc.type == TextureCache::BindingType::RenderTarget &&
		       metadata.clear_register_valid &&
		       DecodePackedColorClear(format, metadata.clear_word, clear);
	}
	if (cmask) {
		return false;
	}
	switch (code) {
		case 0x00:
		case 0x40:
		case 0x80:
		case 0xc0: break;
		default: return false;
	}
	clear = {};
	if (code == 0x00) {
		return true;
	}
	switch (format) {
		case vk::Format::eR8Unorm:
		case vk::Format::eR8G8Unorm:
		case vk::Format::eR8G8B8A8Unorm:
		case vk::Format::eR8G8B8A8Srgb:
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eB8G8R8A8Srgb:
		case vk::Format::eA2B10G10R10UnormPack32:
		case vk::Format::eA2R10G10B10UnormPack32:
		case vk::Format::eR5G6B5UnormPack16:
		case vk::Format::eA1R5G5B5UnormPack16:
		case vk::Format::eR4G4B4A4UnormPack16:
		case vk::Format::eR16Unorm:
		case vk::Format::eR16G16Unorm:
		case vk::Format::eR16G16B16A16Unorm:
		case vk::Format::eR16Sfloat:
		case vk::Format::eR16G16Sfloat:
		case vk::Format::eR16G16B16A16Sfloat:
		case vk::Format::eR32Sfloat:
		case vk::Format::eR32G32Sfloat:
		case vk::Format::eR32G32B32A32Sfloat:
		case vk::Format::eB10G11R11UfloatPack32: break;
		default: return false;
	}
	const float          rgb   = (code & 0x80u) != 0 ? 1.0f : 0.0f;
	const float          alpha = (code & 0x40u) != 0 ? 1.0f : 0.0f;
	std::array<float, 4> channels {rgb, rgb, rgb, alpha};
	if (!metadata.dcc_alpha_msb) {
		std::swap(channels[0], channels[3]);
	}
	// DCC clear decoding clamps missing lanes before applying the format swizzle.
	const auto components = vk::componentCount(format);
	if (components == 1) {
		channels[0] = channels[3];
	} else if (components == 2) {
		channels[1] = channels[3];
	}
	switch (format) {
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eB8G8R8A8Srgb:
		case vk::Format::eA2R10G10B10UnormPack32:
		case vk::Format::eA1R5G5B5UnormPack16:
		case vk::Format::eR5G6B5UnormPack16: std::swap(channels[0], channels[2]); break;
		case vk::Format::eR4G4B4A4UnormPack16:
			std::reverse(channels.begin(), channels.end());
			break;
		default: break;
	}
	clear.float32 = channels;
	return true;
}

[[nodiscard]] std::vector<vk::BufferImageCopy> BuildDepthCopies(const ImageInfo& info,
                                                              uint64_t slice_stride,
                                                              vk::ImageAspectFlags aspect) {
	std::vector<vk::BufferImageCopy> copies(info.resources.layers);
	for (uint32_t layer = 0; layer < info.resources.layers; ++layer) {
		auto& copy             = copies[layer];
		copy.bufferOffset      = slice_stride * layer;
		copy.bufferRowLength   = info.pitch;
		copy.bufferImageHeight = info.extent.height;
		copy.imageSubresource  = {aspect, 0, layer, 1};
		copy.imageExtent       = {info.extent.width, info.extent.height, 1};
	}
	return copies;
}

[[nodiscard]] std::vector<GpuTileInfo> BuildDepthTiles(const ImageInfo& info) {
	TileBlockLayout block {};
	EXIT_NOT_IMPLEMENTED(
	    !TileGetBlockLayout(TileBlockFamily::Depth64KB, info.bytes_per_block, block));
	const auto               full_slice_size = info.data.size / info.resources.layers;
	std::vector<GpuTileInfo> tiles;
	tiles.reserve(info.resources.layers);
	for (uint32_t layer = 0; layer < info.resources.layers; ++layer) {
		const auto offset = full_slice_size * layer;
		tiles.push_back({block.family, block.bytes_per_element, offset, full_slice_size, offset,
		                 full_slice_size, 0, info.extent.width, info.extent.height, 1, info.pitch});
		tiles.back().surface_z = layer;
	}
	return tiles;
}

} // namespace

TextureCache::TextureCache(GraphicContext& graphics, CommandScheduler& scheduler,
                           PageManager& page_manager, BufferCache& buffer_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_page_manager(page_manager),
      m_blit_helper(graphics, scheduler),
      m_tiler(graphics, scheduler, buffer_cache.GetUtilityBuffer(MemoryUsage::Stream)),
      m_buffer_cache(buffer_cache),
      m_readback_linear_images(Config::ReadbackLinearImagesEnabled()) {
	if (m_graphics.CanReportMemoryUsage()) {
		constexpr int64_t GiB = 1024ll * 1024 * 1024;
		const auto        budget =
		    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
		const auto threshold = std::min<int64_t>(budget, 8 * GiB);
		m_pressure_gc_memory = static_cast<uint64_t>(
		    std::max<int64_t>(std::min(budget - 6 * threshold / 10, budget - GiB), GiB + GiB / 2));
		m_critical_gc_memory = static_cast<uint64_t>(
		    std::max<int64_t>(std::min(budget - 2 * threshold / 10, budget - GiB / 2), 3 * GiB));
		m_trigger_gc_memory = static_cast<uint64_t>(std::max<int64_t>((budget - threshold) / 2, 0));
	}
}

TextureCache::~TextureCache() {
	for (const auto& cached: m_clear_fill_pipelines) {
		if (cached.handle != nullptr) {
			m_graphics.device.destroyPipeline(cached.handle, nullptr);
		}
	}
	if (m_clear_fill_vert != nullptr) {
		m_graphics.device.destroyShaderModule(m_clear_fill_vert, nullptr);
	}
	if (m_clear_fill_frag_sint != nullptr) {
		m_graphics.device.destroyShaderModule(m_clear_fill_frag_sint, nullptr);
	}
	if (m_clear_fill_frag_uint != nullptr) {
		m_graphics.device.destroyShaderModule(m_clear_fill_frag_uint, nullptr);
	}
	if (m_clear_fill_frag != nullptr) {
		m_graphics.device.destroyShaderModule(m_clear_fill_frag, nullptr);
	}
	if (m_clear_fill_layout != nullptr) {
		m_graphics.device.destroyPipelineLayout(m_clear_fill_layout, nullptr);
	}
	if (m_clear_fill_desc_layout != nullptr) {
		m_graphics.device.destroyDescriptorSetLayout(m_clear_fill_desc_layout, nullptr);
	}
	if (m_clear_predicate_pipeline != nullptr) {
		m_graphics.device.destroyPipeline(m_clear_predicate_pipeline, nullptr);
	}
	if (m_clear_predicate_layout != nullptr) {
		m_graphics.device.destroyPipelineLayout(m_clear_predicate_layout, nullptr);
	}
	if (m_clear_predicate_desc_layout != nullptr) {
		m_graphics.device.destroyDescriptorSetLayout(m_clear_predicate_desc_layout, nullptr);
	}
	m_slot_images.ForEach([&](ImageId id, const Image& image) {
		if (image.registered) {
			UnregisterImage(id);
		}
	});
}

// Step 3b: record one fallback-reason counter per failed GPU-predicate attempt.
const char* ClearFbReasonName(TextureCache::ClearFbReason reason) {
	using enum TextureCache::ClearFbReason;
	switch (reason) {
		case NoCondRendering: return "nocond";
		case Resources: return "resources";
		case Unsupported: return "unsupported";
		case NoCandidates: return "nocand";
		case TooManyCandidates: return "toomany";
		case MetaRange: return "metarange";
		case DepthOrMsaa: return "depthmsaa";
		case IntFormat: return "intfmt";
		case PipelineFail: return "pipefail";
	}
	return "unknown";
}

void RecordClearFb(TextureCache::ClearFbReason reason) {
	if (!TelemetrySubcountersEnabled()) {
		return;
	}
	using enum TextureCache::ClearFbReason;
	switch (reason) {
		case NoCondRendering: TelemetryCountClearFbNoCond(); break;
		case Resources: TelemetryCountClearFbResources(); break;
		case Unsupported: TelemetryCountClearFbUnsupported(); break;
		case NoCandidates: TelemetryCountClearFbNoCand(); break;
		case TooManyCandidates: TelemetryCountClearFbTooMany(); break;
		case MetaRange: TelemetryCountClearFbMetaRange(); break;
		case DepthOrMsaa: TelemetryCountClearFbDepthMsaa(); break;
		case IntFormat: TelemetryCountClearFbIntFormat(); break;
		case PipelineFail: TelemetryCountClearFbPipeline(); break;
	}
	TelemetryCountClearGpuPredicateFallback();
}

// Step 3: persistent resources for the color-clear predicate check shader, created on
// first GPU-predicate use. Returns false on any allocation failure (caller falls back).
bool TextureCache::ClearPredicateEnsureResources() {
	if (m_clear_predicate_pipeline != nullptr) {
		return m_clear_predicate_buffer != nullptr;
	}
	const vk::DescriptorSetLayoutBinding bindings[] {
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = std::size(bindings);
	layout_info.pBindings    = bindings;
	if (m_graphics.device.createDescriptorSetLayout(&layout_info, nullptr,
	                                                &m_clear_predicate_desc_layout) !=
	    vk::Result::eSuccess) {
		return false;
	}
	struct ClearPredicatePush {
		uint32_t meta_base_dword;
		uint32_t slice_stride_dword;
		uint32_t slice_dwords;
		uint32_t slice_count;
		uint32_t code_count;
		uint32_t code0;
		uint32_t code1;
		uint32_t code2;
		uint32_t code3;
	};
	static_assert(sizeof(ClearPredicatePush) == 36);
	const vk::PushConstantRange push_range {vk::ShaderStageFlagBits::eCompute, 0,
	                                        sizeof(ClearPredicatePush)};
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &m_clear_predicate_desc_layout;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_range;
	if (m_graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                           &m_clear_predicate_layout) !=
	    vk::Result::eSuccess) {
		return false;
	}
	const auto module =
	    CompileSPV({CLEAR_PREDICATE_CHECK_SPV, std::size(CLEAR_PREDICATE_CHECK_SPV)},
	               m_graphics.device);
	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage  = vk::ShaderStageFlagBits::eCompute;
	stage.module = module;
	stage.pName  = "main";
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage  = stage;
	pipeline_info.layout = m_clear_predicate_layout;
	vk::Pipeline pipeline = nullptr;
	const auto   result =
	    m_graphics.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &pipeline);
	m_graphics.device.destroyShaderModule(module, nullptr);
	if (result != vk::Result::eSuccess) {
		return false;
	}
	m_clear_predicate_pipeline = pipeline;
	SetVulkanObjectNameF(m_graphics.device, m_clear_predicate_pipeline, "Clear Predicate Check");
	// One predicate dword per candidate code (up to 4) per slice; 64 slices of headroom.
	m_clear_predicate_buffer = std::make_unique<Buffer>(
	    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0,
	    vk::BufferUsageFlagBits::eStorageBuffer |
	        vk::BufferUsageFlagBits::eConditionalRenderingEXT,
	    64 * 4 * sizeof(uint32_t));
	return m_clear_predicate_buffer != nullptr;
}

bool TextureCache::SameBacking(const ImageInfo& cached, const ImageInfo& requested,
                               bool exact_format) {
	if (cached.data.address != requested.data.address) {
		return false;
	}
	if (cached.data.size != requested.data.size) {
		return false;
	}
	if (cached.extent != requested.extent) {
		return false;
	}
	if (cached.resources.levels < requested.resources.levels ||
	    cached.resources.layers < requested.resources.layers) {
		return false;
	}
	if (cached.samples != requested.samples) {
		return false;
	}
	if (cached.bytes_per_block != requested.bytes_per_block) {
		return false;
	}
	if (cached.tile_mode != requested.tile_mode) {
		return false;
	}
	if (cached.GetColorTransform() != requested.GetColorTransform()) {
		return false;
	}
	if (!ImageViewOps::FormatsCompatible(cached.pixel_format, requested.pixel_format) ||
	    cached.type != requested.type) {
		return false;
	}
	if (exact_format && cached.pixel_format != requested.pixel_format) {
		return false;
	}
	return true;
}

TextureCache::BindingType TextureCache::UploadBinding(const Image& image) {
	if (image.info.IsDepth()) {
		if (image.info.tile_mode == Prospero::TileMode::kDepth ||
		    image.info.tile_mode == Prospero::TileMode::kLinear) {
			return BindingType::DepthTarget;
		}
		return BindingType::Texture;
	}
	if (image.usage.render_target) {
		return BindingType::RenderTarget;
	}
	if (image.usage.video_out) {
		return BindingType::VideoOut;
	}
	return image.usage.storage ? BindingType::Storage : BindingType::Texture;
}

ImageId TextureCache::InsertImage(const ImageInfo& info) {
	const auto id = m_slot_images.insert(m_graphics, m_scheduler, info);
	Profiler::Add(Profiler::Counter::ImagesCreated);
	if (!info.data.Empty()) {
		RegisterImage(id);
	}
	return id;
}

void TextureCache::RegisterImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (image.registered || image.info.data.Empty()) {
		EXIT("TextureCache: invalid image registration\n");
	}
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(image.info.data.address, image.info.data.size, pages)) {
		EXIT("TextureCache: image registration is outside the guest address space\n");
	}
	ForEachPage(image.info.data.address, image.info.data.size, [this, id](uint64_t page) {
		m_image_page_table[page].push_back(id);
	});
	image.registered = true;
	NoteBindlessStateChange(image);
	image.lru_id     = m_lru_cache.Insert(id, LruClock());
	m_total_used_memory += image.AccountedSize();
}

void TextureCache::UnregisterImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	if (image.bindless_pinned) {
		NoteBindlessStateChange(image);
		image.bindless_pinned = false;
		if (on_bindless_unregister) {
			on_bindless_unregister(id);
		}
	}
	UntrackImage(id);
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(image.info.data.address, image.info.data.size, pages)) {
		EXIT("TextureCache: registered image is outside the guest address space\n");
	}
	ForEachPage(image.info.data.address, image.info.data.size, [this, id](uint64_t page) {
		auto* owners = m_image_page_table.Find(page);
		if (owners == nullptr || !owners->Erase(id)) {
			EXIT("TextureCache: image missing from page owner index\n");
		}
	});
	m_lru_cache.Free(image.lru_id);
	const auto accounted = image.AccountedSize();
	if (accounted > m_total_used_memory) {
		EXIT("TextureCache: image accounting underflow\n");
	}
	m_total_used_memory -= accounted;
	image.registered = false;
}

void TextureCache::DeleteImage(ImageId id) {
	auto* image = m_slot_images.try_get(id);
	if (image == nullptr || !image->registered) {
		return;
	}
	if (!image->depth_id) {
		std::vector<ImageId> associations;
		m_slot_images.ForEach([&](ImageId candidate, const Image& associated) {
			if (associated.depth_id == id) {
				associations.push_back(candidate);
			}
		});
		for (const auto association: associations) {
			FreeImage(association);
		}
	}
	if (image->IsGpuModified()) {
		EXIT("TextureCache: deleting a GPU-modified image without resolving its contents\n");
	}
	m_download_images.erase(id);
	if (image->info.HasMetadata()) {
		const auto metadata = m_surface_metas.find(image->info.metadata.range.address);
		if (metadata != m_surface_metas.end() &&
		    image->info.metadata.kind == ImageMetadataKind::Htile &&
		    metadata->second.type == MetaDataInfo::Type::HTile) {
			// A later binding may have reused this address for another metadata type.
			m_surface_metas.erase(metadata);
		}
	}
	UnregisterImage(id);
	// Also between guest command buffers (a guest unmap runs there): command buffers submitted
	// earlier may still be sampling the image on the GPU. Destroying it at once let them read
	// freed memory (device loss at the title, 2026-10-04).
	m_scheduler.DeferRelease([this, id] { ReleaseImage(id); });
}

void TextureCache::ReleaseImage(ImageId id) {
	const auto* image = m_slot_images.try_get(id);
	if (image != nullptr && on_image_release && on_image_release(*image)) {
		// Work recorded with the repointed slot runs until the current tick is done.
		m_scheduler.DeferRelease([this, id] { ReleaseImage(id); });
		return;
	}
	if (image != nullptr && !m_scheduler.IsFree(image->tick_accessed_last)) {
		// Used after it was dropped, in a tick the GPU has not finished: keep it until then.
		static std::atomic<uint32_t> reported = 0;
		if (reported.fetch_add(1) < 16) {
			LOGF("TextureCache: image %ux%u guest=0x%016" PRIx64 " used in tick %" PRIu64
			     " after its release was queued; destruction waits for that tick\n",
			     image->info.extent.width, image->info.extent.height, image->info.data.address,
			     image->tick_accessed_last);
		}
		m_scheduler.DeferRelease([this, id] { ReleaseImage(id); });
		return;
	}
	m_slot_images.erase(id);
	Profiler::Add(Profiler::Counter::ImagesDestroyed);
}

void TextureCache::FreeImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (image.IsGpuModified()) {
		image.ClearGpuModified();
	}
	DeleteImage(id);
}

void TextureCache::TouchImage(Image& image) {
	image.frame_accessed_last = m_graphics.presented_frames.load(std::memory_order_relaxed);
	if (image.registered) {
		m_lru_cache.Touch(image.lru_id, LruClock());
	}
}

void TextureCache::MarkAsMaybeDirty(ImageId id, Image& image) {
	image.MarkMaybeCpuDirty();
	if (image.NeedsMaybeCpuHash()) {
		image.SetMaybeCpuHash(image.HashGuestEdges());
	}
	UntrackImage(id);
}

void TextureCache::TrackImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	const auto image_begin = image.info.data.address;
	const auto image_end   = image.info.data.End();
	if (image_begin == image.track_addr && image_end == image.track_addr_end) {
		return;
	}
	if (!image.IsTracked()) {
		image.track_addr     = image_begin;
		image.track_addr_end = image_end;
		m_page_manager.UpdatePageWatchers<true>(image_begin, image.info.data.size);
		return;
	}
	if (image_begin < image.track_addr) {
		TrackImageHead(id);
	}
	if (image.track_addr_end < image_end) {
		TrackImageTail(id);
	}
}

void TextureCache::TrackImageHead(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	const auto image_begin = image.info.data.address;
	if (image_begin == image.track_addr) {
		return;
	}
	if (!image.IsTracked() || image_begin > image.track_addr) {
		EXIT("TextureCache: invalid image head tracking range\n");
	}
	const auto size  = image.track_addr - image_begin;
	image.track_addr = image_begin;
	m_page_manager.UpdatePageWatchers<true>(image_begin, size);
}

void TextureCache::TrackImageTail(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.registered) {
		return;
	}
	const auto image_end = image.info.data.End();
	if (image_end == image.track_addr_end) {
		return;
	}
	if (!image.IsTracked() || image.track_addr_end > image_end) {
		EXIT("TextureCache: invalid image tail tracking range\n");
	}
	const auto address   = image.track_addr_end;
	const auto size      = image_end - address;
	image.track_addr_end = image_end;
	m_page_manager.UpdatePageWatchers<true>(address, size);
}

void TextureCache::UntrackImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (!image.IsTracked()) {
		return;
	}
	const auto address   = image.track_addr;
	const auto size      = image.track_addr_end - image.track_addr;
	image.track_addr     = 0;
	image.track_addr_end = 0;
	if (size != 0) {
		m_page_manager.UpdatePageWatchers<false>(address, size);
	}
}

void TextureCache::UntrackImageHead(ImageId id) {
	auto&      image = m_slot_images[id];
	const auto begin = image.info.data.address;
	if (!image.IsTracked() || begin < image.track_addr) {
		return;
	}
	const auto address = Common::AlignDown(begin + TRACKER_PAGE_SIZE, TRACKER_PAGE_SIZE);
	const auto size    = address - begin;
	image.track_addr   = address;
	if (image.track_addr == image.track_addr_end) {
		MarkAsMaybeDirty(id, image);
	}
	if (size != 0) {
		m_page_manager.UpdatePageWatchers<false>(begin, size);
	}
}

void TextureCache::UntrackImageTail(ImageId id) {
	auto&      image = m_slot_images[id];
	const auto end   = image.info.data.End();
	if (!image.IsTracked() || image.track_addr_end < end) {
		return;
	}
	const auto address   = Common::AlignDown(end, TRACKER_PAGE_SIZE);
	const auto size      = end - address;
	image.track_addr_end = address;
	if (image.track_addr == image.track_addr_end) {
		MarkAsMaybeDirty(id, image);
	}
	if (size != 0) {
		m_page_manager.UpdatePageWatchers<false>(address, size);
	}
}

void TextureCache::TrackImageDownload(ImageId id, Image& image) {
	if (m_readback_linear_images && !image.info.IsTiled() && !image.info.data.Empty()) {
		if (!image.IsGpuModified()) {
			EXIT("TextureCache: cannot enroll a non-GPU-owned image for download\n");
		}
		m_download_images.insert(id);
	}
}

TextureCache::ImageIds TextureCache::FindImagesInRegion(uint64_t address, uint64_t size,
                                                        bool page_overlap) const {
	ImagePageTable::PageRange pages {};
	if (!ImagePageTable::TryGetPageRange(address, size, pages)) {
		return {};
	}

	uint32_t query_epoch = ++m_image_query_epoch;
	if (query_epoch == 0) {
		m_slot_images.ForEach([](ImageId, const Image& image) { image.query_epoch = 0; });
		query_epoch = ++m_image_query_epoch;
	}

	ImageIds result;
	ForEachPage(address, size, [&](uint64_t page) {
		const auto* owners = m_image_page_table.Find(page);
		if (owners == nullptr) {
			return;
		}
		owners->ForEach([&](ImageId id) {
			auto* image = m_slot_images.try_get(id);
			if (image == nullptr) {
				return;
			}
			if (image->query_epoch == query_epoch) {
				return;
			}
			image->query_epoch = query_epoch;
			if (image->Overlaps(address, size, page_overlap)) {
				result.push_back(id);
			}
		});
	});
	return result;
}

ImageId TextureCache::GetNullImage(const ImageDesc& desc) {
	const auto key = std::pair {desc.info.pixel_format, desc.info.type};
	if (const auto found = m_null_images.find(key); found != m_null_images.end()) {
		return found->second;
	}
	ImageInfo info {};
	info.pixel_format    = desc.info.pixel_format;
	info.guest_format    = desc.info.guest_format;
	info.type            = desc.info.type;
	info.extent          = {1, 1, 1};
	info.resources       = {1, 1};
	info.pitch           = 1;
	info.bytes_per_block = std::max(desc.info.bytes_per_block, 1u);
	info.samples         = 1;
	info.tile_mode       = Prospero::TileMode::kLinear;
	info.mip_layout[0]   = {0, info.bytes_per_block, 1, 1};
	const auto id        = InsertImage(info);
	m_null_images.emplace(key, id);
	return id;
}

void TextureCache::ValidateImageDesc(const ImageDesc& desc) const {
	ImageOps::Validate(desc.info);
	if (desc.view_info.format == vk::Format::eUndefined || desc.view_info.level_count == 0 ||
	    desc.view_info.layer_count == 0 ||
	    desc.view_info.base_level >= desc.info.resources.levels ||
	    desc.view_info.level_count > desc.info.resources.levels - desc.view_info.base_level ||
	    (!desc.info.IsVolume() &&
	     (desc.view_info.base_layer >= desc.info.resources.layers ||
	      desc.view_info.layer_count > desc.info.resources.layers - desc.view_info.base_layer))) {
		EXIT("TextureCache: invalid image view description\n");
	}
	if (desc.type == BindingType::DepthTarget && !IsSupportedDepthTargetFormat(desc.info)) {
		EXIT("TextureCache: unsupported depth image description\n");
	}
	if (desc.type == BindingType::VideoOut && !IsSupportedVideoOutFormat(desc.info)) {
		EXIT("TextureCache: unsupported video-out image description\n");
	}
	if (desc.type == BindingType::VideoOut &&
	    desc.info.metadata.compression == VideoOutCompression::Unsupported) {
		EXIT("TextureCache: unsupported compressed video-out description\n");
	}
}

void TextureCache::PrepareImageCopy(Image& image) {
	if (image.IsCpuDirty()) {
		image.RefreshComplete();
	}
}

void TextureCache::RefreshCopySource(ImageId id) {
	auto& image = m_slot_images[id];
	RefreshImage(id);
	if (image.IsDefinitelyCpuDirty()) {
		EXIT("TextureCache: image copy source remained CPU-dirty after refresh\n");
	}
}

bool TextureCache::CopyD16(Image& destination, Image& source) {
	const bool source_depth      = source.info.IsDepth();
	const bool destination_depth = destination.info.IsDepth();
	if (source_depth == destination_depth) {
		return false;
	}
	auto&      depth          = source_depth ? source : destination;
	auto&      color          = source_depth ? destination : source;
	const auto transfer_bytes = DepthAspectTransferBytes(depth.backing.format);
	if (depth.info.bytes_per_block != sizeof(uint16_t) ||
	    color.info.bytes_per_block != sizeof(uint16_t) || transfer_bytes != sizeof(uint32_t)) {
		return false;
	}
	EXIT_IF(source.backing.samples != 1 || destination.backing.samples != 1 ||
	        source.info.resources.levels != 1 || destination.info.resources.levels != 1 ||
	        source.info.extent != destination.info.extent ||
	        source.info.resources.layers != destination.info.resources.layers);

	const auto     layers = depth.info.resources.layers;
	const uint64_t depth_slice =
	    static_cast<uint64_t>(depth.info.pitch) * depth.info.extent.height * transfer_bytes;
	const uint64_t color_slice =
	    static_cast<uint64_t>(color.info.pitch) * color.info.extent.height * sizeof(uint16_t);
	EXIT_IF(layers == 0 || depth_slice > UINT64_MAX / layers || color_slice > UINT64_MAX / layers);
	const auto                       depth_size = depth_slice * layers;
	const auto                       color_size = color_slice * layers;
	std::vector<vk::BufferImageCopy> depth_copies(layers);
	std::vector<vk::BufferImageCopy> color_copies(layers);
	for (uint32_t layer = 0; layer < layers; layer++) {
		depth_copies[layer].bufferOffset      = depth_slice * layer;
		depth_copies[layer].bufferRowLength   = depth.info.pitch;
		depth_copies[layer].bufferImageHeight = depth.info.extent.height;
		depth_copies[layer].imageSubresource  = {vk::ImageAspectFlagBits::eDepth, 0, layer, 1};
		depth_copies[layer].imageExtent       = depth.info.extent;
		color_copies[layer].bufferOffset      = color_slice * layer;
		color_copies[layer].bufferRowLength   = color.info.pitch;
		color_copies[layer].bufferImageHeight = color.info.extent.height;
		color_copies[layer].imageSubresource  = {vk::ImageAspectFlagBits::eColor, 0, layer, 1};
		color_copies[layer].imageExtent       = color.info.extent;
	}

	auto                         depth_buffer = m_tiler.GetScratchBuffer(depth_size);
	auto                         color_buffer = m_tiler.GetScratchBuffer(color_size, depth_buffer.buffer);
	const TileManager::D16Layout promote_layout {
	    .width               = depth.info.extent.width,
	    .height              = depth.info.extent.height,
	    .layers              = layers,
	    .source_row_stride   = static_cast<uint64_t>(color.info.pitch) * sizeof(uint16_t),
	    .target_row_stride   = static_cast<uint64_t>(depth.info.pitch) * transfer_bytes,
	    .source_slice_stride = color_slice,
	    .target_slice_stride = depth_slice,
	};
	const bool d32 = DepthAspectTransferFormat(depth.backing.format) == vk::Format::eD32Sfloat;
	if (source_depth) {
		source.Download(depth_copies, depth_buffer.buffer, depth_buffer.offset, depth_buffer.size);
		m_tiler.ConvertD16(depth_buffer, color_buffer, TileManager::D16Direction::Demote, d32,
		                   {.width               = promote_layout.width,
		                    .height              = promote_layout.height,
		                    .layers              = promote_layout.layers,
		                    .source_row_stride   = promote_layout.target_row_stride,
		                    .target_row_stride   = promote_layout.source_row_stride,
		                    .source_slice_stride = promote_layout.target_slice_stride,
		                    .target_slice_stride = promote_layout.source_slice_stride});
		destination.Upload(color_copies, color_buffer.buffer, color_buffer.offset,
		                   color_buffer.size);
	} else {
		source.Download(color_copies, color_buffer.buffer, color_buffer.offset, color_buffer.size);
		m_tiler.ConvertD16(color_buffer, depth_buffer, TileManager::D16Direction::Promote, d32,
		                   promote_layout);
		destination.Upload(depth_copies, depth_buffer.buffer, depth_buffer.offset,
		                   depth_buffer.size);
	}
	return true;
}

void TextureCache::CopyImage(ImageId destination_id, ImageId source_id) {
	RefreshCopySource(source_id);
	auto& destination = m_slot_images[destination_id];
	auto& source      = m_slot_images[source_id];
	TrackImage(destination_id);
	if (source.backing.samples != destination.backing.samples) {
		EXIT("TextureCache: cannot issue an unequal-sample image copy\n");
	}
	PrepareImageCopy(destination);
	if (source.IsBufferModified()) {
		if (source.info.data == destination.info.data) {
			destination.MarkBufferModified();
		}
		return;
	}
	const bool source_depth = source.info.IsDepth();
	const bool dest_depth   = destination.info.IsDepth();
	const bool direct_copy =
	    source.info.GetColorTransform() == destination.info.GetColorTransform() &&
	    (source.backing.image_type == destination.backing.image_type ||
	     (source.backing.image_type != vk::ImageType::e1D &&
	      destination.backing.image_type != vk::ImageType::e1D)) &&
	    (source.backing.format == destination.backing.format ||
	     (!source_depth && !dest_depth &&
	      vk::blockSize(source.backing.format) == vk::blockSize(destination.backing.format)));
	if (direct_copy) {
		destination.CopyImage(source);
	} else if (!CopyD16(destination, source)) {
		if (source.backing.samples != 1 || destination.backing.samples != 1) {
			EXIT("TextureCache: cross-format multisample image copy is unsupported\n");
		}
		auto& copy_buffer = m_buffer_cache.GetUtilityBuffer(MemoryUsage::DeviceLocal);
		destination.CopyImageWithBuffer(source, copy_buffer, m_tiler);
	}
	if (source.IsGpuModified()) {
		destination.MarkGpuModified();
	}
	destination.ClearBufferModified();
}

void TextureCache::CopyImageMip(ImageId destination_id, ImageId source_id, uint32_t mip,
                                uint32_t layer) {
	RefreshCopySource(source_id);
	auto& destination = m_slot_images[destination_id];
	auto& source      = m_slot_images[source_id];
	TrackImage(destination_id);
	if (source.IsBufferModified() || source.backing.samples != destination.backing.samples) {
		EXIT("TextureCache: invalid mip-copy ownership or sample count\n");
	}
	destination.CopyMip(source, mip, layer);
	if (source.IsGpuModified()) {
		destination.MarkGpuModified();
	}
}

ImageId TextureCache::ResolveDepthOverlap(const ImageInfo& requested, BindingType binding,
                                          ImageId cached_id) {
	auto& cached = m_slot_images[cached_id];
	if ((!cached.info.IsDepth() && !requested.IsDepth()) ||
	    cached.info.tile_mode != requested.tile_mode) {
		return {};
	}
	const bool stencil_match = requested.HasStencil() == cached.info.HasStencil();
	const bool bpp_match     = requested.bytes_per_block == cached.info.bytes_per_block;
	// PPSA04264
	const bool raw_d16_texture =
	    binding == BindingType::Texture && cached.info.IsDepth() &&
	    cached.info.guest_format == Prospero::BufferFormat::k16UNorm &&
	    requested.guest_format == Prospero::BufferFormat::k16UInt &&
	    requested.pixel_format == vk::Format::eR16Uint && cached.backing.samples == 1 &&
	    requested.samples == 1 && requested.data == cached.info.data &&
	    requested.extent == cached.info.extent && requested.resources == cached.info.resources &&
	    requested.type == cached.info.type && requested.pitch == cached.info.pitch &&
	    !requested.HasStencil() && !cached.info.HasStencil() && !requested.HasMetadata() &&
	    !cached.info.HasMetadata();
	// PPSA04264
	const bool retain_cached_layout =
	    requested.samples == 1 && cached.info.samples == 1 && cached.backing.samples == 1 &&
	    requested.bytes_per_block == cached.info.bytes_per_block &&
	    requested.data.address == cached.info.data.address &&
	    requested.data.size < cached.info.data.size && requested.extent == cached.info.extent &&
	    requested.resources.levels == 1 && cached.info.resources.levels == 1 &&
	    requested.resources.layers != 0 && cached.info.resources.layers != 0 &&
	    requested.resources.layers < cached.info.resources.layers &&
	    requested.type == cached.info.type && requested.pitch == cached.info.pitch &&
	    requested.mip_layout[0].offset == 0 &&
	    cached.info.mip_layout[0].offset == 0 &&
	    requested.mip_layout[0].size == requested.data.size &&
	    cached.info.mip_layout[0].size == cached.info.data.size &&
	    requested.data.size % requested.resources.layers == 0 &&
	    cached.info.data.size % cached.info.resources.layers == 0 &&
	    requested.data.size / requested.resources.layers ==
	        cached.info.data.size / cached.info.resources.layers &&
	    !requested.HasStencil() && !cached.info.HasStencil() && !requested.HasMetadata() &&
	    !cached.info.HasMetadata();
	bool recreate = cached.info.resources < requested.resources ||
	                requested.IsVolume() != cached.info.IsVolume();
	switch (binding) {
		case BindingType::Texture:
			recreate |= requested.IsDepth() && !cached.info.IsDepth();
			recreate |= raw_d16_texture;
			break;
		case BindingType::Storage: recreate |= cached.info.IsDepth(); break;
		case BindingType::RenderTarget: recreate |= cached.info.IsDepth(); break;
		case BindingType::DepthTarget:
			recreate |= !cached.info.IsDepth();
			recreate |= cached.info.IsDepth() && !(stencil_match && bpp_match);
			break;
		case BindingType::VideoOut: recreate |= cached.info.IsDepth(); break;
	}
	if (!recreate) {
		return cached_id;
	}
	RefreshImage(cached_id);
	auto info = requested;
	if (retain_cached_layout) {
		info.data       = cached.info.data;
		info.resources  = cached.info.resources;
		info.mip_layout = cached.info.mip_layout;
	} else {
		info.resources = std::max(requested.resources, cached.info.resources);
	}
	info.htile_clear_mask     = 0;
	const auto replacement_id = InsertImage(info);
	auto&      replacement    = m_slot_images[replacement_id];
	replacement.usage         = cached.usage;
	if (cached.binding.is_bound || cached.binding.is_target) {
		cached.binding.needs_rebind = true;
	}
	if (cached.backing.samples == replacement.backing.samples) {
		const bool copy_supported =
		    cached.backing.samples == 1 || cached.backing.format == replacement.backing.format ||
		    (!cached.info.IsDepth() && !replacement.info.IsDepth() &&
		     ImageViewOps::FormatsCompatible(cached.backing.format, replacement.backing.format));
		if (copy_supported) {
			CopyImage(replacement_id, cached_id);
		} else {
			LOGF_COLOR(Log::Color::BrightYellow,
			           "TextureCache: unsupported cross-format multisample depth copy\n");
		}
	} else if (cached.backing.samples == 1 && replacement.backing.samples > 1 &&
	           replacement.info.IsDepth()) {
		RefreshCopySource(cached_id);
		if (cached.IsBufferModified() || cached.IsDefinitelyCpuDirty()) {
			EXIT("TextureCache: multisample depth conversion source is not native-current\n");
		}
		PrepareImageCopy(replacement);
		m_blit_helper.ReinterpretColorAsMsDepth(cached, replacement);
		CommitGpuWrite(replacement);
	} else {
		LOGF_COLOR(Log::Color::BrightYellow,
		           "TextureCache: unsupported unequal-sample depth overlap copy (%u -> %u)\n",
		           cached.backing.samples, replacement.backing.samples);
	}
	FreeImage(cached_id);
	return replacement_id;
}

TextureCache::OverlapResult TextureCache::ResolveOverlap(const ImageInfo& requested,
                                                         BindingType binding, ImageId cached_id,
                                                         ImageId merged_id) {
	auto owner = m_slot_images.try_get(cached_id);
	if (owner == nullptr) {
		return {merged_id};
	}
	auto&      cached       = *owner;
	const auto current_tick  = m_scheduler.CurrentTick();
	const auto current_frame = m_graphics.presented_frames.load(std::memory_order_relaxed);
	const bool safe_to_delete =
	    current_tick - std::min(current_tick, cached.tick_accessed_last) > NumFramesBeforeRemoval &&
	    (!AliasAgeByFrames() || current_frame - std::min(current_frame, cached.frame_accessed_last) >
	                                AliasFramesBeforeRemoval);

	const uint32_t requested_block = requested.bytes_per_block * requested.samples;
	const uint32_t cached_block    = cached.info.bytes_per_block * cached.info.samples;
	if (requested.data.address == cached.info.data.address &&
	    requested.BlockExtent() == cached.info.BlockExtent() && requested_block == cached_block) {
		if (const auto depth_id = ResolveDepthOverlap(requested, binding, cached_id)) {
			return {depth_id};
		}
		// Equal pitch does not imply equal mip placement: a changed extent can move
		// a level into or out of the mip tail. These are separate guest layouts.
		if (requested.tile_mode != cached.info.tile_mode ||
		    (requested.resources == cached.info.resources &&
		     requested.mip_layout != cached.info.mip_layout)) {
			if (safe_to_delete) {
				FreeImage(cached_id);
			}
			return {merged_id};
		}
		if (requested.GetColorTransform() != cached.info.GetColorTransform()) {
			return {ExpandImage(requested, cached_id)};
		}
		if (requested.IsBlock() && !cached.info.IsBlock()) {
			return {ExpandImage(requested, cached_id)};
		}
		// Volume depth is not an array-layer count. A larger depth can retain the
		// same block-slice layout while requiring a larger native image.
		if ((requested.IsVolume() || cached.info.IsVolume()) &&
		    (requested.data.size == cached.info.data.size ||
		     (requested.type == cached.info.type && requested.resources == cached.info.resources &&
		      requested.extent.width == cached.info.extent.width &&
		      requested.extent.height == cached.info.extent.height &&
		      requested.extent.depth > cached.info.extent.depth))) {
			return {ExpandImage(requested, cached_id)};
		}
		// PPSA08394
		// A view cannot change the native image type or grow its extent.
		if (requested.data.size == cached.info.data.size &&
		    requested.resources == cached.info.resources &&
		    ImageViewOps::FormatsCompatible(cached.info.pixel_format, requested.pixel_format) &&
		    (requested.type != cached.info.type
		         ? requested.extent == cached.info.extent
		         : requested.extent.width > cached.info.extent.width &&
		               requested.extent.height >= cached.info.extent.height &&
		               requested.extent.depth >= cached.info.extent.depth)) {
			return {ExpandImage(requested, cached_id)};
		}
		// PS5 mip tails can expose more levels without increasing the guest allocation.
		if (requested.pixel_format == cached.info.pixel_format &&
		    requested.type == cached.info.type && requested.resources > cached.info.resources &&
		    (requested.data.size > cached.info.data.size ||
		     (requested.data.size == cached.info.data.size &&
		      requested.extent == cached.info.extent &&
		      cached.info.resources.levels > 1 &&
		      requested.resources.layers == cached.info.resources.layers))) {
			return {ExpandImage(requested, cached_id)};
		}
		if (requested.type != cached.info.type) {
			return {merged_id};
		}
		if (requested.pixel_format != cached.info.pixel_format ||
		    requested.data.size <= cached.info.data.size) {
			const auto result_id = merged_id ? merged_id : cached_id;
			const auto result    = m_slot_images.try_get(result_id);
			return {result != nullptr && ImageViewOps::FormatsCompatible(result->info.pixel_format,
			                                                             requested.pixel_format)
			            ? result_id
			            : ImageId {}};
		}
		EXIT("TextureCache: unresolvable equal-address image overlap, address=0x%016" PRIx64
		     " requested=%ux%u "
		     "cached=%ux%u requested_size=0x%016" PRIx64 " cached_size=0x%016" PRIx64
		     " type=%u/%u tile=%u/%u\n",
		     requested.data.address, requested.resources.levels, requested.resources.layers,
		     cached.info.resources.levels, cached.info.resources.layers, requested.data.size,
		     cached.info.data.size, static_cast<uint32_t>(requested.type),
		     static_cast<uint32_t>(cached.info.type), static_cast<uint32_t>(requested.tile_mode),
		     static_cast<uint32_t>(cached.info.tile_mode));
	}

	const int32_t requested_mip = requested.MipOf(cached.info);
	if (requested_mip >= 0) {
		const int32_t layer = requested.SliceOf(cached.info, requested_mip);
		return {cached_id, requested_mip, layer};
	}

	const int32_t mip = cached.info.MipOf(requested);
	if (mip >= 0) {
		const int32_t layer = cached.info.SliceOf(requested, mip);
		if (!merged_id) {
			return {ExpandImage(requested, cached_id)};
		}
		cached.binding.needs_rebind |= cached.binding.is_bound || cached.binding.is_target;
		if (cached.binding.is_target && !m_slot_images[merged_id].binding.is_target) {
			NoteBindlessStateChange(m_slot_images[merged_id]);
		}
		m_slot_images[merged_id].binding.is_target |= cached.binding.is_target;
		CopyImageMip(merged_id, cached_id, static_cast<uint32_t>(mip),
		             static_cast<uint32_t>(layer));
		FreeImage(cached_id);
		return {merged_id};
	}
	if (requested.data.address >= cached.info.data.address && safe_to_delete) {
		FreeImage(cached_id);
	}
	return {merged_id};
}

ImageId TextureCache::ExpandImage(const ImageInfo& info, ImageId source_id) {
	// Measurement only: the expansion runs identically with or without subcounters.
	const bool subcounters = TelemetrySubcountersEnabled();
	const auto expand_begin_us = subcounters ? TelemetryNowUs() : 0;
	RefreshCopySource(source_id);
	const auto expanded_id = InsertImage(info);
	auto&      source      = m_slot_images[source_id];
	if (source.binding.is_bound || source.binding.is_target) {
		source.binding.needs_rebind = true;
	}
	InitializeImage(expanded_id);
	const int32_t mip = source.info.MipOf(info);
	const int32_t layer = source.info.SliceOf(info, mip);
	if (layer >= 0) {
		CopyImageMip(expanded_id, source_id, static_cast<uint32_t>(mip),
		             static_cast<uint32_t>(layer));
	} else {
		CopyImage(expanded_id, source_id);
	}
	FreeImage(source_id);
	if (subcounters) {
		TelemetryAddDrawTargetsExpand(TelemetryNowUs() - expand_begin_us);
		TelemetryCountCacheExpand();
	}
	return expanded_id;
}

struct TextureCache::TextureTransfer {
	TextureUploadLayout              layout;
	std::vector<vk::BufferImageCopy> regions;
	std::vector<GpuTileInfo>         tiles;
	ColorTransform                   color_transform = ColorTransform::None;
	bool                             valid           = false;

	[[nodiscard]] uint64_t LinearSize() const {
		uint64_t size = 0;
		for (const auto& tile: tiles) {
			size = std::max(size, tile.linear_offset + tile.linear_size);
		}
		return size;
	}
};

struct TextureCache::ImageDownload {
	TextureTransfer texture;
	bool                depth_target = false;
	bool                valid        = false;
};

TextureCache::TextureTransfer
TextureCache::BuildTextureTransfer(const Image& image, BindingType binding,
                                    TransferDirection direction) const {
	const auto& info             = image.info;
	const bool  upload           = direction == TransferDirection::Upload;
	const bool  render_target    = binding == BindingType::RenderTarget;
	const bool  video_out        = binding == BindingType::VideoOut;
	uint32_t    layers           = info.TransferLayers();
	bool        volume           = info.IsVolume();
	const char* owner            = "TextureCache readback";

	TextureTransfer transfer;
	transfer.color_transform = info.GetColorTransform();
	if (upload) {
		if ((render_target || video_out) &&
		    (info.resources.layers == 0 || info.data.size % info.resources.layers != 0 ||
		     info.samples != 1 || image.backing.samples != 1)) {
			EXIT("TextureCache: invalid color-attachment upload\n");
		}
		owner = "TextureCache";
		if (render_target) {
			owner = "RenderTarget";
		} else if (binding == BindingType::Storage) {
			owner = "StorageTextureCache";
		} else if (video_out) {
			if (info.metadata.compression != VideoOutCompression::Uncompressed) {
				EXIT("TextureCache: invalid color-attachment upload\n");
			}
			layers = info.resources.layers;
			volume = false;
			owner  = "VideoOut";
		}
	}

	transfer.layout = TextureCalcUploadLayout(info.guest_format, info.extent.width,
	                                         info.extent.height, info.resources.levels, layers,
	                                         info.tile_mode, info.data.size, volume, owner);
	transfer.regions = TextureBuildImageCopies(transfer.layout);
	if (info.IsDepth()) {
		for (auto& region: transfer.regions) {
			region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eDepth;
		}
	}
	if (transfer.layout.surface.description.tile_mode != Prospero::TileMode::kLinear) {
		if (!TextureBuildGpuTileInfos(info.data.size, transfer.regions, transfer.layout,
		                              info.resources.levels, transfer.tiles)) {
			return transfer;
		}
	}
	transfer.valid = true;
	return transfer;
}

TextureCache::ImageDownload TextureCache::BuildDownload(const Image& image) const {
	const auto&  info    = image.info;
	const auto   binding = UploadBinding(image);
	ImageDownload transfer {.depth_target = binding == BindingType::DepthTarget};
	if (info.samples != 1 || image.backing.samples != 1) {
		return transfer;
	}
	if (transfer.depth_target) {
		transfer.valid = IsSupportedDepthPlaneReadback(info) && info.resources.layers != 0 &&
		             info.data.size % info.resources.layers == 0 &&
		             Prospero::NumBytesPerElement(info.guest_format) == info.bytes_per_block;
		return transfer;
	}
	if (info.metadata.compression != VideoOutCompression::Uncompressed) {
		return transfer;
	}
	transfer.texture = BuildTextureTransfer(image, binding, TransferDirection::Download);
	transfer.valid   = transfer.texture.valid;
	return transfer;
}

void TextureCache::UploadImage(Image& image, Buffer& source, uint64_t source_offset) {
	auto& destination = image.depth_id ? m_slot_images[image.depth_id] : image;
	const auto binding = image.depth_id ? BindingType::DepthTarget : UploadBinding(image);
	const auto  upload  = [&](std::vector<vk::BufferImageCopy>& copies, TileManager::Result linear) {
		for (auto& copy: copies) {
			copy.bufferOffset += linear.offset;
		}
		destination.Upload(copies, linear.buffer, linear.offset, linear.size);
	};

	if (binding != BindingType::DepthTarget) {
		const auto& info = image.info;
		auto transfer = BuildTextureTransfer(image, binding, TransferDirection::Upload);
		if (!transfer.valid) {
			EXIT("TextureCache: invalid texture upload: binding=%u addr=0x%016" PRIx64
			     " size=0x%016" PRIx64 " format=%u tile=%u family=%u extent=%ux%ux%u "
			     "pitch=%u levels=%u layers=%u samples=%u\n",
			     static_cast<uint32_t>(binding), info.data.address, info.data.size,
			     static_cast<uint32_t>(info.guest_format), static_cast<uint32_t>(info.tile_mode),
			     static_cast<uint32_t>(transfer.layout.surface.texture.block.family), info.extent.width,
			     info.extent.height, info.extent.depth, info.pitch, info.resources.levels,
			     info.resources.layers, info.samples);
		}
		TileManager::Result linear {source.Handle(), source_offset, info.data.size};
		if (!transfer.tiles.empty()) {
			linear = m_tiler.Detile(source.Handle(), source_offset, info.data.size,
			                        transfer.LinearSize(), transfer.tiles, transfer.color_transform);
		} else if (transfer.color_transform != ColorTransform::None) {
			linear = m_tiler.TransformColor(linear, transfer.color_transform, true);
		}
		upload(transfer.regions, linear);
		return;
	}

	// The stencil plane has its own row pitch and shares the native image
	// with the depth plane.
	auto info = destination.info;
	if (image.depth_id) {
		info.data            = image.info.data;
		info.resources       = {image.stencil_subresources.level_count,
		                        image.stencil_subresources.layer_count};
		info.guest_format    = Prospero::BufferFormat::k8UInt;
		info.bytes_per_block = 1;
		if (info.IsTiled()) info.pitch = TileGetDepthPitch(info.extent.width, 1, 0);
	}
	if (info.samples != 1 || destination.backing.samples != 1 ||
	    info.resources.layers == 0 || info.data.size % info.resources.layers != 0 ||
	    Prospero::NumBytesPerElement(info.guest_format) != info.bytes_per_block) {
		EXIT("TextureCache: invalid depth upload\n");
	}
	const auto          layers          = info.resources.layers;
	const auto          full_slice_size = info.data.size / layers;
	auto copies = BuildDepthCopies(info, full_slice_size, image.depth_id
	                                                        ? vk::ImageAspectFlagBits::eStencil
	                                                        : vk::ImageAspectFlagBits::eDepth);
	if (image.depth_id) {
		for (auto& copy: copies) {
			copy.imageSubresource.baseArrayLayer += image.stencil_subresources.base_layer;
		}
	}
	TileManager::Result linear {source.Handle(), source_offset, source.Size() - source_offset};
	if (info.IsTiled()) {
		const auto tiles = BuildDepthTiles(info);
		linear =
		    m_tiler.Detile(source.Handle(), source_offset, info.data.size, info.data.size, tiles);
	}
	const auto transfer_bytes = image.depth_id ? 1u : DepthAspectTransferBytes(info.pixel_format);
	if (transfer_bytes != info.bytes_per_block) {
		const uint64_t texels_per_slice = static_cast<uint64_t>(info.pitch) * info.extent.height;
		EXIT_NOT_IMPLEMENTED(info.bytes_per_block != sizeof(uint16_t) ||
		                     transfer_bytes != sizeof(uint32_t) || texels_per_slice > UINT32_MAX ||
		                     texels_per_slice > UINT64_MAX / transfer_bytes);
		const uint64_t transfer_slice = texels_per_slice * transfer_bytes;
		EXIT_NOT_IMPLEMENTED(transfer_slice > UINT64_MAX / layers);
		auto promoted = m_tiler.GetScratchBuffer(transfer_slice * layers, linear.buffer);
		m_tiler.ConvertD16(
		    linear, promoted, TileManager::D16Direction::Promote,
		    info.pixel_format == vk::Format::eD32SfloatS8Uint,
		    {.width               = info.extent.width,
		     .height              = info.extent.height,
		     .layers              = layers,
		     .source_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint16_t),
		     .target_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint32_t),
		     .source_slice_stride = full_slice_size,
		     .target_slice_stride = transfer_slice});
		linear = promoted;
		for (uint32_t layer = 0; layer < layers; layer++) {
			copies[layer].bufferOffset = transfer_slice * layer;
		}
	}
	upload(copies, linear);
}

void TextureCache::InitializeImage(ImageId id) {
	auto& image = m_slot_images[id];
	if (image.info.data.Empty()) {
		return;
	}
	TrackImage(id);
	if (image.info.metadata.compression != VideoOutCompression::Uncompressed) {
		if (image.IsCpuDirty()) {
			image.RefreshComplete();
		}
		return;
	}
	if (image.info.samples > 1) {
		return;
	}
	const bool upload = image.IsBufferModified() || image.IsCpuDirty();
	if (upload) {
		const auto [source, source_offset] =
		    m_buffer_cache.ObtainBufferForImage(image.info.data.address, image.info.data.size);
		if (source == nullptr) {
			EXIT("TextureCache: failed to obtain image upload source\n");
		}
		UploadImage(image, *source, source_offset);
		image.ClearBufferModified();
	}
	if (image.IsCpuDirty()) {
		image.RefreshComplete();
	}
}

// Step 3: GPU predicate replacing the synchronous metadata readback. Computes the
// decodable (code, clear color) pairs on the CPU, runs the check shader over the buffer
// cache's GPU copy, then issues one conditional clear per matching slice. Keeps the
// range GPU-modified so a later CPU reader downloads it lazily. False on any unsupported
// case (caller takes the old synchronous path).
// Step 3b: cached DecodeColorClear enumeration per clear-relevant desc fields.
// DecodeColorClear reads exactly (metadata kind, view format, binding type,
// clear_word, clear_register_valid, dcc_alpha_msb): the register-clear branch needs
// kind+type+validity+clear_word (via DecodePackedColorClear, which reads only the
// format and the packed word), the DCC branch needs kind+format+alpha_msb for the
// lane mapping. ImageMetadataInfo holds a single 32-bit clear_word (no second word),
// the guest num format is already folded into view.format at image creation, and no
// swizzle beyond view.format is consulted. None of these change per slice, so the
// 256-code loop runs once per distinct image configuration.
// Step 3b regression fix: re-audited after adding integer fill variants; the key is
// complete, no field is missing.
TextureCache::ClearCandidatesResult
TextureCache::LookupClearCandidates(const ImageDesc& desc) {
	const ClearCandidatesKey key {.kind                  = desc.info.metadata.kind,
	                             .format                = desc.view_info.format,
	                             .type                  = desc.type,
	                             .clear_word            = desc.info.metadata.clear_word,
	                             .clear_register_valid  = desc.info.metadata.clear_register_valid,
	                             .dcc_alpha_msb         = desc.info.metadata.dcc_alpha_msb};
	if (const auto found = m_clear_candidates_cache.find(key);
	    found != m_clear_candidates_cache.end()) {
		return found->second;
	}
	ClearCandidatesResult result;
	for (uint32_t code = 0; code < 256; code++) {
		ClearPredicateCandidate candidate {};
		candidate.code = static_cast<uint8_t>(code);
		if (DecodeColorClear(desc, candidate.code, candidate.clear)) {
			result.candidates.push_back(candidate);
			if (result.candidates.size() > 4) {
				result.too_many  = true;
				result.candidates.clear();
				break;
			}
		}
	}
	if (m_clear_candidates_cache.size() > 256) {
		m_clear_candidates_cache.clear();
	}
	m_clear_candidates_cache.emplace(key, result);
	return result;
}

TextureCache::ClearFillClass TextureCache::ClearFillClassFor(vk::Format format) {
	return ClassifyClearFill(format);
}

TextureCache::ClearPredicateResult
TextureCache::TryMaterializeColorClearGpu(ImageId id, const ImageDesc& desc, uint32_t first,
                                          uint32_t count, uint64_t slice_size,
                                          uint64_t range_address, uint64_t range_size,
                                          ClearFbReason* out_reason) {
	ClearFbReason reason = ClearFbReason::Unsupported;
	const auto    fail   = [&](ClearFbReason why) {
        if (out_reason != nullptr) {
            *out_reason = why;
        }
        RecordClearFb(why);
        return ClearPredicateResult::SyncFallback;
    };
	if (!m_graphics.conditional_rendering_enabled) {
		return fail(ClearFbReason::NoCondRendering);
	}
	if (!ClearPredicateEnsureResources()) {
		return fail(ClearFbReason::Resources);
	}
	if (desc.type == BindingType::VideoOut || desc.info.IsVolume() ||
	    desc.info.resources.levels != 1 || slice_size == 0 || count == 0 || count > 64 ||
	    slice_size > UINT32_MAX || range_size != slice_size * count ||
	    (range_address & 3u) != 0 || (slice_size & 3u) != 0) {
		return fail(ClearFbReason::Unsupported);
	}
	(void)first;
	// Step 3b regression fix: every check that can fail in the draw half runs here,
	// before the dispatch half records anything. After the dispatch the check shader
	// has overwritten matched metadata slices with 0xFFFFFFFF (fused writeback), so a
	// later sync fallback would read 0xFF -- which decodes nothing on integer formats
	// -- and skip the clear the image needs. That left integer targets uncleared and
	// Device-lost the fight entry with null-descriptor draws.
	{
		const auto& view  = desc.view_info;
		auto&       image = m_slot_images[id];
		if (image.depth_id || image.info.IsDepth() || image.info.samples != 1 ||
		    desc.info.resources.levels != 1 || view.level_count != 1) {
			return fail(ClearFbReason::DepthOrMsaa);
		}
		if (ClearFillPipelineFor(view.format) == nullptr) {
			return fail(ClearFbReason::PipelineFail);
		}
	}
	const auto lookup = LookupClearCandidates(desc);
	if (lookup.too_many) {
		return fail(ClearFbReason::TooManyCandidates);
	}
	if (lookup.candidates.empty()) {
		// No metadata code decodes for this image (only Cmask without a decodable
		// register clear reaches here: DCC always decodes code 0x00). The old path's
		// writes are still no-ops here (every slice hits "DecodeColorClear fails ->
		// continue" before ClearImage/WriteBacking), BUT the synchronous ReadMemory
		// before that loop is load-bearing: it downloads a ~4 MB window around the
		// range into guest memory, unprotects unrelated GPU-dirty pages and clears
		// their tracker state, which later readers (TryReadBacking callers without
		// a HasGpuDirtyBytes guard) observe. Skipping it starves those readers of
		// fresh bytes and removes the wait that ordered the GPU write before their
		// reads, which Device-lost the fighter select (null-descriptor DrawIndex at
		// address 0x0). Only skip under the explicit opt-in flag.
		if (ClearSkipNocandEnabled()) {
			if (TelemetrySubcountersEnabled()) {
				TelemetryCountClearGpuPredicate();
				TelemetryCountClearGpuPredicateNocand();
			}
			return ClearPredicateResult::SkippedNoCand;
		}
		return fail(ClearFbReason::NoCandidates);
	}
	// reason is assigned by the dispatch/draw halves below.
	if (!TryMaterializeColorClearGpuDispatch(id, desc, count, slice_size, range_address,
	                                        range_size, lookup.candidates.data(),
	                                        lookup.candidates.size(), &reason)) {
		if (out_reason != nullptr) {
			*out_reason = reason;
		}
		RecordClearFb(reason);
		return ClearPredicateResult::SyncFallback;
	}
	return ClearPredicateResult::Applied;
}

// Step 3 dispatch half: binds the metadata GPU copy and the predicate buffer, runs the
// check compute. Conditional clears follow in the second half below.
bool TextureCache::TryMaterializeColorClearGpuDispatch(
    ImageId id, const ImageDesc& desc, uint32_t count, uint64_t slice_size,
    uint64_t range_address, uint64_t range_size, const ClearPredicateCandidate* candidates,
    size_t candidate_count, ClearFbReason* out_reason) {
	(void)id;
	(void)desc;
	const auto fail = [&](ClearFbReason why) {
		if (out_reason != nullptr) {
			*out_reason = why;
		}
		return false;
	};
	// ObtainBuffer with is_written=false never uploads stale CPU bytes over GPU-written
	// ranges (sync only uploads CPU-modified ranges), so the check reads the guest
	// dispatch's metadata writes.
	const auto [meta, meta_offset] =
	    m_buffer_cache.ObtainBuffer(range_address, range_size, false, false);
	if (meta == nullptr) {
		return fail(ClearFbReason::Resources);
	}
	const auto& limits              = m_graphics.GetPhysicalDeviceProperties().limits;
	const uint64_t descriptor_align = std::max<uint64_t>(limits.minStorageBufferOffsetAlignment, 4);
	const uint64_t meta_desc_offset = Common::AlignDown(meta_offset, descriptor_align);
	const uint64_t meta_base        = meta_offset - meta_desc_offset;
	const uint64_t meta_range       = Common::AlignUp(meta_base + range_size, 4);
	if (meta_range > limits.maxStorageBufferRange) {
		return fail(ClearFbReason::MetaRange);
	}
	struct ClearPredicatePush {
		uint32_t meta_base_dword;
		uint32_t slice_stride_dword;
		uint32_t slice_dwords;
		uint32_t slice_count;
		uint32_t code_count;
		uint32_t code0;
		uint32_t code1;
		uint32_t code2;
		uint32_t code3;
	};
	ClearPredicatePush push {};
	push.meta_base_dword    = static_cast<uint32_t>(meta_base / 4);
	push.slice_stride_dword = static_cast<uint32_t>(slice_size / 4);
	push.slice_dwords       = static_cast<uint32_t>(slice_size / 4);
	push.slice_count        = count;
	push.code_count         = static_cast<uint32_t>(candidate_count);
	uint32_t codes[4]       = {};
	for (size_t i = 0; i < candidate_count; i++) {
		const uint32_t byte = candidates[i].code;
		codes[i]            = byte | (byte << 8) | (byte << 16) | (byte << 24);
	}
	push.code0 = codes[0];
	push.code1 = candidate_count > 1 ? codes[1] : 0;
	push.code2 = candidate_count > 2 ? codes[2] : 0;
	push.code3 = candidate_count > 3 ? codes[3] : 0;
	auto& command = m_scheduler.Current();
	command.EndRendering();
	auto native = command.Recorder();
	vk::BufferMemoryBarrier pre[2] {};
	pre[0].srcAccessMask       = vk::AccessFlagBits::eMemoryWrite;
	pre[0].dstAccessMask       = vk::AccessFlagBits::eShaderRead;
	pre[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	pre[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	pre[0].buffer              = meta->Handle();
	pre[0].offset              = meta_offset;
	pre[0].size                = range_size;
	pre[1]                     = pre[0];
	pre[1].dstAccessMask       = vk::AccessFlagBits::eShaderWrite;
	pre[1].buffer              = m_clear_predicate_buffer->Handle();
	pre[1].offset              = 0;
	pre[1].size                = count * 4 * sizeof(uint32_t);
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eComputeShader, {}, 0, nullptr, 2, pre, 0,
	                       nullptr);
	const vk::DescriptorBufferInfo infos[] {
	    {meta->Handle(), meta_desc_offset, meta_range},
	    {m_clear_predicate_buffer->Handle(), 0, count * 4 * sizeof(uint32_t)},
	};
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t i = 0; i < writes.size(); i++) {
		writes[i].dstBinding      = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[i].pBufferInfo     = &infos[i];
	}
	native.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_clear_predicate_layout, 0,
	                            writes);
	native.bindPipeline(vk::PipelineBindPoint::eCompute, m_clear_predicate_pipeline);
	native.pushConstants(m_clear_predicate_layout, vk::ShaderStageFlagBits::eCompute, 0,
	                     sizeof(push), &push);
	native.dispatch((count + 63u) / 64u, 1, 1);
	vk::BufferMemoryBarrier mid {};
	mid.srcAccessMask       = vk::AccessFlagBits::eShaderWrite;
	mid.dstAccessMask       = vk::AccessFlagBits::eConditionalRenderingReadEXT;
	mid.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	mid.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	mid.buffer              = m_clear_predicate_buffer->Handle();
	mid.offset              = 0;
	mid.size                = count * 4 * sizeof(uint32_t);
	native.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
	                       vk::PipelineStageFlagBits::eConditionalRenderingEXT, {}, 0, nullptr, 1,
	                       &mid, 0, nullptr);
	// The check shader also wrote 0xFFFFFFFF over matched metadata slices; order that
	// write before the later image uses (a readback would download it lazily).
	vk::BufferMemoryBarrier meta_after {};
	meta_after.srcAccessMask       = vk::AccessFlagBits::eShaderWrite;
	meta_after.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	meta_after.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	meta_after.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	meta_after.buffer              = meta->Handle();
	meta_after.offset              = meta_offset;
	meta_after.size                = range_size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &meta_after, 0,
	                       nullptr);
	return TryMaterializeColorClearGpuDraw(id, desc, count, candidates, candidate_count,
	                                     out_reason);
}

// Step 3 draw half: predicated fullscreen-triangle draws write the clear color. One
// (slice, candidate) section each; the predicate dword gates the draw. The check shader
// already wrote 0xFFFFFFFF over matched slices in the metadata GPU copy (the old CPU
// WriteBacking), so the range stays GPU-modified for a later lazy download.
bool TextureCache::TryMaterializeColorClearGpuDraw(ImageId id, const ImageDesc& desc,
                                                  uint32_t count,
                                                  const ClearPredicateCandidate* candidates,
                                                  size_t candidate_count, ClearFbReason* out_reason) {
	const auto fail = [&](ClearFbReason why) {
		if (out_reason != nullptr) {
			*out_reason = why;
		}
		return false;
	};
	const auto& view  = desc.view_info;
	auto&       image = m_slot_images[id];
	// Step 3b regression fix: depth/MSAA and pipeline checks run in the Gpu entry
	// before the dispatch is recorded; reaching here always succeeds.
	const auto pipeline = ClearFillPipelineFor(view.format);
	if (pipeline == nullptr) {
		// Unreachable: the Gpu entry pre-created this pipeline before the dispatch.
		// Kept as a fallback (not an EXIT) so a future format-class mismatch can only
		// cost a sync readback, never a crash.
		return fail(ClearFbReason::PipelineFail);
	}
	auto& command = m_scheduler.Current();
	command.EndRendering();
	auto native = command.Recorder();
	TrackImage(id);
	image.Transit(vk::ImageLayout::eColorAttachmentOptimal,
	              vk::AccessFlagBits2::eColorAttachmentWrite, {}, native);
	for (uint32_t slice = 0; slice < count; slice++) {
		ImageViewInfo layer_view {};
		layer_view.format      = view.format;
		layer_view.type        = view.layer_count == 1 ? vk::ImageViewType::e2D
		                                               : vk::ImageViewType::e2DArray;
		layer_view.base_level  = view.base_level;
		layer_view.base_layer  = view.base_layer + slice;
		layer_view.layer_count = 1;
		layer_view.usage       = vk::ImageUsageFlagBits::eColorAttachment;
		const auto attachment_view = image.FindView(layer_view);
		for (size_t c = 0; c < candidate_count; c++) {
			vk::ConditionalRenderingBeginInfoEXT begin {};
			begin.buffer = m_clear_predicate_buffer->Handle();
			begin.offset = (slice * 4 + c) * sizeof(uint32_t);
			native.beginConditionalRendering(begin);
			vk::RenderingAttachmentInfo attachment {};
			attachment.imageView   = attachment_view;
			attachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
			attachment.loadOp      = vk::AttachmentLoadOp::eLoad;
			attachment.storeOp     = vk::AttachmentStoreOp::eStore;
			vk::RenderingInfo rendering {};
			rendering.renderArea.extent = {std::max(image.info.extent.width >> view.base_level, 1u),
			                               std::max(image.info.extent.height >> view.base_level,
			                                        1u)};
			rendering.layerCount           = 1;
			rendering.colorAttachmentCount = 1;
			rendering.pColorAttachments    = &attachment;
			native.beginRendering(rendering);
			native.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
			// Step 3b regression fix: push the candidate color through the union member
			// matching the attachment class; the pipeline's fragment module reads the
			// same 16 bytes as float/uint/sint.
			switch (ClearFillClassFor(view.format)) {
				case ClearFillClass::Uint:
					native.pushConstants(m_clear_fill_layout, vk::ShaderStageFlagBits::eFragment, 0,
					                     sizeof(candidates[c].clear.uint32),
					                     &candidates[c].clear.uint32);
					break;
				case ClearFillClass::Sint:
					native.pushConstants(m_clear_fill_layout, vk::ShaderStageFlagBits::eFragment, 0,
					                     sizeof(candidates[c].clear.int32),
					                     &candidates[c].clear.int32);
					break;
				default:
					native.pushConstants(m_clear_fill_layout, vk::ShaderStageFlagBits::eFragment, 0,
					                     sizeof(candidates[c].clear.float32),
					                     &candidates[c].clear.float32);
					break;
			}
			const vk::Viewport viewport {0.0f, 0.0f,
			                             static_cast<float>(std::max(image.info.extent.width >>
			                                                         view.base_level,
			                                                         1u)),
			                             static_cast<float>(std::max(image.info.extent.height >>
			                                                         view.base_level,
			                                                         1u)),
			                             0.0f, 1.0f};
			const vk::Rect2D scissor {{0, 0},
			                          {std::max(image.info.extent.width >> view.base_level, 1u),
			                           std::max(image.info.extent.height >> view.base_level,
			                                    1u)}};
			native.setViewportWithCount(1, &viewport);
			native.setScissorWithCount(1, &scissor);
			native.draw(3, 1, 0, 0);
			native.endRendering();
			native.endConditionalRendering();
		}
	}
	CommitGpuWrite(image);
	if (TelemetrySubcountersEnabled()) {
		for (uint32_t slice = 0; slice < count; slice++) {
			(void)slice;
			TelemetryCountCacheClear();
		}
	}
	return true;
}

// Step 3 fill pipeline: fullscreen-triangle vertex shader plus a fragment shader writing
// the push-constant clear color. Dynamic rendering with the image's own color format.
// The fragment module matches the attachment class (float/uint/sint); the vertex shader
// only uses gl_VertexIndex (no push constants or descriptors), so the fragment-only push
// range in the shared layout stays valid.
vk::Pipeline TextureCache::ClearFillPipelineFor(vk::Format format) {
	for (const auto& cached: m_clear_fill_pipelines) {
		if (cached.format == format) {
			return cached.handle;
		}
	}
	if (m_clear_fill_layout == nullptr) {
		const vk::DescriptorSetLayoutCreateInfo descriptor_info {};
		if (m_graphics.device.createDescriptorSetLayout(&descriptor_info, nullptr,
		                                                &m_clear_fill_desc_layout) !=
		    vk::Result::eSuccess) {
			return nullptr;
		}
		struct ClearFillPush {
			float color[4];
		};
		static_assert(sizeof(ClearFillPush) == 16);
		const vk::PushConstantRange push_range {vk::ShaderStageFlagBits::eFragment, 0,
		                                        sizeof(ClearFillPush)};
		vk::PipelineLayoutCreateInfo layout_info {};
		layout_info.setLayoutCount         = 1;
		layout_info.pSetLayouts            = &m_clear_fill_desc_layout;
		layout_info.pushConstantRangeCount = 1;
		layout_info.pPushConstantRanges    = &push_range;
		if (m_graphics.device.createPipelineLayout(&layout_info, nullptr, &m_clear_fill_layout) !=
		    vk::Result::eSuccess) {
			return nullptr;
		}
		m_clear_fill_vert =
		    CompileSPV({CLEAR_PREDICATE_FILL_VERT_SPV, std::size(CLEAR_PREDICATE_FILL_VERT_SPV)},
		               m_graphics.device);
		m_clear_fill_frag = CompileSPV(
		    {CLEAR_PREDICATE_FILL_FRAG_SPV, std::size(CLEAR_PREDICATE_FILL_FRAG_SPV)},
		    m_graphics.device);
		// Step 3b regression fix: integer attachments need a matching fragment output
		// type (uvec4/ivec4); the float module cannot clear them.
		m_clear_fill_frag_uint = CompileSPV(
		    {CLEAR_PREDICATE_FILL_FRAG_UINT_SPV, std::size(CLEAR_PREDICATE_FILL_FRAG_UINT_SPV)},
		    m_graphics.device);
		m_clear_fill_frag_sint = CompileSPV(
		    {CLEAR_PREDICATE_FILL_FRAG_SINT_SPV, std::size(CLEAR_PREDICATE_FILL_FRAG_SINT_SPV)},
		    m_graphics.device);
	}
	vk::ShaderModule frag = m_clear_fill_frag;
	switch (ClearFillClassFor(format)) {
		case ClearFillClass::Uint: frag = m_clear_fill_frag_uint; break;
		case ClearFillClass::Sint: frag = m_clear_fill_frag_sint; break;
		default: break;
	}
	if (frag == nullptr) {
		return nullptr;
	}
	std::array<vk::PipelineShaderStageCreateInfo, 2> stages {};
	stages[0].stage  = vk::ShaderStageFlagBits::eVertex;
	stages[0].module = m_clear_fill_vert;
	stages[0].pName  = "main";
	stages[1].stage  = vk::ShaderStageFlagBits::eFragment;
	stages[1].module = frag;
	stages[1].pName  = "main";
	vk::PipelineVertexInputStateCreateInfo vertex_input {};
	vk::PipelineInputAssemblyStateCreateInfo input_assembly {};
	input_assembly.topology = vk::PrimitiveTopology::eTriangleList;
	vk::PipelineViewportStateCreateInfo viewport {};
	viewport.viewportCount = 1;
	viewport.scissorCount  = 1;
	vk::PipelineRasterizationStateCreateInfo rasterization {};
	rasterization.lineWidth = 1.0f;
	vk::PipelineMultisampleStateCreateInfo multisample {};
	multisample.rasterizationSamples = vk::SampleCountFlagBits::e1;
	vk::PipelineDepthStencilStateCreateInfo depth_stencil {};
	vk::PipelineColorBlendAttachmentState blend_attachment {};
	blend_attachment.colorWriteMask =
	    vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
	    vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
	vk::PipelineColorBlendStateCreateInfo blend {};
	blend.attachmentCount = 1;
	blend.pAttachments    = &blend_attachment;
	std::array<vk::DynamicState, 2> dynamics {vk::DynamicState::eViewport,
	                                          vk::DynamicState::eScissor};
	vk::PipelineDynamicStateCreateInfo dynamic {};
	dynamic.dynamicStateCount = static_cast<uint32_t>(dynamics.size());
	dynamic.pDynamicStates    = dynamics.data();
	vk::PipelineRenderingCreateInfo rendering {};
	rendering.colorAttachmentCount    = 1;
	rendering.pColorAttachmentFormats = &format;
	vk::GraphicsPipelineCreateInfo create {};
	create.stageCount          = static_cast<uint32_t>(stages.size());
	create.pStages             = stages.data();
	create.pVertexInputState   = &vertex_input;
	create.pInputAssemblyState = &input_assembly;
	create.pViewportState      = &viewport;
	create.pRasterizationState = &rasterization;
	create.pMultisampleState   = &multisample;
	create.pDepthStencilState  = &depth_stencil;
	create.pColorBlendState    = &blend;
	create.pDynamicState       = &dynamic;
	create.layout              = m_clear_fill_layout;
	create.pNext               = &rendering;
	vk::Pipeline pipeline = nullptr;
	if (m_graphics.device.createGraphicsPipelines(nullptr, 1, &create, nullptr, &pipeline) !=
	    vk::Result::eSuccess) {
		return nullptr;
	}
	SetVulkanObjectNameF(m_graphics.device, pipeline, "Clear Predicate Fill");
	m_clear_fill_pipelines.push_back({format, pipeline});
	return pipeline;
}

void TextureCache::MaterializeColorClear(ImageId id, const ImageDesc& desc,
                                        uint32_t metadata_base_layer) {
	if (desc.info.metadata.kind != ImageMetadataKind::Dcc &&
	    desc.info.metadata.kind != ImageMetadataKind::Cmask) {
		return;
	}
	const auto range        = desc.info.metadata.range;
	const auto current_tick = m_scheduler.CurrentTick();
	const auto layers       = desc.info.TransferLayers();
	// Measurement only: same clear path with or without subcounters.
	const bool subcounters = TelemetrySubcountersEnabled();
	const auto& view           = desc.view_info;
	const bool  volume_texture = desc.info.IsVolume() && view.type == vk::ImageViewType::e3D;
	const auto  first          = volume_texture ? 0u : metadata_base_layer;
	const auto  image_first    = volume_texture ? 0u : view.base_layer;
	const auto  count          = volume_texture ? desc.info.extent.depth : view.layer_count;
	{
		const auto tracker_begin_us = subcounters ? TelemetryNowUs() : 0;
		std::scoped_lock lock {m_lock};
		auto& image         = m_slot_images[id];
		image.info.metadata = desc.info.metadata;
		// Native color metadata must not retain a reused HTile/CMask/FMask clear flag.
		m_surface_metas.erase(range.address);
		if (range.size == 0 || desc.info.resources.levels != 1) {
			return;
		}
		// A fast-clear stamps the whole DCC metadata range with one uniform code; an ordinary
		// draw also touches these bytes (per-block compression state) but leaves a mixed
		// pattern, so it never passes the all-same-code check below -- yet it re-marks the range
		// GPU-dirty just the same, so gating purely on that flag re-triggers the synchronous
		// readback on every draw to an already-bound render target. Gate on the scheduler tick
		// instead: within one still-unsubmitted recording no GPU write can land in guest memory
		// after the first check, so a later FindImage for the same slices of the same image
		// this tick is redundant. Other slices of the image (another layer or face) keep their
		// own check, a CPU store to the metadata (a guest-side clear) is caught through the
		// tracker below, and a real re-clear in a later recording still gets caught, since the
		// tick will have advanced.
		const bool already_resolved_this_tick =
		    image.dcc_clear_checked_tick == current_tick &&
		    image.dcc_clear_checked_first == first && image.dcc_clear_checked_count == count;
		image.dcc_clear_checked_tick  = current_tick;
		image.dcc_clear_checked_first = first;
		image.dcc_clear_checked_count = count;
		const bool cpu_modified = already_resolved_this_tick &&
		                          m_buffer_cache.IsRegionCpuModified(range.address, range.size);
		if (subcounters) {
			TelemetryAddDrawTargetsClearTracker(TelemetryNowUs() - tracker_begin_us);
		}
		if (already_resolved_this_tick && !cpu_modified) {
			return;
		}
	}
	// These one-mip surfaces use complete 4 KiB color metadata blocks.
	constexpr uint64_t MetadataBlockSize = 0x1000;
	if (!range.Valid() || range.address % MetadataBlockSize != 0 || layers == 0 ||
	    range.size % layers != 0 || (range.size / layers) % MetadataBlockSize != 0) {
		EXIT("TextureCache: color metadata slices must contain aligned 4 KiB blocks\n");
	}
	if (first >= layers || count > layers - first) {
		EXIT("TextureCache: color view exceeds its native metadata slices\n");
	}
	// Finish native metadata writes before reading backing bytes. This can submit the scheduler,
	// so discovery runs before final draw uploads and never holds the texture lock across it.
	const bool gpu_wait_log = ClearWaitLogEnabled();
	const bool skip_clear_wait = SkipClearWaitEnabled();
	uint64_t   gpu_wait_start_us = 0;
	uint32_t   clear_materialized = 0;
	uint32_t   clear_skip_decode = 0;
	uint32_t   clear_skip_all_of = 0;
	if (m_buffer_cache.IsRegionGpuModified(range.address, range.size)) {
		// Experimental: skip the synchronous readback to measure the FPS ceiling.
		if (skip_clear_wait) {
			return;
		}
		// Step 3: replace the synchronous wait with a GPU predicate when possible.
		// Step 3b: per-fallback detail under KYTY_LOG_CLEAR_WAIT=1 (two small LOGFs:
		// geometry first, then format/kind/slice/count plus the reason string).
		if (ClearGpuPredicateEnabled()) {
			ClearFbReason      fb_reason = ClearFbReason::Unsupported;
			const auto         predicate_result =
			    TryMaterializeColorClearGpu(id, desc, first, count, range.size / layers,
			                               range.address, range.size, &fb_reason);
			using enum ClearPredicateResult;
			if (predicate_result == Applied || predicate_result == SkippedNoCand) {
				if (subcounters && predicate_result == Applied) {
					TelemetryCountClearGpuPredicate();
				}
				return;
			}
			if (gpu_wait_log) {
				static uint32_t fb_logged = 0;
				if (fb_logged < 200) {
					const auto slice_bytes = layers != 0 ? range.size / layers : 0;
					const auto extent      = desc.info.extent;
					LOGF("ClearFb: addr=0x%016" PRIx64 " size=0x%016" PRIx64
					     " image=%u,%u,%u\n",
					     range.address, range.size, extent.width, extent.height,
					     extent.depth);
					LOGF("ClearFbInfo: format=%u kind=%u slice=%llu count=%u reason=%s\n",
					     static_cast<uint32_t>(view.format),
					     static_cast<unsigned>(desc.info.metadata.kind),
					     static_cast<unsigned long long>(slice_bytes),
					     static_cast<unsigned>(count), ClearFbReasonName(fb_reason));
					fb_logged++;
				}
			}
		}
		// Exactly one wait counter per real GPU wait (skipped waits are not waits).
		TelemetryCountClearGpuWait();
		TelemetryCountGpusyncClear();
		const auto gpu_wait_begin_us = subcounters ? TelemetryNowUs() : 0;
		if (gpu_wait_log) {
			gpu_wait_start_us = TelemetryNowUs();
		}
		m_buffer_cache.ReadMemory(range.address, range.size, false);
		if (subcounters) {
			TelemetryAddDrawTargetsClearGpuWait(TelemetryNowUs() - gpu_wait_begin_us);
		}
	}
	const auto slice_size = range.size / layers;
	for (uint32_t slice = 0; slice < count; slice++) {
		const auto address = range.address + slice_size * (first + slice);
		uint8_t code = 0;
		const auto readback_begin_us = subcounters ? TelemetryNowUs() : 0;
		if (!LibKernel::Memory::TryReadBacking(address, &code, sizeof(code))) {
			EXIT("TextureCache: failed to read color metadata backing\n");
		}
		vk::ClearValue clear {};
		if (!DecodeColorClear(desc, code, clear.color)) {
			if (subcounters) {
				TelemetryAddDrawTargetsClearReadback(TelemetryNowUs() - readback_begin_us);
			}
			clear_skip_decode++;
			continue;
		}
		if (ClearScratchEnabled()) {
			if (!IsClearSliceUniform(address, slice_size, code)) {
				if (subcounters) {
					TelemetryAddDrawTargetsClearReadback(TelemetryNowUs() - readback_begin_us);
				}
				clear_skip_all_of++;
				continue;
			}
			if (subcounters) {
				TelemetryAddDrawTargetsClearReadback(TelemetryNowUs() - readback_begin_us);
			}
		} else {
			std::vector<uint8_t> bytes(static_cast<size_t>(slice_size));
			if (!LibKernel::Memory::TryReadBacking(address, bytes.data(), bytes.size())) {
				EXIT("TextureCache: failed to read color metadata slice\n");
			}
			if (!std::all_of(bytes.begin(), bytes.end(), [code](uint8_t byte) { return byte == code; })) {
				if (subcounters) {
					TelemetryAddDrawTargetsClearReadback(TelemetryNowUs() - readback_begin_us);
				}
				clear_skip_all_of++;
				continue;
			}
			if (subcounters) {
				TelemetryAddDrawTargetsClearReadback(TelemetryNowUs() - readback_begin_us);
			}
		}
		{
			std::scoped_lock lock {m_lock};
			ClearImage(m_scheduler.Current(), id, view.format,
			           {vk::ImageAspectFlagBits::eColor, view.base_level, view.level_count,
			            image_first + slice, 1}, clear);
			// Measurement only: counts actual color-clear materializations per frame.
			if (TelemetrySubcountersEnabled()) {
				TelemetryCountCacheClear();
				clear_materialized++;
			}
		}
		// Publish the conversion's expanded keys without treating them as guest writes
		// to overlapping image data. Invalidate the buffer before updating its backing.
		if (desc.type != BindingType::VideoOut) {
			const auto writeback_begin_us = subcounters ? TelemetryNowUs() : 0;
			// The scratch path reuses the thread-local 64 KiB chunk; the legacy path keeps
			// the per-slice vector. Both invalidate before writing, both fill with 0xFF.
			if (ClearScratchEnabled()) {
				m_buffer_cache.InvalidateMemory(address, slice_size);
				WriteClearSliceFF(address, slice_size);
			} else {
				std::vector<uint8_t> ff(static_cast<size_t>(slice_size), uint8_t {0xff});
				m_buffer_cache.InvalidateMemory(address, slice_size);
				LibKernel::Memory::WriteBacking(address, ff.data(), ff.size());
			}
			if (subcounters) {
				TelemetryAddDrawTargetsClearWriteback(TelemetryNowUs() - writeback_begin_us);
			}
		}
	}
	// Measurement only: no logic changes. Log per-GPU-wait clear outcomes under KYTY_LOG_CLEAR_WAIT=1.
	if (gpu_wait_log && gpu_wait_start_us != 0) {
		static uint32_t session_logged = 0;
		if (session_logged < 200) {
			const auto wait_us = TelemetryNowUs() - gpu_wait_start_us;
			const auto image_extent = desc.info.extent;
			const auto completed_tick = m_scheduler.IsFree(current_tick - 1) ? (current_tick - 1) : 0;
			const char* result = clear_materialized > 0   ? "clear materialized"
			                   : clear_skip_decode > 0    ? "skip decode"
			                   : clear_skip_all_of > 0    ? "skip all_of"
			                   :                            "no slices";
			LOGF("ClearWait: addr=0x%016" PRIx64 " size=0x%016" PRIx64 " format=%u "
			     "image=%u,%u,%u last_gpu_tick=%llu completed_tick=%llu wait_us=%llu result=%s\n",
			     range.address, range.size, static_cast<uint32_t>(view.format),
			     image_extent.width, image_extent.height, image_extent.depth,
			     static_cast<unsigned long long>(current_tick),
			     static_cast<unsigned long long>(completed_tick),
			     static_cast<unsigned long long>(wait_us), result);
			// Step 3a: the last guest-GPU op that marked this range GPU-written, plus
			// the first code byte read after the wait. 8 specifiers, 8 arguments.
			const auto origin = BufferCache::FindGpuWriteOrigin(range.address, range.size);
			uint8_t    first_code = 0;
			(void)LibKernel::Memory::TryReadBacking(range.address, &first_code, sizeof(first_code));
			LOGF("ClearWaitWriter: op=%s value=0x%016" PRIx64 " size=0x%016" PRIx64
			     " shader=0x%016" PRIx64 " tick=%llu draw=%u dispatch=%u code=0x%02x\n",
			     BufferCache::GpuWriteOpName(origin.op),
			     static_cast<uint64_t>(origin.value), static_cast<uint64_t>(origin.size),
			     static_cast<uint64_t>(origin.shader_hash),
			     static_cast<unsigned long long>(origin.tick),
			     static_cast<unsigned>(origin.frame_draw),
			     static_cast<unsigned>(origin.frame_dispatch), first_code);
			session_logged++;
		}
	}
}


void TextureCache::RefreshImage(ImageId id) {
	// Measurement only: the refresh runs identically with or without subcounters.

	const bool subcounters = TelemetrySubcountersEnabled();
	const auto refresh_begin_us = subcounters ? TelemetryNowUs() : 0;
	auto& image = m_slot_images[id];
	if (image.depth_id &&
	    (m_slot_images[image.depth_id].info.metadata.stencil_compressed ||
	     m_slot_images[image.depth_id].info.samples != 1)) {
		return;
	}
	TrackImage(id);
	if (image.IsMaybeCpuDirty()) {
		const auto hash = image.HashGuestEdges();
		if (image.NeedsMaybeCpuHash()) {
			image.SetMaybeCpuHash(hash);
			return;
		}
		(void)image.ResolveMaybeCpuHash(hash);
	}
	bool cpu_dirty = image.IsBufferModified() || image.IsDefinitelyCpuDirty();
	if (image.info.metadata.compression != VideoOutCompression::Uncompressed) {
		if (cpu_dirty) {
			EXIT("TextureCache: compressed guest image refresh is unsupported\n");
		}
		return;
	}
	if (!cpu_dirty) {
		return;
	}
	InitializeImage(id);
	if (subcounters) {
		TelemetryAddDrawTargetsRefresh(TelemetryNowUs() - refresh_begin_us);
		TelemetryCountCacheRefresh();
	}
}

ImageId TextureCache::AssociateStencil(ImageId depth_id, GuestRange stencil) {
	if (!stencil.Valid()) {
		EXIT("TextureCache: invalid stencil association range\n");
	}
	auto& depth = m_slot_images[depth_id];
	if (!depth.info.IsDepth() || !depth.info.HasStencil()) {
		EXIT("TextureCache: stencil association requires a depth/stencil image\n");
	}

	ImageId association {};
	for (const auto id: FindImagesInRegion(stencil.address, stencil.size, false)) {
		const auto owner = m_slot_images.try_get(id);
		if (owner != nullptr && owner->info.data == stencil &&
		    owner->info.extent == depth.info.extent) {
			association = id;
		}
	}
	if (!association) {
		ImageInfo info {};
		info.data   = stencil;
		info.extent = depth.info.extent;
		association = InsertImage(info);
	}
	auto& record = m_slot_images[association];
	TouchImage(record);
	record.depth_id             = depth_id;
	record.stencil_subresources = depth.stencil_subresources;
	return association;
}

ImageId TextureCache::FindImage(ImageDesc& desc, bool exact_format) {
	auto& command = m_scheduler.Current();
	if (command.IsInvalid()) {
		EXIT("TextureCache: image lookup requires a valid command buffer\n");
	}
	// Measurement only: same lookup path with or without subcounters.
	const bool subcounters = TelemetrySubcountersEnabled();
	const auto lookup_begin_us = subcounters ? TelemetryNowUs() : 0;
	if (subcounters) {
		TelemetryCountCacheLookup();
	}
	ValidateImageDesc(desc);
	if (desc.info.data.Empty()) {
		std::scoped_lock lock {m_lock};
		return GetNullImage(desc);
	}
	const auto metadata_base_layer = desc.view_info.base_layer;

	ImageId result {};
	uint32_t candidate_count = 0;
	bool same_backing_hit = false;
	bool overlap_hit = false;
	bool inserted = false;
	{
		std::scoped_lock lock {m_lock};
		const auto       candidates =
		    FindImagesInRegion(desc.info.data.address, desc.info.data.size, false);
		candidate_count = static_cast<uint32_t>(candidates.size());

		for (const auto id: candidates) {
			const auto& image = m_slot_images[id];
			if (SameBacking(image.info, desc.info, exact_format)) {
				result = id;
			}
		}
		same_backing_hit = static_cast<bool>(result);

		int32_t view_mip   = -1;
		int32_t view_layer = -1;
		if (!result) {
			const auto overlap_begin_us = subcounters ? TelemetryNowUs() : 0;
			for (const auto candidate: candidates) {
				view_mip                = -1;
				view_layer              = -1;
				const auto& merged_info = result ? m_slot_images[result].info : desc.info;
				const auto  overlap     = ResolveOverlap(merged_info, desc.type, candidate, result);
				if (overlap.image) {
					result     = overlap.image;
					view_mip   = overlap.mip;
					view_layer = overlap.layer;
				}
			}
			if (subcounters) {
				TelemetryAddDrawTargetsOverlap(TelemetryNowUs() - overlap_begin_us);
			}
			overlap_hit = static_cast<bool>(result);
		}

		if (result) {
			auto& resolved = m_slot_images[result];
			if (exact_format && resolved.info.pixel_format != desc.info.pixel_format) {
				result = {};
			} else if (resolved.info.resources < desc.info.resources) {
				FreeImage(result);
				result = {};
			}
		}
		if (!result) {
			const auto insert_begin_us = subcounters ? TelemetryNowUs() : 0;
			result         = InsertImage(desc.info);
			auto& inserted_image = m_slot_images[result];
			if (m_buffer_cache.HasGpuDirtyBytes(inserted_image.info.data.address,
			                                    inserted_image.info.data.size)) {
				inserted_image.MarkBufferModified();
			}
			if (subcounters) {
				TelemetryAddDrawTargetsCacheInsert(TelemetryNowUs() - insert_begin_us);
			}
			inserted = true;
		}
		auto& image = m_slot_images[result];
		if (view_mip >= 0) {
			desc.view_info.base_level = static_cast<uint32_t>(view_mip);
		}
		if (view_layer >= 0) {
			desc.view_info.base_layer = static_cast<uint32_t>(view_layer);
		}
		image.tick_accessed_last = m_scheduler.CurrentTick();
		TouchImage(image);
		if (subcounters) {
			TelemetryAddDrawTargetsCacheLookup(TelemetryNowUs() - lookup_begin_us);
			TelemetryAddCacheCandidates(candidate_count);
			if (same_backing_hit) {
				TelemetryCountCacheSameBackingHit();
			} else if (overlap_hit) {
				TelemetryCountCacheOverlapHit();
			}
			if (inserted) {
				TelemetryCountCacheInsert();
			}
			TelemetrySetCacheImages(static_cast<uint32_t>(m_slot_images.size()));
		}
	}
	const auto clear_begin_us = subcounters ? TelemetryNowUs() : 0;
	MaterializeColorClear(result, desc, metadata_base_layer);
	if (subcounters) {
		TelemetryAddDrawTargetsClear(TelemetryNowUs() - clear_begin_us);
	}
	if (desc.type == BindingType::VideoOut &&
	    desc.info.metadata.compression != VideoOutCompression::Uncompressed) {
		std::scoped_lock lock {m_lock};
		const auto& image = m_slot_images[result];
		const bool guest_dirty = image.IsBufferModified() || image.IsCpuDirty();
		const bool native_current =
		    (image.usage.render_target || image.IsGpuModified()) && !guest_dirty;
		if (!native_current) {
			EXIT("TextureCache: compressed video-out read requires clean native GPU "
			     "contents\n");
		}
	}
	return result;
}

void TextureCache::UpdateImage(ImageId id) {
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	TouchImage(image);
	RefreshImage(id);
}

ImageId TextureCache::FindImageFromRange(uint64_t address, uint64_t size, bool ensure_valid) {
	if (!GuestRange {address, size}.Valid()) {
		return {};
	}
	std::scoped_lock lock {m_lock};
	ImageIds         matches;
	// Only an image that starts at the address qualifies, and it covers that byte: one page to
	// query instead of every page of the range (a multi-megabyte texel buffer on every draw).
	for (const auto id: FindImagesInRegion(address, 1, false)) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr || owner->info.data.address != address) {
			continue;
		}
		if (ensure_valid && owner->depth_id) {
			owner = m_slot_images.try_get(owner->depth_id);
		}
		if (owner == nullptr || (ensure_valid && !owner->SafeToDownload())) {
			continue;
		}
		matches.push_back(id);
	}
	ImageId selected {};
	if (matches.size() == 1) {
		selected = matches.front();
	} else {
		for (const auto id: matches) {
			const auto& image = m_slot_images[id];
			if (image.info.data.size == size) {
				selected = id;
				break;
			}
		}
	}
	if (selected && ensure_valid) {
		const auto owner = m_slot_images.try_get(selected);
		if (owner != nullptr && owner->depth_id) {
			selected = owner->depth_id;
		}
	}
	return selected;
}

vk::ImageView TextureCache::FindTexture(ImageId id, const ImageDesc& desc) {
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	TouchImage(image);
	if (!image.info.data.Empty()) {
		if (!image.registered || image.depth_id || image.binding.needs_rebind) {
			EXIT("TextureCache: texture requires rediscovery before final acquisition\n");
		}
	}
	if (desc.type == BindingType::Storage) {
		image.MarkGpuModified();
	}
	if (!image.info.data.Empty()) {
		RefreshImage(id);
		if (image.info.HasStencil() &&
		    desc.info.data.address >= image.info.stencil.address &&
		    desc.info.data.End() <= image.info.stencil.End()) {
			for (const auto stencil_id:
			     FindImagesInRegion(image.info.stencil.address, image.info.stencil.size, false)) {
				const auto* stencil = m_slot_images.try_get(stencil_id);
				if (stencil != nullptr && stencil->depth_id == id &&
				    stencil->info.data == image.info.stencil) {
					RefreshImage(stencil_id);
					break;
				}
			}
		}
	}
	switch (desc.type) {
		case BindingType::Texture: break;
		case BindingType::Storage:
			if (!image.info.data.Empty()) {
				CommitGpuWrite(image);
			}
			TrackImageDownload(id, image);
			break;
		default: EXIT("TextureCache: invalid texture binding\n");
	}
	return image.FindView(desc.view_info);
}

vk::ImageView TextureCache::FindRenderTarget(ImageId id, const ImageDesc& desc) {
	if (desc.type != BindingType::RenderTarget) {
		EXIT("TextureCache: invalid color-target binding\n");
	}
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	if (!image.registered || image.depth_id || image.binding.needs_rebind) {
		EXIT("TextureCache: color target requires rediscovery before final acquisition\n");
	}
	TouchImage(image);
	image.MarkGpuModified();
	image.usage.render_target = true;
	RefreshImage(id);
	CommitGpuWrite(image);
	TrackImageDownload(id, image);
	return image.FindView(desc.view_info);
}

vk::ImageView TextureCache::FindDepthTarget(ImageId id, const ImageDesc& desc) {
	if (desc.type != BindingType::DepthTarget) {
		EXIT("TextureCache: invalid depth-target binding\n");
	}
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	if (!image.registered || image.depth_id || image.binding.needs_rebind) {
		EXIT("TextureCache: depth target requires rediscovery before final acquisition\n");
	}
	TouchImage(image);
	image.MarkGpuModified();
	image.usage.depth_target = true;
	if (desc.info.HasStencil()) {
		const auto layers = image.info.resources.layers;
		EXIT_IF(layers == 0 || image.info.data.size % layers != 0 ||
		        desc.info.data.address < image.info.data.address);
		const auto slice_size = image.info.data.size / layers;
		const auto offset     = desc.info.data.address - image.info.data.address;
		EXIT_IF(slice_size == 0 || offset % slice_size != 0 || offset / slice_size >= layers ||
		        desc.info.resources.layers > layers - offset / slice_size);
		image.stencil_subresources = {0, desc.info.resources.levels,
		                             static_cast<uint32_t>(offset / slice_size),
		                             desc.info.resources.layers};
	}
	image.info.stencil = desc.info.stencil;
	image.info.metadata = desc.info.metadata;
	if (desc.info.HasMetadata()) {
		m_surface_metas.emplace(desc.info.metadata.range.address,
		                        MetaDataInfo {.type       = MetaDataInfo::Type::HTile,
		                                      .clear_mask = image.info.htile_clear_mask});
	}
	RefreshImage(id);
	CommitGpuWrite(image);
	if (desc.info.HasStencil()) {
		RefreshImage(AssociateStencil(id, desc.info.stencil));
	}
	return image.FindView(desc.view_info);
}

void TextureCache::MarkGpuWritten(ImageId id) {
	std::scoped_lock lock {m_lock};
	auto&            image = m_slot_images[id];
	if (!image.registered || image.depth_id) {
		EXIT("TextureCache: cannot mark an unavailable image GPU-written\n");
	}
	TrackImage(id);
	CommitGpuWrite(image);
	if (image.info.HasStencil()) {
		const auto stencil_id = AssociateStencil(id, image.info.stencil);
		TrackImage(stencil_id);
		CommitGpuWrite(m_slot_images[stencil_id]);
	}
}

void TextureCache::CommitGpuWrite(Image& image) {
	if (!image.depth_id && image.backing.image == nullptr) {
		EXIT("TextureCache: GPU writes require a native image or stencil association\n");
	}
	image.ClearBufferModified();
	if (image.IsCpuDirty()) {
		image.RefreshComplete();
	}
	// Aliases aged by frames stay alive side by side. A single owner among them: this write
	// supersedes the bytes of every other live image overlapping it, so none of those may later
	// be written back (collector, linear readback, buffer sync) over it. Their native contents
	// stay as they are; the tick rule freed them without a write-back. Scanned only when the
	// owner changes, not on every bind.
	if (AliasAgeByFrames() && !image.alias_owner && !image.depth_id && image.registered &&
	    !image.info.data.Empty()) {
		for (const auto id: FindImagesInRegion(image.info.data.address, image.info.data.size, false)) {
			auto* other = m_slot_images.try_get(id);
			if (other == nullptr || other == &image || other->depth_id || !other->registered) {
				continue;
			}
			other->alias_owner = false;
			if (other->IsGpuModified()) {
				other->ClearGpuModified();
			}
		}
		image.alias_owner = true;
	}
	image.MarkGpuModified();
}

bool TextureCache::ClearImageFromBuffer(CommandBuffer& command, uint64_t address, uint64_t size,
                                        uint32_t packed_clear) {
	if (command.IsInvalid() || !GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid image clear\n");
	}
	std::scoped_lock      lock {m_lock};
	ImageId               selected {};
	ImageId               stencil_id {};
	vk::ImageAspectFlags  aspect {};
	ImageSubresourceRange subresources;
	for (const auto id: FindImagesInRegion(address, size, false)) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr) {
			continue;
		}
		vk::ImageAspectFlags candidate {};
		ImageId              candidate_id = id;
		ImageSubresourceRange candidate_subresources {
		    0, owner->info.resources.levels, 0, owner->info.TransferLayers()};
		if (owner->depth_id && owner->info.data.address == address &&
		    owner->info.data.size == size) {
			candidate              = vk::ImageAspectFlagBits::eStencil;
			candidate_id           = owner->depth_id;
			candidate_subresources = owner->stencil_subresources;
			owner        = m_slot_images.try_get(candidate_id);
			if (owner == nullptr || owner->backing.image == nullptr || !owner->info.HasStencil()) {
				continue;
			}
		} else if (!owner->depth_id && owner->info.data.address == address &&
		           owner->info.data.size == size) {
			candidate = owner->info.IsDepth() ? vk::ImageAspectFlagBits::eDepth
			                                  : vk::ImageAspectFlagBits::eColor;
		}
		if (!candidate) {
			continue;
		}
		if (selected && (selected != candidate_id || subresources != candidate_subresources)) {
			return false;
		}
		selected     = candidate_id;
		stencil_id   = candidate == vk::ImageAspectFlagBits::eStencil ? id : ImageId {};
		aspect       = candidate;
		subresources = candidate_subresources;
	}
	if (!selected) {
		return false;
	}
	auto&          image = m_slot_images[selected];
	vk::ClearValue clear {};
	if (aspect == vk::ImageAspectFlagBits::eColor) {
		if (!DecodeColorDwordFill(image.info.pixel_format, packed_clear, clear.color)) {
			return false;
		}
	} else {
		uint8_t stencil_clear = 0;
		if ((aspect == vk::ImageAspectFlagBits::eDepth &&
		     !DecodePackedDepthClear(image.info.pixel_format, packed_clear, clear.depthStencil.depth)) ||
		    (aspect == vk::ImageAspectFlagBits::eStencil &&
		     !DecodePackedStencilClear(packed_clear, stencil_clear))) {
			return false;
		}
		clear.depthStencil.stencil = stencil_clear;
	}
	ClearImage(command, selected, image.backing.format,
	           {aspect, subresources.base_level, subresources.level_count,
	            subresources.base_layer, subresources.layer_count}, clear);
	if (stencil_id) {
		TrackImage(stencil_id);
		CommitGpuWrite(m_slot_images[stencil_id]);
	}
	return true;
}

void TextureCache::ClearImage(CommandBuffer& command, ImageId id, vk::Format format,
                              const vk::ImageSubresourceRange& range, const vk::ClearValue& clear) {
	auto& image = m_slot_images[id];
	const auto aspects = image.info.IsDepth() ? ImageViewOps::DepthAspectMask(image.backing.format)
	                                          : vk::ImageAspectFlagBits::eColor;
	EXIT_IF(range.baseMipLevel >= image.info.resources.levels);
	const auto layers = image.info.IsVolume()
	                        ? std::max(image.info.extent.depth >> range.baseMipLevel, 1u)
	                        : image.backing.layers;
	EXIT_IF(command.IsInvalid() || image.depth_id || !range.aspectMask || range.levelCount == 0 ||
	        range.levelCount > image.info.resources.levels - range.baseMipLevel ||
	        range.layerCount == 0 || range.baseArrayLayer >= layers ||
	        range.layerCount > layers - range.baseArrayLayer ||
	        (range.aspectMask & aspects) != range.aspectMask);
	const bool full_subresources = range.baseMipLevel == 0 &&
	                               range.levelCount == image.info.resources.levels &&
	                               range.baseArrayLayer == 0 && range.layerCount == layers;
	const bool full_image = range.aspectMask == aspects && full_subresources;
	TrackImage(id);
	if (!full_image && (image.IsBufferModified() || image.IsCpuDirty())) {
		InitializeImage(id);
		if (image.info.samples == 1 && (image.IsBufferModified() || image.IsCpuDirty())) {
			EXIT("TextureCache: image clear retained guest ownership\n");
		}
	}
	if (image.info.HasStencil() && (range.aspectMask & vk::ImageAspectFlagBits::eStencil)) {
		const auto stencil_id = AssociateStencil(id, image.info.stencil);
		if (!full_subresources) {
			RefreshImage(stencil_id);
		} else {
			TrackImage(stencil_id);
			CommitGpuWrite(m_slot_images[stencil_id]);
		}
	}
	command.EndRendering();
	// Transfer clears use the backing format; aliased clears must encode through their view.
	if (format != image.backing.format || (image.info.IsVolume() && !full_image)) {
		EXIT_NOT_IMPLEMENTED(range.aspectMask != vk::ImageAspectFlagBits::eColor ||
		                     range.levelCount != 1);
		ImageViewInfo view {};
		view.format = format;
		view.type   = range.layerCount == 1 ? vk::ImageViewType::e2D : vk::ImageViewType::e2DArray;
		view.base_level  = range.baseMipLevel;
		view.base_layer  = range.baseArrayLayer;
		view.layer_count = range.layerCount;
		view.usage       = vk::ImageUsageFlagBits::eColorAttachment;
		image.Transit(vk::ImageLayout::eColorAttachmentOptimal,
		              vk::AccessFlagBits2::eColorAttachmentWrite, {}, command.Recorder());
		vk::RenderingAttachmentInfo attachment {};
		attachment.imageView   = image.FindView(view);
		attachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
		attachment.loadOp      = vk::AttachmentLoadOp::eClear;
		attachment.storeOp     = vk::AttachmentStoreOp::eStore;
		attachment.clearValue  = clear;
		vk::RenderingInfo rendering {};
		rendering.renderArea.extent = {
		    std::max(image.info.extent.width >> range.baseMipLevel, 1u),
		    std::max(image.info.extent.height >> range.baseMipLevel, 1u)};
		rendering.layerCount           = range.layerCount;
		rendering.colorAttachmentCount = 1;
		rendering.pColorAttachments    = &attachment;
		command.Recorder().beginRendering(&rendering);
		command.Recorder().endRendering();
		CommitGpuWrite(image);
		return;
	}
	image.Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {},
	              command.Recorder());
	auto native_range = range;
	if (image.info.IsVolume()) {
		native_range.baseArrayLayer = 0;
		native_range.layerCount     = 1;
	}
	if (range.aspectMask == vk::ImageAspectFlagBits::eColor) {
		command.Recorder().clearColorImage(image.backing.image, vk::ImageLayout::eTransferDstOptimal,
		                                 &clear.color, 1, &native_range);
	} else {
		command.Recorder().clearDepthStencilImage(image.backing.image,
		                                        vk::ImageLayout::eTransferDstOptimal,
		                                        &clear.depthStencil, 1, &native_range);
	}
	CommitGpuWrite(image);
}

void TextureCache::InvalidateMemory(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid memory-invalidation range\n");
	}
	std::scoped_lock lock {m_lock};
	InvalidateCpuAliases(address, size);
}

void TextureCache::DownloadDepth(Image& image, Buffer& destination, uint64_t destination_offset) {
	const auto&    info             = image.info;
	const auto     layers           = info.resources.layers;
	const auto     full_slice_size  = info.data.size / layers;
	const auto     transfer_bytes   = DepthAspectTransferBytes(info.pixel_format);
	const uint64_t texels_per_slice = static_cast<uint64_t>(info.pitch) * info.extent.height;
	EXIT_NOT_IMPLEMENTED(transfer_bytes == 0 || texels_per_slice > UINT32_MAX ||
	                     texels_per_slice > UINT64_MAX / transfer_bytes ||
	                     texels_per_slice > UINT64_MAX / info.bytes_per_block);
	const uint64_t transfer_slice = texels_per_slice * transfer_bytes;
	const uint64_t guest_slice    = texels_per_slice * info.bytes_per_block;
	EXIT_NOT_IMPLEMENTED(transfer_slice > UINT64_MAX / layers);
	const uint64_t transfer_size = transfer_slice * layers;
	EXIT_NOT_IMPLEMENTED(guest_slice > full_slice_size);
	auto copies = BuildDepthCopies(info, full_slice_size, vk::ImageAspectFlagBits::eDepth);
	if (transfer_bytes == info.bytes_per_block) {
		if (!info.IsTiled()) {
			for (auto& copy: copies) {
				copy.bufferOffset += destination_offset;
			}
			image.Download(copies, destination.Handle(), destination_offset, info.data.size);
			return;
		}
		const auto tiles = BuildDepthTiles(info);
		m_tiler.TileImage(image, copies, destination.Handle(), destination_offset, info.data.size,
		                  info.data.size, tiles);
		return;
	}
	EXIT_NOT_IMPLEMENTED(info.bytes_per_block != sizeof(uint16_t) ||
	                     transfer_bytes != sizeof(uint32_t));
	for (uint32_t layer = 0; layer < layers; layer++) {
		copies[layer].bufferOffset = transfer_slice * layer;
	}
	auto host_linear = m_tiler.GetScratchBuffer(transfer_size);
	image.Download(copies, host_linear.buffer, 0, host_linear.size);
	const bool tiled        = info.IsTiled();
	auto       guest_linear = tiled ? m_tiler.GetScratchBuffer(info.data.size, host_linear.buffer)
	                                : TileManager::Result {destination.Handle(), destination_offset,
	                                                       destination.Size() - destination_offset};
	m_tiler.ConvertD16(host_linear, guest_linear, TileManager::D16Direction::Demote,
	                   DepthAspectTransferFormat(info.pixel_format) == vk::Format::eD32Sfloat,
	                   {.width               = info.extent.width,
	                    .height              = info.extent.height,
	                    .layers              = layers,
	                    .source_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint32_t),
	                    .target_row_stride   = static_cast<uint64_t>(info.pitch) * sizeof(uint16_t),
	                    .source_slice_stride = transfer_slice,
	                    .target_slice_stride = full_slice_size});
	if (!tiled) {
		return;
	}
	const auto tiles = BuildDepthTiles(info);
	m_tiler.Tile(guest_linear.buffer, guest_linear.offset, info.data.size, destination.Handle(),
	             destination_offset, info.data.size, tiles);
}

void TextureCache::DownloadImage(Image& image, Buffer& destination, uint64_t destination_offset,
                                     uint64_t destination_size, ImageDownload transfer) {
	if (!transfer.valid) {
		EXIT("TextureCache: invalid image download transfer\n");
	}
	if (transfer.depth_target) {
		if (destination_size != image.info.data.size) {
			EXIT("TextureCache: partial depth image download is unsupported\n");
		}
		DownloadDepth(image, destination, destination_offset);
		return;
	}

	auto&      texture   = transfer.texture;
	const auto transform = texture.color_transform;
	if (texture.tiles.empty()) {
		if (transform != ColorTransform::None) {
			auto linear = m_tiler.GetScratchBuffer(destination_size);
			image.Download(texture.regions, linear.buffer, 0, linear.size);
			m_tiler.TransformColor(linear,
			                       {destination.Handle(), destination_offset, destination_size},
			                       transform, false);
			return;
		}
		for (auto& copy: texture.regions) {
			copy.bufferOffset += destination_offset;
		}
		image.Download(texture.regions, destination.Handle(), destination_offset, destination_size);
		return;
	}

	m_tiler.TileImage(image, texture.regions, destination.Handle(), destination_offset,
	                  destination_size, texture.LinearSize(), texture.tiles, transform);
}

bool BufferCache::SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	const auto selected = m_texture_cache.FindImageFromRange(vaddr, size);
	if (!selected) {
		return false;
	}

	std::scoped_lock lock {m_texture_cache.m_lock};
	auto& image = m_texture_cache.m_slot_images[selected];
	// The GPU thread owns image retirement; CPU invalidation can dirty this image after lookup.
	if (!image.SafeToDownload()) {
		return false;
	}
	if (!buffer.IsInBounds(image.info.data.address, 1)) {
		return false;
	}
	const auto buf_offset = buffer.Offset(image.info.data.address);
	const auto available  = buffer.Size() - buf_offset;
	uint32_t   levels     = 0;
	uint64_t   copy_size  = 0;
	if (image.info.IsVolume()) {
		// Volume mips contain strided block slices, so a mip's linear span cannot prove that
		// every retained slice fits. Keep volume synchronization whole-image only.
		if (!buffer.IsInBounds(image.info.data.address, image.info.data.size)) {
			return false;
		}
		levels    = image.info.resources.levels;
		copy_size = image.info.data.size;
	} else {
		for (; levels < image.info.resources.levels; ++levels) {
			const auto& mip = image.info.mip_layout[levels];
			if (mip.size == 0 || mip.offset > available || mip.size > available - mip.offset) {
				break;
			}
			copy_size = std::max(copy_size, mip.offset + mip.size);
		}
	}
	if (copy_size == 0) {
		return false;
	}
	auto transfer = m_texture_cache.BuildDownload(image);
	if (!transfer.valid) {
		return false;
	}
	if (transfer.depth_target && copy_size != image.info.data.size) {
		return false;
	}
	if (!transfer.depth_target && levels < image.info.resources.levels) {
		auto& texture = transfer.texture;
		std::erase_if(texture.regions, [levels](const vk::BufferImageCopy& region) {
			return region.imageSubresource.mipLevel >= levels;
		});
		if (texture.regions.empty()) {
			return false;
		}
		if (!texture.tiles.empty()) {
			texture.tiles.clear();
			if (!TextureBuildGpuTileInfos(copy_size, texture.regions, texture.layout, levels,
			                              texture.tiles)) {
				return false;
			}
		}
	}
	m_texture_cache.DownloadImage(image, buffer, buf_offset, copy_size, std::move(transfer));
	return true;
}

bool TextureCache::DownloadImageMemory(ImageId id, bool evicting) {
	auto& image = m_slot_images[id];
	if (image.depth_id) {
		return false;
	}
	auto transfer = BuildDownload(image);
	if (!transfer.valid || !image.SafeToDownload()) {
		return false;
	}
	const auto range = image.info.data;
	auto&      ring  = m_buffer_cache.GetUtilityBuffer(MemoryUsage::Download);
	// An image larger than the download ring gets a staging buffer of its own for this one
	// transfer: the ring cannot map it, and its regions cannot be cut into ring-sized pieces
	// the way a buffer download can.
	std::shared_ptr<Buffer> dedicated;
	Buffer*                 download = &ring;
	uint8_t*                mapped   = nullptr;
	uint64_t                offset   = 0;
	if (range.size > ring.Size()) {
		dedicated = std::make_shared<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                     AllFlags, range.size);
		download  = dedicated.get();
		mapped    = dedicated->Mapped().data();
		if (mapped == nullptr) {
			return false;
		}
	} else {
		std::tie(mapped, offset) =
		    ring.Map(range.size, std::max<uint64_t>(image.info.bytes_per_block, 4));
		if (mapped == nullptr) {
			EXIT("TextureCache: failed to map reusable download buffer\n");
		}
		ring.Commit();
	}
	if (!LibKernel::Memory::TryReadBacking(range.address, mapped, range.size)) {
		return false;
	}
	const bool guard      = evicting && EvictWriteBackGuard();
	uint64_t   guest_hash = 0;
	if (guard && !HashGuestBacking(range.address, range.size, guest_hash)) {
		return false;
	}
	download->Flush(offset, range.size);

	DownloadImage(image, *download, offset, range.size, std::move(transfer));
	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eTransferWrite |
	                        vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = download->Handle();
	barrier.offset              = offset;
	barrier.size                = range.size;
	m_scheduler.EndRendering();
	m_scheduler.Current().Recorder().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                                               vk::PipelineStageFlagBits::eHost, {}, 0, nullptr,
	                                               1, &barrier, 0, nullptr);
	m_scheduler.DeferPriorityOperation([download, dedicated, range, mapped, offset, guard,
	                                    guest_hash] {
		download->Invalidate(offset, range.size);
		uint64_t current = 0;
		if (guard && (!HashGuestBacking(range.address, range.size, current) ||
		              current != guest_hash)) {
			// The guest wrote (or unmapped) these bytes after the image was evicted: they are
			// newer than its pixels.
			static std::atomic<uint32_t> reported = 0;
			if (reported.fetch_add(1, std::memory_order_relaxed) < 16) {
				LOGF("TextureCache: evicted image write-back dropped, guest=0x%016" PRIx64
				     " size=0x%" PRIx64 ": the guest bytes changed before it landed\n",
				     range.address, range.size);
			}
			return;
		}
		LibKernel::Memory::WriteBacking(range.address, mapped, range.size);
	});
	return true;
}

void TextureCache::InvalidateMemoryFromGPU(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		return;
	}
	std::scoped_lock lock {m_lock};
	for (const auto id: FindImagesInRegion(address, size, true)) {
		auto& image = m_slot_images[id];
		if (image.info.data.address != address) {
			continue;
		}
		if (image.IsGpuModified()) {
			image.ClearGpuModified();
		}
		image.MarkBufferModified();
	}
}

void TextureCache::InvalidateCpuAliases(uint64_t address, uint64_t size) {
	const auto page_begin = Common::AlignDown(address, TRACKER_PAGE_SIZE);
	const auto page_end   = Common::AlignUp(address + size, TRACKER_PAGE_SIZE);
	for (const auto id: FindImagesInRegion(address, size, true)) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr) {
			continue;
		}
		if (owner->Overlaps(address, size)) {
			owner->InvalidateCpuWrite(address, size);
			UntrackImage(id);
			continue;
		}
		const auto image_begin = owner->info.data.address;
		const auto image_end   = owner->info.data.End();
		if (page_end < image_end) {
			UntrackImageHead(id);
		} else if (image_begin < page_begin) {
			UntrackImageTail(id);
		} else {
			MarkAsMaybeDirty(id, *owner);
		}
	}
}

bool TextureCache::IsMeta(uint64_t address) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	return found != m_surface_metas.end();
}

bool TextureCache::IsMetaCleared(uint64_t address, uint32_t slice) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end() || slice >= 32) {
		return false;
	}
	return (found->second.clear_mask & (1u << slice)) != 0;
}

bool TextureCache::ClearMeta(uint64_t address) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end()) {
		return false;
	}
	found->second.clear_mask = UINT32_MAX;
	return true;
}

bool TextureCache::TouchMeta(uint64_t address, uint32_t slice, bool is_clear) {
	std::scoped_lock lock {m_lock};
	const auto       found = m_surface_metas.find(address);
	if (found == m_surface_metas.end() || slice >= 32) {
		return false;
	}
	if (is_clear) {
		found->second.clear_mask |= 1u << slice;
	} else {
		found->second.clear_mask &= ~(1u << slice);
	}
	return true;
}

void TextureCache::UnmapMemory(uint64_t address, uint64_t size) {
	if (!GuestRange {address, size}.Valid()) {
		EXIT("TextureCache: invalid unmap range\n");
	}
	std::scoped_lock lock {m_lock};
	for (auto metadata = m_surface_metas.begin(); metadata != m_surface_metas.end();) {
		const auto base = metadata->first;
		if (base >= address && base < address + size) {
			metadata = m_surface_metas.erase(metadata);
		} else {
			++metadata;
		}
	}
	auto images = FindImagesInRegion(address, size, false);
	for (const auto id: images) {
		auto owner = m_slot_images.try_get(id);
		if (owner == nullptr) {
			continue;
		}
		FreeImage(id);
	}
}

uint64_t TextureCache::LruClock() const noexcept {
	return m_graphics.presented_frames.load(std::memory_order_relaxed) + m_gc_tick / 512;
}

namespace {
// Why the collector does not free: over each TexStats window, candidates it visited and what
// happened to them.
struct GcSkipStats {
	uint64_t visited      = 0;
	uint64_t freed        = 0;
	uint64_t freed_mb     = 0;
	uint64_t gone         = 0; // no longer registered
	uint64_t depth        = 0; // part of a depth/stencil association
	uint64_t gpu_tiled    = 0; // GPU-written tiled: needs a write-back it cannot do
	uint64_t gpu_unpress  = 0; // GPU-written, not under pressure
	uint64_t gpu_download = 0; // GPU-written, write-back failed
	uint64_t young        = 0; // runs where the LRU had nothing old enough
	uint64_t written_back = 0; // GPU-written tiled images written back, then freed
};
GcSkipStats g_gc_skip;
} // namespace

void TextureCache::RunGarbageCollector() {
	std::scoped_lock lock {m_lock};
	m_gc_tick++;
	const uint64_t clock = LruClock();
	if (m_graphics.CanReportMemoryUsage()) {
		// Pressure is judged by this cache's own image bytes, which RegisterImage and
		// UnregisterImage already track. Device-wide usage counts the images this cache has
		// spilled to host memory (and every buffer), so it reads "critical" no matter how
		// small the cache is -- and never falls, so the collector cannot tell that it is the
		// one holding too much. Spider-Man's world then grew to 7.6 GB of images on a 6 GB
		// card, all of it spilled, and every shader access crossed PCIe (3 fps).
		// KYTY_GC_DEVICE_BYTES=1 restores the old policy, as in the buffer cache.
		static const bool device_bytes = std::getenv("KYTY_GC_DEVICE_BYTES") != nullptr;
		if (device_bytes) {
			m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
		}
	}
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}
	// The collector runs once per guest submission -- this title submits hundreds of times a
	// frame -- so a budget counted per run is really that many evictions per frame, and every
	// eviction is an image recreated, synchronized and submitted. Budget the work per frame
	// instead: 7 600 submissions a frame came from a per-run budget of 256.
	if (m_gc_budget_frame != clock) {
		m_gc_budget_frame       = clock;
		m_gc_freed_bytes_frame  = 0;
		m_gc_freed_images_frame = 0;
		m_gc_written_back_bytes_frame = 0;
	}
	const auto collect = [&](bool allow_aggressive) {
		bool pressured  = m_total_used_memory >= m_pressure_gc_memory;
		bool aggressive = allow_aggressive && m_total_used_memory >= m_critical_gc_memory;
		// Ages in frames. This collector runs once per guest submission, and a title can
		// submit a hundred times a frame: an age counted in its ticks evicted textures inside
		// the frame that used them, to recreate them the next (11 000 images a second at the
		// title menu). An image used in the frame being recorded is never a candidate, at any
		// pressure; over budget the collector reaches back only as far as the previous frame.
		const uint64_t age = std::min<uint64_t>(aggressive ? 1 : pressured ? 4 : 16, clock);
		// Under pressure the work is budgeted per frame, in bytes actually freed, and in
		// proportion to the excess, so the cache converges on its threshold instead of either
		// churning (a per-run budget: 7 600 submissions a frame) or growing without bound (a
		// small fixed count, spent on GPU-written render targets that are skipped: 13.5 GB).
		// Only frees count against it. A cap on images per frame bounds the synchronization
		// that recreating them costs.
		constexpr uint64_t MiB             = 1024 * 1024;
		constexpr size_t   MaxFreesInFrame = 1024;
		uint64_t           byte_budget     = 0;
		size_t             deletions       = 10;
		if (pressured || aggressive) {
			const auto threshold = aggressive ? m_critical_gc_memory : m_pressure_gc_memory;
			const auto excess    = m_total_used_memory - threshold;
			byte_budget = aggressive ? std::max<uint64_t>(64 * MiB, excess / 4)
			                         : std::max<uint64_t>(16 * MiB, excess / 8);
			if (m_gc_freed_bytes_frame >= byte_budget ||
			    m_gc_freed_images_frame >= MaxFreesInFrame) {
				return;
			}
			deletions = MaxFreesInFrame - m_gc_freed_images_frame;
		}
		// Skipped candidates (GPU-written tiled images, depth associations) do not use up the
		// budget, so look further down the LRU than the number of frees wanted.
		const size_t         visit_limit = pressured || aggressive ? deletions * 4 : deletions;
		std::vector<ImageId> candidates;
		candidates.reserve(std::min<size_t>(visit_limit, 4096));
		// Deleting depth recursively deletes its stencil association, so finish LRU traversal
		// first.
		m_lru_cache.ForEachItemBelow(clock - age, [&](ImageId id) {
			candidates.push_back(id);
			return candidates.size() >= visit_limit || candidates.size() >= 4096;
		});
		if (candidates.empty() && (pressured || aggressive)) {
			g_gc_skip.young++;
		}
		for (const auto id: candidates) {
			if (deletions == 0) {
				break;
			}
			if (!(pressured || aggressive)) {
				--deletions;
			}
			g_gc_skip.visited++;
			auto owner = m_slot_images.try_get(id);
			if (owner == nullptr || !owner->registered) {
				g_gc_skip.gone++;
				continue;
			}
			if (owner->depth_id) {
				g_gc_skip.depth++;
				continue;
			}
			if (owner->bindless_pinned) {
				continue;
			}
			if (owner->IsGpuModified()) {
				const bool safe = owner->SafeToDownload();
				// A GPU-written tiled image was never evicted: render targets and textures the
				// GPU decompresses are almost all of a 4K title's images, so under pressure the
				// collector found nothing else it could free (Spider-Man: 3 500 of every 4 000
				// candidates) and the cache grew to 13.5 GB until the driver crashed in
				// present. Under pressure such an image is now written back, detiled, like any
				// other, within a per-frame write-back budget; candidates are never ones used in
				// the frame being recorded. KYTY_GC_KEEP_TILED=1 restores the old rule.
				static const bool keep_tiled = std::getenv("KYTY_GC_KEEP_TILED") != nullptr;
				constexpr uint64_t WriteBackPerFrame = 256 * MiB;
				if (safe && owner->info.IsTiled() &&
				    (keep_tiled || !(pressured || aggressive) ||
				     (m_gc_written_back_bytes_frame != 0 &&
				      m_gc_written_back_bytes_frame + owner->info.data.size > WriteBackPerFrame))) {
					g_gc_skip.gpu_tiled++;
					continue;
				}
				if (safe && !pressured) {
					g_gc_skip.gpu_unpress++;
					continue;
				}
				if (safe && !DownloadImageMemory(id, true)) {
					g_gc_skip.gpu_download++;
					continue;
				}
				if (safe && owner->info.IsTiled()) {
					m_gc_written_back_bytes_frame += owner->info.data.size;
					g_gc_skip.written_back++;
				}
			}
			const auto freed = owner->AccountedSize();
			g_gc_skip.freed++;
			g_gc_skip.freed_mb += freed >> 20u;
			FreeImage(id);
			if (pressured || aggressive) {
				--deletions;
				m_gc_freed_images_frame++;
				m_gc_freed_bytes_frame += freed;
				if (m_gc_freed_bytes_frame >= byte_budget) {
					break;
				}
			}
			if (m_total_used_memory < m_critical_gc_memory && aggressive) {
				deletions >>= 2;
				aggressive = false;
			}
			if (m_total_used_memory < m_pressure_gc_memory && pressured) {
				deletions >>= 1;
				pressured = false;
			}
		}
	};
	collect(false);
	if (m_total_used_memory >= m_critical_gc_memory) {
		collect(true);
	}
	if (m_gc_tick % 500 == 0) {
		uint64_t images = 0;
		uint64_t bytes  = 0;
		m_slot_images.ForEach([&](ImageId, const Image& image) {
			if (image.registered) {
				images++;
				bytes += image.AccountedSize();
			}
		});
		LOGF("TexStats: runs=%" PRIu64 " frame=%" PRIu64 " images=%" PRIu64 " image_bytes=%" PRIu64
		     "MB device_used=%" PRIu64 "MB pressure=%" PRIu64 "MB critical=%" PRIu64 "MB\n",
		     m_gc_tick, m_graphics.presented_frames.load(std::memory_order_relaxed), images,
		     bytes >> 20u, m_total_used_memory >> 20u, m_pressure_gc_memory >> 20u,
		     m_critical_gc_memory >> 20u);
		LOGF("TexGc: visited=%" PRIu64 " freed=%" PRIu64 " (%" PRIu64 "MB) written_back=%" PRIu64
		     " gone=%" PRIu64 " depth=%" PRIu64 " gpu_tiled=%" PRIu64 " gpu_unpressured=%" PRIu64
		     " gpu_download_failed=%" PRIu64 " nothing_old_enough=%" PRIu64 "\n",
		     g_gc_skip.visited, g_gc_skip.freed, g_gc_skip.freed_mb, g_gc_skip.written_back,
		     g_gc_skip.gone, g_gc_skip.depth, g_gc_skip.gpu_tiled, g_gc_skip.gpu_unpress,
		     g_gc_skip.gpu_download, g_gc_skip.young);
		g_gc_skip = {};
	}
}

void TextureCache::ProcessDownloadImages() {
	std::scoped_lock lock {m_lock};
	for (const auto id: m_download_images) {
		const auto owner = m_slot_images.try_get(id);
		if (owner != nullptr && owner->registered && owner->IsGpuModified()) {
			(void)DownloadImageMemory(id);
		}
	}
	m_download_images.clear();
}

} // namespace Libs::Graphics
