#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/image/blitHelper.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/image/tiler.h"

#include <functional>
#include <map>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class Buffer;
class BufferCache;
class CommandBuffer;
class CommandScheduler;
class RenderExecutor;
struct TextureCacheTestAccess;

class TextureCache {
public:
	// Called before a bindless-pinned image is unregistered.
	std::function<void(ImageId)> on_bindless_unregister;
	// Called right before an image is destroyed; true keeps it one more GPU tick (a bindless slot
	// still held one of its views and was repointed).
	std::function<bool(const Image&)> on_image_release;
	enum class BindingType : uint8_t { Texture, Storage, RenderTarget, DepthTarget, VideoOut };

	// Step 3b regression fix: fill fragment module depends on the attachment class.
	// Public so the anonymous-namespace classifier in textureCache.cpp can name it.
	enum class ClearFillClass : uint8_t { Float, Uint, Sint };
	// TryMaterializeColorClearGpu* attempt records one of these; the counters print in
	// [FRAME_TELEMETRY_CLEARFB].
	enum class ClearFbReason : uint8_t {
		NoCondRendering,
		Resources,
		Unsupported,
		NoCandidates,
		TooManyCandidates,
		MetaRange,
		DepthOrMsaa,
		IntFormat,
		PipelineFail,
	};
	// Step 3b: outcome of the GPU-predicate attempt. SkippedNoCand means no metadata code
	// decodes for this image, so the old path is a proven no-op and the wait is skipped
	// without a readback.
	enum class ClearPredicateResult : uint8_t { Applied, SyncFallback, SkippedNoCand };

	struct ImageDesc {
		ImageInfo     info;
		ImageViewInfo view_info;
		BindingType   type = BindingType::Texture;
	};

	TextureCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	             BufferCache& buffer_cache);
	~TextureCache();
	KYTY_CLASS_NO_COPY(TextureCache);

	[[nodiscard]] ImageId       FindImage(ImageDesc& desc, bool exact_format = false);
	void                        UpdateImage(ImageId id);
	[[nodiscard]] ImageId       FindImageFromRange(uint64_t address, uint64_t size,
	                                               bool ensure_valid = true);
	[[nodiscard]] vk::ImageView FindTexture(ImageId id, const ImageDesc& desc);
	[[nodiscard]] vk::ImageView FindRenderTarget(ImageId id, const ImageDesc& desc);
	[[nodiscard]] vk::ImageView FindDepthTarget(ImageId id, const ImageDesc& desc);
	[[nodiscard]] Image&        GetImage(ImageId id) {
		auto& image = m_slot_images[id];
		TouchImage(image);
		return image;
	}
	void MarkGpuWritten(ImageId id);

	[[nodiscard]] bool ClearImageFromBuffer(CommandBuffer& command, uint64_t address, uint64_t size,
	                                        uint32_t packed_clear);
	void               InvalidateMemory(uint64_t address, uint64_t size);
	void               InvalidateMemoryFromGPU(uint64_t address, uint64_t size);

	[[nodiscard]] bool IsMeta(uint64_t address);
	[[nodiscard]] bool IsMetaCleared(uint64_t address, uint32_t slice);
	[[nodiscard]] bool ClearMeta(uint64_t address);
	[[nodiscard]] bool TouchMeta(uint64_t address, uint32_t slice, bool is_clear);

	void UnmapMemory(uint64_t address, uint64_t size);
	void ProcessDownloadImages();
	void RunGarbageCollector();

private:
	enum class TransferDirection { Upload, Download };
	struct TextureTransfer;
	struct ImageDownload;

	struct MetaDataInfo {
		enum class Type : uint8_t { CMask, FMask, HTile };

		Type     type;
		uint32_t clear_mask = UINT32_MAX;
	};

	struct OverlapResult {
		ImageId image;
		int32_t mip   = -1;
		int32_t layer = -1;
	};

	using ImageIds       = InlinePageOwnerList<ImageId, 16>;
	using ImagePageTable = MultiLevelPageTable<ImageIds, 20, 44, 14>;

	// Callers have validated the nonempty 44-bit range with TryGetPageRange.
	template <typename Func>
	static void ForEachPage(uint64_t address, size_t size, Func&& func) {
		using FuncReturn = typename std::invoke_result<Func, uint64_t>::type;
		static constexpr bool RETURNS_BOOL = std::is_same_v<FuncReturn, bool>;
		const uint64_t page_end = (address + size - 1) >> ImagePageTable::kPageBits;
		for (uint64_t page = address >> ImagePageTable::kPageBits; page <= page_end; ++page) {
			if constexpr (RETURNS_BOOL) {
				if (func(page)) {
					break;
				}
			} else {
				func(page);
			}
		}
	}

	[[nodiscard]] ImageId     InsertImage(const ImageInfo& info);
	[[nodiscard]] ImageId     GetNullImage(const ImageDesc& desc);
	void                      RegisterImage(ImageId id);
	void                      UnregisterImage(ImageId id);
	void                      DeleteImage(ImageId id);
	void                      ReleaseImage(ImageId id);
	void                      FreeImage(ImageId id);
	void                      TouchImage(Image& image);
	void                      TrackImage(ImageId id);
	void                      TrackImageHead(ImageId id);
	void                      TrackImageTail(ImageId id);
	void                      UntrackImage(ImageId id);
	void                      UntrackImageHead(ImageId id);
	void                      UntrackImageTail(ImageId id);
	void                      MarkAsMaybeDirty(ImageId id, Image& image);
	void                      TrackImageDownload(ImageId id, Image& image);
	[[nodiscard]] static bool SameBacking(const ImageInfo& cached, const ImageInfo& requested,
	                                      bool exact_format);
	[[nodiscard]] static BindingType UploadBinding(const Image& image);

	// Caller holds m_lock; it also serializes the per-image query epoch.
	[[nodiscard]] ImageIds      FindImagesInRegion(uint64_t address, uint64_t size,
	                                               bool page_overlap) const;
	[[nodiscard]] OverlapResult ResolveOverlap(const ImageInfo& requested, BindingType binding,
	                                           ImageId cached, ImageId merged);
	[[nodiscard]] ImageId       ResolveDepthOverlap(const ImageInfo& requested, BindingType binding,
	                                                ImageId cached);
	[[nodiscard]] ImageId       ExpandImage(const ImageInfo& info, ImageId source);
	void                        RefreshImage(ImageId id);
	void                        MaterializeColorClear(ImageId id, const ImageDesc& desc,
	                                                uint32_t metadata_base_layer);
	// Step 3: GPU predicate for the color-clear metadata wait. Checks the decodable codes
	// on the GPU and clears the image conditionally.
	// Step 3b: on SyncFallback, out_reason (when non-null) receives the failing check so
	// the caller can log it; every false return from the Dispatch/Draw halves assigns it.
	[[nodiscard]] ClearPredicateResult
	TryMaterializeColorClearGpu(ImageId id, const ImageDesc& desc, uint32_t first, uint32_t count,
	                           uint64_t slice_size, uint64_t range_address, uint64_t range_size,
	                           ClearFbReason* out_reason = nullptr);
	[[nodiscard]] bool ClearPredicateEnsureResources();
	// Step 3 dispatch half: records the check compute and the conditional clears.
	// Candidates are (code byte, clear color) pairs decodable for this image.
	struct ClearPredicateCandidate {
		uint8_t             code;
		vk::ClearColorValue clear;
	};
	// Step 3b: cached decodable (code, clear color) pairs per clear-relevant desc fields,
	// so the 256-code DecodeColorClear loop doesn't run every frame.
	struct ClearCandidatesKey {
		ImageMetadataKind kind;
		vk::Format        format;
		BindingType       type;
		uint32_t          clear_word;
		bool              clear_register_valid;
		bool              dcc_alpha_msb;
		auto operator<=>(const ClearCandidatesKey&) const = default;
	};
	struct ClearCandidatesResult {
		bool                                 too_many = false;
		std::vector<ClearPredicateCandidate> candidates;
	};
	ClearCandidatesResult LookupClearCandidates(const ImageDesc& desc);
	[[nodiscard]] ClearFillClass ClearFillClassFor(vk::Format format);
	[[nodiscard]] bool TryMaterializeColorClearGpuDispatch(
	    ImageId id, const ImageDesc& desc, uint32_t count, uint64_t slice_size,
	    uint64_t range_address, uint64_t range_size, const ClearPredicateCandidate* candidates,
	    size_t candidate_count, ClearFbReason* out_reason);
	// Step 3 draw half: one predicated fullscreen-triangle draw per (slice, candidate).
	// Conditional rendering gates draws, unlike transfer clears.
	[[nodiscard]] bool TryMaterializeColorClearGpuDraw(ImageId id, const ImageDesc& desc,
	                                                  uint32_t count,
	                                                  const ClearPredicateCandidate* candidates,
	                                                  size_t candidate_count, ClearFbReason* out_reason);
	[[nodiscard]] vk::Pipeline ClearFillPipelineFor(vk::Format format);
	void                        InitializeImage(ImageId id);
	[[nodiscard]] TextureTransfer
	BuildTextureTransfer(const Image& image, BindingType binding, TransferDirection direction) const;
	[[nodiscard]] ImageDownload BuildDownload(const Image& image) const;
	void UploadImage(Image& image, Buffer& source, uint64_t source_offset);
	void DownloadImage(Image& image, Buffer& destination, uint64_t destination_offset,
	                       uint64_t destination_size, ImageDownload transfer);
	void DownloadDepth(Image& image, Buffer& destination, uint64_t destination_offset);
	void CommitGpuWrite(Image& image);
	// Caller holds m_lock. Volume layer ranges select depth slices.
	void ClearImage(CommandBuffer& command, ImageId id, vk::Format format,
	                const vk::ImageSubresourceRange& range, const vk::ClearValue& clear);
	void PrepareImageCopy(Image& image);
	void RefreshCopySource(ImageId id);
	[[nodiscard]] bool CopyD16(Image& destination, Image& source);
	void               CopyImage(ImageId destination, ImageId source);
	[[nodiscard]] ImageId AssociateStencil(ImageId depth, GuestRange stencil);
	void CopyImageMip(ImageId destination, ImageId source, uint32_t mip, uint32_t layer);
	void ValidateImageDesc(const ImageDesc& desc) const;

	void               InvalidateCpuAliases(uint64_t address, uint64_t size);
	// evicting: the collector writes the image back before freeing it; the write is dropped if
	// the guest bytes change before it lands (KYTY_EVICT_WRITEBACK_GUARD).
	[[nodiscard]] bool DownloadImageMemory(ImageId id, bool evicting = false);

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	TrackingSpinLock                                  m_lock;
	PageManager&                                      m_page_manager;
	BlitHelper                                        m_blit_helper;
	TileManager                                       m_tiler;
	BufferCache&                                      m_buffer_cache;
	// Step 3: persistent GPU resources for the color-clear predicate check shader.
	vk::DescriptorSetLayout m_clear_predicate_desc_layout = nullptr;
	vk::PipelineLayout      m_clear_predicate_layout      = nullptr;
	vk::Pipeline            m_clear_predicate_pipeline    = nullptr;
	// Step 3 fill path: fullscreen-triangle pipeline writing the clear color. One
	// pipeline per color format; cached by format value.
	vk::DescriptorSetLayout m_clear_fill_desc_layout = nullptr;
	vk::PipelineLayout      m_clear_fill_layout      = nullptr;
	vk::ShaderModule        m_clear_fill_vert        = nullptr;
	vk::ShaderModule        m_clear_fill_frag        = nullptr;
	struct ClearFillPipeline {
		vk::Format   format = vk::Format::eUndefined;
		vk::Pipeline handle = nullptr;
	};
	std::vector<ClearFillPipeline> m_clear_fill_pipelines;
	std::unique_ptr<Buffer> m_clear_predicate_buffer;
	vk::ShaderModule        m_clear_fill_frag_uint = nullptr;
	vk::ShaderModule        m_clear_fill_frag_sint = nullptr;
	// Step 3b: decodable clear-candidate pairs per clear-relevant desc fields. Bounded:
	// cleared once it grows past 256 entries.
	std::map<ClearCandidatesKey, ClearCandidatesResult> m_clear_candidates_cache;
	Common::SlotVector<Image>                         m_slot_images;
	ImagePageTable                                    m_image_page_table;
	std::map<std::pair<vk::Format, Prospero::ImageType>, ImageId> m_null_images;
	Common::LeastRecentlyUsedCache<ImageId, uint64_t> m_lru_cache;
	std::unordered_set<ImageId>                       m_download_images;
	std::map<uint64_t, MetaDataInfo>                  m_surface_metas;
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t                                          m_trigger_gc_memory  = 0;
	uint64_t                                          m_pressure_gc_memory = 1536ull * 1024 * 1024;
	uint64_t         m_critical_gc_memory     = 3ull * 1024 * 1024 * 1024;
	uint64_t         m_gc_tick                = 0;
	// Evictions are budgeted per frame, not per collector run: see RunGarbageCollector.
	uint64_t         m_gc_budget_frame        = 0;
	uint64_t         m_gc_freed_bytes_frame   = 0;
	size_t           m_gc_freed_images_frame  = 0;
	uint64_t         m_gc_written_back_bytes_frame = 0;
	// The LRU clock: presented frames, advanced by the collector's own ticks as well so a
	// stretch without presents (a loading screen) still ages its entries.
	[[nodiscard]] uint64_t LruClock() const noexcept;
	mutable uint32_t m_image_query_epoch      = 0;
	bool             m_readback_linear_images = false;

	friend struct TextureCacheTestAccess;
	friend class BufferCache;
	friend class RenderExecutor;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_
