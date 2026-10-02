#include "common/assert.h"
#include "common/common.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <bit>
#include <cstring>
namespace Libs::Graphics {

CommandBuffer::CommandBuffer(CommandScheduler& scheduler)
    : m_context(scheduler.Context()), m_graphics(scheduler.Graphics()) {}

bool CommandBuffer::IsInvalid() const {
	return m_buffer == nullptr;
}

vk::CommandBuffer CommandBuffer::Handle() const {
	EXIT_IF(IsInvalid());
	return m_buffer;
}

void CommandBuffer::Begin() {
	EXIT_IF(m_rendering || IsInvalid());
	auto buffer = Handle();

	vk::CommandBufferBeginInfo begin_info {};
	begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

	auto result = buffer.begin(&begin_info);

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	// New Vulkan command buffer => previous vkCmdBind*/vkCmdSet* state is gone.
	m_cached_vertex_binding  = {};
	m_cached_index_binding   = {};
	m_cached_viewport        = {};
	m_cached_scissor         = {};
	m_cached_depth_bias      = {};
	m_cached_pipeline        = {};
	m_cached_descriptor_sets = {};
}

bool CommandBuffer::BindVertexBuffersCached(uint32_t first_binding, uint32_t count,
                                            const vk::Buffer* buffers,
                                            const vk::DeviceSize* offsets,
                                            const vk::DeviceSize* sizes) {
	auto& cached = m_cached_vertex_binding;
	if (cached.valid && cached.first_binding == first_binding && cached.count == count &&
	    count <= CachedVertexBinding::MaxBindings) {
		bool same = true;
		for (uint32_t i = 0; i < count; i++) {
			if (cached.buffers[i] != buffers[i] || cached.offsets[i] != offsets[i] ||
			    cached.sizes[i] != sizes[i]) {
				same = false;
				break;
			}
		}
		if (same) {
			return false;
		}
	}
	cached.first_binding = first_binding;
	cached.count         = count <= CachedVertexBinding::MaxBindings
	                           ? count
	                           : CachedVertexBinding::MaxBindings;
	for (uint32_t i = 0; i < cached.count; i++) {
		cached.buffers[i] = buffers[i];
		cached.offsets[i] = offsets[i];
		cached.sizes[i]   = sizes[i];
	}
	cached.valid = true;
	return true;
}

bool CommandBuffer::BindIndexBufferCached(vk::Buffer buffer, vk::DeviceSize offset,
                                          vk::IndexType index_type) {
	auto& cached = m_cached_index_binding;
	if (cached.valid && cached.buffer == buffer && cached.offset == offset &&
	    cached.index_type == index_type) {
		return false;
	}
	cached = {.buffer = buffer, .offset = offset, .index_type = index_type, .valid = true};
	return true;
}

namespace {

[[nodiscard]] bool ViewportsEqual(const vk::Viewport& a, const vk::Viewport& b) {
	// Bit-exact compare: NaN payloads must not compare equal across draws.
	return std::memcmp(&a, &b, sizeof(vk::Viewport)) == 0;
}

[[nodiscard]] bool ScissorsEqual(const vk::Rect2D& a, const vk::Rect2D& b) {
	return std::memcmp(&a, &b, sizeof(vk::Rect2D)) == 0;
}

} // namespace

bool CommandBuffer::SetViewportWithCountCached(uint32_t count,
                                               const vk::Viewport* viewports) const {
	auto& cached = m_cached_viewport;
	if (cached.valid && cached.count == count && count <= MaxCachedViewports) {
		bool same = true;
		for (uint32_t i = 0; i < count; i++) {
			if (!ViewportsEqual(cached.viewports[i], viewports[i])) {
				same = false;
				break;
			}
		}
		if (same) {
			return false;
		}
	}
	cached.count = count <= MaxCachedViewports ? count : MaxCachedViewports;
	for (uint32_t i = 0; i < cached.count; i++) {
		cached.viewports[i] = viewports[i];
	}
	cached.valid = true;
	return true;
}

bool CommandBuffer::SetScissorWithCountCached(uint32_t count,
                                              const vk::Rect2D* scissors) const {
	auto& cached = m_cached_scissor;
	if (cached.valid && cached.count == count && count <= MaxCachedViewports) {
		bool same = true;
		for (uint32_t i = 0; i < count; i++) {
			if (!ScissorsEqual(cached.scissors[i], scissors[i])) {
				same = false;
				break;
			}
		}
		if (same) {
			return false;
		}
	}
	cached.count = count <= MaxCachedViewports ? count : MaxCachedViewports;
	for (uint32_t i = 0; i < cached.count; i++) {
		cached.scissors[i] = scissors[i];
	}
	cached.valid = true;
	return true;
}

bool CommandBuffer::SetDepthBiasCached(bool enable, float constant_factor, float clamp,
                                       float slope_factor) const {
	auto& cached = m_cached_depth_bias;
	if (cached.valid && cached.enable == enable &&
	    std::memcmp(&cached.constant_factor, &constant_factor, sizeof(float)) == 0 &&
	    std::memcmp(&cached.clamp, &clamp, sizeof(float)) == 0 &&
	    std::memcmp(&cached.slope_factor, &slope_factor, sizeof(float)) == 0) {
		return false;
	}
	cached.enable          = enable;
	cached.constant_factor = constant_factor;
	cached.clamp           = clamp;
	cached.slope_factor    = slope_factor;
	cached.valid           = true;
	return true;
}

bool CommandBuffer::BindPipelineCached(vk::PipelineBindPoint bind_point,
                                       vk::Pipeline pipeline) const {
	auto& cached = m_cached_pipeline;
	vk::Pipeline& slot =
	    bind_point == vk::PipelineBindPoint::eGraphics ? cached.graphics_pipeline
	                                                   : cached.compute_pipeline;
	if (slot == pipeline) {
		return false;
	}
	slot = pipeline;
	return true;
}

bool CommandBuffer::BindDescriptorSetsCached(vk::PipelineBindPoint bind_point,
                                             vk::PipelineLayout layout, uint32_t first_set,
                                             const vk::DescriptorSet* sets,
                                             uint32_t set_count) const {
	// Only the (layout, set0) and (layout, set1) pairs CommitBindings emits are
	// cached. A layout change invalidates both slots; push descriptors bypass
	// vkCmdBindDescriptorSets entirely and invalidate via
	// InvalidatePushDescriptorCache.
	if (set_count == 0 || sets == nullptr) {
		return false;
	}
	auto& binding = bind_point == vk::PipelineBindPoint::eGraphics
	                      ? m_cached_descriptor_sets.graphics
	                      : m_cached_descriptor_sets.compute;
	if (binding.layout != layout) {
		binding.layout  = layout;
		binding.set0    = nullptr;
		binding.set1    = nullptr;
		binding.has_set0 = false;
		binding.has_set1 = false;
	}
	bool same = true;
	for (uint32_t i = 0; i < set_count; i++) {
		const uint32_t set_index = first_set + i;
		if (set_index == 0) {
			if (!binding.has_set0 || binding.set0 != sets[i]) {
				same = false;
				break;
			}
		} else if (set_index == 1) {
			if (!binding.has_set1 || binding.set1 != sets[i]) {
				same = false;
				break;
			}
		} else {
			same = false;
			break;
		}
	}
	if (same) {
		return false;
	}
	for (uint32_t i = 0; i < set_count; i++) {
		const uint32_t set_index = first_set + i;
		if (set_index == 0) {
			binding.set0    = sets[i];
			binding.has_set0 = true;
		} else if (set_index == 1) {
			binding.set1    = sets[i];
			binding.has_set1 = true;
		}
	}
	return true;
}

void CommandBuffer::InvalidatePushDescriptorCache(vk::PipelineBindPoint bind_point) const {
	// Push descriptors have no set handle to compare, so any push on this bind
	// point invalidates the cached set 0 for its layout.
	auto& binding = bind_point == vk::PipelineBindPoint::eGraphics
	                      ? m_cached_descriptor_sets.graphics
	                      : m_cached_descriptor_sets.compute;
	binding.set0    = nullptr;
	binding.has_set0 = false;
}

void CommandBuffer::End() const {
	EndRendering();
	auto buffer = Handle();

	auto result = buffer.end();

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::SetDebugInfo(uint32_t op, uint64_t submit_id, uint32_t arg0, uint32_t arg1,
                                 uint32_t arg2, uint32_t arg3, uint64_t arg4) {
	m_debug_op        = op;
	m_debug_submit_id = submit_id;
	m_debug_arg0      = arg0;
	m_debug_arg1      = arg1;
	m_debug_arg2      = arg2;
	m_debug_arg3      = arg3;
	m_debug_arg4      = arg4;
	if (m_graphics.diagnostic_checkpoints_enabled && m_buffer) {
		const auto* marker = RecordDiagnosticCheckpoint({.op        = op,
		                                                 .submit_id = submit_id,
		                                                 .arg0      = arg0,
		                                                 .arg1      = arg1,
		                                                 .arg2      = arg2,
		                                                 .arg3      = arg3,
		                                                 .arg4      = arg4});
		if (marker != nullptr) {
			Handle().setCheckpointNV(marker);
		}
	}
}

void CommandBuffer::BeginRendering(const RenderState& state) const {
	if (m_rendering && m_render_state == state) {
		return;
	}
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.num_color_attachments > RENDER_COLOR_ATTACHMENTS_MAX);
	EndRendering();

	std::array<vk::RenderingAttachmentInfo, RENDER_COLOR_ATTACHMENTS_MAX> colors {};
	for (uint32_t i = 0; i < state.num_color_attachments; i++) {
		const auto& attachment = state.color_attachments[i];
		colors[i].imageView    = attachment.image_view;
		colors[i].imageLayout  = attachment.image_layout;
		colors[i].loadOp =
		    attachment.is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
		colors[i].storeOp                 = vk::AttachmentStoreOp::eStore;
		colors[i].clearValue.color.uint32 = attachment.clear_value;
	}

	const auto&                 depth_stencil = state.depth_stencil_attachment;
	vk::RenderingAttachmentInfo depth {};
	depth.imageView   = depth_stencil.image_view;
	depth.imageLayout = depth_stencil.image_layout;
	depth.loadOp =
	    depth_stencil.depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	depth.storeOp                       = vk::AttachmentStoreOp::eStore;
	depth.clearValue.depthStencil.depth = std::bit_cast<float>(depth_stencil.clear_value[0]);

	vk::RenderingAttachmentInfo stencil {};
	stencil.imageView   = depth_stencil.image_view;
	stencil.imageLayout = depth_stencil.image_layout;
	stencil.loadOp =
	    depth_stencil.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	stencil.storeOp                         = vk::AttachmentStoreOp::eStore;
	stencil.clearValue.depthStencil.stencil = depth_stencil.clear_value[1];

	vk::RenderingInfo rendering {};
	rendering.renderArea.extent    = {state.width, state.height};
	rendering.layerCount           = state.num_layers;
	rendering.colorAttachmentCount = state.num_color_attachments;
	rendering.pColorAttachments    = colors.data();
	rendering.pDepthAttachment     = depth_stencil.has_depth ? &depth : nullptr;
	rendering.pStencilAttachment   = depth_stencil.has_stencil ? &stencil : nullptr;
	Handle().beginRendering(rendering);
	m_render_state = state;
	m_rendering    = true;
}

void CommandBuffer::EndRendering() const {
	if (!m_rendering) {
		return;
	}
	Handle().endRendering();
	m_rendering    = false;
	m_render_state = {};
}

} // namespace Libs::Graphics
