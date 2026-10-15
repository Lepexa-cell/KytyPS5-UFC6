#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDRECORDER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDRECORDER_H_

#include "common/assert.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <cstdint>
#include <cstring>

namespace Libs::Graphics {

// The Vulkan commands of the hot recording paths, with vk::CommandBuffer's names and arguments
// (ported from IDXTRI's wolverine-perf branch, d9958e28 + ad1e1396).
// Bound to a raw command buffer it records directly; bound to a threaded scheduler
// (KYTY_RECORD_THREAD) it copies the arguments, and every array they point to, into the
// scheduler's command chunk and queues the call for the recording thread. Nothing the caller
// passes by pointer is used after the method returns.
class CommandRecorder {
public:
	// NOLINTNEXTLINE(google-explicit-constructor): raw command buffers stay usable as recorders.
	CommandRecorder(vk::CommandBuffer direct): m_direct(direct) {}
	CommandRecorder(CommandScheduler& scheduler, vk::CommandBuffer direct)
	    : m_scheduler(&scheduler), m_direct(direct) {}

	[[nodiscard]] bool Threaded() const noexcept { return m_direct == nullptr; }
	[[nodiscard]] bool operator==(std::nullptr_t) const noexcept {
		return m_direct == nullptr && m_scheduler == nullptr;
	}
	// Direct-mode recorders still compare against the scheduler's shadow: CommandBuffer
	// passes its scheduler even when it hands out the raw command buffer, so dedup works
	// identically in both modes (see context.cpp Recorder()).
	void BindScheduler(CommandScheduler& scheduler) { m_scheduler = &scheduler; }

	void bindPipeline(vk::PipelineBindPoint bind_point, vk::Pipeline pipeline) const {
		if (m_scheduler != nullptr && RecordPipelineCacheEnabled()) {
			auto&                          cache = m_scheduler->RecordCache();
			vk::Pipeline&                  known_pipeline = bind_point == vk::PipelineBindPoint::eCompute
				                                                ? cache.compute_pipeline
				                                                : cache.graphics_pipeline;
			bool&                          known = bind_point == vk::PipelineBindPoint::eCompute
				                                       ? cache.compute_pipeline_known
				                                       : cache.graphics_pipeline_known;
			// A graphics bind never touches the compute entry and vice versa: each half of
			// the pair is compared only against its own bind point.
			if (known && known_pipeline == pipeline) {
				GetFrameTelemetry().record_pipeline_skipped++;
				return;
			}
			known_pipeline = pipeline;
			known          = true;
			GetFrameTelemetry().record_pipeline_emitted++;
		}
		Run([=](vk::CommandBuffer command) { command.bindPipeline(bind_point, pipeline); });
	}
	void draw(uint32_t vertices, uint32_t instances, uint32_t first_vertex,
	          uint32_t first_instance) const {
		Run([=](vk::CommandBuffer command) {
			command.draw(vertices, instances, first_vertex, first_instance);
		});
	}
	void drawIndexed(uint32_t indices, uint32_t instances, uint32_t first_index, int32_t offset,
	                 uint32_t first_instance) const {
		Run([=](vk::CommandBuffer command) {
			command.drawIndexed(indices, instances, first_index, offset, first_instance);
		});
	}
	void drawIndirect(vk::Buffer buffer, vk::DeviceSize offset, uint32_t count,
	                  uint32_t stride) const {
		Run([=](vk::CommandBuffer command) {
			command.drawIndirect(buffer, offset, count, stride);
		});
	}
	void drawIndexedIndirect(vk::Buffer buffer, vk::DeviceSize offset, uint32_t count,
	                         uint32_t stride) const {
		Run([=](vk::CommandBuffer command) {
			command.drawIndexedIndirect(buffer, offset, count, stride);
		});
	}
	void drawMeshTasksEXT(uint32_t x, uint32_t y, uint32_t z) const {
		Run([=](vk::CommandBuffer command) { command.drawMeshTasksEXT(x, y, z); });
	}
	void drawMeshTasksIndirectEXT(vk::Buffer buffer, vk::DeviceSize offset, uint32_t count,
	                              uint32_t stride) const {
		Run([=](vk::CommandBuffer command) {
			command.drawMeshTasksIndirectEXT(buffer, offset, count, stride);
		});
	}
	void dispatch(uint32_t x, uint32_t y, uint32_t z) const {
		Run([=](vk::CommandBuffer command) { command.dispatch(x, y, z); });
	}
	void dispatchIndirect(vk::Buffer buffer, vk::DeviceSize offset) const {
		Run([=](vk::CommandBuffer command) { command.dispatchIndirect(buffer, offset); });
	}
	void bindIndexBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::IndexType type) const {
		Run([=](vk::CommandBuffer command) { command.bindIndexBuffer(buffer, offset, type); });
	}
	void bindVertexBuffers2(uint32_t first, uint32_t count, const vk::Buffer* buffers,
	                        const vk::DeviceSize* offsets, const vk::DeviceSize* sizes,
	                        const vk::DeviceSize* strides) const {
		if (!Threaded()) {
			m_direct.bindVertexBuffers2(first, count, buffers, offsets, sizes, strides);
			return;
		}
		const auto* b = m_scheduler->Stash(buffers, count);
		const auto* o = m_scheduler->Stash(offsets, count);
		const auto* s = m_scheduler->Stash(sizes, sizes != nullptr ? count : 0);
		const auto* d = m_scheduler->Stash(strides, strides != nullptr ? count : 0);
		m_scheduler->Record([=](vk::CommandBuffer command) {
			command.bindVertexBuffers2(first, count, b, o, s, d);
		});
	}
	void pushConstants(vk::PipelineLayout layout, vk::ShaderStageFlags stages, uint32_t offset,
	                   uint32_t size, const void* values) const {
		if (!Threaded()) {
			m_direct.pushConstants(layout, stages, offset, size, values);
			return;
		}
		const auto* data = m_scheduler->Stash(static_cast<const std::byte*>(values), size);
		m_scheduler->Record([=](vk::CommandBuffer command) {
			command.pushConstants(layout, stages, offset, size, data);
		});
	}
	void bindDescriptorSets(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
	                        uint32_t first, uint32_t count, const vk::DescriptorSet* sets,
	                        uint32_t dynamic_count, const uint32_t* dynamic_offsets) const {
		if (!Threaded()) {
			m_direct.bindDescriptorSets(bind_point, layout, first, count, sets, dynamic_count,
			                            dynamic_offsets);
			return;
		}
		const auto* s = m_scheduler->Stash(sets, count);
		const auto* d = m_scheduler->Stash(dynamic_offsets, dynamic_count);
		m_scheduler->Record([=](vk::CommandBuffer command) {
			command.bindDescriptorSets(bind_point, layout, first, count, s, dynamic_count, d);
		});
	}
	void pushDescriptorSetKHR(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
	                          uint32_t set, uint32_t count,
	                          const vk::WriteDescriptorSet* writes) const {
		if (!Threaded()) {
			m_direct.pushDescriptorSetKHR(bind_point, layout, set, count, writes);
			return;
		}
		const auto* copy = StashWrites(writes, count);
		m_scheduler->Record([=](vk::CommandBuffer command) {
			command.pushDescriptorSetKHR(bind_point, layout, set, count, copy);
		});
	}
	// vkUpdateDescriptorSets of a set allocated for this command buffer, then its bind: both move
	// to the recording thread (the update is driver work too). The set must not be in use yet.
	void updateAndBindDescriptorSet(vk::Device device, vk::PipelineBindPoint bind_point,
	                                vk::PipelineLayout layout, vk::DescriptorSet set,
	                                uint32_t count, const vk::WriteDescriptorSet* writes) const {
		if (!Threaded()) {
			device.updateDescriptorSets(count, writes, 0, nullptr);
			m_direct.bindDescriptorSets(bind_point, layout, 0, 1, &set, 0, nullptr);
			return;
		}
		const auto* copy = StashWrites(writes, count);
		m_scheduler->Record([=](vk::CommandBuffer command) {
			device.updateDescriptorSets(count, copy, 0, nullptr);
			command.bindDescriptorSets(bind_point, layout, 0, 1, &set, 0, nullptr);
		});
	}
	void pipelineBarrier(vk::PipelineStageFlags source, vk::PipelineStageFlags destination,
	                     vk::DependencyFlags flags, uint32_t memory_count,
	                     const vk::MemoryBarrier* memory, uint32_t buffer_count,
	                     const vk::BufferMemoryBarrier* buffers, uint32_t image_count,
	                     const vk::ImageMemoryBarrier* images) const {
		if (!Threaded()) {
			m_direct.pipelineBarrier(source, destination, flags, memory_count, memory, buffer_count,
			                         buffers, image_count, images);
			return;
		}
		RequireNoNext(memory, memory_count);
		RequireNoNext(buffers, buffer_count);
		RequireNoNext(images, image_count);
		const auto* m = m_scheduler->Stash(memory, memory_count);
		const auto* b = m_scheduler->Stash(buffers, buffer_count);
		const auto* i = m_scheduler->Stash(images, image_count);
		m_scheduler->Record([=](vk::CommandBuffer command) {
			command.pipelineBarrier(source, destination, flags, memory_count, m, buffer_count, b,
			                        image_count, i);
		});
	}
	void pipelineBarrier2(const vk::DependencyInfo& dependency) const {
		if (!Threaded()) {
			m_direct.pipelineBarrier2(dependency);
			return;
		}
		EXIT_IF(dependency.pNext != nullptr);
		RequireNoNext(dependency.pMemoryBarriers, dependency.memoryBarrierCount);
		RequireNoNext(dependency.pBufferMemoryBarriers, dependency.bufferMemoryBarrierCount);
		RequireNoNext(dependency.pImageMemoryBarriers, dependency.imageMemoryBarrierCount);
		auto copy = dependency;
		copy.pMemoryBarriers =
		    m_scheduler->Stash(dependency.pMemoryBarriers, dependency.memoryBarrierCount);
		copy.pBufferMemoryBarriers = m_scheduler->Stash(dependency.pBufferMemoryBarriers,
		                                                dependency.bufferMemoryBarrierCount);
		copy.pImageMemoryBarriers =
		    m_scheduler->Stash(dependency.pImageMemoryBarriers, dependency.imageMemoryBarrierCount);
		m_scheduler->Record([copy](vk::CommandBuffer command) { command.pipelineBarrier2(copy); });
	}
	void copyBuffer(vk::Buffer source, vk::Buffer destination, uint32_t count,
	                const vk::BufferCopy* regions) const {
		if (!Threaded()) {
			m_direct.copyBuffer(source, destination, count, regions);
			return;
		}
		const auto* r = m_scheduler->Stash(regions, count);
		m_scheduler->Record(
		    [=](vk::CommandBuffer command) { command.copyBuffer(source, destination, count, r); });
	}
	void copyBuffer(vk::Buffer source, vk::Buffer destination, const vk::BufferCopy& region) const {
		copyBuffer(source, destination, 1, &region);
	}
	void updateBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size,
	                  const void* data) const {
		if (!Threaded()) {
			m_direct.updateBuffer(buffer, offset, size, data);
			return;
		}
		const auto* copy = m_scheduler->Stash(static_cast<const std::byte*>(data), size);
		m_scheduler->Record([=](vk::CommandBuffer command) {
			command.updateBuffer(buffer, offset, size, copy);
		});
	}
	void writeTimestamp(vk::PipelineStageFlagBits stage, vk::QueryPool pool, uint32_t query) const {
		Run([=](vk::CommandBuffer command) { command.writeTimestamp(stage, pool, query); });
	}
	void fillBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size,
	                uint32_t value) const {
		Run([=](vk::CommandBuffer command) { command.fillBuffer(buffer, offset, size, value); });
	}
	void copyBufferToImage(vk::Buffer source, vk::Image destination, vk::ImageLayout layout,
	                       uint32_t count, const vk::BufferImageCopy* regions) const {
		if (!Threaded()) {
			m_direct.copyBufferToImage(source, destination, layout, count, regions);
			return;
		}
		const auto* r = m_scheduler->Stash(regions, count);
		m_scheduler->Record([=](vk::CommandBuffer command) {
			command.copyBufferToImage(source, destination, layout, count, r);
		});
	}
	void copyBufferToImage(vk::Buffer source, vk::Image destination, vk::ImageLayout layout,
	                       const vk::BufferImageCopy& region) const {
		copyBufferToImage(source, destination, layout, 1, &region);
	}
	void copyImageToBuffer(vk::Image source, vk::ImageLayout layout, vk::Buffer destination,
	                       uint32_t count, const vk::BufferImageCopy* regions) const {
		if (!Threaded()) {
			m_direct.copyImageToBuffer(source, layout, destination, count, regions);
			return;
		}
		const auto* r = m_scheduler->Stash(regions, count);
		m_scheduler->Record([=](vk::CommandBuffer command) {
			command.copyImageToBuffer(source, layout, destination, count, r);
		});
	}
	void copyImageToBuffer(vk::Image source, vk::ImageLayout layout, vk::Buffer destination,
	                       const vk::BufferImageCopy& region) const {
		copyImageToBuffer(source, layout, destination, 1, &region);
	}
	void copyImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	               vk::ImageLayout destination_layout, uint32_t count,
	               const vk::ImageCopy* regions) const {
		if (!Threaded()) {
			m_direct.copyImage(source, source_layout, destination, destination_layout, count,
			                   regions);
			return;
		}
		const auto* r = m_scheduler->Stash(regions, count);
		m_scheduler->Record([=](vk::CommandBuffer command) {
			command.copyImage(source, source_layout, destination, destination_layout, count, r);
		});
	}
	void copyImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	               vk::ImageLayout destination_layout, const vk::ImageCopy& region) const {
		copyImage(source, source_layout, destination, destination_layout, 1, &region);
	}
	void resolveImage(vk::Image source, vk::ImageLayout source_layout, vk::Image destination,
	                  vk::ImageLayout destination_layout, const vk::ImageResolve& region) const {
		Run([=](vk::CommandBuffer command) {
			command.resolveImage(source, source_layout, destination, destination_layout, 1,
			                     &region);
		});
	}
	void clearColorImage(vk::Image image, vk::ImageLayout layout, const vk::ClearColorValue* color,
	                     uint32_t count, const vk::ImageSubresourceRange* ranges) const {
		if (!Threaded()) {
			m_direct.clearColorImage(image, layout, color, count, ranges);
			return;
		}
		const auto  value = *color;
		const auto* r     = m_scheduler->Stash(ranges, count);
		m_scheduler->Record([=](vk::CommandBuffer command) {
			command.clearColorImage(image, layout, &value, count, r);
		});
	}
	void clearDepthStencilImage(vk::Image image, vk::ImageLayout layout,
	                            const vk::ClearDepthStencilValue* value, uint32_t count,
	                            const vk::ImageSubresourceRange* ranges) const {
		if (!Threaded()) {
			m_direct.clearDepthStencilImage(image, layout, value, count, ranges);
			return;
		}
		const auto  clear = *value;
		const auto* r     = m_scheduler->Stash(ranges, count);
		m_scheduler->Record([=](vk::CommandBuffer command) {
			command.clearDepthStencilImage(image, layout, &clear, count, r);
		});
	}
	template <typename Writes>
	void pushDescriptorSetKHR(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
	                          uint32_t set, const Writes& writes) const {
		pushDescriptorSetKHR(bind_point, layout, set, static_cast<uint32_t>(std::size(writes)),
		                     std::data(writes));
	}
	void beginRendering(const vk::RenderingInfo* info) const { beginRendering(*info); }
	void beginRendering(const vk::RenderingInfo& info) const {
		if (!Threaded()) {
			m_direct.beginRendering(info);
			return;
		}
		EXIT_IF(info.pNext != nullptr);
		RequireNoNext(info.pColorAttachments, info.colorAttachmentCount);
		RequireNoNext(info.pDepthAttachment, info.pDepthAttachment != nullptr ? 1u : 0u);
		RequireNoNext(info.pStencilAttachment, info.pStencilAttachment != nullptr ? 1u : 0u);
		auto copy = info;
		copy.pColorAttachments =
		    m_scheduler->Stash(info.pColorAttachments, info.colorAttachmentCount);
		copy.pDepthAttachment =
		    m_scheduler->Stash(info.pDepthAttachment, info.pDepthAttachment != nullptr ? 1u : 0u);
		copy.pStencilAttachment = m_scheduler->Stash(info.pStencilAttachment,
		                                             info.pStencilAttachment != nullptr ? 1u : 0u);
		m_scheduler->Record([copy](vk::CommandBuffer command) { command.beginRendering(copy); });
	}
	// Step 3: VK_EXT_conditional_rendering. Each predicate dword gates one conditional
	// section; the predicate buffer carries eConditionalRenderingEXT usage.
	void beginConditionalRendering(const vk::ConditionalRenderingBeginInfoEXT& info) const {
		if (!Threaded()) {
			m_direct.beginConditionalRenderingEXT(&info);
			return;
		}
		const auto copy = *m_scheduler->Stash(&info, 1);
		m_scheduler->Record(
		    [copy](vk::CommandBuffer command) { command.beginConditionalRenderingEXT(&copy); });
	}
	void endConditionalRendering() const {
		Run([](vk::CommandBuffer command) { command.endConditionalRenderingEXT(); });
	}
	void endRendering() const {
		Run([](vk::CommandBuffer command) { command.endRendering(); });
	}
	void setCheckpointNV(const void* marker) const {
		Run([=](vk::CommandBuffer command) { command.setCheckpointNV(marker); });
	}
	void setViewportWithCount(uint32_t count, const vk::Viewport* viewports) const {
		if (!Threaded()) {
			if (!RecordViewportWc(count, viewports, nullptr, 0, true)) {
				return;
			}
			m_direct.setViewportWithCount(count, viewports);
			return;
		}
		if (!RecordViewportWc(count, viewports, nullptr, 0, true)) {
			return;
		}
		const auto* v = m_scheduler->Stash(viewports, count);
		m_scheduler->Record(
		    [=](vk::CommandBuffer command) { command.setViewportWithCount(count, v); });
	}
	void setScissorWithCount(uint32_t count, const vk::Rect2D* scissors) const {
		if (!Threaded()) {
			if (!RecordViewportWc(0, nullptr, scissors, count, false)) {
				return;
			}
			m_direct.setScissorWithCount(count, scissors);
			return;
		}
		if (!RecordViewportWc(0, nullptr, scissors, count, false)) {
			return;
		}
		const auto* s = m_scheduler->Stash(scissors, count);
		m_scheduler->Record(
		    [=](vk::CommandBuffer command) { command.setScissorWithCount(count, s); });
	}
	// Non-WithCount form used by the blit helper: distinct Vulkan dynamic state, tracked
	// in its own shadow entry (never compared against WithCount values). Viewport and
	// scissor are independent commands: each compares only its own values.
	void setViewport(uint32_t first, uint32_t count, const vk::Viewport* viewports) const {
		if (m_scheduler != nullptr && RecordDynamicCacheEnabled()) {
			auto& cache = m_scheduler->RecordCache();
			const bool same =
			    cache.vp_viewport_known && cache.vp_first == first && cache.vp_count == count &&
			    count <= RecordedStateCache::kMaxViewports &&
			    ViewportsEqual(cache.vp_viewports.data(), viewports, count);
			if (same) {
				GetFrameTelemetry().record_dynamic_skipped++;
				return;
			}
			if (count <= RecordedStateCache::kMaxViewports) {
				if (count != 0) {
					std::memcpy(cache.vp_viewports.data(), viewports,
					            count * sizeof(vk::Viewport));
				}
				cache.vp_first          = first;
				cache.vp_count          = count;
				cache.vp_viewport_known = true;
			} else {
				cache.vp_viewport_known = false;
			}
			GetFrameTelemetry().record_dynamic_emitted++;
		}
		if (!Threaded()) {
			m_direct.setViewport(first, count, viewports);
			return;
		}
		const auto* v = m_scheduler->Stash(viewports, count);
		m_scheduler->Record(
		    [=](vk::CommandBuffer command) { command.setViewport(first, count, v); });
	}
	void setScissor(uint32_t first, uint32_t count, const vk::Rect2D* scissors) const {
		if (m_scheduler != nullptr && RecordDynamicCacheEnabled()) {
			auto& cache = m_scheduler->RecordCache();
			const bool same =
			    cache.vp_scissor_known && cache.vp_scissor_first == first &&
			    cache.vp_scissor_count == count && count <= RecordedStateCache::kMaxViewports &&
			    ScissorsEqual(cache.vp_scissors.data(), scissors, count);
			if (same) {
				GetFrameTelemetry().record_dynamic_skipped++;
				return;
			}
			if (count <= RecordedStateCache::kMaxViewports) {
				if (count != 0) {
					std::memcpy(cache.vp_scissors.data(), scissors, count * sizeof(vk::Rect2D));
				}
				cache.vp_scissor_first = first;
				cache.vp_scissor_count = count;
				cache.vp_scissor_known = true;
			} else {
				cache.vp_scissor_known = false;
			}
			GetFrameTelemetry().record_dynamic_emitted++;
		}
		if (!Threaded()) {
			m_direct.setScissor(first, count, scissors);
			return;
		}
		const auto* s = m_scheduler->Stash(scissors, count);
		m_scheduler->Record(
		    [=](vk::CommandBuffer command) { command.setScissor(first, count, s); });
	}
	void setLineWidth(float width) const {
		if (m_scheduler != nullptr && RecordDynamicCacheEnabled()) {
			auto& cache = m_scheduler->RecordCache();
			if (cache.line_width_known &&
			    std::memcmp(&cache.line_width, &width, sizeof(float)) == 0) {
				GetFrameTelemetry().record_dynamic_skipped++;
				return;
			}
			cache.line_width       = width;
			cache.line_width_known = true;
			GetFrameTelemetry().record_dynamic_emitted++;
		}
		Run([=](vk::CommandBuffer command) { command.setLineWidth(width); });
	}
	void setBlendConstants(const float constants[4]) const {
		const std::array<float, 4> values {constants[0], constants[1], constants[2], constants[3]};
		if (m_scheduler != nullptr && RecordDynamicCacheEnabled()) {
			auto& cache = m_scheduler->RecordCache();
			if (cache.blend_constants_known && cache.blend_constants == values) {
				GetFrameTelemetry().record_dynamic_skipped++;
				return;
			}
			cache.blend_constants       = values;
			cache.blend_constants_known = true;
			GetFrameTelemetry().record_dynamic_emitted++;
		}
		Run([=](vk::CommandBuffer command) { command.setBlendConstants(values.data()); });
	}
	void setCullMode(vk::CullModeFlags mode) const {
		if (m_scheduler != nullptr && RecordDynamicCacheEnabled()) {
			auto& cache = m_scheduler->RecordCache();
			if (cache.cull_mode_known && cache.cull_mode == mode) {
				GetFrameTelemetry().record_dynamic_skipped++;
				return;
			}
			cache.cull_mode       = mode;
			cache.cull_mode_known = true;
			GetFrameTelemetry().record_dynamic_emitted++;
		}
		Run([=](vk::CommandBuffer command) { command.setCullMode(mode); });
	}
	void setFrontFace(vk::FrontFace face) const {
		if (m_scheduler != nullptr && RecordDynamicCacheEnabled()) {
			auto& cache = m_scheduler->RecordCache();
			if (cache.front_face_known && cache.front_face == face) {
				GetFrameTelemetry().record_dynamic_skipped++;
				return;
			}
			cache.front_face       = face;
			cache.front_face_known = true;
			GetFrameTelemetry().record_dynamic_emitted++;
		}
		Run([=](vk::CommandBuffer command) { command.setFrontFace(face); });
	}
	void setDepthBoundsTestEnable(vk::Bool32 enable) const {
		if (m_scheduler != nullptr && RecordDynamicCacheEnabled()) {
			auto& cache = m_scheduler->RecordCache();
			if (cache.depth_bounds_test_enable_known && cache.depth_bounds_test_enable == enable) {
				GetFrameTelemetry().record_dynamic_skipped++;
				return;
			}
			cache.depth_bounds_test_enable       = enable;
			cache.depth_bounds_test_enable_known = true;
			GetFrameTelemetry().record_dynamic_emitted++;
		}
		Run([=](vk::CommandBuffer command) { command.setDepthBoundsTestEnable(enable); });
	}
	void setDepthBounds(float min_bounds, float max_bounds) const {
		if (m_scheduler != nullptr && RecordDynamicCacheEnabled()) {
			auto& cache = m_scheduler->RecordCache();
			if (cache.depth_bounds_known && cache.depth_min_bounds == min_bounds &&
			    cache.depth_max_bounds == max_bounds) {
				GetFrameTelemetry().record_dynamic_skipped++;
				return;
			}
			cache.depth_min_bounds   = min_bounds;
			cache.depth_max_bounds   = max_bounds;
			cache.depth_bounds_known = true;
			GetFrameTelemetry().record_dynamic_emitted++;
		}
		Run([=](vk::CommandBuffer command) { command.setDepthBounds(min_bounds, max_bounds); });
	}
	void setDepthTestEnable(vk::Bool32 enable) const {
		if (m_scheduler != nullptr && RecordDynamicCacheEnabled()) {
			auto& cache = m_scheduler->RecordCache();
			if (cache.depth_test_enable_known && cache.depth_test_enable == enable) {
				GetFrameTelemetry().record_dynamic_skipped++;
				return;
			}
			cache.depth_test_enable       = enable;
			cache.depth_test_enable_known = true;
			GetFrameTelemetry().record_dynamic_emitted++;
		}
		Run([=](vk::CommandBuffer command) { command.setDepthTestEnable(enable); });
	}
	void setDepthWriteEnable(vk::Bool32 enable) const {
		if (m_scheduler != nullptr && RecordDynamicCacheEnabled()) {
			auto& cache = m_scheduler->RecordCache();
			if (cache.depth_write_enable_known && cache.depth_write_enable == enable) {
				GetFrameTelemetry().record_dynamic_skipped++;
				return;
			}
			cache.depth_write_enable       = enable;
			cache.depth_write_enable_known = true;
			GetFrameTelemetry().record_dynamic_emitted++;
		}
		Run([=](vk::CommandBuffer command) { command.setDepthWriteEnable(enable); });
	}
	void setDepthCompareOp(vk::CompareOp op) const {
		if (m_scheduler != nullptr && RecordDynamicCacheEnabled()) {
			auto& cache = m_scheduler->RecordCache();
			if (cache.depth_compare_op_known && cache.depth_compare_op == op) {
				GetFrameTelemetry().record_dynamic_skipped++;
				return;
			}
			cache.depth_compare_op       = op;
			cache.depth_compare_op_known = true;
			GetFrameTelemetry().record_dynamic_emitted++;
		}
		Run([=](vk::CommandBuffer command) { command.setDepthCompareOp(op); });
	}
	void setDepthBiasEnable(vk::Bool32 enable) const {
		if (m_scheduler != nullptr && RecordDynamicCacheEnabled()) {
			auto& cache = m_scheduler->RecordCache();
			if (cache.depth_bias_enable_known && cache.depth_bias_enable == enable) {
				GetFrameTelemetry().record_dynamic_skipped++;
				return;
			}
			cache.depth_bias_enable       = enable;
			cache.depth_bias_enable_known = true;
			GetFrameTelemetry().record_dynamic_emitted++;
		}
		Run([=](vk::CommandBuffer command) { command.setDepthBiasEnable(enable); });
	}
	void setDepthBias(float constant, float clamp, float slope) const {
		if (m_scheduler != nullptr && RecordDynamicCacheEnabled()) {
			auto& cache = m_scheduler->RecordCache();
			if (cache.depth_bias_known && cache.depth_bias_constant == constant &&
			    cache.depth_bias_clamp == clamp && cache.depth_bias_slope == slope) {
				GetFrameTelemetry().record_dynamic_skipped++;
				return;
			}
			cache.depth_bias_constant = constant;
			cache.depth_bias_clamp    = clamp;
			cache.depth_bias_slope    = slope;
			cache.depth_bias_known    = true;
			GetFrameTelemetry().record_dynamic_emitted++;
		}
		Run([=](vk::CommandBuffer command) { command.setDepthBias(constant, clamp, slope); });
	}
	void setStencilTestEnable(vk::Bool32 enable) const {
		if (m_scheduler != nullptr && RecordDynamicCacheEnabled()) {
			auto& cache = m_scheduler->RecordCache();
			if (cache.stencil_test_enable_known && cache.stencil_test_enable == enable) {
				GetFrameTelemetry().record_dynamic_skipped++;
				return;
			}
			cache.stencil_test_enable       = enable;
			cache.stencil_test_enable_known = true;
			GetFrameTelemetry().record_dynamic_emitted++;
		}
		Run([=](vk::CommandBuffer command) { command.setStencilTestEnable(enable); });
	}
	void setStencilOp(vk::StencilFaceFlags face, vk::StencilOp fail, vk::StencilOp pass,
	                  vk::StencilOp depth_fail, vk::CompareOp compare) const {
		if (m_scheduler != nullptr && RecordDynamicCacheEnabled()) {
			auto& cache = m_scheduler->RecordCache();
			// eStencilOp is declared dynamic by every renderer graphics pipeline (see
			// shaders.cpp dynamic_states); the helper graphics pipelines never record
			// stencil state, so a value comparison is valid here.
			if (StencilOpMatches(cache, face, fail, pass, depth_fail, compare)) {
				GetFrameTelemetry().record_dynamic_skipped++;
				return;
			}
			StencilOpUpdate(cache, face, fail, pass, depth_fail, compare);
			GetFrameTelemetry().record_dynamic_emitted++;
		}
		Run([=](vk::CommandBuffer command) {
			command.setStencilOp(face, fail, pass, depth_fail, compare);
		});
	}
	void setStencilCompareMask(vk::StencilFaceFlags face, uint32_t mask) const {
		if (m_scheduler != nullptr && RecordDynamicCacheEnabled()) {
			auto& cache = m_scheduler->RecordCache();
			if (StencilMaskMatches(cache, face, mask, true)) {
				GetFrameTelemetry().record_dynamic_skipped++;
				return;
			}
			StencilMaskUpdate(cache, face, mask, true);
			GetFrameTelemetry().record_dynamic_emitted++;
		}
		Run([=](vk::CommandBuffer command) { command.setStencilCompareMask(face, mask); });
	}
	void setStencilWriteMask(vk::StencilFaceFlags face, uint32_t mask) const {
		if (m_scheduler != nullptr && RecordDynamicCacheEnabled()) {
			auto& cache = m_scheduler->RecordCache();
			if (StencilMaskMatches(cache, face, mask, false)) {
				GetFrameTelemetry().record_dynamic_skipped++;
				return;
			}
			StencilMaskUpdate(cache, face, mask, false);
			GetFrameTelemetry().record_dynamic_emitted++;
		}
		Run([=](vk::CommandBuffer command) { command.setStencilWriteMask(face, mask); });
	}
	void setStencilReference(vk::StencilFaceFlags face, uint32_t reference) const {
		if (m_scheduler != nullptr && RecordDynamicCacheEnabled()) {
			auto& cache = m_scheduler->RecordCache();
			if (StencilReferenceMatches(cache, face, reference)) {
				GetFrameTelemetry().record_dynamic_skipped++;
				return;
			}
			StencilReferenceUpdate(cache, face, reference);
			GetFrameTelemetry().record_dynamic_emitted++;
		}
		Run([=](vk::CommandBuffer command) { command.setStencilReference(face, reference); });
	}
	void setColorWriteEnableEXT(uint32_t count, const vk::Bool32* enables) const {
		if (m_scheduler != nullptr && RecordDynamicCacheEnabled()) {
			auto& cache = m_scheduler->RecordCache();
			// eColorWriteEnableEXT exists only when the pipeline declares it (renderer
			// pipelines with color_count != 0, never on macOS): the only recorder caller
			// runs right after binding such a pipeline, and helper pipelines never call
			// this. Count is part of the comparison; only the first `count` entries are
			// compared.
			const bool same = cache.color_write_known && cache.color_write_count == count &&
			                  count <= RecordedStateCache::kMaxColorAttachments &&
			                  ColorWriteEqual(cache.color_write_enables.data(), enables, count);
			if (same) {
				GetFrameTelemetry().record_dynamic_skipped++;
				return;
			}
			if (count <= RecordedStateCache::kMaxColorAttachments) {
				if (count != 0) {
					std::memcpy(cache.color_write_enables.data(), enables,
					            count * sizeof(vk::Bool32));
				}
				cache.color_write_count = count;
				cache.color_write_known = true;
			} else {
				cache.color_write_known = false;
			}
			GetFrameTelemetry().record_dynamic_emitted++;
		}
		if (!Threaded()) {
			m_direct.setColorWriteEnableEXT(count, enables);
			return;
		}
		const auto* e = m_scheduler->Stash(enables, count);
		m_scheduler->Record(
		    [=](vk::CommandBuffer command) { command.setColorWriteEnableEXT(count, e); });
	}
	void setAttachmentFeedbackLoopEnableEXT(vk::ImageAspectFlags aspects) const {
		// Single draw-time caller whose value follows the bound pipeline's feedback state;
		// with no validity signal to compare safely, always record.
		if (m_scheduler != nullptr) {
			GetFrameTelemetry().record_dynamic_emitted++;
		}
		Run([=](vk::CommandBuffer command) {
			command.setAttachmentFeedbackLoopEnableEXT(aspects);
		});
	}

private:
	// Bitwise float compare: SetGraphicsDynamicParams copies register bits verbatim, so
	// equal values must produce equal bits (no arithmetic reaches the recorder).
	static bool FloatBitsEqual(float a, float b) {
		uint32_t ua = 0;
		uint32_t ub = 0;
		std::memcpy(&ua, &a, sizeof(float));
		std::memcpy(&ub, &b, sizeof(float));
		return ua == ub;
	}
	static bool ViewportsEqual(const vk::Viewport* a, const vk::Viewport* b, uint32_t count) {
		for (uint32_t i = 0; i < count; i++) {
			if (!FloatBitsEqual(a[i].x, b[i].x) || !FloatBitsEqual(a[i].y, b[i].y) ||
			    !FloatBitsEqual(a[i].width, b[i].width) ||
			    !FloatBitsEqual(a[i].height, b[i].height) ||
			    !FloatBitsEqual(a[i].minDepth, b[i].minDepth) ||
			    !FloatBitsEqual(a[i].maxDepth, b[i].maxDepth)) {
				return false;
			}
		}
		return true;
	}
	static bool ScissorsEqual(const vk::Rect2D* a, const vk::Rect2D* b, uint32_t count) {
		for (uint32_t i = 0; i < count; i++) {
			if (a[i].offset.x != b[i].offset.x || a[i].offset.y != b[i].offset.y ||
			    a[i].extent.width != b[i].extent.width ||
			    a[i].extent.height != b[i].extent.height) {
				return false;
			}
		}
		return true;
	}
	static bool ColorWriteEqual(const vk::Bool32* a, const vk::Bool32* b, uint32_t count) {
		for (uint32_t i = 0; i < count; i++) {
			if (a[i] != b[i]) {
				return false;
			}
		}
		return true;
	}
	// Stencil state is per-face Vulkan state: front-only updates compare only the front
	// entry. Combined front+back updates compare both, so neither face suppresses the other.
	static bool StencilOpFaceMatches(const RecordedStateCache::StencilFaceState& state,
	                                 vk::StencilOp fail, vk::StencilOp pass,
	                                 vk::StencilOp depth_fail, vk::CompareOp compare) {
		return state.op_known && state.op_fail == fail && state.op_pass == pass &&
		       state.op_depth_fail == depth_fail && state.op_compare == compare;
	}
	static bool StencilOpMatches(const RecordedStateCache& cache, vk::StencilFaceFlags face,
	                             vk::StencilOp fail, vk::StencilOp pass, vk::StencilOp depth_fail,
	                             vk::CompareOp compare) {
		const bool front = (face & vk::StencilFaceFlagBits::eFront) != vk::StencilFaceFlags {};
		const bool back  = (face & vk::StencilFaceFlagBits::eBack) != vk::StencilFaceFlags {};
		if (front && !StencilOpFaceMatches(cache.stencil_front, fail, pass, depth_fail, compare)) {
			return false;
		}
		if (back && !StencilOpFaceMatches(cache.stencil_back, fail, pass, depth_fail, compare)) {
			return false;
		}
		return front || back;
	}
	static void StencilOpFaceUpdate(RecordedStateCache::StencilFaceState& state,
	                                vk::StencilOp fail, vk::StencilOp pass,
	                                vk::StencilOp depth_fail, vk::CompareOp compare) {
		state.op_fail       = fail;
		state.op_pass       = pass;
		state.op_depth_fail = depth_fail;
		state.op_compare    = compare;
		state.op_known      = true;
	}
	static void StencilOpUpdate(RecordedStateCache& cache, vk::StencilFaceFlags face,
	                            vk::StencilOp fail, vk::StencilOp pass, vk::StencilOp depth_fail,
	                            vk::CompareOp compare) {
		if ((face & vk::StencilFaceFlagBits::eFront) != vk::StencilFaceFlags {}) {
			StencilOpFaceUpdate(cache.stencil_front, fail, pass, depth_fail, compare);
		}
		if ((face & vk::StencilFaceFlagBits::eBack) != vk::StencilFaceFlags {}) {
			StencilOpFaceUpdate(cache.stencil_back, fail, pass, depth_fail, compare);
		}
	}
	static bool StencilMaskMatches(const RecordedStateCache& cache, vk::StencilFaceFlags face,
	                               uint32_t mask, bool compare_mask) {
		const bool front = (face & vk::StencilFaceFlagBits::eFront) != vk::StencilFaceFlags {};
		const bool back  = (face & vk::StencilFaceFlagBits::eBack) != vk::StencilFaceFlags {};
		if (front) {
			const auto& state = cache.stencil_front;
			const bool  known = compare_mask ? state.compare_mask_known : state.write_mask_known;
			const auto  value = compare_mask ? state.compare_mask : state.write_mask;
			if (!known || value != mask) {
				return false;
			}
		}
		if (back) {
			const auto& state = cache.stencil_back;
			const bool  known = compare_mask ? state.compare_mask_known : state.write_mask_known;
			const auto  value = compare_mask ? state.compare_mask : state.write_mask;
			if (!known || value != mask) {
				return false;
			}
		}
		return front || back;
	}
	static void StencilMaskUpdate(RecordedStateCache& cache, vk::StencilFaceFlags face,
	                              uint32_t mask, bool compare_mask) {
		if ((face & vk::StencilFaceFlagBits::eFront) != vk::StencilFaceFlags {}) {
			auto& state = cache.stencil_front;
			if (compare_mask) {
				state.compare_mask       = mask;
				state.compare_mask_known = true;
			} else {
				state.write_mask       = mask;
				state.write_mask_known = true;
			}
		}
		if ((face & vk::StencilFaceFlagBits::eBack) != vk::StencilFaceFlags {}) {
			auto& state = cache.stencil_back;
			if (compare_mask) {
				state.compare_mask       = mask;
				state.compare_mask_known = true;
			} else {
				state.write_mask       = mask;
				state.write_mask_known = true;
			}
		}
	}
	static bool StencilReferenceMatches(const RecordedStateCache& cache,
	                                    vk::StencilFaceFlags face, uint32_t reference) {
		const bool front = (face & vk::StencilFaceFlagBits::eFront) != vk::StencilFaceFlags {};
		const bool back  = (face & vk::StencilFaceFlagBits::eBack) != vk::StencilFaceFlags {};
		if (front) {
			const auto& state = cache.stencil_front;
			if (!state.reference_known || state.reference != reference) {
				return false;
			}
		}
		if (back) {
			const auto& state = cache.stencil_back;
			if (!state.reference_known || state.reference != reference) {
				return false;
			}
		}
		return front || back;
	}
	static void StencilReferenceUpdate(RecordedStateCache& cache, vk::StencilFaceFlags face,
	                                   uint32_t reference) {
		if ((face & vk::StencilFaceFlagBits::eFront) != vk::StencilFaceFlags {}) {
			cache.stencil_front.reference       = reference;
			cache.stencil_front.reference_known = true;
		}
		if ((face & vk::StencilFaceFlagBits::eBack) != vk::StencilFaceFlags {}) {
			cache.stencil_back.reference       = reference;
			cache.stencil_back.reference_known = true;
		}
	}
	template <typename F>
	void Run(F&& fn) const {
		if (!Threaded()) {
			fn(m_direct);
		} else {
			m_scheduler->Record(std::forward<F>(fn));
		}
	}
	template <typename T>
	static void RequireNoNext(const T* items, uint32_t count) {
		for (uint32_t i = 0; i < count; i++) {
			EXIT_IF(items[i].pNext != nullptr);
		}
	}
	// A deep copy of descriptor writes: the writes and the info arrays they point to.
	const vk::WriteDescriptorSet* StashWrites(const vk::WriteDescriptorSet* writes,
	                                          uint32_t                      count) const {
		RequireNoNext(writes, count);
		auto* copy = const_cast<vk::WriteDescriptorSet*>(m_scheduler->Stash(writes, count));
		for (uint32_t i = 0; i < count; i++) {
			auto& write      = copy[i];
			write.pImageInfo = m_scheduler->Stash(
			    write.pImageInfo, write.pImageInfo != nullptr ? write.descriptorCount : 0);
			write.pBufferInfo = m_scheduler->Stash(
			    write.pBufferInfo, write.pBufferInfo != nullptr ? write.descriptorCount : 0);
			write.pTexelBufferView =
			    m_scheduler->Stash(write.pTexelBufferView,
			                       write.pTexelBufferView != nullptr ? write.descriptorCount : 0);
		}
		return copy;
	}

	CommandScheduler* m_scheduler = nullptr;
	vk::CommandBuffer m_direct    = nullptr;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDRECORDER_H_
