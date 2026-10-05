#include "graphics/host_gpu/renderer/renderDraw.h"

#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/gpuTiming.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

std::pair<int32_t, uint32_t> ResolveDrawOffsets(uint32_t index_offset,
	                                           const ShaderVertexInputInfo& vs_input_info) {
	auto     vertex_offset   = static_cast<int32_t>(index_offset);
	uint32_t instance_offset = 0;
	if (!vs_input_info.fetch_embedded) {
		return {vertex_offset, instance_offset};
	}

	EXIT_IF(!vs_input_info.stage);
	const auto& program   = *vs_input_info.stage.program;
	const auto& resources = *vs_input_info.stage.resources;
	if (index_offset == 0 &&
	    program.info.vertex_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.vertex_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			vertex_offset = static_cast<int32_t>(resources.user_data[index]);
		}
	}
	if (program.info.instance_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.instance_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			instance_offset = resources.user_data[index];
		}
	}

	return {vertex_offset, instance_offset};
}

static std::atomic<uint32_t> g_draw_state_log_count   = 0;
static std::atomic<uint32_t> g_draw_input_log_count   = 0;
static std::atomic<uint32_t> g_mrt_state_log_count    = 0;

static std::atomic<uint32_t> g_framebuffer_skip_log_count = 0;

static float ConvertPolygonOffsetConstantFactor(float guest_factor, const HW::PolyOffset& offset,
                                                vk::Format host_depth_format) {
	if (offset.db_is_float_fmt) {
		return guest_factor;
	}

	// Fixed-point guest bias (e.g. D24) cannot be represented 1:1 by the
	// floating-point host attachment (D32_SFLOAT). Without a rescale the same
	// integer bias is a smaller NDC offset, which shows up as shadow acne
	// (speckle on the octagon canvas).
	int host_depth_bits = 0;
	switch (host_depth_format) {
		case vk::Format::eD16Unorm:
		case vk::Format::eD16UnormS8Uint: host_depth_bits = 16; break;
		case vk::Format::eD24UnormS8Uint: host_depth_bits = 24; break;
		case vk::Format::eD32Sfloat:
		case vk::Format::eD32SfloatS8Uint:
			// Fixed-point guest bias on a 32-bit float host attachment: keep the
			// guest intent with a 2x base rescale for the D24 -> float mapping.
			// Dropping it would leave the shadow acne behind.
			return guest_factor * 2.0f;
		default:
			// A fixed-point guest bias cannot be represented exactly by a floating-point host
			// attachment without VK_EXT_depth_bias_control.
			return guest_factor;
	}
	return std::ldexp(guest_factor, host_depth_bits + offset.neg_num_db_bits);
}

static const char* RenderColorTypeName(const RenderColorInfo& color) {
	return color.image_id ? "RenderTexture" : "NoColorOutput";
}

static void LogFramebufferSkip(const char* draw_name, const RenderColorInfo& color,
                               const RenderDepthInfo& depth, const CommandBuffer& buffer,
                               uint32_t index_count, uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	auto log_id = g_framebuffer_skip_log_count.fetch_add(1, std::memory_order_relaxed);
	if (log_id >= 128) {
		return;
	}

	LOGF(
	    "DrawFramebufferSkip[%u]: %s color=%s color_addr=0x%010" PRIx64 " color_size=0x%016" PRIx64
	    " color_image=%s depth_format=%s depth_image=%s depth_vaddr_num=%d target_mask=0x%08" PRIx32
	    " prim=%u index_count=%u flags=0x%08" PRIx32 "\n",
	    log_id, draw_name, RenderColorTypeName(color), color.desc.info.data.address,
	    color.desc.info.data.size, color.image_id ? "yes" : "no",
	    vk::to_string(depth.desc.view_info.format).c_str(), depth.image_id ? "yes" : "no",
	    static_cast<int>(!depth.desc.info.data.Empty()) +
	        static_cast<int>(depth.desc.info.HasStencil()),
	    ctx.GetRenderTargetMask(), static_cast<uint32_t>(ucfg.GetPrimType()), index_count, flags);
}

static void LogMrtState(const char* draw_name, const CommandBuffer& buffer,
                        const ShaderPixelInputInfo& ps_input_info) {
	const auto& ctx            = buffer.GetRegisters();
	const auto& sh_regs        = ctx.GetShaderRegisters();
	const auto  rt_mask        = ctx.GetRenderTargetMask();
	const auto  cb_shader_mask = sh_regs.m_cbShaderMask;
	const auto& bc0            = ctx.GetBlendControl(0);

	auto log_id = g_mrt_state_log_count.fetch_add(1);
	if (log_id >= 32) {
		return;
	}

	LOGF("MrtState[%u]: %s rt_mask=0x%08" PRIx32 " cb_shader_mask=0x%08" PRIx32
	     " blend0=%s src=%u dst=%u alpha_src=%u alpha_dst=%u sep_alpha=%s\n",
	     log_id, draw_name, rt_mask, cb_shader_mask, bc0.enable ? "true" : "false",
	     bc0.color_srcblend, bc0.color_destblend, bc0.alpha_srcblend, bc0.alpha_destblend,
	     bc0.separate_alpha_blend ? "true" : "false");

	for (uint32_t i = 0; i < 8; i++) {
		const auto& rt  = ctx.GetRenderTarget(i);
		const auto& bc  = ctx.GetBlendControl(i);
		const auto  ctm = (rt_mask >> (i * 4u)) & 0x0fu;
		const auto  csm = (cb_shader_mask >> (i * 4u)) & 0x0fu;

		if (rt.base.addr == 0 && ps_input_info.target_output_mode[i] == 0 && ctm == 0 && csm == 0 &&
		    !bc.enable) {
			continue;
		}

		LOGF("MrtState[%u]: slot=%u addr=0x%010" PRIx64
		     " target_mask=0x%x shader_mask=0x%x out_mode=%u"
		     " fmt=0x%08" PRIx32 " nfmt=0x%08" PRIx32 " order=0x%08" PRIx32
		     " width=%u height=%u tile=%u"
		     " blend=%s src=%u dst=%u alpha_src=%u alpha_dst=%u\n",
		     log_id, i, rt.base.addr, ctm, csm, ps_input_info.target_output_mode[i],
		     static_cast<uint32_t>(rt.info.format), static_cast<uint32_t>(rt.info.channel_type),
		     static_cast<uint32_t>(rt.info.channel_order), rt.attrib2.width + 1,
		     rt.attrib2.height + 1, static_cast<uint32_t>(rt.attrib3.tile_mode),
		     bc.enable ? "true" : "false", bc.color_srcblend, bc.color_destblend, bc.alpha_srcblend,
		     bc.alpha_destblend);
	}
}

static void LogDrawTargetState(const char* draw_name, const RenderColorInfo& color,
                               const RenderDepthInfo& depth, const CommandBuffer& buffer,
                               const ShaderPixelInputInfo& ps_input_info, uint32_t index_count,
                               uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (!color.image_id) {
		return;
	}

	auto log_id = g_draw_state_log_count.fetch_add(1);
	if (log_id >= 192) {
		return;
	}

	const auto& cc             = ctx.GetColorControl();
	const auto& bc             = ctx.GetBlendControl(color.target_slot);
	const auto& dc             = ctx.GetDepthControl();
	const auto& vp             = ctx.GetScreenViewport();
	const auto& vp0            = vp.viewports[0];
	const auto& ps_resources   = ps_input_info.stage.program->info;
	const auto  sampled_images = std::count_if(
	    ps_resources.images.begin(), ps_resources.images.end(), [](const auto& image) {
		    return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
	    });

	const auto extent = color.Extent();
	const auto sc     = calc_final_scissor(vp, ctx.GetScanModeControl(), extent, 0);

	LOGF(
	    "DrawTargetState[%u]: frame=%d %s target=%s addr=0x%010" PRIx64
	    " extent=%ux%u prim=%u index_count=%u flags=0x%08" PRIx32 " color_mask=0x%08" PRIx32
	    " cc_mode=%u cc_op=0x%02x"
	    " blend=%s src=%u dst=%u comb=%u ps_tex=%d sampled=%d storage=%d ps_kill=%s target_mode0=%u"
	    " depth_test=%s depth_write=%s depth_func=%u depth_clear=%s viewport=(%.1f,%.1f %.1fx%.1f) "
	    "scissor=(%d,%d)-(%d,%d)\n",
	    log_id, buffer.GetContext().GetGpu().GetFrameNum(), draw_name, RenderColorTypeName(color),
	    color.desc.info.data.address, extent.width, extent.height,
	    static_cast<uint32_t>(ucfg.GetPrimType()), index_count, flags, ctx.GetRenderTargetMask(),
	    cc.mode, cc.op,
	    bc.enable ? "true" : "false", bc.color_srcblend, bc.color_destblend, bc.color_comb_fcn,
	    static_cast<int>(ps_resources.images.size()), static_cast<int>(sampled_images),
	    static_cast<int>(ps_resources.images.size() - sampled_images),
	    ps_input_info.ps_pixel_kill_enable ? "true" : "false", ps_input_info.target_output_mode[0],
	    dc.z_enable ? "true" : "false", dc.z_write_enable ? "true" : "false", dc.zfunc,
	    depth.depth_clear_enable ? "true" : "false", vp0.xoffset - vp0.xscale,
	    vp0.yoffset - vp0.yscale, vp0.xscale * 2.0f, vp0.yscale * 2.0f, sc.left, sc.top, sc.right,
	    sc.bottom);

	LogMrtState(draw_name, buffer, ps_input_info);
}

static void LogDrawInputState(const CommandBuffer& buffer, const RenderColorInfo& color,
                              const ShaderVertexInputInfo& vs_input_info,
                              uint32_t index_type_and_size, uint32_t index_count,
                              const void* index_addr) {
	auto log_id = g_draw_input_log_count.fetch_add(1);
	if (log_id >= 64) {
		return;
	}

	LOGF("DrawInputState[%u]: frame=%d target=%s addr=0x%010" PRIx64
	     " index_type=%u index_count=%u index_addr=0x%016" PRIx64
	     " vs_resources=%d vs_buffers=%d\n",
	     log_id, buffer.GetContext().GetGpu().GetFrameNum(), RenderColorTypeName(color),
	     color.desc.info.data.address, index_type_and_size, index_count,
	     reinterpret_cast<uint64_t>(index_addr), vs_input_info.resources_num,
	     vs_input_info.buffers_num);

	for (int bi = 0; bi < vs_input_info.buffers_num; bi++) {
		const auto& b = vs_input_info.buffers[bi];
		LOGF("DrawInputState[%u]: vb[%d] addr=0x%010" PRIx64
		     " stride=%u records=%u fetch_index=%u\n",
		     log_id, bi, b.addr, b.stride, b.num_records, b.fetch_index);

		const auto* bytes = reinterpret_cast<const uint8_t*>(b.addr);
		if (bytes != nullptr && b.stride != 0) {
			const uint32_t records = std::min<uint32_t>(b.num_records, 4u);
			for (uint32_t rec = 0; rec < records; rec++) {
				const auto* rec_bytes = bytes + static_cast<uint64_t>(rec) * b.stride;
				const auto  dword_num = std::min<uint32_t>(b.stride / 4u, 12u);
				uint32_t    raw[12]   = {};
				float       flt[12]   = {};
				for (uint32_t i = 0; i < dword_num; i++) {
					std::memcpy(&raw[i], rec_bytes + i * 4u, sizeof(raw[i]));
					std::memcpy(&flt[i], rec_bytes + i * 4u, sizeof(flt[i]));
				}
				LOGF("DrawInputState[%u]: vb[%d].rec[%u] stride=%u dwords=%u raw=%08" PRIx32
				     " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32
				     " %08" PRIx32 " %08" PRIx32 " %08" PRIx32
				     " f=(%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f)\n",
				     log_id, bi, rec, b.stride, dword_num, raw[0], raw[1], raw[2], raw[3], raw[4],
				     raw[5], raw[6], raw[7], raw[8], flt[0], flt[1], flt[2], flt[3], flt[4], flt[5],
				     flt[6], flt[7], flt[8]);

				for (int ai = 0; ai < vs_input_info.resources_num; ai++) {
					const auto& r  = vs_input_info.resources[ai];
					const auto& rd = vs_input_info.resources_dst[ai];
					if (rd.buffer_index != bi) {
						continue;
					}
					const auto offset = static_cast<uint32_t>(r.Base48() - b.addr);
					if (offset + 4u <= b.stride &&
					    r.Format() == Prospero::BufferFormat::k8_8_8_8UNorm) {
						uint32_t packed = 0;
						std::memcpy(&packed, rec_bytes + offset, sizeof(packed));
						const auto r8 = (packed >> 0u) & 0xffu;
						const auto g8 = (packed >> 8u) & 0xffu;
						const auto b8 = (packed >> 16u) & 0xffu;
						const auto a8 = (packed >> 24u) & 0xffu;
						LOGF("DrawInputState[%u]: vb[%d].rec[%u].attr[%d] dst=v%d fmt=56 "
						     "rgba8=%02" PRIx32 "%02" PRIx32 "%02" PRIx32 "%02" PRIx32
						     " rgba=(%.3f,%.3f,%.3f,%.3f)\n",
						     log_id, bi, rec, ai, rd.register_start, r8, g8, b8, a8,
						     static_cast<double>(r8) / 255.0, static_cast<double>(g8) / 255.0,
						     static_cast<double>(b8) / 255.0, static_cast<double>(a8) / 255.0);
					}
				}
			}
		}

		for (int ai = 0; ai < vs_input_info.resources_num; ai++) {
			const auto& r  = vs_input_info.resources[ai];
			const auto& rd = vs_input_info.resources_dst[ai];
			if (rd.buffer_index != bi) {
				continue;
			}
			LOGF("DrawInputState[%u]: attr[%d] offset=%u dst=v%d regs=%d fetch_index=%u "
			     "sharp=%08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 "\n",
			     log_id, ai, static_cast<uint32_t>(r.Base48() - b.addr), rd.register_start,
			     rd.registers_num, rd.fetch_index, r.fields[0], r.fields[1], r.fields[2], r.fields[3]);
		}
	}
}

static void SetGraphicsDynamicParams(const CommandBuffer& buffer, vk::CommandBuffer vk_buffer,
                                     const ShaderVertexInputInfo& vs_input_info,
                                     const RenderDepthInfo& depth, const RenderState& rendering,
                                     bool depth_only, bool ps_active) {
	KYTY_PROFILER_FUNCTION();

	// Fast guard: a draw with no active rasterization target (no pixel shader
	// and no depth/stencil attachment work) consumes no viewport/scissor/
	// depth dynamic state — skip the whole block. Depth-only shadow passes
	// still need depth state, so they only skip the viewport/scissor setup
	// when no depth attachment is bound either.
	const bool has_depth_stencil_state =
	    depth.depth_test_enable || depth.depth_write_enable || depth.stencil_test_enable;
	if (!ps_active && !has_depth_stencil_state && !depth.image_id) {
		return;
	}
	const bool skip_viewport_scissor = depth_only && !depth.image_id;

	const auto& ctx = buffer.GetRegisters();
	const auto&        vp  = ctx.GetScreenViewport();
	const vk::Extent2D framebuffer_extent {rendering.width, rendering.height};
	const auto& outputs = vs_input_info.stage.program->info.outputs;
	const bool  indexed_viewports =
	    std::any_of(outputs.begin(), outputs.end(), [](const auto& output) {
		    return output.kind == ShaderRecompiler::IR::StageOutputKind::ViewportIndex;
	    });
	constexpr uint32_t viewport_slots = std::size(HW::ScreenViewport {}.viewports);
	std::array<vk::Viewport, viewport_slots> viewports {};
	std::array<vk::Rect2D, viewport_slots>   scissors {};
	const uint32_t viewport_count = indexed_viewports ? viewport_slots : 1;
	for (uint32_t i = 0; i < viewport_count; i++) {
		const auto& guest    = vp.viewports[i];
		auto&       viewport = viewports[i];
		if (ctx.GetClipControl().clip_disable) {
			const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
			viewport.width  = static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u));
			viewport.height = static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u));
		} else {
			viewport.x      = guest.xoffset - guest.xscale;
			viewport.y      = guest.yoffset - guest.yscale;
			viewport.width  = guest.xscale * 2.0f;
			viewport.height = guest.yscale * 2.0f;
			// TAA jitter is the fractional part of the guest offset above. It is
			// forwarded unchanged: rounding it collapses the pattern and the
			// octagon floor grid shimmers.
		}
		viewport.minDepth =
		    guest.zoffset - (ctx.GetClipControl().dx_clip_space ? 0.0f : guest.zscale);
		viewport.maxDepth = guest.zscale + guest.zoffset;

		const auto final_scissor =
		    calc_final_scissor(vp, ctx.GetScanModeControl(), framebuffer_extent, i);
		auto& scissor  = scissors[i];
		scissor.offset = {final_scissor.left, final_scissor.top};
		scissor.extent = {static_cast<uint32_t>(final_scissor.right - final_scissor.left),
		                  static_cast<uint32_t>(final_scissor.bottom - final_scissor.top)};
		if (viewport.width == 0.0f) {
			// Keep empty slots at their guest index; Vulkan requires a positive viewport width.
			viewport.width = 1.0f;
			scissor.extent = {0, 0};
		}
	}
	// UFC hot loop: identical viewport/scissor pairs repeat across thousands of
	// draws per frame; skip the driver call when the cached values match.
	// Rasterizer-discarded draws (guard above) never reach here; depth-only
	// passes without a depth attachment skip viewport/scissor entirely.
	if (!skip_viewport_scissor) {
		if (buffer.SetViewportWithCountCached(viewport_count, viewports.data())) {
			vk_buffer.setViewportWithCount(viewport_count, viewports.data());
		}
		if (buffer.SetScissorWithCountCached(viewport_count, scissors.data())) {
			vk_buffer.setScissorWithCount(viewport_count, scissors.data());
		}
	}

	float line_width = ctx.GetLineWidth();
	if (line_width != 1.0f) {
		static bool logged = false;
		if (!logged) {
			LOGF("Render: temporary: clamping Vulkan line width %f to 1.0 because wideLines is "
			     "not enabled\n",
			     line_width);
			logged = true;
		}
		line_width = 1.0f;
	}
	if (buffer.SetLineWidthCached(line_width)) {
		vk_buffer.setLineWidth(line_width);
	}
	if (!depth_only) {
		// Depth-only shadow passes write no color: blend constants are
		// unused by the pipeline (and blending is statically disabled),
		// so skip the driver call on the 4 shadow cascades.
		const auto&      blend = ctx.GetBlendColor();
		const std::array blend_constants {blend.red, blend.green, blend.blue, blend.alpha};
		if (buffer.SetBlendConstantsCached(blend_constants.data())) {
			vk_buffer.setBlendConstants(blend_constants.data());
		}
	}
	if (buffer.SetDepthStateCached(depth.depth_test_enable, depth.depth_write_enable,
	                              depth.depth_compare_op)) {
		vk_buffer.setDepthTestEnable(depth.depth_test_enable ? VK_TRUE : VK_FALSE);
		vk_buffer.setDepthWriteEnable(depth.depth_write_enable ? VK_TRUE : VK_FALSE);
		vk_buffer.setDepthCompareOp(depth.depth_compare_op);
	}

	const auto& mode              = ctx.GetModeControl();
	const auto& poly_offset       = ctx.GetPolyOffset();
	const bool  use_front         = mode.poly_offset_front_enable && !mode.cull_front;
	const bool  use_back          = mode.poly_offset_back_enable && !mode.cull_back;
	const bool  depth_bias_enable = use_front || use_back;
	// Vulkan has one bias for both faces. Prefer a visible front face when both are enabled.
	const float guest_constant_factor =
	    use_front ? poly_offset.front_offset : poly_offset.back_offset;
	const float constant_factor = depth_bias_enable
	                                  ? ConvertPolygonOffsetConstantFactor(
	                                        guest_constant_factor, poly_offset,
	                                        depth.desc.view_info.format)
	                                  : 0.0f;
	const float slope_factor =
	    depth_bias_enable
	        ? (use_front ? poly_offset.front_scale : poly_offset.back_scale) / 16.0f
	        : 0.0f;
	// Depth bias is disabled on almost every draw; cache enable+values together
	// so the common (disabled, 0, 0, 0) case skips both driver calls.
	if (buffer.SetDepthBiasCached(depth_bias_enable, constant_factor, poly_offset.clamp,
	                              slope_factor)) {
		vk_buffer.setDepthBiasEnable(depth_bias_enable ? VK_TRUE : VK_FALSE);
		if (depth_bias_enable) {
			vk_buffer.setDepthBias(constant_factor, poly_offset.clamp, slope_factor);
		}
	}

	if (buffer.SetStencilStateCached(depth.stencil_test_enable, depth.stencil_front,
	                                 depth.stencil_back)) {
		vk_buffer.setStencilTestEnable(depth.stencil_test_enable ? VK_TRUE : VK_FALSE);
		if (depth.stencil_test_enable) {
			const auto set_stencil = [&](vk::StencilFaceFlagBits face,
			                             const vk::StencilOpState& state) {
				vk_buffer.setStencilOp(face, state.failOp, state.passOp, state.depthFailOp,
				                       state.compareOp);
				vk_buffer.setStencilCompareMask(face, state.compareMask);
				vk_buffer.setStencilWriteMask(face, state.writeMask);
				vk_buffer.setStencilReference(face, state.reference);
			};
			set_stencil(vk::StencilFaceFlagBits::eFront, depth.stencil_front);
			set_stencil(vk::StencilFaceFlagBits::eBack, depth.stencil_back);
		}
	}

#if defined(__APPLE__)
	// MoltenVK has no VK_EXT_color_write_enable; the pipeline is created without the
	// eColorWriteEnableEXT dynamic state and relies on the static colorWriteMask instead.
#else
	vk::Bool32 enable[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	for (uint32_t slot = 0; slot < rendering.num_color_attachments; slot++) {
		enable[slot] = rendering.color_attachments[slot].image_view != nullptr;
	}
	if (rendering.num_color_attachments != 0 &&
	    buffer.SetColorWriteEnableCached(rendering.num_color_attachments, enable)) {
		vk_buffer.setColorWriteEnableEXT(rendering.num_color_attachments, enable);
	}
#endif
}

static bool DrawHasValidVertexShader(const HW::Shader& sh_ctx) {

	const auto& vs = sh_ctx.GetVs();
	return vs.es_regs.data_addr != 0;
}

static bool PixelShaderHasDepthOrCoverageSideEffects(const HW::ShaderRegisters& sh_regs) {
	const auto& db = sh_regs.db_shader_control;
	return db.shader_kill_enable || db.shader_z_export_enable || db.shader_mask_export_enable ||
	       db.shader_dual_export_enable || db.shader_execute_on_noop;
}

struct DrawRenderState {
	RenderDepthInfo       depth_info;
	RenderColorInfo       color_info[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	uint32_t              color_count                              = 0;
	bool                  ps_active                                = true;
	std::array<ShaderVertexInputInfo, 3> vertex_info;
	ShaderPixelInputInfo  ps_input_info;
	PipelineCache::GraphicsPrograms programs;
};

struct DrawCallInfo {
	CommandBufferDebugOp debug_op       = CommandBufferDebugOp::DrawIndex;
	uint32_t             index_count    = 0;
	uint32_t             instance_count = 0;
	uint32_t             first_instance = 0;

	[[nodiscard]] bool IsIndexed() const { return debug_op == CommandBufferDebugOp::DrawIndex; }
	[[nodiscard]] const char* Name() const { return IsIndexed() ? "DrawIndex" : "DrawIndexAuto"; }
};

RenderState RenderExecutor::AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
                                                 uint32_t color_count, RenderDepthInfo& depth,
                                                 vk::ImageAspectFlags& feedback_aspects,
                                                 std::span<PreparedBindings* const> stages) {
	EXIT_IF(colors == nullptr || color_count > RENDER_COLOR_ATTACHMENTS_MAX);
	feedback_aspects = {};
	auto&       cache = m_context.GetTextureCache();
	RenderState state {};
	state.width                 = std::numeric_limits<uint32_t>::max();
	state.height                = std::numeric_limits<uint32_t>::max();
	state.num_layers            = std::numeric_limits<uint32_t>::max();
	state.num_color_attachments = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		auto& target = colors[i];
		EXIT_IF(!target.image_id);
		const auto owner = cache.m_slot_images.try_get(target.image_id);
		if (owner == nullptr || (!owner->registered && !owner->info.data.Empty()) ||
		    owner->binding.needs_rebind) {
			EXIT("color target changed after render-state discovery\n");
		}
		const auto image_view = cache.FindRenderTarget(target.image_id, target.desc);
		auto&      image      = cache.GetImage(target.image_id);
		EXIT_IF(image.backing.samples != target.desc.info.samples || image_view == nullptr);
		const auto& view   = target.desc.view_info;
		const auto  layout = image.binding.is_bound ? vk::ImageLayout::eGeneral
		                                            : vk::ImageLayout::eColorAttachmentOptimal;
		image.binding.attachment_layout = layout;
		image.binding.attachment_access =
		    vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite;
		image.Transit(layout, image.binding.attachment_access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Handle());
		const auto extent       = target.Extent();
		state.width             = std::min(state.width, extent.width);
		state.height            = std::min(state.height, extent.height);
		state.num_layers        = std::min(state.num_layers, view.layer_count);
		state.num_color_attachments = std::max(state.num_color_attachments, target.target_slot + 1);
		auto& attachment            = state.color_attachments[target.target_slot];
		attachment.image_view   = image_view;
		attachment.image_layout = layout;
	}
	if (depth.image_id) {
		const auto owner = cache.m_slot_images.try_get(depth.image_id);
		if (owner == nullptr || !owner->registered || owner->binding.needs_rebind) {
			EXIT("depth target changed after render-state discovery\n");
		}
		const auto  image_view = cache.FindDepthTarget(depth.image_id, depth.desc);
		const auto& metadata   = depth.desc.info.metadata;
		if (metadata.kind == ImageMetadataKind::Htile && depth.depth_clear_enable &&
		    !cache.ClearMeta(metadata.range.address)) {
			EXIT("failed to acquire HTile metadata for a depth clear\n");
		}
		const bool meta_clear =
		    metadata.kind == ImageMetadataKind::Htile &&
		    cache.IsMetaCleared(metadata.range.address, depth.desc.view_info.base_layer);
		depth.depth_load_clear_enable = depth.depth_clear_enable || meta_clear;
		if (meta_clear &&
		    !cache.TouchMeta(metadata.range.address, depth.desc.view_info.base_layer, false)) {
			EXIT("failed to consume HTile clear state\n");
		}
		auto& image = cache.GetImage(depth.image_id);
		EXIT_IF(image_view == nullptr || image.backing.samples != depth.desc.info.samples);
		const auto draw_writes = depth.AttachmentWriteAspects();
		vk::ImageAspectFlags sampled_aspects;
		for (const auto* stage: stages) {
			for (const auto& binding: stage->images) {
				if (binding.image_id != depth.image_id ||
				    binding.desc.type != TextureCache::BindingType::Texture) continue;
				const auto native =
				    std::ranges::find(image.views, binding.image_view, &CachedImageView::view);
				EXIT_IF(native == image.views.end());
				sampled_aspects |= native->info.aspect;
				feedback_aspects |= DepthFeedbackAspects(draw_writes, depth.desc.view_info,
				                                         native->info);
			}
		}
		const auto per_fragment_writes = depth.AttachmentWriteAspects(false);
		if ((feedback_aspects & per_fragment_writes) &&
		    !m_context.GetGraphics().attachment_feedback_loop_enabled) {
			// Without VK_EXT_attachment_feedback_loop_* (the AMD Windows driver), the draw samples
			// the depth target it writes in the GENERAL layout: not defined by Vulkan, but what
			// the hardware does anyway, and better than ending the emulator.
			static std::atomic_bool warned {false};
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				LOGF("Warning: depth attachment feedback loop without host support; "
				     "using the GENERAL layout\n");
			}
		}
		auto layout = depth_attachment_layout(depth);
		if (sampled_aspects & ~DepthReadableAspects(layout)) {
			layout = m_context.GetGraphics().attachment_feedback_loop_enabled
			             ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
			             : vk::ImageLayout::eGeneral;
		}
		// The attachment store writes even when guest depth/stencil tests do not.
		const auto access = vk::AccessFlagBits2::eDepthStencilAttachmentRead |
		                    vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
		image.binding.attachment_layout = layout;
		image.binding.attachment_access = access;
		const auto& view                = depth.desc.view_info;
		image.Transit(layout, access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Handle());
		state.width               = std::min(state.width, depth.desc.info.extent.width);
		state.height              = std::min(state.height, depth.desc.info.extent.height);
		state.num_layers          = std::min(state.num_layers, view.layer_count);
		const auto aspects        = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		auto&      attachment     = state.depth_stencil_attachment;
		attachment.image_view     = image_view;
		attachment.image_layout   = layout;
		attachment.clear_value[0] = std::bit_cast<uint32_t>(depth.depth_clear_value);
		attachment.clear_value[1] = depth.stencil_clear_value;
		attachment.has_depth      = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eDepth);
		attachment.depth_clear    = depth.depth_load_clear_enable;
		attachment.has_stencil    = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eStencil);
		attachment.stencil_clear  = depth.stencil_clear_enable;
	}
	if (color_count == 0 && !depth.image_id) {
		const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
		state.width        = limits.maxFramebufferWidth;
		state.height       = limits.maxFramebufferHeight;
	}
	if (state.num_layers == std::numeric_limits<uint32_t>::max()) {
		state.num_layers = 1;
	}
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.width == std::numeric_limits<uint32_t>::max() ||
	        state.height == std::numeric_limits<uint32_t>::max());
	return state;
}

static uint32_t DrawColorOutputMask(const HW::Context& ctx) {
	const auto& sh_regs     = ctx.GetShaderRegisters();
	const auto  write_mask  = ctx.GetRenderTargetMask() & sh_regs.m_cbShaderMask;
	uint32_t    output_mask = 0;
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		if (sh_regs.target_output_mode[slot] != 0 &&
		    render_target_mask_slot(write_mask, slot) != 0) {
			output_mask |= 1u << slot;
		}
	}
	return output_mask;
}

enum class CbColorMode : uint8_t {
	Disable            = 0,
	Normal             = 1,
	EliminateFastClear = 2,
	Resolve            = 3,
	FmaskDecompress    = 5,
	DccDecompress      = 6,
};

static bool ConsumeMetadataColorOperation(const CommandBuffer& buffer) {
	const auto& ctx  = buffer.GetRegisters();
	const auto  mode = ctx.GetColorControl().mode;
	// These special modes run color-buffer metadata or decompression operations. The shader is a
	// vehicle for that operation, and its exported color must not be applied as a normal draw.
	// Kyty stores expanded Vulkan images rather than compressed guest surfaces, so no equivalent
	// hardware pass is emitted. Tracked DCC clear state is materialized on attachment bind;
	// future CMask/FMask support can consume its state through the same TextureCache path.
	return mode == static_cast<uint8_t>(CbColorMode::EliminateFastClear) ||
	       mode == static_cast<uint8_t>(CbColorMode::FmaskDecompress) ||
	       mode == static_cast<uint8_t>(CbColorMode::DccDecompress);
}

struct DrawEmitInfo {
	int32_t  vertex_offset = 0;
	uint32_t first_vertex  = 0;
	uint32_t first_instance = 0;
};

struct DrawIndexBufferSource {
	uint64_t      address   = 0;
	const void*   host_data = nullptr;
	uint64_t      size      = 0;
	vk::IndexType type      = vk::IndexType::eUint16;
	uint32_t      guest_element_size = 0;
};

struct PreparedIndexBuffer {
	vk::Buffer     buffer = nullptr;
	vk::DeviceSize offset = 0;
	vk::IndexType  type   = vk::IndexType::eUint16;
};

// Flat O(1) byte-size LUT for vertex buffer formats, indexed by the raw
// BufferFormat code. Byte sizes repeat per data format across the 7 number
// formats (UNorm/SNorm/UScaled/SScaled/UInt/SInt/Float share one row stride),
// so a plain table beats the runtime switch + GetFormatComponentType branch
// chain on the per-draw hot path.
constexpr uint32_t kVertexFormatByteSizeCount = 78;
constexpr std::array<uint8_t, kVertexFormatByteSizeCount> kVertexFormatByteSizes = {
    /* 0: kInvalid */ 0,
    /* 1-6: 8-bit x1 */ 1, 1, 1, 1, 1, 1,
    /* 7-13: 16-bit x1 */ 2, 2, 2, 2, 2, 2, 2,
    /* 14-19: 8-bit x2 */ 2, 2, 2, 2, 2, 2,
    /* 20-22: 32-bit x1 */ 4, 4, 4,
    /* 23-29: 16-bit x2 */ 4, 4, 4, 4, 4, 4, 4,
    /* 30-43: packed 32-bit x2 (11_11_10, 10_11_11) */ 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4,
    /* 44-55: packed 32-bit x2 (2_10_10_10, 10_10_10_2) */ 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4,
    /* 56-61: 8-bit x4 */ 4, 4, 4, 4, 4, 4,
    /* 62-64: 32-bit x2 */ 8, 8, 8,
    /* 65-71: 16-bit x4 */ 8, 8, 8, 8, 8, 8, 8,
    /* 72-74: 32-bit x3 */ 12, 12, 12,
    /* 75-77: 32-bit x4 */ 16, 16, 16,
};

[[nodiscard]] uint32_t VertexFormatByteSize(Prospero::BufferFormat format) {
	const auto index = static_cast<uint32_t>(format);
	return index < kVertexFormatByteSizes.size() ? kVertexFormatByteSizes[index] : 0;
}

static uint64_t VertexBufferDescriptorSize(int binding, const ShaderVertexInputInfo& info) {
	const auto& buffer = info.buffers[binding];
	if (buffer.stride != 0 || buffer.num_records == 0) {
		return static_cast<uint64_t>(buffer.stride) * buffer.num_records;
	}

	uint64_t size = 0;
	for (int i = 0; i < info.resources_num; i++) {
		if (info.resources_dst[i].buffer_index != binding) {
			continue;
		}
		const auto& resource = info.resources[i];
		// RDNA2 OOB_SELECT=2 only checks NumRecords != 0. A constant attribute still
		// fetches its entire format; NumRecords is not a byte count in this mode.
		// O(1) format-size LUT instead of the runtime GetFormatInfo() switch chain.
		const uint64_t extent = resource.OutOfBounds() == 2
		                            ? resource.Base48() - buffer.addr +
		                                  VertexFormatByteSize(resource.Format())
		                            : buffer.num_records;
		size = std::max(size, extent);
	}
	return size;
}

struct VertexBufferRange {
	uint64_t                     base_address  = 0;
	uint64_t                     requested_end = 0;
	uint64_t                     acquired_end  = 0;
	std::pair<Buffer*, uint64_t> binding;

	[[nodiscard]] uint64_t RequestedSize() const { return requested_end - base_address; }

	[[nodiscard]] bool operator<(const VertexBufferRange& other) const {
		return base_address < other.base_address;
	}
};

static void SortVertexBufferRanges(VertexBufferRange* ranges, uint32_t count) {
	// Hot path: AcquireVertexBuffers runs per draw (~16k/frame) with only 1-4
	// ranges, so a generic std::sort with iterators/lambda is pure overhead.
	// Inline compare+swap network: 1 compare for 2, 3 for 3, 5 for 4 entries.
	auto swap_if = [&](uint32_t a, uint32_t b) {
		if (ranges[b] < ranges[a]) {
			const auto tmp = ranges[a];
			ranges[a]       = ranges[b];
			ranges[b]       = tmp;
		}
	};
	switch (count) {
		case 0:
		case 1: return;
		case 2: swap_if(0, 1); return;
		case 3:
			swap_if(0, 1);
			swap_if(1, 2);
			swap_if(0, 1);
			return;
		case 4:
			swap_if(0, 1);
			swap_if(2, 3);
			swap_if(0, 2);
			swap_if(1, 3);
			swap_if(1, 2);
			return;
		default: break;
	}
	std::sort(ranges, ranges + count,
	          [](const VertexBufferRange& left, const VertexBufferRange& right) {
		          return left < right;
	          });
}

struct PreparedVertexBuffers {
	static constexpr uint32_t MaxBuffers = ShaderVertexInputInfo::RES_MAX;

	std::array<vk::Buffer, MaxBuffers>     buffers {};
	std::array<vk::DeviceSize, MaxBuffers> offsets {};
	std::array<vk::DeviceSize, MaxBuffers> sizes {};
	uint32_t                               count = 0;
};

// Octagon hot loop: identical vertex buffers rebind thousands of times per frame.
// The guest descriptor fingerprint below lets AcquireVertexBuffers skip the range
// collect/sort/merge work on an exact repeat. The host ObtainBuffer calls are NOT
// skipped: they keep the CPU->GPU upload honest and refresh bindings the cache
// may have evicted or merged since (ForEachUploadRange invariant).
struct VertexBufferFingerprint {
	// Flat Pod copy of the guest descriptors actually consumed: per-slot address,
	// stride and record count plus the attribute mapping (resources/resources_dst).
	// The attribute extent sources (resources_num + resources[]/resources_dst[]) feed
	// VertexBufferDescriptorSize for the stride==0 path, so they are part of the key.
	int                                                   buffers_num   = -1;
	int                                                   resources_num = -1;
	std::array<ShaderVertexInputBuffer, ShaderVertexInputInfo::RES_MAX> buffers {};
	std::array<ShaderBufferResource, ShaderVertexInputInfo::RES_MAX> resources {};
	std::array<ShaderVertexDestination, ShaderVertexInputInfo::RES_MAX> resources_dst {};
	std::array<uint64_t, ShaderVertexInputInfo::RES_MAX> sizes {};
	std::array<VertexBufferRange, ShaderVertexInputInfo::RES_MAX> merged_ranges {};
	uint32_t merged_count = 0;
	bool     valid        = false;
};

[[nodiscard]] static bool MatchVertexBufferFingerprint(const VertexBufferFingerprint& cached,
                                               const ShaderVertexInputInfo& info) {
	if (!cached.valid || cached.buffers_num != info.buffers_num ||
	    cached.resources_num != info.resources_num) {
		return false;
	}
	for (int i = 0; i < info.buffers_num; i++) {
		const auto& a = cached.buffers[i];
		const auto& b = info.buffers[i];
		if (a.addr != b.addr || a.stride != b.stride || a.num_records != b.num_records ||
		    a.fetch_index != b.fetch_index) {
			return false;
		}
	}
	for (int i = 0; i < info.resources_num; i++) {
		const auto& a = cached.resources[i];
		const auto& b = info.resources[i];
		if (a.fields[0] != b.fields[0] || a.fields[1] != b.fields[1] ||
		    a.fields[2] != b.fields[2] || a.fields[3] != b.fields[3]) {
			return false;
		}
		const auto& da = cached.resources_dst[i];
		const auto& db = info.resources_dst[i];
		if (da.register_start != db.register_start || da.registers_num != db.registers_num ||
		    da.attr_id != db.attr_id || da.fetch_index != db.fetch_index ||
		    da.buffer_index != db.buffer_index) {
			return false;
		}
	}
	return true;
}

static PreparedVertexBuffers AcquireVertexBuffers(CommandBuffer&               buffer,
                                                  const ShaderVertexInputInfo& vs_input_info) {
	EXIT_IF(vs_input_info.buffers_num < 0 ||
	        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX);

	// Collect the non-empty guest vertex ranges.
	std::array<uint64_t, ShaderVertexInputInfo::RES_MAX>          sizes {};
	std::array<VertexBufferRange, ShaderVertexInputInfo::RES_MAX> ranges {};
	uint32_t                                                      range_count = 0;
	// Fast path: the merged guest ranges are already known from the previous draw.
	// Sizes are recomputed by VertexBufferDescriptorSize into the fingerprint on the
	// slow path, and the fingerprint match above guarantees they are identical here.
	thread_local VertexBufferFingerprint t_vertex_cache;
	const bool range_hit = MatchVertexBufferFingerprint(t_vertex_cache, vs_input_info);
	// The merged guest ranges: freshly computed on the slow path, replayed on a hit.
	// Host bindings are always re-acquired below (honest CPU->GPU upload + refresh).
	std::array<VertexBufferRange, ShaderVertexInputInfo::RES_MAX> merged_ranges {};
	uint32_t                                                      merged_count = 0;
	if (!range_hit) {
		for (int i = 0; i < vs_input_info.buffers_num; i++) {
			const auto& vertex = vs_input_info.buffers[i];
			const auto  size   = VertexBufferDescriptorSize(i, vs_input_info);
			sizes[i]           = size;
			if (size == 0) {
				continue;
			}
			if (vertex.addr == 0 || size > UINT64_MAX - vertex.addr) {
				EXIT("invalid vertex buffer range: addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
				     vertex.addr, size);
			}
			ranges[range_count++] = {vertex.addr, vertex.addr + size};
		}

		SortVertexBufferRanges(ranges.data(), range_count);

		// Merge overlapping or touching ranges before acquiring host buffers.
		for (uint32_t i = 0; i < range_count; i++) {
			const auto& range = ranges[i];
			if (merged_count != 0 &&
			    merged_ranges[merged_count - 1].requested_end >= range.base_address) {
				merged_ranges[merged_count - 1].requested_end =
				    std::max(merged_ranges[merged_count - 1].requested_end, range.requested_end);
				continue;
			}
			merged_ranges[merged_count++] = {range.base_address, range.requested_end};
		}

		// Publish the slow-path result for the next exact-repeat draw.
		t_vertex_cache.buffers_num   = vs_input_info.buffers_num;
		t_vertex_cache.resources_num = vs_input_info.resources_num;
		for (int i = 0; i < vs_input_info.buffers_num; i++) {
			t_vertex_cache.buffers[i] = vs_input_info.buffers[i];
		}
		for (int i = 0; i < vs_input_info.resources_num; i++) {
			t_vertex_cache.resources[i]     = vs_input_info.resources[i];
			t_vertex_cache.resources_dst[i] = vs_input_info.resources_dst[i];
		}
		for (int i = 0; i < ShaderVertexInputInfo::RES_MAX; i++) {
			t_vertex_cache.sizes[i] = sizes[i];
		}
		for (uint32_t i = 0; i < merged_count; i++) {
			t_vertex_cache.merged_ranges[i] = {merged_ranges[i].base_address,
			                                   merged_ranges[i].requested_end};
		}
		t_vertex_cache.merged_count = merged_count;
		t_vertex_cache.valid        = true;
	} else {
		for (int i = 0; i < ShaderVertexInputInfo::RES_MAX; i++) {
			sizes[i] = t_vertex_cache.sizes[i];
		}
		merged_count = t_vertex_cache.merged_count;
		for (uint32_t i = 0; i < merged_count; i++) {
			merged_ranges[i] = {t_vertex_cache.merged_ranges[i].base_address,
			                    t_vertex_cache.merged_ranges[i].requested_end};
		}
	}

	auto& cache = buffer.GetContext().GetBufferCache();
	for (uint32_t i = 0; i < merged_count; i++) {
		auto& range = merged_ranges[i];
		// PPSA20298
		const auto size =
		    Libs::LibKernel::Memory::ClampRangeSize(range.base_address, range.RequestedSize());
		range.acquired_end = range.base_address + size;
		range.binding      = cache.ObtainBuffer(range.base_address, size, false);
	}

	// Rebuild slot bindings, offsetting non-empty slots into their acquired merged range.
	PreparedVertexBuffers prepared;
	prepared.count         = static_cast<uint32_t>(vs_input_info.buffers_num);
	vk::Buffer null_buffer = nullptr;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = sizes[i];
		if (size == 0) {
			if (null_buffer == nullptr) {
				null_buffer = cache.GetBuffer(NULL_BUFFER_ID).Handle();
			}
			prepared.buffers[i] = null_buffer;
			prepared.offsets[i] = 0;
			continue;
		}

		const auto range = std::find_if(merged_ranges.begin(), merged_ranges.begin() + merged_count,
		                                [&](const VertexBufferRange& value) {
			                                return vertex.addr >= value.base_address &&
			                                       vertex.addr < value.acquired_end;
		                                });
		if (range == merged_ranges.begin() + merged_count) {
			EXIT("vertex buffer address is outside the acquired range: addr=0x%016" PRIx64 "\n",
			     vertex.addr);
		}

		prepared.buffers[i] = range->binding.first->Handle();
		prepared.offsets[i] = range->binding.second + vertex.addr - range->base_address;
		prepared.sizes[i]   = std::min(size, range->acquired_end - vertex.addr);
	}

	return prepared;
}

static void SetDrawDebugPhase(CommandBuffer& buffer, uint64_t submit_id, const DrawCallInfo& draw,
                              uint32_t phase) {
	buffer.SetDebugInfo(static_cast<uint32_t>(draw.debug_op), submit_id, phase, draw.index_count, 0,
	                    draw.instance_count, draw.first_instance);
}

static bool GetDrawTopology(const HW::UserConfig& ucfg, vk::PrimitiveTopology& topology) {

	topology = vk::PrimitiveTopology::ePointList;

	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kNone: return false;
		case Prospero::PrimitiveType::kPointList:
			topology = vk::PrimitiveTopology::ePointList;
			break;
		case Prospero::PrimitiveType::kLineList: topology = vk::PrimitiveTopology::eLineList; break;
		case Prospero::PrimitiveType::kLineStrip:
			topology = vk::PrimitiveTopology::eLineStrip;
			break;
		case Prospero::PrimitiveType::kTriList:
			topology = vk::PrimitiveTopology::eTriangleList;
			break;
		case Prospero::PrimitiveType::kTriFan:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		case Prospero::PrimitiveType::kTriStrip:
			topology = vk::PrimitiveTopology::eTriangleStrip;
			break;
		case Prospero::PrimitiveType::kPatch:
			if (!Config::TessellationEnabled()) return false;
			[[fallthrough]];
		case Prospero::PrimitiveType::kRectList:
		case Prospero::PrimitiveType::kRectListLegacy:
			topology = vk::PrimitiveTopology::ePatchList;
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		default: {
			static std::atomic_bool logged = false;
			if (!logged.exchange(true, std::memory_order_relaxed)) {
				std::printf("Skipping draw with unknown primitive type: %u\n",
				            static_cast<uint32_t>(ucfg.GetPrimType()));
			}
			return false;
		}
	}

	return true;
}

static bool ResolvePrimitiveRestart(const CommandBuffer& buffer,
                                    const DrawIndexBufferSource& source) {
	const auto control = buffer.GetUserConfig().GetPrimitiveResetControl();
	EXIT_NOT_IMPLEMENTED((control & ~0x3u) != 0);
	if ((control & 0x1u) == 0) {
		return false;
	}
	switch (buffer.GetUserConfig().GetPrimType()) {
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip: break;
		default: return false;
	}

	const auto element_size = source.guest_element_size;
	const auto index_mask   = UINT32_MAX >> ((4 - element_size) * 8);
	const auto reset_index  = buffer.GetRegisters().GetPrimitiveResetIndex();
	if ((control & 0x2u) != 0 && (reset_index & ~index_mask) != 0) {
		return false;
	}
	const auto restart_index = reset_index & index_mask;
	if (restart_index == index_mask) {
		// Use native restart; the 8-bit path widens its marker to 0xffff.
		return true;
	}

	// A game can set a custom reset value without using it in the index buffer.
	// Keep restart off in that case; fail if we actually find the value.
	// Scan before preparing draw resources: readback can restart the command buffer.
	EXIT_NOT_IMPLEMENTED(source.address == 0);
	const auto* indices = reinterpret_cast<const uint8_t*>(source.address);
	for (uint64_t offset = 0; offset < source.size; offset += element_size) {
		uint32_t index = 0;
		std::memcpy(&index, indices + offset, element_size);
		EXIT_NOT_IMPLEMENTED(index == restart_index);
	}
	return false;
}

bool DrawUsesSingleSample(std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
                          const HW::AaConfig& aa_config) {
	bool single_sample = std::ranges::all_of(
	    colors, [](const auto& color) { return color.desc.info.samples == 1u; });
	if (depth.image_id) {
		return single_sample && depth.desc.info.samples == 1u;
	}
	return colors.empty() ? render_sample_count(aa_config.msaa_num_samples) == 1u : single_sample;
}

struct ShaderRefreshKey {
	uint64_t es_addr           = 0;
	uint64_t gs_addr           = 0;
	uint64_t ps_addr           = 0;
	uint64_t gs_user_data_addr = 0;
	uint32_t color_output_mask = 0;
	uint32_t shader_stages     = 0;
	uint32_t prim_type         = 0;
	uint32_t gs_user_sgpr      = 0;
	uint32_t ps_user_sgpr      = 0;
	std::array<Prospero::ColorComponentMapping, 8> export_mapping {};
	uint32_t gs_user_sgpr_values[16]                      = {};
	uint32_t ps_user_sgpr_values[16]                      = {};
	bool     ps_active                                    = false;
	bool     indexed                                      = false;
	bool     single_sample                                = false;

	[[nodiscard]] bool operator==(const ShaderRefreshKey&) const noexcept = default;
};

struct ShaderRefreshEntry {
	ShaderRefreshKey                     key;
	PipelineCache::GraphicsPrograms      programs;
	std::array<ShaderVertexInputInfo, 3> vertex_info;
	ShaderPixelInputInfo                 ps_input_info;
	bool                                 valid = false;
};

static void RefreshShaders(CommandBuffer& buffer, const DrawCallInfo& draw,
                           uint32_t color_output_mask, DrawRenderState& state) {
	auto& ctx    = buffer.GetRegisters();
	auto& sh_ctx = buffer.GetShaders();

	const auto& vertex_shader_info = sh_ctx.GetVs();
	const auto& pixel_shader_info  = sh_ctx.GetPs();
	const auto& shader_regs        = ctx.GetShaderRegisters();

	thread_local ShaderRefreshEntry t_refresh;
	ShaderRefreshKey                key {};
	key.es_addr           = vertex_shader_info.es_regs.data_addr;
	key.gs_addr           = vertex_shader_info.gs_regs.data_addr;
	key.ps_addr           = pixel_shader_info.ps_regs.data_addr;
	key.gs_user_data_addr = vertex_shader_info.gs_regs.user_data_addr;
	key.color_output_mask = color_output_mask;
	key.shader_stages     = ctx.GetShaderStages();
	key.prim_type         = static_cast<uint32_t>(buffer.GetUserConfig().GetPrimType());
	key.gs_user_sgpr      = vertex_shader_info.gs_regs.rsrc2.user_sgpr;
	key.ps_user_sgpr      = pixel_shader_info.ps_regs.rsrc2.user_sgpr;
	key.ps_active         = state.ps_active;
	key.indexed           = draw.IsIndexed();
	key.single_sample     = DrawUsesSingleSample(std::span {state.color_info, state.color_count},
	                                            state.depth_info, ctx.GetAaConfig());
	const auto gs_words = std::min<uint32_t>(key.gs_user_sgpr, 16u);
	const auto ps_words = std::min<uint32_t>(key.ps_user_sgpr, 16u);
	if (gs_words != 0) {
		std::memcpy(key.gs_user_sgpr_values, vertex_shader_info.gs_user_sgpr.value,
		            gs_words * sizeof(uint32_t));
	}
	if (ps_words != 0) {
		std::memcpy(key.ps_user_sgpr_values, pixel_shader_info.ps_user_sgpr.value,
		            ps_words * sizeof(uint32_t));
	}

	state.programs      = {};
	state.ps_input_info = {};
	std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX>
	    target_export_mapping {};
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		const auto& rt = ctx.GetRenderTarget(slot);
		if ((color_output_mask & (1u << slot)) != 0 && rt.base.addr != 0) {
			target_export_mapping[slot] =
			    TextureGetRenderTargetFormat(rt.info.format, rt.info.channel_type,
			                                 rt.info.channel_order)
			        .export_mapping;
			key.export_mapping[slot] = target_export_mapping[slot];
		}
	}
	if (t_refresh.valid && t_refresh.key == key) {
		state.programs      = t_refresh.programs;
		state.vertex_info   = t_refresh.vertex_info;
		state.ps_input_info = t_refresh.ps_input_info;
		return;
	}
	auto& pipeline_cache = buffer.GetContext().GetPipelineCache();
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "GetGraphicsPrograms");
	}
	state.programs = pipeline_cache.GetGraphicsPrograms(
	    vertex_shader_info, pixel_shader_info, shader_regs, ctx, buffer.GetUserConfig(),
	    target_export_mapping, state.ps_active, state.vertex_info, state.ps_input_info,
	    DrawUsesSingleSample(std::span {state.color_info, state.color_count}, state.depth_info,
	                         ctx.GetAaConfig()));
	t_refresh.key           = key;
	t_refresh.programs      = state.programs;
	t_refresh.vertex_info   = state.vertex_info;
	t_refresh.ps_input_info = state.ps_input_info;
	t_refresh.valid         = static_cast<bool>(state.programs.vertex[0]) &&
	                  (!state.ps_active || static_cast<bool>(state.programs.pixel));
}

// KYTY_LOG_DRAWS=<first_frame>:<count>: from that frame on, log count draws with their shader
// hashes, counts and whether their programs exist (-1: not resolved, the draw is empty). A census
// of what a scene submits.
static void LogDrawCensusLine(CommandBuffer& buffer, const char* kind, uint32_t count,
                              uint32_t instances, int vs_program, int ps_program, int ps_active) {
	static const auto setting = [] {
		struct Setting {
			uint64_t first_frame = 0;
			uint32_t count       = 0;
		} result;
		if (const char* value = std::getenv("KYTY_LOG_DRAWS"); value != nullptr) {
			char* end          = nullptr;
			result.first_frame = std::strtoull(value, &end, 10);
			if (end != nullptr && *end == ':') {
				result.count = static_cast<uint32_t>(std::strtoul(end + 1, nullptr, 10));
			}
		}
		return result;
	}();
	static std::atomic<uint32_t> used {0};
	if (setting.count == 0) {
		return;
	}
	const auto frame = static_cast<uint64_t>(buffer.GetContext().GetGpu().GetFrameNum());
	// KYTY_LOG_DRAWS_TRIGGER=<file>: also wait until that file exists (checked once per frame), so
	// the census can start at a screen reached by hand.
	static const char*           trigger    = std::getenv("KYTY_LOG_DRAWS_TRIGGER");
	static std::atomic<uint64_t> checked    = UINT64_MAX;
	static std::atomic_bool      triggered  = trigger == nullptr;
	if (!triggered && checked.exchange(frame) != frame) {
		if (FILE* file = std::fopen(trigger, "rb"); file != nullptr) {
			std::fclose(file);
			triggered = true;
			LOGF("Draw census: triggered at frame %" PRIu64 "\n", frame);
		}
	}
	if (!triggered || frame < setting.first_frame ||
	    used.fetch_add(1, std::memory_order_relaxed) >= setting.count) {
		return;
	}
	const auto& sh_ctx = buffer.GetShaders();
	const auto  hash   = [](uint64_t addr) { return addr != 0 ? ShaderDeclaredHash(addr) : 0ull; };
	LOGF("Draw census: frame=%" PRIu64 " %s es=%016" PRIx64 " gs=%016" PRIx64 " ps=%016" PRIx64
	     " count=%u instances=%u vs_program=%d ps_program=%d ps_active=%d\n",
	     frame, kind, hash(sh_ctx.GetVs().es_regs.data_addr),
	     hash(sh_ctx.GetVs().gs_regs.data_addr), hash(sh_ctx.GetPs().ps_regs.data_addr), count,
	     instances, vs_program, ps_program, ps_active);
}

static void LogDrawCensus(CommandBuffer& buffer, const DrawCallInfo& draw,
                          const DrawRenderState& state) {
	LogDrawCensusLine(buffer, draw.Name(), draw.index_count, draw.instance_count,
	                  state.programs.vertex[0] ? 1 : 0, state.programs.pixel ? 1 : 0,
	                  state.ps_active ? 1 : 0);
}

// Diagnostics: the draw parameters of the watched shader's draws in the watched frame (see
// InWatchedShaderFrame); each line precedes that draw's WatchShader bindings.
void RenderExecutor::LogWatchedDraw(const DrawCallInfo& draw, const DrawRenderState& state,
                                    uint32_t index_offset, int32_t base_vertex,
                                    uint32_t first_vertex, int32_t host_vertex_offset,
                                    uint32_t host_first_instance, bool indirect) {
	// Hot path (~16k draws/frame): one env-cached check before touching atomics.
	if (!WatchedShaderEnabled()) {
		return;
	}
	static std::atomic<uint32_t> logged {0};
	const auto*                  program = state.vertex_info[0].stage.program;
	if (program == nullptr ||
	    !InWatchedShaderFrame(program->shader_hash,
	                          m_context.GetGraphics().presented_frames.load(std::memory_order_relaxed),
	                          logged)) {
		return;
	}
	LOGF("WatchShader draw: %s count=%u instances=%u first_instance=%u index_offset=%u "
	     "base_vertex=%d first_vertex=%u host_vertex_offset=%d host_first_instance=%u "
	     "indirect=%d\n",
	     draw.Name(), draw.index_count, draw.instance_count, draw.first_instance, index_offset,
	     base_vertex, first_vertex, host_vertex_offset, host_first_instance, indirect ? 1 : 0);
}

bool RenderExecutor::PrepareDrawRenderState(CommandBuffer& buffer, const DrawCallInfo& draw,
                                            uint32_t            render_target_slice_offset,
	                                        DrawRenderState& state) {
	const auto& shader_regs       = buffer.GetRegisters().GetShaderRegisters();
	const auto  color_output_mask = DrawColorOutputMask(buffer.GetRegisters());
	state.ps_active = buffer.GetShaders().GetPs().ps_regs.data_addr != 0 &&
	                  (color_output_mask != 0 ||
	                   PixelShaderHasDepthOrCoverageSideEffects(shader_regs));
	RefreshShaders(buffer, draw, color_output_mask, state);
	LogDrawCensus(buffer, draw, state);
	if (!state.programs.vertex[0] || (state.ps_active && !state.programs.pixel)) {
		return false;
	}
	uint32_t mrt_mask = 0;
	if (state.ps_active) {
		for (const auto& output: state.ps_input_info.stage.program->info.outputs) {
			if (output.kind == ShaderRecompiler::IR::StageOutputKind::Mrt) {
				mrt_mask |= 1u << output.index;
			}
		}
	}
	mrt_mask &= color_output_mask;
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "ResolveRenderColorTarget");
	}
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		if ((mrt_mask & (1u << slot)) != 0) {
			ResolveRenderColorTarget(buffer, state.color_info[state.color_count],
			                         render_target_slice_offset, slot);
			if (state.color_info[state.color_count].image_id) {
				state.color_count++;
			}
		}
	}
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "ResolveRenderDepthTarget");
	}
	ResolveRenderDepthTarget(buffer, state.depth_info);

	if (state.color_count == 0 && !state.depth_info.image_id && !state.ps_active) {
		LogFramebufferSkip(draw.Name(), state.color_info[0], state.depth_info, buffer,
		                   draw.index_count, 0);
		return false;
	}
	// Shader outputs identify active attachments; finalize centroid after resolving them.
	// Bypass the thread_local shader cache here: the re-refresh is rare (a single-sample flip
	// after targets resolve), and the key already holds the pre-resolve value.
	if (state.ps_active &&
	    state.ps_input_info.ps_single_sample !=
	        DrawUsesSingleSample(std::span {state.color_info, state.color_count}, state.depth_info,
	                             buffer.GetRegisters().GetAaConfig())) {
		state.programs      = {};
		state.ps_input_info = {};
		std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX>
		    target_export_mapping {};
		for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
			const auto& rt = buffer.GetRegisters().GetRenderTarget(slot);
			if ((color_output_mask & (1u << slot)) != 0 && rt.base.addr != 0) {
				target_export_mapping[slot] =
				    TextureGetRenderTargetFormat(rt.info.format, rt.info.channel_type,
				                                 rt.info.channel_order)
				        .export_mapping;
			}
		}
		state.programs = buffer.GetContext().GetPipelineCache().GetGraphicsPrograms(
		    buffer.GetShaders().GetVs(), buffer.GetShaders().GetPs(),
		    buffer.GetRegisters().GetShaderRegisters(), buffer.GetRegisters(),
		    buffer.GetUserConfig(), target_export_mapping, state.ps_active, state.vertex_info,
		    state.ps_input_info,
		    DrawUsesSingleSample(std::span {state.color_info, state.color_count},
		                         state.depth_info, buffer.GetRegisters().GetAaConfig()));
	}

	return true;
}

static PreparedIndexBuffer PrepareIndexBuffer(CommandBuffer&               buffer,
                                              const DrawIndexBufferSource& source) {
	PreparedIndexBuffer prepared;
	if (source.size == 0) {
		return prepared;
	}
	prepared.type = source.type;
	// Fast path: octagon frames rebind the same guest index buffer thousands
	// of times. When addr/size/type repeat, reuse the resolved host binding
	// and skip FindBuffer/ObtainBuffer. host_data uploads bypass the cache.
	// NOTE: the buffer cache may evict or merge host allocations between
	// draws, so the cached handle is only a hint: callers re-validate via
	// BindIndexBufferCached + the cache's own identity below on mismatch.
	struct IndexCacheEntry {
		uint64_t      address = 0;
		uint64_t      size    = 0;
		vk::IndexType type    = vk::IndexType::eUint16;
		vk::Buffer    buffer  = nullptr;
		vk::DeviceSize offset = 0;
		bool          valid   = false;
	};
	thread_local IndexCacheEntry t_index_cache;
	const bool index_cache_hit =
	    t_index_cache.valid && source.host_data == nullptr && t_index_cache.address == source.address &&
	    t_index_cache.size == source.size && t_index_cache.type == source.type &&
	    t_index_cache.buffer != nullptr;
	if (index_cache_hit) {
		prepared.buffer = t_index_cache.buffer;
		prepared.offset = t_index_cache.offset;
		return prepared;
	}
	if (source.host_data != nullptr) {
		auto& stream = buffer.GetContext().GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
		prepared.offset = stream.Copy(source.host_data, source.size, 16);
		prepared.buffer = stream.Handle();
	} else {
		auto [buffer_ptr, offset] =
		    buffer.GetContext().GetBufferCache().ObtainBuffer(source.address, source.size, false);
		prepared.buffer = buffer_ptr->Handle();
		prepared.offset = offset;
	}
	if (source.host_data == nullptr) {
		t_index_cache.address = source.address;
		t_index_cache.size    = source.size;
		t_index_cache.type    = source.type;
		t_index_cache.buffer  = prepared.buffer;
		t_index_cache.offset  = prepared.offset;
		t_index_cache.valid   = true;
	}
	return prepared;
}

static void CommitVertexBuffers(CommandBuffer& buffer, vk::CommandBuffer vk_buffer,
                                const PreparedVertexBuffers& prepared) {
	for (uint32_t i = 0; i < prepared.count; i++) {
		EXIT_IF(prepared.buffers[i] == nullptr);
	}
	if (prepared.count != 0) {
		// Guest descriptor bounds must survive allocation merging in the cache. Frostbite rebinds
		// identical buffers thousands of times per frame; skip the redundant driver call.
		if (buffer.BindVertexBuffersCached(0, prepared.count, prepared.buffers.data(),
		                                   prepared.offsets.data(), prepared.sizes.data())) {
			vk_buffer.bindVertexBuffers2(0, prepared.count, prepared.buffers.data(),
			                             prepared.offsets.data(), prepared.sizes.data(), nullptr);
		}
	}
}

static void CommitIndexBuffer(CommandBuffer& buffer, vk::CommandBuffer vk_buffer,
                             const PreparedIndexBuffer& prepared) {
	if (prepared.buffer == nullptr) {
		return;
	}
	if (buffer.BindIndexBufferCached(prepared.buffer, prepared.offset, prepared.type)) {
		vk_buffer.bindIndexBuffer(prepared.buffer, prepared.offset, prepared.type);
	}
}

[[nodiscard]] static bool DrawScissorIsEmpty(const CommandBuffer& buffer,
                                      const DrawRenderState& state) {
	// Zero-area scissor cull: the scissor test discards 100% of fragments, so
	// skip command-buffer emission entirely. Mirrors SetGraphicsDynamicParams:
	// slot 0 unless the VS can emit a viewport index, clamped to the draw's
	// framebuffer extent. Uses the resolved color/depth extents (mip-aware)
	// rather than re-resolving targets. Rasterizer-discard-equivalent draws
	// with no color/depth work never reach here (PrepareDrawRenderState).
	const auto& ctx = buffer.GetRegisters();
	const auto& vp  = ctx.GetScreenViewport();
	const auto& outputs =
	    state.vertex_info[0].stage.program->info.outputs;
	const bool indexed_viewports =
	    std::any_of(outputs.begin(), outputs.end(), [](const auto& output) {
		    return output.kind == ShaderRecompiler::IR::StageOutputKind::ViewportIndex;
	    });
	const uint32_t slot_count =
	    indexed_viewports ? static_cast<uint32_t>(std::size(HW::ScreenViewport {}.viewports)) : 1u;
	vk::Extent2D framebuffer_extent {0, 0};
	for (uint32_t i = 0; i < state.color_count; i++) {
		const auto extent = state.color_info[i].Extent();
		framebuffer_extent.width =
		    framebuffer_extent.width == 0 ? extent.width : std::min(framebuffer_extent.width, extent.width);
		framebuffer_extent.height =
		    framebuffer_extent.height == 0 ? extent.height : std::min(framebuffer_extent.height, extent.height);
	}
	if (state.depth_info.image_id &&
	    state.depth_info.desc.view_info.format != vk::Format::eUndefined) {
		const auto width = std::max(state.depth_info.desc.info.extent.width, 1u);
		const auto height = std::max(state.depth_info.desc.info.extent.height, 1u);
		framebuffer_extent.width =
		    framebuffer_extent.width == 0 ? width : std::min(framebuffer_extent.width, width);
		framebuffer_extent.height =
		    framebuffer_extent.height == 0 ? height : std::min(framebuffer_extent.height, height);
	}
	if (framebuffer_extent.width == 0 || framebuffer_extent.height == 0) {
		return false;
	}
	for (uint32_t i = 0; i < slot_count; i++) {
		const auto scissor = calc_final_scissor(vp, ctx.GetScanModeControl(), framebuffer_extent, i);
		if (scissor.right > scissor.left && scissor.bottom > scissor.top) {
			return false;
		}
	}
	return true;
}

static void LogDrawStateIfNeeded(const CommandBuffer& buffer, const DrawCallInfo& draw,
	                             const DrawRenderState& state, uint32_t index_type_and_size,
                                 const void* index_addr) {
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	if (!draw.IsIndexed() && !Prospero::IsRectList(buffer.GetUserConfig().GetPrimType())) {
		return;
	}

	if (state.ps_active) {
		LogDrawTargetState(draw.Name(), state.color_info[0], state.depth_info, buffer,
		                   state.ps_input_info, draw.index_count, 0);
	}
	LogDrawInputState(buffer, state.color_info[0], state.vertex_info[0], index_type_and_size,
	                  draw.index_count, index_addr);
}

static void EmitDrawPrimitives(const HW::UserConfig& ucfg, vk::CommandBuffer vk_buffer,
                               const DrawCallInfo& draw, const DrawEmitInfo& emit) {
	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kPointList:
		case Prospero::PrimitiveType::kLineList:
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriList:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip:
		case Prospero::PrimitiveType::kRectList:
		case Prospero::PrimitiveType::kRectListLegacy:
		case Prospero::PrimitiveType::kPatch:
			if (draw.IsIndexed()) {
				vk_buffer.drawIndexed(draw.index_count, draw.instance_count, 0, emit.vertex_offset,
				                      emit.first_instance);
			} else {
				vk_buffer.draw(draw.index_count, draw.instance_count, emit.first_vertex,
				               emit.first_instance);
			}
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			EXIT_NOT_IMPLEMENTED((draw.index_count & 0x3u) != 0);
			for (uint32_t i = 0; i < draw.index_count; i += 4) {
				if (draw.IsIndexed()) {
					vk_buffer.drawIndexed(4, draw.instance_count, i, emit.vertex_offset,
					                      emit.first_instance);
				} else {
					vk_buffer.draw(4, draw.instance_count, i + emit.first_vertex,
					               emit.first_instance);
				}
			}
			break;
		default: EXIT("unknown primitive type: %u\n", static_cast<uint32_t>(ucfg.GetPrimType()));
	}
}

void RenderExecutor::ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer,
                                         const DrawCallInfo& draw, DrawRenderState& state,
                                         vk::PrimitiveTopology topology, const DrawEmitInfo& emit,
                                         const DrawIndexBufferSource& index_source,
	                                     bool primitive_restart_enable) {
	// Degenerate guard: a zero-count draw must never emit command-buffer work.
	// Callers early-out on args, but indirect-sized state can still collapse
	// here; return before any barrier, binding or pipeline work. This skips
	// vkCmdBindPipeline/vkCmdBindDescriptorSets only because no draw executes.
	if (draw.index_count == 0 || draw.instance_count == 0) {
		return;
	}
	auto& ucfg = buffer.GetUserConfig();
	// A draw observes every prior compute write, so the single coalesced
	// compute barrier (if any) is emitted once here instead of once per
	// dispatch. Non-compute work between dispatches funnels through this path.
	FlushPendingComputeBarrier();
	const auto vertex_stages =
	    std::span {state.vertex_info.data(), state.programs.VertexStageCount()};
	const bool mesh_active = state.vertex_info[0].stage.program->stage == ShaderType::Mesh;
	uint32_t   mesh_groups = 0;
	if (mesh_active) {
		const auto& mesh = state.vertex_info[0].mesh;
		static std::atomic_bool restart_warned = false;
		if (primitive_restart_enable && !restart_warned.exchange(true, std::memory_order_relaxed)) {
			std::printf("Warning: primitive restart is not implemented for mesh shaders; "
			            "continuing draw (primitive=%u indexed=%u)\n",
			            static_cast<uint32_t>(ucfg.GetPrimType()), draw.IsIndexed());
		}
		if (mesh.primitives_per_group == 0) {
			EXIT("unsupported mesh draw: primitive=%u indexed=%u restart=%u\n",
			     static_cast<uint32_t>(ucfg.GetPrimType()), draw.IsIndexed(), primitive_restart_enable);
		}
		const auto primitives = mesh.InputPrimitiveCount(draw.index_count);
		if (primitives == 0 || draw.instance_count == 0) {
			return;
		}
		mesh_groups        = (primitives - 1u) / mesh.primitives_per_group + 1u;
		const auto& limits = m_context.GetGraphics().mesh_shader_properties;
		if (mesh_groups > limits.maxMeshWorkGroupCount[0] ||
		    draw.instance_count > limits.maxMeshWorkGroupCount[1] ||
		    static_cast<uint64_t>(mesh_groups) * draw.instance_count >
		        limits.maxMeshWorkGroupTotalCount) {
			EXIT("mesh draw exceeds host workgroup limits: %ux%u\n", mesh_groups,
			     draw.instance_count);
		}
	}

	if (mesh_active && draw.IsIndexed()) {
		// Register the original guest indices for shader reads; PrepareGraphicsBindings
		// synchronizes registered BDA ranges before any draw commands are committed.
		(void)m_context.GetBufferCache().FindBuffer(
		    index_source.address, static_cast<uint64_t>(draw.index_count) *
		                              index_source.guest_element_size);
	}
	LogDrawPhase(draw.Name(), "PrepareBindings");
	auto&                            bindings = m_graphics_bindings;
	std::array<PreparedBindings*, 4> descriptor_stages {};
	uint32_t                         stage_count = 0;
	for (uint32_t i = 0; i < vertex_stages.size(); i++) {
		PrepareBindings(state.vertex_info[i].stage, bindings.vertex[i]);
		descriptor_stages[stage_count++] = &bindings.vertex[i];
	}
	// Depth-only shadow cascade fast path: no color targets and no active pixel
	// shader. Only vertex position + depth state are consumed, so skip the
	// pixel-stage material/texture descriptor binding and the color-blend
	// setup below (static pipeline state already disables blending).
	const bool depth_only = state.color_count == 0 && !state.ps_active;
	if (state.ps_active) {
		if (!bindings.pixel) bindings.pixel.emplace();
		PrepareBindings(state.ps_input_info.stage, *bindings.pixel);
		descriptor_stages[stage_count++] = &*bindings.pixel;
	}
	const auto stages = std::span {descriptor_stages.data(), stage_count};
	const uint64_t bindings_prepare_us = CommandProcessor::TelemetryNowUs();
	PrepareGraphicsBindings(stages, std::span {state.color_info, state.color_count});
	if (GuestGpu::IsGpuThread()) {
		m_context.GetGpu().GraphicsProcessor().TelemetryAddDrawBindings(
		    CommandProcessor::TelemetryNowUs() - bindings_prepare_us);
	}
	// Zero-area scissor cull: geometry is 100% clipped, so skip buffer,
	// pipeline and command-buffer work entirely. Memory-writing draws cannot
	// be skipped: others may read what they write.
	const auto writes_memory = [](const ShaderStageRuntime& runtime) {
		return HasShaderBufferWrites(runtime) ||
		       std::ranges::any_of(runtime.program->info.images, [](const auto& image) {
			       return image.written &&
			              image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Storage;
		       });
	};
	bool may_defer = !(state.ps_active && writes_memory(state.ps_input_info.stage));
	for (const auto& stage: vertex_stages) {
		may_defer = may_defer && !writes_memory(stage.stage);
	}
	bool writes_any_memory = !may_defer;
	if (!writes_any_memory && !mesh_active && DrawScissorIsEmpty(buffer, state)) {
		ResetBindings();
		return;
	}
	PreparedVertexBuffers vertex_bindings;
	PreparedIndexBuffer   index_binding;
	uint64_t              vertex_us = 0;
	if (!mesh_active) {
		LogDrawPhase(draw.Name(), "PrepareVertexBuffers");
		const uint64_t vertex_acquire_us = CommandProcessor::TelemetryNowUs();
		vertex_bindings = AcquireVertexBuffers(buffer, state.vertex_info[0]);
		// Index-buffer fast path: hundreds of crowd draws reuse one guest
		// buffer. Reuse only on exact addr/size/type with the same host
		// owner and no CPU/GPU modification since (ForEachUploadRange
		// invariant); host_data uploads and mesh draws bypass it.
		bool index_hit = false;
		if (index_source.host_data == nullptr && index_source.address != 0 && index_source.size != 0 &&
		    m_draw_index_cache.valid && m_draw_index_cache.buffer != nullptr &&
		    m_draw_index_cache.address == index_source.address &&
		    m_draw_index_cache.size == index_source.size &&
		    m_draw_index_cache.type == index_source.type) {
			auto&       cache = m_context.GetBufferCache();
			const auto  owner = cache.FindBuffer(index_source.address, index_source.size);
			// BufferId is truthy only for a live owner; additionally require
			// the owner to cover the range so merge/evict cannot alias it.
			if (owner && cache.GetBuffer(owner).IsInBounds(index_source.address, index_source.size) &&
			    cache.GetBuffer(owner).Handle() == m_draw_index_cache.buffer &&
			    !cache.HasGpuDirtyBytes(index_source.address, index_source.size)) {
				index_binding.buffer = m_draw_index_cache.buffer;
				index_binding.offset = m_draw_index_cache.offset;
				index_binding.type   = m_draw_index_cache.type;
				index_hit            = true;
			}
		}
		if (!index_hit) {
			index_binding = PrepareIndexBuffer(buffer, index_source);
			if (index_source.host_data == nullptr && index_source.address != 0 && index_source.size != 0 &&
			    index_binding.buffer != nullptr) {
				m_draw_index_cache.address = index_source.address;
				m_draw_index_cache.size    = index_source.size;
				m_draw_index_cache.type    = index_source.type;
				m_draw_index_cache.buffer  = index_binding.buffer;
				m_draw_index_cache.offset  = index_binding.offset;
				m_draw_index_cache.valid   = true;
			} else {
				m_draw_index_cache.valid = false;
			}
		}
		vertex_us += CommandProcessor::TelemetryNowUs() - vertex_acquire_us;
	}
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "CreatePipeline");
	}
	// A draw that writes no memory may be skipped while its new pipeline compiles in the
	// background; its render targets are not touched. One that writes buffers or storage images
	// waits for the pipeline, as others may read what it writes. Identical
	// shader programs across consecutive crowd/stadium draws usually mean an
	// identical pipeline: the L1 MRU inside TryGetGraphicsPipeline then hits
	// on the full key without hashing or locking, so no caller-side pipeline
	// cache is kept here (its key would duplicate GraphicsPipelineKey and
	// risk stale reuse). vkCmdBindPipeline stays unconditional below.
	const uint64_t pipe_start_us = CommandProcessor::TelemetryNowUs();
	auto* deferred_pipeline = m_context.GetPipelineCache().TryGetGraphicsPipeline(
	    std::span {state.color_info, state.color_count}, state.depth_info, vertex_stages, buffer,
	    state.ps_active ? &state.ps_input_info : nullptr, topology, primitive_restart_enable,
	    state.programs, may_defer);
	if (GuestGpu::IsGpuThread()) {
		m_context.GetGpu().GraphicsProcessor().TelemetryAddDrawPipe(
		    CommandProcessor::TelemetryNowUs() - pipe_start_us);
	}
	if (deferred_pipeline == nullptr) {
		return;
	}
	auto& pipeline = *deferred_pipeline;
	vk::ImageAspectFlags feedback_aspects;
	const auto rendering =
	    AcquireRenderTargets(buffer, state.color_info, state.color_count, state.depth_info,
	                         feedback_aspects, stages);

	// Resource preparation above may synchronously finish and restart the scheduler. From this
	// point onward, every operation targets the current command buffer and cannot touch guest
	// memory.
	auto vk_buffer = buffer.Handle();
	SetDrawDebugPhase(buffer, submit_id, draw, draw.IsIndexed() ? 0x100u : 0x200u);
	if (!mesh_active) {
		const uint64_t vertex_commit_us = CommandProcessor::TelemetryNowUs();
		CommitVertexBuffers(buffer, vk_buffer, vertex_bindings);
		vertex_us += CommandProcessor::TelemetryNowUs() - vertex_commit_us;
	}
	if (state.ps_active && !draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x300u);
	}
	const uint64_t bindings_commit_us = CommandProcessor::TelemetryNowUs();
	CommitBindings(buffer, vk::PipelineBindPoint::eGraphics, pipeline, stages);
	if (GuestGpu::IsGpuThread()) {
		m_context.GetGpu().GraphicsProcessor().TelemetryAddDrawBindings(
		    CommandProcessor::TelemetryNowUs() - bindings_commit_us);
	}
	if (mesh_active) {
		const uint32_t draw_data[] {
		    draw.index_count,
		    draw.IsIndexed() ? static_cast<uint32_t>(emit.vertex_offset) : emit.first_vertex,
		    emit.first_instance, index_source.guest_element_size,
		    static_cast<uint32_t>(index_source.address),
		    static_cast<uint32_t>(index_source.address >> 32u)};
		static_assert(std::size(draw_data) == ShaderRecompiler::IR::PushData::MeshDrawDwordCount);
		vk_buffer.pushConstants(pipeline.pipeline_layout,
		                        vk::ShaderStageFlagBits::eMeshEXT |
		                            vk::ShaderStageFlagBits::eFragment,
		                        0, sizeof(draw_data), draw_data);
	} else {
		const uint64_t index_commit_us = CommandProcessor::TelemetryNowUs();
		CommitIndexBuffer(buffer, vk_buffer, index_binding);
		vertex_us += CommandProcessor::TelemetryNowUs() - index_commit_us;
	}
	if (GuestGpu::IsGpuThread()) {
		m_context.GetGpu().GraphicsProcessor().TelemetryAddDrawVertex(vertex_us);
	}

	SetGraphicsDynamicParams(buffer, vk_buffer, vertex_stages.back(), state.depth_info, rendering,
	                         depth_only, state.ps_active);
	if (m_context.GetGraphics().attachment_feedback_loop_enabled) {
		vk_buffer.setAttachmentFeedbackLoopEnableEXT(feedback_aspects);
	}

	LogDrawPhase(draw.Name(), "BeginRendering");
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x400u);
	}
	m_context.GetCommandScheduler().BeginRendering(rendering);
	// The pipeline is bound unconditionally: internal clears/warmup can rebind
	// pipeline state behind any cache, so a skipped bind would draw with a stale
	// pipeline (GPU page fault).
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.pipeline);
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x500u);
	}
	GpuTiming::Before(m_context, buffer);
	const uint64_t emit_start_us = CommandProcessor::TelemetryNowUs();
	if (mesh_active) {
		vk_buffer.drawMeshTasksEXT(mesh_groups, draw.instance_count, 1);
	} else {
		EmitDrawPrimitives(ucfg, vk_buffer, draw, emit);
	}
	if (GuestGpu::IsGpuThread()) {
		m_context.GetGpu().GraphicsProcessor().TelemetryAddDrawEmit(
		    CommandProcessor::TelemetryNowUs() - emit_start_us);
	}
	GpuTiming::After(m_context, buffer,
	                 state.ps_active ? state.ps_input_info.stage.program->shader_hash
	                                 : vertex_stages.back().stage.program->shader_hash,
	                 GpuTiming::Draw);

	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x600u);
	}
	vk::PipelineStageFlags shader_write_stages = {};
	for (const auto& stage: vertex_stages) {
		if (HasShaderBufferWrites(stage.stage)) {
			shader_write_stages |= ShaderPipelineStages(NativeShaderStage(stage.logical_stage));
		}
	}
	if (state.ps_active && HasShaderBufferWrites(state.ps_input_info.stage)) {
		shader_write_stages |= vk::PipelineStageFlagBits::eFragmentShader;
	}
	if (shader_write_stages) {
		m_context.GetCommandScheduler().EndRendering();
		ShaderWriteBarrier(vk_buffer, shader_write_stages);
	}
	LogDrawPhase(draw.Name(), "DrawComplete");
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x700u);
	}
}

void RenderExecutor::DrawIndex(uint64_t submit_id, CommandBuffer& buffer,
                               const DrawIndexArgs& args) {
	// Degenerate early-out first: culled passes often submit zero indices.
	// Instant return before any state setup, buffer requests or
	// ExecutePreparedDraw (validation above is kept, scheduler/lock work is
	// skipped).
	EXIT_IF(buffer.IsInvalid());
	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	if (args.index_count == 0 || args.instance_count == 0) {
		return;
	}
	KYTY_PROFILER_FUNCTION();

	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndex), submit_id,
	                    args.index_count, 0, 1, args.instance_count,
	                    reinterpret_cast<uint64_t>(args.index_addr));

	Common::LockGuard lock(m_context.GetMutex());
	if (args.index_count == 0 || args.instance_count == 0) {
		LogDrawCensusLine(buffer, "DrawIndex-empty", args.index_count, args.instance_count, -1, -1,
		                  -1);
		return;
	}

	if (ConsumeMetadataColorOperation(buffer) || DepthStencilCopy(buffer) ||
	    ResolveColorTargets(buffer, args.render_target_slice_offset)) {
		ResetBindings();
		return;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		return;
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndex():Shader:\n");
		uc_print("GraphicsRenderDrawIndex():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndex():Parameters:\n"
		     "\t index_type_and_size = 0x%08" PRIx32 "\n"
		     "\t index_count         = 0x%08" PRIx32 "\n"
		     "\t index_addr          = 0x%016" PRIx64 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t base_vertex         = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     args.index_type_and_size, args.index_count,
		     reinterpret_cast<uint64_t>(args.index_addr), args.instance_count,
		     static_cast<uint32_t>(args.base_vertex), args.first_instance);
	}

	uc_check(ucfg);

	hw_check(buffer);

	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	if (!GetDrawTopology(ucfg, topology)) {
		return;
	}

	DrawIndexBufferSource index_source {};
	index_source.address = reinterpret_cast<uint64_t>(args.index_addr);
	switch (static_cast<Prospero::IndexType>(args.index_type_and_size)) {
		case Prospero::IndexType::kIndex16:
			index_source.type               = vk::IndexType::eUint16;
			index_source.guest_element_size = 2;
			break;
		case Prospero::IndexType::kIndex32:
			index_source.type               = vk::IndexType::eUint32;
			index_source.guest_element_size = 4;
			break;
		case Prospero::IndexType::kIndex8:
			index_source.guest_element_size = 1;
			break;
		default: EXIT("unknown index_type_and_size: %u\n", args.index_type_and_size);
	}
	index_source.size = static_cast<uint64_t>(args.index_count) * index_source.guest_element_size;
	const bool primitive_restart = ResolvePrimitiveRestart(buffer, index_source);

	static thread_local std::vector<uint16_t> t_expanded_indices;
	if (index_source.guest_element_size == 1) {
		EXIT_NOT_IMPLEMENTED(args.index_addr == nullptr);
		const auto* src = static_cast<const uint8_t*>(args.index_addr);
		// Hot path: reuse the thread-local buffer across draws; resize only grows.
		t_expanded_indices.resize(args.index_count);
		for (uint32_t i = 0; i < args.index_count; i++) {
			t_expanded_indices[i] = primitive_restart && src[i] == 0xffu ? 0xffffu : src[i];
		}
		index_source.host_data = t_expanded_indices.data();
		index_source.size      = t_expanded_indices.size() * sizeof(uint16_t);
	}

	const DrawCallInfo draw {CommandBufferDebugOp::DrawIndex, args.index_count,
	                        args.instance_count, args.first_instance};
	DrawRenderState state {};
	if (!PrepareDrawRenderState(buffer, draw, args.render_target_slice_offset, state)) {
		ResetBindings();
		return;
	}

	LogDrawStateIfNeeded(buffer, draw, state, args.index_type_and_size,
	                     args.index_addr);

	const bool indirect = args.offset_source == DrawOffsetSource::IndirectArgs;
	const auto [vertex_offset, instance_offset] =
	    indirect ? std::pair<int32_t, uint32_t> {0, args.first_instance}
	             : ResolveDrawOffsets(ucfg.GetIndexOffset(), state.vertex_info[0]);

	DrawEmitInfo emit {};
	emit.vertex_offset  = vertex_offset + args.base_vertex;
	emit.first_instance = instance_offset;
	LogWatchedDraw(draw, state, ucfg.GetIndexOffset(), args.base_vertex, 0, emit.vertex_offset,
	               emit.first_instance, indirect);

	ExecutePreparedDraw(submit_id, buffer, draw, state, topology, emit, index_source,
	                    primitive_restart);
	ResetBindings();
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void RenderExecutor::DrawAuto(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args) {
	// Degenerate early-out first, mirroring DrawIndex: instant return before
	// PopPendingOperations/lock/state setup, buffer requests or
	// ExecutePreparedDraw (validation above is kept).
	EXIT_IF(buffer.IsInvalid());
	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	if (args.vertex_count == 0 || args.instance_count == 0) {
		return;
	}
	KYTY_PROFILER_FUNCTION();

	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndexAuto), submit_id,
	                    args.vertex_count, 0, args.first_vertex, args.instance_count,
	                    args.first_instance);

	Common::LockGuard lock(m_context.GetMutex());
	if (args.vertex_count == 0 || args.instance_count == 0) {
		LogDrawCensusLine(buffer, "DrawIndexAuto-empty", args.vertex_count, args.instance_count, -1,
		                  -1, -1);
		return;
	}

	if (ConsumeMetadataColorOperation(buffer) || DepthStencilCopy(buffer) ||
	    ResolveColorTargets(buffer, args.render_target_slice_offset)) {
		ResetBindings();
		return;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		return;
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndexAuto():Shader:\n");
		uc_print("GraphicsRenderDrawIndexAuto():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndexAuto():Parameters:\n"
		     "\t vertex_count        = 0x%08" PRIx32 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t first_vertex        = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     args.vertex_count, args.instance_count, args.first_vertex, args.first_instance);
	}

	uc_check(ucfg);

	hw_check(buffer);

	const DrawCallInfo draw {CommandBufferDebugOp::DrawIndexAuto,
	                         args.vertex_count, args.instance_count, args.first_instance};

	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	if (!GetDrawTopology(ucfg, topology)) {
		ResetBindings();
		return;
	}
	DrawRenderState state {};
	if (!PrepareDrawRenderState(buffer, draw, args.render_target_slice_offset, state)) {
		ResetBindings();
		return;
	}

	const bool rect_list = Prospero::IsRectList(ucfg.GetPrimType());
	if (rect_list && state.vertex_info[0].buffers_num == 0 &&
	    state.vertex_info[0].stage.program->param_export_mask == 0 &&
	    state.ps_input_info.input_num != 0) {
		if (graphics_debug_dump_enabled()) {
			LOGF("DrawIndexAuto: skipping rect-list draw with no VS param exports and PS inputs: "
			     "ps_inputs=%u ps=0x%016" PRIx64 " es=0x%016" PRIx64 " gs=0x%016" PRIx64 "\n",
			     state.ps_input_info.input_num, sh_ctx.GetPs().ps_regs.data_addr,
			     sh_ctx.GetVs().es_regs.data_addr, sh_ctx.GetVs().gs_regs.data_addr);
		}
		ResetBindings();
		return;
	}

	LogDrawStateIfNeeded(buffer, draw, state, 0, nullptr);

	const bool indirect = args.offset_source == DrawOffsetSource::IndirectArgs;
	const auto [vertex_offset, instance_offset] =
	    indirect ? std::pair<int32_t, uint32_t> {0, args.first_instance}
	             : ResolveDrawOffsets(ucfg.GetIndexOffset(), state.vertex_info[0]);
	DrawEmitInfo emit {};
	emit.first_vertex = static_cast<uint32_t>(vertex_offset + static_cast<int32_t>(args.first_vertex));
	emit.first_instance = instance_offset;
	LogWatchedDraw(draw, state, ucfg.GetIndexOffset(), 0, args.first_vertex,
	               static_cast<int32_t>(emit.first_vertex), emit.first_instance, indirect);

	DrawIndexBufferSource index_source {};
	ExecutePreparedDraw(submit_id, buffer, draw, state, topology, emit, index_source, false);
	ResetBindings();
}

bool RenderExecutor::ResolveColorTargets(CommandBuffer& buffer, uint32_t render_target_slice_offset) {
	const auto& hw = buffer.GetRegisters();
	if (hw.GetColorControl().mode != 3) {
		return false;
	}

	const auto& src_rt = hw.GetRenderTarget(0);
	const auto& dst_rt = hw.GetRenderTarget(1);
	if (src_rt.base.addr == 0 || dst_rt.base.addr == 0) {
		return false;
	}

	RenderColorInfo src {};
	RenderColorInfo dst {};
	ResolveRenderColorTarget(buffer, src, render_target_slice_offset, 0, true, true);
	ResolveRenderColorTarget(buffer, dst, render_target_slice_offset, 1, true, true);
	if (!src.image_id || !dst.image_id) {
		return false;
	}
	if (src.desc.info.data.address == dst.desc.info.data.address &&
	    src.guest_mip_level == dst.guest_mip_level &&
	    src.guest_array_layer == dst.guest_array_layer) {
		return true;
	}

	auto& cache = m_context.GetTextureCache();
	cache.MarkGpuWritten(dst.image_id);
	auto& source      = cache.GetImage(src.image_id);
	auto& destination = cache.GetImage(dst.image_id);
	destination.Resolve(source, {src.guest_mip_level, 1, src.guest_array_layer, 1},
	                    {dst.guest_mip_level, 1, dst.guest_array_layer, 1});
	return true;
}

} // namespace Libs::Graphics
