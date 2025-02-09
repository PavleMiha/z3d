const std = @import("std");
const glm = @import("ziglm");
const math = std.math;
const zm = @import("zmath");
const glfw = @import("zglfw");
const wgpu = @import("zgpu");

const vec3 = glm.vec3;
const mat4 = glm.mat4;

const Transform = struct {
    parent: u32,
    local_pos: vec3,
    local_rot: vec3,
    local_scale: vec3,
    dirty: bool,
    local_to_world: mat4,
};

const Mesh = struct {
    vertex_buffer: wgpu.BufferHandle,
    index_buffer: wgpu.BufferHandle,
    index_count: u32,
};

const Vertex = struct {
    position: [3]f32,
    color: [3]f32,
};

const NUM_MESHES = 1024;
const GameState = struct {
    camera_transform: Transform,
    transforms: Transform[NUM_MESHES],
    meshes: Mesh[NUM_MESHES],
};

const Resources = struct {
    // zig fmt: off
    const wgsl_vs =
    \\  @group(0) @binding(0) var<uniform> object_to_clip: mat4x4<f32>;
    \\  struct VertexOut {
    \\      @builtin(position) position_clip: vec4<f32>,
    \\      @location(0) color: vec3<f32>,
    \\  }
    \\  @vertex fn main(
    \\      @location(0) position: vec3<f32>,
    \\      @location(1) color: vec3<f32>,
    \\  ) -> VertexOut {
    \\      var output: VertexOut;
    \\      output.position_clip = vec4(position, 1.0) * object_to_clip;
    \\      output.color = color;
    \\      return output;
    \\  }
    ;
    const wgsl_fs =
    \\  @fragment fn main(
    \\      @location(0) color: vec3<f32>,
    \\  ) -> @location(0) vec4<f32> {
    \\      return vec4(color, 1.0);
    \\  }
// zig fmt: on
    ;

    allocator: std.mem.Allocator,
    gfx_context: wgpu.GraphicsContext,

    pipeline: wgpu.RenderPipelineHandle,
    bind_group: wgpu.BindGroupHandle,

    vertex_buffer: wgpu.BufferHandle,
    index_buffer: wgpu.BufferHandle,

    depth_texture: wgpu.TextureHandle,
    depth_texture_view: wgpu.TextureViewHandle,

    fn createDepthTexture(gctx: *wgpu.GraphicsContext) struct {
        texture: wgpu.TextureHandle,
        view: wgpu.TextureViewHandle,
    } {
        const texture = gctx.createTexture(.{
            .usage = .{ .render_attachment = true },
            .dimension = .tdim_2d,
            .size = .{
                .width = gctx.swapchain_descriptor.width,
                .height = gctx.swapchain_descriptor.height,
                .depth_or_array_layers = 1,
            },
            .format = .depth32_float,
            .mip_level_count = 1,
            .sample_count = 1,
        });
        const view = gctx.createTextureView(texture, .{});
        return .{ .texture = texture, .view = view };
    }

    fn init(allocator: std.mem.Allocator, window: *glfw.Window) !Resources {
        const gctx = try wgpu.GraphicsContext.create(
            allocator,
            .{
                .window = window,
                .fn_getTime = @ptrCast(&glfw.getTime),
                .fn_getFramebufferSize = @ptrCast(&glfw.Window.getFramebufferSize),
                .fn_getWin32Window = @ptrCast(&glfw.getWin32Window),
                .fn_getX11Display = @ptrCast(&glfw.getX11Display),
                .fn_getX11Window = @ptrCast(&glfw.getX11Window),
                .fn_getWaylandDisplay = @ptrCast(&glfw.getWaylandDisplay),
                .fn_getWaylandSurface = @ptrCast(&glfw.getWaylandWindow),
                .fn_getCocoaWindow = @ptrCast(&glfw.getCocoaWindow),
            },
            .{},
        );
        errdefer gctx.destroy(allocator);

        const bind_group_layout = gctx.createBindGroupLayout(&.{
            wgpu.bufferEntry(0, .{ .vertex = true }, .uniform, true, 0),
        });
        defer gctx.releaseResource(bind_group_layout);

        const pipeline_layout = gctx.createPipelineLayout(&.{bind_group_layout});
        defer gctx.releaseResource(pipeline_layout);

        const pipeline = pipeline: {
            const vs_module = wgpu.createWgslShaderModule(gctx.device, wgsl_vs, "vs");
            defer vs_module.release();

            const fs_module = wgpu.createWgslShaderModule(gctx.device, wgsl_fs, "fs");
            defer fs_module.release();

            const color_targets = [_]wgpu.ColorTargetState{.{
                .format = wgpu.GraphicsContext.swapchain_format,
            }};

            const vertex_attributes = [_]wgpu.VertexAttribute{
                .{ .format = .float32x3, .offset = 0, .shader_location = 0 },
                .{ .format = .float32x3, .offset = @offsetOf(Vertex, "color"), .shader_location = 1 },
            };
            const vertex_buffers = [_]wgpu.VertexBufferLayout{.{
                .array_stride = @sizeOf(Vertex),
                .attribute_count = vertex_attributes.len,
                .attributes = &vertex_attributes,
            }};

            const pipeline_descriptor = wgpu.RenderPipelineDescriptor{
                .vertex = wgpu.VertexState{
                    .module = vs_module,
                    .entry_point = "main",
                    .buffer_count = vertex_buffers.len,
                    .buffers = &vertex_buffers,
                },
                .primitive = wgpu.PrimitiveState{
                    .front_face = .ccw,
                    .cull_mode = .none,
                    .topology = .triangle_list,
                },
                .depth_stencil = &wgpu.DepthStencilState{
                    .format = .depth32_float,
                    .depth_write_enabled = true,
                    .depth_compare = .less,
                },
                .fragment = &wgpu.FragmentState{
                    .module = fs_module,
                    .entry_point = "main",
                    .target_count = color_targets.len,
                    .targets = &color_targets,
                },
            };
            break :pipeline gctx.createRenderPipeline(pipeline_layout, pipeline_descriptor);
        };

        const bind_group = gctx.createBindGroup(bind_group_layout, &.{
            .{ .binding = 0, .buffer_handle = gctx.uniforms.buffer, .offset = 0, .size = @sizeOf(zm.Mat) },
        });

        // Create a vertex buffer.
        const vertex_buffer = gctx.createBuffer(.{
            .usage = .{ .copy_dst = true, .vertex = true },
            .size = 3 * @sizeOf(Vertex),
        });
        const vertex_data = [_]Vertex{
            .{ .position = [3]f32{ 0.0, 0.5, 0.0 }, .color = [3]f32{ 1.0, 0.0, 0.0 } },
            .{ .position = [3]f32{ -0.5, -0.5, 0.0 }, .color = [3]f32{ 0.0, 1.0, 0.0 } },
            .{ .position = [3]f32{ 0.5, -0.5, 0.0 }, .color = [3]f32{ 0.0, 0.0, 1.0 } },
        };
        gctx.queue.writeBuffer(gctx.lookupResource(vertex_buffer).?, 0, Vertex, vertex_data[0..]);

        // Create an index buffer.
        const index_buffer = gctx.createBuffer(.{
            .usage = .{ .copy_dst = true, .index = true },
            .size = 3 * @sizeOf(u32),
        });
        const index_data = [_]u32{ 0, 1, 2 };
        gctx.queue.writeBuffer(gctx.lookupResource(index_buffer).?, 0, u32, index_data[0..]);

        // Create a depth texture and its 'view'.
        const depth = createDepthTexture(gctx);

        return Resources{
            .gctx = gctx,
            .pipeline = pipeline,
            .bind_group = bind_group,
            .vertex_buffer = vertex_buffer,
            .index_buffer = index_buffer,
            .depth_texture = depth.texture,
            .depth_texture_view = depth.view,
        };
    }

    fn deinit(self: *Resources) void {
        self.gctx.destroy(self.allocator);
    }
};

fn draw(resources: *Resources) void {
    const gctx = resources.gctx;
    const fb_width = gctx.swapchain_descriptor.width;
    const fb_height = gctx.swapchain_descriptor.height;
    const t = @as(f32, @floatCast(gctx.stats.time));

    const cam_world_to_view = zm.lookAtLh(
        zm.f32x4(3.0, 3.0, -3.0, 1.0),
        zm.f32x4(0.0, 0.0, 0.0, 1.0),
        zm.f32x4(0.0, 1.0, 0.0, 0.0),
    );
    const cam_view_to_clip = zm.perspectiveFovLh(
        0.25 * math.pi,
        @as(f32, @floatFromInt(fb_width)) / @as(f32, @floatFromInt(fb_height)),
        0.01,
        200.0,
    );
    const cam_world_to_clip = zm.mul(cam_world_to_view, cam_view_to_clip);

    const back_buffer_view = gctx.swapchain.getCurrentTextureView();
    defer back_buffer_view.release();

    const commands = commands: {
        const encoder = gctx.device.createCommandEncoder(null);
        defer encoder.release();

        pass: {
            const vb_info = gctx.lookupResourceInfo(resources.vertex_buffer) orelse break :pass;
            const ib_info = gctx.lookupResourceInfo(resources.index_buffer) orelse break :pass;
            const pipeline = gctx.lookupResource(resources.pipeline) orelse break :pass;
            const bind_group = gctx.lookupResource(resources.bind_group) orelse break :pass;
            const depth_view = gctx.lookupResource(resources.depth_texture_view) orelse break :pass;

            const color_attachments = [_]wgpu.RenderPassColorAttachment{.{
                .view = back_buffer_view,
                .load_op = .clear,
                .store_op = .store,
            }};
            const depth_attachment = wgpu.RenderPassDepthStencilAttachment{
                .view = depth_view,
                .depth_load_op = .clear,
                .depth_store_op = .store,
                .depth_clear_value = 1.0,
            };
            const render_pass_info = wgpu.RenderPassDescriptor{
                .color_attachment_count = color_attachments.len,
                .color_attachments = &color_attachments,
                .depth_stencil_attachment = &depth_attachment,
            };
            const pass = encoder.beginRenderPass(render_pass_info);
            defer {
                pass.end();
                pass.release();
            }

            pass.setVertexBuffer(0, vb_info.gpuobj.?, 0, vb_info.size);
            pass.setIndexBuffer(ib_info.gpuobj.?, .uint32, 0, ib_info.size);

            pass.setPipeline(pipeline);

            // Draw triangle 1.
            {
                const object_to_world = zm.mul(zm.rotationY(t), zm.translation(-1.0, 0.0, 0.0));
                const object_to_clip = zm.mul(object_to_world, cam_world_to_clip);

                const mem = gctx.uniformsAllocate(zm.Mat, 1);
                mem.slice[0] = zm.transpose(object_to_clip);

                pass.setBindGroup(0, bind_group, &.{mem.offset});
                pass.drawIndexed(3, 1, 0, 0, 0);
            }

            // Draw triangle 2.
            {
                const object_to_world = zm.mul(zm.rotationY(0.75 * t), zm.translation(1.0, 0.0, 0.0));
                const object_to_clip = zm.mul(object_to_world, cam_world_to_clip);

                const mem = gctx.uniformsAllocate(zm.Mat, 1);
                mem.slice[0] = zm.transpose(object_to_clip);

                pass.setBindGroup(0, bind_group, &.{mem.offset});
                pass.drawIndexed(3, 1, 0, 0, 0);
            }
        }
        {
            const color_attachments = [_]wgpu.RenderPassColorAttachment{.{
                .view = back_buffer_view,
                .load_op = .load,
                .store_op = .store,
            }};
            const render_pass_info = wgpu.RenderPassDescriptor{
                .color_attachment_count = color_attachments.len,
                .color_attachments = &color_attachments,
            };
            const pass = encoder.beginRenderPass(render_pass_info);
            defer {
                pass.end();
                pass.release();
            }

            //zgui.backend.draw(pass);
        }

        break :commands encoder.finish(null);
    };
    defer commands.release();

    gctx.submit(&.{commands});

    if (gctx.present() == .swap_chain_resized) {
        // Release old depth texture.
        gctx.releaseResource(resources.depth_texture_view);
        gctx.destroyResource(resources.depth_texture);

        // Create a new depth texture to match the new window size.
        const depth = resources.createDepthTexture(gctx);
        resources.depth_texture = depth.texture;
        resources.depth_texture_view = depth.view;
    }
}

pub fn main() !void {
    try glfw.init();
    defer glfw.terminate();

    glfw.windowHint(.client_api, .no_api);

    const window = try glfw.Window.create(1600, 1000, "z3d", null);
    defer window.destroy();
    window.setSizeLimits(400, 400, -1, -1);

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();

    var resources = try Resources.init(allocator, window);
    defer resources.deinit();

    const scale_factor = scale_factor: {
        const scale = window.getContentScale();
        break :scale_factor @max(scale[0], scale[1]);
    };

    // setup your graphics context here
    _ = scale_factor;
    while (!window.shouldClose()) {
        glfw.pollEvents();

        if (window.getKey(.space) == .press) {
            draw(resources);
            break;
        }
        // render your things here

        window.swapBuffers();
    }
}
