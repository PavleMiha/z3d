const std = @import("std");

const zglfw = @import("zglfw");
const zgpu = @import("zgpu");

pub fn main() !void {
    try zglfw.init();
    defer zglfw.terminate();

    const window = try zglfw.createWindow(800, 600, "zig-gamedev: minimal_glfw_gl", null);

    defer zglfw.destroyWindow(window);

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();
    
    const gctx = try zgpu.GraphicsContext.create(
    allocator,
    .{
        .window = window,
        .fn_getTime = @ptrCast(&zglfw.getTime),
        .fn_getFramebufferSize = @ptrCast(&zglfw.Window.getFramebufferSize),

        // optional fields
        .fn_getWin32Window = @ptrCast(&zglfw.getWin32Window),
        .fn_getX11Display = @ptrCast(&zglfw.getX11Display),
        .fn_getX11Window = @ptrCast(&zglfw.getX11Window),
        .fn_getWaylandDisplay = @ptrCast(&zglfw.getWaylandDisplay),
        .fn_getWaylandSurface = @ptrCast(&zglfw.getWaylandWindow),
        .fn_getCocoaWindow = @ptrCast(&zglfw.getCocoaWindow),
    },
    .{}, // default context creation options
    );
    defer zgpu.GraphicsContext.destroy(gctx, allocator);



    while (!window.shouldClose()) {
        zglfw.pollEvents();

        // render your things here
        
        window.swapBuffers();
    }
}