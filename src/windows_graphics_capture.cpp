#include "windows_graphics_capture.hpp"
#include <algorithm>

WindowsGraphicsCapture::WindowsGraphicsCapture() {
    initialize_d3d();
}

WindowsGraphicsCapture::~WindowsGraphicsCapture() {
    cleanup();
}

bool WindowsGraphicsCapture::is_initialized() const {
    return initialized;
}

cv::Size WindowsGraphicsCapture::get_screen_size() const {
    return screen_size;
}

cv::Point WindowsGraphicsCapture::get_screen_center() const {
    return cv::Point(screen_size.width / 2, screen_size.height / 2);
}

cv::Rect WindowsGraphicsCapture::calculate_fov_region(int fov_width, int fov_height) {
    auto center = get_screen_center();
    auto x = center.x - fov_width / 2;
    auto y = center.y - fov_height / 2;
    
    // Ensure FOV doesn't go outside screen bounds
    x = (x > 0) ? x : 0;
    y = (y > 0) ? y : 0;
    
    auto actual_width = (fov_width < (screen_size.width - x)) ? fov_width : (screen_size.width - x);
    auto actual_height = (fov_height < (screen_size.height - y)) ? fov_height : (screen_size.height - y);
    
    return cv::Rect(x, y, actual_width, actual_height);
}

cv::Mat WindowsGraphicsCapture::capture_fov(int fov_width, int fov_height) {
    auto full_screen = capture_screen();
    if (full_screen.empty()) {
        return cv::Mat();
    }
    
    auto fov_region = calculate_fov_region(fov_width, fov_height);
    return full_screen(fov_region);
}

cv::Mat WindowsGraphicsCapture::capture_screen() {
    if (!initialized) {
        logger.error("[WGC][ERROR] Screen capture not initialized!");
        return cv::Mat();
    }
    
    auto desktop_resource = static_cast<IDXGIResource*>(nullptr);
    auto frame_info = DXGI_OUTDUPL_FRAME_INFO();
    
    // Try multiple times with increasing timeout
    auto hr = S_OK;
    for (int attempt = 0; attempt < 3; ++attempt) {
        hr = desktop_duplication->AcquireNextFrame(500, &frame_info, &desktop_resource);
        if (SUCCEEDED(hr)) {
            break;
        }
        
        // If it's a timeout error, wait a bit and try again
        if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
            Sleep(50); // Wait 50ms before retry
            continue;
        }
        
        // Handle access lost error - try to reinitialize
        if (hr == DXGI_ERROR_ACCESS_LOST_ERROR) {
            logger.warning("[WGC][WARNING] Desktop duplication access lost, attempting to reinitialize...");
            if (reinitialize_capture()) {
                continue; // Try again with new duplication
            } else {
                logger.error("[WGC][ERROR] Failed to reinitialize desktop duplication");
                return cv::Mat();
            }
        }
        
        // For other errors, log and return
        logger.error("[WGC][ERROR] Failed to acquire next frame: " + std::to_string(hr));
        return cv::Mat();
    }
    
    if (FAILED(hr)) {
        logger.error("[WGC][ERROR] Failed to acquire next frame after retries: " + std::to_string(hr));
        return cv::Mat();
    }
    
    ID3D11Texture2D* desktop_texture = nullptr;
    hr = desktop_resource->QueryInterface(IID_ID3D11Texture2D_, (void**)&desktop_texture);
    desktop_resource->Release();
    
    if (FAILED(hr)) {
        desktop_duplication->ReleaseFrame();
        logger.error("[WGC][ERROR] Failed to get ID3D11Texture2D interface: " + std::to_string(hr));
        return cv::Mat();
    }
    
    // Get texture description
    D3D11_TEXTURE2D_DESC texture_desc;
    desktop_texture->GetDesc(&texture_desc);
    
    // Update screen size if not set
    if (screen_size.width == 0 || screen_size.height == 0) {
        screen_size = cv::Size(texture_desc.Width, texture_desc.Height);
        logger.info("[WGC][INFO] Screen size detected: " + std::to_string(screen_size.width) + "x" + std::to_string(screen_size.height));
    }
    
    // Create staging texture for CPU access
    D3D11_TEXTURE2D_DESC staging_desc = texture_desc;
    staging_desc.Usage = D3D11_USAGE_STAGING;
    staging_desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    staging_desc.BindFlags = 0;
    staging_desc.MiscFlags = 0;
    
    ID3D11Texture2D* staging_texture = nullptr;
    hr = d3d_device->CreateTexture2D(&staging_desc, nullptr, &staging_texture);
    
    if (SUCCEEDED(hr)) {
        // Copy desktop texture to staging texture
        d3d_context->CopyResource(staging_texture, desktop_texture);
        
        // Map staging texture to get pixel data
        D3D11_MAPPED_SUBRESOURCE mapped_resource;
        hr = d3d_context->Map(staging_texture, 0, D3D11_MAP_READ, 0, &mapped_resource);
        
        if (SUCCEEDED(hr)) {
            // Create OpenCV Mat from pixel data
            cv::Mat frame(texture_desc.Height, texture_desc.Width, CV_8UC4, mapped_resource.pData, mapped_resource.RowPitch);
            cv::Mat frame_bgr;
            cv::cvtColor(frame, frame_bgr, cv::COLOR_BGRA2BGR);
            
            d3d_context->Unmap(staging_texture, 0);
            staging_texture->Release();
            desktop_texture->Release();
            desktop_duplication->ReleaseFrame();
            
            if (frame_bgr.empty()) {
                logger.error("[WGC][ERROR] Failed to convert captured frame!");
            }
            return frame_bgr;
        }
        
        staging_texture->Release();
    }
    
    desktop_texture->Release();
    desktop_duplication->ReleaseFrame();
    return cv::Mat();
}

bool WindowsGraphicsCapture::initialize_d3d() {
    // Create D3D11 device
    D3D_FEATURE_LEVEL feature_level;
    HRESULT hr = D3D11CreateDevice(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
        nullptr, 0, D3D11_SDK_VERSION,
        &d3d_device, &feature_level, &d3d_context);
    
    if (FAILED(hr)) {
        logger.error("[WGC][ERROR] Failed to create D3D11 device - HRESULT: " + std::to_string(hr));
        return false;
    }
    
    // Get DXGI factory
    hr = CreateDXGIFactory1(IID_IDXGIFactory1_, (void**)&dxgi_factory);
    if (FAILED(hr)) {
        logger.error("[WGC][ERROR] Failed to create DXGI factory - HRESULT: " + std::to_string(hr));
        logger.error("[WGC][ERROR] Possible causes:");
        logger.error("[WGC][ERROR] 1. Video drivers out of date");
        logger.error("[WGC][ERROR] 2. DirectX not installed");
        logger.error("[WGC][ERROR] 3. Insufficient permissions");
        return false;
    }
    
    // Get primary adapter
    hr = dxgi_factory->EnumAdapters(0, &dxgi_adapter);
    if (FAILED(hr)) {
        logger.error("[WGC][ERROR] Failed to get DXGI adapter - HRESULT: " + std::to_string(hr));
        return false;
    }
    
    // Get primary output
    hr = dxgi_adapter->EnumOutputs(0, &dxgi_output);
    if (FAILED(hr)) {
        logger.error("[WGC][ERROR] Failed to get DXGI output - HRESULT: " + std::to_string(hr));
        return false;
    }
    
    // Get IDXGIOutput1 interface
    hr = dxgi_output->QueryInterface(IID_IDXGIOutput1_, (void**)&dxgi_output1);
    if (FAILED(hr)) {
        logger.error("[WGC][ERROR] Failed to get IDXGIOutput1 interface - HRESULT: " + std::to_string(hr));
        return false;
    }
    
    // Create desktop duplication
    hr = dxgi_output1->DuplicateOutput(d3d_device, &desktop_duplication);
    if (FAILED(hr)) {
        logger.error("[WGC][ERROR] Failed to create desktop duplication - HRESULT: " + std::to_string(hr));
        logger.error("[WGC][ERROR] Possible causes:");
        logger.error("[WGC][ERROR] 1. Application does not have permission to capture screen");
        logger.error("[WGC][ERROR] 2. Another application is already capturing");
        logger.error("[WGC][ERROR] 3. Windows Graphics Capture not supported");
        return false;
    }
    
    initialized = true;
    
    // Check if we have proper permissions
    if (!check_permissions()) {
        logger.error("[WGC][ERROR] Insufficient permissions for screen capture!");
        logger.error("[WGC][ERROR] Please run as administrator or check Windows privacy settings");
        cleanup();
        return false;
    }
    
    return true;
}

void WindowsGraphicsCapture::cleanup() {
    if (desktop_duplication) {
        desktop_duplication->Release();
        desktop_duplication = nullptr;
    }
    if (dxgi_output1) {
        dxgi_output1->Release();
        dxgi_output1 = nullptr;
    }
    if (dxgi_output) {
        dxgi_output->Release();
        dxgi_output = nullptr;
    }
    if (dxgi_adapter) {
        dxgi_adapter->Release();
        dxgi_adapter = nullptr;
    }
    if (dxgi_factory) {
        dxgi_factory->Release();
        dxgi_factory = nullptr;
    }
    if (d3d_context) {
        d3d_context->Release();
        d3d_context = nullptr;
    }
    if (d3d_device) {
        d3d_device->Release();
        d3d_device = nullptr;
    }
}

bool WindowsGraphicsCapture::check_permissions() {
    // Check if running with administrator privileges
    BOOL is_admin = FALSE;
    PSID admin_group = nullptr;
    SID_IDENTIFIER_AUTHORITY nt_authority = SECURITY_NT_AUTHORITY;
    
    if (AllocateAndInitializeSid(&nt_authority, 2, SECURITY_BUILTIN_DOMAIN_RID,
        DOMAIN_ALIAS_RID_ADMINS, 0, 0, 0, 0, 0, 0, &admin_group)) {
        CheckTokenMembership(nullptr, admin_group, &is_admin);
        FreeSid(admin_group);
    }
    
    if (!is_admin) {
        logger.warning("[WGC][WARNING] Not running as administrator - screen capture may fail");
        logger.warning("[WGC][WARNING] Consider running as administrator for better compatibility");
    }
    
    // Check if desktop duplication is working
    if (!desktop_duplication) {
        logger.error("[WGC][ERROR] Desktop duplication not initialized");
        return false;
    }
    
    return true;
}

bool WindowsGraphicsCapture::reinitialize_capture() {
    logger.info("[WGC][INFO] Attempting to reinitialize screen capture...");
    
    // Clean up existing resources
    cleanup();
    
    // Wait a bit before reinitializing
    Sleep(200);
    
    // Try to reinitialize
    if (initialize_d3d()) {
        logger.info("[WGC][INFO] Successfully reinitialized screen capture");
        return true;
    } else {
        logger.error("[WGC][ERROR] Failed to reinitialize screen capture");
        return false;
    }
} 