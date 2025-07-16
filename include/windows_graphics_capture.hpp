#pragma once

#include "logger.hpp"
#include <opencv2/opencv.hpp>
#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>
#include <dxgi1_2.h>
#include <dxgi1_3.h>

// Use COM interface IDs directly - FIXED FOR COMPATIBILITY
// Using explicit GUIDs to avoid compiler issues with __uuidof

// IID for ID3D11Texture2D
static const GUID IID_ID3D11Texture2D_ = { 0x6f15aaf2, 0xd208, 0x4e89, { 0x9a, 0xb4, 0x48, 0x95, 0x35, 0xd3, 0x4f, 0x9c } };
// IID for IDXGIFactory1
static const GUID IID_IDXGIFactory1_ = { 0x770aae78, 0xf26f, 0x4dba, { 0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87 } };
// IID for IDXGIOutput1
static const GUID IID_IDXGIOutput1_ = { 0x00cddea8, 0x939b, 0x4b83, { 0xa3, 0x40, 0xa6, 0x85, 0x22, 0x66, 0x66, 0xcc } };

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "windowsapp.lib")

class WindowsGraphicsCapture {
private:
    ID3D11Device* d3d_device = nullptr;
    ID3D11DeviceContext* d3d_context = nullptr;
    IDXGIOutputDuplication* desktop_duplication = nullptr;
    IDXGIOutput* dxgi_output = nullptr;
    IDXGIAdapter* dxgi_adapter = nullptr;
    IDXGIFactory1* dxgi_factory = nullptr;
    IDXGIOutput1* dxgi_output1 = nullptr;
    
    bool initialized = false;

public:
    WindowsGraphicsCapture();
    ~WindowsGraphicsCapture();
    
    bool is_initialized() const;
    cv::Mat capture_screen();

private:
    bool initialize_d3d();
    void cleanup();
}; 