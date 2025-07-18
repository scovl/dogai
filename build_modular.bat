@echo off
echo ========================================
echo Build YOLOv8 Video Detection - Modular
echo ========================================
echo.
echo GPU Optimization for AMD RX 7600 XT
echo Target: 120+ FPS with GPU acceleration
echo.

REM Check for AMD GPU
echo Checking GPU configuration...
wmic path win32_VideoController get name | findstr /i "AMD" >nul
if %errorlevel% equ 0 (
    echo [INFO] AMD GPU detected - enabling CPU optimizations for high FPS
    set GPU_OPTIONS=-DUSE_GPU=OFF -DOPTIMIZE_FOR_AMD=ON
) else (
    echo [INFO] AMD GPU not detected - using CPU optimizations
    set GPU_OPTIONS=-DUSE_GPU=OFF
)

REM Verify if build directory exists
if not exist "build" (
    echo Creating build directory...
    mkdir build
)

cd build

REM Configure CMake with GPU optimizations
echo Configuring CMake with GPU optimizations...
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release %GPU_OPTIONS%

if %ERRORLEVEL% neq 0 (
    echo [ERROR] CMake configuration failed!
    pause
    exit /b 1
)

REM Compile with optimizations
echo Compiling project with optimizations...
cmake --build . --config Release --parallel 8

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Compilation failed!
    pause
    exit /b 1
)

REM Copy configuration file
echo Copying configuration file...
copy ..\blood.cfg bin\ >nul 2>&1

echo.
echo =========================================
echo Build completed successfully!
echo =========================================
echo.
echo Performance optimizations applied:
echo - GPU acceleration: %GPU_OPTIONS%
echo - Target FPS: 120+ FPS
echo - Build type: Release with optimizations
echo.
echo Executable: build\bin\video_object_detection.exe
echo.
echo To run with high FPS:
echo   build\bin\video_object_detection.exe
echo.
echo Configuration file:
echo   build\bin\blood.cfg
echo   - Normal mode: 120 FPS
echo   - Maximum mode: 144 FPS (change performance_mode = maximum)
echo.
pause 