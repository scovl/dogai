@echo off
echo ========================================
echo Build YOLOv8 Video Detection - Modular
echo ========================================

REM Verify if build directory exists
if not exist "build" (
    echo Make build directory...
    mkdir build
)

cd build

REM Configure CMake
echo Configuring CMake...
cmake .. -G "Visual Studio 17 2022" -A x64

if %ERRORLEVEL% neq 0 (
    echo [ERRO] Falha na configuração do CMake!
    pause
    exit /b 1
)

REM Compile
echo Compiling project...
cmake --build . --config Release

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to compile!
    pause
    exit /b 1
)

echo.
echo =========================================
echo Compilation completed successfully!
echo =========================================
echo.
echo Executable: build\bin\Release\video_object_detection.exe
echo.
echo To execute:
echo   cd build\bin\Release
echo   video_object_detection.exe
echo.
pause 