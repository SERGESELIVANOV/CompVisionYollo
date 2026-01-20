@echo off
echo Building GUI project...

set MSBUILD_PATH="C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe"

if not exist %MSBUILD_PATH% (
    echo MSBuild not found at %MSBUILD_PATH%
    echo Please check Visual Studio installation
    pause
    exit /b 1
)

%MSBUILD_PATH% GUI\GUI.vcxproj /p:Configuration=Debug /p:Platform=x64

if %ERRORLEVEL% neq 0 (
    echo Build failed!
    pause
    exit /b 1
)

echo Build successful!
echo GUI executable should be in GUI\x64\Debug\GUI.exe
pause