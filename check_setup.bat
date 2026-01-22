@echo off
echo Checking setup...

echo Current directory: %CD%
echo.

echo Checking GUI executable locations:
set GUI_FOUND=0

if exist "x64\Release\GUI.exe" (
    echo ✓ GUI.exe found at x64\Release\GUI.exe
    set GUI_FOUND=1
) else (
    echo ✗ GUI.exe not found at x64\Release\GUI.exe
)

if exist "GUI\x64\Release\GUI.exe" (
    echo ✓ GUI.exe found at GUI\x64\Release\GUI.exe
    set GUI_FOUND=1
) else (
    echo ✗ GUI.exe not found at GUI\x64\Release\GUI.exe
)

if exist "Release\GUI.exe" (
    echo ✓ GUI.exe found at Release\GUI.exe
    set GUI_FOUND=1
) else (
    echo ✗ GUI.exe not found at Release\GUI.exe
)

if %GUI_FOUND%==0 (
    echo ✗ No GUI.exe found in expected locations
    echo Please build the GUI project in Release mode
)

echo.
echo Checking Qt Installer Framework:
if exist "C:\Qt\Tools\QtInstallerFramework\4.10\bin\binarycreator.exe" (
    echo ✓ Qt IFW found at C:\Qt\Tools\QtInstallerFramework\4.10
) else (
    echo ✗ Qt IFW not found at expected location
)

echo.
echo Checking installer structure:
if exist "installer\config\config.xml" (
    echo ✓ config.xml found
) else (
    echo ✗ config.xml not found
)

if exist "installer\packages\compvision.app\meta\package.xml" (
    echo ✓ package.xml found
) else (
    echo ✗ package.xml not found
)

echo.
echo Next steps:
echo 1. Build GUI project in Release mode
echo 2. Run prepare_release.bat  
echo 3. Go to installer folder and run build_installer.bat
echo.
pause