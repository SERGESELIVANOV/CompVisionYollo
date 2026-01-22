@echo off
echo Preparing release build...

REM Создаем папку release
if not exist "release" mkdir release

REM Ищем GUI.exe в разных возможных местах
set GUI_FOUND=0

if exist "x64\Release\GUI.exe" (
    echo Found GUI.exe in x64\Release\
    copy "x64\Release\GUI.exe" "release\"
    set GUI_FOUND=1
) else if exist "GUI\x64\Release\GUI.exe" (
    echo Found GUI.exe in GUI\x64\Release\
    copy "GUI\x64\Release\GUI.exe" "release\"
    set GUI_FOUND=1
) else if exist "Release\GUI.exe" (
    echo Found GUI.exe in Release\
    copy "Release\GUI.exe" "release\"
    set GUI_FOUND=1
) else (
    echo Error: GUI.exe not found!
    echo Searched in:
    echo - x64\Release\GUI.exe
    echo - GUI\x64\Release\GUI.exe  
    echo - Release\GUI.exe
    echo.
    echo Please build the GUI project in Release mode first.
    pause
    exit /b 1
)

if %GUI_FOUND%==1 (
    echo ✓ GUI.exe copied to release folder
)

REM Используем windeployqt для копирования Qt зависимостей
echo Running windeployqt...
windeployqt.exe release\GUI.exe --release --no-translations --no-compiler-runtime

REM Копируем OpenCV DLL
echo Copying OpenCV DLLs...
if exist "C:\opencv\build\x64\vc16\bin\opencv_world4120.dll" (
    copy "C:\opencv\build\x64\vc16\bin\opencv_world4120.dll" "release\"
    echo ✓ OpenCV DLL copied
) else (
    echo ⚠ Warning: OpenCV DLL not found at expected location
)

REM Копируем папку с материалами
echo Copying Materials...
if exist "GUI\Materials" (
    xcopy "GUI\Materials" "release\Materials\" /E /I /H /Y
    echo ✓ Materials copied
) else (
    echo ⚠ Warning: Materials folder not found
)

echo.
echo Release preparation completed!
echo Contents of release folder:
dir /b release\
echo.
pause