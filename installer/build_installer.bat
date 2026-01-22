@echo off
echo Building installer...

REM Устанавливаем путь к Qt Installer Framework
set QT_IFW_DIR=C:\Qt\Tools\QtInstallerFramework\4.10

REM Проверяем наличие binarycreator
if not exist "%QT_IFW_DIR%\bin\binarycreator.exe" (
    echo Error: Qt Installer Framework not found at %QT_IFW_DIR%!
    pause
    exit /b 1
)

REM Проверяем наличие файлов
if not exist "..\release\GUI.exe" (
    echo Error: GUI.exe not found!
    pause
    exit /b 1
)

REM Создаем/очищаем папку data
if exist "packages\compvision.app\data" rmdir /s /q "packages\compvision.app\data"
mkdir "packages\compvision.app\data"
xcopy "..\release\*" "packages\compvision.app\data\" /E /I /H /Y /EXCLUDE:exclude.txt

echo .git > exclude.txt
echo *.pdb >> exclude.txt
echo *.log >> exclude.txt
echo *.tlog >> exclude.txt
echo microsoft >> exclude.txt

if not exist "packages\compvision.app\data\GUI.exe" (
    echo Copy failed!
    pause
    exit /b 1
)

REM Создаем установщик (деинсталлятор создается автоматически)
echo Creating installer...
"%QT_IFW_DIR%\bin\binarycreator.exe" ^
    --offline-only ^
    -c config\config.xml ^
    -p packages ^
    ComputerVisionYOLO_Installer.exe

if %errorlevel% equ 0 (
    echo.
    echo ============================================
    echo ✓ Installer created successfully!
    echo File: ComputerVisionYOLO_Installer.exe
    echo Location: %CD%\ComputerVisionYOLO_Installer.exe
    echo.
    echo Size: 
    for %%A in ("ComputerVisionYOLO_Installer.exe") do echo   %%~zA bytes
    echo.
    echo After installation, users can uninstall via:
    echo - MaintenanceTool.exe in the program folder
    echo - Windows "Programs and Features"
    echo ============================================
) else (
    echo ✗ Error creating installer! Code: %errorlevel%
)

if exist "exclude.txt" del exclude.txt
pause