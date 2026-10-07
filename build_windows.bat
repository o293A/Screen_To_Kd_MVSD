@echo off
REM Builds kd_ocr.exe (needs Rust: https://rustup.rs). The result is a single self-contained file.
cargo build --release
if errorlevel 1 (echo Build failed & pause & exit /b 1)
copy /Y target\release\kd_ocr.exe kd_ocr.exe >nul
if errorlevel 1 (echo Could not copy the exe & pause & exit /b 1)

REM Permanently delete the target folder (no recycle bin)
if exist target rmdir /s /q target

echo.
echo Done: kd_ocr.exe
echo You can put kd_ocr.exe anywhere you want: it is the whole app, nothing else is needed.
echo (settings.txt is created next to it on first launch)
pause
