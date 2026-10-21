@echo off
setlocal
pushd "%~dp0" || exit /b 1
set "webcam_python=%~dp0.venv\Scripts\python.exe"
if not exist "%webcam_python%" set "webcam_python=%~dp0..\.venv\Scripts\python.exe"
if not exist "%webcam_python%" (
    echo Python virtual environment missing. See README.md.
    popd
    exit /b 1
)
"%webcam_python%" -X utf8 -m src.webcam %*
set "webcam_exit=%errorlevel%"
popd
exit /b %webcam_exit%
