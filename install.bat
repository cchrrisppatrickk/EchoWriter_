@echo off
setlocal enabledelayedexpansion
title Instalador de EchoWriter AI
echo ===================================================
echo   Instalador de App de Transcripcion y Diarizacion
echo ===================================================
echo.

:: --------------------------------------------------
:: Step 1: Validar pre-requisitos
:: --------------------------------------------------
echo [1/8] Comprobando pre-requisitos...

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python no esta instalado o no esta en el PATH de Windows.
    echo Por favor, instala Python 3.10 o posterior y marca "Add Python to PATH".
    pause
    exit /b 1
)

where git >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Git no esta instalado o no esta en el PATH de Windows.
    echo Por favor, instala Git y intentalo de nuevo.
    pause
    exit /b 1
)

:: Obtener version de Python
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v
echo   Version de Python: %PYTHON_VERSION%

where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] FFmpeg no se detecto en el PATH de Windows.
    echo WhisperX requiere FFmpeg para decodificar audio y video.
    echo Si no lo tienes instalado, abre una consola como Administrador y ejecuta:
    echo   winget install ffmpeg
    echo.
    echo Presiona una tecla una vez que hayas instalado FFmpeg para continuar...
    pause
) else (
    echo   Version de FFmpeg: OK
)
echo.

:: --------------------------------------------------
:: Step 2: Crear entorno virtual
:: --------------------------------------------------
echo [2/8] Creando entorno virtual (venv)...
if exist venv (
    echo   El entorno virtual ya existe, omitiendo creacion.
) else (
    python -m venv venv
)
echo   OK
echo.

:: --------------------------------------------------
:: Step 3: Activar entorno virtual
:: --------------------------------------------------
echo [3/8] Activando entorno virtual...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] No se pudo activar el entorno virtual.
    pause
    exit /b 1
)
echo   OK
echo.

:: --------------------------------------------------
:: Step 4: Actualizar pip
:: --------------------------------------------------
echo [4/8] Actualizando pip...
python -m pip install --upgrade pip
echo   OK
echo.

:: --------------------------------------------------
:: Step 5: Detectar GPU NVIDIA y CUDA
:: --------------------------------------------------
echo [5/8] Detectando GPU NVIDIA y version de CUDA...

where nvidia-smi >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] No se detecto una GPU NVIDIA o 'nvidia-smi' no esta en el PATH.
    echo Este proyecto requiere una GPU NVIDIA para ejecutar WhisperX localmente.
    pause
    exit /b 1
)

:: Parsear version de CUDA desde nvidia-smi
set CUDA_VER=
for /f "tokens=9 delims= " %%v in ('nvidia-smi ^| findstr /C:"CUDA Version"') do set CUDA_VER=%%v

if "!CUDA_VER!"=="" (
    echo [ERROR] No se pudo detectar la version de CUDA.
    pause
    exit /b 1
)

:: Extraer version mayor y menor
for /f "tokens=1,2 delims=." %%a in ("!CUDA_VER!") do (
    set CUDA_MAJOR=%%a
    set CUDA_MINOR=%%b
)
echo   CUDA detectado: !CUDA_VER!

:: Mapear a la rueda de PyTorch CUDA correspondiente
set CU_TAG=
if !CUDA_MAJOR! GEQ 13 set CU_TAG=cu128
if !CUDA_MAJOR! EQU 12 (
    if !CUDA_MINOR! GEQ 8 (
        set CU_TAG=cu128
    ) else if !CUDA_MINOR! GEQ 4 (
        set CU_TAG=cu124
    ) else (
        set CU_TAG=cu121
    )
)
if !CUDA_MAJOR! EQU 11 set CU_TAG=cu118

if "!CU_TAG!"=="" (
    echo [ERROR] Version de CUDA no compatible. Se requiere CUDA 11.8 o superior.
    pause
    exit /b 1
)

echo   Plataforma PyTorch seleccionada: !CU_TAG!
echo   OK
echo.

:: --------------------------------------------------
:: Step 6: Instalar requisitos adicionales (Gradio)
:: --------------------------------------------------
echo [6/8] Instalando dependencias de la aplicacion (Gradio)...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Fallo la instalacion de requisitos adicionales.
    pause
    exit /b 1
)
echo   OK
echo.

:: --------------------------------------------------
:: Step 7: Instalar WhisperX
:: --------------------------------------------------
echo [7/8] Instalando WhisperX desde Git...
pip install git+https://github.com/m-bain/WhisperX.git
if %errorlevel% neq 0 (
    echo [ERROR] Fallo la instalacion de WhisperX.
    pause
    exit /b 1
)
echo   OK
echo.

:: --------------------------------------------------
:: Step 8: Instalar PyTorch CUDA-enabled
:: --------------------------------------------------
echo [8/8] Instalanado PyTorch, TorchVision y TorchAudio (!CU_TAG!)...
echo   (Asegurando binarios compatibles con GPU)
pip uninstall torch torchvision torchaudio -y 2>nul
pip install torch torchvision torchaudio --upgrade --no-cache-dir --index-url https://download.pytorch.org/whl/!CU_TAG!
if %errorlevel% neq 0 (
    echo [ERROR] Fallo la instalacion de PyTorch.
    pause
    exit /b 1
)
echo   OK
echo.

echo ===================================================
echo   Instalacion completada con exito.
echo.
echo   Para iniciar la aplicacion, ejecuta:
echo     start_app.bat
echo ===================================================
echo.
pause


