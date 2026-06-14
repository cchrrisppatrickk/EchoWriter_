@echo off
echo Iniciando App de Transcripcion y Diarizacion...
if not exist venv (
    echo [ERROR] No se detecto el entorno virtual 'venv'.
    echo Por favor, ejecuta primero 'install.bat' para instalar todas las dependencias.
    pause
    exit /b 1
)

call venv\Scripts\activate
if %errorlevel% neq 0 (
    echo [ERROR] No se pudo activar el entorno virtual.
    pause
    exit /b 1
)

python app.py
if %errorlevel% neq 0 (
    echo.
    echo La aplicacion ha finalizado con un codigo de error (%errorlevel%).
    pause
)
