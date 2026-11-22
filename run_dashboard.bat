@echo off
REM ============================================================
REM  Lanzador del dashboard Streamlit - Catastro Bogota
REM  Ejecutar con doble clic (Windows).
REM ============================================================

REM Ir a la carpeta del proyecto (donde se encuentra este archivo)
cd /d "%~dp0"

REM Verificar que Streamlit este instalado
where streamlit >nul 2>nul
if errorlevel 1 (
    echo Streamlit no encontrado. Instalando dependencias...
    echo.
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [ERROR] No se pudieron instalar las dependencias.
        echo Verifica que Python este instalado y agregado al PATH.
        pause
        exit /b 1
    )
)

echo.
echo Iniciando dashboard en http://localhost:8501
echo Cierra esta ventana o presiona Ctrl+C para detenerlo.
echo.
streamlit run dashboard\app.py

pause
