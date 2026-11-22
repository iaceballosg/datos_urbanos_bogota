#!/usr/bin/env bash
# ============================================================
#  Lanzador del dashboard Streamlit — Catastro Bogotá (macOS / Linux)
#  Uso: ./run_dashboard.sh
# ============================================================
set -e

# Ir a la carpeta del proyecto (donde se encuentra este archivo)
cd "$(dirname "$0")"

# Verificar que Streamlit esté instalado
if ! command -v streamlit >/dev/null 2>&1; then
    echo "Streamlit no encontrado. Instalando dependencias..."
    python3 -m pip install -r requirements.txt || python -m pip install -r requirements.txt
fi

echo
echo "Iniciando dashboard en http://localhost:8501"
echo "(Ctrl+C para detenerlo)"
echo
streamlit run dashboard/app.py
