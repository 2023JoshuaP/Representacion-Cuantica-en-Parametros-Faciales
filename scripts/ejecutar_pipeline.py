"""Ejecuta en orden el pipeline reproducible de frames a resultados finales.

Uso:
    python scripts/ejecutar_pipeline.py

Requiere el video fuente de la tesis en la ruta esperada por 01_extraer_frames.py,
además de ffmpeg, ffprobe y las dependencias de requirements.txt.
"""

from pathlib import Path
import subprocess
import sys


SCRIPTS_DIR = Path(__file__).resolve().parent
STEPS = [
    "01_extraer_frames.py",
    "02_detectar_landmarks.py",
    "03_calcular_parametros.py",
    "04_normalizar_parametros.py",
    "05_primer_circuito.py",
    "06_generar_circuitos_60.py",
    "07_generar_circuitos_extendidos_60.py",
    "08_consolidar_tres_datos.py",
    "09_preparar_tres_datos_finales.py",
    "10_generar_circuitos_tres_datos.py",
    "11_generar_circuitos_extendidos_tres_datos.py",
    "12_comparar_fidelidades.py",
    "13_swap_test.py",
    "14_crear_figura_informe.py",
    "16_crear_figura_circuitos_extendidos_ABC.py",
]


def main():
    for position, name in enumerate(STEPS, start=1):
        print(f"\n[{position}/{len(STEPS)}] {name}")
        subprocess.run([sys.executable, str(SCRIPTS_DIR / name)], check=True)


if __name__ == "__main__":
    main()
