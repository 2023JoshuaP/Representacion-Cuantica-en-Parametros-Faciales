"""Paso 4: normaliza los parámetros crudos a [0,1] para la codificación angular."""

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "parametros_60" / "parametros_crudos_60.csv"
OUTPUT_DIR = ROOT / "parametros_60"
GEOMETRIC_FEATURES = ["EAR_33", "EAR_362", "MAR", "yaw_proxy"]
LUMA_FEATURE = "luma_mediana_0_255"


def minmax(value, lower, upper):
    if not upper > lower:
        raise ValueError(f"Rango inválido: [{lower}, {upper}]")
    normalized = (value - lower) / (upper - lower)
    # No se recorta: un valor fuera del rango debe causar un error visible.
    if not 0 <= normalized <= 1:
        raise ValueError(f"Valor normalizado fuera de [0,1]: {normalized}")
    return normalized


def main():
    if not RAW_PATH.is_file():
        raise FileNotFoundError("Faltan parámetros crudos. Ejecuta antes 03_calcular_parametros.py.")
    rows = list(csv.DictReader(RAW_PATH.open(encoding="utf-8", newline="")))
    if len(rows) != 60:
        raise ValueError(f"Se esperaban 60 filas; se encontraron {len(rows)}.")
    raw = {feature: [float(row[feature]) for row in rows] for feature in GEOMETRIC_FEATURES + [LUMA_FEATURE]}
    limits = {
        feature: {"method": "min_max_global_60_frames", "min": min(values), "max": max(values)}
        for feature, values in raw.items() if feature in GEOMETRIC_FEATURES
    }
    limits[LUMA_FEATURE] = {
        "method": "division_por_255", "min": 0.0, "max": 255.0,
        "note": "Rango conocido de luma digital de 8 bits; no se amplifica el contraste entre registros.",
    }

    normalized_rows = []
    for row in rows:
        output = dict(row)
        for feature in GEOMETRIC_FEATURES:
            output[f"{feature}_norm"] = minmax(float(row[feature]), limits[feature]["min"], limits[feature]["max"])
        output[f"{LUMA_FEATURE}_norm"] = minmax(float(row[LUMA_FEATURE]), 0.0, 255.0)
        normalized_rows.append(output)

    normalized_features = [f"{feature}_norm" for feature in GEOMETRIC_FEATURES + [LUMA_FEATURE]]
    if not all(math.isfinite(float(row[feature])) and 0 <= float(row[feature]) <= 1
               for row in normalized_rows for feature in normalized_features):
        raise ValueError("La validación de normalización falló.")

    destination = OUTPUT_DIR / "parametros_normalizados_60.csv"
    with destination.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=normalized_rows[0].keys())
        writer.writeheader()
        writer.writerows(normalized_rows)
    normalization_record = {
        "purpose": "Paso 4: normalización previa a Ry(pi*x).",
        "rows": len(normalized_rows),
        "limits": limits,
        "policy": "Los rasgos geométricos usan el mínimo y máximo global de los 60 frames. La luminancia se divide por 255. No se ajustaron límites por comparación ni se recortaron valores.",
    }
    (OUTPUT_DIR / "normalizacion.json").write_text(
        json.dumps(normalization_record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    figure, axes = plt.subplots(len(normalized_features), 1, figsize=(10, 12), sharex=True, constrained_layout=True)
    colors = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}
    for axis, feature in zip(axes, normalized_features):
        for sequence in ("A", "B", "C"):
            subset = [row for row in normalized_rows if row["registro"] == sequence]
            axis.plot([int(row["orden_en_secuencia"]) for row in subset],
                      [float(row[feature]) for row in subset], marker="o", markersize=3,
                      linewidth=1.3, label=sequence, color=colors[sequence])
        axis.set_ylabel(feature)
        axis.set_ylim(-0.05, 1.05)
        axis.grid(alpha=0.25)
        axis.legend(title="Registro", ncol=3, loc="best")
    axes[-1].set_xlabel("Orden dentro de la secuencia (1–20)")
    figure.suptitle("Parámetros normalizados para codificación angular", fontsize=13)
    figure.savefig(OUTPUT_DIR / "series_parametros_normalizados.png", dpi=180)
    plt.close(figure)
    print("Normalización válida: 60 filas y cinco parámetros en [0,1].")
    print(f"Tabla: {destination}")
    print(f"Límites: {OUTPUT_DIR / 'normalizacion.json'}")


if __name__ == "__main__":
    main()
