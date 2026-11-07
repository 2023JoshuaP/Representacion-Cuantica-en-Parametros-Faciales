"""Paso 8: resume las tres secuencias y propone roles para la comparación final.

Cada secuencia se representa por la mediana de los 20 valores normalizados de
cada parámetro. La selección de par cercano usa distancia euclidiana sin pesos.
"""

import csv
import json
import math
from itertools import combinations
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "parametros_60" / "parametros_normalizados_60.csv"
OUTPUT_DIR = ROOT / "tres_datos"
FEATURES = [
    ("EAR_33_norm", "EAR_33"),
    ("EAR_362_norm", "EAR_362"),
    ("MAR_norm", "MAR"),
    ("yaw_proxy_norm", "yaw_proxy"),
    ("luma_mediana_0_255_norm", "luminancia"),
]


def euclidean(vector_a, vector_b):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vector_a, vector_b)))


def main():
    if not INPUT.is_file():
        raise FileNotFoundError("Faltan parámetros normalizados. Ejecuta antes 04_normalizar_parametros.py.")
    rows = list(csv.DictReader(INPUT.open(encoding="utf-8", newline="")))
    if len(rows) != 60:
        raise ValueError(f"Se esperaban 60 filas, no {len(rows)}.")
    sequences = ("A", "B", "C")
    summary = []
    vectors = {}
    for sequence in sequences:
        subset = [row for row in rows if row["registro"] == sequence]
        if len(subset) != 20:
            raise ValueError(f"{sequence}: se esperaban 20 frames, no {len(subset)}.")
        vector = [median(float(row[key]) for row in subset) for key, _ in FEATURES]
        vectors[sequence] = vector
        summary.append({
            "secuencia_origen": sequence,
            "numero_frames": len(subset),
            **{label: value for (_, label), value in zip(FEATURES, vector)},
        })

    distances = []
    for left, right in combinations(sequences, 2):
        distances.append({
            "secuencia_1": left,
            "secuencia_2": right,
            "distancia_euclidiana": euclidean(vectors[left], vectors[right]),
        })
    closest = min(distances, key=lambda item: item["distancia_euclidiana"])
    close_pair = (closest["secuencia_1"], closest["secuencia_2"])
    different = next(sequence for sequence in sequences if sequence not in close_pair)
    ordered_roles = {
        "dato_A_cercano": close_pair[0],
        "dato_B_cercano": close_pair[1],
        "dato_C_diferente": different,
    }

    OUTPUT_DIR.mkdir(exist_ok=True)
    with (OUTPUT_DIR / "vectores_representativos.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    with (OUTPUT_DIR / "distancias_clasicas.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=distances[0].keys())
        writer.writeheader()
        writer.writerows(distances)
    selection = {
        "purpose": "Selección objetiva de tres datos ilustrativos para la actividad.",
        "record_definition": "Mediana de cada parámetro normalizado a través de 20 frames consecutivos de una secuencia.",
        "distance": "Euclidiana sin pesos sobre cinco parámetros normalizados.",
        "feature_order": [label for _, label in FEATURES],
        "note_luminance": "La luminancia conserva escala digital Y/255; no se amplificó hasta el rango observado de la muestra.",
        "distances": distances,
        "suggested_roles": ordered_roles,
        "interpretation": "Los roles se proponen por cercanía clásica previa a la codificación cuántica. Son ilustrativos, no una validación clínica ni una selección representativa.",
    }
    (OUTPUT_DIR / "seleccion_sugerida.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    positions = list(range(len(FEATURES)))
    labels = [label for _, label in FEATURES]
    for sequence, color in zip(sequences, ("#1f77b4", "#ff7f0e", "#2ca02c")):
        axis.plot(positions, vectors[sequence], marker="o", linewidth=2, label=f"Secuencia {sequence}", color=color)
    axis.set(xticks=positions, xticklabels=labels, ylim=(-0.05, 1.05),
             ylabel="Valor normalizado", title="Vectores representativos: mediana de 20 frames")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(OUTPUT_DIR / "vectores_representativos.png", dpi=180)
    plt.close(figure)

    print("Tres vectores representativos calculados mediante mediana de 20 frames.")
    for row in distances:
        print(f"d({row['secuencia_1']}, {row['secuencia_2']}) = {row['distancia_euclidiana']:.6f}")
    print("Roles sugeridos:", ordered_roles)


if __name__ == "__main__":
    main()
