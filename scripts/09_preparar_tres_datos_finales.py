"""Paso 9: fija tres frames finales y crea su representación clásica de 40 bits."""

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "parametros_60" / "parametros_normalizados_60.csv"
OUTPUT_DIR = ROOT / "tres_datos_finales"
FEATURES = [
    ("EAR_33", "EAR_33_norm", "EAR_33"),
    ("EAR_362", "EAR_362_norm", "EAR_362"),
    ("MAR", "MAR_norm", "MAR"),
    ("yaw_proxy", "yaw_proxy_norm", "yaw_proxy"),
    ("luma_mediana_0_255", "luma_mediana_0_255_norm", "luminancia"),
]
# Selección basada en vecindad temporal confirmada para A/B y contraste clásico
# frente a C. Las etiquetas finales no modifican los registros de origen.
SELECTION = {
    "A": ("A", 777),
    "B": ("A", 778),
    "C": ("C", 2555),
}


def quantize_eight_bits(value):
    """Cuantización declarada: entero más cercano de 0 a 255; .5 hacia arriba."""
    if not 0 <= value <= 1:
        raise ValueError(f"Valor normalizado fuera de [0,1]: {value}")
    integer = math.floor(255 * value + 0.5)
    return integer, format(integer, "08b")


def euclidean(left, right):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(left, right)))


def main():
    if not INPUT.is_file():
        raise FileNotFoundError("Faltan parámetros normalizados. Ejecuta antes 04_normalizar_parametros.py.")
    source = list(csv.DictReader(INPUT.open(encoding="utf-8", newline="")))
    final_rows, vectors = [], {}
    for final_name, (origin, frame) in SELECTION.items():
        matches = [row for row in source if row["registro"] == origin
                   and int(row["frame_indice_base_cero"]) == frame]
        if len(matches) != 1:
            raise ValueError(f"No se encontró de forma única {final_name}: {origin}_{frame}.")
        row = matches[0]
        values = [float(row[normalized]) for _, normalized, _ in FEATURES]
        integers_and_bits = [quantize_eight_bits(value) for value in values]
        vectors[final_name] = values
        final_rows.append({
            "dato_final": final_name,
            "secuencia_origen": origin,
            "frame_indice_base_cero": frame,
            "tiempo_s": row["tiempo_s"],
            "archivo": row["archivo"],
            **{f"{label}_crudo": row[raw] for raw, _, label in FEATURES},
            **{f"{label}_normalizado": value for (_, _, label), value in zip(FEATURES, values)},
            **{f"{label}_entero_8bits": integer for (_, _, label), (integer, _) in zip(FEATURES, integers_and_bits)},
            **{f"{label}_bitstring": bits for (_, _, label), (_, bits) in zip(FEATURES, integers_and_bits)},
            "bitstring_40bits": "".join(bits for _, bits in integers_and_bits),
            "bitstring_agrupado": " | ".join(bits for _, bits in integers_and_bits),
        })

    distances = []
    for left, right in (("A", "B"), ("A", "C"), ("B", "C")):
        distances.append({"comparacion": f"{left}–{right}",
                          "distancia_euclidiana_5_parametros": euclidean(vectors[left], vectors[right])})
    if not distances[0]["distancia_euclidiana_5_parametros"] < min(
        distances[1]["distancia_euclidiana_5_parametros"], distances[2]["distancia_euclidiana_5_parametros"]
    ):
        raise AssertionError("La selección no cumple que A–B sea el par clásicamente más cercano.")

    OUTPUT_DIR.mkdir(exist_ok=True)
    with (OUTPUT_DIR / "tres_datos_bitstrings.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=final_rows[0].keys())
        writer.writeheader()
        writer.writerows(final_rows)
    with (OUTPUT_DIR / "distancias_clasicas_finales.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=distances[0].keys())
        writer.writeheader()
        writer.writerows(distances)
    manifest = {
        "purpose": "Tres datos finales para la comparación cuántica solicitada.",
        "selection": {
            "A": "A_0777 (31,08 s): primer frame del par temporalmente cercano.",
            "B": "A_0778 (31,12 s): frame consecutivo, 0,04 s después de A.",
            "C": "C_2555 (102,20 s): otro episodio, con patrón de parámetros distinto.",
        },
        "selection_status": "Ilustrativa y declarada; no representa una muestra clínica ni se modificaron valores.",
        "classical_representation": "Cinco valores normalizados cuantizados individualmente a enteros de 8 bits; concatenación de 40 bits.",
        "quantization": "floor(255*x + 0.5), sin modificar x; bit más significativo a la izquierda.",
        "feature_order": [label for _, _, label in FEATURES],
        "expected_order": "A–B debe ser la menor distancia clásica; A–C y B–C deben ser mayores antes de calcular estados cuánticos.",
        "classical_distances": distances,
    }
    (OUTPUT_DIR / "procedencia.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Tres datos finales preparados con bitstrings de 40 bits.")
    for row in final_rows:
        print(f"{row['dato_final']} = {row['secuencia_origen']}_{row['frame_indice_base_cero']}: {row['bitstring_agrupado']}")
    for row in distances:
        print(f"d({row['comparacion']}) = {row['distancia_euclidiana_5_parametros']:.6f}")


if __name__ == "__main__":
    main()
