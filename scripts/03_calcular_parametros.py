"""
Paso 3: calcula cinco parámetros faciales crudos de los 60 frames.
"""

import csv
import json
import math
from pathlib import Path
from statistics import fmean, stdev

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
LANDMARKS_JSON = ROOT / "landmarks_60" / "landmarks_normalizados.json"
OUTPUT_DIR = ROOT / "parametros_60"

EYE_33 = [33, 160, 158, 133, 153, 144]
EYE_362 = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 13, 291, 14]
NOSE_TIP = 1
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


def point_px(landmarks, index, width, height):
    point = landmarks[index]
    return np.array([point["x"] * (width - 1), point["y"] * (height - 1)], dtype=float)


def distance(a, b):
    return float(np.linalg.norm(a - b))


def ear(landmarks, indices, width, height):
    """Eye Aspect Ratio: (|p2-p6| + |p3-p5|) / (2|p1-p4|)."""
    p1, p2, p3, p4, p5, p6 = [point_px(landmarks, i, width, height) for i in indices]
    horizontal = distance(p1, p4)
    if horizontal <= 0:
        raise ValueError("Ancho de ojo nulo; no se puede calcular EAR.")
    return (distance(p2, p6) + distance(p3, p5)) / (2 * horizontal)


def mar(landmarks, width, height):
    """Mouth Aspect Ratio: distancia vertical labios / ancho de boca."""
    corner_1, upper, corner_2, lower = [point_px(landmarks, i, width, height) for i in MOUTH]
    horizontal = distance(corner_1, corner_2)
    if horizontal <= 0:
        raise ValueError("Ancho de boca nulo; no se puede calcular MAR.")
    return distance(upper, lower) / horizontal


def yaw_proxy(landmarks, width, height):
    """Posición horizontal relativa de nariz entre extremos oculares, no grados."""
    nose = point_px(landmarks, NOSE_TIP, width, height)[0]
    eye_33 = point_px(landmarks, 33, width, height)[0]
    eye_362 = point_px(landmarks, 362, width, height)[0]
    left, right = sorted((eye_33, eye_362))
    if right - left <= 0:
        raise ValueError("Distancia interocular horizontal nula.")
    return (nose - left) / (right - left)


def face_luma_median(image, landmarks):
    """Mediana de Y=0.2126R+0.7152G+0.0722B dentro del óvalo facial."""
    height, width = image.shape[:2]
    polygon = np.array(
        [[round(landmarks[i]["x"] * (width - 1)), round(landmarks[i]["y"] * (height - 1))]
         for i in FACE_OVAL], dtype=np.int32
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    blue, green, red = cv2.split(image)
    luma = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    values = luma[mask == 255]
    if values.size == 0:
        raise ValueError("Máscara facial vacía.")
    return float(np.median(values)), int(values.size)


def main():
    if not LANDMARKS_JSON.is_file():
        raise FileNotFoundError("Faltan landmarks. Ejecuta antes 02_detectar_landmarks.py.")
    records = json.loads(LANDMARKS_JSON.read_text(encoding="utf-8"))
    if len(records) != 60:
        raise ValueError(f"Se esperaban 60 registros de landmarks; se encontraron {len(records)}.")
    OUTPUT_DIR.mkdir(exist_ok=True)
    rows = []
    for record in records:
        image_path = ROOT / record["archivo"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"No se pudo abrir {image_path}")
        height, width = image.shape[:2]
        points = record["landmarks"]
        if len(points) != 478:
            raise ValueError(f"{image_path.name}: se esperaban 478 landmarks.")
        luma, face_pixels = face_luma_median(image, points)
        row = {
            "registro": record["registro"],
            "orden_en_secuencia": record["orden_en_secuencia"],
            "frame_indice_base_cero": record["frame_indice_base_cero"],
            "tiempo_s": record["tiempo_s"],
            "archivo": record["archivo"],
            "EAR_33": ear(points, EYE_33, width, height),
            "EAR_362": ear(points, EYE_362, width, height),
            "MAR": mar(points, width, height),
            "yaw_proxy": yaw_proxy(points, width, height),
            "luma_mediana_0_255": luma,
            "pixeles_en_mascara_facial": face_pixels,
        }
        if not all(math.isfinite(float(value)) for key, value in row.items()
                   if key not in {"registro", "archivo"}):
            raise ValueError(f"Valor no finito en {image_path.name}")
        rows.append(row)

    raw_path = OUTPUT_DIR / "parametros_crudos_60.csv"
    with raw_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    features = ["EAR_33", "EAR_362", "MAR", "yaw_proxy", "luma_mediana_0_255"]
    summary = []
    for sequence in ("A", "B", "C"):
        subset = [row for row in rows if row["registro"] == sequence]
        for feature in features:
            values = [float(row[feature]) for row in subset]
            summary.append({"registro": sequence, "parametro": feature, "n": len(values),
                            "media": fmean(values), "desviacion_estandar": stdev(values),
                            "minimo": min(values), "maximo": max(values)})
    with (OUTPUT_DIR / "resumen_parametros.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    figure, axes = plt.subplots(len(features), 1, figsize=(10, 12), sharex=True, constrained_layout=True)
    colors = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}
    for axis, feature in zip(axes, features):
        for sequence in ("A", "B", "C"):
            subset = [row for row in rows if row["registro"] == sequence]
            axis.plot([row["orden_en_secuencia"] for row in subset],
                      [row[feature] for row in subset], marker="o", markersize=3,
                      linewidth=1.3, label=sequence, color=colors[sequence])
        axis.set_ylabel(feature)
        axis.grid(alpha=0.25)
        axis.legend(title="Registro", ncol=3, loc="best")
    axes[-1].set_xlabel("Orden dentro de la secuencia (1–20)")
    figure.suptitle("Parámetros crudos por frame; sin normalización", fontsize=13)
    figure.savefig(OUTPUT_DIR / "series_parametros_crudos.png", dpi=180)
    plt.close(figure)

    manifest = {
        "purpose": "Paso 3: extracción de parámetros crudos, sin normalización ni codificación cuántica.",
        "parameters": {
            "EAR_33": "EAR con landmarks [33,160,158,133,153,144].",
            "EAR_362": "EAR con landmarks [362,385,387,263,373,380].",
            "MAR": "distancia entre labios 13–14 / ancho de boca 61–291.",
            "yaw_proxy": "posición x de nariz 1, relativa al intervalo horizontal entre 33 y 362; no es yaw en grados.",
            "luma_mediana_0_255": "mediana de Y=0.2126R+0.7152G+0.0722B dentro del óvalo facial.",
        },
        "coordinates": "Las distancias geométricas se calcularon en píxeles del frame 1280x720.",
        "status": "Pendiente de revisión de rangos antes de normalizar.",
    }
    (OUTPUT_DIR / "procedencia.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Parámetros calculados sin normalizar: {len(rows)} frames.")
    print(f"Tabla: {raw_path}")
    print(f"Gráfica: {OUTPUT_DIR / 'series_parametros_crudos.png'}")


if __name__ == "__main__":
    main()
