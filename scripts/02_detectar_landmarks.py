"""
Paso 2: detecta y revisa landmarks faciales en los 60 frames extraídos.
"""

import csv
import hashlib
import json
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = ROOT / "frames_20"
METADATA = FRAMES_DIR / "frames_20_metadata.csv"
MODEL_DIR = ROOT / "modelos"
MODEL_PATH = MODEL_DIR / "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
OUTPUT_DIR = ROOT / "landmarks_60"

# Índices oficiales del modelo Face Landmarker, usados solo para visualización.
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 13, 291, 14]


def ensure_model():
    """Obtiene el bundle oficial y deja registrada su huella SHA-256."""
    MODEL_DIR.mkdir(exist_ok=True)
    if not MODEL_PATH.is_file():
        print("Descargando el modelo oficial FaceLandmarker (una sola vez)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    digest = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    (MODEL_DIR / "face_landmarker.sha256").write_text(
        f"{digest}  {MODEL_PATH.name}\n", encoding="utf-8"
    )
    return digest


def draw_polyline(image, landmarks, indices, color, closed=True):
    height, width = image.shape[:2]
    points = [(round(landmarks[i].x * (width - 1)), round(landmarks[i].y * (height - 1)))
              for i in indices]
    for start, end in zip(points, points[1:] + ([points[0]] if closed else [])):
        cv2.line(image, start, end, color, 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(image, point, 2, color, -1, cv2.LINE_AA)


def create_contact_sheet(paths: list[Path], output: Path):
    """Crea una hoja 5x4 para comprobar detección y contornos visualmente."""
    width, height, label_height = 320, 180, 24
    sheet = Image.new("RGB", (5 * width, 4 * (height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for position, path in enumerate(paths):
        with Image.open(path) as image:
            image = image.resize((width, height), Image.Resampling.LANCZOS)
            x = (position % 5) * width
            y = (position // 5) * (height + label_height)
            sheet.paste(image, (x, y))
        name = path.stem.replace("_landmarks", "")
        draw.text((x + 5, y + height + 5), name, fill="black", font=font)
    sheet.save(output, quality=88)


def main():
    if not METADATA.is_file():
        raise FileNotFoundError("Falta frames_20_metadata.csv. Ejecuta primero 01_extraer_frames.py.")
    model_sha256 = ensure_model()
    output_images = OUTPUT_DIR / "imagenes_anotadas"
    output_images.mkdir(parents=True, exist_ok=True)
    metadata = list(csv.DictReader(METADATA.open(encoding="utf-8", newline="")))
    if len(metadata) != 60:
        raise ValueError(f"Se esperaban 60 frames; se encontraron {len(metadata)}.")

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        # Los 60 frames están ordenados cronológicamente. VIDEO permite el
        # seguimiento temporal sin omitir frames, adecuado para esta secuencia.
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        output_facial_transformation_matrixes=True,
    )

    results, summary, outputs_by_sequence = [], [], {"A": [], "B": [], "C": []}
    with FaceLandmarker.create_from_options(options) as detector:
        for row in metadata:
            image_path = ROOT / row["archivo"]
            mp_image = mp.Image.create_from_file(str(image_path))
            timestamp_ms = round(int(row["frame_indice_base_cero"]) * 1000 / 25)
            result = detector.detect_for_video(mp_image, timestamp_ms)
            number_of_faces = len(result.face_landmarks)
            valid = number_of_faces == 1 and len(result.face_landmarks[0]) == 478
            summary.append({
                **row,
                "caras_detectadas": number_of_faces,
                "landmarks_por_rostro": len(result.face_landmarks[0]) if number_of_faces else 0,
                "deteccion_valida": valid,
            })
            if not valid:
                continue

            landmarks = result.face_landmarks[0]
            results.append({
                "registro": row["registro"],
                "orden_en_secuencia": int(row["orden_en_secuencia"]),
                "frame_indice_base_cero": int(row["frame_indice_base_cero"]),
                "tiempo_s": float(row["tiempo_s"]),
                "archivo": row["archivo"],
                "landmarks": [{"x": point.x, "y": point.y, "z": point.z} for point in landmarks],
            })
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"No se pudo abrir {image_path}")
            draw_polyline(image, landmarks, FACE_OVAL, (255, 255, 0))
            draw_polyline(image, landmarks, LEFT_EYE, (0, 255, 0))
            draw_polyline(image, landmarks, RIGHT_EYE, (0, 255, 0))
            draw_polyline(image, landmarks, MOUTH, (255, 0, 255))
            destination = output_images / f"{Path(row['archivo']).stem}_landmarks.jpg"
            if not cv2.imwrite(str(destination), image):
                raise ValueError(f"No se pudo guardar {destination}")
            outputs_by_sequence[row["registro"]].append(destination)

    with (OUTPUT_DIR / "resumen_deteccion.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    (OUTPUT_DIR / "landmarks_normalizados.json").write_text(
        json.dumps(results, ensure_ascii=False), encoding="utf-8"
    )
    for sequence, paths in outputs_by_sequence.items():
        if len(paths) != 20:
            raise RuntimeError(f"{sequence}: solo {len(paths)} detecciones válidas; no se calcularán parámetros.")
        create_contact_sheet(paths, OUTPUT_DIR / f"{sequence}_landmarks_contact_sheet.jpg")
    manifest = {
        "purpose": "Paso 2: detección de landmarks, sin cálculo de características.",
        "model_url": MODEL_URL,
        "model_file": MODEL_PATH.relative_to(ROOT).as_posix(),
        "model_sha256": model_sha256,
        "mediapipe_version": mp.__version__,
        "configuration": {
            "running_mode": "VIDEO",
            "timestamp_rule": "frame_index * 1000 / 25, en orden cronológico",
            "num_faces": 1,
            "min_face_detection_confidence": 0.5,
            "min_face_presence_confidence": 0.5,
            "expected_landmarks_per_face": 478,
        },
        "visualized_contours": {"face_oval": FACE_OVAL, "left_eye": LEFT_EYE,
                                "right_eye": RIGHT_EYE, "mouth": MOUTH},
    }
    (OUTPUT_DIR / "procedencia.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Detección completada: 60/60 frames con un rostro y 478 landmarks.")
    print("Revisa A_landmarks_contact_sheet.jpg, B_landmarks_contact_sheet.jpg y C_landmarks_contact_sheet.jpg.")


if __name__ == "__main__":
    main()
