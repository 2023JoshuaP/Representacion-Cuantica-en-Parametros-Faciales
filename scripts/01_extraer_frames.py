"""
Paso 1: extrae las tres secuencias visualmente confirmadas.
"""

import csv
import hashlib
import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT.parent / (
    "DROWSINESS DETECTION/code/datasets/NITYMED/DSM_Dataset-HDTV720/"
    "Microsleep/Male/HDTV720/P1042756_720.mp4"
)

# Inclusivo: cada secuencia contiene exactamente 20 fotogramas consecutivos.
SEQUENCES = {
    "A": {"first_frame": 770, "last_frame": 789, "window": "30.8–31.6 s"},
    "B": {"first_frame": 1000, "last_frame": 1019, "window": "40.0–40.8 s"},
    "C": {"first_frame": 2555, "last_frame": 2574, "window": "102.2–103.0 s"},
}


def video_metadata():
    output = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=avg_frame_rate,r_frame_rate,width,height,nb_frames,start_time",
        "-of", "json", str(VIDEO),
    ], text=True)
    stream = json.loads(output)["streams"][0]
    fps = Fraction(stream["avg_frame_rate"])
    if fps != 25 or Fraction(stream["r_frame_rate"]) != fps:
        raise ValueError(f"Se esperaba video de 25 FPS; se encontró {fps} FPS.")
    if stream["start_time"] != "0.000000":
        raise ValueError("Se esperaba que el video iniciara en t=0.")
    return stream, fps


def extract_frame(frame_index: int, destination: Path):
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", str(VIDEO), "-vf", f"select=eq(n\\,{frame_index})",
        "-frames:v", "1", "-q:v", "2", str(destination),
    ], check=True)


def contact_sheet(sequence: str, paths: list[Path], output: Path):
    thumb_width, thumb_height = 256, 144
    columns, label_height = 5, 24
    rows = 4
    sheet = Image.new("RGB", (columns * thumb_width, rows * (thumb_height + label_height)), "white")
    font = ImageFont.load_default()
    for position, path in enumerate(paths):
        with Image.open(path) as image:
            thumbnail = image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        x = (position % columns) * thumb_width
        y = (position // columns) * (thumb_height + label_height)
        sheet.paste(thumbnail, (x, y))
        draw = ImageDraw.Draw(sheet)
        frame = int(path.stem.rsplit("_", 1)[1])
        draw.text((x + 5, y + thumb_height + 5), f"{sequence}-{position + 1:02d} | n={frame} | {frame / 25:.2f} s", fill="black", font=font)
    sheet.save(output, quality=88)


def main():
    if not VIDEO.is_file():
        raise FileNotFoundError(f"No se encontró el video: {VIDEO}")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("Se requieren ffmpeg y ffprobe para este paso.")

    stream, fps = video_metadata()
    output_dir = ROOT / "frames_20"
    output_dir.mkdir(exist_ok=True)
    rows = []

    for sequence, specification in SEQUENCES.items():
        indices = range(specification["first_frame"], specification["last_frame"] + 1)
        if len(indices) != 20:
            raise AssertionError(f"{sequence} debe contener 20 frames.")
        paths = []
        for ordinal, index in enumerate(indices, start=1):
            path = output_dir / f"{sequence}_{index:04d}.jpg"
            extract_frame(index, path)
            with Image.open(path) as image:
                if image.size != (stream["width"], stream["height"]):
                    raise ValueError(f"Resolución inesperada en {path.name}: {image.size}")
            paths.append(path)
            rows.append({
                "registro": sequence,
                "orden_en_secuencia": ordinal,
                "frame_indice_base_cero": index,
                "tiempo_s": f"{index / float(fps):.2f}",
                "ventana_confirmada": specification["window"],
                "archivo": path.relative_to(ROOT).as_posix(),
                "sha256_jpeg": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
        contact_sheet(sequence, paths, output_dir / f"{sequence}_contact_sheet.jpg")

    with (output_dir / "frames_20_metadata.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "purpose": "Paso 1: extracción de 20 frames consecutivos por registro para análisis facial.",
        "source_video": str(VIDEO.relative_to(ROOT.parent)),
        "stream": stream,
        "fps": str(fps),
        "frame_indexing": "Base cero; t = indice / FPS.",
        "sequences": SEQUENCES,
        "note": "Las ventanas fueron confirmadas visualmente por el investigador; este script no reclasifica episodios.",
    }
    (output_dir / "procedencia.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Extracción terminada: {len(rows)} frames en {output_dir}")
    print("Revisa A_contact_sheet.jpg, B_contact_sheet.jpg y C_contact_sheet.jpg antes del paso 2.")


if __name__ == "__main__":
    main()
