"""Crea una figura compacta de los tres frames finales y sus landmarks."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "figuras_informe"
DATA = [
    ("A", "A_0777", "31,08 s"),
    ("B", "A_0778", "31,12 s"),
    ("C", "C_2555", "102,20 s"),
]


def main():
    width, height, header = 420, 236, 28
    sheet = Image.new("RGB", (3 * width, 2 * (height + header) + 30), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for column, (data, identifier, timestamp) in enumerate(DATA):
        raw = ROOT / "frames_20" / f"{identifier}.jpg"
        marked = ROOT / "landmarks_60" / "imagenes_anotadas" / f"{identifier}_landmarks.jpg"
        for row, path in enumerate((raw, marked)):
            with Image.open(path) as image:
                image = image.resize((width, height), Image.Resampling.LANCZOS)
            x, y = column * width, row * (height + header)
            sheet.paste(image, (x, y))
            label = (f"Dato {data}: {identifier}, {timestamp}" if row == 0
                     else "Landmarks: rostro, ojos y boca")
            draw.text((x + 6, y + height + 7), label, fill="black", font=font)
    draw.text((8, 2 * (height + header) + 8),
              "Fila superior: frame original. Fila inferior: óvalo facial (celeste), ojos (verde) y boca (magenta).",
              fill="black", font=font)
    OUTPUT_DIR.mkdir(exist_ok=True)
    destination = OUTPUT_DIR / "frames_landmarks_datos_ABC.png"
    sheet.save(destination, quality=90)
    print(destination)


if __name__ == "__main__":
    main()
