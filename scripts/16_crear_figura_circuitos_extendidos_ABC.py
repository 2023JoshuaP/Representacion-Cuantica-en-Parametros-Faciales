"""Compone los tres circuitos extendidos principales para el informe."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "tres_datos_finales" / "circuitos_extendidos"
OUTPUT_DIR = ROOT / "figuras_informe"


def main():
    names = ("A", "B", "C")
    images = []
    for name in names:
        path = SOURCE_DIR / f"circuito_extendido_dato_{name}.png"
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    width, height = images[0].size
    gap, label_height = 18, 28
    sheet = Image.new("RGB", (2 * width + gap, 2 * (height + label_height) + gap), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    positions = [(0, 0), (width + gap, 0), ((width + gap) // 2, height + label_height + gap)]
    for name, image, (x, y) in zip(names, images, positions):
        sheet.paste(image, (x, y))
        draw.text((x + 8, y + height + 7), f"Circuito extendido: dato {name}", fill="black", font=font)
    OUTPUT_DIR.mkdir(exist_ok=True)
    output = OUTPUT_DIR / "circuitos_extendidos_ABC.png"
    sheet.save(output, quality=95)
    print(output)


if __name__ == "__main__":
    main()
