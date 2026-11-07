"""Compone los tres circuitos principales en una sola figura para el informe."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "tres_datos_finales" / "circuitos"
OUTPUT_DIR = ROOT / "figuras_informe"


def main():
    names = ("A", "B", "C")
    images = []
    for name in names:
        path = SOURCE_DIR / f"circuito_dato_{name}.png"
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    gap, label_height = 18, 26
    width = sum(image.width for image in images) + gap * (len(images) - 1)
    height = max(image.height for image in images) + label_height
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    x = 0
    for name, image in zip(names, images):
        sheet.paste(image, (x, 0))
        draw.text((x + 8, image.height + 7), f"Dato {name}", fill="black", font=font)
        x += image.width + gap
    OUTPUT_DIR.mkdir(exist_ok=True)
    output = OUTPUT_DIR / "circuitos_principales_ABC.png"
    sheet.save(output, quality=95)
    print(output)


if __name__ == "__main__":
    main()
