# Representación cuántica de parámetros faciales

Actividad de **Trabajo Interdisciplinar III**. A partir de tres frames del video
NITYMED, el proyecto extrae cinco parámetros faciales, los codifica en estados
cuánticos simulados y compara los datos A, B y C mediante fidelidad.

> Es una simulación ideal con Qiskit `Statevector`; no usa hardware cuántico,
> no diagnostica somnolencia y no demuestra ventaja cuántica ni mejora del
> modelo de tesis.

## Estructura

```text
scripts/                 Pipeline reproducible (01–16) y ejecutor principal.
frames_20/               60 frames extraídos del video fuente.
landmarks_60/            Landmarks y anotaciones visuales de los frames.
parametros_60/           Parámetros crudos, normalizados y sus gráficas.
tres_datos_finales/      Bitstrings, circuitos, fidelidades y Swap Test de A/B/C.
figuras_informe/         Figuras compuestas usadas en el informe.
docs/latex/              Carátula, informe LaTeX y PDF final.
recursos/                Logo institucional usado en la carátula.
legacy/duracion_anterior/ Enfoque anterior por duración; no es la entrega final.
```

Los directorios `circuitos_60/`, `circuitos_60_extendidos/`, `circuitos_frame/`
y `tres_datos/` contienen resultados exploratorios intermedios del pipeline.

## Datos finales

| Dato | Frame | Tiempo | Rol |
|---|---:|---:|---|
| A | `A_0777` | 31,08 s | Frame de referencia |
| B | `A_0778` | 31,12 s | Frame consecutivo y cercano a A |
| C | `C_2555` | 102,20 s | Frame de contraste |

Cada frame se resume con `EAR_33`, `EAR_362`, `MAR`, `yaw_proxy` y luminancia
facial. Tras normalizar los valores a `[0,1]`, cada parámetro se cuantiza a 8
bits: cada dato clásico tiene 40 bits. La codificación cuántica principal usa
cinco qubits con `Ry(pi*x)`. También se muestra una versión extendida con
`H + Ry + CNOT` y una verificación adicional mediante Swap Test.

Resultados exactos de fidelidad:

| Comparación | Fidelidad |
|---|---:|
| A–B | 0,991998 |
| A–C | 0,002385 |
| B–C | 0,001624 |

## Instalación y ejecución

Desde la raíz del repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/ejecutar_pipeline.py
```

El paso 1 requiere `ffmpeg` y `ffprobe`, así como el video fuente de la tesis
en la ruta declarada en `scripts/01_extraer_frames.py`. Los pasos posteriores
registran resultados en los directorios indicados arriba. Para reproducir las
versiones verificadas localmente, se puede usar `requirements-lock.txt`.

## Informe

El PDF de entrega es:

```text
docs/latex/entrega_final_con_caratula.pdf
```

Para compilarlo nuevamente se necesita una distribución LaTeX con `latexmk`:

```bash
cd docs/latex
latexmk -pdf -interaction=nonstopmode -halt-on-error caratula.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error informe_final.tex
pdfunite caratula.pdf informe_final.pdf entrega_final_con_caratula.pdf
```

## Límites metodológicos

- A y B fueron escogidos por proximidad temporal; C ilustra contraste clásico.
- `yaw_proxy` es una posición horizontal relativa de la nariz, no un ángulo en grados.
- La luminancia caracteriza la imagen, no la somnolencia.
- La selección de tres frames es didáctica y no representa validación clínica ni generalización.
- Los archivos del video fuente y de las anotaciones de la tesis no se modifican.
