"""Paso 6: genera los 60 circuitos de cinco qubits y sus imágenes Matplotlib."""

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector


ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "parametros_60" / "parametros_normalizados_60.csv"
OUTPUT_DIR = ROOT / "circuitos_60"
IMAGE_DIR = OUTPUT_DIR / "imagenes"
FEATURES = [
    ("EAR_33_norm", "EAR (contorno 33)"),
    ("EAR_362_norm", "EAR (contorno 362)"),
    ("MAR_norm", "MAR"),
    ("yaw_proxy_norm", "Orientación horizontal"),
    ("luma_mediana_0_255_norm", "Luminancia facial"),
]


def make_circuit(values, name):
    register = QuantumRegister(5, "q")
    circuit = QuantumCircuit(register, name=name)
    for qubit, value in enumerate(values):
        if not 0 <= value <= 1:
            raise ValueError(f"Valor fuera de [0,1]: {value}")
        circuit.ry(math.pi * value, qubit, label=fr"R_y($\pi\cdot${value:.3f})")
    return circuit


def main():
    if not INPUT.is_file():
        raise FileNotFoundError("Faltan parámetros normalizados. Ejecuta antes 04_normalizar_parametros.py.")
    rows = list(csv.DictReader(INPUT.open(encoding="utf-8", newline="")))
    if len(rows) != 60:
        raise ValueError(f"Se esperaban 60 frames, no {len(rows)}.")
    if [row["registro"] for row in rows].count("A") != 20 or [row["registro"] for row in rows].count("B") != 20 or [row["registro"] for row in rows].count("C") != 20:
        raise ValueError("Se esperaban 20 frames por registro.")
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    metadata = []
    for position, row in enumerate(rows, start=1):
        identifier = f"{row['registro']}_{int(row['frame_indice_base_cero']):04d}"
        values = [float(row[key]) for key, _ in FEATURES]
        circuit = make_circuit(values, f"Frame_{identifier}")
        state = Statevector.from_instruction(circuit)
        norm_squared = float(sum(abs(amplitude) ** 2 for amplitude in state.data))
        if not state.is_valid() or not math.isclose(norm_squared, 1.0, abs_tol=1e-12):
            raise RuntimeError(f"Estado no normalizado para {identifier}.")
        image_path = IMAGE_DIR / f"circuito_{identifier}.png"
        figure = circuit.draw(output="mpl", fold=-1, style={"name": "iqp"})
        figure.suptitle(f"{identifier} | $\\theta_k=\\pi x_k$", y=1.03, fontsize=12)
        figure.savefig(image_path, dpi=130, bbox_inches="tight")
        plt.close(figure)
        metadata.append({
            "registro": row["registro"],
            "orden_en_secuencia": row["orden_en_secuencia"],
            "frame_indice_base_cero": row["frame_indice_base_cero"],
            "tiempo_s": row["tiempo_s"],
            "archivo_frame": row["archivo"],
            **{f"q{index}_{key}": value for index, ((key, _), value) in enumerate(zip(FEATURES, values))},
            **{f"q{index}_theta_rad": math.pi * value for index, value in enumerate(values)},
            "dimension_estado": len(state.data),
            "norma_cuadrada": norm_squared,
            "imagen_circuito": image_path.relative_to(ROOT).as_posix(),
            "sha256_imagen": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        })
        print(f"[{position:02d}/60] {identifier}")

    with (OUTPUT_DIR / "circuitos_60_metadata.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=metadata[0].keys())
        writer.writeheader()
        writer.writerows(metadata)
    manifest = {
        "purpose": "Paso 6: un circuito producto de 5 qubits por cada uno de los 60 frames; sin medidas ni fidelidades.",
        "encoding": "q_k = Ry(pi*x_k)|0>, con valores normalizados de parametros_normalizados_60.csv.",
        "qubit_mapping": [{"qubit": f"q_{index}", "parameter": label}
                          for index, (_, label) in enumerate(FEATURES)],
        "input_file": INPUT.relative_to(ROOT).as_posix(),
        "input_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        "circuits": len(metadata),
        "qubits_per_circuit": 5,
        "state_dimension": 32,
        "state_validation": "Norma al cuadrado comprobada como 1 para cada frame.",
    }
    (OUTPUT_DIR / "procedencia.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Generación terminada: 60 circuitos, 60 PNG y 60 estados normalizados.")


if __name__ == "__main__":
    main()
