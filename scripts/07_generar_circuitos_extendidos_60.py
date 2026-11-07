"""Paso 7: extensión H + Ry + CNOT para los mismos 60 frames.

Mantiene intactos los circuitos producto del paso 6. Esta es otra codificación,
no una mejora demostrada de detección.
"""

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
OUTPUT_DIR = ROOT / "circuitos_60_extendidos"
IMAGE_DIR = OUTPUT_DIR / "imagenes"
FEATURES = [
    ("EAR_33_norm", "EAR (contorno 33)"),
    ("EAR_362_norm", "EAR (contorno 362)"),
    ("MAR_norm", "MAR"),
    ("yaw_proxy_norm", "Orientación horizontal"),
    ("luma_mediana_0_255_norm", "Luminancia facial"),
]


def make_extended_circuit(values, name):
    """Feature map: H en todos, Ry(pi*x) y CNOT lineal q0→q1→...→q4."""
    register = QuantumRegister(5, "q")
    circuit = QuantumCircuit(register, name=name)
    for qubit in range(5):
        circuit.h(qubit)
    circuit.barrier(label="superposición")
    for qubit, value in enumerate(values):
        if not 0 <= value <= 1:
            raise ValueError(f"Valor fuera de [0,1]: {value}")
        circuit.ry(math.pi * value, qubit, label=fr"R_y($\pi\cdot${value:.3f})")
    circuit.barrier(label="parámetros")
    for control in range(4):
        circuit.cx(control, control + 1)
    return circuit


def main():
    if not INPUT.is_file():
        raise FileNotFoundError("Faltan parámetros normalizados. Ejecuta antes 04_normalizar_parametros.py.")
    rows = list(csv.DictReader(INPUT.open(encoding="utf-8", newline="")))
    if len(rows) != 60:
        raise ValueError(f"Se esperaban 60 frames, no {len(rows)}.")
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    metadata = []
    for position, row in enumerate(rows, start=1):
        identifier = f"{row['registro']}_{int(row['frame_indice_base_cero']):04d}"
        values = [float(row[key]) for key, _ in FEATURES]
        circuit = make_extended_circuit(values, f"Extend_{identifier}")
        state = Statevector.from_instruction(circuit)
        norm_squared = float(sum(abs(amplitude) ** 2 for amplitude in state.data))
        if not state.is_valid() or not math.isclose(norm_squared, 1.0, abs_tol=1e-12):
            raise RuntimeError(f"Estado no normalizado para {identifier}.")
        image_path = IMAGE_DIR / f"circuito_extendido_{identifier}.png"
        figure = circuit.draw(output="mpl", fold=-1, style={"name": "iqp"})
        figure.suptitle(f"Feature map extendido: {identifier} | H + $R_y(\\pi x)$ + CNOT", y=1.03, fontsize=12)
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

    with (OUTPUT_DIR / "circuitos_extendidos_60_metadata.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=metadata[0].keys())
        writer.writeheader()
        writer.writerows(metadata)
    manifest = {
        "purpose": "Paso 7: codificación extendida de cada frame, sin mediciones ni fidelidades.",
        "feature_map": [
            "Inicializar los cinco qubits en |0>.",
            "Aplicar H a q0, q1, q2, q3 y q4.",
            "Aplicar Ry(pi*x_k) al qubit asociado al parámetro k.",
            "Aplicar CNOT lineales q0→q1, q1→q2, q2→q3 y q3→q4.",
        ],
        "qubit_mapping": [{"qubit": f"q_{index}", "parameter": label}
                          for index, (_, label) in enumerate(FEATURES)],
        "interpretation": "Es una segunda codificación. H y CNOT cambian los estados y, por ello, las fidelidades respecto de la codificación producto simple. No añade datos ni demuestra mejor detección.",
        "input_file": INPUT.relative_to(ROOT).as_posix(),
        "input_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        "circuits": len(metadata), "qubits_per_circuit": 5, "state_dimension": 32,
        "state_validation": "Norma al cuadrado comprobada como 1 para cada frame.",
    }
    (OUTPUT_DIR / "procedencia.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Generación terminada: 60 circuitos extendidos, 60 PNG y 60 estados normalizados.")


if __name__ == "__main__":
    main()
