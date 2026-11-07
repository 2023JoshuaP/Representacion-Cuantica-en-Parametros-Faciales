"""Paso 11: genera los tres circuitos extendidos H + Ry + CNOT principales."""

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector


ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "tres_datos_finales" / "tres_datos_bitstrings.csv"
OUTPUT_DIR = ROOT / "tres_datos_finales" / "circuitos_extendidos"
FEATURES = [
    ("EAR_33_normalizado", "EAR (contorno 33)"),
    ("EAR_362_normalizado", "EAR (contorno 362)"),
    ("MAR_normalizado", "MAR"),
    ("yaw_proxy_normalizado", "Orientación horizontal"),
    ("luminancia_normalizado", "Luminancia facial"),
]


def extended_circuit(values, name):
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
        raise FileNotFoundError("Faltan los tres datos finales. Ejecuta antes 09_preparar_tres_datos_finales.py.")
    records = list(csv.DictReader(INPUT.open(encoding="utf-8", newline="")))
    if [record["dato_final"] for record in records] != ["A", "B", "C"]:
        raise ValueError("Se esperaban A, B y C.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = []
    for record in records:
        values = [float(record[key]) for key, _ in FEATURES]
        label = record["dato_final"]
        circuit = extended_circuit(values, f"Extend_Dato_{label}")
        state = Statevector.from_instruction(circuit)
        norm_squared = float(sum(abs(amplitude) ** 2 for amplitude in state.data))
        if not state.is_valid() or not math.isclose(norm_squared, 1.0, abs_tol=1e-12):
            raise RuntimeError(f"Estado inválido para {label}.")
        image = OUTPUT_DIR / f"circuito_extendido_dato_{label}.png"
        figure = circuit.draw(output="mpl", fold=-1, style={"name": "iqp"})
        figure.suptitle(
            f"Dato {label}: {record['secuencia_origen']}_{int(record['frame_indice_base_cero']):04d} | H + $R_y(\\pi x)$ + CNOT",
            y=1.03, fontsize=12,
        )
        figure.savefig(image, dpi=180, bbox_inches="tight")
        plt.close(figure)
        metadata.append({
            "dato_final": label,
            "origen": f"{record['secuencia_origen']}_{record['frame_indice_base_cero']}",
            "tiempo_s": record["tiempo_s"],
            **{f"q{index}_{name}": value for index, ((_, name), value) in enumerate(zip(FEATURES, values))},
            "dimension_estado": len(state.data), "norma_cuadrada": norm_squared,
            "imagen_circuito": image.relative_to(ROOT).as_posix(),
        })
        print(f"Dato {label}: circuito extendido de 5 qubits, norma² = {norm_squared:.12f}")
    with (OUTPUT_DIR / "circuitos_extendidos_principales_metadata.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=metadata[0].keys())
        writer.writeheader()
        writer.writerows(metadata)
    (OUTPUT_DIR / "procedencia.json").write_text(json.dumps({
        "purpose": "Paso 11: tres circuitos extendidos principales.",
        "operations": ["H en los cinco qubits", "Ry(pi*x_k) por parámetro",
                       "CNOT q0→q1→q2→q3→q4"],
        "note": "Se compararán por fidelidad y se contrastarán contra la codificación producto simple.",
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Tres circuitos extendidos principales generados.")


if __name__ == "__main__":
    main()
