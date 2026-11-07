"""Paso 10: genera los tres circuitos principales de la entrega."""

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
OUTPUT_DIR = ROOT / "tres_datos_finales" / "circuitos"
FEATURES = [
    ("EAR_33_normalizado", "EAR (contorno 33)"),
    ("EAR_362_normalizado", "EAR (contorno 362)"),
    ("MAR_normalizado", "MAR"),
    ("yaw_proxy_normalizado", "Orientación horizontal"),
    ("luminancia_normalizado", "Luminancia facial"),
]


def circuit_from_values(values, name):
    register = QuantumRegister(5, "q")
    circuit = QuantumCircuit(register, name=name)
    for qubit, value in enumerate(values):
        if not 0 <= value <= 1:
            raise ValueError(f"Valor fuera de [0,1]: {value}")
        circuit.ry(math.pi * value, qubit, label=fr"R_y($\pi\cdot${value:.3f})")
    return circuit


def main():
    if not INPUT.is_file():
        raise FileNotFoundError("Faltan los tres datos finales. Ejecuta antes 09_preparar_tres_datos_finales.py.")
    records = list(csv.DictReader(INPUT.open(encoding="utf-8", newline="")))
    if [record["dato_final"] for record in records] != ["A", "B", "C"]:
        raise ValueError("Se esperaban exactamente los datos finales A, B y C.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = []
    for record in records:
        values = [float(record[key]) for key, _ in FEATURES]
        label = record["dato_final"]
        circuit = circuit_from_values(values, f"Dato_{label}")
        state = Statevector.from_instruction(circuit)
        norm_squared = float(sum(abs(amplitude) ** 2 for amplitude in state.data))
        if not state.is_valid() or not math.isclose(norm_squared, 1.0, abs_tol=1e-12):
            raise RuntimeError(f"Estado inválido para {label}.")
        image = OUTPUT_DIR / f"circuito_dato_{label}.png"
        figure = circuit.draw(output="mpl", fold=-1, style={"name": "iqp"})
        figure.suptitle(
            f"Dato {label}: {record['secuencia_origen']}_{int(record['frame_indice_base_cero']):04d} | $\\theta_k=\\pi x_k$",
            y=1.05, fontsize=13,
        )
        figure.savefig(image, dpi=180, bbox_inches="tight")
        plt.close(figure)
        metadata.append({
            "dato_final": label,
            "origen": f"{record['secuencia_origen']}_{record['frame_indice_base_cero']}",
            "tiempo_s": record["tiempo_s"],
            **{f"q{index}_{name}": value for index, ((_, name), value) in enumerate(zip(FEATURES, values))},
            **{f"q{index}_theta_rad": math.pi * value for index, value in enumerate(values)},
            "dimension_estado": len(state.data),
            "norma_cuadrada": norm_squared,
            "imagen_circuito": image.relative_to(ROOT).as_posix(),
        })
        print(f"Dato {label}: circuito de 5 qubits, norma² = {norm_squared:.12f}")

    with (OUTPUT_DIR / "circuitos_principales_metadata.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=metadata[0].keys())
        writer.writeheader()
        writer.writerows(metadata)
    manifest = {
        "purpose": "Paso 10: los tres circuitos principales para comparar fidelidad.",
        "encoding": "q_k = Ry(pi*x_k)|0>, con x_k normalizado y orden fijo de cinco parámetros.",
        "qubit_mapping": [{"qubit": f"q_{index}", "parameter": label}
                          for index, (_, label) in enumerate(FEATURES)],
        "note": "No se aplican puertas H, CNOT ni mediciones; la codificación producto permite interpretación directa.",
        "state_dimension": 32,
    }
    (OUTPUT_DIR / "procedencia.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Tres circuitos principales generados.")


if __name__ == "__main__":
    main()
