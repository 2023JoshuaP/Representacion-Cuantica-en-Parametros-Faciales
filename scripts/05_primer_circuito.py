"""Paso 5: prepara y dibuja el circuito de cinco qubits para un frame ejemplo."""

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # genera PNG sin requerir interfaz gráfica
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector


ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "parametros_60" / "parametros_normalizados_60.csv"
OUTPUT_DIR = ROOT / "circuitos_frame"

# El orden se fija aquí y se mantendrá para todos los 60 circuitos posteriores.
FEATURES = [
    ("EAR_33_norm", "EAR (contorno 33)"),
    ("EAR_362_norm", "EAR (contorno 362)"),
    ("MAR_norm", "MAR"),
    ("yaw_proxy_norm", "Orientación horizontal"),
    ("luma_mediana_0_255_norm", "Luminancia facial"),
]


def select_example(rows):
    """Primer frame de A: A_0770, elegido por orden temporal, no por resultado."""
    selected = [row for row in rows if row["registro"] == "A" and row["orden_en_secuencia"] == "1"]
    if len(selected) != 1:
        raise ValueError("No se encontró de forma única el primer frame de A.")
    return selected[0]


def make_circuit(values):
    register = QuantumRegister(len(values), "q")
    circuit = QuantumCircuit(register, name="Frame_A_0770")
    for qubit, value in enumerate(values):
        if not 0 <= value <= 1:
            raise ValueError(f"El valor normalizado debe pertenecer a [0,1], no {value}.")
        theta = math.pi * value
        circuit.ry(theta, qubit, label=fr"R_y($\pi\cdot${value:.3f})")
    return circuit


def main():
    if not INPUT.is_file():
        raise FileNotFoundError("Faltan parámetros normalizados. Ejecuta antes 04_normalizar_parametros.py.")
    rows = list(csv.DictReader(INPUT.open(encoding="utf-8", newline="")))
    row = select_example(rows)
    values = [float(row[key]) for key, _ in FEATURES]
    circuit = make_circuit(values)
    state = Statevector.from_instruction(circuit)
    if not state.is_valid():
        raise RuntimeError("El estado resultante no está normalizado.")
    OUTPUT_DIR.mkdir(exist_ok=True)
    identifier = f"{row['registro']}_{int(row['frame_indice_base_cero']):04d}"
    figure = circuit.draw(output="mpl", fold=-1, style={"name": "iqp"})
    figure.suptitle(
        f"Codificación angular de {identifier}  |  $\\theta_k=\\pi x_k$",
        y=1.05,
        fontsize=14,
    )
    figure.savefig(OUTPUT_DIR / f"circuito_{identifier}.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    payload = {
        "purpose": "Paso 5: circuito ilustrativo de un frame; no hay mediciones ni comparación todavía.",
        "frame": {
            "registro": row["registro"], "orden_en_secuencia": int(row["orden_en_secuencia"]),
            "frame_indice_base_cero": int(row["frame_indice_base_cero"]),
            "tiempo_s": float(row["tiempo_s"]), "archivo": row["archivo"],
        },
        "encoding": "Un qubit por parámetro: Ry(pi*x), donde x pertenece a [0,1].",
        "qubits": [
            {"qubit": f"q_{index}", "parameter": label, "normalized_value": value,
             "theta_rad": math.pi * value}
            for index, ((_, label), value) in enumerate(zip(FEATURES, values))
        ],
        "state_dimension": len(state.data),
        "state_norm_squared": float(sum(abs(amplitude) ** 2 for amplitude in state.data)),
    }
    (OUTPUT_DIR / f"circuito_{identifier}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Circuito generado para {identifier}: {circuit.num_qubits} qubits.")
    print(f"Norma al cuadrado del estado: {payload['state_norm_squared']:.12f}")
    for item in payload["qubits"]:
        print(f"{item['qubit']}: {item['parameter']} = {item['normalized_value']:.6f}; "
              f"theta = {item['theta_rad']:.6f} rad")


if __name__ == "__main__":
    main()
