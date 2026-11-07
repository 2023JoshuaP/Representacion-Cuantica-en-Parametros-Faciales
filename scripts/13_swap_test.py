"""Paso 13: estima la fidelidad con Swap Test sobre los estados extendidos.

Es una simulación ideal mediante Statevector. No se ejecuta en hardware cuántico.
"""

import csv
import json
import math
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity


ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "tres_datos_finales" / "tres_datos_bitstrings.csv"
OUTPUT_DIR = ROOT / "tres_datos_finales" / "swap_test"
FEATURES = ["EAR_33_normalizado", "EAR_362_normalizado", "MAR_normalizado",
            "yaw_proxy_normalizado", "luminancia_normalizado"]
SHOTS = 8192
SEED = 20261107


def prepare_extended(circuit, qubits, values):
    """Prepara H + Ry(pi*x) + CNOT lineal en los cinco qubits indicados."""
    for qubit in qubits:
        circuit.h(qubit)
    for qubit, value in zip(qubits, values):
        circuit.ry(math.pi * value, qubit)
    for index in range(len(qubits) - 1):
        circuit.cx(qubits[index], qubits[index + 1])


def extended_state(values):
    circuit = QuantumCircuit(5)
    prepare_extended(circuit, list(range(5)), values)
    return Statevector.from_instruction(circuit)


def swap_test(values_left, values_right, name):
    """Qubit 0 auxiliar; 1–5 registro izquierdo; 6–10 registro derecho."""
    circuit = QuantumCircuit(11, name=name)
    auxiliary = 0
    left = list(range(1, 6))
    right = list(range(6, 11))
    prepare_extended(circuit, left, values_left)
    prepare_extended(circuit, right, values_right)
    circuit.barrier(label="estados")
    circuit.h(auxiliary)
    for left_qubit, right_qubit in zip(left, right):
        circuit.cswap(auxiliary, left_qubit, right_qubit)
    circuit.h(auxiliary)
    return circuit


def main():
    if not INPUT.is_file():
        raise FileNotFoundError("Faltan los tres datos finales. Ejecuta antes 09_preparar_tres_datos_finales.py.")
    records = list(csv.DictReader(INPUT.open(encoding="utf-8", newline="")))
    if [record["dato_final"] for record in records] != ["A", "B", "C"]:
        raise ValueError("Se esperaban A, B y C.")
    values = {record["dato_final"]: [float(record[key]) for key in FEATURES] for record in records}
    states = {name: extended_state(vector) for name, vector in values.items()}
    OUTPUT_DIR.mkdir(exist_ok=True)
    rows = []
    for pair_index, (left, right) in enumerate(combinations(("A", "B", "C"), 2)):
        circuit = swap_test(values[left], values[right], f"Swap_{left}_{right}")
        state = Statevector.from_instruction(circuit)
        p0_exact = float(state.probabilities(qargs=[0])[0])
        fidelity_exact = float(state_fidelity(states[left], states[right]))
        np.testing.assert_allclose(2 * p0_exact - 1, fidelity_exact, atol=1e-12, rtol=0)
        state.seed(SEED + pair_index)
        counts = state.sample_counts(shots=SHOTS, qargs=[0])
        n0 = int(counts.get("0", 0))
        p0_estimated = n0 / SHOTS
        fidelity_estimated = 2 * p0_estimated - 1
        standard_error_approx = 2 * math.sqrt(p0_estimated * (1 - p0_estimated) / SHOTS)
        rows.append({
            "comparacion": f"{left}–{right}",
            "qubits_totales": circuit.num_qubits,
            "F_exacta": fidelity_exact,
            "P0_exacta": p0_exact,
            "N0": n0,
            "shots": SHOTS,
            "P0_estimada": p0_estimated,
            "F_estimada_swap_test": fidelity_estimated,
            "error_estandar_aproximado": standard_error_approx,
        })
        if left == "A" and right == "B":
            figure = circuit.draw(output="mpl", fold=-1, style={"name": "iqp"})
            figure.suptitle("Swap Test extendido para comparar A y B", y=1.02, fontsize=13)
            figure.savefig(OUTPUT_DIR / "swap_test_A_B.png", dpi=160, bbox_inches="tight")
            plt.close(figure)

    with (OUTPUT_DIR / "resultados_swap_test.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    labels = [row["comparacion"] for row in rows]
    exact = [row["F_exacta"] for row in rows]
    estimated = [row["F_estimada_swap_test"] for row in rows]
    positions = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    axis.bar(positions - 0.18, exact, 0.36, label="Fidelidad exacta", color="#1f77b4")
    axis.bar(positions + 0.18, estimated, 0.36, label=f"Swap Test ({SHOTS} shots)", color="#ff7f0e")
    for position, value in zip(positions - 0.18, exact):
        axis.text(position, value + 0.025, f"{value:.5f}", ha="center", va="bottom", fontsize=9)
    axis.set(xticks=positions, xticklabels=labels, ylim=(-0.08, 1.12), ylabel="Fidelidad",
             title="Fidelidad exacta y estimación mediante Swap Test")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(OUTPUT_DIR / "comparacion_swap_test.png", dpi=180)
    plt.close(figure)

    manifest = {
        "purpose": "Paso 13: estimación simulada de fidelidad con Swap Test.",
        "registers": "1 qubit auxiliar + 5 qubits para el primer estado + 5 para el segundo = 11 qubits.",
        "formula": "P(0)=(1+F)/2; F_estimada=2*N0/shots-1.",
        "state_preparation": "Codificación extendida H + Ry(pi*x) + CNOT en cada registro de cinco qubits.",
        "shots": SHOTS,
        "seed": SEED,
        "execution": "Statevector ideal y muestreo de probabilidades; no hardware cuántico ni ruido físico.",
    }
    (OUTPUT_DIR / "procedencia.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Swap Test completado:")
    for row in rows:
        print(f"{row['comparacion']}: F exacta={row['F_exacta']:.6f}; "
              f"F estimada={row['F_estimada_swap_test']:.6f}; N0={row['N0']}/{SHOTS}")


if __name__ == "__main__":
    main()
