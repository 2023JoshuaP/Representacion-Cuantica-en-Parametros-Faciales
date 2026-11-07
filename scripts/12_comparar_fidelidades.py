"""Paso 12: compara A, B y C por fidelidad cuántica exacta en simulación."""

import csv
import json
import math
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector, state_fidelity


ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "tres_datos_finales" / "tres_datos_bitstrings.csv"
OUTPUT_DIR = ROOT / "tres_datos_finales" / "comparacion_cuantica"
FEATURES = ["EAR_33_normalizado", "EAR_362_normalizado", "MAR_normalizado",
            "yaw_proxy_normalizado", "luminancia_normalizado"]
FEATURE_LABELS = ["EAR_33", "EAR_362", "MAR", "yaw_proxy", "luminancia"]


def simple_circuit(values, name):
    register = QuantumRegister(5, "q")
    circuit = QuantumCircuit(register, name=name)
    for qubit, value in enumerate(values):
        circuit.ry(math.pi * value, qubit)
    return circuit


def extended_circuit(values, name):
    register = QuantumRegister(5, "q")
    circuit = QuantumCircuit(register, name=name)
    for qubit in range(5):
        circuit.h(qubit)
    for qubit, value in enumerate(values):
        circuit.ry(math.pi * value, qubit)
    for control in range(4):
        circuit.cx(control, control + 1)
    return circuit


def main():
    if not INPUT.is_file():
        raise FileNotFoundError("Faltan los tres datos finales. Ejecuta antes 09_preparar_tres_datos_finales.py.")
    records = list(csv.DictReader(INPUT.open(encoding="utf-8", newline="")))
    if [record["dato_final"] for record in records] != ["A", "B", "C"]:
        raise ValueError("Se esperaban los tres datos A, B y C.")
    values = {record["dato_final"]: [float(record[key]) for key in FEATURES] for record in records}
    states_simple = {name: Statevector.from_instruction(simple_circuit(vector, f"Simple_{name}"))
                     for name, vector in values.items()}
    states_extended = {name: Statevector.from_instruction(extended_circuit(vector, f"Extend_{name}"))
                       for name, vector in values.items()}
    results, factors = [], []
    for left, right in combinations(("A", "B", "C"), 2):
        factor_values = [math.cos(math.pi * (x - y) / 2) ** 2
                         for x, y in zip(values[left], values[right])]
        theory = math.prod(factor_values)
        simple_qiskit = float(state_fidelity(states_simple[left], states_simple[right]))
        simple_inner = float(abs(states_simple[left].inner(states_simple[right])) ** 2)
        extended_qiskit = float(state_fidelity(states_extended[left], states_extended[right]))
        np.testing.assert_allclose([simple_qiskit, simple_inner, extended_qiskit],
                                   [theory, theory, theory], rtol=0, atol=1e-12)
        results.append({
            "comparacion": f"{left}–{right}",
            "F_teorica_producto": theory,
            "F_simple_state_fidelity": simple_qiskit,
            "F_simple_producto_interno_cuadrado": simple_inner,
            "F_extendida_state_fidelity": extended_qiskit,
            "diferencia_extendida_menos_simple": extended_qiskit - simple_qiskit,
        })
        for feature, factor in zip(FEATURE_LABELS, factor_values):
            factors.append({"comparacion": f"{left}–{right}", "parametro": feature,
                            "fidelidad_del_parametro": factor})

    # Verifica explícitamente la expectativa declarada desde la distancia clásica.
    result_by_pair = {row["comparacion"]: row for row in results}
    if not result_by_pair["A–B"]["F_simple_state_fidelity"] > max(
        result_by_pair["A–C"]["F_simple_state_fidelity"],
        result_by_pair["B–C"]["F_simple_state_fidelity"],
    ):
        raise AssertionError("La fidelidad no conserva la expectativa de cercanía A–B.")

    OUTPUT_DIR.mkdir(exist_ok=True)
    with (OUTPUT_DIR / "fidelidades.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    with (OUTPUT_DIR / "fidelidad_por_parametro.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=factors[0].keys())
        writer.writeheader()
        writer.writerows(factors)

    labels = [row["comparacion"] for row in results]
    simple_values = [row["F_simple_state_fidelity"] for row in results]
    extended_values = [row["F_extendida_state_fidelity"] for row in results]
    positions = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    axis.bar(positions - 0.17, simple_values, 0.34, label="Simple: $R_y(\\pi x)$", color="#1f77b4")
    axis.bar(positions + 0.17, extended_values, 0.34, label="Extendida: H + $R_y$ + CNOT", color="#ff7f0e", alpha=0.72)
    for position, value in zip(positions - 0.17, simple_values):
        axis.text(position, value + 0.025, f"{value:.6f}", ha="center", va="bottom", fontsize=9)
    axis.set(xticks=positions, xticklabels=labels, ylim=(0, 1.12), ylabel="Fidelidad",
             title="Fidelidad exacta entre los tres estados de cinco qubits")
    axis.axhline(1, color="gray", linewidth=0.8, linestyle=":")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(OUTPUT_DIR / "fidelidades.png", dpi=180)
    plt.close(figure)

    manifest = {
        "purpose": "Paso 12: comparación exacta por fidelidad de estados puros en simulación clásica.",
        "method": "Qiskit state_fidelity(state1, state2), comprobada con |<psi|phi>|^2.",
        "simple_formula": "Producto sobre k de cos^2(pi*(x_k-y_k)/2).",
        "extended_result": "La fidelidad extendida coincide con la simple para estos circuitos; las H y CNOT fijas no alteran la fidelidad en esta codificación.",
        "expectation": "A–B, seleccionado como par cercanamente temporal y clásicamente próximo, debe presentar mayor fidelidad que A–C y B–C.",
        "execution": "Statevector ideal; no hardware cuántico ni ruido físico.",
    }
    (OUTPUT_DIR / "procedencia.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Fidelidades calculadas y verificadas:")
    for row in results:
        print(f"{row['comparacion']}: F = {row['F_simple_state_fidelity']:.12f} "
              f"(extendida = {row['F_extendida_state_fidelity']:.12f})")


if __name__ == "__main__":
    main()
