#!/usr/bin/env python3

# Quantum_Pauli_fermion_boson


"""
QUANTUM BOSON V2 — Loto 7/39
Python 3.11
Qiskit 1.4.4
Dependencies: qiskit, numpy, scipy

Two quantum orbitals:
    u = U(theta)|0>
    v = U(theta)|1>

Bosonic occupations:
    four particles in u;
    three particles in v.

For a legal seven-label set S:
    c(S) = coefficient of z^3 in product(u_i + z*v_i)

The corresponding permanent is:
    permanent = 4! * 3! * c(S)

History-controlled quantum filter:
    h(x) = CAP * tanh(x W / CAP)
    K_i(x) = exp(h_i(x) - max(h(x)))

The accepted ancilla=1 branch implements K.
Illegal basis labels have zero accepted amplitude.

Conditional Loto distribution:
    P(S|x) = |c_K(S)|^2 / Z(x)

Z sums over all legal sets of seven different labels.
Bosonic symmetry does not itself prohibit repeated labels.
Distinct Loto labels are an explicit conditioning event.

Qiskit executes orbital circuits during optimization.
Training evaluates the quantum filter analytically.
NEXT executes and verifies the explicit Qiskit filter.

Permanent contractions and normalization use a compact exact
representation. No dense 42-qubit statevector is allocated.

No classifier, random sampling, noise, row shuffling,
duplicate repair or cosmetic combination filters.
"""


import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import csv
import hashlib
import math
import sys
import traceback

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, islice, permutations
from pathlib import Path

import numpy as np
import qiskit

from scipy.optimize import minimize
from scipy.special import logsumexp

from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate
from qiskit.quantum_info import Statevector


FILES = (
    Path("/data/loto7_4710_k86.csv"),
    Path("/data/loto7_4710_k86_loto_2977.csv"),
    Path("/data/loto7_4710_k86_loto_plus_1733.csv"),
)


NUMBER_COUNT = 39
DRAW_SIZE = 7
U_COUNT = 4
V_COUNT = 3

FEATURE_DIM = 8
CAP = 4.0
QUANTUM_DEPTH = 2
THETA_RIDGE_RATIO = 0.01

CANDIDATES = (
    (1, 0.03),
    (1, 0.30),
    (3, 0.03),
    (3, 0.30),
)

INITIAL_ORDERS = (17, 11)

OPTIMIZER_STEPS = 600
OPTIMIZER_CONTINUATIONS = 3
GRADIENT_LIMIT = 5e-4

TRAIN_BATCH = 256
MAP_BATCH = 4096
NORMALIZATION_TOLERANCE = 2e-7


def say(*lines):
    for line in lines:
        print(line, flush=True)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def checked_controlled_ry(angle, controls, control_state):
    angle = float(angle)

    require(
        math.isfinite(angle)
        and controls >= 1
        and 0 <= control_state < (1 << controls),
        "Neispravna kontrolisana RY kapija.",
    )

    cosine = math.cos(angle / 2.0)
    sine = math.sin(angle / 2.0)

    require(
        abs(cosine * cosine + sine * sine - 1.0) < 1e-12,
        "Osnovna RY rotacija nije numerički unitarna.",
    )

    # Qiskit 1.4 internally calls a determinant during synthesis.
    # Handle numerical backend flags only inside this construction.
    # The resulting filter amplitudes are checked independently.
    with np.errstate(
        divide="ignore",
        invalid="ignore",
        over="raise",
        under="ignore",
    ):
        gate = RYGate(angle).control(
            num_ctrl_qubits=controls,
            ctrl_state=int(control_state),
        )

    require(
        gate.num_qubits == controls + 1,
        "Pogrešan broj kubita kontrolisane RY kapije.",
    )

    return gate


class QuantumOrbitals:
    def __init__(self, number_count, depth):
        require(
            number_count >= 2 and depth >= 1,
            "Neispravne dimenzije kvantnog modela.",
        )

        self.m = int(number_count)
        self.depth = int(depth)
        self.nq = max(1, (self.m - 1).bit_length())
        self.dimension = 1 << self.nq
        self.parameter_count = 2 * self.nq * self.depth

        self.fourier = self.make_fourier()

        self.inputs = (
            Statevector.from_int(0, self.dimension),
            Statevector.from_int(1, self.dimension),
        )

    def make_fourier(self):
        circuit = QuantumCircuit(self.nq, name="QFT")

        for upper in range(self.nq - 1, -1, -1):
            circuit.h(upper)

            for lower in range(upper - 1, -1, -1):
                circuit.cp(
                    math.pi / (2 ** (upper - lower)),
                    upper,
                    lower,
                )

        for lower in range(self.nq // 2):
            circuit.swap(lower, self.nq - 1 - lower)

        return circuit

    def circuit(self, theta):
        theta = np.asarray(theta, dtype=float)

        require(
            theta.shape == (self.parameter_count,)
            and np.all(np.isfinite(theta)),
            "Neispravni kvantni parametri.",
        )

        circuit = QuantumCircuit(self.nq, name="BosonOrbitals")

        circuit.compose(
            self.fourier,
            qubits=list(range(self.nq)),
            inplace=True,
        )

        position = 0

        for layer in range(self.depth):
            for qubit in range(self.nq):
                circuit.ry(float(theta[position]), qubit)
                position += 1

            for qubit in range(self.nq):
                circuit.rz(float(theta[position]), qubit)
                position += 1

            if layer % 2 == 0:
                for qubit in range(self.nq - 1):
                    circuit.cx(qubit, qubit + 1)
            else:
                for qubit in range(self.nq - 1, 0, -1):
                    circuit.cx(qubit, qubit - 1)

        require(
            position == self.parameter_count,
            "Neusaglašena konstrukcija kvantnog kola.",
        )

        return circuit

    def full_amplitudes(self, theta):
        circuit = self.circuit(theta)

        result = np.column_stack(
            [
                np.asarray(
                    state.evolve(circuit).data,
                    dtype=np.complex128,
                )
                for state in self.inputs
            ]
        )

        require(
            np.all(np.isfinite(result)),
            "Qiskit je vratio nevažeće amplitude.",
        )

        require(
            np.max(np.abs(
                result.conj().T @ result - np.eye(2)
            )) < 2e-9,
            "Qiskit izlazne orbitale nisu ortonormalne.",
        )

        return result

    def basis(self, theta):
        # Projection is part of the legal-label postselection.
        return self.full_amplitudes(theta)[:self.m, :]

    def amplitude_derivative(self, theta, parameter):
        # Exact amplitude derivative for one RY/RZ occurrence.
        positive = np.asarray(theta, dtype=float).copy()
        negative = positive.copy()

        positive[parameter] += math.pi
        negative[parameter] -= math.pi

        return (
            self.basis(positive) - self.basis(negative)
        ) / 4.0

    def filtered_amplitudes(self, theta, h):
        h = np.asarray(h, dtype=float)

        require(
            h.shape == (self.m,)
            and np.all(np.isfinite(h)),
            "Neispravan istorijski amplitudni filter.",
        )

        factors = np.exp(h - float(np.max(h)))

        require(
            np.all(factors > 0)
            and np.all(factors <= 1.0),
            "Amplitudni filter nije kontrakcija.",
        )

        circuit = QuantumCircuit(self.nq + 1, name="BosonFilter")

        circuit.compose(
            self.circuit(theta),
            qubits=list(range(self.nq)),
            inplace=True,
        )

        wires = list(range(self.nq + 1))

        for label, factor in enumerate(factors):
            angle = 2.0 * math.asin(float(factor))

            gate = checked_controlled_ry(
                angle,
                self.nq,
                label,
            )

            circuit.append(gate, wires)

        columns = []
        illegal_maximum = 0.0

        for orbital in range(2):
            state = Statevector.from_int(
                orbital,
                2 * self.dimension,
            ).evolve(circuit)

            require(
                np.all(np.isfinite(state.data)),
                "Qiskit filter je vratio nevažeće stanje.",
            )

            require(
                abs(float(np.vdot(
                    state.data, state.data
                ).real) - 1.0) < 2e-9,
                "Qiskit stanje filtera nije normalizovano.",
            )

            accepted = np.asarray(
                state.data[self.dimension:],
                dtype=np.complex128,
            )

            columns.append(accepted[:self.m])

            if self.m < self.dimension:
                illegal_maximum = max(
                    illegal_maximum,
                    float(np.max(np.abs(accepted[self.m:]))),
                )

        result = np.column_stack(columns)
        reference = factors[:, None] * self.basis(theta)

        error = float(np.max(np.abs(result - reference)))

        require(
            illegal_maximum < 1e-10,
            "Nelegalne oznake imaju prihvaćenu amplitudu.",
        )

        require(
            error < 2e-9,
            f"Qiskit amplitudni filter nije prošao: {error:.3e}",
        )

        return result, error


@dataclass(frozen=True)
class BosonSpace:
    u_count: int
    v_count: int
    states: tuple
    edges: tuple
    start: int
    end: int

    @property
    def draw_size(self):
        return self.u_count + self.v_count

    @property
    def size(self):
        return len(self.states)


@lru_cache(maxsize=8)
def make_space(u_count, v_count):
    require(
        u_count >= 1 and v_count >= 1,
        "Obe orbitale moraju imati pozitivno zauzeće.",
    )

    total = u_count + v_count
    states = []

    for level in range(total + 1):
        for a in range(u_count + 1):
            for b in range(v_count + 1):
                if a + b != level:
                    continue

                for c in range(u_count + 1):
                    for d in range(v_count + 1):
                        if c + d == level:
                            states.append((a, b, c, d))

    lookup = {
        state: index
        for index, state in enumerate(states)
    }

    edges = []

    for conjugate_orbital in range(2):
        for ordinary_orbital in range(2):
            sources = []
            destinations = []

            for index, state in enumerate(states):
                destination = list(state)
                destination[conjugate_orbital] += 1
                destination[2 + ordinary_orbital] += 1

                destination_index = lookup.get(tuple(destination))

                if destination_index is not None:
                    sources.append(index)
                    destinations.append(destination_index)

            edges.append(
                (
                    conjugate_orbital,
                    ordinary_orbital,
                    np.asarray(sources, dtype=np.int64),
                    np.asarray(destinations, dtype=np.int64),
                )
            )

    return BosonSpace(
        u_count=u_count,
        v_count=v_count,
        states=tuple(states),
        edges=tuple(edges),
        start=lookup[(0, 0, 0, 0)],
        end=lookup[(u_count, v_count, u_count, v_count)],
    )


def normalize_columns(b):
    norms = np.sqrt(np.sum(np.abs(b) ** 2, axis=0))

    require(
        np.all(np.isfinite(norms))
        and np.all(norms > 1e-12),
        "Orbitala ima nultu ili neispravnu normu.",
    )

    # Independent column scaling cancels after normalization.
    # Do not apply QR: mixing orbitals changes the 4+3 bosonic state.
    return b / norms[None, :]


def boson_amplitude(selected, v_count, with_gradient=False):
    selected = np.asarray(selected, dtype=np.complex128)

    require(
        selected.ndim == 3
        and selected.shape[2] == 2
        and np.all(np.isfinite(selected)),
        "Neispravan ulaz bosonske amplitude.",
    )

    n, k, _ = selected.shape

    require(
        0 <= v_count <= k,
        "Neispravno bosonsko zauzeće.",
    )

    polynomial = np.zeros(
        (n, v_count + 1),
        dtype=np.complex128,
    )
    polynomial[:, 0] = 1.0

    cache = [polynomial] if with_gradient else None

    for position in range(k):
        previous = polynomial

        polynomial = (
            previous * selected[:, position, 0][:, None]
        )

        if v_count:
            polynomial[:, 1:] += (
                previous[:, :-1]
                * selected[:, position, 1][:, None]
            )

        if with_gradient:
            cache.append(polynomial)

    amplitude = polynomial[:, v_count]

    require(
        np.all(np.isfinite(amplitude)),
        "Bosonska amplituda nije konačna.",
    )

    if not with_gradient:
        return amplitude, None

    reverse = np.zeros_like(polynomial)
    reverse[:, v_count] = 1.0

    derivative = np.zeros_like(selected)

    for position in range(k - 1, -1, -1):
        previous = cache[position]

        derivative[:, position, 0] = np.sum(
            reverse * previous,
            axis=1,
        )

        if v_count:
            derivative[:, position, 1] = np.sum(
                reverse[:, 1:] * previous[:, :-1],
                axis=1,
            )

        previous_reverse = (
            reverse * selected[:, position, 0][:, None]
        )

        if v_count:
            previous_reverse[:, :-1] += (
                reverse[:, 1:]
                * selected[:, position, 1][:, None]
            )

        reverse = previous_reverse

    return amplitude, derivative


def boson_partition(b, weights, space, with_gradient=False):
    b = np.asarray(b, dtype=np.complex128)
    weights = np.asarray(weights, dtype=float)

    require(
        b.ndim == 2
        and b.shape[1] == 2
        and weights.ndim == 2
        and weights.shape[1] == len(b)
        and len(weights) > 0
        and np.all(np.isfinite(b))
        and np.all(np.isfinite(weights))
        and np.all(weights > 0),
        "Neispravni ulazi bosonske normalizacije.",
    )

    n, m = weights.shape

    forward = np.zeros(
        (n, space.size),
        dtype=np.complex128,
    )
    forward[:, space.start] = 1.0

    cache = [forward] if with_gradient else None

    for number in range(m):
        following = forward.copy()

        for a, c, sources, destinations in space.edges:
            coefficient = (
                weights[:, number]
                * np.conj(b[number, a])
                * b[number, c]
            )

            following[:, destinations] += (
                forward[:, sources] * coefficient[:, None]
            )

        forward = following

        if with_gradient:
            cache.append(forward)

    complex_z = forward[:, space.end]

    require(
        np.all(np.isfinite(complex_z)),
        "Normalizaciona konstanta nije konačna.",
    )

    z = complex_z.real

    require(
        np.all(z > 0),
        "Normalizaciona konstanta nije pozitivna.",
    )

    require(
        np.all(np.abs(complex_z.imag) <= 1e-8 * z),
        "Normalizacija ima preveliki imaginarni ostatak.",
    )

    log_z = np.log(z)

    if not with_gradient:
        return log_z, None, None

    reverse = np.zeros_like(forward)
    reverse[:, space.end] = 1.0 / z

    gradient_b = np.zeros_like(b)
    occupation = np.zeros((n, m), dtype=float)

    for number in range(m - 1, -1, -1):
        previous = cache[number]
        previous_reverse = reverse.copy()

        for a, c, sources, destinations in space.edges:
            edge_value = np.sum(
                previous[:, sources]
                * reverse[:, destinations],
                axis=1,
            )

            coefficient = (
                weights[:, number]
                * np.conj(b[number, a])
                * b[number, c]
            )

            occupation[:, number] += (
                coefficient * edge_value
            ).real

            gradient_b[number, a] += (
                2.0
                * np.sum(
                    weights[:, number]
                    * b[number, c]
                    * edge_value
                )
            )

            previous_reverse[:, sources] += (
                reverse[:, destinations]
                * coefficient[:, None]
            )

        reverse = previous_reverse

    require(
        np.all(np.isfinite(gradient_b))
        and np.all(np.isfinite(occupation)),
        "Izvod bosonske normalizacije nije konačan.",
    )

    require(
        np.max(np.abs(
            occupation.sum(axis=1) - space.draw_size
        )) < 1e-5,
        "Zbir uslovnih zauzeća nije veličina kombinacije.",
    )

    require(
        np.min(occupation) >= -1e-5
        and np.max(occupation) <= 1.0 + 1e-5,
        "Uslovno zauzeće legalnog skupa je van [0,1].",
    )

    return log_z, gradient_b, occupation


def load_draws(path):
    raw = path.read_bytes()
    fingerprint = hashlib.sha256(raw).hexdigest()

    rows = []
    blank = 0

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for line_number, row in enumerate(csv.reader(stream), 1):
            if not row or all(not item.strip() for item in row):
                blank += 1
                continue

            require(
                len(row) == DRAW_SIZE,
                f"{path.name}, red {line_number}: potrebno je 7 kolona.",
            )

            try:
                numbers = tuple(int(item.strip()) for item in row)
            except ValueError as error:
                raise RuntimeError(
                    f"{path.name}, red {line_number}: neispravan ceo broj."
                ) from error

            require(
                len(set(numbers)) == DRAW_SIZE
                and min(numbers) >= 1
                and max(numbers) <= NUMBER_COUNT,
                f"{path.name}, red {line_number}: neispravna sedmorka.",
            )

            rows.append(
                tuple(value - 1 for value in sorted(numbers))
            )

    require(
        len(rows) >= 100,
        f"{path.name}: nedovoljno istorijskih kola.",
    )

    say(
        "",
        f"ULAZ: {path.name}",
        f"Redova: {len(rows)} | praznih redova: {blank}",
        f"SHA256: {fingerprint}",
        "Redosled kola ostaje redosled iz CSV-a.",
    )

    return np.asarray(rows, dtype=np.int64), fingerprint


def make_context(draws, targets, lookback, number_count=39):
    targets = np.asarray(targets, dtype=np.int64)

    require(
        targets.ndim == 1
        and np.all(targets >= 1)
        and np.all(targets <= len(draws)),
        "Neispravni indeksi istorijskih konteksta.",
    )

    width = number_count + 1
    result = np.zeros(
        (len(targets), lookback * width),
        dtype=float,
    )

    for lag in range(1, lookback + 1):
        previous = targets - lag
        valid = np.flatnonzero(previous >= 0)
        offset = (lag - 1) * width

        if len(valid):
            result[
                valid[:, None],
                offset + draws[previous[valid]],
            ] = 1.0

            result[valid, offset + number_count] = 1.0

    return result


def fit_feature_map(raw):
    mean = raw.mean(axis=0)
    centered = raw - mean

    covariance = centered.T @ centered / len(raw)
    values, vectors = np.linalg.eigh(covariance)

    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]

    threshold = max(float(values[0]), 1.0) * 1e-12
    dimension = min(
        FEATURE_DIM,
        int(np.count_nonzero(values > threshold)),
    )

    require(
        dimension > 0,
        "Istorijski ulazi nemaju promenljive karakteristike.",
    )

    vectors = vectors[:, :dimension].copy()
    values = values[:dimension]

    for column in range(dimension):
        pivot = int(np.argmax(np.abs(vectors[:, column])))

        if vectors[pivot, column] < 0:
            vectors[:, column] *= -1.0

    return mean, vectors / np.sqrt(values)[None, :]


def unpack(parameters, source, feature_count):
    count = source.parameter_count
    theta = parameters[:count]
    w = parameters[count:].reshape(feature_count, source.m)

    return theta, w


def log_amplitudes(x, w):
    t = np.tanh((x @ w) / CAP)
    return CAP * t, 1.0 - t * t


def probability_terms(
    b,
    w,
    x,
    targets,
    space,
    with_gradient=False,
):
    h, derivative = log_amplitudes(x, w)
    weights = np.exp(2.0 * h)

    log_z, normalizer_gradient, occupation = boson_partition(
        b,
        weights,
        space,
        with_gradient,
    )

    amplitude, amplitude_gradient = boson_amplitude(
        b[targets],
        space.v_count,
        with_gradient,
    )

    magnitude = np.abs(amplitude)

    require(
        np.all(np.isfinite(magnitude))
        and np.all(magnitude > 0),
        "Posmatrano kolo ima nultu ili nevažeću amplitudu.",
    )

    indices = np.arange(len(targets))[:, None]

    logp = (
        2.0 * np.log(magnitude)
        + 2.0 * np.sum(h[indices, targets], axis=1)
        - log_z
    )

    require(
        np.all(np.isfinite(logp))
        and np.max(logp) <= 1e-7,
        "Neispravna verovatnoća posmatranog kola.",
    )

    return (
        logp,
        derivative,
        normalizer_gradient,
        occupation,
        amplitude,
        amplitude_gradient,
    )


def objective(
    parameters,
    source,
    x,
    targets,
    ridge,
    space,
    gradient=True,
):
    theta, w = unpack(parameters, source, x.shape[1])

    # Qiskit orbital execution occurs at every objective evaluation.
    b = source.basis(theta)

    n = len(targets)
    total_nll = 0.0

    gradient_b = np.zeros_like(b) if gradient else None
    gradient_w = np.zeros_like(w) if gradient else None

    for start in range(0, n, TRAIN_BATCH):
        stop = min(start + TRAIN_BATCH, n)

        batch_x = x[start:stop]
        batch_targets = targets[start:stop]

        (
            logp,
            derivative,
            normalizer_gradient,
            occupation,
            amplitude,
            amplitude_gradient,
        ) = probability_terms(
            b,
            w,
            batch_x,
            batch_targets,
            space,
            with_gradient=gradient,
        )

        total_nll -= float(np.sum(logp))

        if gradient:
            gradient_b += normalizer_gradient

            selected_gradient = -2.0 * np.conj(
                amplitude_gradient
                / amplitude[:, None, None]
            )

            np.add.at(
                gradient_b,
                batch_targets.ravel(),
                selected_gradient.reshape(-1, 2),
            )

            gradient_h = 2.0 * occupation
            rows = np.arange(len(batch_targets))[:, None]
            gradient_h[rows, batch_targets] -= 2.0

            gradient_w += (
                batch_x.T @ (gradient_h * derivative)
            )

    loss = (
        total_nll / n
        + ridge * float(np.sum(w * w))
        + ridge * THETA_RIDGE_RATIO * float(np.dot(theta, theta))
    )

    require(np.isfinite(loss), "Funkcija cilja nije konačna.")

    if not gradient:
        return loss

    gradient_b /= n

    gradient_w = (
        gradient_w / n + 2.0 * ridge * w
    )

    gradient_theta = np.empty(
        source.parameter_count,
        dtype=float,
    )

    for parameter in range(source.parameter_count):
        db = source.amplitude_derivative(theta, parameter)

        gradient_theta[parameter] = (
            float(np.vdot(gradient_b, db).real)
            + 2.0 * ridge * THETA_RIDGE_RATIO * theta[parameter]
        )

    result = np.concatenate(
        (gradient_theta, gradient_w.ravel())
    )

    require(
        np.all(np.isfinite(result)),
        "Gradijent nije konačan.",
    )

    return loss, result


@dataclass
class Model:
    source: QuantumOrbitals
    theta: np.ndarray
    w: np.ndarray
    b: np.ndarray
    mean: np.ndarray
    projection: np.ndarray
    lookback: int
    ridge: float
    train_loss: float
    gradient_norm: float

    def transform(self, raw):
        return (np.asarray(raw) - self.mean) @ self.projection

    def orbitals(self, raw_row, explicit_filter=False):
        x = self.transform(
            np.asarray(raw_row).reshape(1, -1)
        )

        h, _ = log_amplitudes(x, self.w)

        if explicit_filter:
            amplitudes, error = self.source.filtered_amplitudes(
                self.theta,
                h[0],
            )
        else:
            factors = np.exp(h[0] - float(np.max(h[0])))
            amplitudes = factors[:, None] * self.b
            error = None

        return normalize_columns(amplitudes), error


def fit_model(raw, targets, lookback, ridge, label):
    mean, projection = fit_feature_map(raw)
    x = (raw - mean) @ projection

    source = QuantumOrbitals(NUMBER_COUNT, QUANTUM_DEPTH)
    space = make_space(U_COUNT, V_COUNT)
    fitted = []

    for order in INITIAL_ORDERS:
        indices = np.arange(
            1,
            source.parameter_count + 1,
            dtype=float,
        )

        theta0 = 0.08 * np.sin(indices * order / 13.0)
        w0 = np.zeros((x.shape[1], NUMBER_COUNT), dtype=float)

        parameters = np.concatenate((theta0, w0.ravel()))
        iteration = 0

        say(
            "",
            label,
            f"Lag={lookback} | λ={ridge:g}"
            f" | početne faze={order}"
            f" | prelaza={len(targets)}",
            f"Parametara kola: {source.parameter_count}",
            f"Parametara istorijskog upravljanja: {w0.size}",
        )

        def function(current):
            return objective(
                current,
                source,
                x,
                targets,
                ridge,
                space,
            )

        def callback(current):
            nonlocal iteration
            iteration += 1

            if iteration % 25 == 0:
                say(f"Optimizacija: {iteration} iteracija")

        try:
            converged = False
            result = None
            maximum_gradient = math.inf

            for continuation in range(OPTIMIZER_CONTINUATIONS):
                result = minimize(
                    function,
                    parameters,
                    method="L-BFGS-B",
                    jac=True,
                    callback=callback,
                    options={
                        "maxiter": OPTIMIZER_STEPS,
                        "maxls": 60,
                        "maxcor": 20,
                        "ftol": 1e-13,
                        "gtol": 1e-6,
                    },
                )

                parameters = result.x.copy()
                loss, gradient = function(parameters)

                maximum_gradient = float(
                    np.max(np.abs(gradient))
                )

                if (
                    result.success
                    and maximum_gradient <= GRADIENT_LIMIT
                ):
                    converged = True
                    break

                say(
                    f"Nastavak {continuation + 1}"
                    f" | max|grad|={maximum_gradient:.3e}",
                    str(result.message),
                )

            require(
                converged,
                "Početak nije dostigao uslov konvergencije.",
            )

            theta, w = unpack(parameters, source, x.shape[1])

            fitted.append(
                Model(
                    source=source,
                    theta=theta.copy(),
                    w=w.copy(),
                    b=source.basis(theta).copy(),
                    mean=mean.copy(),
                    projection=projection.copy(),
                    lookback=lookback,
                    ridge=ridge,
                    train_loss=float(loss),
                    gradient_norm=maximum_gradient,
                )
            )

            say(
                f"Prihvaćen lokalni minimum: {loss:.9f}",
                f"Max|grad|: {maximum_gradient:.3e}",
            )

        except (
            RuntimeError,
            FloatingPointError,
            np.linalg.LinAlgError,
        ) as error:
            say(
                f"Početak {order} nije prihvaćen:",
                str(error),
            )

    require(
        bool(fitted),
        f"{label}: nijedan početak nije prošao numeričke uslove.",
    )

    return min(fitted, key=lambda model: model.train_loss)


def evaluate_nll(model, raw, targets):
    space = make_space(U_COUNT, V_COUNT)
    total = 0.0

    for start in range(0, len(targets), TRAIN_BATCH):
        stop = min(start + TRAIN_BATCH, len(targets))
        x = model.transform(raw[start:stop])

        logp = probability_terms(
            model.b,
            model.w,
            x,
            targets[start:stop],
            space,
        )[0]

        total -= float(np.sum(logp))

    return total / len(targets)


def exhaustive_maximum(orbitals, space, label, verbose=True):
    m = len(orbitals)
    k = space.draw_size

    expected = math.comb(m, k)
    iterator = combinations(range(m), k)

    log_z = float(
        boson_partition(
            orbitals,
            np.ones((1, m)),
            space,
        )[0][0]
    )

    visited = 0
    best = None
    best_score = -math.inf
    log_total = -math.inf
    previous_progress = 0

    while True:
        block = list(islice(iterator, MAP_BATCH))

        if not block:
            break

        selected = np.asarray(block, dtype=np.int64)

        amplitude, _ = boson_amplitude(
            orbitals[selected],
            space.v_count,
        )

        magnitude = np.abs(amplitude)
        scores = np.full(len(selected), -np.inf)

        nonzero = magnitude > 0

        scores[nonzero] = (
            2.0 * np.log(magnitude[nonzero]) - log_z
        )

        require(
            not np.any(np.isnan(scores))
            and not np.any(np.isposinf(scores)),
            "Nevažeća verovatnoća tokom pretrage.",
        )

        index = int(np.argmax(scores))
        value = float(scores[index])

        if value > best_score:
            best_score = value
            best = tuple(int(v) for v in selected[index])

        log_total = float(
            np.logaddexp(log_total, logsumexp(scores))
        )

        visited += len(selected)
        progress = visited // 2_000_000

        if verbose and progress > previous_progress:
            say(f"{label}: pregledano {visited:,}/{expected:,}")
            previous_progress = progress

    require(
        visited == expected and best is not None,
        "Pretraga maksimuma nije potpuna.",
    )

    require(
        np.isfinite(best_score)
        and -math.log(expected) - 1e-7 <= best_score <= 1e-7,
        "Neispravna verovatnoća maksimuma.",
    )

    probability_sum = math.exp(log_total)

    require(
        abs(probability_sum - 1.0) <= NORMALIZATION_TOLERANCE,
        f"Normalizacija nije prošla: {probability_sum:.12g}",
    )

    return best, best_score, probability_sum, visited


def verify_two_boson_state():
    source = QuantumOrbitals(6, 1)

    theta = 0.05 * np.sin(
        np.arange(source.parameter_count) + 1.0
    )

    dimension = source.dimension
    initial = np.zeros(dimension * dimension, dtype=complex)

    initial[0 + dimension * 1] = 1.0 / math.sqrt(2.0)
    initial[1 + dimension * 0] = 1.0 / math.sqrt(2.0)

    circuit = QuantumCircuit(2 * source.nq)
    unitary = source.circuit(theta)

    circuit.compose(
        unitary,
        qubits=list(range(source.nq)),
        inplace=True,
    )

    circuit.compose(
        unitary,
        qubits=list(range(source.nq, 2 * source.nq)),
        inplace=True,
    )

    actual = Statevector(initial).evolve(circuit).data
    orbitals = source.full_amplitudes(theta)
    expected = np.empty_like(actual)

    for first in range(dimension):
        for second in range(dimension):
            expected[first + dimension * second] = (
                orbitals[first, 0] * orbitals[second, 1]
                + orbitals[first, 1] * orbitals[second, 0]
            ) / math.sqrt(2.0)

    error = float(np.max(np.abs(actual - expected)))

    require(
        error < 2e-10,
        "Puna dvobosonska Qiskit provera nije prošla.",
    )

    return error


def self_checks():
    say("", "Pokrećem ugrađene Qiskit i bosonske samoprovere.")

    source = QuantumOrbitals(7, 1)
    space = make_space(2, 1)

    theta = 0.07 * np.sin(
        np.arange(source.parameter_count) + 1.0
    )

    dimension = source.dimension
    labels = np.arange(dimension)

    for column in (0, 1, 3):
        actual = Statevector.from_int(
            column,
            dimension,
        ).evolve(source.fourier).data

        expected = (
            np.exp(2j * math.pi * labels * column / dimension)
            / math.sqrt(dimension)
        )

        require(
            np.max(np.abs(actual - expected)) < 2e-10,
            "QFT provera nije prošla.",
        )

    epsilon = 1e-6
    amplitude_error = 0.0

    for parameter in range(source.parameter_count):
        positive = theta.copy()
        negative = theta.copy()

        positive[parameter] += epsilon
        negative[parameter] -= epsilon

        finite = (
            source.basis(positive) - source.basis(negative)
        ) / (2.0 * epsilon)

        exact = source.amplitude_derivative(theta, parameter)

        amplitude_error = max(
            amplitude_error,
            float(np.max(np.abs(finite - exact))),
        )

    require(
        amplitude_error < 2e-7,
        "Provera izvoda kvantnih amplituda nije prošla.",
    )

    x = np.array(
        [[0.2, -0.4], [-0.3, 0.5], [0.7, 0.1]],
        dtype=float,
    )

    targets = np.array(
        [[0, 2, 5], [1, 3, 6], [0, 4, 6]],
        dtype=np.int64,
    )

    w = (
        0.02 * np.sin(np.arange(2 * source.m) + 1.0)
    ).reshape(2, source.m)

    parameters = np.concatenate((theta, w.ravel()))
    ridge = 0.03

    _, analytic = objective(
        parameters,
        source,
        x,
        targets,
        ridge,
        space,
    )

    numerical = np.empty_like(parameters)

    for parameter in range(len(parameters)):
        positive = parameters.copy()
        negative = parameters.copy()

        positive[parameter] += epsilon
        negative[parameter] -= epsilon

        numerical[parameter] = (
            objective(
                positive,
                source,
                x,
                targets,
                ridge,
                space,
                gradient=False,
            )
            - objective(
                negative,
                source,
                x,
                targets,
                ridge,
                space,
                gradient=False,
            )
        ) / (2.0 * epsilon)

    gradient_error = float(np.max(np.abs(analytic - numerical)))

    require(
        gradient_error < 3e-5,
        f"Provera ukupnog gradijenta nije prošla: {gradient_error:.3e}",
    )

    h = np.linspace(-0.3, 0.4, source.m)
    filtered, filter_error = source.filtered_amplitudes(theta, h)
    b = normalize_columns(filtered)

    log_z, _, occupation = boson_partition(
        b,
        np.ones((1, source.m)),
        space,
        with_gradient=True,
    )

    states = list(combinations(range(source.m), space.draw_size))
    indices = np.asarray(states, dtype=np.int64)

    amplitudes, _ = boson_amplitude(
        b[indices],
        space.v_count,
    )

    raw_probabilities = np.abs(amplitudes) ** 2
    direct_z = float(raw_probabilities.sum())

    require(
        abs(float(log_z[0]) - math.log(direct_z)) < 1e-9,
        "Direktna provera normalizacije nije prošla.",
    )

    probabilities = raw_probabilities / direct_z
    exact_occupation = np.zeros(source.m)

    for selected, probability in zip(states, probabilities):
        exact_occupation[np.asarray(selected)] += probability

    require(
        np.max(np.abs(exact_occupation - occupation[0])) < 1e-8,
        "Provera uslovnih zauzeća nije prošla.",
    )

    best, logp, probability_sum, _ = exhaustive_maximum(
        b,
        space,
        "Mala provera",
        verbose=False,
    )

    maximum_index = int(np.argmax(probabilities))

    require(
        best == states[maximum_index]
        and abs(math.exp(logp) - probabilities[maximum_index]) < 1e-9,
        "Provera maksimuma nije prošla.",
    )

    selected = b[indices[:1]]
    original, _ = boson_amplitude(selected, space.v_count)
    reordered, _ = boson_amplitude(
        selected[:, ::-1, :],
        space.v_count,
    )

    require(
        np.max(np.abs(original - reordered)) < 1e-12,
        "Bosonska simetrija nije prošla.",
    )

    # Independent small permanent, without calling the coefficient routine.
    matrix = np.column_stack(
        (
            selected[0, :, 0],
            selected[0, :, 0],
            selected[0, :, 1],
        )
    )

    permanent = 0.0j

    for permutation in permutations(range(3)):
        term = 1.0 + 0.0j

        for row, column in enumerate(permutation):
            term *= matrix[row, column]

        permanent += term

    require(
        abs(
            permanent
            - math.factorial(2) * math.factorial(1) * original[0]
        ) < 1e-11,
        "Veza amplitude i permanenta nije prošla.",
    )

    # Repeated labels are not forbidden by bosonic symmetry.
    repeated, _ = boson_amplitude(
        np.ones((1, 3, 2), dtype=complex),
        1,
    )

    require(
        abs(repeated[0] - 3.0) < 1e-12,
        "Provera bosonskog ponovljenog zauzeća nije prošla.",
    )

    for u_count, v_count in ((1, 1), (2, 1)):
        boundary_space = make_space(u_count, v_count)
        m = boundary_space.draw_size
        boundary_b = np.ones((m, 2), dtype=complex)

        _, _, total, visited = exhaustive_maximum(
            boundary_b,
            boundary_space,
            "Granična provera",
            verbose=False,
        )

        require(
            visited == 1 and abs(total - 1.0) < 1e-10,
            "Granični slučaj jedne legalne kombinacije nije prošao.",
        )

    toy_draws = np.array(
        [[0, 1, 2], [2, 3, 4], [1, 4, 5]],
        dtype=np.int64,
    )

    context = make_context(
        toy_draws,
        [1],
        lookback=3,
        number_count=6,
    )

    require(
        np.count_nonzero(context[0, :6]) == 3
        and context[0, 6] == 1
        and np.all(context[0, 7:] == 0),
        "Provera prvog istorijskog prelaza nije prošla.",
    )

    symmetry_error = verify_two_boson_state()

    say(
        "Ugrađene samoprovere su prošle.",
        f"Greška izvoda amplituda: {amplitude_error:.3e}",
        f"Greška ukupnog gradijenta: {gradient_error:.3e}",
        f"Greška Qiskit filtera: {filter_error:.3e}",
        f"Greška pune dvobosonske simulacije: {symmetry_error:.3e}",
        f"Zbir male raspodele: {probability_sum:.12f}",
    )


def longest_run(numbers):
    best = 1
    current = 1

    for left, right in zip(numbers, numbers[1:]):
        current = current + 1 if right == left + 1 else 1
        best = max(best, current)

    return best


def context_distribution_distance(first, second, space):
    log_z_first = float(
        boson_partition(
            first,
            np.ones((1, len(first))),
            space,
        )[0][0]
    )

    log_z_second = float(
        boson_partition(
            second,
            np.ones((1, len(second))),
            space,
        )[0][0]
    )

    # A fixed deterministic collection of legal subsets.
    # This is a diagnostic only, not a maximum search.
    subsets = np.asarray(
        list(islice(
            combinations(range(len(first)), space.draw_size),
            128,
        )),
        dtype=np.int64,
    )

    amplitude_first, _ = boson_amplitude(
        first[subsets],
        space.v_count,
    )

    amplitude_second, _ = boson_amplitude(
        second[subsets],
        space.v_count,
    )

    probability_first = (
        np.abs(amplitude_first) ** 2 * math.exp(-log_z_first)
    )

    probability_second = (
        np.abs(amplitude_second) ** 2 * math.exp(-log_z_second)
    )

    return float(
        np.max(np.abs(probability_first - probability_second))
    )


def process_file(path, draws, fingerprint):
    targets = draws[1:].copy()
    n = len(targets)

    split60 = int(0.60 * n)
    split70 = int(0.70 * n)
    split80 = int(0.80 * n)

    require(
        50 < split60 < split70 < split80 < n,
        "Nedovoljno podataka za hronološku podelu.",
    )

    contexts = {
        lag: make_context(
            draws,
            np.arange(1, len(draws)),
            lag,
            NUMBER_COUNT,
        )
        for lag in sorted({item[0] for item in CANDIDATES})
    }

    folds = (
        (split60, split70),
        (split70, split80),
    )

    scores = []

    for lookback, ridge in CANDIDATES:
        raw = contexts[lookback]
        total_nll = 0.0
        total_count = 0

        for fold, (train_end, validation_end) in enumerate(folds, 1):
            model = fit_model(
                raw[:train_end],
                targets[:train_end],
                lookback,
                ridge,
                f"{path.name} / validacija {fold}",
            )

            count = validation_end - train_end

            score = evaluate_nll(
                model,
                raw[train_end:validation_end],
                targets[train_end:validation_end],
            )

            total_nll += count * score
            total_count += count

            say(f"Validacioni NLL: {score:.9f}")

        score = total_nll / total_count
        scores.append((score, lookback, ridge))

        say(
            f"KANDIDAT: lag={lookback} | λ={ridge:g}",
            f"Zbirni validacioni NLL: {score:.9f}",
        )

    validation_nll, lookback, ridge = min(
        scores,
        key=lambda item: item[0],
    )

    raw = contexts[lookback]

    say(
        "",
        f"IZABRANO: lag={lookback} | λ={ridge:g}",
        f"Validacioni NLL: {validation_nll:.9f}",
    )

    test_model = fit_model(
        raw[:split80],
        targets[:split80],
        lookback,
        ridge,
        f"{path.name} / odvojeni test",
    )

    test_nll = evaluate_nll(
        test_model,
        raw[split80:],
        targets[split80:],
    )

    say(
        f"ODVOJENI TEST: {n - split80} prelaza",
        f"NLL cele sedmorke: {test_nll:.9f}",
    )

    final_model = fit_model(
        raw,
        targets,
        lookback,
        ridge,
        f"{path.name} / završno učenje",
    )

    say(f"ZAVRŠNO UČENJE: svih {n} istorijskih prelaza.")

    raw_next = make_context(
        draws,
        [len(draws)],
        lookback,
        NUMBER_COUNT,
    )[0]

    orbitals, filter_error = final_model.orbitals(
        raw_next,
        explicit_filter=True,
    )

    space = make_space(U_COUNT, V_COUNT)

    _, _, occupations = boson_partition(
        orbitals,
        np.ones((1, NUMBER_COUNT)),
        space,
        with_gradient=True,
    )

    require(
        abs(float(occupations.sum()) - DRAW_SIZE) < 1e-7,
        "Zbir uslovnih NEXT zauzeća nije sedam.",
    )

    sensitivity = 0.0

    for previous_raw in raw[-8:]:
        previous, _ = final_model.orbitals(previous_raw)

        sensitivity = max(
            sensitivity,
            context_distribution_distance(
                orbitals,
                previous,
                space,
            ),
        )

    say(
        "",
        f"Qiskit filter proveren: greška {filter_error:.3e}",
        "Promena verovatnoće na dijagnostičkim skupovima:"
        f" {sensitivity:.9e}",
        "Potpuna pretraga svih legalnih sedmorki.",
    )

    best, logp, probability_sum, visited = exhaustive_maximum(
        orbitals,
        space,
        path.name,
    )

    require(
        len(best) == DRAW_SIZE
        and len(set(best)) == DRAW_SIZE
        and min(best) >= 0
        and max(best) < NUMBER_COUNT,
        "NEXT kombinacija nije legalna.",
    )

    warnings = []

    run = longest_run(best)
    span = best[-1] - best[0]

    repetitions = int(np.count_nonzero(
        np.all(draws == np.asarray(best)[None, :], axis=1)
    ))

    if run >= 5:
        warnings.append(
            f"NEXT sadrži niz od {run} uzastopnih brojeva."
        )

    if span <= 9:
        warnings.append(
            "Svih sedam brojeva nalazi se u rasponu od deset brojeva."
        )

    if repetitions:
        warnings.append(
            f"NEXT se pojavljuje {repetitions} puta u istoriji."
        )

    if sensitivity < 1e-12:
        warnings.append(
            "Na proveravanih 128 dijagnostičkih skupova"
            " nije utvrđena jasna promena zbog istorije."
        )

    return {
        "name": path.name,
        "hash": fingerprint,
        "next": tuple(value + 1 for value in best),
        "validation_nll": validation_nll,
        "test_nll": test_nll,
        "probability": math.exp(logp),
        "probability_sum": probability_sum,
        "visited": visited,
        "sensitivity": sensitivity,
        "filter_error": filter_error,
        "warnings": warnings,
    }


def main():
    np.seterr(
        divide="raise",
        over="raise",
        invalid="raise",
        under="ignore",
    )

    require(
        U_COUNT + V_COUNT == DRAW_SIZE,
        "Bosonska zauzeća nisu usaglašena sa veličinom kombinacije.",
    )

    say(
        "QUANTUM BOSON V2 — Loto 7/39",
        f"Python: {sys.version.split()[0]}",
        f"Qiskit: {qiskit.__version__}",
        "Dve kompleksne orbitale nastaju izvršavanjem Qiskit kola.",
        "Bosonska zauzeća: četiri čestice u u, tri čestice u v.",
        "Sedam čestica koristi sedam logičkih registara od šest kubita.",
        "Simulacija čuva dve različite orbitale i njihova zauzeća.",
        "Amplitudni filter koristi pomoćni kubit.",
        "Trening koristi analitički oblik kvantnog filtera.",
        "NEXT izvršava i proverava eksplicitni Qiskit filter.",
        "Simetrične amplitude računaju se preko koeficijenta permanenta.",
        "Različiti loto brojevi dolaze iz uslovljavanja na legalne skupove.",
        "Bosoni nemaju Paulijevo isključivanje.",
        "Normalizacija i permanentne kontrakcije su klasični tačni proračuni.",
        "Nema slučajnog generisanja, šuma ili uzorkovanja merenja.",
        "Optimizacija je lokalna; globalni minimum nije garantovan.",
    )

    resolved = [path.resolve() for path in FILES]

    require(
        len(set(resolved)) == len(FILES),
        "Ulazne putanje nisu različite.",
    )

    datasets = []
    fingerprints = []

    for path in FILES:
        require(path.is_file(), f"Fajl nije pronađen: {path}")

        draws, fingerprint = load_draws(path)
        datasets.append((path, draws, fingerprint))
        fingerprints.append(fingerprint)

    require(
        len(set(fingerprints)) == len(fingerprints),
        "Najmanje dva ulaza imaju isti sadržaj.",
    )

    self_checks()

    results = [
        process_file(path, draws, fingerprint)
        for path, draws, fingerprint in datasets
    ]

    for first in range(len(results)):
        for second in range(first + 1, len(results)):
            if results[first]["next"] == results[second]["next"]:
                results[first]["warnings"].append(
                    "Ista NEXT kombinacija kao u "
                    + results[second]["name"]
                    + "; proveriti uticaj istorije i degeneraciju."
                )

                results[second]["warnings"].append(
                    "Ista NEXT kombinacija kao u "
                    + results[first]["name"]
                    + "; proveriti uticaj istorije i degeneraciju."
                )

    say("", "ZAVRŠNA TRI NEXT REZULTATA")

    for result in results:
        numbers = " ".join(
            f"{number:02d}" for number in result["next"]
        )

        say(
            "",
            result["name"],
            f"NEXT: {numbers}",
            f"Validacioni NLL: {result['validation_nll']:.9f}",
            f"Odvojeni test NLL: {result['test_nll']:.9f}",
            f"Modelna verovatnoća NEXT: {result['probability']:.12g}",
            f"Zbir svih verovatnoća: {result['probability_sum']:.12f}",
            f"Pregledanih kombinacija: {result['visited']:,}",
            f"Greška Qiskit filtera: {result['filter_error']:.3e}",
            f"Dijagnostička osetljivost: {result['sensitivity']:.9e}",
        )

        for warning in result["warnings"]:
            say(f"DIJAGNOSTIKA: {warning}")

    say(
        "",
        "Obrada je završena.",
        "NEXT je numerički maksimum naučene bosonske raspodele.",
        "Potpuna pretraga proverava maksimum i zbir verovatnoća.",
        "Prediktivna vrednost procenjuje se odvojenim testom.",
    )


if __name__ == "__main__":
    try:
        main()

    except (
        OSError,
        UnicodeError,
        csv.Error,
        ValueError,
        RuntimeError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as error:
        say(
            "",
            f"ZAUSTAVLJENO: {error}",
            "Završni skup od tri NEXT rezultata nije potvrđen.",
        )

        traceback.print_exc()
        sys.exit(1)



"""
QUANTUM BOSON V2 — Loto 7/39
Python: 3.11.13
Qiskit: 1.4.4
Dve kompleksne orbitale nastaju izvršavanjem Qiskit kola.
Bosonska zauzeća: četiri čestice u u, tri čestice u v.
Sedam čestica koristi sedam logičkih registara od šest kubita.
Simulacija čuva dve različite orbitale i njihova zauzeća.
Amplitudni filter koristi pomoćni kubit.
Trening koristi analitički oblik kvantnog filtera.
NEXT izvršava i proverava eksplicitni Qiskit filter.
Simetrične amplitude računaju se preko koeficijenta permanenta.
Različiti loto brojevi dolaze iz uslovljavanja na legalne skupove.
Bosoni nemaju Paulijevo isključivanje.
Normalizacija i permanentne kontrakcije su klasični tačni proračuni.
Nema slučajnog generisanja, šuma ili uzorkovanja merenja.
Optimizacija je lokalna; globalni minimum nije garantovan.

ULAZ: loto7_4710_k86.csv
Redova: 4710 | praznih redova: 0
SHA256: e51a5ab684401459535996f50fd0b0540b0af6d2b86bca49825cf381d759c85d
Redosled kola ostaje redosled iz CSV-a.

ULAZ: loto7_4710_k86_loto_2977.csv
Redova: 2977 | praznih redova: 0
SHA256: 69ae55edbff87383f9bb4a25df4ba6b44acbf23404db6e7fa3642de0442479f4
Redosled kola ostaje redosled iz CSV-a.

ULAZ: loto7_4710_k86_loto_plus_1733.csv
Redova: 1733 | praznih redova: 0
SHA256: 9d708eb88434b55a38bbc69474a869955e0295f1bed401cf10dc177d7d8fef39
Redosled kola ostaje redosled iz CSV-a.

Pokrećem ugrađene Qiskit i bosonske samoprovere.
Ugrađene samoprovere su prošle.
Greška izvoda amplituda: 9.629e-11
Greška ukupnog gradijenta: 4.946e-10
Greška Qiskit filtera: 1.755e-16
Greška pune dvobosonske simulacije: 9.310e-17
Zbir male raspodele: 1.000000000000

loto7_4710_k86.csv / validacija 1
Lag=1 | λ=0.03 | početne faze=17 | prelaza=2825
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Optimizacija: 175 iteracija
Optimizacija: 200 iteracija
Optimizacija: 225 iteracija
Optimizacija: 250 iteracija
Optimizacija: 275 iteracija
Prihvaćen lokalni minimum: 16.508661992
Max|grad|: 8.069e-07

loto7_4710_k86.csv / validacija 1
Lag=1 | λ=0.03 | početne faze=11 | prelaza=2825
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Optimizacija: 175 iteracija
Optimizacija: 200 iteracija
Optimizacija: 225 iteracija
Optimizacija: 250 iteracija
Optimizacija: 275 iteracija
Optimizacija: 300 iteracija
Prihvaćen lokalni minimum: 16.508661992
Max|grad|: 6.929e-07
Validacioni NLL: 16.595398998

loto7_4710_k86.csv / validacija 2
Lag=1 | λ=0.03 | početne faze=17 | prelaza=3296
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Optimizacija: 175 iteracija
Optimizacija: 200 iteracija
Optimizacija: 225 iteracija
Optimizacija: 250 iteracija
Optimizacija: 275 iteracija
Optimizacija: 300 iteracija
Prihvaćen lokalni minimum: 16.514950607
Max|grad|: 7.693e-07

loto7_4710_k86.csv / validacija 2
Lag=1 | λ=0.03 | početne faze=11 | prelaza=3296
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Optimizacija: 175 iteracija
Optimizacija: 200 iteracija
Optimizacija: 225 iteracija
Optimizacija: 250 iteracija
Optimizacija: 275 iteracija
Optimizacija: 300 iteracija
Prihvaćen lokalni minimum: 16.514950607
Max|grad|: 9.782e-07
Validacioni NLL: 16.593588508
KANDIDAT: lag=1 | λ=0.03
Zbirni validacioni NLL: 16.594493753

loto7_4710_k86.csv / validacija 1
Lag=1 | λ=0.3 | početne faze=17 | prelaza=2825
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Prihvaćen lokalni minimum: 16.600194710
Max|grad|: 7.635e-07

loto7_4710_k86.csv / validacija 1
Lag=1 | λ=0.3 | početne faze=11 | prelaza=2825
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Prihvaćen lokalni minimum: 16.600194710
Max|grad|: 7.330e-07
Validacioni NLL: 16.662071838

loto7_4710_k86.csv / validacija 2
Lag=1 | λ=0.3 | početne faze=17 | prelaza=3296
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Prihvaćen lokalni minimum: 16.607575652
Max|grad|: 4.682e-07

loto7_4710_k86.csv / validacija 2
Lag=1 | λ=0.3 | početne faze=11 | prelaza=3296
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Prihvaćen lokalni minimum: 16.607575652
Max|grad|: 9.036e-07
Validacioni NLL: 16.642097930
KANDIDAT: lag=1 | λ=0.3
Zbirni validacioni NLL: 16.652084884

loto7_4710_k86.csv / validacija 1
Lag=3 | λ=0.03 | početne faze=17 | prelaza=2825
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Optimizacija: 175 iteracija
Optimizacija: 200 iteracija
Optimizacija: 225 iteracija
Optimizacija: 250 iteracija
Optimizacija: 275 iteracija
Prihvaćen lokalni minimum: 16.503804140
Max|grad|: 3.293e-06

loto7_4710_k86.csv / validacija 1
Lag=3 | λ=0.03 | početne faze=11 | prelaza=2825
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Optimizacija: 175 iteracija
Optimizacija: 200 iteracija
Optimizacija: 225 iteracija
Optimizacija: 250 iteracija
Optimizacija: 275 iteracija
Optimizacija: 300 iteracija
Prihvaćen lokalni minimum: 16.503804140
Max|grad|: 8.025e-07
Validacioni NLL: 16.601662778

loto7_4710_k86.csv / validacija 2
Lag=3 | λ=0.03 | početne faze=17 | prelaza=3296
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Optimizacija: 175 iteracija
Optimizacija: 200 iteracija
Optimizacija: 225 iteracija
Optimizacija: 250 iteracija
Optimizacija: 275 iteracija
Optimizacija: 300 iteracija
Prihvaćen lokalni minimum: 16.513091437
Max|grad|: 8.194e-07

loto7_4710_k86.csv / validacija 2
Lag=3 | λ=0.03 | početne faze=11 | prelaza=3296
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Optimizacija: 175 iteracija
Optimizacija: 200 iteracija
Optimizacija: 225 iteracija
Optimizacija: 250 iteracija
Optimizacija: 275 iteracija
Optimizacija: 300 iteracija
Optimizacija: 325 iteracija
Prihvaćen lokalni minimum: 16.513091437
Max|grad|: 4.896e-06
Validacioni NLL: 16.600924931
KANDIDAT: lag=3 | λ=0.03
Zbirni validacioni NLL: 16.601293854

loto7_4710_k86.csv / validacija 1
Lag=3 | λ=0.3 | početne faze=17 | prelaza=2825
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Prihvaćen lokalni minimum: 16.597713561
Max|grad|: 8.525e-07

loto7_4710_k86.csv / validacija 1
Lag=3 | λ=0.3 | početne faze=11 | prelaza=2825
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Prihvaćen lokalni minimum: 16.597713561
Max|grad|: 8.094e-07
Validacioni NLL: 16.665219134

loto7_4710_k86.csv / validacija 2
Lag=3 | λ=0.3 | početne faze=17 | prelaza=3296
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Prihvaćen lokalni minimum: 16.606668983
Max|grad|: 8.366e-07

loto7_4710_k86.csv / validacija 2
Lag=3 | λ=0.3 | početne faze=11 | prelaza=3296
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Prihvaćen lokalni minimum: 16.606668983
Max|grad|: 4.140e-06
Validacioni NLL: 16.646909722
KANDIDAT: lag=3 | λ=0.3
Zbirni validacioni NLL: 16.656064428

IZABRANO: lag=1 | λ=0.03
Validacioni NLL: 16.594493753

loto7_4710_k86.csv / odvojeni test
Lag=1 | λ=0.03 | početne faze=17 | prelaza=3767
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Optimizacija: 175 iteracija
Optimizacija: 200 iteracija
Optimizacija: 225 iteracija
Optimizacija: 250 iteracija
Prihvaćen lokalni minimum: 16.522831624
Max|grad|: 9.270e-07

loto7_4710_k86.csv / odvojeni test
Lag=1 | λ=0.03 | početne faze=11 | prelaza=3767
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Optimizacija: 175 iteracija
Optimizacija: 200 iteracija
Optimizacija: 225 iteracija
Optimizacija: 250 iteracija
Optimizacija: 275 iteracija
Prihvaćen lokalni minimum: 16.522831624
Max|grad|: 7.742e-07
ODVOJENI TEST: 942 prelaza
NLL cele sedmorke: 16.585957021

loto7_4710_k86.csv / završno učenje
Lag=1 | λ=0.03 | početne faze=17 | prelaza=4709
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Optimizacija: 175 iteracija
Optimizacija: 200 iteracija
Optimizacija: 225 iteracija
Optimizacija: 250 iteracija
Optimizacija: 275 iteracija
Prihvaćen lokalni minimum: 16.529134677
Max|grad|: 7.459e-07

loto7_4710_k86.csv / završno učenje
Lag=1 | λ=0.03 | početne faze=11 | prelaza=4709
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Optimizacija: 175 iteracija
Optimizacija: 200 iteracija
Optimizacija: 225 iteracija
Optimizacija: 250 iteracija
Optimizacija: 275 iteracija
Optimizacija: 300 iteracija
Prihvaćen lokalni minimum: 16.529134677
Max|grad|: 9.750e-07
ZAVRŠNO UČENJE: svih 4709 istorijskih prelaza.

Qiskit filter proveren: greška 7.988e-15
Promena verovatnoće na dijagnostičkim skupovima: 8.826650365e-08
Potpuna pretraga svih legalnih sedmorki.
loto7_4710_k86.csv: pregledano 2,002,944/15,380,937
loto7_4710_k86.csv: pregledano 4,001,792/15,380,937
loto7_4710_k86.csv: pregledano 6,000,640/15,380,937
loto7_4710_k86.csv: pregledano 8,003,584/15,380,937
loto7_4710_k86.csv: pregledano 10,002,432/15,380,937
loto7_4710_k86.csv: pregledano 12,001,280/15,380,937
loto7_4710_k86.csv: pregledano 14,000,128/15,380,937

loto7_4710_k86_loto_2977.csv / validacija 1
Lag=1 | λ=0.03 | početne faze=17 | prelaza=1785
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Optimizacija: 175 iteracija
Optimizacija: 200 iteracija
Optimizacija: 225 iteracija
Optimizacija: 250 iteracija
Prihvaćen lokalni minimum: 16.491674603
Max|grad|: 9.314e-07

loto7_4710_k86_loto_2977.csv / validacija 1
Lag=1 | λ=0.03 | početne faze=11 | prelaza=1785
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Optimizacija: 175 iteracija
Optimizacija: 200 iteracija
Optimizacija: 225 iteracija
Optimizacija: 250 iteracija
Prihvaćen lokalni minimum: 16.491674603
Max|grad|: 8.118e-07
Validacioni NLL: 16.575736751

loto7_4710_k86_loto_2977.csv / validacija 2
Lag=1 | λ=0.03 | početne faze=17 | prelaza=2083
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Optimizacija: 175 iteracija
Optimizacija: 200 iteracija
Optimizacija: 225 iteracija
Prihvaćen lokalni minimum: 16.496516040
Max|grad|: 9.517e-07

loto7_4710_k86_loto_2977.csv / validacija 2
Lag=1 | λ=0.03 | početne faze=11 | prelaza=2083
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Optimizacija: 175 iteracija
Optimizacija: 200 iteracija
Optimizacija: 225 iteracija
Prihvaćen lokalni minimum: 16.496516040
Max|grad|: 8.657e-07
Validacioni NLL: 16.629110727
KANDIDAT: lag=1 | λ=0.03
Zbirni validacioni NLL: 16.602378887

loto7_4710_k86_loto_2977.csv / validacija 1
Lag=1 | λ=0.3 | početne faze=17 | prelaza=1785
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Prihvaćen lokalni minimum: 16.582958532
Max|grad|: 8.470e-07

loto7_4710_k86_loto_2977.csv / validacija 1
Lag=1 | λ=0.3 | početne faze=11 | prelaza=1785
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Prihvaćen lokalni minimum: 16.582958532
Max|grad|: 8.945e-07
Validacioni NLL: 16.621886165

loto7_4710_k86_loto_2977.csv / validacija 2
Lag=1 | λ=0.3 | početne faze=17 | prelaza=2083
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Prihvaćen lokalni minimum: 16.587223964
Max|grad|: 6.365e-07

loto7_4710_k86_loto_2977.csv / validacija 2
Lag=1 | λ=0.3 | početne faze=11 | prelaza=2083
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija

loto7_4710_k86_loto_2977.csv / validacija 2
Lag=3 | λ=0.03 | početne faze=17 | prelaza=2083
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Optimizacija: 175 iteracija
Optimizacija: 200 iteracija
Optimizacija: 225 iteracija
Prihvaćen lokalni minimum: 16.493965349
Max|grad|: 8.877e-07

loto7_4710_k86_loto_2977.csv / validacija 2
Lag=3 | λ=0.03 | početne faze=11 | prelaza=2083
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Optimizacija: 175 iteracija
Optimizacija: 200 iteracija
Optimizacija: 225 iteracija
Prihvaćen lokalni minimum: 16.493965349
Max|grad|: 1.645e-06
Validacioni NLL: 16.580088503
KANDIDAT: lag=3 | λ=0.03
Zbirni validacioni NLL: 16.589940692

loto7_4710_k86_loto_2977.csv / validacija 1
Lag=3 | λ=0.3 | početne faze=17 | prelaza=1785
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Prihvaćen lokalni minimum: 16.582778123
Max|grad|: 2.348e-06

loto7_4710_k86_loto_2977.csv / validacija 1
Lag=3 | λ=0.3 | početne faze=11 | prelaza=1785
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Prihvaćen lokalni minimum: 16.582778123
Max|grad|: 7.982e-07
Validacioni NLL: 16.635904085

loto7_4710_k86_loto_2977.csv / validacija 2
Lag=3 | λ=0.3 | početne faze=17 | prelaza=2083
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Prihvaćen lokalni minimum: 16.585694577
Max|grad|: 8.582e-07

loto7_4710_k86_loto_2977.csv / validacija 2
Lag=3 | λ=0.3 | početne faze=11 | prelaza=2083
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Prihvaćen lokalni minimum: 16.585694577
Max|grad|: 6.933e-07
Validacioni NLL: 16.633778455
KANDIDAT: lag=3 | λ=0.3
Zbirni validacioni NLL: 16.634843056

IZABRANO: lag=3 | λ=0.03
Validacioni NLL: 16.589940692

loto7_4710_k86_loto_2977.csv / odvojeni test
Lag=3 | λ=0.03 | početne faze=17 | prelaza=2380
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Optimizacija: 175 iteracija
Optimizacija: 200 iteracija
Optimizacija: 225 iteracija
Prihvaćen lokalni minimum: 16.495917660
Max|grad|: 6.409e-07

loto7_4710_k86_loto_2977.csv / odvojeni test
Lag=3 | λ=0.03 | početne faze=11 | prelaza=2380
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Optimizacija: 175 iteracija
Optimizacija: 200 iteracija
Optimizacija: 225 iteracija
Prihvaćen lokalni minimum: 16.495917660
Max|grad|: 4.086e-06
ODVOJENI TEST: 596 prelaza
NLL cele sedmorke: 16.622196800

loto7_4710_k86_loto_2977.csv / završno učenje
Lag=3 | λ=0.03 | početne faze=17 | prelaza=2976
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Optimizacija: 175 iteracija
Optimizacija: 200 iteracija
Optimizacija: 225 iteracija
Prihvaćen lokalni minimum: 16.515640341
Max|grad|: 8.667e-07

loto7_4710_k86_loto_2977.csv / završno učenje
Lag=3 | λ=0.03 | početne faze=11 | prelaza=2976
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Optimizacija: 175 iteracija
Optimizacija: 200 iteracija
Optimizacija: 225 iteracija
Prihvaćen lokalni minimum: 16.515640341
Max|grad|: 9.469e-07
ZAVRŠNO UČENJE: svih 2976 istorijskih prelaza.

Qiskit filter proveren: greška 7.778e-15
Promena verovatnoće na dijagnostičkim skupovima: 5.872522738e-08
Potpuna pretraga svih legalnih sedmorki.
loto7_4710_k86_loto_2977.csv: pregledano 2,002,944/15,380,937
loto7_4710_k86_loto_2977.csv: pregledano 4,001,792/15,380,937
loto7_4710_k86_loto_2977.csv: pregledano 6,000,640/15,380,937
loto7_4710_k86_loto_2977.csv: pregledano 8,003,584/15,380,937
loto7_4710_k86_loto_2977.csv: pregledano 10,002,432/15,380,937
loto7_4710_k86_loto_2977.csv: pregledano 12,001,280/15,380,937
loto7_4710_k86_loto_2977.csv: pregledano 14,000,128/15,380,937

loto7_4710_k86_loto_plus_1733.csv / validacija 1
Lag=1 | λ=0.03 | početne faze=17 | prelaza=1039
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Optimizacija: 175 iteracija
Optimizacija: 200 iteracija
Prihvaćen lokalni minimum: 16.448453658
Max|grad|: 8.103e-07

loto7_4710_k86_loto_plus_1733.csv / validacija 1
Lag=1 | λ=0.03 | početne faze=11 | prelaza=1039
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Optimizacija: 175 iteracija
Prihvaćen lokalni minimum: 16.478379519
Max|grad|: 4.362e-07
Validacioni NLL: 16.781742938

loto7_4710_k86_loto_plus_1733.csv / validacija 2
Lag=1 | λ=0.03 | početne faze=17 | prelaza=1212
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Optimizacija: 175 iteracija
Optimizacija: 200 iteracija
Optimizacija: 225 iteracija
Optimizacija: 250 iteracija
Prihvaćen lokalni minimum: 16.442524028
Max|grad|: 9.729e-07

loto7_4710_k86_loto_plus_1733.csv / validacija 2
Lag=1 | λ=0.03 | početne faze=11 | prelaza=1212
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Prihvaćen lokalni minimum: 16.517226007
Max|grad|: 4.720e-07
Validacioni NLL: 16.645218497
KANDIDAT: lag=1 | λ=0.03
Zbirni validacioni NLL: 16.713480717

loto7_4710_k86_loto_plus_1733.csv / validacija 1
Lag=1 | λ=0.3 | početne faze=17 | prelaza=1039
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Prihvaćen lokalni minimum: 16.550914587
Max|grad|: 9.257e-07

loto7_4710_k86_loto_plus_1733.csv / validacija 1
Lag=1 | λ=0.3 | početne faze=11 | prelaza=1039
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Prihvaćen lokalni minimum: 16.550914587
Max|grad|: 1.256e-06
Validacioni NLL: 16.731505346

loto7_4710_k86_loto_plus_1733.csv / validacija 2
Lag=1 | λ=0.3 | početne faze=17 | prelaza=1212
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Prihvaćen lokalni minimum: 16.573865165
Max|grad|: 9.260e-07

loto7_4710_k86_loto_plus_1733.csv / validacija 2
Lag=1 | λ=0.3 | početne faze=11 | prelaza=1212
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Prihvaćen lokalni minimum: 16.586128324
Max|grad|: 8.217e-07
Validacioni NLL: 16.666056506
KANDIDAT: lag=1 | λ=0.3
Zbirni validacioni NLL: 16.698780926

loto7_4710_k86_loto_plus_1733.csv / validacija 1
Lag=3 | λ=0.03 | početne faze=17 | prelaza=1039
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Optimizacija: 175 iteracija
Prihvaćen lokalni minimum: 16.444179131
Max|grad|: 1.571e-06

loto7_4710_k86_loto_plus_1733.csv / validacija 1
Lag=3 | λ=0.03 | početne faze=11 | prelaza=1039
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Optimizacija: 175 iteracija
Prihvaćen lokalni minimum: 16.471860842
Max|grad|: 6.586e-07
Validacioni NLL: 16.715009449

loto7_4710_k86_loto_plus_1733.csv / validacija 2
Lag=3 | λ=0.03 | početne faze=17 | prelaza=1212
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Optimizacija: 175 iteracija
Optimizacija: 200 iteracija
Optimizacija: 225 iteracija
Optimizacija: 250 iteracija
Optimizacija: 275 iteracija
Prihvaćen lokalni minimum: 16.428293906
Max|grad|: 1.452e-06

loto7_4710_k86_loto_plus_1733.csv / validacija 2
Lag=3 | λ=0.03 | početne faze=11 | prelaza=1212
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Optimizacija: 175 iteracija
Prihvaćen lokalni minimum: 16.500965254
Max|grad|: 7.337e-07
Validacioni NLL: 16.650008716
KANDIDAT: lag=3 | λ=0.03
Zbirni validacioni NLL: 16.682509083

loto7_4710_k86_loto_plus_1733.csv / validacija 1
Lag=3 | λ=0.3 | početne faze=17 | prelaza=1039
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Prihvaćen lokalni minimum: 16.547219454
Max|grad|: 5.502e-07

loto7_4710_k86_loto_plus_1733.csv / validacija 1
Lag=3 | λ=0.3 | početne faze=11 | prelaza=1039
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Prihvaćen lokalni minimum: 16.547219454
Max|grad|: 5.594e-07
Validacioni NLL: 16.691474216

loto7_4710_k86_loto_plus_1733.csv / validacija 2
Lag=3 | λ=0.3 | početne faze=17 | prelaza=1212
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Prihvaćen lokalni minimum: 16.565547150
Max|grad|: 8.087e-07

loto7_4710_k86_loto_plus_1733.csv / validacija 2
Lag=3 | λ=0.3 | početne faze=11 | prelaza=1212
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Prihvaćen lokalni minimum: 16.577452308
Max|grad|: 9.834e-07
Validacioni NLL: 16.657815460
KANDIDAT: lag=3 | λ=0.3
Zbirni validacioni NLL: 16.674644838

IZABRANO: lag=3 | λ=0.3
Validacioni NLL: 16.674644838

loto7_4710_k86_loto_plus_1733.csv / odvojeni test
Lag=3 | λ=0.3 | početne faze=17 | prelaza=1385
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Prihvaćen lokalni minimum: 16.577729534
Max|grad|: 9.868e-07

loto7_4710_k86_loto_plus_1733.csv / odvojeni test
Lag=3 | λ=0.3 | početne faze=11 | prelaza=1385
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Optimizacija: 175 iteracija
Optimizacija: 200 iteracija
Optimizacija: 225 iteracija
Prihvaćen lokalni minimum: 16.577729534
Max|grad|: 3.640e-07
ODVOJENI TEST: 347 prelaza
NLL cele sedmorke: 16.622653595

loto7_4710_k86_loto_plus_1733.csv / završno učenje
Lag=3 | λ=0.3 | početne faze=17 | prelaza=1732
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Prihvaćen lokalni minimum: 16.587652801
Max|grad|: 7.672e-07

loto7_4710_k86_loto_plus_1733.csv / završno učenje
Lag=3 | λ=0.3 | početne faze=11 | prelaza=1732
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Prihvaćen lokalni minimum: 16.610455485
Max|grad|: 4.832e-07
ZAVRŠNO UČENJE: svih 1732 istorijskih prelaza.

Qiskit filter proveren: greška 8.364e-15
Promena verovatnoće na dijagnostičkim skupovima: 1.474428468e-07
Potpuna pretraga svih legalnih sedmorki.
loto7_4710_k86_loto_plus_1733.csv: pregledano 2,002,944/15,380,937
loto7_4710_k86_loto_plus_1733.csv: pregledano 4,001,792/15,380,937
loto7_4710_k86_loto_plus_1733.csv: pregledano 6,000,640/15,380,937
loto7_4710_k86_loto_plus_1733.csv: pregledano 8,003,584/15,380,937
loto7_4710_k86_loto_plus_1733.csv: pregledano 10,002,432/15,380,937
loto7_4710_k86_loto_plus_1733.csv: pregledano 12,001,280/15,380,937
loto7_4710_k86_loto_plus_1733.csv: pregledano 14,000,128/15,380,937



ZAVRŠNA TRI NEXT REZULTATA

loto7_4710_k86.csv
NEXT: 25 x 27 y 29 z 31
Validacioni NLL: 16.594493753
Odvojeni test NLL: 16.585957021
Modelna verovatnoća NEXT: 1.09236306282e-06
Zbir svih verovatnoća: 1.000000000000
Pregledanih kombinacija: 15,380,937
Greška Qiskit filtera: 7.988e-15
Dijagnostička osetljivost: 8.826650365e-08
DIJAGNOSTIKA: NEXT sadrži niz od 7 uzastopnih brojeva.
DIJAGNOSTIKA: Svih sedam brojeva nalazi se u rasponu od deset brojeva.

loto7_4710_k86_loto_2977.csv
NEXT: 25 x 29 y 31 z 33
Validacioni NLL: 16.589940692
Odvojeni test NLL: 16.622196800
Modelna verovatnoća NEXT: 4.98637196598e-07
Zbir svih verovatnoća: 1.000000000000
Pregledanih kombinacija: 15,380,937
Greška Qiskit filtera: 7.778e-15
Dijagnostička osetljivost: 5.872522738e-08
DIJAGNOSTIKA: NEXT sadrži niz od 6 uzastopnih brojeva.
DIJAGNOSTIKA: Svih sedam brojeva nalazi se u rasponu od deset brojeva.

loto7_4710_k86_loto_plus_1733.csv
NEXT: 15 x 18 y 20 z 32
Validacioni NLL: 16.674644838
Odvojeni test NLL: 16.622653595
Modelna verovatnoća NEXT: 9.61858791937e-07
Zbir svih verovatnoća: 1.000000000000
Pregledanih kombinacija: 15,380,937
Greška Qiskit filtera: 8.364e-15
Dijagnostička osetljivost: 1.474428468e-07

Obrada je završena.
NEXT je numerički maksimum naučene bosonske raspodele.
Potpuna pretraga proverava maksimum i zbir verovatnoća.
Prediktivna vrednost procenjuje se odvojenim testom.
"""



"""
Dve slobodno naučene orbitale sa zauzećima 4+3. 
Bosonska konstrukcija i veza sa permanentom, a orbitale će nastajati u Qiskit kolu. 
Tačna normalizacija nad legalnim sedmorkama. 
Kod bosona se različiti loto brojevi dobijaju uslovljavanjem na legalne skupove, a ne Paulijevim isključivanjem.
Qiskit kolo stvara kompleksne orbitale, uz očuvanje bosonskih zauzeća 4+3, simetričnih amplituda i permanenta.
Istorijski amplitudni filter koristi istu kontrolisane RY kapije kao fermionski kod. 
Normalizacija i maksimum računaju se tačno preko kompaktne bosonske reprezentacije. 
Nema gustog vektora sa 42 kubita.
"""
