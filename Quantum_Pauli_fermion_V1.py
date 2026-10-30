#!/usr/bin/env python3

# Quantum_Pauli_fermion_boson
 

"""
Quantum Pauli Fermion V1
Python 3.11
Qiskit 1.4.4
Dependencies: qiskit, numpy, scipy

Quantum orbitals:
    B(theta)[i,j] = <i|U(theta)|j>

History-controlled quantum filter:
    h(x) = CAP * tanh(x W / CAP)
    K(x)[i,i] = exp(h_i(x) - max(h(x)))

The filter uses controlled RY gates and an accepted ancilla=1 branch.

Fermionic distribution:
    A(x) = K(x) B(theta)
    P(S|x) = |det(A(x)[S,:])|^2 / det(A(x)^H A(x))

Qiskit executes the orbital circuit during fitting.
Training evaluates the filter analytically.
NEXT verifies and executes the explicit Qiskit filter.

Seven-particle amplitudes use the compact Slater representation.
No dense 42-qubit statevector is allocated.

Numerical correction:
    Local floating-point handling during controlled-RY construction.
    Checked determinants throughout all self-checks.
    Explicit validation of Qiskit filter amplitudes.

No sampling, random initialization, noise or row shuffling.
No classifier, rounding, duplicate repair or cosmetic filters.
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
from itertools import combinations, islice
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

FEATURE_DIM = 8
CAP = 4.0
QUANTUM_DEPTH = 2

CANDIDATES = (
    (1, 0.03),
    (1, 0.30),
    (3, 0.03),
    (3, 0.30),
)

INITIALIZATIONS = (0, 1)
THETA_RIDGE_RATIO = 0.01

OPTIMIZER_STEPS = 600
OPTIMIZER_CONTINUATIONS = 3
GRADIENT_LIMIT = 5e-4

MAP_BATCH = 4096
NORMALIZATION_TOLERANCE = 2e-7


def say(*lines):
    for line in lines:
        print(line, flush=True)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def checked_slogdet(matrix):
    matrix = np.asarray(matrix)

    require(
        matrix.ndim >= 2
        and matrix.shape[-1] == matrix.shape[-2]
        and matrix.shape[-1] > 0,
        "Determinanta zahteva nepraznu kvadratnu matricu.",
    )
    require(
        np.all(np.isfinite(matrix)),
        "Ulaz determinante sadrži nevažeće vrednosti.",
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
        over="ignore",
        under="ignore",
    ):
        phase, logarithm = np.linalg.slogdet(matrix)

    require(
        np.all(np.isfinite(phase)),
        "Neispravna faza determinante.",
    )

    zero = phase == 0

    valid_zero = zero & np.isneginf(logarithm)
    valid_nonzero = (
        (~zero)
        & np.isfinite(logarithm)
        & (np.abs(np.abs(phase) - 1.0) < 1e-7)
    )

    require(
        np.all(valid_zero | valid_nonzero),
        "Neispravan rezultat determinante.",
    )

    return phase, logarithm


def checked_determinant(matrix):
    matrix = np.asarray(matrix)

    require(
        matrix.ndim == 2,
        "Skalarna determinanta zahteva jednu matricu.",
    )

    phase, logarithm = checked_slogdet(matrix)
    phase = complex(phase)
    logarithm = float(logarithm)

    if phase == 0:
        return 0.0 + 0.0j

    require(
        logarithm <= math.log(np.finfo(float).max),
        "Determinanta je izvan numeričkog opsega.",
    )

    value = phase * math.exp(logarithm)

    require(
        np.isfinite(value.real) and np.isfinite(value.imag),
        "Nevažeća skalarna determinanta.",
    )

    return value


def positive_logdet(gram):
    phase, logarithm = checked_slogdet(gram)

    require(
        np.all(np.isfinite(logarithm)),
        "Singularna normalizaciona matrica.",
    )
    require(
        np.all(np.abs(phase - 1.0) < 1e-6),
        "Normalizaciona matrica nije pozitivno definitna.",
    )

    return logarithm


def checked_controlled_ry(angle, control_count, control_state):
    angle = float(angle)

    require(
        math.isfinite(angle),
        "Ugao kontrolisane RY kapije nije konačan.",
    )
    require(
        control_count >= 1
        and 0 <= control_state < (1 << control_count),
        "Neispravno kontrolno stanje RY kapije.",
    )

    cosine = math.cos(angle / 2.0)
    sine = math.sin(angle / 2.0)

    require(
        abs(cosine * cosine + sine * sine - 1.0) < 1e-12,
        "Osnovna RY rotacija nije numerički unitarna.",
    )

    # Qiskit 1.4 may call np.linalg.det during synthesis.
    # Some numerical backends raise spurious divide/invalid flags.
    # Handle only these flags locally. Global settings are restored
    # automatically when leaving this context.
    #
    # The resulting filter is independently validated against its
    # expected complex amplitudes in filtered_amplitudes().
    with np.errstate(
        divide="ignore",
        invalid="ignore",
        over="raise",
        under="ignore",
    ):
        gate = RYGate(angle).control(
            num_ctrl_qubits=control_count,
            ctrl_state=int(control_state),
        )

    require(
        gate.num_qubits == control_count + 1,
        "Kontrolisana RY kapija ima pogrešan broj kubita.",
    )

    return gate


def orthonormal_basis(amplitudes):
    amplitudes = np.asarray(amplitudes, dtype=np.complex128)

    require(
        amplitudes.ndim == 2
        and amplitudes.shape[0] >= amplitudes.shape[1]
        and np.all(np.isfinite(amplitudes)),
        "Neispravna orbitalna matrica.",
    )

    q, r = np.linalg.qr(amplitudes, mode="reduced")

    require(
        np.all(np.abs(np.diag(r)) > 1e-12),
        "Orbitalna matrica nema potreban rang.",
    )
    require(
        np.max(np.abs(
            q.conj().T @ q - np.eye(q.shape[1])
        )) < 1e-9,
        "Ortonormalizacija nije prošla.",
    )

    return q


class QuantumOrbitals:
    def __init__(self, number_count, particle_count, depth):
        require(
            1 <= particle_count <= number_count and depth >= 1,
            "Neispravne dimenzije kvantnog modela.",
        )

        self.m = int(number_count)
        self.k = int(particle_count)
        self.depth = int(depth)
        self.nq = max(1, (self.m - 1).bit_length())
        self.dimension = 1 << self.nq
        self.parameter_count = 2 * self.nq * self.depth

        self.fourier = self.make_fourier()

        self.inputs = tuple(
            Statevector.from_int(j, self.dimension)
            for j in range(self.k)
        )

    def make_fourier(self):
        circuit = QuantumCircuit(self.nq, name="QFT")

        for upper in range(self.nq - 1, -1, -1):
            circuit.h(upper)

            for lower in range(upper - 1, -1, -1):
                angle = math.pi / (2 ** (upper - lower))
                circuit.cp(angle, upper, lower)

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

        circuit = QuantumCircuit(self.nq, name="FermionOrbitals")
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
                result.conj().T @ result - np.eye(self.k)
            )) < 2e-9,
            "Qiskit orbitalni izlazi nisu ortonormalni.",
        )

        return result

    def basis(self, theta):
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

        scale = float(np.max(h))
        factors = np.exp(h - scale)

        require(
            np.all(factors > 0)
            and np.all(factors <= 1.0),
            "Amplitudni filter nije kontrakcija.",
        )

        circuit = QuantumCircuit(self.nq + 1, name="HistoryFilter")
        circuit.compose(
            self.circuit(theta),
            qubits=list(range(self.nq)),
            inplace=True,
        )

        ancilla = self.nq
        wires = list(range(self.nq)) + [ancilla]

        for label, factor in enumerate(factors):
            angle = 2.0 * math.asin(float(factor))

            controlled = checked_controlled_ry(
                angle,
                self.nq,
                label,
            )

            circuit.append(controlled, wires)

        columns = []
        illegal_maximum = 0.0

        for orbital in range(self.k):
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

        require(
            illegal_maximum < 1e-10,
            "Nelegalne oznake imaju prihvaćenu amplitudu.",
        )

        reference = factors[:, None] * self.basis(theta)
        error = float(np.max(np.abs(result - reference)))

        require(
            error < 2e-9,
            f"Qiskit amplitudni filter nije prošao: {error:.3e}",
        )

        return result, error


def load_draws(path):
    raw = path.read_bytes()
    fingerprint = hashlib.sha256(raw).hexdigest()

    rows = []
    blank = 0

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        for line_number, row in enumerate(csv.reader(handle), 1):
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

    draws = np.asarray(rows, dtype=np.int64)

    say(
        "",
        f"ULAZ: {path.name}",
        f"Redova: {len(draws)} | praznih redova: {blank}",
        f"SHA256: {fingerprint}",
        "Redosled kola ostaje redosled iz CSV-a.",
    )

    return draws, fingerprint


def make_context(
    draws,
    target_indices,
    lookback,
    number_count=39,
):
    targets = np.asarray(target_indices, dtype=np.int64)

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

    values = values[:dimension]
    vectors = vectors[:, :dimension].copy()

    for column in range(dimension):
        pivot = int(np.argmax(np.abs(vectors[:, column])))

        if vectors[pivot, column] < 0:
            vectors[:, column] *= -1.0

    projection = vectors / np.sqrt(values)[None, :]

    return mean, projection


def unpack(parameters, source, feature_count):
    count = source.parameter_count
    theta = parameters[:count]
    w = parameters[count:].reshape(feature_count, source.m)

    return theta, w


def log_amplitudes(x, w):
    t = np.tanh((x @ w) / CAP)
    return CAP * t, 1.0 - t * t


def distribution_terms(b, w, x, targets):
    h, link_derivative = log_amplitudes(x, w)
    weights = np.exp(2.0 * h)

    gram = np.einsum(
        "ni,ia,ib->nab",
        weights,
        b.conj(),
        b,
        optimize=True,
    )

    normalizer = positive_logdet(gram)
    selected = b[targets]

    _, logabs = checked_slogdet(selected)

    require(
        np.all(np.isfinite(logabs)),
        "Posmatrano kolo ima nultu numeričku amplitudu.",
    )

    sample_indices = np.arange(len(targets))[:, None]

    logp = (
        2.0 * logabs
        + 2.0 * np.sum(h[sample_indices, targets], axis=1)
        - normalizer
    )

    require(
        np.all(np.isfinite(logp))
        and np.max(logp) <= 1e-7,
        "Neispravna log-verovatnoća posmatranog kola.",
    )

    return logp, weights, gram, selected, link_derivative


def objective(
    parameters,
    source,
    x,
    targets,
    ridge,
    gradient=True,
):
    theta, w = unpack(parameters, source, x.shape[1])

    b = source.basis(theta)

    logp, weights, gram, selected, derivative = distribution_terms(
        b,
        w,
        x,
        targets,
    )

    loss = (
        -float(logp.mean())
        + ridge * float(np.sum(w * w))
        + ridge * THETA_RIDGE_RATIO * float(np.dot(theta, theta))
    )

    require(np.isfinite(loss), "Nevažeća funkcija cilja.")

    if not gradient:
        return loss

    n = len(targets)

    inverse_gram = np.linalg.inv(gram)
    inverse_selected = np.linalg.inv(selected)

    b_inverse = np.einsum(
        "ia,nab->nib",
        b,
        inverse_gram,
        optimize=True,
    )

    occupations = weights * np.einsum(
        "nib,ib->ni",
        b_inverse,
        b.conj(),
        optimize=True,
    ).real

    require(
        np.min(occupations) >= -1e-6
        and np.max(occupations) <= 1.0 + 1e-6,
        "Neispravne fermionske zauzetosti tokom učenja.",
    )

    require(
        np.max(
            np.abs(occupations.sum(axis=1) - source.k)
        ) < 1e-5,
        "Zbir fermionskih zauzetosti nije broj čestica.",
    )

    gradient_h = 2.0 * occupations
    sample_indices = np.arange(n)[:, None]
    gradient_h[sample_indices, targets] -= 2.0

    gradient_w = (
        x.T @ (gradient_h * derivative) / n
        + 2.0 * ridge * w
    )

    gradient_b = (
        2.0
        * np.einsum(
            "ni,nib->ib",
            weights,
            b_inverse,
            optimize=True,
        )
        / n
    )

    selected_gradient = (
        -2.0
        * inverse_selected.conj().transpose(0, 2, 1)
        / n
    )

    np.add.at(
        gradient_b,
        targets.ravel(),
        selected_gradient.reshape(-1, source.k),
    )

    gradient_theta = np.empty(
        source.parameter_count,
        dtype=float,
    )

    for parameter in range(source.parameter_count):
        db = source.amplitude_derivative(theta, parameter)

        gradient_theta[parameter] = (
            float(np.vdot(gradient_b, db).real)
            + 2.0
            * ridge
            * THETA_RIDGE_RATIO
            * theta[parameter]
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

    def basis(self, raw_row, explicit_filter=False):
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
            shift = float(np.max(h[0]))

            amplitudes = (
                np.exp(h[0] - shift)[:, None]
                * self.b
            )

            error = None

        q = orthonormal_basis(amplitudes)

        return q, amplitudes, error


def fit_model(raw, targets, lookback, ridge, label):
    mean, projection = fit_feature_map(raw)
    x = (raw - mean) @ projection

    source = QuantumOrbitals(
        NUMBER_COUNT,
        DRAW_SIZE,
        QUANTUM_DEPTH,
    )

    fitted = []

    for initialization in INITIALIZATIONS:
        theta0 = np.zeros(
            source.parameter_count,
            dtype=float,
        )

        if initialization:
            theta0 = 0.025 * np.sin(
                np.arange(
                    source.parameter_count,
                    dtype=float,
                ) + 1.0
            )

        w0 = np.zeros(
            (x.shape[1], NUMBER_COUNT),
            dtype=float,
        )

        parameters = np.concatenate(
            (theta0, w0.ravel())
        )

        iteration = 0

        say(
            "",
            label,
            f"Lag={lookback} | λ={ridge:g}"
            f" | početak={initialization}"
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
            )

        def callback(current):
            nonlocal iteration
            iteration += 1

            if iteration % 25 == 0:
                say(f"Optimizacija: {iteration} iteracija")

        try:
            converged = False
            last_gradient = math.inf
            result = None

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

                last_gradient = float(
                    np.max(np.abs(gradient))
                )

                if (
                    result.success
                    and last_gradient <= GRADIENT_LIMIT
                ):
                    converged = True
                    break

                say(
                    f"Nastavak {continuation + 1}"
                    f" | max|grad|={last_gradient:.3e}",
                    str(result.message),
                )

            require(
                converged,
                "Početak nije dostigao uslov konvergencije.",
            )

            theta, w = unpack(
                parameters,
                source,
                x.shape[1],
            )

            b = source.basis(theta)

            fitted.append(
                Model(
                    source=source,
                    theta=theta.copy(),
                    w=w.copy(),
                    b=b.copy(),
                    mean=mean.copy(),
                    projection=projection.copy(),
                    lookback=lookback,
                    ridge=ridge,
                    train_loss=float(loss),
                    gradient_norm=last_gradient,
                )
            )

            say(
                f"Prihvaćen lokalni minimum: {loss:.9f}",
                f"Max|grad|: {last_gradient:.3e}",
            )

        except (
            RuntimeError,
            FloatingPointError,
            np.linalg.LinAlgError,
        ) as error:
            say(
                f"Početak {initialization} nije prihvaćen:",
                str(error),
            )

    require(
        bool(fitted),
        f"{label}: nijedan početak nije prošao numeričke uslove.",
    )

    return min(
        fitted,
        key=lambda model: model.train_loss,
    )


def evaluate_nll(model, raw, targets):
    total = 0.0

    for start in range(0, len(targets), 512):
        stop = min(start + 512, len(targets))

        x = model.transform(raw[start:stop])

        logp = distribution_terms(
            model.b,
            model.w,
            x,
            targets[start:stop],
        )[0]

        total -= float(np.sum(logp))

    return total / len(targets)


def exhaustive_maximum(q, label, verbose=True):
    m, k = q.shape

    expected = math.comb(m, k)
    iterator = combinations(range(m), k)

    visited = 0
    best = None
    best_score = -math.inf
    log_total = -math.inf
    previous_progress = 0

    while True:
        block = list(islice(iterator, MAP_BATCH))

        if not block:
            break

        indices = np.asarray(block, dtype=np.int64)
        _, logabs = checked_slogdet(q[indices])
        scores = 2.0 * logabs

        require(
            not np.any(np.isnan(scores)),
            "NaN tokom potpune pretrage.",
        )

        local = int(np.argmax(scores))
        value = float(scores[local])

        if value > best_score:
            best_score = value
            best = tuple(int(v) for v in indices[local])

        log_total = float(
            np.logaddexp(
                log_total,
                logsumexp(scores),
            )
        )

        visited += len(indices)
        progress = visited // 2_000_000

        if verbose and progress > previous_progress:
            say(
                f"{label}: pregledano {visited:,}/{expected:,}"
            )
            previous_progress = progress

    require(
        visited == expected and best is not None,
        "Pretraga maksimuma nije potpuna.",
    )

    require(
        np.isfinite(best_score) and best_score <= 1e-7,
        "Neispravna verovatnoća maksimuma.",
    )

    probability_sum = math.exp(log_total)

    require(
        abs(probability_sum - 1.0)
        <= NORMALIZATION_TOLERANCE,
        f"Normalizacija nije prošla: {probability_sum:.12g}",
    )

    require(
        best_score >= -math.log(expected) - 1e-7,
        "Maksimum je manji od dozvoljene donje granice.",
    )

    return best, best_score, probability_sum, visited


def verify_two_particle_state():
    source = QuantumOrbitals(6, 2, 1)

    theta = 0.03 * np.sin(
        np.arange(
            source.parameter_count,
            dtype=float,
        ) + 1.0
    )

    dimension = source.dimension

    initial = np.zeros(
        dimension * dimension,
        dtype=complex,
    )

    initial[0 + dimension * 1] = 1.0 / math.sqrt(2.0)
    initial[1 + dimension * 0] = -1.0 / math.sqrt(2.0)

    circuit = QuantumCircuit(2 * source.nq)
    unitary = source.circuit(theta)

    circuit.compose(
        unitary,
        qubits=list(range(source.nq)),
        inplace=True,
    )

    circuit.compose(
        unitary,
        qubits=list(
            range(source.nq, 2 * source.nq)
        ),
        inplace=True,
    )

    actual = Statevector(initial).evolve(circuit).data
    orbitals = source.full_amplitudes(theta)
    expected = np.empty_like(actual)

    for first in range(dimension):
        for second in range(dimension):
            amplitude = (
                orbitals[first, 0] * orbitals[second, 1]
                - orbitals[first, 1] * orbitals[second, 0]
            ) / math.sqrt(2.0)

            expected[
                first + dimension * second
            ] = amplitude

    error = float(
        np.max(np.abs(actual - expected))
    )

    require(
        error < 2e-10,
        "Puna dvoročestična Qiskit provera nije prošla.",
    )

    for label in range(dimension):
        require(
            abs(
                actual[label + dimension * label]
            ) < 2e-10,
            "Paulijevo isključivanje nije prošlo.",
        )

    return error


def self_checks():
    say(
        "",
        "Pokrećem ugrađene Qiskit i matematičke samoprovere.",
    )

    # Explicit singular-determinant check.
    phase, logarithm = checked_slogdet(
        np.array(
            [[1.0, 1.0], [1.0, 1.0]],
            dtype=complex,
        )
    )

    require(
        phase == 0 and np.isneginf(logarithm),
        "Provera nulte determinante nije prošla.",
    )

    require(
        checked_determinant(
            np.array([[0.0]], dtype=complex)
        ) == 0,
        "Skalarna nulta determinanta nije prošla.",
    )

    require(
        abs(
            checked_determinant(
                np.array(
                    [[1j, 0.0], [0.0, 2.0]],
                    dtype=complex,
                )
            ) - 2j
        ) < 1e-12,
        "Provera faze determinante nije prošla.",
    )

    source = QuantumOrbitals(9, 3, 1)

    theta = 0.02 * np.sin(
        np.arange(
            source.parameter_count,
            dtype=float,
        ) + 1.0
    )

    dimension = source.dimension
    labels = np.arange(dimension)

    for column in (0, 1, 3):
        actual = Statevector.from_int(
            column,
            dimension,
        ).evolve(source.fourier).data

        expected = (
            np.exp(
                2j * math.pi * labels * column / dimension
            )
            / math.sqrt(dimension)
        )

        require(
            np.max(
                np.abs(actual - expected)
            ) < 2e-10,
            "QFT provera redosleda kubita nije prošla.",
        )

    epsilon = 1e-6
    amplitude_error = 0.0

    for parameter in range(source.parameter_count):
        positive = theta.copy()
        negative = theta.copy()

        positive[parameter] += epsilon
        negative[parameter] -= epsilon

        finite_difference = (
            source.basis(positive)
            - source.basis(negative)
        ) / (2.0 * epsilon)

        exact = source.amplitude_derivative(
            theta,
            parameter,
        )

        amplitude_error = max(
            amplitude_error,
            float(np.max(
                np.abs(finite_difference - exact)
            )),
        )

    require(
        amplitude_error < 2e-7,
        "Provera izvoda kvantnih amplituda nije prošla.",
    )

    x = np.array(
        [
            [0.2, -0.4],
            [-0.3, 0.5],
            [0.7, 0.1],
            [-0.1, -0.2],
        ],
        dtype=float,
    )

    targets = np.array(
        [
            [0, 3, 7],
            [1, 4, 8],
            [0, 2, 5],
            [2, 6, 8],
        ],
        dtype=np.int64,
    )

    w = (
        0.02
        * np.sin(
            np.arange(x.shape[1] * source.m) + 1.0
        )
    ).reshape(x.shape[1], source.m)

    parameters = np.concatenate(
        (theta, w.ravel())
    )

    ridge = 0.03

    _, analytic_gradient = objective(
        parameters,
        source,
        x,
        targets,
        ridge,
    )

    numerical_gradient = np.empty_like(parameters)

    for parameter in range(len(parameters)):
        positive = parameters.copy()
        negative = parameters.copy()

        positive[parameter] += epsilon
        negative[parameter] -= epsilon

        positive_loss = objective(
            positive,
            source,
            x,
            targets,
            ridge,
            gradient=False,
        )

        negative_loss = objective(
            negative,
            source,
            x,
            targets,
            ridge,
            gradient=False,
        )

        numerical_gradient[parameter] = (
            positive_loss - negative_loss
        ) / (2.0 * epsilon)

    gradient_error = float(
        np.max(np.abs(
            analytic_gradient - numerical_gradient
        ))
    )

    require(
        gradient_error < 3e-5,
        "Provera ukupnog gradijenta nije prošla:"
        f" {gradient_error:.3e}",
    )

    h = np.linspace(-0.4, 0.5, source.m)

    filtered, filter_error = source.filtered_amplitudes(
        theta,
        h,
    )

    q = orthonormal_basis(filtered)

    best, best_logp, probability_sum, _ = exhaustive_maximum(
        q,
        "Mala samoprovera",
        verbose=False,
    )

    projector = q @ q.conj().T
    occupations = np.diag(projector).real

    require(
        np.min(occupations) >= -1e-10
        and np.max(occupations) <= 1.0 + 1e-10
        and abs(
            float(occupations.sum()) - source.k
        ) < 1e-9,
        "Provera zauzeća nije prošla.",
    )

    exact_occupations = np.zeros(source.m)
    direct_best = None
    direct_maximum = -1.0

    for selected in combinations(
        range(source.m),
        source.k,
    ):
        matrix = q[np.asarray(selected)]
        determinant = checked_determinant(matrix)
        probability = float(abs(determinant) ** 2)

        exact_occupations[
            np.asarray(selected)
        ] += probability

        if probability > direct_maximum:
            direct_maximum = probability
            direct_best = selected

    require(
        np.max(np.abs(
            exact_occupations - occupations
        )) < 1e-9,
        "Enumeracija fermionskih zauzeća nije prošla.",
    )

    require(
        best == direct_best
        and abs(
            math.exp(best_logp) - direct_maximum
        ) < 1e-10,
        "Provera maksimuma nije prošla.",
    )

    repeated = filtered[[0, 0, 2], :]

    require(
        abs(
            checked_determinant(repeated)
        ) < 1e-10,
        "Ponovljeni broj nema nultu fermionsku amplitudu.",
    )

    original = checked_determinant(
        filtered[[0, 1, 2], :]
    )

    exchanged = checked_determinant(
        filtered[[1, 0, 2], :]
    )

    require(
        abs(original + exchanged) < 1e-10,
        "Promena znaka pri razmeni čestica nije prošla.",
    )

    for k in (1, 6):
        boundary_source = QuantumOrbitals(6, k, 1)

        boundary_theta = np.zeros(
            boundary_source.parameter_count
        )

        boundary_q = orthonormal_basis(
            boundary_source.basis(boundary_theta)
        )

        exhaustive_maximum(
            boundary_q,
            "Granična provera",
            verbose=False,
        )

    toy_draws = np.array(
        [
            [0, 1, 2],
            [2, 3, 4],
            [1, 4, 5],
        ],
        dtype=np.int64,
    )

    context = make_context(
        toy_draws,
        [1],
        lookback=3,
        number_count=6,
    )

    require(
        np.count_nonzero(
            context[0, :6]
        ) == 3
        and context[0, 6] == 1
        and np.all(context[0, 7:] == 0),
        "Provera prvog istorijskog prelaza nije prošla.",
    )

    antisymmetry_error = verify_two_particle_state()

    say(
        "Ugrađene samoprovere su prošle.",
        f"Greška izvoda amplituda: {amplitude_error:.3e}",
        f"Greška ukupnog gradijenta: {gradient_error:.3e}",
        f"Greška Qiskit filtera: {filter_error:.3e}",
        "Greška pune dvoročestične simulacije:"
        f" {antisymmetry_error:.3e}",
        f"Zbir male raspodele: {probability_sum:.12f}",
    )


def longest_run(numbers):
    best = 1
    current = 1

    for left, right in zip(numbers, numbers[1:]):
        current = current + 1 if right == left + 1 else 1
        best = max(best, current)

    return best


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
        for lag in sorted(
            {item[0] for item in CANDIDATES}
        )
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

        for fold, (
            train_end,
            validation_end,
        ) in enumerate(folds, 1):
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

        scores.append(
            (score, lookback, ridge)
        )

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

    say(
        f"ZAVRŠNO UČENJE: svih {n} istorijskih prelaza."
    )

    raw_next = make_context(
        draws,
        [len(draws)],
        lookback,
        NUMBER_COUNT,
    )[0]

    q_next, filtered, filter_error = final_model.basis(
        raw_next,
        explicit_filter=True,
    )

    next_projector = q_next @ q_next.conj().T
    occupations = np.diag(next_projector).real

    require(
        np.min(occupations) >= -1e-9
        and np.max(occupations) <= 1.0 + 1e-9
        and abs(
            float(occupations.sum()) - DRAW_SIZE
        ) < 1e-8,
        "NEXT fermionske zauzetosti nisu ispravne.",
    )

    log_acceptance = float(
        positive_logdet(
            filtered.conj().T @ filtered
        )
    )

    require(
        log_acceptance <= 1e-7,
        "Verovatnoća prihvatanja filtera prelazi jedan.",
    )

    sensitivity = 0.0

    for previous_raw in raw[-8:]:
        q_previous, _, _ = final_model.basis(
            previous_raw
        )

        previous_projector = (
            q_previous @ q_previous.conj().T
        )

        sensitivity = max(
            sensitivity,
            float(np.linalg.norm(
                next_projector - previous_projector
            )),
        )

    say(
        "",
        "Qiskit filter proveren:"
        f" greška {filter_error:.3e}",
        "Log-verovatnoća prihvatanja filtera:"
        f" {log_acceptance:.9f}",
        "Osetljivost potprostora na istoriju:"
        f" {sensitivity:.9e}",
        "Potpuna pretraga svih legalnih sedmorki.",
    )

    best, logp, probability_sum, visited = exhaustive_maximum(
        q_next,
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

    repetitions = int(
        np.count_nonzero(
            np.all(
                draws == np.asarray(best)[None, :],
                axis=1,
            )
        )
    )

    if run >= 5:
        warnings.append(
            f"NEXT sadrži niz od {run} uzastopnih brojeva."
        )

    if span <= 9:
        warnings.append(
            "Svih sedam brojeva nalazi se"
            " u rasponu od deset brojeva."
        )

    if repetitions:
        warnings.append(
            f"NEXT se pojavljuje {repetitions} puta u istoriji."
        )

    if sensitivity < 1e-8:
        warnings.append(
            "Izlazni potprostor slabo reaguje"
            " na proverene kontekste."
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
        "log_acceptance": log_acceptance,
        "warnings": warnings,
    }


def main():
    np.seterr(
        divide="raise",
        over="raise",
        invalid="raise",
        under="ignore",
    )

    say(
        "QUANTUM PAULI FERMION V1 — Loto 7/39",
        f"Python: {sys.version.split()[0]}",
        f"Qiskit: {qiskit.__version__}",
        "Orbitalne amplitude nastaju izvršavanjem Qiskit kola.",
        "Sedam orbitala predstavlja sedam fermiona.",
        "Svaka orbitala koristi registar od šest kubita.",
        "Amplitudni filter koristi još jedan pomoćni kubit.",
        "Sedmočestično stanje koristi kompaktnu Slaterovu reprezentaciju.",
        "Antisimetrija i Paulijevo isključivanje ulaze kroz determinantu.",
        "Trening koristi analitički oblik kvantnog filtera.",
        "NEXT koristi eksplicitnu Qiskit proveru tog filtera.",
        "Optimizator i determinantalni proračun su klasični.",
        "Nema klasičnog regresora koji zamenjuje orbitalno kvantno kolo.",
        "Nema slučajnog generisanja niti uzorkovanja merenja.",
        "Optimizacija je lokalna; globalni minimum nije garantovan.",
        "Uključena je lokalna numerička obrada kontrolisane RY kapije.",
        "Sve direktne determinante u samoproverama koriste proverenu funkciju.",
    )

    resolved = [
        path.resolve()
        for path in FILES
    ]

    require(
        len(set(resolved)) == len(FILES),
        "Ulazne putanje nisu različite.",
    )

    datasets = []
    fingerprints = []

    for path in FILES:
        require(
            path.is_file(),
            f"Fajl nije pronađen: {path}",
        )

        draws, fingerprint = load_draws(path)

        datasets.append(
            (path, draws, fingerprint)
        )

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
            if (
                results[first]["next"]
                == results[second]["next"]
            ):
                results[first]["warnings"].append(
                    "Ista NEXT kombinacija kao u "
                    + results[second]["name"]
                    + "; proveriti osetljivost istorije"
                    + " i degeneraciju."
                )

                results[second]["warnings"].append(
                    "Ista NEXT kombinacija kao u "
                    + results[first]["name"]
                    + "; proveriti osetljivost istorije"
                    + " i degeneraciju."
                )

    say(
        "",
        "ZAVRŠNA TRI NEXT REZULTATA",
    )

    for result in results:
        numbers = " ".join(
            f"{value:02d}"
            for value in result["next"]
        )

        say(
            "",
            result["name"],
            f"NEXT: {numbers}",
            f"Validacioni NLL: {result['validation_nll']:.9f}",
            f"Odvojeni test NLL: {result['test_nll']:.9f}",
            "Modelna verovatnoća NEXT:"
            f" {result['probability']:.12g}",
            "Zbir svih verovatnoća:"
            f" {result['probability_sum']:.12f}",
            "Pregledanih kombinacija:"
            f" {result['visited']:,}",
            "Osetljivost na istoriju:"
            f" {result['sensitivity']:.9e}",
            "Greška Qiskit filtera:"
            f" {result['filter_error']:.3e}",
            "Log-prihvatanje filtera:"
            f" {result['log_acceptance']:.9f}",
        )

        for warning in result["warnings"]:
            say(f"DIJAGNOSTIKA: {warning}")

    say(
        "",
        "Obrada je završena.",
        "NEXT je numerički maksimum naučene fermionske raspodele.",
        "Potpuna pretraga potvrđuje maksimum i normalizaciju.",
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
QUANTUM PAULI FERMION V1 — Loto 7/39
Python: 3.11.13
Qiskit: 1.4.4
Orbitalne amplitude nastaju izvršavanjem Qiskit kola.
Sedam orbitala predstavlja sedam fermiona.
Svaka orbitala koristi registar od šest kubita.
Amplitudni filter koristi još jedan pomoćni kubit.
Sedmočestično stanje koristi kompaktnu Slaterovu reprezentaciju.
Antisimetrija i Paulijevo isključivanje ulaze kroz determinantu.
Trening koristi analitički oblik kvantnog filtera.
NEXT koristi eksplicitnu Qiskit proveru tog filtera.
Optimizator i determinantalni proračun su klasični.
Nema klasičnog regresora koji zamenjuje orbitalno kvantno kolo.
Nema slučajnog generisanja niti uzorkovanja merenja.
Optimizacija je lokalna; globalni minimum nije garantovan.
Uključena je lokalna numerička obrada kontrolisane RY kapije.
Sve direktne determinante u samoproverama koriste proverenu funkciju.

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

Pokrećem ugrađene Qiskit i matematičke samoprovere.
Ugrađene samoprovere su prošle.
Greška izvoda amplituda: 8.271e-11
Greška ukupnog gradijenta: 6.460e-10
Greška Qiskit filtera: 1.665e-16
Greška pune dvoročestične simulacije: 7.473e-17
Zbir male raspodele: 1.000000000000

loto7_4710_k86.csv / validacija 1
Lag=1 | λ=0.03 | početak=0 | prelaza=2825
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Prihvaćen lokalni minimum: 17.657092298
Max|grad|: 7.199e-07

loto7_4710_k86.csv / validacija 1
Lag=1 | λ=0.03 | početak=1 | prelaza=2825
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Prihvaćen lokalni minimum: 17.768156004
Max|grad|: 8.429e-07
Validacioni NLL: 17.872690606

loto7_4710_k86.csv / validacija 2
Lag=1 | λ=0.03 | početak=0 | prelaza=3296
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Prihvaćen lokalni minimum: 17.716473379
Max|grad|: 3.294e-06

loto7_4710_k86.csv / validacija 2
Lag=1 | λ=0.03 | početak=1 | prelaza=3296
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Prihvaćen lokalni minimum: 17.749962406
Max|grad|: 8.771e-07
Validacioni NLL: 17.652896132
KANDIDAT: lag=1 | λ=0.03
Zbirni validacioni NLL: 17.762793369

loto7_4710_k86.csv / validacija 1
Lag=1 | λ=0.3 | početak=0 | prelaza=2825
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Prihvaćen lokalni minimum: 17.772400222
Max|grad|: 8.170e-07

loto7_4710_k86.csv / validacija 1
Lag=1 | λ=0.3 | početak=1 | prelaza=2825
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Optimizacija: 175 iteracija
Prihvaćen lokalni minimum: 17.743596100
Max|grad|: 1.008e-06
Validacioni NLL: 17.719540467

loto7_4710_k86.csv / validacija 2
Lag=1 | λ=0.3 | početak=0 | prelaza=3296
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Prihvaćen lokalni minimum: 17.806715009
Max|grad|: 6.980e-07

loto7_4710_k86.csv / validacija 2
Lag=1 | λ=0.3 | početak=1 | prelaza=3296
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Prihvaćen lokalni minimum: 17.821878144
Max|grad|: 6.970e-07
Validacioni NLL: 17.716953504
KANDIDAT: lag=1 | λ=0.3
Zbirni validacioni NLL: 17.718246985

loto7_4710_k86.csv / validacija 1
Lag=3 | λ=0.03 | početak=0 | prelaza=2825
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Prihvaćen lokalni minimum: 17.611971745
Max|grad|: 7.038e-07

loto7_4710_k86.csv / validacija 1
Lag=3 | λ=0.03 | početak=1 | prelaza=2825
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Prihvaćen lokalni minimum: 17.761173060
Max|grad|: 9.413e-07
Validacioni NLL: 17.856217755

loto7_4710_k86.csv / validacija 2
Lag=3 | λ=0.03 | početak=0 | prelaza=3296
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Optimizacija: 175 iteracija
Prihvaćen lokalni minimum: 17.711769927
Max|grad|: 8.466e-07

loto7_4710_k86.csv / validacija 2
Lag=3 | λ=0.03 | početak=1 | prelaza=3296
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Prihvaćen lokalni minimum: 17.820903877
Max|grad|: 5.025e-07
Validacioni NLL: 17.679752415
KANDIDAT: lag=3 | λ=0.03
Zbirni validacioni NLL: 17.767985085

loto7_4710_k86.csv / validacija 1
Lag=3 | λ=0.3 | početak=0 | prelaza=2825
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Prihvaćen lokalni minimum: 17.676647904
Max|grad|: 9.875e-07

loto7_4710_k86.csv / validacija 1
Lag=3 | λ=0.3 | početak=1 | prelaza=2825
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Prihvaćen lokalni minimum: 17.832023729
Max|grad|: 1.364e-06
Validacioni NLL: 17.913262517

loto7_4710_k86.csv / validacija 2
Lag=3 | λ=0.3 | početak=0 | prelaza=3296
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Prihvaćen lokalni minimum: 17.831490903
Max|grad|: 5.972e-07

loto7_4710_k86.csv / validacija 2
Lag=3 | λ=0.3 | početak=1 | prelaza=3296
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Prihvaćen lokalni minimum: 17.919552856
Max|grad|: 1.188e-06
Validacioni NLL: 17.831455073
KANDIDAT: lag=3 | λ=0.3
Zbirni validacioni NLL: 17.872358795

IZABRANO: lag=1 | λ=0.3
Validacioni NLL: 17.718246985

loto7_4710_k86.csv / odvojeni test
Lag=1 | λ=0.3 | početak=0 | prelaza=3767
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Prihvaćen lokalni minimum: 17.792943350
Max|grad|: 7.239e-07

loto7_4710_k86.csv / odvojeni test
Lag=1 | λ=0.3 | početak=1 | prelaza=3767
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Prihvaćen lokalni minimum: 17.824776504
Max|grad|: 5.335e-07
ODVOJENI TEST: 942 prelaza
NLL cele sedmorke: 17.783645604

loto7_4710_k86.csv / završno učenje
Lag=1 | λ=0.3 | početak=0 | prelaza=4709
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Prihvaćen lokalni minimum: 17.764213187
Max|grad|: 2.379e-06

loto7_4710_k86.csv / završno učenje
Lag=1 | λ=0.3 | početak=1 | prelaza=4709
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Prihvaćen lokalni minimum: 17.856085535
Max|grad|: 4.726e-07
ZAVRŠNO UČENJE: svih 4709 istorijskih prelaza.

Qiskit filter proveren: greška 9.719e-15
Log-verovatnoća prihvatanja filtera: -4.513148369
Osetljivost potprostora na istoriju: 2.083450897e-01
Potpuna pretraga svih legalnih sedmorki.
loto7_4710_k86.csv: pregledano 2,002,944/15,380,937
loto7_4710_k86.csv: pregledano 4,001,792/15,380,937
loto7_4710_k86.csv: pregledano 6,000,640/15,380,937
loto7_4710_k86.csv: pregledano 8,003,584/15,380,937
loto7_4710_k86.csv: pregledano 10,002,432/15,380,937
loto7_4710_k86.csv: pregledano 12,001,280/15,380,937
loto7_4710_k86.csv: pregledano 14,000,128/15,380,937

loto7_4710_k86_loto_2977.csv / validacija 1
Lag=1 | λ=0.03 | početak=0 | prelaza=1785
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Prihvaćen lokalni minimum: 17.705412485
Max|grad|: 6.429e-07

loto7_4710_k86_loto_2977.csv / validacija 1
Lag=1 | λ=0.03 | početak=1 | prelaza=1785
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Prihvaćen lokalni minimum: 17.681359534
Max|grad|: 7.832e-07
Validacioni NLL: 17.771550242

loto7_4710_k86_loto_2977.csv / validacija 2
Lag=1 | λ=0.03 | početak=0 | prelaza=2083
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Prihvaćen lokalni minimum: 17.667117969
Max|grad|: 7.358e-07

loto7_4710_k86_loto_2977.csv / validacija 2
Lag=1 | λ=0.03 | početak=1 | prelaza=2083
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Prihvaćen lokalni minimum: 17.754701376
Max|grad|: 5.902e-07
Validacioni NLL: 17.745517692
KANDIDAT: lag=1 | λ=0.03
Zbirni validacioni NLL: 17.758555843

loto7_4710_k86_loto_2977.csv / validacija 1
Lag=1 | λ=0.3 | početak=0 | prelaza=1785
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Prihvaćen lokalni minimum: 17.809691019
Max|grad|: 7.117e-07

loto7_4710_k86_loto_2977.csv / validacija 1
Lag=1 | λ=0.3 | početak=1 | prelaza=1785
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Optimizacija: 175 iteracija
Prihvaćen lokalni minimum: 17.767851798
Max|grad|: 4.417e-07
Validacioni NLL: 17.572326640

loto7_4710_k86_loto_2977.csv / validacija 2
Lag=1 | λ=0.3 | početak=0 | prelaza=2083
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Prihvaćen lokalni minimum: 17.773400812
Max|grad|: 7.083e-07

loto7_4710_k86_loto_2977.csv / validacija 2
Lag=1 | λ=0.3 | početak=1 | prelaza=2083
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Prihvaćen lokalni minimum: 17.812173856
Max|grad|: 5.536e-07
Validacioni NLL: 17.718556464
KANDIDAT: lag=1 | λ=0.3
Zbirni validacioni NLL: 17.645318670

loto7_4710_k86_loto_2977.csv / validacija 1
Lag=3 | λ=0.03 | početak=0 | prelaza=1785
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Prihvaćen lokalni minimum: 17.697514489
Max|grad|: 6.654e-07

loto7_4710_k86_loto_2977.csv / validacija 1
Lag=3 | λ=0.03 | početak=1 | prelaza=1785
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
Prihvaćen lokalni minimum: 17.569436611
Max|grad|: 6.681e-07
Validacioni NLL: 17.797405875

loto7_4710_k86_loto_2977.csv / validacija 2
Lag=3 | λ=0.03 | početak=0 | prelaza=2083
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Prihvaćen lokalni minimum: 17.684703096
Max|grad|: 6.123e-07

loto7_4710_k86_loto_2977.csv / validacija 2
Lag=3 | λ=0.03 | početak=1 | prelaza=2083
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Optimizacija: 175 iteracija
Prihvaćen lokalni minimum: 17.711541283
Max|grad|: 4.562e-07
Validacioni NLL: 17.752845073
KANDIDAT: lag=3 | λ=0.03
Zbirni validacioni NLL: 17.775162920

loto7_4710_k86_loto_2977.csv / validacija 1
Lag=3 | λ=0.3 | početak=0 | prelaza=1785
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Prihvaćen lokalni minimum: 17.773805720
Max|grad|: 3.442e-07

loto7_4710_k86_loto_2977.csv / validacija 1
Lag=3 | λ=0.3 | početak=1 | prelaza=1785
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Prihvaćen lokalni minimum: 17.710741781
Max|grad|: 5.310e-07
Validacioni NLL: 17.804858417

loto7_4710_k86_loto_2977.csv / validacija 2
Lag=3 | λ=0.3 | početak=0 | prelaza=2083
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Prihvaćen lokalni minimum: 17.775315581
Max|grad|: 4.410e-07

loto7_4710_k86_loto_2977.csv / validacija 2
Lag=3 | λ=0.3 | početak=1 | prelaza=2083
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Prihvaćen lokalni minimum: 17.846265681
Max|grad|: 4.124e-07
Validacioni NLL: 17.759402696
KANDIDAT: lag=3 | λ=0.3
Zbirni validacioni NLL: 17.782168754

IZABRANO: lag=1 | λ=0.3
Validacioni NLL: 17.645318670

loto7_4710_k86_loto_2977.csv / odvojeni test
Lag=1 | λ=0.3 | početak=0 | prelaza=2380
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Prihvaćen lokalni minimum: 17.732533929
Max|grad|: 1.738e-07

loto7_4710_k86_loto_2977.csv / odvojeni test
Lag=1 | λ=0.3 | početak=1 | prelaza=2380
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Prihvaćen lokalni minimum: 17.838520293
Max|grad|: 4.086e-07
ODVOJENI TEST: 596 prelaza
NLL cele sedmorke: 17.734029894

loto7_4710_k86_loto_2977.csv / završno učenje
Lag=1 | λ=0.3 | početak=0 | prelaza=2976
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Prihvaćen lokalni minimum: 17.862268701
Max|grad|: 1.418e-06

loto7_4710_k86_loto_2977.csv / završno učenje
Lag=1 | λ=0.3 | početak=1 | prelaza=2976
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Prihvaćen lokalni minimum: 17.854657076
Max|grad|: 3.825e-07
ZAVRŠNO UČENJE: svih 2976 istorijskih prelaza.

Qiskit filter proveren: greška 9.007e-15
Log-verovatnoća prihvatanja filtera: -6.497888870
Osetljivost potprostora na istoriju: 2.159306952e-01
Potpuna pretraga svih legalnih sedmorki.
loto7_4710_k86_loto_2977.csv: pregledano 2,002,944/15,380,937
loto7_4710_k86_loto_2977.csv: pregledano 4,001,792/15,380,937
loto7_4710_k86_loto_2977.csv: pregledano 6,000,640/15,380,937
loto7_4710_k86_loto_2977.csv: pregledano 8,003,584/15,380,937
loto7_4710_k86_loto_2977.csv: pregledano 10,002,432/15,380,937
loto7_4710_k86_loto_2977.csv: pregledano 12,001,280/15,380,937
loto7_4710_k86_loto_2977.csv: pregledano 14,000,128/15,380,937

loto7_4710_k86_loto_plus_1733.csv / validacija 1
Lag=1 | λ=0.03 | početak=0 | prelaza=1039
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
Prihvaćen lokalni minimum: 17.584156980
Max|grad|: 5.439e-07

loto7_4710_k86_loto_plus_1733.csv / validacija 1
Lag=1 | λ=0.03 | početak=1 | prelaza=1039
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Prihvaćen lokalni minimum: 17.684290277
Max|grad|: 8.791e-07
Validacioni NLL: 18.129726156

loto7_4710_k86_loto_plus_1733.csv / validacija 2
Lag=1 | λ=0.03 | početak=0 | prelaza=1212
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Prihvaćen lokalni minimum: 17.715170895
Max|grad|: 5.753e-07

loto7_4710_k86_loto_plus_1733.csv / validacija 2
Lag=1 | λ=0.03 | početak=1 | prelaza=1212
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Prihvaćen lokalni minimum: 17.638795629
Max|grad|: 4.474e-07
Validacioni NLL: 17.719426043
KANDIDAT: lag=1 | λ=0.03
Zbirni validacioni NLL: 17.924576100

loto7_4710_k86_loto_plus_1733.csv / validacija 1
Lag=1 | λ=0.3 | početak=0 | prelaza=1039
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Prihvaćen lokalni minimum: 17.790241779
Max|grad|: 3.196e-06

loto7_4710_k86_loto_plus_1733.csv / validacija 1
Lag=1 | λ=0.3 | početak=1 | prelaza=1039
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Prihvaćen lokalni minimum: 17.851768464
Max|grad|: 7.548e-07
Validacioni NLL: 17.837810902

loto7_4710_k86_loto_plus_1733.csv / validacija 2
Lag=1 | λ=0.3 | početak=0 | prelaza=1212
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
Prihvaćen lokalni minimum: 17.664471189
Max|grad|: 6.414e-07

loto7_4710_k86_loto_plus_1733.csv / validacija 2
Lag=1 | λ=0.3 | početak=1 | prelaza=1212
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Prihvaćen lokalni minimum: 17.766792184
Max|grad|: 7.814e-07
Validacioni NLL: 17.523452271
KANDIDAT: lag=1 | λ=0.3
Zbirni validacioni NLL: 17.680631587

loto7_4710_k86_loto_plus_1733.csv / validacija 1
Lag=3 | λ=0.03 | početak=0 | prelaza=1039
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Optimizacija: 150 iteracija
Prihvaćen lokalni minimum: 17.568586551
Max|grad|: 6.825e-07

loto7_4710_k86_loto_plus_1733.csv / validacija 1
Lag=3 | λ=0.03 | početak=1 | prelaza=1039
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Prihvaćen lokalni minimum: 17.667537177
Max|grad|: 9.516e-07
Validacioni NLL: 18.060839454

loto7_4710_k86_loto_plus_1733.csv / validacija 2
Lag=3 | λ=0.03 | početak=0 | prelaza=1212
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Prihvaćen lokalni minimum: 17.667430871
Max|grad|: 9.171e-07

loto7_4710_k86_loto_plus_1733.csv / validacija 2
Lag=3 | λ=0.03 | početak=1 | prelaza=1212
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Prihvaćen lokalni minimum: 17.662798572
Max|grad|: 1.470e-06
Validacioni NLL: 17.704908422
KANDIDAT: lag=3 | λ=0.03
Zbirni validacioni NLL: 17.882873938

loto7_4710_k86_loto_plus_1733.csv / validacija 1
Lag=3 | λ=0.3 | početak=0 | prelaza=1039
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Prihvaćen lokalni minimum: 17.841392417
Max|grad|: 8.884e-07

loto7_4710_k86_loto_plus_1733.csv / validacija 1
Lag=3 | λ=0.3 | početak=1 | prelaza=1039
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Prihvaćen lokalni minimum: 17.768071579
Max|grad|: 8.609e-07
Validacioni NLL: 17.955014128

loto7_4710_k86_loto_plus_1733.csv / validacija 2
Lag=3 | λ=0.3 | početak=0 | prelaza=1212
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Prihvaćen lokalni minimum: 18.139951385
Max|grad|: 1.867e-07

loto7_4710_k86_loto_plus_1733.csv / validacija 2
Lag=3 | λ=0.3 | početak=1 | prelaza=1212
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Prihvaćen lokalni minimum: 17.744329268
Max|grad|: 9.272e-07
Validacioni NLL: 17.687905040
KANDIDAT: lag=3 | λ=0.3
Zbirni validacioni NLL: 17.821459584

IZABRANO: lag=1 | λ=0.3
Validacioni NLL: 17.680631587

loto7_4710_k86_loto_plus_1733.csv / odvojeni test
Lag=1 | λ=0.3 | početak=0 | prelaza=1385
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Prihvaćen lokalni minimum: 17.770585115
Max|grad|: 9.904e-07

loto7_4710_k86_loto_plus_1733.csv / odvojeni test
Lag=1 | λ=0.3 | početak=1 | prelaza=1385
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Prihvaćen lokalni minimum: 17.831088573
Max|grad|: 9.202e-07
ODVOJENI TEST: 347 prelaza
NLL cele sedmorke: 17.950427341

loto7_4710_k86_loto_plus_1733.csv / završno učenje
Lag=1 | λ=0.3 | početak=0 | prelaza=1732
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Optimizacija: 125 iteracija
Prihvaćen lokalni minimum: 17.734484021
Max|grad|: 6.361e-07

loto7_4710_k86_loto_plus_1733.csv / završno učenje
Lag=1 | λ=0.3 | početak=1 | prelaza=1732
Parametara kola: 24
Parametara istorijskog upravljanja: 312
Optimizacija: 25 iteracija
Optimizacija: 50 iteracija
Optimizacija: 75 iteracija
Optimizacija: 100 iteracija
Prihvaćen lokalni minimum: 17.740765842
Max|grad|: 8.376e-07
ZAVRŠNO UČENJE: svih 1732 istorijskih prelaza.

Qiskit filter proveren: greška 1.046e-14
Log-verovatnoća prihvatanja filtera: -4.169094899
Osetljivost potprostora na istoriju: 2.414512988e-01
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
NEXT: 04 x 11 y 28 z 37
Validacioni NLL: 17.718246985
Odvojeni test NLL: 17.783645604
Modelna verovatnoća NEXT: 4.33046274651e-06
Zbir svih verovatnoća: 1.000000000000
Pregledanih kombinacija: 15,380,937
Osetljivost na istoriju: 2.083450897e-01
Greška Qiskit filtera: 9.719e-15
Log-prihvatanje filtera: -4.513148369

loto7_4710_k86_loto_2977.csv
NEXT: 09 x 20 y 29 z 38
Validacioni NLL: 17.645318670
Odvojeni test NLL: 17.734029894
Modelna verovatnoća NEXT: 6.40806617937e-06
Zbir svih verovatnoća: 1.000000000000
Pregledanih kombinacija: 15,380,937
Osetljivost na istoriju: 2.159306952e-01
Greška Qiskit filtera: 9.007e-15
Log-prihvatanje filtera: -6.497888870

loto7_4710_k86_loto_plus_1733.csv
NEXT: 02 x 10 y 23 z 34
Validacioni NLL: 17.680631587
Odvojeni test NLL: 17.950427341
Modelna verovatnoća NEXT: 5.67873052744e-06
Zbir svih verovatnoća: 1.000000000000
Pregledanih kombinacija: 15,380,937
Osetljivost na istoriju: 2.414512988e-01
Greška Qiskit filtera: 1.046e-14
Log-prihvatanje filtera: -4.169094899

Obrada je završena.
NEXT je numerički maksimum naučene fermionske raspodele.
Potpuna pretraga potvrđuje maksimum i normalizaciju.
Prediktivna vrednost procenjuje se odvojenim testom.
"""



"""
Kvantni model sa Qiskitom. 
Parametrizovana Qiskit kola, kompleksne amplitude i lokalna simulacija, sa izlaznom raspodelom izvedenom iz kvantnog stanja. 
Klasični optimizator može da podešava parametre. 
Slaterova determinanta i Paulijevo isključivanje, amplitude parametrizovanog Qiskit kola. 
Istorija će upravljati amplitudnim filterom sa pomoćnim kubitom. 
Sedam registara od po šest kubita obradjeno preko kompaktne Slaterove reprezentacije, bez gustog vektora stanja sa 42 kubita.

- Amplitude parametrizovanog Qiskit kola.
- Istorija upravlja kontrolisanim kvantnim amplitudnim filterom.
- Slaterova determinanta, antisimetrija i Paulijevo isključivanje.

Kompaktna simulacija fermionskog stanja. 
Qiskit simulira orbitalna kola; Slaterova determinanta tačno povezuje njihove amplitude u sedmočestično stanje. 
Ne pravi se gust vektor sa \(2^{42}\) amplituda. 
Tokom treninga amplitudni filter računa se analitički, a pre NEXT pretrage proverava se njegovim stvarnim Qiskit kolom sa pomoćnim kubitom.
"""
