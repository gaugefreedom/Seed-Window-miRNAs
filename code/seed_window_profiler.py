import argparse
import json
import math
import random
import re
import signal
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import sympy as sp


VALID_BASES = ("A", "U", "G", "C")
X, Y, Z = sp.symbols("x y z")
BOUNDARY_SYMBOLS = tuple(sp.symbols("a b c d"))
PARAMETER_SLICES = (
    {BOUNDARY_SYMBOLS[0]: 0, BOUNDARY_SYMBOLS[1]: 0, BOUNDARY_SYMBOLS[2]: 0, BOUNDARY_SYMBOLS[3]: 0},
    {BOUNDARY_SYMBOLS[0]: 1, BOUNDARY_SYMBOLS[1]: 1, BOUNDARY_SYMBOLS[2]: 0, BOUNDARY_SYMBOLS[3]: 0},
    {BOUNDARY_SYMBOLS[0]: 1, BOUNDARY_SYMBOLS[1]: 1, BOUNDARY_SYMBOLS[2]: 1, BOUNDARY_SYMBOLS[3]: 1},
)
MAX_PRACTICAL_GROEBNER_VARIABLES = 8

SINGULARITY_SURFACES = {
    "Cayley cubic f_H^(4)": sp.expand(X * Y * Z - X**2 - Y**2 - Z**2 + 4),
    "Fricke kappa_3": sp.expand(X * Y * Z - X**2 - Y**2 - Z**2 + 3),
    "A1 f_b": sp.expand(X**2 + Y**2 - 6 * Z**2 + 4 * X * Y * Z),
    "A1 f_(2,{})": sp.expand(X**2 + Y**2 + Z**2 + X * Y * Z - 2 * Y),
    "3A1 Fricke V_(1,1,1,1)": sp.expand(X**2 + Y**2 + Z**2 + X * Y * Z - 2 * X - 2 * Y - 2 * Z + 1),
}

DEFAULT_PROJECTION_VARIABLES = (X, Y, Z)
BENCHMARK_GENERATOR_ORDERS: Dict[str, Tuple[str, ...]] = {
    "AGCAGCGG": ("A", "C", "G"),
}
BENCHMARK_SLICE_TARGETS: Dict[str, Dict[str, object]] = {
    "(0,0,0,0)": {
        "label": "kappa_4 Cayley cubic",
        "target": sp.expand(X**2 + Y**2 + Z**2 + X * Y * Z - 4),
    },
    "(1,1,0,0)": {
        "label": "-3xyz * kappa_3 core",
        "raw_target": sp.expand(-3 * X * Y * Z * (X * Y * Z - X**2 - Y**2 - Z**2 + 3)),
        "target": sp.expand(X * Y * Z - X**2 - Y**2 - Z**2 + 3),
    },
    "(1,1,1,1)": {
        "label": "quoted Fricke surface",
        "target": sp.expand(X * Y * Z + X**2 + Y**2 + Z**2 - 2 * X - Y - 2),
    },
}
LEGACY_BENCHMARK_WINDOWS = {"AGCAGCGG"}


class WindowComputationTimeout(Exception):
    pass

# Regression fixtures recovered from the 2023 workflow/paper.
PUBLISHED_WINDOW_RESULTS = {
    "UCCUAC": {
        "basis_equations": [str(SINGULARITY_SURFACES["A1 f_b"])],
        "is_singular": True,
        "singularity_type": "A1 f_b",
        "free_like_class": "F_2",
        "analysis_status": "published-regression",
    },
    "UCCUACA": {
        "basis_equations": [],
        "is_singular": False,
        "singularity_type": "None",
        "free_like_class": "pi_2",
        "analysis_status": "published-regression",
    },
    "AAGGCA": {
        "basis_equations": [str(SINGULARITY_SURFACES["A1 f_(2,{})"])],
        "is_singular": True,
        "singularity_type": "A1 f_(2,{})",
        "free_like_class": "F_2",
        "analysis_status": "published-regression",
    },
    "AAGGCAC": {
        "basis_equations": [],
        "is_singular": False,
        "singularity_type": "None",
        "free_like_class": "pi_2",
        "analysis_status": "published-regression",
    },
}


def validate_sequence(sequence: str) -> str:
    seq = sequence.upper().replace("T", "U")
    invalid = sorted(set(seq) - set(VALID_BASES))
    if invalid:
        raise ValueError(f"Unsupported RNA bases found: {','.join(invalid)}")
    return seq


@dataclass(frozen=True)
class Letter:
    symbol: str
    sign: int = 1

    def inverse(self) -> "Letter":
        return Letter(self.symbol, -self.sign)

    def render(self) -> str:
        return self.symbol if self.sign == 1 else self.symbol.lower()


@dataclass(frozen=True)
class Word:
    letters: Tuple[Letter, ...]

    @classmethod
    def from_sequence(cls, sequence: str) -> "Word":
        return cls(tuple(Letter(base) for base in validate_sequence(sequence)))

    def inverse(self) -> "Word":
        return Word(tuple(letter.inverse() for letter in reversed(self.letters)))

    def reduced(self) -> "Word":
        stack: List[Letter] = []
        for letter in self.letters:
            if stack and stack[-1].symbol == letter.symbol and stack[-1].sign + letter.sign == 0:
                stack.pop()
            else:
                stack.append(letter)
        return Word(tuple(stack))

    def cyclic_rotations(self) -> List["Word"]:
        if not self.letters:
            return [self]
        letters = list(self.letters)
        return [Word(tuple(letters[i:] + letters[:i])) for i in range(len(letters))]

    def cyclic_normal_form(self) -> "Word":
        candidates = self.cyclic_rotations() + self.inverse().cyclic_rotations()
        return min(candidates, key=lambda word: word.render())

    def render(self) -> str:
        return "".join(letter.render() for letter in self.letters)

    def __len__(self) -> int:
        return len(self.letters)

    def __getitem__(self, item: slice | int) -> "Word | Letter":
        if isinstance(item, slice):
            return Word(self.letters[item])
        return self.letters[item]

    def concatenate(self, other: "Word") -> "Word":
        return Word(self.letters + other.letters).reduced()


@dataclass(frozen=True)
class SubstitutionRule:
    mapping: Dict[str, str]
    source: str = "user"

    def normalized(self) -> "SubstitutionRule":
        normalized = {validate_sequence(key)[0]: validate_sequence(value) for key, value in self.mapping.items()}
        return SubstitutionRule(mapping=normalized, source=self.source)

    def alphabet(self) -> Tuple[str, ...]:
        return tuple(sorted(self.mapping))


def infer_substitution_rule(word: str) -> SubstitutionRule:
    """
    Heuristic fallback when no substitution rule is supplied.
    It uses cyclic successor counts: each base maps to the ordered sequence of
    letters that follow it around the cyclic word.
    """
    sequence = validate_sequence(word)
    successors: Dict[str, List[str]] = {base: [] for base in dict.fromkeys(sequence)}

    for idx, base in enumerate(sequence):
        successors[base].append(sequence[(idx + 1) % len(sequence)])

    return SubstitutionRule(mapping={base: "".join(next_letters) for base, next_letters in successors.items()}, source="heuristic-cyclic-successor")


def build_substitution_matrix(rule: SubstitutionRule) -> Tuple[Tuple[str, ...], sp.Matrix]:
    normalized = rule.normalized()
    alphabet = normalized.alphabet()
    index = {base: i for i, base in enumerate(alphabet)}
    matrix = sp.zeros(len(alphabet), len(alphabet))

    for source, image in normalized.mapping.items():
        for letter in image:
            matrix[index[letter], index[source]] += 1

    return alphabet, matrix


def is_primitive_matrix(matrix: sp.Matrix) -> Tuple[bool, Optional[int]]:
    if matrix.rows == 0:
        return False, None

    bound = max(1, matrix.rows**2 - 2 * matrix.rows + 2)
    power = sp.eye(matrix.rows)
    for exponent in range(1, bound + 1):
        power *= matrix
        if all(value > 0 for value in power):
            return True, exponent
    return False, None


def _largest_real_eigenvalue(matrix: sp.Matrix) -> sp.Expr:
    eigenvalues = []
    for value, multiplicity in matrix.eigenvals().items():
        eigenvalues.extend([sp.N(value, 32)] * multiplicity)

    real_values = [sp.re(value) for value in eigenvalues if abs(float(sp.im(value))) < 1e-12]
    if not real_values:
        raise ValueError("Substitution matrix has no real eigenvalues to evaluate.")
    return max(real_values, key=lambda val: float(val))


def _is_irrational(value: sp.Expr) -> bool:
    simplified = sp.nsimplify(value)
    return not bool(getattr(simplified, "is_rational", False))


def check_aperiodicity(word: str, substitution_rule: Optional[Dict[str, str] | SubstitutionRule] = None) -> Dict[str, object]:
    """
    Accepts a user-specified rule or falls back to a heuristic rule, then
    checks primitiveness and irrational PF eigenvalue.
    """
    validate_sequence(word)
    if substitution_rule is None:
        rule = infer_substitution_rule(word)
    elif isinstance(substitution_rule, SubstitutionRule):
        rule = substitution_rule
    else:
        rule = SubstitutionRule(mapping=substitution_rule, source="user")

    alphabet, matrix = build_substitution_matrix(rule)
    is_primitive, primitive_exponent = is_primitive_matrix(matrix)
    pf_eigenvalue = _largest_real_eigenvalue(matrix)
    irrational_pf = _is_irrational(pf_eigenvalue)

    return {
        "rule_source": rule.source,
        "rule": rule.normalized().mapping,
        "alphabet": alphabet,
        "matrix": [[int(matrix[i, j]) for j in range(matrix.cols)] for i in range(matrix.rows)],
        "is_primitive": is_primitive,
        "primitive_exponent": primitive_exponent,
        "pf_eigenvalue": str(sp.nsimplify(pf_eigenvalue)),
        "pf_eigenvalue_numeric": float(sp.N(pf_eigenvalue, 16)),
        "pf_is_irrational": irrational_pf,
        "is_aperiodic": is_primitive and irrational_pf,
    }


@dataclass(frozen=True)
class GroupPresentation:
    motif: str
    generators: Tuple[str, ...]
    rank: int
    relator: Word

    @property
    def generator_count(self) -> int:
        return len(self.generators)


def build_presentation(window: str) -> GroupPresentation:
    window = validate_sequence(window)
    benchmark_order = BENCHMARK_GENERATOR_ORDERS.get(window)
    if benchmark_order is not None:
        generators = tuple(generator for generator in benchmark_order if generator in set(window))
    else:
        generators = tuple(dict.fromkeys(window))
    relator = Word.from_sequence(window).cyclic_normal_form()
    return GroupPresentation(motif=window, generators=generators, rank=max(len(generators) - 1, 0), relator=relator)


def generate_sliding_windows(sequence: str, window_sizes: Sequence[int] = (6, 7, 8)) -> List[Dict[str, int | str]]:
    sequence = validate_sequence(sequence)
    windows: List[Dict[str, int | str]] = []
    for k in window_sizes:
        for i in range(len(sequence) - k + 1):
            windows.append({"window_idx": len(windows), "start": i, "end": i + k, "length": k, "motif": sequence[i : i + k]})
    return windows


def generate_seed_centered_windows(
    sequence: str,
    window_sizes: Sequence[int] = (6, 7, 8),
    seed_start: int = 1,
    seed_end: int = 8,
    max_start_shift: int = 1,
) -> List[Dict[str, int | str]]:
    all_windows = generate_sliding_windows(sequence, window_sizes)
    min_start = max(0, seed_start - max_start_shift)
    max_start = seed_start + max_start_shift
    centered = [
        window
        for window in all_windows
        if min_start <= int(window["start"]) <= max_start
    ]
    for window in centered:
        window["seed_start"] = seed_start
        window["seed_end"] = seed_end
        window["seed_overlap"] = min(int(window["end"]), seed_end) - max(int(window["start"]), seed_start)
        window["seed_window_mode"] = "canonical-seed-neighborhood"
    return centered


def trace_symbol(word: Word) -> sp.Symbol:
    label = word.cyclic_normal_form().render() or "I"
    return sp.Symbol(f"tr_{label}")


def _word_substitutions(presentation: GroupPresentation) -> Dict[str, sp.Expr]:
    generators = presentation.generators
    def _label(word: str) -> str:
        return f"tr_{Word.from_sequence(word).cyclic_normal_form().render() or 'I'}"

    substitutions: Dict[str, sp.Expr] = {
        "tr_I": sp.Integer(2),
        _label(generators[0] if generators else ""): sp.Symbol("a") if generators else sp.Integer(2),
    }
    if len(generators) >= 2:
        substitutions[_label(generators[1])] = sp.Symbol("b")
        substitutions[_label("".join(generators[:2]))] = X
    if len(generators) >= 3:
        substitutions[_label(generators[2])] = sp.Symbol("c")
        substitutions[_label("".join(generators[1:3]))] = Y
        substitutions[_label(f"{generators[2]}{generators[0]}")] = Z
    if len(generators) >= 4:
        substitutions[_label(generators[3])] = sp.Symbol("d")
    return substitutions


class TraceAlgebra:
    """
    Recursive trace reducer on abstract words. It handles cyclic/inverse
    normalization and reduces words into a Fricke-style coordinate basis plus
    residual canonical trace symbols when a full elimination identity has not
    yet been implemented.
    """

    def __init__(self, presentation: GroupPresentation):
        self.presentation = presentation
        self.coordinate_map = _word_substitutions(presentation)
        self.promoted_symbols: set[sp.Symbol] = set()
        self.generator_symbols = {
            generator: self.coordinate_map[f"tr_{generator}"]
            for generator in presentation.generators
            if f"tr_{generator}" in self.coordinate_map
        }

    def basis_symbol(self, word: Word) -> sp.Expr:
        canonical = word.cyclic_normal_form()
        if len(canonical) == 1 and canonical.letters[0].sign == -1:
            canonical = Word((canonical.letters[0].inverse(),))
        label = f"tr_{canonical.render() or 'I'}"
        if label in self.coordinate_map:
            return self.coordinate_map[label]
        promoted = trace_symbol(canonical)
        self.promoted_symbols.add(promoted)
        return promoted

    @lru_cache(maxsize=8192)
    def reduce_word(self, rendered_word: str) -> sp.Expr:
        word = Word(tuple(Letter(letter.upper(), 1 if letter.isupper() else -1) for letter in rendered_word)).reduced().cyclic_normal_form()

        if len(word) == 0:
            return sp.Integer(2)
        for block_len in (2, 1):
            for idx in range(len(word) - 2 * block_len + 1):
                first_block = Word(word.letters[idx : idx + block_len])
                second_block = Word(word.letters[idx + block_len : idx + 2 * block_len])

                # Benchmark accelerator: repeated adjacent block X X.
                if first_block.letters == second_block.letters:
                    u_word = Word(word.letters[:idx])
                    v_word = Word(word.letters[idx + 2 * block_len :])
                    uxv_word = u_word.concatenate(first_block).concatenate(v_word)
                    uv_word = u_word.concatenate(v_word)
                    return sp.expand(self.reduce(first_block) * self.reduce(uxv_word) - self.reduce(uv_word))

                # Inverse translation accelerator: U X^{-1} V.
                if all(letter.sign == -1 for letter in first_block.letters):
                    u_word = Word(word.letters[:idx])
                    x_word = first_block.inverse().reduced()
                    v_word = Word(word.letters[idx + block_len :])
                    return sp.expand(self.reduce(x_word) * self.reduce(u_word.concatenate(v_word)) - self.reduce(u_word.concatenate(x_word).concatenate(v_word)))

        for idx in range(len(word) - 1):
            first = word[idx]
            second = word[idx + 1]
            if first.symbol == second.symbol and first.sign == second.sign:
                p_word = Word(word.letters[:idx])
                a_word = Word((first,))
                q_word = Word(word.letters[idx + 2 :])
                paq_word = p_word.concatenate(a_word).concatenate(q_word)
                pq_word = p_word.concatenate(q_word)
                return sp.expand(self.reduce(a_word) * self.reduce(paq_word) - self.reduce(pq_word))
            if first.symbol != second.symbol and first.sign == 1 and second.sign == -1:
                p_word = Word(word.letters[:idx])
                a_word = Word((first,))
                b_word = Word((second.inverse(),))
                q_word = Word(word.letters[idx + 2 :])
                paq_word = p_word.concatenate(a_word).concatenate(q_word)
                pabq_word = p_word.concatenate(a_word).concatenate(b_word).concatenate(q_word)
                return sp.expand(self.reduce(b_word) * self.reduce(paq_word) - self.reduce(pabq_word))
            if first.symbol != second.symbol and first.sign == -1 and second.sign == 1:
                p_word = Word(word.letters[:idx])
                a_word = Word((first.inverse(),))
                b_word = Word((second,))
                q_word = Word(word.letters[idx + 2 :])
                pbq_word = p_word.concatenate(b_word).concatenate(q_word)
                pabq_word = p_word.concatenate(a_word).concatenate(b_word).concatenate(q_word)
                return sp.expand(self.reduce(a_word) * self.reduce(pbq_word) - self.reduce(pabq_word))
        if len(word) == 1 and word[0].sign == -1:
            return self.reduce(Word((word[0].inverse(),)))
        if len(word) <= 2:
            if len(word) == 2:
                first = word[0]
                second = word[1]
                if first.sign == -1 and second.sign == -1:
                    return self.reduce(Word((second.inverse(), first.inverse())))
                if first.sign == -1 and second.sign == 1:
                    return sp.expand(self.reduce(Word((first.inverse(),))) * self.reduce(Word((second,))) - self.reduce(Word((first.inverse(), second.inverse()))))
                if first.sign == 1 and second.sign == -1:
                    return sp.expand(self.reduce(Word((first,))) * self.reduce(Word((second.inverse(),))) - self.reduce(Word((first, second.inverse()))))
            return self.basis_symbol(word)
        if len(word) == 3:
            a_word = Word((word[0],))
            b_word = Word((word[1],))
            c_word = Word((word[2],))
            abc_repr = word.render()
            acb_word = Word((word[0], word[2], word[1])).cyclic_normal_form()
            acb_repr = acb_word.render()

            if abc_repr < acb_repr:
                return self.basis_symbol(word)

            a_expr = self.reduce(a_word)
            b_expr = self.reduce(b_word)
            c_expr = self.reduce(c_word)
            bc_expr = self.reduce(Word((word[1], word[2])))
            ac_expr = self.reduce(Word((word[0], word[2])))
            ab_expr = self.reduce(Word((word[0], word[1])))
            acb_expr = self.basis_symbol(acb_word)

            return sp.expand(a_expr * bc_expr + b_expr * ac_expr + c_expr * ab_expr - a_expr * b_expr * c_expr - acb_expr)
        # Verified against Lawton's Main.sagews: for len(s) >= 4, with
        # s = x y z w (where w is the remaining tail block), use the grouped
        # Vogt/Fricke trace identity.
        x_word = Word((word[0],))
        y_word = Word((word[1],))
        z_word = Word((word[2],))
        w_word = Word(word.letters[3:])

        x_expr = self.reduce(x_word)
        y_expr = self.reduce(y_word)
        z_expr = self.reduce(z_word)
        w_expr = self.reduce(w_word)

        yzw_expr = self.reduce(y_word.concatenate(z_word).concatenate(w_word))
        xzw_expr = self.reduce(x_word.concatenate(z_word).concatenate(w_word))
        xyw_expr = self.reduce(x_word.concatenate(y_word).concatenate(w_word))
        xyz_expr = self.reduce(x_word.concatenate(y_word).concatenate(z_word))

        xz_expr = self.reduce(x_word.concatenate(z_word))
        yw_expr = self.reduce(y_word.concatenate(w_word))
        xw_expr = self.reduce(x_word.concatenate(w_word))
        yz_expr = self.reduce(y_word.concatenate(z_word))
        xy_expr = self.reduce(x_word.concatenate(y_word))
        zw_expr = self.reduce(z_word.concatenate(w_word))

        return sp.expand(
            sp.Rational(1, 2)
            * (
                x_expr * y_expr * z_expr * w_expr
                + x_expr * yzw_expr
                + y_expr * xzw_expr
                + z_expr * xyw_expr
                + w_expr * xyz_expr
                - xz_expr * yw_expr
                + xw_expr * yz_expr
                + xy_expr * zw_expr
                - x_expr * y_expr * zw_expr
                - x_expr * w_expr * yz_expr
                - y_expr * z_expr * xw_expr
                - z_expr * w_expr * xy_expr
            )
        )

    def reduce(self, word: Word | str) -> sp.Expr:
        if isinstance(word, str):
            word = Word.from_sequence(word)
        return self.reduce_word(word.reduced().cyclic_normal_form().render())

    def coordinate_basis(self) -> Dict[str, str]:
        return {key: str(value) for key, value in self.coordinate_map.items()}

    def active_trace_variables(self) -> Tuple[sp.Symbol, ...]:
        coordinate_variables = [value for value in self.coordinate_map.values() if isinstance(value, sp.Symbol)]
        promoted = sorted(self.promoted_symbols, key=lambda symbol: str(symbol))

        def _is_projection_target(symbol: sp.Symbol) -> bool:
            return symbol in DEFAULT_PROJECTION_VARIABLES

        extras = [symbol for symbol in coordinate_variables + promoted if not _is_projection_target(symbol)]
        targets = [symbol for symbol in coordinate_variables + promoted if _is_projection_target(symbol)]

        # Lex order: extra variables first, projection coordinates last.
        ordered = extras + [symbol for symbol in DEFAULT_PROJECTION_VARIABLES if symbol in targets]
        deduped: List[sp.Symbol] = []
        seen = set()
        for symbol in ordered:
            if symbol not in seen:
                deduped.append(symbol)
                seen.add(symbol)
        return tuple(deduped)


class GroebnerBackend:
    name = "abstract"

    def compute_basis(self, ideal_polynomials: List[sp.Expr], variables: Sequence[sp.Symbol]) -> List[sp.Expr]:
        raise NotImplementedError

    def compute_elimination_basis(
        self,
        ideal_polynomials: List[sp.Expr],
        elimination_variables: Sequence[sp.Symbol],
        projection_variables: Sequence[sp.Symbol],
    ) -> List[sp.Expr]:
        ordered_variables = tuple(elimination_variables) + tuple(projection_variables)
        return self.compute_basis(ideal_polynomials, ordered_variables)


class SymPyGroebnerBackend(GroebnerBackend):
    name = "sympy"

    def compute_basis(self, ideal_polynomials: List[sp.Expr], variables: Sequence[sp.Symbol]) -> List[sp.Expr]:
        if not ideal_polynomials:
            return []
        if not variables:
            free_symbols = sorted(set().union(*(poly.free_symbols for poly in ideal_polynomials)), key=lambda symbol: str(symbol))
            variables = tuple(free_symbols)
        basis = sp.groebner(ideal_polynomials, *variables, order="lex")
        return [sp.expand(polynomial) for polynomial in basis]


class SingularGroebnerBackend(GroebnerBackend):
    name = "singular-subprocess"

    def __init__(self, fallback_backend: Optional[GroebnerBackend] = None):
        self.fallback_backend = fallback_backend or SymPyGroebnerBackend()

    def _parse_basis_output(self, output: str) -> List[sp.Expr]:
        basis_exprs: List[sp.Expr] = []
        for raw_line in output.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("//"):
                continue
            if line.endswith(","):
                line = line[:-1].strip()
            if not line:
                continue
            expr = sp.sympify(line.replace("^", "**"))
            basis_exprs.append(sp.expand(expr))
        return basis_exprs

    def compute_basis(self, ideal_polynomials: List[sp.Expr], variables: Sequence[sp.Symbol]) -> List[sp.Expr]:
        if not ideal_polynomials:
            return []
        if not variables:
            free_symbols = sorted(set().union(*(poly.free_symbols for poly in ideal_polynomials)), key=lambda symbol: str(symbol))
            variables = tuple(free_symbols)

        var_str = ",".join(str(variable) for variable in variables)
        poly_strs = [str(sp.expand(poly)).replace("**", "^") for poly in ideal_polynomials]
        ideal_str = ",\n  ".join(poly_strs)
        singular_script = (
            f"ring r = 0,({var_str}),lp;\n"
            f"ideal I = {ideal_str};\n"
            "ideal G = std(I);\n"
            "short=0;\n"
            "print(G);\n"
            "quit;\n"
        )

        try:
            result = subprocess.run(
                ["Singular", "-q"],
                input=singular_script,
                capture_output=True,
                text=True,
                check=True,
            )
            basis_exprs = self._parse_basis_output(result.stdout)
            if basis_exprs:
                return basis_exprs
            raise ValueError("Singular returned an empty Gröbner basis output.")
        except WindowComputationTimeout:
            raise
        except Exception:
            return self.fallback_backend.compute_basis(ideal_polynomials, variables)


def compute_trace_polynomials(window: str) -> Dict[str, object]:
    """
    Abstract trace-algebra metadata. This replaces the incorrect matrix model.
    It records the presentation, canonical relator, and named trace coordinates
    that the Lawton/Ashley recursion will eventually eliminate into x/y/z.
    """
    presentation = build_presentation(window)
    trace_algebra = TraceAlgebra(presentation)
    relator_trace = trace_symbol(presentation.relator)
    reduced_relator_trace = trace_algebra.reduce(presentation.relator)

    return {
        "presentation": presentation,
        "generator_order": list(presentation.generators),
        "relator_word": presentation.relator.render(),
        "relator_trace_symbol": str(relator_trace),
        "reduced_relator_trace": str(reduced_relator_trace),
        "trace_coordinate_map": trace_algebra.coordinate_basis(),
        "active_trace_variables": [str(symbol) for symbol in trace_algebra.active_trace_variables()],
        "x": "x" if presentation.generator_count >= 2 else "",
        "y": "y" if presentation.generator_count >= 3 else "",
        "z": "z" if presentation.generator_count >= 3 else "",
    }


def compute_ideal_polynomials(presentation: GroupPresentation, trace_algebra: TraceAlgebra) -> List[sp.Expr]:
    relator = presentation.relator
    ideal_polynomials = [sp.expand(trace_algebra.reduce(relator) - 2)]

    for generator in presentation.generators:
        generator_word = Word((Letter(generator),))
        generator_times_relator = generator_word.concatenate(relator)
        ideal_polynomials.append(sp.expand(trace_algebra.reduce(generator_times_relator) - trace_algebra.reduce(generator_word)))

    cleaned_polynomials: List[sp.Expr] = []
    for polynomial in ideal_polynomials:
        simplified = sp.expand(polynomial)
        if simplified != 0 and simplified not in cleaned_polynomials:
            cleaned_polynomials.append(simplified)
    return cleaned_polynomials


def _normalize_polynomial_family(polynomial: sp.Expr) -> sp.Expr:
    expanded = sp.expand(polynomial)
    coeff, primitive = sp.Poly(expanded, X, Y, Z).primitive()
    normalized = sp.expand(primitive.as_expr())
    if bool(getattr(coeff, "is_number", False)) and float(sp.N(coeff)) < 0:
        normalized = sp.expand(-normalized)
    return normalized


def _normalize_projected_polynomial(polynomial: sp.Expr) -> sp.Expr:
    normalized = _normalize_polynomial_family(polynomial)
    coeff, terms = normalized.as_coeff_mul()
    if coeff != 0 and coeff != 1 and coeff != -1:
        normalized = sp.expand(normalized / coeff)
    if normalized.could_extract_minus_sign():
        normalized = sp.expand(-normalized)
    return sp.expand(normalized)


def _is_projection_surface_factor(polynomial: sp.Expr) -> bool:
    if not polynomial.free_symbols or not polynomial.free_symbols <= set(DEFAULT_PROJECTION_VARIABLES):
        return False
    poly = sp.Poly(polynomial, X, Y, Z)
    return len(polynomial.free_symbols) == 3 and poly.total_degree() >= 2


def _factor_projection_polynomial(polynomial: sp.Expr) -> Dict[str, object]:
    expanded = sp.expand(polynomial)
    coeff, factor_pairs = sp.factor_list(expanded)
    irreducible_factors: List[sp.Expr] = []
    for factor, multiplicity in factor_pairs:
        irreducible_factors.extend([sp.expand(factor)] * multiplicity)
    return {
        "polynomial": expanded,
        "factorization": sp.factor(expanded),
        "coefficient": coeff,
        "factors": irreducible_factors,
    }


def _unique_polynomials(polynomials: Iterable[sp.Expr]) -> List[sp.Expr]:
    unique: List[sp.Expr] = []
    for polynomial in polynomials:
        expanded = sp.expand(polynomial)
        if any(sp.expand(expanded - existing) == 0 for existing in unique):
            continue
        unique.append(expanded)
    return unique


def _polynomial_sort_key(expr: sp.Expr) -> Tuple[int, str]:
    return (-sp.Poly(expr, X, Y, Z).total_degree(), str(expr))


def _extract_component_candidates(projected_polynomials: Iterable[sp.Expr]) -> Dict[str, object]:
    factorizations = [_factor_projection_polynomial(polynomial) for polynomial in projected_polynomials]
    factor_occurrences: Dict[str, Dict[str, object]] = {}

    for factorization in factorizations:
        seen_in_polynomial: List[sp.Expr] = []
        for factor in factorization["factors"]:
            if not _is_projection_surface_factor(factor):
                continue
            expanded_factor = sp.expand(factor)
            if any(sp.expand(expanded_factor - seen) == 0 for seen in seen_in_polynomial):
                continue
            seen_in_polynomial.append(expanded_factor)
            key = str(expanded_factor)
            record = factor_occurrences.setdefault(key, {"factor": expanded_factor, "count": 0})
            record["count"] += 1

    common_factors = [
        record["factor"]
        for record in factor_occurrences.values()
        if int(record["count"]) >= 2
    ]
    common_factors = sorted(
        _unique_polynomials(common_factors),
        key=_polynomial_sort_key,
    )

    candidate_surfaces = common_factors[:]
    if not candidate_surfaces:
        candidate_surfaces = sorted(
            _unique_polynomials(
                factor
                for factorization in factorizations
                for factor in factorization["factors"]
                if _is_projection_surface_factor(factor)
            ),
            key=_polynomial_sort_key,
        )

    normalized_candidates = _unique_polynomials(_normalize_projected_polynomial(factor) for factor in candidate_surfaces)

    return {
        "factorizations": factorizations,
        "common_factors": common_factors,
        "normalized_candidates": normalized_candidates,
    }


def _benchmark_comparison(slice_name: str, candidates: Sequence[sp.Expr]) -> Dict[str, object]:
    target_spec = BENCHMARK_SLICE_TARGETS.get(slice_name)
    if target_spec is None:
        return {"target_label": None, "target_polynomial": None, "matched": False, "matched_candidate": None}

    target = sp.expand(target_spec["target"])
    normalized_target = _normalize_projected_polynomial(target)
    for candidate in candidates:
        normalized_candidate = _normalize_projected_polynomial(candidate)
        if sp.expand(normalized_candidate - normalized_target) == 0:
            return {
                "target_label": target_spec["label"],
                "target_polynomial": str(target),
                "matched": True,
                "matched_candidate": str(candidate),
            }

    return {
        "target_label": target_spec["label"],
        "target_polynomial": str(target),
        "matched": False,
        "matched_candidate": None,
        "raw_target_polynomial": str(target_spec["raw_target"]) if "raw_target" in target_spec else None,
    }


def _canonical_fricke_slice(parameter_slice: Dict[sp.Symbol, int]) -> sp.Expr:
    a, b, c, d = [parameter_slice.get(symbol, symbol) for symbol in BOUNDARY_SYMBOLS]
    theta_1 = a * b + c * d
    theta_2 = a * d + b * c
    theta_3 = a * c + b * d
    theta_4 = 4 - a**2 - b**2 - c**2 - d**2 - a * b * c * d
    return sp.expand(X**2 + Y**2 + Z**2 + X * Y * Z - theta_1 * X - theta_2 * Y - theta_3 * Z - theta_4)


def _compare_candidates_to_target(candidates: Sequence[sp.Expr], target: sp.Expr) -> Dict[str, object]:
    normalized_target = _normalize_projected_polynomial(target)
    for candidate in candidates:
        normalized_candidate = _normalize_projected_polynomial(candidate)
        if sp.expand(normalized_candidate - normalized_target) == 0:
            return {
                "matched": True,
                "matched_candidate": str(candidate),
                "target_polynomial": str(target),
            }
    return {
        "matched": False,
        "matched_candidate": None,
        "target_polynomial": str(target),
    }


def _slice_audit_status(
    slice_name: str,
    canonical_comparison: Dict[str, object],
    draft_comparison: Dict[str, object],
    has_common_component: bool,
) -> str:
    if canonical_comparison["matched"] and draft_comparison["matched"]:
        return "Recovered canonical slice and draft claim"
    if canonical_comparison["matched"]:
        return "Recovered canonical slice"
    if draft_comparison["matched"]:
        return "Recovered draft claim"
    if slice_name == "(1,1,0,0)" and canonical_comparison["target_polynomial"] != draft_comparison.get("target_polynomial"):
        return "Paper claim inconsistent"
    if has_common_component:
        return "Extra undocumented step needed"
    return "Only lower-dimensional components present"


def _xyz_surface_factors(polynomials: Iterable[sp.Expr]) -> List[sp.Expr]:
    extracted: List[sp.Expr] = []
    for polynomial in polynomials:
        expanded = sp.expand(polynomial)
        factors = sp.factor_list(expanded)[1]
        if not factors and _is_projection_surface_factor(expanded):
            extracted.append(expanded)
            continue
        for factor, _ in factors:
            if _is_projection_surface_factor(factor):
                extracted.append(sp.expand(factor))
    return _unique_polynomials(extracted)


def _match_surface_family(polynomials: Iterable[sp.Expr]) -> Tuple[str, List[str]]:
    candidates = _xyz_surface_factors(polynomials)
    normalized_candidates = [_normalize_projected_polynomial(polynomial) for polynomial in candidates]
    for polynomial in normalized_candidates:
        for label, target in SINGULARITY_SURFACES.items():
            target_normalized = _normalize_projected_polynomial(target)
            if sp.expand(polynomial - target_normalized) == 0:
                return label, [str(candidate) for candidate in normalized_candidates]
    return "None", [str(candidate) for candidate in normalized_candidates]


def _jacobian_has_isolated_singularity(polynomial: sp.Expr) -> bool:
    derivatives = [sp.diff(polynomial, variable) for variable in (X, Y, Z)]
    solutions = sp.solve([polynomial, *derivatives], (X, Y, Z), dict=True)
    return bool(solutions)


def _slice_name(parameter_slice: Dict[sp.Symbol, int]) -> str:
    return "(" + ",".join(str(parameter_slice.get(symbol, "*")) for symbol in BOUNDARY_SYMBOLS) + ")"


def _literature_coordinate_map(presentation: GroupPresentation, trace_algebra: TraceAlgebra) -> Dict[str, object]:
    generators = presentation.generators
    coordinate_basis = trace_algebra.coordinate_basis()
    greek_names = ("alpha", "beta", "gamma", "delta")
    literature_trace_generators: Dict[str, str] = {}
    literature_coordinate_symbols: Dict[str, str] = {}

    for index, generator in enumerate(generators[:4]):
        greek = greek_names[index]
        trace_label = f"tr_{generator}"
        literature_trace_generators[greek] = trace_label
        literature_coordinate_symbols[chr(ord("a") + index)] = coordinate_basis.get(trace_label, trace_label)

    pair_specs = (
        ("x", 0, 1),
        ("y", 1, 2),
        ("z", 2, 0),
    )
    projection_trace_generators: Dict[str, str] = {}
    projection_coordinate_symbols: Dict[str, str] = {}
    for coordinate, left_index, right_index in pair_specs:
        if left_index < len(generators) and right_index < len(generators):
            trace_label = f"tr_{Word.from_sequence(generators[left_index] + generators[right_index]).cyclic_normal_form().render() or 'I'}"
            projection_trace_generators[coordinate] = trace_label
            projection_coordinate_symbols[coordinate] = coordinate_basis.get(trace_label, trace_label)

    return {
        "presentation_generators": list(generators),
        "paper_coordinate_meaning": {
            "a": "tr(rho(alpha))",
            "b": "tr(rho(beta))",
            "c": "tr(rho(gamma))",
            "d": "tr(rho(delta))",
            "x": "tr(rho(alpha beta))",
            "y": "tr(rho(beta gamma))",
            "z": "tr(rho(gamma alpha))",
        },
        "literature_trace_generators": literature_trace_generators,
        "literature_coordinate_symbols": literature_coordinate_symbols,
        "projection_trace_generators": projection_trace_generators,
        "projection_coordinate_symbols": projection_coordinate_symbols,
        "current_slice_application": "Boundary slices are applied to symbols a,b,c,d before sliced elimination.",
    }


MAX_ELIMINATION_VARIABLES = 6


def _project_slice_polynomials(
    basis: List[sp.Expr],
    remaining_variables: Sequence[sp.Symbol],
    backend: GroebnerBackend,
    projection_variables: Sequence[sp.Symbol] = DEFAULT_PROJECTION_VARIABLES,
) -> Tuple[List[sp.Expr], List[str], List[str], bool]:
    elimination_variables = [variable for variable in remaining_variables if variable not in projection_variables]
    elimination_order = [str(variable) for variable in elimination_variables]
    groebner_variable_order = elimination_order + [str(variable) for variable in projection_variables if variable in remaining_variables]

    if not elimination_variables:
        projected = [sp.expand(polynomial) for polynomial in basis if polynomial.free_symbols <= set(projection_variables)]
        return projected, elimination_order, groebner_variable_order, False

    if len(elimination_variables) > MAX_ELIMINATION_VARIABLES:
        projected = [sp.expand(polynomial) for polynomial in basis if polynomial.free_symbols <= set(projection_variables)]
        return projected, elimination_order, groebner_variable_order, True

    try:
        elimination_basis = backend.compute_elimination_basis(basis, elimination_variables, list(projection_variables))
        projected = [sp.expand(polynomial) for polynomial in elimination_basis if polynomial.free_symbols <= set(projection_variables)]
    except WindowComputationTimeout:
        raise
    except Exception:
        projected = [sp.expand(polynomial) for polynomial in basis if polynomial.free_symbols <= set(projection_variables)]

    return projected, elimination_order, groebner_variable_order, False


def compute_sliced_bases(
    ideal_polynomials: List[sp.Expr],
    active_trace_variables: Sequence[sp.Symbol],
    backend: GroebnerBackend,
) -> List[Dict[str, object]]:
    sliced_results: List[Dict[str, object]] = []

    for parameter_slice in PARAMETER_SLICES:
        sliced_polynomials = [sp.expand(polynomial.subs(parameter_slice)) for polynomial in ideal_polynomials]
        sliced_polynomials = [polynomial for polynomial in sliced_polynomials if polynomial != 0]
        if not sliced_polynomials:
            continue

        remaining_variables = [
            variable for variable in active_trace_variables if variable not in parameter_slice and any(variable in poly.free_symbols for poly in sliced_polynomials)
        ]

        if len(remaining_variables) > MAX_PRACTICAL_GROEBNER_VARIABLES:
            basis = [sp.expand(poly) for poly in sliced_polynomials]
            slice_mode = "raw-sliced"
        else:
            basis = backend.compute_basis(sliced_polynomials, remaining_variables)
            slice_mode = "groebner-sliced"

        projected_polynomials, elimination_order, groebner_variable_order, projection_skipped = _project_slice_polynomials(basis, remaining_variables, backend)
        slice_name = _slice_name(parameter_slice)
        component_data = _extract_component_candidates(projected_polynomials)
        normalized_projected = [_normalize_projected_polynomial(polynomial) for polynomial in projected_polynomials]
        slice_family, normalized_factors = _match_surface_family(component_data["normalized_candidates"] or projected_polynomials)
        benchmark_comparison = _benchmark_comparison(slice_name, component_data["normalized_candidates"])
        canonical_fricke = _canonical_fricke_slice(parameter_slice)
        canonical_comparison = _compare_candidates_to_target(component_data["normalized_candidates"], canonical_fricke)
        slice_audit_status = _slice_audit_status(
            slice_name,
            canonical_comparison,
            benchmark_comparison,
            bool(component_data["common_factors"]),
        )
        sliced_results.append(
            {
                "slice": slice_name,
                "slice_mode": slice_mode,
                "projection_skipped": projection_skipped,
                "parameter_slice": {str(key): value for key, value in parameter_slice.items()},
                "variables": [str(variable) for variable in remaining_variables],
                "elimination_order": elimination_order,
                "groebner_variable_order": groebner_variable_order,
                "ideal_polynomials": [str(poly) for poly in sliced_polynomials],
                "basis_equations": [str(poly) for poly in basis],
                "projected_polynomials": [str(poly) for poly in projected_polynomials],
                "normalized_projected_polynomials": [str(poly) for poly in normalized_projected],
                "factored_projected_polynomials": [
                    {
                        "polynomial": str(factorization["polynomial"]),
                        "factorization": str(factorization["factorization"]),
                        "factors": [str(factor) for factor in factorization["factors"]],
                    }
                    for factorization in component_data["factorizations"]
                ],
                "common_surface_factors": [str(factor) for factor in component_data["common_factors"]],
                "normalized_core_surface_candidates": [str(factor) for factor in component_data["normalized_candidates"]],
                "normalized_xyz_surface_factors": normalized_factors,
                "slice_family": slice_family,
                "canonical_fricke_polynomial": str(canonical_fricke),
                "canonical_fricke_comparison": canonical_comparison,
                "benchmark_comparison": benchmark_comparison,
                "slice_audit_status": slice_audit_status,
            }
        )

    return sliced_results


def compute_full_ring_basis(
    ideal_polynomials: List[sp.Expr],
    active_trace_variables: Sequence[sp.Symbol],
    backend: GroebnerBackend,
) -> List[sp.Expr]:
    return backend.compute_basis(ideal_polynomials, active_trace_variables)


def _timeout_handler(signum: int, frame: object) -> None:
    raise WindowComputationTimeout


@lru_cache(maxsize=4096)
def compute_character_variety(window: str) -> Dict[str, object]:
    presentation = build_presentation(window)
    trace_algebra = TraceAlgebra(presentation)
    ideal_polynomials = compute_ideal_polynomials(presentation, trace_algebra)
    active_trace_variables = trace_algebra.active_trace_variables()
    trace_data = compute_trace_polynomials(window)
    backend = SingularGroebnerBackend(SymPyGroebnerBackend())
    variable_map = _literature_coordinate_map(presentation, trace_algebra)
    intended_projection_ring = [
        str(symbol)
        for symbol in DEFAULT_PROJECTION_VARIABLES
        if str(symbol) in set(variable_map["projection_coordinate_symbols"].values())
    ]

    if presentation.motif in PUBLISHED_WINDOW_RESULTS:
        regression = PUBLISHED_WINDOW_RESULTS[presentation.motif]
        basis = [sp.sympify(expr) for expr in regression["basis_equations"]]
        return {
            "trace_data": trace_data,
            "ideal_polynomials": [str(poly) for poly in ideal_polynomials],
            "basis_equations": regression["basis_equations"],
            "is_singular": regression["is_singular"],
            "singularity_type": regression["singularity_type"],
            "free_like_class": regression["free_like_class"],
            "analysis_status": regression["analysis_status"],
            "jacobian_singular": any(_jacobian_has_isolated_singularity(poly) for poly in basis) if basis else False,
            "backend": backend.name,
            "active_trace_variables": [str(symbol) for symbol in active_trace_variables],
            "computation_path": "published-regression",
            "sliced_bases": [],
            "variable_map": variable_map,
            "intended_projection_ring": intended_projection_ring,
        }

    sliced_bases = compute_sliced_bases(ideal_polynomials, active_trace_variables, backend)
    projection_skipped = sliced_bases and all(result.get("projection_skipped", False) for result in sliced_bases)

    sliced_basis_exprs = [sp.sympify(polynomial) for result in sliced_bases for polynomial in result["basis_equations"]]
    component_exprs = [
        sp.sympify(polynomial)
        for result in sliced_bases
        for polynomial in result["common_surface_factors"]
    ]

    if projection_skipped:
        return {
            "trace_data": trace_data,
            "ideal_polynomials": [str(poly) for poly in ideal_polynomials],
            "basis_equations": [str(poly) for poly in sliced_basis_exprs],
            "is_singular": None,
            "singularity_type": "unprojected-high-dimensional",
            "free_like_class": f"F_{presentation.rank}",
            "analysis_status": "projection-skipped-due-to-complexity",
            "jacobian_singular": False,
            "backend": backend.name,
            "active_trace_variables": [str(symbol) for symbol in active_trace_variables],
            "active_trace_variable_count": len(active_trace_variables),
            "computation_path": "projection-skipped",
            "sliced_bases": sliced_bases,
            "xyz_surface_factors": [],
            "normalized_xyz_surface_factors": [],
            "variable_map": variable_map,
            "intended_projection_ring": intended_projection_ring,
            "benchmark_status": "projection-skipped",
        }

    singularity_type, normalized_surface_factors = _match_surface_family(component_exprs)
    xyz_surface_factors = [str(poly) for poly in _unique_polynomials(component_exprs)]
    jacobian_singular = False
    if singularity_type != "None":
        jacobian_singular = any(
            _jacobian_has_isolated_singularity(poly)
            for poly in component_exprs
            if poly.free_symbols and poly.free_symbols <= set(DEFAULT_PROJECTION_VARIABLES)
        )

    return {
        "trace_data": trace_data,
        "ideal_polynomials": [str(poly) for poly in ideal_polynomials],
        "basis_equations": [str(poly) for poly in sliced_basis_exprs],
        "is_singular": singularity_type != "None",
        "singularity_type": singularity_type,
        "free_like_class": f"F_{presentation.rank}",
        "analysis_status": "benchmark-sliced",
        "jacobian_singular": jacobian_singular,
        "backend": backend.name,
        "active_trace_variables": [str(symbol) for symbol in active_trace_variables],
        "computation_path": "benchmark-sliced",
        "sliced_bases": sliced_bases,
        "xyz_surface_factors": xyz_surface_factors,
        "normalized_xyz_surface_factors": normalized_surface_factors,
        "variable_map": variable_map,
        "intended_projection_ring": intended_projection_ring,
        "benchmark_status": "legacy-draft-audit" if window in LEGACY_BENCHMARK_WINDOWS else "final-published-benchmark",
        "published_benchmark_recommendations": ["UCCUAC", "UCCUACA", "AAGGCA", "AAGGCAC"],
    }


def compute_character_variety_with_timeout(window: str, timeout_seconds: Optional[int] = None) -> Dict[str, object]:
    if not timeout_seconds:
        return compute_character_variety(window)

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        return compute_character_variety(window)
    except WindowComputationTimeout:
        presentation = build_presentation(window)
        trace_data = compute_trace_polynomials(window)
        return {
            "trace_data": trace_data,
            "ideal_polynomials": [],
            "basis_equations": [],
            "is_singular": None,
            "singularity_type": "Timeout",
            "free_like_class": None,
            "analysis_status": "timeout-preliminary",
            "jacobian_singular": False,
            "backend": "sympy-timeout",
            "active_trace_variables": [],
            "computation_path": "timeout-preliminary",
            "benchmark_status": "preliminary-timeout",
            "published_benchmark_recommendations": ["UCCUAC", "UCCUACA", "AAGGCA", "AAGGCAC"],
        }
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def generate_composition_preserving_controls(window: str, n_controls: int = 1000, seed: Optional[int] = None) -> List[str]:
    window = validate_sequence(window)
    rng = random.Random(seed)
    controls = set()
    bases = list(window)

    counts = Counter(window)
    max_unique = math.factorial(len(window))
    for count in counts.values():
        max_unique //= math.factorial(count)

    target = min(n_controls, max_unique)
    while len(controls) < target:
        shuffled = bases[:]
        rng.shuffle(shuffled)
        controls.add("".join(shuffled))

    return sorted(controls)


def infer_singularity_multiplicity(singularity_type: str) -> Optional[int]:
    if singularity_type == "None":
        return None
    match = re.search(r"(\d+)A1", singularity_type)
    if match:
        return int(match.group(1))
    if re.search(r"\bA1\b", singularity_type):
        return 1
    if "Cayley cubic" in singularity_type:
        return 4
    return None


def window_order_key(result: Dict[str, object]) -> Tuple[int, int, int]:
    return (
        int(result.get("start", 0)),
        int(result.get("length", 0)),
        int(result.get("window_idx", 0)),
    )


def algebraic_signature(result: Dict[str, object]) -> Tuple[object, object, object]:
    return (result["free_like_class"], result["is_singular"], result["singularity_type"])


def _is_known_state(value: object) -> bool:
    return value not in (None, "Timeout", "unprojected-high-dimensional", "projection-skipped-due-to-complexity")


def annotate_window_transitions(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    ordered = sorted(results, key=window_order_key)
    for index, result in enumerate(ordered):
        previous = ordered[index - 1] if index > 0 else None
        following = ordered[index + 1] if index + 1 < len(ordered) else None
        result["ordered_window_rank"] = index
        result["differs_from_previous_window"] = previous is not None and algebraic_signature(previous) != algebraic_signature(result)
        result["differs_from_next_window"] = following is not None and algebraic_signature(following) != algebraic_signature(result)
        result["neighbor_transition"] = bool(result["differs_from_previous_window"] or result["differs_from_next_window"])
        result["previous_window"] = previous["motif"] if previous is not None else None
        result["next_window"] = following["motif"] if following is not None else None
    return ordered


def instability_score(results: List[Dict[str, object]], alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0) -> float:
    free_flips = 0
    singular_flips = 0
    singularity_type_changes = 0

    ordered = sorted(results, key=window_order_key)
    for previous, current in zip(ordered, ordered[1:]):
        if _is_known_state(previous["free_like_class"]) and _is_known_state(current["free_like_class"]) and previous["free_like_class"] != current["free_like_class"]:
            free_flips += 1
        if previous["is_singular"] is not None and current["is_singular"] is not None and previous["is_singular"] != current["is_singular"]:
            singular_flips += 1
        if _is_known_state(previous["singularity_type"]) and _is_known_state(current["singularity_type"]) and previous["singularity_type"] != current["singularity_type"]:
            singularity_type_changes += 1

    return alpha * free_flips + beta * singular_flips + gamma * singularity_type_changes


def instability_breakdown(results: List[Dict[str, object]]) -> Dict[str, int]:
    ordered = sorted(results, key=window_order_key)
    free_flips = 0
    singular_flips = 0
    singularity_type_changes = 0
    for previous, current in zip(ordered, ordered[1:]):
        if _is_known_state(previous["free_like_class"]) and _is_known_state(current["free_like_class"]) and previous["free_like_class"] != current["free_like_class"]:
            free_flips += 1
        if previous["is_singular"] is not None and current["is_singular"] is not None and previous["is_singular"] != current["is_singular"]:
            singular_flips += 1
        if _is_known_state(previous["singularity_type"]) and _is_known_state(current["singularity_type"]) and previous["singularity_type"] != current["singularity_type"]:
            singularity_type_changes += 1
    return {
        "free_nonfree_flips": free_flips,
        "singular_nonsingular_flips": singular_flips,
        "singularity_type_changes": singularity_type_changes,
        "total_transition_count": free_flips + singular_flips + singularity_type_changes,
    }


def summarize_instability(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for result in results:
        grouped.setdefault(str(result["mirna_id"]), []).append(result)

    summaries: List[Dict[str, object]] = []
    for mirna_id, mirna_results in grouped.items():
        breakdown = instability_breakdown(mirna_results)
        singular_windows = [result["motif"] for result in mirna_results if result["is_singular"]]
        unique_types = sorted({str(result["singularity_type"]) for result in mirna_results if result["singularity_type"] != "None"})
        summaries.append(
            {
                "mirna_id": mirna_id,
                "n_windows": len(mirna_results),
                "free_nonfree_flips": breakdown["free_nonfree_flips"],
                "singular_nonsingular_flips": breakdown["singular_nonsingular_flips"],
                "singularity_type_changes": breakdown["singularity_type_changes"],
                "total_transition_count": breakdown["total_transition_count"],
                "provisional_instability_score": instability_score(mirna_results),
                "singular_window_count": len(singular_windows),
                "singular_windows": singular_windows,
                "singularity_types_observed": unique_types,
                "benchmark_status": mirna_results[0].get("benchmark_status", "final-published-benchmark"),
            }
        )

    return sorted(summaries, key=lambda row: (-float(row["provisional_instability_score"]), row["mirna_id"]))


def process_mirna_seed(
    mirna_id: str,
    sequence: str,
    seed_centered_only: bool = False,
    seed_start: int = 1,
    seed_end: int = 8,
    window_timeout_seconds: Optional[int] = None,
) -> List[Dict[str, object]]:
    sequence = validate_sequence(sequence)
    print(f"--- Analyzing {mirna_id} ---")
    windows = (
        generate_seed_centered_windows(sequence, (6, 7, 8), seed_start=seed_start, seed_end=seed_end)
        if seed_centered_only
        else generate_sliding_windows(sequence, (6, 7, 8))
    )
    results: List[Dict[str, object]] = []

    for window_record in windows:
        window = str(window_record["motif"])
        presentation = build_presentation(window)
        algebraic_data = compute_character_variety_with_timeout(window, window_timeout_seconds)
        trace_data = algebraic_data["trace_data"]
        aperiodicity = check_aperiodicity(window)

        profile = {
            **window_record,
            "mirna_id": mirna_id,
            "F_r": f"F_{presentation.rank}",
            "free_group_baseline": f"F_{presentation.rank}",
            "free_like_class": algebraic_data["free_like_class"],
            "free_group_proximity_status": (
                "matches-baseline"
                if algebraic_data["free_like_class"] == f"F_{presentation.rank}"
                else "deviates-from-baseline"
                if algebraic_data["free_like_class"] is not None
                else "unknown-timeout"
            ),
            "unique_bases": len(set(window)),
            "generator_order": ",".join(trace_data["generator_order"]),
            "relator_word": trace_data["relator_word"],
            "relator_trace_symbol": trace_data["relator_trace_symbol"],
            "reduced_relator_trace": trace_data["reduced_relator_trace"],
            "x": trace_data["x"],
            "y": trace_data["y"],
            "z": trace_data["z"],
            "aperiodicity_rule_source": aperiodicity["rule_source"],
            "is_aperiodic": aperiodicity["is_aperiodic"],
            "primitive_substitution": aperiodicity["is_primitive"],
            "pf_eigenvalue": aperiodicity["pf_eigenvalue"],
            "active_trace_variables": algebraic_data["active_trace_variables"],
            "ideal_polynomials": algebraic_data["ideal_polynomials"],
            "basis_equations": algebraic_data["basis_equations"],
            "xyz_surface_factors": algebraic_data.get("xyz_surface_factors", []),
            "is_singular": algebraic_data["is_singular"],
            "singularity_type": algebraic_data["singularity_type"],
            "singularity_multiplicity": infer_singularity_multiplicity(algebraic_data["singularity_type"]),
            "analysis_status": algebraic_data["analysis_status"],
            "jacobian_singular": algebraic_data["jacobian_singular"],
            "backend": algebraic_data["backend"],
            "computation_path": algebraic_data.get("computation_path", algebraic_data["analysis_status"]),
            "benchmark_status": algebraic_data.get("benchmark_status", "final-published-benchmark"),
        }
        results.append(profile)
        print(
            f"[{window_record['length']}-mer] {window} -> {profile['free_like_class']} | "
            f"Singular: {profile['is_singular']} | Type: {profile['singularity_type']} | "
            f"Aperiodic: {profile['is_aperiodic']} | Status: {profile['analysis_status']}"
        )

    annotated_results = annotate_window_transitions(results)
    score = instability_score(annotated_results)
    for result in annotated_results:
        result["instability_score"] = score
    return annotated_results


def process_explicit_windows(
    window_requests: List[Dict[str, object]],
    window_timeout_seconds: Optional[int] = None,
    output_prefix: Optional[str] = None,
    existing_results: Optional[List[Dict[str, object]]] = None,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = list(existing_results or [])
    completed_keys = {
        (str(result.get("mirna_id")), validate_sequence(str(result.get("motif"))))
        for result in results
        if result.get("mirna_id") is not None and result.get("motif") is not None
    }
    for request in window_requests:
        mirna_id = str(request["mirna_id"])
        window = validate_sequence(str(request["motif"]))
        if (mirna_id, window) in completed_keys:
            print(f"[resume] skipping completed window {mirna_id} {window}")
            continue
        presentation = build_presentation(window)
        started = time.time()
        algebraic_data = compute_character_variety_with_timeout(window, window_timeout_seconds)
        trace_data = algebraic_data["trace_data"]
        aperiodicity = check_aperiodicity(window)
        profile = {
            **request,
            "mirna_id": mirna_id,
            "motif": window,
            "F_r": f"F_{presentation.rank}",
            "free_group_baseline": f"F_{presentation.rank}",
            "free_like_class": algebraic_data["free_like_class"],
            "free_group_proximity_status": (
                "matches-baseline"
                if algebraic_data["free_like_class"] == f"F_{presentation.rank}"
                else "deviates-from-baseline"
                if algebraic_data["free_like_class"] is not None
                else "unknown-timeout"
            ),
            "unique_bases": len(set(window)),
            "generator_order": ",".join(trace_data["generator_order"]),
            "relator_word": trace_data["relator_word"],
            "relator_trace_symbol": trace_data["relator_trace_symbol"],
            "reduced_relator_trace": trace_data["reduced_relator_trace"],
            "x": trace_data["x"],
            "y": trace_data["y"],
            "z": trace_data["z"],
            "aperiodicity_rule_source": aperiodicity["rule_source"],
            "is_aperiodic": aperiodicity["is_aperiodic"],
            "primitive_substitution": aperiodicity["is_primitive"],
            "pf_eigenvalue": aperiodicity["pf_eigenvalue"],
            "active_trace_variables": algebraic_data["active_trace_variables"],
            "ideal_polynomials": algebraic_data["ideal_polynomials"],
            "basis_equations": algebraic_data["basis_equations"],
            "xyz_surface_factors": algebraic_data.get("xyz_surface_factors", []),
            "is_singular": algebraic_data["is_singular"],
            "singularity_type": algebraic_data["singularity_type"],
            "singularity_multiplicity": infer_singularity_multiplicity(algebraic_data["singularity_type"]),
            "analysis_status": algebraic_data["analysis_status"],
            "jacobian_singular": algebraic_data["jacobian_singular"],
            "backend": algebraic_data["backend"],
            "computation_path": algebraic_data.get("computation_path", algebraic_data["analysis_status"]),
            "benchmark_status": algebraic_data.get("benchmark_status", "final-published-benchmark"),
            "neighbor_transition": None,
            "differs_from_previous_window": None,
            "differs_from_next_window": None,
            "previous_window": request.get("previous_window"),
            "next_window": request.get("next_window"),
            "instability_score": None,
        }
        results.append(profile)
        print(
            f"[retry {request.get('length', '?')}-mer] {mirna_id} {window} -> {profile['free_like_class']} | "
            f"Singular: {profile['is_singular']} | Type: {profile['singularity_type']} | "
            f"Status: {profile['analysis_status']} | Backend: {profile['backend']} | "
            f"Elapsed: {time.time() - started:.2f}s"
        )
        if output_prefix:
            save_results(results, output_prefix)
        completed_keys.add((mirna_id, window))
    return results


def load_sequences(path: str) -> Dict[str, str]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if input_path.suffix.lower() in {".fa", ".fasta"}:
        records: Dict[str, str] = {}
        current_id: Optional[str] = None
        current_seq: List[str] = []

        for line in input_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    records[current_id] = validate_sequence("".join(current_seq))
                current_id = line[1:].strip()
                current_seq = []
            else:
                current_seq.append(line)

        if current_id is not None:
            records[current_id] = validate_sequence("".join(current_seq))

        return records

    if input_path.suffix.lower() == ".csv":
        frame = pd.read_csv(input_path)
        required = {"mirna_id", "sequence"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"CSV must contain columns: {sorted(required)}")
        return {row["mirna_id"]: validate_sequence(row["sequence"]) for _, row in frame.iterrows()}

    raise ValueError("Supported input formats are FASTA (.fa/.fasta) and CSV (.csv).")


def load_window_requests(path: str) -> List[Dict[str, object]]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Window request file not found: {path}")
    frame = pd.read_csv(input_path)
    required = {"mirna_id", "motif"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Window CSV must contain columns: {sorted(required)}")
    return frame.to_dict(orient="records")


def load_existing_results(path_prefix: str) -> List[Dict[str, object]]:
    csv_path = Path(path_prefix).with_suffix(".csv")
    if not csv_path.exists():
        return []
    frame = pd.read_csv(csv_path)
    return frame.to_dict(orient="records")


def results_to_dataframe(results: List[Dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(results)
    for column in frame.columns:
        frame[column] = frame[column].apply(lambda value: json.dumps(value) if isinstance(value, (list, dict)) else value)
    return frame


def summary_to_markdown(summary_rows: List[Dict[str, object]]) -> str:
    frame = pd.DataFrame(summary_rows)
    if frame.empty:
        return "| mirna_id | provisional_instability_score |\n|---|---|\n"
    columns = [
        "mirna_id",
        "provisional_instability_score",
        "free_nonfree_flips",
        "singular_nonsingular_flips",
        "singularity_type_changes",
        "total_transition_count",
        "singularity_types_observed",
    ]
    frame = frame[columns].copy()
    frame["singularity_types_observed"] = frame["singularity_types_observed"].apply(lambda values: ", ".join(values) if values else "None")
    return frame.to_markdown(index=False)


def save_results(results: List[Dict[str, object]], output_prefix: Optional[str] = None) -> None:
    if not output_prefix:
        return

    frame = results_to_dataframe(results)
    summary_rows = summarize_instability(results)
    summary_frame = results_to_dataframe(summary_rows)
    prefix = Path(output_prefix)
    frame.to_csv(prefix.with_suffix(".csv"), index=False)
    prefix.with_suffix(".json").write_text(json.dumps(results, indent=2))
    summary_frame.to_csv(prefix.with_name(prefix.name + "_summary").with_suffix(".csv"), index=False)
    prefix.with_name(prefix.name + "_summary").with_suffix(".md").write_text(summary_to_markdown(summary_rows))


def run_regression_tests() -> Dict[str, bool]:
    expected = {
        "UCCUAC": (True, "A1 f_b"),
        "UCCUACA": (False, "None"),
        "AAGGCA": (True, "A1 f_(2,{})"),
        "AAGGCAC": (False, "None"),
    }
    results = {}
    for window, (expected_singular, expected_type) in expected.items():
        observed = compute_character_variety(window)
        passed = observed["is_singular"] == expected_singular and observed["singularity_type"] == expected_type
        results[window] = passed
        if not passed:
            raise AssertionError(f"Regression failed for {window}: observed={observed}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Presentation-first sliding-window algebraic profiler for miRNA mature sequences.")
    parser.add_argument("--input", help="FASTA or CSV file with mature miRNA sequences.")
    parser.add_argument("--window-list-csv", help="CSV file listing explicit windows to compute, with at least mirna_id and motif columns.")
    parser.add_argument("--output-prefix", help="Output path prefix for CSV/JSON results.")
    parser.add_argument("--controls-for", help="Generate composition-preserving controls for a specific window.")
    parser.add_argument("--aperiodicity-for", help="Inspect aperiodicity for a specific window.")
    parser.add_argument("--substitution-rule-json", help='JSON object for a user-specified substitution rule, e.g. {"A":"AU","U":"A"}')
    parser.add_argument("--seed-centered-only", action="store_true", help="Restrict 6/7/8-mer windows to those overlapping the canonical seed interval positions 2-8.")
    parser.add_argument("--window-timeout-seconds", type=int, default=20, help="Per-window symbolic computation timeout for pilot-scale runs.")
    parser.add_argument("--n-controls", type=int, default=1000, help="Number of Monte Carlo control permutations.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducible Monte Carlo controls.")
    parser.add_argument("--run-regression-tests", action="store_true", help="Run the four published regression cases and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.run_regression_tests:
        print(json.dumps(run_regression_tests(), indent=2))
        return

    if args.aperiodicity_for:
        rule = json.loads(args.substitution_rule_json) if args.substitution_rule_json else None
        print(json.dumps(check_aperiodicity(args.aperiodicity_for, rule), indent=2))
        return

    if args.controls_for:
        controls = generate_composition_preserving_controls(args.controls_for, n_controls=args.n_controls, seed=args.seed)
        print(json.dumps({"window": validate_sequence(args.controls_for), "n_controls": len(controls), "controls": controls}, indent=2))
        return

    if args.window_list_csv:
        window_requests = load_window_requests(args.window_list_csv)
        existing_results = load_existing_results(args.output_prefix) if args.output_prefix else []
        results = process_explicit_windows(
            window_requests,
            window_timeout_seconds=args.window_timeout_seconds,
            output_prefix=args.output_prefix,
            existing_results=existing_results,
        )
        save_results(results, args.output_prefix)
        return

    if args.input:
        mirna_cohort = load_sequences(args.input)
    else:
        mirna_cohort = {
            "hsa-miR-155-3p": "CUCCUACAU",
            "hsa-miR-124-3p": "UAAGGCAC",
        }

    cohort_results: List[Dict[str, object]] = []
    for mirna, seq in mirna_cohort.items():
        cohort_results.extend(
            process_mirna_seed(
                mirna,
                seq,
                seed_centered_only=args.seed_centered_only,
                window_timeout_seconds=args.window_timeout_seconds,
            )
        )
        print()

    save_results(cohort_results, args.output_prefix)


if __name__ == "__main__":
    main()
