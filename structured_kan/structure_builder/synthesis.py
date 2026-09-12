"""Compress operator-coloured rooted term graphs using algebraic equivalence.

The equivalence relation is deliberately conservative:

* addition and multiplication are commutative;
* identity-linked adjacent nodes with the same operator are associative and
  therefore flattened;
* + and * are never substituted for each other;
* a larger builder may cover a smaller term by neutralising an unmatched
  additive branch to 0 or an unmatched multiplicative branch to 1.

The resulting relation is a rooted, colour-preserving term embedding modulo
associativity and commutativity (AC), with explicit unit slack (ACU envelope).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp
from scipy.sparse import lil_matrix

from .graph_diagnostics import exact_isomorphism_classes, rooted_core_graph
from .terms import Term, is_identity_edge, term_from_spec


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "algebraic_coloured_builder_catalogue"


def load_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load either a record list or a corpus payload containing ``selected``."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload, {"source": path.stem}
    if not isinstance(payload, dict) or not isinstance(payload.get("selected"), list):
        raise ValueError(f"{path} must contain a record list or a 'selected' list")
    provenance = {key: value for key, value in payload.items() if key != "selected"}
    return list(payload["selected"]), provenance


def input_dimension(record: dict[str, Any]) -> int:
    """Return the explicit arity parameter used to instantiate a builder."""
    variables = record.get("operator_variable_spec", {}).get("variables", ())
    dimension = len(variables)
    if dimension < 1:
        raise ValueError(f"{record.get('equation', record.get('index'))} has no input variables")
    return dimension


def builder_call(builder: str) -> str:
    """Render the input-dimension-adaptive template identity."""
    return f"{builder}(input_dimension)"


def dimension_profile(values: Iterable[int]) -> dict[str, Any]:
    observed = [int(value) for value in values]
    if not observed:
        return {
            "equation_count": 0,
            "median": None,
            "mean_absolute_deviation": None,
            "minimum": None,
            "maximum": None,
            "counts": {},
        }
    median = float(np.median(observed))
    mean_absolute_deviation = sum(abs(value - median) for value in observed) / len(observed)
    return {
        "equation_count": len(observed),
        "median": median,
        "mean_absolute_deviation": mean_absolute_deviation,
        "minimum": min(observed),
        "maximum": max(observed),
        "counts": {
            str(value): observed.count(value) for value in sorted(set(observed))
        },
    }


def dimension_dispersion(values: Iterable[int]) -> float:
    observed = [int(value) for value in values]
    profile = dimension_profile(observed)
    if profile["median"] is None:
        return 0.0
    return sum(abs(value - profile["median"]) for value in observed)


def dimension_selection_penalty(dimension: int, profile: dict[str, Any]) -> float:
    """Robust soft distance; broad builders make weaker dimension claims."""
    if profile["median"] is None:
        return 0.0
    scale = 1.0 + float(profile["mean_absolute_deviation"])
    return abs(float(dimension) - float(profile["median"])) / scale


def objective_class_weights(
    class_members: Iterable[list[int]], mode: str
) -> list[float]:
    counts = [len(members) for members in class_members]
    if mode == "frequency":
        return [float(count) for count in counts]
    if mode == "sqrt-frequency":
        return [math.sqrt(count) for count in counts]
    if mode == "log-frequency":
        return [math.log1p(count) for count in counts]
    if mode == "uniform-class":
        return [1.0 for _ in counts]
    raise ValueError(f"unknown class weighting mode: {mode}")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


class ProgressReporter:
    """Append-only machine-readable stage and merge progress."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.sequence = 0
        path.write_text("", encoding="utf-8")

    def emit(self, event: str, **values: Any) -> None:
        self.sequence += 1
        row = {
            "sequence": self.sequence,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "event": event,
            **values,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        print(json.dumps(row, separators=(",", ":")), flush=True)








def merge_root_operator(left: Term, right: Term) -> str | None:
    """Choose the common root colour, treating unary roots as colour-neutral."""
    if left.operator == right.operator:
        return left.operator
    if left.arity == 1:
        return right.operator
    if right.arity == 1:
        return left.operator
    return None


def embedding_root_compatible(target: Term, builder: Term) -> bool:
    """A unary target can use either arithmetic colour in a larger builder."""
    return target.operator == builder.operator or target.arity == 1


@dataclass(frozen=True)
class Slack:
    additive_branches: int = 0
    multiplicative_branches: int = 0
    raw_attachments: int = 0
    nodes: int = 0
    edges: int = 0

    @property
    def weighted(self) -> int:
        return (
            self.additive_branches
            + 2 * self.multiplicative_branches
            + self.raw_attachments
            + self.nodes
            + self.edges
        )

    def __add__(self, other: "Slack") -> "Slack":
        return Slack(
            self.additive_branches + other.additive_branches,
            self.multiplicative_branches + other.multiplicative_branches,
            self.raw_attachments + other.raw_attachments,
            self.nodes + other.nodes,
            self.edges + other.edges,
        )

    def payload(self) -> dict[str, int]:
        return {
            "weighted": self.weighted,
            "additive_branches": self.additive_branches,
            "multiplicative_branches": self.multiplicative_branches,
            "raw_attachments": self.raw_attachments,
            "nodes": self.nodes,
            "edges": self.edges,
        }


def neutral_slack(parent_operator: str, branch: Term) -> Slack:
    additive = int(parent_operator in {"+", "sum", "sum_direct", "sum_phi"})
    multiplicative = int(parent_operator in {"*", "product", "product_direct", "product_phi"})
    return Slack(
        additive_branches=additive,
        multiplicative_branches=multiplicative,
        raw_attachments=sum(node.raw_arity for node in walk_terms(branch)),
        nodes=branch.nodes,
        edges=branch.edges + 1,
    )


def _better(left: Slack | None, right: Slack | None) -> Slack | None:
    if left is None:
        return right
    if right is None:
        return left
    return min(left, right, key=lambda value: (
        value.weighted,
        value.nodes + value.edges,
        value.multiplicative_branches,
        value.additive_branches,
    ))


def encode_lexicographic_vectors(
    vectors: list[tuple[int, ...]], maximum_selected_terms: int
) -> list[int]:
    """Encode bounded integer vectors without losing lexicographic order."""
    if not vectors:
        return []
    width = len(vectors[0])
    maxima = [max(abs(vector[index]) for vector in vectors) for index in range(width)]
    coefficients = [0] * width
    coefficients[-1] = 1
    for index in range(width - 2, -1, -1):
        maximum_lower_swing = 2 * maximum_selected_terms * sum(
            maxima[lower] * coefficients[lower]
            for lower in range(index + 1, width)
        )
        coefficients[index] = maximum_lower_swing + 1
    return [
        sum(value * coefficient for value, coefficient in zip(vector, coefficients))
        for vector in vectors
    ]


def slack_order_vector(value: Slack) -> tuple[int, int, int, int]:
    return (
        value.weighted,
        value.nodes + value.edges,
        value.multiplicative_branches,
        value.additive_branches,
    )


@lru_cache(maxsize=None)
def embedding_child_assignment(
    target: Term, builder: Term
) -> tuple[tuple[int, int], ...] | None:
    """Return the child matching used by the minimum-slack embedding.

    The result exposes the same Hungarian assignment used by
    :func:`embedding_slack`.  It is useful for learning how many copies of an
    otherwise fixed child pattern are actually required by a target.
    """
    if not embedding_root_compatible(target, builder):
        return None
    if target.raw_arity > builder.raw_arity:
        return None
    if len(target.children) > len(builder.children):
        return None
    if not target.children:
        return ()

    neutral = [neutral_slack(builder.operator, child) for child in builder.children]
    child_slacks: dict[tuple[int, int], Slack] = {}
    vectors: list[tuple[int, ...]] = []
    coordinates: list[tuple[int, int]] = []
    for target_index, target_child in enumerate(target.children):
        for builder_index, builder_child in enumerate(builder.children):
            child_slack = embedding_slack(target_child, builder_child)
            if child_slack is None:
                continue
            child_slacks[target_index, builder_index] = child_slack
            child_vector = slack_order_vector(child_slack)
            neutral_vector = slack_order_vector(neutral[builder_index])
            vectors.append(tuple(
                child_value - neutral_value
                for child_value, neutral_value in zip(child_vector, neutral_vector)
            ))
            coordinates.append((target_index, builder_index))
    if not vectors:
        return None
    encoded = encode_lexicographic_vectors(vectors, len(target.children))
    encoded_by_coordinate = dict(zip(coordinates, encoded))
    valid_limit = max(abs(value) for value in encoded) * len(target.children) + 1
    invalid = valid_limit * 4
    cost_matrix = np.full(
        (len(target.children), len(builder.children)), invalid, dtype=np.int64
    )
    for coordinate, value in encoded_by_coordinate.items():
        cost_matrix[coordinate] = value
    target_indices, builder_indices = linear_sum_assignment(cost_matrix)
    selected = tuple(zip(target_indices.tolist(), builder_indices.tolist()))
    if any(coordinate not in child_slacks for coordinate in selected):
        return None
    return selected


@lru_cache(maxsize=None)
def embedding_slack(target: Term, builder: Term) -> Slack | None:
    """Minimum ACU excess for embedding a target, with unary colour identified."""
    if not embedding_root_compatible(target, builder):
        return None
    if target.raw_arity > builder.raw_arity:
        return None
    base = Slack(raw_attachments=builder.raw_arity - target.raw_arity)
    target_children = target.children
    builder_children = builder.children
    if len(target_children) > len(builder_children):
        return None

    if not target_children:
        for child in builder_children:
            base += neutral_slack(builder.operator, child)
        return base

    selected = embedding_child_assignment(target, builder)
    if selected is None:
        return None
    neutral = [neutral_slack(builder.operator, child) for child in builder_children]
    used_builders = {builder_index for _, builder_index in selected}
    for target_index, builder_index in selected:
        child_slack = embedding_slack(
            target_children[target_index], builder_children[builder_index]
        )
        if child_slack is None:
            raise RuntimeError("cached embedding assignment became incompatible")
        base += child_slack
    for builder_index, value in enumerate(neutral):
        if builder_index not in used_builders:
            base += value
    return base


@lru_cache(maxsize=None)
def capacity_units(term: Term) -> int:
    """Arithmetic nodes plus independent raw-role route capacity."""
    return term.nodes + sum(
        node.raw_arity
        for node in walk_terms(term)
    )


def walk_terms(term: Term) -> tuple[Term, ...]:
    output = [term]
    for child in term.children:
        output.extend(walk_terms(child))
    return tuple(output)


@lru_cache(maxsize=None)
def embedding_unused_capacity(target: Term, builder: Term) -> int | None:
    """Minimum recursively aligned unused builder capacity.

    An unused arithmetic node or independent raw-role route costs one, while
    additive and multiplicative neutral branches receive no extra
    operator-specific multiplier.
    """
    if not embedding_root_compatible(target, builder):
        return None
    if target.raw_arity > builder.raw_arity:
        return None
    if len(target.children) > len(builder.children):
        return None
    base = builder.raw_arity - target.raw_arity
    if not target.children:
        return base + sum(capacity_units(child) for child in builder.children)

    baseline = [capacity_units(child) for child in builder.children]
    valid: dict[tuple[int, int], int] = {}
    deltas: list[int] = []
    for target_index, target_child in enumerate(target.children):
        for builder_index, builder_child in enumerate(builder.children):
            child_cost = embedding_unused_capacity(target_child, builder_child)
            if child_cost is None:
                continue
            delta = child_cost - baseline[builder_index]
            valid[target_index, builder_index] = delta
            deltas.append(delta)
    if not valid:
        return None
    bound = max((abs(value) for value in deltas), default=0)
    invalid = (bound + sum(baseline) + 1) * (len(target.children) + 1)
    matrix = np.full(
        (len(target.children), len(builder.children)), invalid, dtype=np.int64
    )
    for coordinate, value in valid.items():
        matrix[coordinate] = value
    target_indices, builder_indices = linear_sum_assignment(matrix)
    selected = tuple(zip(target_indices.tolist(), builder_indices.tolist()))
    if any(coordinate not in valid for coordinate in selected):
        return None
    return base + sum(baseline) + sum(valid[coordinate] for coordinate in selected)


def structural_score_value(
    target: Term,
    builder: Term,
    mode: str,
    unused_normalization_power: float = 0.0,
    builder_capacity_weight: float = 0.0,
) -> float | None:
    if mode == "weighted-slack":
        slack = embedding_slack(target, builder)
        return None if slack is None else float(slack.weighted)
    if mode == "unused-capacity":
        value = embedding_unused_capacity(target, builder)
        if value is None:
            return None
        capacity = float(max(1, capacity_units(builder)))
        return (
            float(value) / capacity ** unused_normalization_power
            + builder_capacity_weight * capacity
        )
    raise ValueError(f"unknown structural score: {mode}")


def exact_classes(terms: Iterable[Term]) -> dict[str, list[int]]:
    classes: dict[str, list[int]] = {}
    for index, term in enumerate(terms):
        classes.setdefault(term.signature(), []).append(index)
    return classes


@lru_cache(maxsize=None)
def merge_envelope(left: Term, right: Term) -> Term | None:
    """Least-slack common coloured superterm for two AC-normalized terms."""
    operator = merge_root_operator(left, right)
    if operator is None:
        return None
    raw_attachment = left.raw_attachment or right.raw_attachment
    raw_arity = max(left.raw_arity, right.raw_arity)
    left_children = left.children
    right_children = right.children

    pair_envelopes: dict[tuple[int, int], Term] = {}
    vectors = []
    coordinates = []
    for left_index, left_child in enumerate(left_children):
        for right_index, right_child in enumerate(right_children):
            merged_child = merge_envelope(left_child, right_child)
            if merged_child is None:
                continue
            left_slack = embedding_slack(left_child, merged_child)
            right_slack = embedding_slack(right_child, merged_child)
            if left_slack is None or right_slack is None:
                continue
            pair_envelopes[left_index, right_index] = merged_child
            unmatched_slack = (
                neutral_slack(operator, left_child)
                + neutral_slack(operator, right_child)
            )
            merged_slack = left_slack + right_slack
            vectors.append((
                merged_slack.weighted - unmatched_slack.weighted,
                (
                    merged_child.nodes + merged_child.edges + 1
                    - (left_child.nodes + left_child.edges + 1)
                    - (right_child.nodes + right_child.edges + 1)
                ),
                (
                    merged_slack.multiplicative_branches
                    - unmatched_slack.multiplicative_branches
                ),
            ))
            coordinates.append((left_index, right_index))

    selected_pairs: list[tuple[int, int]] = []
    if vectors:
        maximum_pairs = min(len(left_children), len(right_children))
        encoded = encode_lexicographic_vectors(vectors, maximum_pairs)
        encoded_by_coordinate = dict(zip(coordinates, encoded))
        size = len(left_children) + len(right_children)
        valid_limit = max(abs(value) for value in encoded) * max(1, maximum_pairs) + 1
        invalid = valid_limit * 4
        cost_matrix = np.zeros((size, size), dtype=np.int64)
        cost_matrix[:len(left_children), :len(right_children)] = invalid
        for coordinate, value in encoded_by_coordinate.items():
            cost_matrix[coordinate] = value
        row_indices, column_indices = linear_sum_assignment(cost_matrix)
        selected_pairs = [
            (row_index, column_index)
            for row_index, column_index in zip(row_indices.tolist(), column_indices.tolist())
            if row_index < len(left_children)
            and column_index < len(right_children)
            and (row_index, column_index) in pair_envelopes
        ]

    used_left = {left_index for left_index, _ in selected_pairs}
    used_right = {right_index for _, right_index in selected_pairs}
    children = [pair_envelopes[pair] for pair in selected_pairs]
    children.extend(
        child for index, child in enumerate(left_children) if index not in used_left
    )
    children.extend(
        child for index, child in enumerate(right_children) if index not in used_right
    )
    children.sort(key=Term.signature)
    return Term(operator, raw_attachment, tuple(children), raw_arity=raw_arity)


@lru_cache(maxsize=None)
def merge_envelope_unused_capacity(left: Term, right: Term) -> Term | None:
    """The same exact common-superterm merge, scored by unused capacity."""
    operator = merge_root_operator(left, right)
    if operator is None:
        return None
    raw_attachment = left.raw_attachment or right.raw_attachment
    raw_arity = max(left.raw_arity, right.raw_arity)
    pair_envelopes: dict[tuple[int, int], Term] = {}
    pair_deltas: dict[tuple[int, int], int] = {}
    for left_index, left_child in enumerate(left.children):
        for right_index, right_child in enumerate(right.children):
            merged_child = merge_envelope_unused_capacity(left_child, right_child)
            if merged_child is None:
                continue
            left_cost = embedding_unused_capacity(left_child, merged_child)
            right_cost = embedding_unused_capacity(right_child, merged_child)
            if left_cost is None or right_cost is None:
                continue
            pair = (left_index, right_index)
            pair_envelopes[pair] = merged_child
            pair_deltas[pair] = (
                left_cost + right_cost
                - capacity_units(left_child) - capacity_units(right_child)
            )

    selected_pairs: list[tuple[int, int]] = []
    if pair_deltas:
        maximum_pairs = min(len(left.children), len(right.children))
        bound = max(abs(value) for value in pair_deltas.values())
        invalid = (bound + 1) * (maximum_pairs + 1) * 4
        size = len(left.children) + len(right.children)
        matrix = np.zeros((size, size), dtype=np.int64)
        matrix[:len(left.children), :len(right.children)] = invalid
        for coordinate, value in pair_deltas.items():
            matrix[coordinate] = value
        row_indices, column_indices = linear_sum_assignment(matrix)
        selected_pairs = [
            (row_index, column_index)
            for row_index, column_index in zip(row_indices.tolist(), column_indices.tolist())
            if row_index < len(left.children)
            and column_index < len(right.children)
            and (row_index, column_index) in pair_envelopes
        ]

    used_left = {left_index for left_index, _ in selected_pairs}
    used_right = {right_index for _, right_index in selected_pairs}
    children = [pair_envelopes[pair] for pair in selected_pairs]
    children.extend(
        child for index, child in enumerate(left.children) if index not in used_left
    )
    children.extend(
        child for index, child in enumerate(right.children) if index not in used_right
    )
    children.sort(key=Term.signature)
    return Term(operator, raw_attachment, tuple(children), raw_arity=raw_arity)


def weighted_cluster_cost(
    members: list[int],
    envelope: Term,
    terms: list[Term],
    weights: list[float],
) -> float:
    total = 0.0
    for member in members:
        slack = embedding_slack(terms[member], envelope)
        if slack is None:
            return 10**12
        total += weights[member] * slack.weighted
    return total


def structural_cluster_cost(
    members: list[int],
    envelope: Term,
    terms: list[Term],
    weights: list[float],
    mode: str,
    unused_normalization_power: float = 0.0,
    builder_capacity_weight: float = 0.0,
) -> float:
    total = 0.0
    for member in members:
        value = structural_score_value(
            terms[member],
            envelope,
            mode,
            unused_normalization_power,
            builder_capacity_weight,
        )
        if value is None:
            return 10**12
        total += weights[member] * value
    return total


def synthesize_catalogue(
    terms: list[Term],
    weights: list[float],
    limit: int,
    progress: ProgressReporter | None = None,
    class_dimensions: list[list[int]] | None = None,
    dimension_weight: float = 0.0,
    consistent_fan_in_objective: bool = False,
    class_validation_groups: list[list[str | None]] | None = None,
    raw_symbol: str = "X_input_dimension",
    structural_score: str = "weighted-slack",
    unused_normalization_power: float = 0.0,
    builder_capacity_weight: float = 0.0,
    max_builder_nodes: int | None = None,
    max_builder_capacity_units: int | None = None,
    allow_partial_when_capped: bool = False,
) -> list[dict[str, Any]]:
    """Greedy ACU agglomeration with structural and soft dimension costs."""
    if max_builder_nodes is not None:
        if max_builder_nodes <= 0:
            raise ValueError("max_builder_nodes must be positive")
        oversized = [term.nodes for term in terms if term.nodes > max_builder_nodes]
        if oversized:
            raise ValueError(
                "input terms exceed max_builder_nodes; filter unsupported target "
                f"classes before construction: {sorted(oversized)}"
            )
    if max_builder_capacity_units is not None:
        if max_builder_capacity_units <= 0:
            raise ValueError("max_builder_capacity_units must be positive")
        oversized = [
            capacity_units(term)
            for term in terms
            if capacity_units(term) > max_builder_capacity_units
        ]
        if oversized:
            raise ValueError(
                "input terms exceed max_builder_capacity_units; filter unsupported "
                f"target classes before construction: {sorted(oversized)}"
            )
    if class_dimensions is None:
        class_dimensions = [[] for _ in terms]
    if len(class_dimensions) != len(terms):
        raise ValueError("class_dimensions must align with terms")
    if class_validation_groups is None:
        class_validation_groups = [
            [None] * len(dimensions) for dimensions in class_dimensions
        ]
    if len(class_validation_groups) != len(terms):
        raise ValueError("class_validation_groups must align with terms")
    clusters = [
        {
            "members": [index],
            "envelope": term,
            "cost": 0.0,
            "dimension_values": list(class_dimensions[index]),
            "dimension_cost": dimension_dispersion(class_dimensions[index]),
        }
        for index, term in enumerate(terms)
    ]
    initial_count = len(clusters)
    while len(clusters) > limit:
        best = None
        for left_index, left in enumerate(clusters):
            for right_index in range(left_index + 1, len(clusters)):
                right = clusters[right_index]
                merge = (
                    merge_envelope_unused_capacity
                    if structural_score == "unused-capacity"
                    else merge_envelope
                )
                envelope = merge(left["envelope"], right["envelope"])
                if envelope is None:
                    continue
                if (
                    max_builder_nodes is not None
                    and envelope.nodes > max_builder_nodes
                ):
                    continue
                if (
                    max_builder_capacity_units is not None
                    and capacity_units(envelope) > max_builder_capacity_units
                ):
                    continue
                members = [*left["members"], *right["members"]]
                cost = (
                    consistent_parametric_cluster_cost(
                        members,
                        envelope,
                        terms,
                        weights,
                        class_dimensions,
                        class_validation_groups,
                    )
                    if consistent_fan_in_objective
                    else structural_cluster_cost(
                        members,
                        envelope,
                        terms,
                        weights,
                        structural_score,
                        unused_normalization_power,
                        builder_capacity_weight,
                    )
                )
                structural_increase = cost - float(left["cost"]) - float(right["cost"])
                dimensions = [*left["dimension_values"], *right["dimension_values"]]
                dimension_cost = dimension_dispersion(dimensions)
                dimension_increase = (
                    dimension_cost
                    - float(left["dimension_cost"])
                    - float(right["dimension_cost"])
                )
                combined_increase = structural_increase + dimension_weight * dimension_increase
                score = (
                    combined_increase,
                    structural_increase,
                    dimension_increase,
                    cost,
                    envelope.nodes + envelope.edges,
                    len(members),
                    envelope.signature(),
                )
                if best is None or score < best[0]:
                    best = (
                        score,
                        left_index,
                        right_index,
                        members,
                        envelope,
                        cost,
                        dimensions,
                        dimension_cost,
                    )
        if best is None:
            if allow_partial_when_capped:
                break
            raise RuntimeError(f"cannot synthesize {limit} root-coloured envelopes")
        (
            score,
            left_index,
            right_index,
            members,
            envelope,
            cost,
            dimensions,
            dimension_cost,
        ) = best
        left_size = len(clusters[left_index]["members"])
        right_size = len(clusters[right_index]["members"])
        clusters[left_index] = {
            "members": members,
            "envelope": envelope,
            "cost": cost,
            "dimension_values": dimensions,
            "dimension_cost": dimension_cost,
        }
        clusters.pop(right_index)
        if progress is not None:
            progress.emit(
                "catalogue_merge_completed",
                merge_number=initial_count - len(clusters),
                clusters_remaining=len(clusters),
                target_builder_count=limit,
                left_class_count=left_size,
                right_class_count=right_size,
                merged_class_count=len(members),
                incremental_objective_structural_slack=score[1],
                merged_objective_structural_slack=cost,
                incremental_weighted_slack=score[1],
                merged_weighted_slack=cost,
                incremental_dimension_dispersion=score[2],
                merged_dimension_dispersion=dimension_cost,
                dimension_weight=dimension_weight,
                structural_objective=(
                    "consistent_parametric_fan_in_slack"
                    if consistent_fan_in_objective
                    else (
                        "exact_envelope_unused_capacity"
                        if structural_score == "unused-capacity"
                        else "maximum_capacity_envelope_slack"
                    )
                ),
                combined_incremental_cost=score[0],
                envelope_arithmetic_nodes=envelope.nodes,
                envelope_arithmetic_edges=envelope.edges,
                envelope_operator_pattern=envelope.notation(raw_symbol),
            )
    return clusters


def attach_dimension_conditioned_envelopes(
    clusters: list[dict[str, Any]],
    representative_terms: list[Term],
    class_dimensions: list[list[int]],
    rewrite_limit: int = 1,
) -> None:
    """Attach a bounded external-derived rewrite family without adding builders."""
    if rewrite_limit < 1:
        raise ValueError("rewrite_limit must be positive")
    for cluster in clusters:
        observed_dimensions = sorted({
            dimension
            for class_index in cluster["members"]
            for dimension in class_dimensions[class_index]
        })
        conditioned: dict[str, Term] = {}
        conditioned_class_counts: dict[str, int] = {}
        conditioned_metadata: dict[str, dict[str, Any]] = {}
        dimension_rewrites: dict[str, list[dict[str, Any]]] = {}
        for dimension in observed_dimensions:
            selected_classes = [
                class_index
                for class_index in cluster["members"]
                if dimension in class_dimensions[class_index]
            ]
            local_terms = [representative_terms[index] for index in selected_classes]
            local_weights = [
                float(class_dimensions[index].count(dimension))
                for index in selected_classes
            ]
            if len(local_terms) == 1:
                super_envelope = local_terms[0]
            else:
                super_envelope = synthesize_catalogue(
                    local_terms, local_weights, 1
                )[0]["envelope"]
            candidate_terms: dict[str, Term] = {}
            candidate_kinds: dict[str, str] = {}

            def add_candidate(term: Term, kind: str) -> None:
                signature = term.signature()
                kind_priority = {
                    "observed_prototype": 0,
                    "pairwise_envelope": 1,
                    "super_envelope": 2,
                }
                if (
                    signature not in candidate_terms
                    or kind_priority[kind] < kind_priority[candidate_kinds[signature]]
                ):
                    candidate_terms[signature] = term
                    candidate_kinds[signature] = kind

            for term in local_terms:
                add_candidate(term, "observed_prototype")
            add_candidate(super_envelope, "super_envelope")
            for left_index, left_term in enumerate(local_terms):
                for right_term in local_terms[left_index + 1:]:
                    pairwise = merge_envelope(left_term, right_term)
                    if pairwise is not None:
                        add_candidate(pairwise, "pairwise_envelope")
            global_envelope = cluster["envelope"]
            current_slacks = []
            for term in local_terms:
                global_slack = embedding_slack(term, global_envelope)
                if global_slack is None:
                    raise RuntimeError("global cluster envelope lost development coverage")
                current_slacks.append(global_slack)
            current_objective = sum(
                weight * slack.weighted
                for weight, slack in zip(local_weights, current_slacks)
            )
            global_objective = current_objective
            selected_rewrites: list[dict[str, Any]] = []
            remaining = set(candidate_terms)
            for rewrite_rank in range(1, rewrite_limit + 1):
                best_candidate = None
                for signature in sorted(remaining):
                    candidate = candidate_terms[signature]
                    trial_slacks = []
                    for term, current_slack in zip(local_terms, current_slacks):
                        candidate_slack = embedding_slack(term, candidate)
                        trial_slacks.append(
                            current_slack
                            if candidate_slack is None
                            else min(
                                current_slack,
                                candidate_slack,
                                key=lambda value: (
                                    value.weighted,
                                    value.nodes + value.edges,
                                    value.multiplicative_branches,
                                ),
                            )
                        )
                    trial_objective = sum(
                        weight * slack.weighted
                        for weight, slack in zip(local_weights, trial_slacks)
                    )
                    score = (
                        trial_objective,
                        candidate.nodes + candidate.edges,
                        candidate.signature(),
                    )
                    if best_candidate is None or score < best_candidate[0]:
                        best_candidate = (
                            score,
                            signature,
                            candidate,
                            trial_slacks,
                        )
                if best_candidate is None or best_candidate[0][0] >= current_objective - 1e-9:
                    break
                score, signature, candidate, current_slacks = best_candidate
                previous_objective = current_objective
                current_objective = score[0]
                remaining.remove(signature)
                selected_rewrites.append({
                    "envelope": candidate,
                    "rewrite_rank": rewrite_rank,
                    "selected_kind": candidate_kinds[signature],
                    "candidate_count": len(candidate_terms),
                    "global_fallback_objective": global_objective,
                    "selected_family_objective": current_objective,
                    "marginal_objective_reduction": (
                        previous_objective - current_objective
                    ),
                })
            if selected_rewrites:
                dimension_key = str(dimension)
                dimension_rewrites[dimension_key] = selected_rewrites
                first = selected_rewrites[0]
                conditioned[dimension_key] = first["envelope"]
                conditioned_class_counts[dimension_key] = len(selected_classes)
                conditioned_metadata[dimension_key] = {
                    key: value for key, value in first.items() if key != "envelope"
                }
        cluster["dimension_envelopes"] = conditioned
        cluster["dimension_envelope_class_counts"] = conditioned_class_counts
        cluster["dimension_envelope_metadata"] = conditioned_metadata
        cluster["dimension_rewrites"] = dimension_rewrites


def _collect_fan_in_observations(
    target: Term,
    builder: Term,
    dimension: int,
    observations: dict[tuple[str, str], dict[str, Any]],
    validation_group: str | None = None,
) -> None:
    """Collect required multiplicities along one fixed-skeleton embedding."""
    selected = embedding_child_assignment(target, builder)
    if selected is None:
        raise RuntimeError("cannot collect fan-in from an incompatible embedding")

    groups: dict[str, list[int]] = {}
    child_by_signature: dict[str, Term] = {}
    for builder_index, child in enumerate(builder.children):
        signature = child.signature()
        groups.setdefault(signature, []).append(builder_index)
        child_by_signature[signature] = child
    used_builder_indices = {builder_index for _, builder_index in selected}
    parent_signature = builder.signature()
    for child_signature, indices in groups.items():
        # A multiplicity-one group is an explicit optional slot (count 0/1),
        # while a larger group is a repeated fan-in slot. Both obey the same
        # rule family and neither can substitute a different child pattern.
        key = (parent_signature, child_signature)
        entry = observations.setdefault(key, {
            "parent_signature": parent_signature,
            "child_signature": child_signature,
            "parent_operator_pattern": builder.notation("X_input_dimension"),
            "child_operator_pattern": child_by_signature[child_signature].notation(
                "X_input_dimension"
            ),
            "maximum_count": len(indices),
            "observations": [],
        })
        entry["observations"].append({
            "input_dimension": int(dimension),
            "required_count": sum(index in used_builder_indices for index in indices),
            "validation_group": validation_group,
        })

    for target_index, builder_index in selected:
        _collect_fan_in_observations(
            target.children[target_index],
            builder.children[builder_index],
            dimension,
            observations,
            validation_group,
        )


def fan_in_rule_count(rule: dict[str, Any], dimension: int) -> int:
    """Instantiate one clipped fan-in rule at an equation's input dimension."""
    maximum = int(rule["maximum_count"])
    minimum = int(rule.get("minimum_count", 0))
    if rule["mode"] == "constant":
        count = int(rule["constant_count"])
    elif rule["mode"] == "monotone_affine":
        count = math.ceil(
            float(rule["slope"]) * int(dimension)
            + float(rule["intercept"])
            - 1e-12
        )
    else:
        raise ValueError(f"unknown fan-in rule mode: {rule['mode']}")
    return min(maximum, max(minimum, count))


def _fit_consistent_fan_in_rule(
    entry: dict[str, Any], validate_groups: bool = True
) -> dict[str, Any]:
    """Fit a capacity-covering constant or monotone affine count rule."""
    raw_observations = list(entry["observations"])
    if not raw_observations:
        raise ValueError("fan-in rule requires at least one observation")
    maximum = int(entry["maximum_count"])
    minimum = min(int(row["required_count"]) for row in raw_observations)
    constant = max(int(row["required_count"]) for row in raw_observations)
    constant_objective = constant * len(raw_observations)

    best: tuple[int, float, float, list[int]] | None = None
    # Quarter-integer slopes provide a compact, interpretable rule family.
    # Non-negative slopes encode the structural prior that more available
    # inputs may require more fan-in, but never less.
    for numerator in range(1, 4 * maximum + 1):
        slope = numerator / 4.0
        intercept = max(
            int(row["required_count"])
            - slope * int(row["input_dimension"])
            for row in raw_observations
        )
        trial = {
            "mode": "monotone_affine",
            "maximum_count": maximum,
            "minimum_count": minimum,
            "slope": slope,
            "intercept": intercept,
        }
        predictions = [
            fan_in_rule_count(trial, int(row["input_dimension"]))
            for row in raw_observations
        ]
        if any(
            prediction < int(row["required_count"])
            for prediction, row in zip(predictions, raw_observations)
        ):
            continue
        score = (sum(predictions), slope, abs(intercept), predictions)
        if best is None or score[:3] < best[:3]:
            best = score

    counts_by_dimension: dict[str, int] = {}
    observation_counts: dict[str, int] = {}
    for row in raw_observations:
        key = str(int(row["input_dimension"]))
        counts_by_dimension[key] = max(
            counts_by_dimension.get(key, 0), int(row["required_count"])
        )
        observation_counts[key] = observation_counts.get(key, 0) + 1

    rule = {
        key: value for key, value in entry.items() if key != "observations"
    }
    if best is not None and best[0] < constant_objective:
        selected_objective, slope, _, _ = best
        intercept = max(
            int(row["required_count"])
            - slope * int(row["input_dimension"])
            for row in raw_observations
        )
        rule.update({
            "mode": "monotone_affine",
            "minimum_count": minimum,
            "slope": slope,
            "intercept": intercept,
            "dimension_informative": True,
        })
    else:
        selected_objective = constant_objective
        rule.update({
            "mode": "constant",
            "minimum_count": constant,
            "constant_count": constant,
            "dimension_informative": False,
        })
    validation_labels = sorted({
        str(row["validation_group"])
        for row in raw_observations
        if row.get("validation_group") is not None
    })
    validation_folds: list[dict[str, Any]] = []
    validation_passed = True
    validation_reason = (
        "not_requested"
        if not validation_labels
        else "rule_is_constant"
    )
    if validate_groups and rule["dimension_informative"] and validation_labels:
        validation_reason = "leave_one_source_out_passed"
        if len(validation_labels) < 2:
            validation_passed = False
            validation_reason = "fewer_than_two_external_sources"
        else:
            for label in validation_labels:
                training = [
                    row for row in raw_observations
                    if str(row.get("validation_group")) != label
                ]
                held_out = [
                    row for row in raw_observations
                    if str(row.get("validation_group")) == label
                ]
                if not training or not held_out:
                    fold_passed = False
                    fold_rule = None
                else:
                    fold_entry = {**entry, "observations": training}
                    fold_rule = _fit_consistent_fan_in_rule(
                        fold_entry, validate_groups=False
                    )
                    fold_passed = all(
                        fan_in_rule_count(
                            fold_rule, int(row["input_dimension"])
                        ) >= int(row["required_count"])
                        for row in held_out
                    )
                validation_folds.append({
                    "held_out_source": label,
                    "held_out_observations": len(held_out),
                    "passed_capacity_coverage": fold_passed,
                    "training_rule_mode": (
                        fold_rule["mode"] if fold_rule is not None else None
                    ),
                })
                validation_passed = validation_passed and fold_passed
            if not validation_passed:
                validation_reason = "leave_one_source_out_capacity_failure"
        if not validation_passed:
            rule.pop("slope", None)
            rule.pop("intercept", None)
            rule.update({
                "mode": "constant",
                "minimum_count": constant,
                "constant_count": constant,
                "dimension_informative": False,
            })
            selected_objective = constant_objective
    rule.update({
        "training_required_count_by_input_dimension": counts_by_dimension,
        "training_observation_count_by_input_dimension": observation_counts,
        "training_observations": len(raw_observations),
        "full_capacity_objective": maximum * len(raw_observations),
        "constant_objective": constant_objective,
        "selected_objective": selected_objective,
        "external_source_validation_passed": validation_passed,
        "external_source_validation_reason": validation_reason,
        "external_source_validation_folds": validation_folds,
        "predicted_count_by_observed_input_dimension": {
            dimension: fan_in_rule_count(rule, int(dimension))
            for dimension in sorted(counts_by_dimension, key=int)
        },
    })
    return rule


def instantiate_consistent_fan_in(cluster: dict[str, Any], dimension: int) -> Term:
    """Instantiate a builder without changing its coloured nested skeleton."""
    skeleton: Term = cluster["envelope"]
    rules = {
        (rule["parent_signature"], rule["child_signature"]): rule
        for rule in cluster.get("consistent_fan_in_rules", ())
    }

    def instantiate(node: Term) -> Term:
        groups: dict[str, list[Term]] = {}
        for child in node.children:
            groups.setdefault(child.signature(), []).append(child)
        children: list[Term] = []
        for child_signature in sorted(groups):
            originals = groups[child_signature]
            rule = rules.get((node.signature(), child_signature))
            retained = (
                len(originals)
                if rule is None
                else fan_in_rule_count(rule, dimension)
            )
            children.extend(instantiate(child) for child in originals[:retained])
        children.sort(key=Term.signature)
        return Term(
            node.operator,
            node.raw_attachment,
            tuple(children),
            raw_arity=node.raw_arity,
        )

    return instantiate(skeleton)


def learn_consistent_fan_in_rules(
    cluster: dict[str, Any],
    representative_terms: list[Term],
    class_dimensions: list[list[int]],
    class_validation_groups: list[list[str | None]] | None = None,
) -> list[dict[str, Any]]:
    """Learn all fixed child-pattern multiplicity rules for one cluster."""
    observations: dict[tuple[str, str], dict[str, Any]] = {}
    if class_validation_groups is None:
        class_validation_groups = [
            [None] * len(dimensions) for dimensions in class_dimensions
        ]
    for class_index in cluster["members"]:
        target = representative_terms[class_index]
        dimensions = class_dimensions[class_index]
        validation_groups = class_validation_groups[class_index]
        if len(dimensions) != len(validation_groups):
            raise ValueError("validation groups must align with class dimensions")
        for dimension, validation_group in zip(dimensions, validation_groups):
            _collect_fan_in_observations(
                target,
                cluster["envelope"],
                dimension,
                observations,
                validation_group,
            )
    return [
        _fit_consistent_fan_in_rule(entry)
        for _, entry in sorted(observations.items())
    ]


def consistent_parametric_cluster_cost(
    members: list[int],
    envelope: Term,
    terms: list[Term],
    weights: list[float],
    class_dimensions: list[list[int]],
    class_validation_groups: list[list[str | None]] | None = None,
) -> float:
    """Development slack after instantiating one coherent count-rule family."""
    cluster = {"members": members, "envelope": envelope}
    cluster["consistent_fan_in_rules"] = learn_consistent_fan_in_rules(
        cluster, terms, class_dimensions, class_validation_groups
    )
    instantiated_by_dimension = {
        dimension: instantiate_consistent_fan_in(cluster, dimension)
        for dimension in {
            dimension
            for member in members
            for dimension in class_dimensions[member]
        }
    }
    total = 0.0
    for member in members:
        dimensions = class_dimensions[member]
        if not dimensions:
            slack = embedding_slack(terms[member], envelope)
            if slack is None:
                return 10**12
            total += weights[member] * slack.weighted
            continue
        per_equation_weight = weights[member] / len(dimensions)
        for dimension in dimensions:
            slack = embedding_slack(
                terms[member], instantiated_by_dimension[dimension]
            )
            if slack is None:
                return 10**12
            total += per_equation_weight * slack.weighted
    return total


def attach_consistent_fan_in_rules(
    clusters: list[dict[str, Any]],
    representative_terms: list[Term],
    class_dimensions: list[list[int]],
    class_validation_groups: list[list[str | None]] | None = None,
    progress: ProgressReporter | None = None,
) -> None:
    """Learn one coherent parametric fan-in family for every fixed builder."""
    for builder_index, cluster in enumerate(clusters, start=1):
        rules = learn_consistent_fan_in_rules(
            cluster,
            representative_terms,
            class_dimensions,
            class_validation_groups,
        )
        cluster["consistent_fan_in_rules"] = rules

        for rule_index, rule in enumerate(rules, start=1):
            if progress is not None:
                progress.emit(
                    "consistent_fan_in_rule_fitted",
                    builder=f"ACB{builder_index:02d}",
                    rule_number=rule_index,
                    parent_operator_pattern=rule["parent_operator_pattern"],
                    child_operator_pattern=rule["child_operator_pattern"],
                    maximum_count=rule["maximum_count"],
                    mode=rule["mode"],
                    minimum_count=rule["minimum_count"],
                    slope=rule.get("slope"),
                    intercept=rule.get("intercept"),
                    constant_count=rule.get("constant_count"),
                    training_required_count_by_input_dimension=rule[
                        "training_required_count_by_input_dimension"
                    ],
                    predicted_count_by_observed_input_dimension=rule[
                        "predicted_count_by_observed_input_dimension"
                    ],
                    full_capacity_objective=rule["full_capacity_objective"],
                    selected_objective=rule["selected_objective"],
                    external_source_validation_passed=rule[
                        "external_source_validation_passed"
                    ],
                    external_source_validation_reason=rule[
                        "external_source_validation_reason"
                    ],
                    external_source_validation_folds=rule[
                        "external_source_validation_folds"
                    ],
                )

        baseline_objective = 0
        parametric_objective = 0
        for class_index in cluster["members"]:
            target = representative_terms[class_index]
            baseline = embedding_slack(target, cluster["envelope"])
            if baseline is None:
                raise RuntimeError("fixed skeleton lost a development member")
            for dimension in class_dimensions[class_index]:
                instantiated = instantiate_consistent_fan_in(cluster, dimension)
                parametric = embedding_slack(target, instantiated)
                if parametric is None:
                    raise RuntimeError(
                        "consistent fan-in rule lost development coverage"
                    )
                baseline_objective += baseline.weighted
                parametric_objective += parametric.weighted
        if progress is not None:
            progress.emit(
                "consistent_builder_parameterization_completed",
                builder=f"ACB{builder_index:02d}",
                fixed_skeleton_operator_pattern=cluster["envelope"].notation(
                    "X_input_dimension"
                ),
                fan_in_groups=len(rules),
                dimension_informative_groups=sum(
                    rule["dimension_informative"] for rule in rules
                ),
                constant_groups=sum(
                    not rule["dimension_informative"] for rule in rules
                ),
                baseline_development_weighted_slack=baseline_objective,
                parametric_development_weighted_slack=parametric_objective,
                development_weighted_slack_reduction=(
                    baseline_objective - parametric_objective
                ),
            )


def refine_catalogue_by_external_swaps(
    clusters: list[dict[str, Any]],
    representative_terms: list[Term],
    class_weights: list[float],
    class_dimensions: list[list[int]],
    maximum_swaps: int,
    progress: ProgressReporter | None = None,
    structural_score: str = "weighted-slack",
    unused_normalization_power: float = 0.0,
    builder_capacity_weight: float = 0.0,
    max_builder_nodes: int | None = None,
    max_builder_capacity_units: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Improve external structural slack with coverage-preserving one-for-one swaps."""
    candidates_by_signature: dict[str, Term] = {}
    for term in representative_terms:
        candidates_by_signature[term.signature()] = term
    for cluster in clusters:
        candidates_by_signature[cluster["envelope"].signature()] = cluster["envelope"]
        for envelope in cluster.get("dimension_envelopes", {}).values():
            candidates_by_signature[envelope.signature()] = envelope
    candidates = [
        candidates_by_signature[signature]
        for signature in sorted(candidates_by_signature)
        if (
            (max_builder_nodes is None
             or candidates_by_signature[signature].nodes <= max_builder_nodes)
            and (max_builder_capacity_units is None
                 or capacity_units(candidates_by_signature[signature])
                 <= max_builder_capacity_units)
        )
    ]
    candidate_index = {
        term.signature(): index for index, term in enumerate(candidates)
    }
    selected = [candidate_index[cluster["envelope"].signature()] for cluster in clusters]

    compatibility: dict[tuple[int, int], tuple[Slack, float]] = {}
    for class_index, target in enumerate(representative_terms):
        for candidate, builder in enumerate(candidates):
            slack = embedding_slack(target, builder)
            if slack is not None:
                score = structural_score_value(
                    target,
                    builder,
                    structural_score,
                    unused_normalization_power,
                    builder_capacity_weight,
                )
                if score is None:
                    raise RuntimeError("structural score disagrees with ACU embedding")
                compatibility[class_index, candidate] = (slack, score)

    def selection_score(selection: list[int]) -> tuple[float, list[int]]:
        assignments = []
        total = 0.0
        for class_index in range(len(representative_terms)):
            available = [
                (
                    compatibility[class_index, candidate][1],
                    (
                        compatibility[class_index, candidate][0].nodes
                        + compatibility[class_index, candidate][0].edges
                        if structural_score == "weighted-slack"
                        else compatibility[class_index, candidate][0].weighted
                    ),
                    (
                        compatibility[class_index, candidate][0].multiplicative_branches
                        if structural_score == "weighted-slack"
                        else compatibility[class_index, candidate][0].nodes
                        + compatibility[class_index, candidate][0].edges
                    ),
                    position,
                )
                for position, candidate in enumerate(selection)
                if (class_index, candidate) in compatibility
            ]
            if not available:
                return float("inf"), []
            best = min(available)
            assignments.append(best[3])
            total += class_weights[class_index] * best[0]
        return total, assignments

    initial_score, assignments = selection_score(selected)
    current_score = initial_score
    swaps = []
    for swap_number in range(1, maximum_swaps + 1):
        selected_set = set(selected)
        best_swap = None
        for remove_position, removed_candidate in enumerate(selected):
            for added_candidate in range(len(candidates)):
                if added_candidate in selected_set:
                    continue
                trial = list(selected)
                trial[remove_position] = added_candidate
                trial_score, trial_assignments = selection_score(trial)
                if not np.isfinite(trial_score) or trial_score >= current_score - 1e-9:
                    continue
                tie_break = (
                    trial_score,
                    candidates[added_candidate].nodes + candidates[added_candidate].edges,
                    candidates[added_candidate].signature(),
                    remove_position,
                )
                if best_swap is None or tie_break < best_swap[0]:
                    best_swap = (
                        tie_break,
                        remove_position,
                        removed_candidate,
                        added_candidate,
                        trial_assignments,
                    )
        if best_swap is None:
            break
        (
            tie_break,
            remove_position,
            removed_candidate,
            added_candidate,
            assignments,
        ) = best_swap
        previous_score = current_score
        current_score = tie_break[0]
        selected[remove_position] = added_candidate
        swap = {
            "swap_number": swap_number,
            "removed_operator_pattern": candidates[removed_candidate].notation(
                "X_input_dimension"
            ),
            "added_operator_pattern": candidates[added_candidate].notation(
                "X_input_dimension"
            ),
            "previous_external_objective": previous_score,
            "new_external_objective": current_score,
            "objective_reduction": previous_score - current_score,
        }
        swaps.append(swap)
        if progress is not None:
            progress.emit("external_swap_refinement_completed", **swap)

    final_score, assignments = selection_score(selected)
    refined = []
    for position, candidate in enumerate(selected):
        members = [
            class_index
            for class_index, assigned_position in enumerate(assignments)
            if assigned_position == position
        ]
        envelope = candidates[candidate]
        refined.append({
            "members": members,
            "envelope": envelope,
            "cost": sum(
                class_weights[class_index]
                * compatibility[class_index, candidate][1]
                for class_index in members
            ),
            "dimension_values": [
                dimension
                for class_index in members
                for dimension in class_dimensions[class_index]
            ],
            "dimension_cost": dimension_dispersion(
                dimension
                for class_index in members
                for dimension in class_dimensions[class_index]
            ),
        })
    return refined, {
        "candidate_envelopes": len(candidates),
        "maximum_builder_nodes": max_builder_nodes,
        "maximum_builder_capacity_units": max_builder_capacity_units,
        "maximum_swaps": maximum_swaps,
        "completed_swaps": len(swaps),
        "initial_external_objective": initial_score,
        "final_external_objective": final_score,
        "objective_reduction": initial_score - final_score,
        "swaps": swaps,
    }


def refine_catalogue_by_candidate_milp(
    clusters: list[dict[str, Any]],
    representative_terms: list[Term],
    class_weights: list[float],
    class_dimensions: list[list[int]],
    limit: int,
    time_limit_seconds: float,
    progress: ProgressReporter | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Globally select builders from source-only synthesized candidates.

    The pool contains every observed term, every current greedy envelope, and
    least-general ACU merges of observed/greedy pairs.  No evaluation equation
    is loaded or inspected.  The MILP selects exactly ``limit`` used builders
    and assigns every development class to one compatible builder.
    """

    candidates_by_signature: dict[str, Term] = {}

    def add(term: Term | None) -> None:
        if term is not None:
            candidates_by_signature.setdefault(term.signature(), term)

    current = [cluster["envelope"] for cluster in clusters]
    for term in (*representative_terms, *current):
        add(term)
    for left_index, left in enumerate(representative_terms):
        for right in representative_terms[left_index + 1:]:
            add(merge_envelope(left, right))
    for envelope in current:
        for term in representative_terms:
            add(merge_envelope(envelope, term))
    for left_index, left in enumerate(current):
        for right in current[left_index + 1:]:
            add(merge_envelope(left, right))

    candidates = [
        candidates_by_signature[signature]
        for signature in sorted(candidates_by_signature)
    ]
    compatibility: dict[tuple[int, int], Slack] = {}
    used_candidates: set[int] = set()
    for class_index, target in enumerate(representative_terms):
        for candidate_index, builder in enumerate(candidates):
            slack = embedding_slack(target, builder)
            if slack is not None:
                compatibility[class_index, candidate_index] = slack
                used_candidates.add(candidate_index)
    candidates = [
        candidate for index, candidate in enumerate(candidates)
        if index in used_candidates
    ]
    compatibility = {}
    for class_index, target in enumerate(representative_terms):
        for candidate_index, builder in enumerate(candidates):
            slack = embedding_slack(target, builder)
            if slack is not None:
                compatibility[class_index, candidate_index] = slack

    target_count = len(representative_terms)
    candidate_count = len(candidates)
    pairs = sorted(compatibility)
    x_offset = 0
    y_offset = candidate_count
    pair_position = {
        pair: y_offset + position for position, pair in enumerate(pairs)
    }
    variable_count = candidate_count + len(pairs)
    objective = np.zeros(variable_count)
    for candidate_index, candidate in enumerate(candidates):
        objective[x_offset + candidate_index] = 1e-6 * (
            candidate.nodes + candidate.edges
        )
    for (class_index, candidate_index), position in pair_position.items():
        slack = compatibility[class_index, candidate_index]
        objective[position] = class_weights[class_index] * slack.weighted

    # One assignment per class, assignment implies selection, every selected
    # builder is used, and exactly the requested catalogue size is selected.
    row_count = target_count + len(pairs) + candidate_count + 1
    matrix = lil_matrix((row_count, variable_count), dtype=float)
    lower = np.full(row_count, -np.inf)
    upper = np.full(row_count, np.inf)
    row_index = 0
    pairs_by_target: dict[int, list[tuple[int, int]]] = defaultdict(list)
    pairs_by_candidate: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for pair, position in pair_position.items():
        pairs_by_target[pair[0]].append((pair[1], position))
        pairs_by_candidate[pair[1]].append((pair[0], position))
    for class_index in range(target_count):
        for _, position in pairs_by_target[class_index]:
            matrix[row_index, position] = 1.0
        lower[row_index] = upper[row_index] = 1.0
        row_index += 1
    for (class_index, candidate_index), position in pair_position.items():
        matrix[row_index, position] = 1.0
        matrix[row_index, x_offset + candidate_index] = -1.0
        upper[row_index] = 0.0
        row_index += 1
    for candidate_index in range(candidate_count):
        matrix[row_index, x_offset + candidate_index] = 1.0
        for _, position in pairs_by_candidate[candidate_index]:
            matrix[row_index, position] = -1.0
        upper[row_index] = 0.0
        row_index += 1
    matrix[row_index, :candidate_count] = 1.0
    lower[row_index] = upper[row_index] = float(limit)

    result = milp(
        c=objective,
        integrality=np.ones(variable_count),
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"time_limit": time_limit_seconds},
    )
    if result.x is None:
        raise RuntimeError(f"candidate-envelope MILP found no incumbent: {result.message}")
    selected = [
        candidate_index for candidate_index in range(candidate_count)
        if result.x[x_offset + candidate_index] > 0.5
    ]
    if len(selected) != limit:
        raise RuntimeError(f"candidate-envelope MILP selected {len(selected)} builders")
    assignments: list[int] = []
    for class_index in range(target_count):
        available = [
            (result.x[position], candidate_index)
            for candidate_index, position in pairs_by_target[class_index]
            if candidate_index in selected
        ]
        assignments.append(max(available)[1])

    refined = []
    for candidate_index in selected:
        members = [
            class_index for class_index, assigned in enumerate(assignments)
            if assigned == candidate_index
        ]
        envelope = candidates[candidate_index]
        refined.append({
            "members": members,
            "envelope": envelope,
            "cost": sum(
                class_weights[class_index]
                * compatibility[class_index, candidate_index].weighted
                for class_index in members
            ),
            "dimension_values": [
                dimension
                for class_index in members
                for dimension in class_dimensions[class_index]
            ],
            "dimension_cost": dimension_dispersion(
                dimension
                for class_index in members
                for dimension in class_dimensions[class_index]
            ),
        })
    summary = {
        "candidate_envelopes": candidate_count,
        "selected_builders": len(refined),
        "objective": float(result.fun),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "optimality_certified": bool(result.success),
        "mip_gap": (
            float(result.mip_gap)
            if getattr(result, "mip_gap", None) is not None else None
        ),
        "time_limit_seconds": time_limit_seconds,
    }
    if progress is not None:
        progress.emit("candidate_milp_refinement_completed", **summary)
    return refined, summary


def minimum_observed_cover(
    terms: list[Term], time_limit_seconds: float
) -> dict[str, Any]:
    """Attempt an exact observed-envelope cover, retaining bounds on timeout."""
    if time_limit_seconds <= 0:
        return {
            "exact_minimum": None,
            "best_incumbent": None,
            "lower_bound": None,
            "optimality_certified": False,
            "solver_status": "skipped",
            "solver_message": "skipped because the configured time limit is non-positive",
            "mip_gap": None,
            "mip_node_count": None,
            "time_limit_seconds": time_limit_seconds,
        }
    count = len(terms)
    matrix = lil_matrix((count, count), dtype=float)
    for target_index, target in enumerate(terms):
        for builder_index, builder in enumerate(terms):
            if embedding_slack(target, builder) is not None:
                matrix[target_index, builder_index] = 1.0
    result = milp(
        c=np.ones(count),
        integrality=np.ones(count),
        bounds=Bounds(np.zeros(count), np.ones(count)),
        constraints=LinearConstraint(
            matrix.tocsr(), np.ones(count), np.full(count, np.inf)
        ),
        options={"time_limit": time_limit_seconds},
    )
    incumbent = (
        int(round(float(result.x.sum()))) if result.x is not None else None
    )
    raw_dual_bound = getattr(result, "mip_dual_bound", None)
    lower_bound = (
        int(np.ceil(float(raw_dual_bound) - 1e-9))
        if raw_dual_bound is not None and np.isfinite(raw_dual_bound)
        else None
    )
    return {
        "exact_minimum": incumbent if result.success else None,
        "best_incumbent": incumbent,
        "lower_bound": lower_bound,
        "optimality_certified": bool(result.success),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "mip_gap": (
            float(result.mip_gap)
            if getattr(result, "mip_gap", None) is not None
            else None
        ),
        "mip_node_count": (
            int(result.mip_node_count)
            if getattr(result, "mip_node_count", None) is not None
            else None
        ),
        "time_limit_seconds": time_limit_seconds,
    }


def solve_catalogue(terms: list[Term], limit: int) -> tuple[list[int], list[int], list[Slack]]:
    """Select at most limit observed coloured envelopes and assign every term."""
    count = len(terms)
    compatible: dict[tuple[int, int], Slack] = {}
    for target_index, target in enumerate(terms):
        for builder_index, builder in enumerate(terms):
            slack = embedding_slack(target, builder)
            if slack is not None:
                compatible[(target_index, builder_index)] = slack

    # x_j chooses builder j; y_ij assigns target i to builder j.
    x_offset = 0
    y_offset = count
    pairs = sorted(compatible)
    pair_position = {pair: y_offset + position for position, pair in enumerate(pairs)}
    variables = count + len(pairs)
    objective = np.zeros(variables)
    objective[:count] = 1e-3
    for pair, position in pair_position.items():
        objective[position] = compatible[pair].weighted

    rows = count + len(pairs) + 1
    matrix = lil_matrix((rows, variables), dtype=float)
    lower = np.full(rows, -np.inf)
    upper = np.full(rows, np.inf)
    row = 0
    for target_index in range(count):
        for pair, position in pair_position.items():
            if pair[0] == target_index:
                matrix[row, position] = 1.0
        lower[row] = upper[row] = 1.0
        row += 1
    for (target_index, builder_index), position in pair_position.items():
        matrix[row, position] = 1.0
        matrix[row, x_offset + builder_index] = -1.0
        upper[row] = 0.0
        row += 1
    matrix[row, :count] = 1.0
    upper[row] = float(limit)

    result = milp(
        c=objective,
        integrality=np.ones(variables),
        bounds=Bounds(np.zeros(variables), np.ones(variables)),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"time_limit": 120.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"catalogue MILP failed: {result.message}")
    selected = [index for index in range(count) if result.x[index] > 0.5]
    assignments = []
    slacks = []
    for target_index in range(count):
        chosen = []
        for builder_index in selected:
            position = pair_position.get((target_index, builder_index))
            if position is not None and result.x[position] > 0.5:
                chosen.append(builder_index)
        if len(chosen) != 1:
            raise RuntimeError(f"target {target_index} has assignments {chosen}")
        assignments.append(chosen[0])
        slacks.append(compatible[(target_index, chosen[0])])
    return selected, assignments, slacks


def evaluate_fixed_catalogue(
    records: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    dimension_weight: float = 0.0,
    ignore_input_dimension: bool = False,
    structural_score: str = "weighted-slack",
    unused_normalization_power: float = 0.0,
    builder_capacity_weight: float = 0.0,
    collapse_univariate_chains: bool = False,
) -> dict[str, Any]:
    """Evaluate frozen templates using structural slack plus soft dimension evidence."""
    assignments = []
    for record in records:
        term = term_from_spec(
            record["operator_variable_spec"],
            collapse_univariate_chains=collapse_univariate_chains,
        )
        dimension = input_dimension(record)
        candidates = []
        for builder_index, cluster in enumerate(clusters):
            if "consistent_fan_in_rules" in cluster:
                instantiated = instantiate_consistent_fan_in(cluster, dimension)
                envelope_options = [(
                    "consistent_fan_in_rule",
                    instantiated,
                    0,
                    "consistent_parametric_fan_in",
                )]
            else:
                envelope_options = [
                    ("global_fallback", cluster["envelope"], 1, "global_envelope")
                ]
                conditioned = cluster.get("dimension_envelopes", {}).get(str(dimension))
                if conditioned is not None:
                    conditioned_kind = cluster.get(
                        "dimension_envelope_metadata", {}
                    ).get(str(dimension), {}).get(
                        "selected_kind", "conditioned_envelope"
                    )
                    envelope_options.append((
                        "dimension_conditioned",
                        conditioned,
                        0,
                        conditioned_kind,
                    ))
            for variant, envelope, variant_priority, envelope_kind in envelope_options:
                slack = embedding_slack(term, envelope)
                if slack is None:
                    continue
                score_value = structural_score_value(
                    term,
                    envelope,
                    structural_score,
                    unused_normalization_power,
                    builder_capacity_weight,
                )
                if score_value is None:
                    raise RuntimeError("structural score disagrees with ACU embedding")
                profile = dimension_profile(cluster.get("dimension_values", ()))
                dimension_penalty = (
                    0.0
                    if ignore_input_dimension
                    else dimension_selection_penalty(dimension, profile)
                )
                combined_cost = score_value + dimension_weight * dimension_penalty
                candidates.append((
                    combined_cost,
                    score_value,
                    (
                        slack.nodes + slack.edges
                        if structural_score == "weighted-slack"
                        else slack.weighted
                    ),
                    (
                        slack.multiplicative_branches
                        if structural_score == "weighted-slack"
                        else slack.nodes + slack.edges
                    ),
                    builder_index,
                    variant_priority,
                    slack,
                    dimension_penalty,
                    variant,
                    envelope_kind,
                ))
        best = min(candidates) if candidates else None
        row = {
            "index": int(record["index"]),
            "equation": record["equation"],
            "expression": record.get("expression"),
            "input_dimension": dimension,
            "target_operator_pattern": term.notation(
                "X" if ignore_input_dimension else "X_input_dimension"
            ),
            "covered": best is not None,
        }
        if best is None:
            row.update({
                "builder": None,
                "builder_call": None,
                "slack": None,
                "structural_score": None,
                "dimension_penalty": None,
                "weighted_dimension_penalty": None,
                "combined_selection_cost": None,
                "builder_variant": None,
                "dimension_conditioning_used": False,
                "conditioned_envelope_kind": None,
                "fan_in_parameterization_used": False,
                "instantiated_operator_pattern": None,
            })
        else:
            builder_index = best[4]
            builder = f"ACB{builder_index + 1:02d}"
            selected_cluster = clusters[builder_index]
            instantiated = (
                instantiate_consistent_fan_in(selected_cluster, dimension)
                if best[8] == "consistent_fan_in_rule"
                else None
            )
            row.update({
                "builder": builder,
                "builder_call": (
                    builder if ignore_input_dimension else builder_call(builder)
                ),
                "slack": best[6].payload(),
                "structural_score": best[1],
                "dimension_penalty": best[7],
                "weighted_dimension_penalty": dimension_weight * best[7],
                "combined_selection_cost": best[0],
                "builder_variant": best[8],
                "dimension_conditioning_used": (
                    best[8] == "dimension_conditioned"
                    or (
                        instantiated is not None
                        and instantiated.signature()
                        != selected_cluster["envelope"].signature()
                    )
                ),
                "conditioned_envelope_kind": best[9],
                "fan_in_parameterization_used": (
                    best[8] == "consistent_fan_in_rule"
                ),
                "instantiated_operator_pattern": (
                    instantiated.notation("X_input_dimension")
                    if instantiated is not None else None
                ),
            })
        assignments.append(row)

    covered = [row for row in assignments if row["covered"]]
    uncovered = [row for row in assignments if not row["covered"]]
    weighted_slack = [row["slack"]["weighted"] for row in covered]
    structural_scores = [row["structural_score"] for row in covered]
    dimension_penalties = [row["dimension_penalty"] for row in covered]
    combined_costs = [row["combined_selection_cost"] for row in covered]
    dimension_counts = {
        str(dimension): sum(row["input_dimension"] == dimension for row in assignments)
        for dimension in sorted({row["input_dimension"] for row in assignments})
    }
    summary = {
        "equations": len(assignments),
        "covered_equations": len(covered),
        "uncovered_equations": len(uncovered),
        "coverage_fraction": len(covered) / len(assignments) if assignments else 0.0,
        "zero_slack_equations": sum(value == 0 for value in weighted_slack),
        "structural_score_mode": structural_score,
        "mean_structural_score_over_covered": (
            sum(structural_scores) / len(structural_scores)
            if structural_scores else None
        ),
        "maximum_structural_score_over_covered": (
            max(structural_scores) if structural_scores else None
        ),
        "mean_weighted_slack_over_covered": (
            sum(weighted_slack) / len(weighted_slack) if weighted_slack else None
        ),
        "maximum_weighted_slack_over_covered": max(weighted_slack) if weighted_slack else None,
        "dimension_weight": dimension_weight,
        "mean_dimension_penalty_over_covered": (
            sum(dimension_penalties) / len(dimension_penalties)
            if dimension_penalties else None
        ),
        "mean_weighted_dimension_penalty_over_covered": (
            dimension_weight * sum(dimension_penalties) / len(dimension_penalties)
            if dimension_penalties else None
        ),
        "mean_combined_selection_cost_over_covered": (
            sum(combined_costs) / len(combined_costs) if combined_costs else None
        ),
        "dimension_conditioned_assignments": sum(
            row["dimension_conditioning_used"] for row in covered
        ),
        "global_fallback_assignments": sum(
            not row["dimension_conditioning_used"] for row in covered
        ),
        "observed_prototype_assignments": sum(
            row["conditioned_envelope_kind"] == "observed_prototype"
            for row in covered
        ),
        "super_envelope_assignments": sum(
            row["conditioned_envelope_kind"] == "super_envelope"
            for row in covered
        ),
        "consistent_fan_in_assignments": sum(
            row["fan_in_parameterization_used"] for row in covered
        ),
        "dimension_changed_fan_in_assignments": sum(
            row["fan_in_parameterization_used"]
            and row["dimension_conditioning_used"]
            for row in covered
        ),
        "total_additive_unit_branches": sum(
            row["slack"]["additive_branches"] for row in covered
        ),
        "total_multiplicative_unit_branches": sum(
            row["slack"]["multiplicative_branches"] for row in covered
        ),
        "builder_templates_used": len({row["builder"] for row in covered}),
        "observed_builder_calls": len({row["builder_call"] for row in covered}),
        "equation_count_by_input_dimension": dimension_counts,
        "input_dimension_is_builder_parameter": not ignore_input_dimension,
        "input_dimension_ignored": ignore_input_dimension,
    }
    return {"summary": summary, "assignments": assignments, "uncovered": uncovered}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--builders", type=int, default=12)
    parser.add_argument(
        "--dimension-weight",
        type=float,
        default=0.0,
        help=(
            "Soft input-dimension weight used during clustering and assignment; "
            "zero reproduces dimension-agnostic matching."
        ),
    )
    parser.add_argument(
        "--class-weighting",
        choices=("frequency", "sqrt-frequency", "log-frequency", "uniform-class"),
        default="frequency",
        help="How strongly repeated exact AC classes influence catalogue synthesis.",
    )
    parser.add_argument(
        "--structural-score",
        choices=("weighted-slack", "unused-capacity"),
        default="weighted-slack",
        help=(
            "Structural clustering and assignment score. unused-capacity keeps "
            "the exact ACU envelope relation but counts only unused arithmetic "
            "nodes and independent raw-role routes."
        ),
    )
    parser.add_argument(
        "--unused-normalization-power",
        type=float,
        default=0.0,
        help=(
            "For unused-capacity scoring, construct with U/E^rho. "
            "rho=0 exactly reproduces the established U objective."
        ),
    )
    parser.add_argument(
        "--builder-capacity-weight",
        type=float,
        default=0.0,
        help=(
            "For unused-capacity scoring, add lambda*E to U/E^rho. "
            "E is the builder's topology capacity_units count."
        ),
    )
    parser.add_argument(
        "--source-weight",
        action="append",
        default=[],
        metavar="SOURCE=WEIGHT",
        help=(
            "Optional positive per-equation corpus weight, repeatable; unspecified "
            "sources use weight 1. Requires frequency class weighting."
        ),
    )
    parser.add_argument(
        "--dimension-conditioned-envelopes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Instantiate each builder with a dimension-specific external envelope "
            "when it improves structural fit, retaining its global fallback."
        ),
    )
    parser.add_argument(
        "--consistent-fan-in-rules",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep one fixed coloured skeleton per builder and let input dimension "
            "control only fixed child-pattern multiplicities through clipped monotone "
            "affine rules, with constant fallback per uninformative group."
        ),
    )
    parser.add_argument(
        "--ignore-input-dimension",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Treat input dimension as descriptive metadata only; it contributes "
            "nothing to synthesis, instantiation, assignment, or reported cost."
        ),
    )
    parser.add_argument(
        "--refinement-swaps",
        type=int,
        default=0,
        help=(
            "Maximum external-only, coverage-preserving one-for-one builder swaps."
        ),
    )
    parser.add_argument(
        "--candidate-milp-refinement",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Globally select the fixed builder catalogue from observed and "
            "source-only synthesized ACU candidates after greedy construction."
        ),
    )
    parser.add_argument(
        "--candidate-milp-time-limit-seconds",
        type=float,
        default=60.0,
        help="Time limit for source-only candidate-envelope MILP refinement.",
    )
    parser.add_argument(
        "--minimum-cover-time-limit-seconds",
        type=float,
        default=60.0,
        help="Time bound for the auxiliary exact observed-envelope cover diagnostic.",
    )
    parser.add_argument(
        "--evaluation-input",
        type=Path,
        default=None,
        help="Optional held-out record set evaluated after the catalogue is frozen.",
    )
    parser.add_argument(
        "--collapse-univariate-chains",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Treat every maximal one-input computation as one learned univariate "
            "edge map (canonical and enabled by default). Use --no-collapse-"
            "univariate-chains only to audit superseded legacy artifacts."
        ),
    )
    args = parser.parse_args()
    if args.dimension_weight < 0:
        parser.error("--dimension-weight must be non-negative")
    if args.refinement_swaps < 0:
        parser.error("--refinement-swaps must be non-negative")
    if not 0.0 <= args.unused_normalization_power <= 1.0:
        parser.error("--unused-normalization-power must be in [0, 1]")
    if args.builder_capacity_weight < 0.0:
        parser.error("--builder-capacity-weight must be non-negative")
    if args.candidate_milp_time_limit_seconds <= 0:
        parser.error("--candidate-milp-time-limit-seconds must be positive")
    if args.consistent_fan_in_rules and args.dimension_conditioned_envelopes:
        parser.error(
            "--consistent-fan-in-rules cannot be combined with the legacy "
            "--dimension-conditioned-envelopes mode"
        )
    if args.ignore_input_dimension and (
        args.dimension_weight != 0
        or args.consistent_fan_in_rules
        or args.dimension_conditioned_envelopes
    ):
        parser.error(
            "--ignore-input-dimension requires --dimension-weight 0 and cannot "
            "be combined with a dimension-conditioned builder mode"
        )
    source_weights: dict[str, float] = {}
    for source_weight in args.source_weight:
        if "=" not in source_weight:
            parser.error("--source-weight must have form SOURCE=WEIGHT")
        source, raw_weight = source_weight.rsplit("=", 1)
        source = source.strip()
        try:
            weight = float(raw_weight)
        except ValueError:
            parser.error(f"invalid source weight: {source_weight}")
        if not source or not math.isfinite(weight) or weight <= 0:
            parser.error(f"source weights must be finite and positive: {source_weight}")
        source_weights[source] = weight
    if source_weights and args.class_weighting != "frequency":
        parser.error("--source-weight currently requires --class-weighting frequency")
    args.output.mkdir(parents=True, exist_ok=True)

    progress = ProgressReporter(args.output / "progress.jsonl")
    analysis_started_at = time.perf_counter()
    default_excepthook = sys.excepthook

    def report_failure(exc_type: type[BaseException], exc_value: BaseException, traceback: Any) -> None:
        progress.emit(
            "analysis_failed",
            error_type=exc_type.__name__,
            error=str(exc_value),
            elapsed_seconds=round(time.perf_counter() - analysis_started_at, 6),
        )
        default_excepthook(exc_type, exc_value, traceback)

    sys.excepthook = report_failure
    progress.emit(
        "analysis_started",
        development_input=str(args.input.resolve()),
        evaluation_input=(
            str(args.evaluation_input.resolve()) if args.evaluation_input is not None else None
        ),
        output_directory=str(args.output.resolve()),
        requested_builder_templates=args.builders,
        dimension_weight=args.dimension_weight,
        dimension_policy=(
            "ignored; descriptive metadata only"
            if args.ignore_input_dimension
            else "soft evidence; never a hard compatibility constraint"
        ),
        class_weighting=args.class_weighting,
        source_weights=source_weights,
        dimension_conditioned_envelopes=args.dimension_conditioned_envelopes,
        consistent_fan_in_rules=args.consistent_fan_in_rules,
        ignore_input_dimension=args.ignore_input_dimension,
        refinement_swaps=args.refinement_swaps,
        unused_normalization_power=args.unused_normalization_power,
        builder_capacity_weight=args.builder_capacity_weight,
        collapse_univariate_chains=args.collapse_univariate_chains,
        equation_model=(
            "equation = builder"
            if args.ignore_input_dimension
            else "equation = builder(input_dimension)"
        ),
        catalogue_counting_unit=(
            "fixed builder skeleton"
            if args.ignore_input_dimension
            else "input-dimension-adaptive builder template"
        ),
    )

    stage_started_at = time.perf_counter()
    records, development_provenance = load_records(args.input)
    development_dimension_counts = {
        str(dimension): sum(input_dimension(record) == dimension for record in records)
        for dimension in sorted({input_dimension(record) for record in records})
    }
    progress.emit(
        "development_records_loaded",
        equation_count=len(records),
        equation_count_by_input_dimension=development_dimension_counts,
        provenance=development_provenance,
        elapsed_seconds=round(time.perf_counter() - stage_started_at, 6),
    )

    stage_started_at = time.perf_counter()
    terms = [
        term_from_spec(
            record["operator_variable_spec"],
            collapse_univariate_chains=args.collapse_univariate_chains,
        )
        for record in records
    ]
    ordinary_graphs = [rooted_core_graph(record["operator_variable_spec"]) for record in records]
    _, ordinary_representatives = exact_isomorphism_classes(ordinary_graphs)
    ordinary_seconds = time.perf_counter() - stage_started_at
    progress.emit(
        "ordinary_classes_computed",
        equation_count=len(records),
        ordinary_operator_coloured_rooted_classes=len(ordinary_representatives),
        elapsed_seconds=round(ordinary_seconds, 6),
    )

    stage_started_at = time.perf_counter()
    classes = exact_classes(terms)
    representatives = [members[0] for members in classes.values()]
    representative_terms = [terms[index] for index in representatives]
    class_counts = [len(members) for members in classes.values()]
    class_weights = objective_class_weights(classes.values(), args.class_weighting)
    class_dimensions = [
        [input_dimension(records[index]) for index in members]
        for members in classes.values()
    ]
    class_validation_groups = [
        [
            str(records[index].get("source_corpus", "unknown_external_source"))
            for index in members
        ]
        for members in classes.values()
    ]
    ac_seconds = time.perf_counter() - stage_started_at
    progress.emit(
        "ac_classes_computed",
        ac_operator_coloured_classes=len(classes),
        smallest_class_equation_count=min(class_counts),
        largest_class_equation_count=max(class_counts),
        class_weighting=args.class_weighting,
        elapsed_seconds=round(ac_seconds, 6),
    )

    progress.emit(
        "minimum_observed_cover_started",
        candidate_observed_envelopes=len(representative_terms),
    )
    stage_started_at = time.perf_counter()
    observed_cover = minimum_observed_cover(
        representative_terms, args.minimum_cover_time_limit_seconds
    )
    observed_cover_minimum = observed_cover["exact_minimum"]
    minimum_cover_seconds = time.perf_counter() - stage_started_at
    progress.emit(
        "minimum_observed_cover_completed",
        **observed_cover,
        candidate_observed_envelopes=len(representative_terms),
        elapsed_seconds=round(minimum_cover_seconds, 6),
    )

    progress.emit(
        "catalogue_synthesis_started",
        initial_ac_classes=len(representative_terms),
        target_builder_templates=args.builders,
        merges_required=max(0, len(representative_terms) - args.builders),
        dimension_weight=args.dimension_weight,
        class_weighting=args.class_weighting,
        structural_objective=(
            "consistent_parametric_fan_in_slack"
            if args.consistent_fan_in_rules
            else (
                "exact_envelope_unused_capacity"
                if args.structural_score == "unused-capacity"
                else "maximum_capacity_envelope_slack"
            )
        ),
    )
    stage_started_at = time.perf_counter()
    clusters = synthesize_catalogue(
        representative_terms,
        class_weights,
        args.builders,
        progress=progress,
        class_dimensions=class_dimensions,
        dimension_weight=args.dimension_weight,
        consistent_fan_in_objective=args.consistent_fan_in_rules,
        class_validation_groups=class_validation_groups,
        raw_symbol=("X" if args.ignore_input_dimension else "X_input_dimension"),
        structural_score=args.structural_score,
        unused_normalization_power=args.unused_normalization_power,
        builder_capacity_weight=args.builder_capacity_weight,
    )
    synthesis_seconds = time.perf_counter() - stage_started_at
    clusters.sort(key=lambda value: value["envelope"].signature())
    candidate_milp_summary = {
        "enabled": False,
        "candidate_envelopes": 0,
        "selected_builders": len(clusters),
        "objective": None,
        "optimality_certified": False,
    }
    if args.candidate_milp_refinement:
        refinement_started_at = time.perf_counter()
        progress.emit(
            "candidate_milp_refinement_started",
            development_ac_classes=len(representative_terms),
            requested_builder_templates=args.builders,
            time_limit_seconds=args.candidate_milp_time_limit_seconds,
            evaluation_loaded=False,
        )
        clusters, candidate_milp_summary = refine_catalogue_by_candidate_milp(
            clusters,
            representative_terms,
            class_weights,
            class_dimensions,
            args.builders,
            args.candidate_milp_time_limit_seconds,
            progress=progress,
        )
        candidate_milp_summary = {
            "enabled": True,
            **candidate_milp_summary,
            "elapsed_seconds": round(
                time.perf_counter() - refinement_started_at, 6
            ),
        }
        clusters.sort(key=lambda value: value["envelope"].signature())
    if args.dimension_conditioned_envelopes:
        conditioning_started_at = time.perf_counter()
        attach_dimension_conditioned_envelopes(
            clusters, representative_terms, class_dimensions
        )
        progress.emit(
            "dimension_conditioned_envelopes_built",
            builder_template_count=len(clusters),
            conditioned_envelope_count=sum(
                len(cluster["dimension_envelopes"]) for cluster in clusters
            ),
            observed_prototype_count=sum(
                metadata["selected_kind"] == "observed_prototype"
                for cluster in clusters
                for metadata in cluster["dimension_envelope_metadata"].values()
            ),
            super_envelope_count=sum(
                metadata["selected_kind"] == "super_envelope"
                for cluster in clusters
                for metadata in cluster["dimension_envelope_metadata"].values()
            ),
            global_fallback_count=len(clusters),
            elapsed_seconds=round(
                time.perf_counter() - conditioning_started_at, 6
            ),
        )
    refinement_summary = {
        "candidate_envelopes": 0,
        "maximum_swaps": args.refinement_swaps,
        "completed_swaps": 0,
        "initial_external_objective": None,
        "final_external_objective": None,
        "objective_reduction": 0.0,
        "swaps": [],
    }
    if args.refinement_swaps:
        refinement_started_at = time.perf_counter()
        progress.emit(
            "external_swap_refinement_started",
            maximum_swaps=args.refinement_swaps,
            builder_template_count=len(clusters),
        )
        clusters, refinement_summary = refine_catalogue_by_external_swaps(
            clusters,
            representative_terms,
            class_weights,
            class_dimensions,
            args.refinement_swaps,
            progress=progress,
            structural_score=args.structural_score,
            unused_normalization_power=args.unused_normalization_power,
            builder_capacity_weight=args.builder_capacity_weight,
        )
        clusters.sort(key=lambda value: value["envelope"].signature())
        if args.dimension_conditioned_envelopes:
            attach_dimension_conditioned_envelopes(
                clusters, representative_terms, class_dimensions
            )
        progress.emit(
            "external_swap_refinement_finished",
            **{
                key: value
                for key, value in refinement_summary.items()
                if key != "swaps"
            },
            elapsed_seconds=round(
                time.perf_counter() - refinement_started_at, 6
            ),
        )
    if args.consistent_fan_in_rules:
        parameterization_started_at = time.perf_counter()
        progress.emit(
            "consistent_fan_in_parameterization_started",
            builder_template_count=len(clusters),
            invariant=(
                "one fixed coloured skeleton; only multiplicities of fixed child "
                "subtree patterns may vary"
            ),
            rule_family="clipped non-negative affine count or constant",
        )
        attach_consistent_fan_in_rules(
            clusters,
            representative_terms,
            class_dimensions,
            class_validation_groups,
            progress=progress,
        )
        progress.emit(
            "consistent_fan_in_parameterization_finished",
            builder_template_count=len(clusters),
            fan_in_groups=sum(
                len(cluster.get("consistent_fan_in_rules", ()))
                for cluster in clusters
            ),
            dimension_informative_groups=sum(
                rule["dimension_informative"]
                for cluster in clusters
                for rule in cluster.get("consistent_fan_in_rules", ())
            ),
            constant_groups=sum(
                not rule["dimension_informative"]
                for cluster in clusters
                for rule in cluster.get("consistent_fan_in_rules", ())
            ),
            elapsed_seconds=round(
                time.perf_counter() - parameterization_started_at, 6
            ),
        )
    progress.emit(
        "catalogue_synthesis_completed",
        selected_builder_templates=len(clusters),
        merges_completed=len(representative_terms) - len(clusters),
        elapsed_seconds=round(synthesis_seconds, 6),
    )
    signature_to_class = {
        signature: class_index for class_index, signature in enumerate(classes)
    }
    stage_started_at = time.perf_counter()
    development_evaluation = evaluate_fixed_catalogue(
        records,
        clusters,
        dimension_weight=args.dimension_weight,
        ignore_input_dimension=args.ignore_input_dimension,
        structural_score=args.structural_score,
        unused_normalization_power=args.unused_normalization_power,
        builder_capacity_weight=args.builder_capacity_weight,
        collapse_univariate_chains=args.collapse_univariate_chains,
    )
    record_assignments = development_evaluation["assignments"]
    for row, term in zip(record_assignments, terms):
        row["ac_class"] = signature_to_class[term.signature()]
    if development_evaluation["uncovered"]:
        raise RuntimeError("synthesized catalogue failed to cover development records")
    assignment_seconds = time.perf_counter() - stage_started_at
    progress.emit(
        "development_assignments_completed",
        equation_count=len(record_assignments),
        zero_slack_equations=sum(
            row["slack"]["weighted"] == 0 for row in record_assignments
        ),
        mean_weighted_slack=(
            sum(row["slack"]["weighted"] for row in record_assignments)
            / len(record_assignments)
        ),
        maximum_weighted_slack=max(
            row["slack"]["weighted"] for row in record_assignments
        ),
        structural_score_mode=args.structural_score,
        mean_structural_score=(
            sum(row["structural_score"] for row in record_assignments)
            / len(record_assignments)
        ),
        maximum_structural_score=max(
            row["structural_score"] for row in record_assignments
        ),
        dimension_weight=args.dimension_weight,
        dimension_conditioned_assignments=sum(
            row["dimension_conditioning_used"] for row in record_assignments
        ),
        mean_dimension_penalty=(
            sum(row["dimension_penalty"] for row in record_assignments)
            / len(record_assignments)
        ),
        mean_combined_selection_cost=(
            sum(row["combined_selection_cost"] for row in record_assignments)
            / len(record_assignments)
        ),
        elapsed_seconds=round(assignment_seconds, 6),
    )

    builders = []
    for catalogue_index, cluster in enumerate(clusters, start=1):
        builder = f"ACB{catalogue_index:02d}"
        members = [row for row in record_assignments if row["builder"] == builder]
        observed_dimensions = sorted({row["input_dimension"] for row in members})
        equation_count_by_dimension = {
            str(dimension): sum(row["input_dimension"] == dimension for row in members)
            for dimension in observed_dimensions
        }
        compatible_classes = cluster["members"] or [
            class_index
            for class_index, term in enumerate(representative_terms)
            if embedding_slack(term, cluster["envelope"]) is not None
        ]
        closest_class = min(
            compatible_classes,
            key=lambda class_index: (
                structural_score_value(
                    representative_terms[class_index],
                    cluster["envelope"],
                    args.structural_score,
                    args.unused_normalization_power,
                    args.builder_capacity_weight,
                ),
                representative_terms[class_index].signature(),
            ),
        )
        closest_record = representatives[closest_class]
        builders.append({
            "builder": builder,
            "template_call": (
                builder if args.ignore_input_dimension else builder_call(builder)
            ),
            "input_dimension_parameter": (
                None if args.ignore_input_dimension else "input_dimension"
            ),
            "closest_observed_index": int(records[closest_record]["index"]),
            "closest_observed_equation": records[closest_record]["equation"],
            "synthetic_operator_tree": cluster["envelope"].payload(),
            "operator_pattern": cluster["envelope"].notation(
                "X" if args.ignore_input_dimension else "X_input_dimension"
            ),
            "arithmetic_nodes": cluster["envelope"].nodes,
            "arithmetic_edges": cluster["envelope"].edges,
            "structural_score_mode": args.structural_score,
            "equation_count": len(members),
            "observed_input_dimensions": observed_dimensions,
            "equation_count_by_input_dimension": equation_count_by_dimension,
            "dimension_profile": dimension_profile(cluster["dimension_values"]),
            "dimension_matching_policy": (
                "ignored; retained as descriptive equation metadata only"
                if args.ignore_input_dimension
                else "fixed coloured skeleton; input dimension controls only repeated "
                "identical-child fan-in through consistent rules"
                if args.consistent_fan_in_rules
                else "soft; structural improvement may override dimension distance"
            ),
            "consistent_fan_in_rules": cluster.get(
                "consistent_fan_in_rules", []
            ),
            "fan_in_groups": len(
                cluster.get("consistent_fan_in_rules", ())
            ),
            "dimension_informative_fan_in_groups": sum(
                rule["dimension_informative"]
                for rule in cluster.get("consistent_fan_in_rules", ())
            ),
            "constant_fan_in_groups": sum(
                not rule["dimension_informative"]
                for rule in cluster.get("consistent_fan_in_rules", ())
            ),
            "dimension_conditioned_envelopes": {
                dimension: {
                    "operator_pattern": envelope.notation("X_input_dimension"),
                    "synthetic_operator_tree": envelope.payload(),
                    "development_ac_class_count": cluster[
                        "dimension_envelope_class_counts"
                    ][dimension],
                    **cluster["dimension_envelope_metadata"][dimension],
                }
                for dimension, envelope in cluster.get(
                    "dimension_envelopes", {}
                ).items()
            },
            "builder_call": (
                builder if args.ignore_input_dimension else builder_call(builder)
            ),
            "weighted_slack_total": sum(row["slack"]["weighted"] for row in members),
            "additive_unit_branches": sum(row["slack"]["additive_branches"] for row in members),
            "multiplicative_unit_branches": sum(row["slack"]["multiplicative_branches"] for row in members),
            "members": [row["equation"] for row in members],
        })
    progress.emit(
        "builder_templates_frozen",
        builder_template_count=len(builders),
        dimension_weight=args.dimension_weight,
        equation_count=len(record_assignments),
        equation_count_by_input_dimension=development_dimension_counts,
        builder_equation_counts={
            row["builder"]: row["equation_count"] for row in builders
        },
    )

    development_dimensions = sorted({row["input_dimension"] for row in record_assignments})
    equation_count_by_dimension = {
        str(dimension): sum(row["input_dimension"] == dimension for row in record_assignments)
        for dimension in development_dimensions
    }
    observed_builder_calls = {
        row["builder_call"] for row in record_assignments
    }

    summary = {
        "equations": len(records),
        "ordinary_operator_coloured_rooted_classes": len(ordinary_representatives),
        "ac_operator_coloured_classes": len(classes),
        "minimum_observed_coloured_envelopes": observed_cover_minimum,
        "observed_envelope_cover_diagnostic": observed_cover,
        "selected_builders": len(builders),
        "selected_builder_templates": len(builders),
        "observed_builder_calls": len(observed_builder_calls),
        "agglomerative_merges": len(representative_terms) - len(builders),
        "builder_kind": "synthetic least-general ACU envelopes",
        "optimization": (
            "source-only global MILP selection over observed, pair-merged, and greedy ACU candidates"
            if args.candidate_milp_refinement
            else
            f"{args.class_weighting} greedy structural agglomeration with input dimension ignored; not a global optimum certificate"
            if args.ignore_input_dimension
            else f"{args.class_weighting} greedy agglomeration with soft dimension dispersion; not a global optimum certificate"
        ),
        "class_weighting": args.class_weighting,
        "structural_score_mode": args.structural_score,
        "collapse_univariate_chains": args.collapse_univariate_chains,
        "dimension_weight": args.dimension_weight,
        "input_dimension_ignored": args.ignore_input_dimension,
        "dimension_conditioned_envelopes_enabled": args.dimension_conditioned_envelopes,
        "consistent_fan_in_rules_enabled": args.consistent_fan_in_rules,
        "dimension_conditioned_envelope_count": sum(
            len(cluster.get("dimension_envelopes", {})) for cluster in clusters
        ),
        "fan_in_groups": sum(
            len(cluster.get("consistent_fan_in_rules", ()))
            for cluster in clusters
        ),
        "dimension_informative_fan_in_groups": sum(
            rule["dimension_informative"]
            for cluster in clusters
            for rule in cluster.get("consistent_fan_in_rules", ())
        ),
        "constant_fan_in_groups": sum(
            not rule["dimension_informative"]
            for cluster in clusters
            for rule in cluster.get("consistent_fan_in_rules", ())
        ),
        "external_swap_refinement": refinement_summary,
        "candidate_milp_refinement": candidate_milp_summary,
        "total_weighted_slack": sum(row["slack"]["weighted"] for row in record_assignments),
        "total_structural_score": sum(row["structural_score"] for row in record_assignments),
        "mean_structural_score_per_equation": sum(
            row["structural_score"] for row in record_assignments
        ) / len(record_assignments),
        "maximum_structural_score_per_equation": max(
            row["structural_score"] for row in record_assignments
        ),
        "total_additive_unit_branches": sum(row["slack"]["additive_branches"] for row in record_assignments),
        "total_multiplicative_unit_branches": sum(row["slack"]["multiplicative_branches"] for row in record_assignments),
        "total_neutralised_raw_attachments": sum(row["slack"]["raw_attachments"] for row in record_assignments),
        "total_neutralised_nodes": sum(row["slack"]["nodes"] for row in record_assignments),
        "total_neutralised_edges": sum(row["slack"]["edges"] for row in record_assignments),
        "mean_neutral_branches_per_equation": sum(
            row["slack"]["additive_branches"] + row["slack"]["multiplicative_branches"]
            for row in record_assignments
        ) / len(record_assignments),
        "mean_weighted_branch_penalty_per_equation": sum(
            row["slack"]["additive_branches"] + 2 * row["slack"]["multiplicative_branches"]
            for row in record_assignments
        ) / len(record_assignments),
        "zero_slack_equations": sum(row["slack"]["weighted"] == 0 for row in record_assignments),
        "mean_weighted_slack_per_equation": sum(
            row["slack"]["weighted"] for row in record_assignments
        ) / len(record_assignments),
        "maximum_weighted_slack_per_equation": max(
            row["slack"]["weighted"] for row in record_assignments
        ),
        "mean_dimension_penalty_per_equation": sum(
            row["dimension_penalty"] for row in record_assignments
        ) / len(record_assignments),
        "mean_weighted_dimension_penalty_per_equation": sum(
            row["weighted_dimension_penalty"] for row in record_assignments
        ) / len(record_assignments),
        "mean_combined_selection_cost_per_equation": sum(
            row["combined_selection_cost"] for row in record_assignments
        ) / len(record_assignments),
        "dimension_conditioned_assignments": sum(
            row["dimension_conditioning_used"] for row in record_assignments
        ),
        "global_fallback_assignments": sum(
            not row["dimension_conditioning_used"] for row in record_assignments
        ),
        "input_dimension_parameterization": {
            "equation_model": (
                "equation = builder"
                if args.ignore_input_dimension
                else "equation = builder(input_dimension)"
            ),
            "catalogue_counting_unit": (
                "fixed builder skeleton"
                if args.ignore_input_dimension
                else "input-dimension-adaptive builder template"
            ),
            "raw_symbol": (
                "X" if args.ignore_input_dimension else "X_input_dimension"
            ),
            "observed_dimensions": development_dimensions,
            "equation_count_by_input_dimension": equation_count_by_dimension,
            "dimension_policy": (
                "ignored; recorded as descriptive metadata only"
                if args.ignore_input_dimension
                else "soft evidence in clustering and assignment; never a hard compatibility constraint"
            ),
            "dimension_weight": args.dimension_weight,
            "dimension_penalty": (
                "disabled; always zero"
                if args.ignore_input_dimension
                else "robust distance from each builder's development median, scaled by its mean absolute deviation"
            ),
            "conditioned_envelopes_enabled": args.dimension_conditioned_envelopes,
            "consistent_fan_in_rules_enabled": args.consistent_fan_in_rules,
            "conditioned_envelope_policy": (
                "disabled; every builder is one fixed coloured skeleton"
                if args.ignore_input_dimension
                else "one invariant coloured skeleton per builder; only fixed child-pattern multiplicities follow a clipped monotone affine rule, or a constant rule when dimension is uninformative"
                if args.consistent_fan_in_rules
                else "instantiate the envelope learned for the equation's input dimension when it gives a better structural fit; otherwise use the same builder's global envelope"
            ),
        },
        "rewrite_system": {
            "associativity": "flatten same-operator nodes only across exact identity state edges",
            "commutativity": "children are unordered",
            "additive_unit": "extra additive branch is fixed to 0",
            "multiplicative_unit": "extra multiplicative branch is fixed to 1",
            "operator_substitution": "forbidden at arity two or greater; unary sum and product are identified",
        },
        "build_progress": {
            "progress_log": "progress.jsonl",
            "ordinary_classification_seconds": round(ordinary_seconds, 6),
            "ac_classification_seconds": round(ac_seconds, 6),
            "minimum_observed_cover_seconds": round(minimum_cover_seconds, 6),
            "catalogue_synthesis_seconds": round(synthesis_seconds, 6),
            "development_assignment_seconds": round(assignment_seconds, 6),
        },
    }
    payload = {
        "development": {
            "input": str(args.input.resolve()),
            "provenance": development_provenance,
            "assignments": record_assignments,
        },
        "summary": summary,
        "builders": builders,
    }
    evaluation = None
    evaluation_provenance = None
    if args.evaluation_input is not None:
        stage_started_at = time.perf_counter()
        evaluation_records, evaluation_provenance = load_records(args.evaluation_input)
        evaluation_dimension_counts = {
            str(dimension): sum(
                input_dimension(record) == dimension for record in evaluation_records
            )
            for dimension in sorted(
                {input_dimension(record) for record in evaluation_records}
            )
        }
        progress.emit(
            "held_out_records_loaded",
            equation_count=len(evaluation_records),
            equation_count_by_input_dimension=evaluation_dimension_counts,
            provenance=evaluation_provenance,
            elapsed_seconds=round(time.perf_counter() - stage_started_at, 6),
        )
        progress.emit(
            "held_out_evaluation_started",
            equation_count=len(evaluation_records),
            frozen_builder_template_count=len(builders),
        )
        stage_started_at = time.perf_counter()
        evaluation = evaluate_fixed_catalogue(
            evaluation_records,
            clusters,
            dimension_weight=args.dimension_weight,
            ignore_input_dimension=args.ignore_input_dimension,
            structural_score=args.structural_score,
            unused_normalization_power=args.unused_normalization_power,
            builder_capacity_weight=args.builder_capacity_weight,
            collapse_univariate_chains=args.collapse_univariate_chains,
        )
        evaluation_seconds = time.perf_counter() - stage_started_at
        payload["evaluation"] = {
            "input": str(args.evaluation_input.resolve()),
            "provenance": evaluation_provenance,
            **evaluation,
        }
        progress.emit(
            "held_out_evaluation_completed",
            **evaluation["summary"],
            elapsed_seconds=round(evaluation_seconds, 6),
        )

    progress.emit(
        "artifact_write_started",
        expected_artifacts=[
            "catalogue.json",
            "development_assignments.jsonl",
            "builders.jsonl",
            *( ["evaluation_assignments.jsonl"] if evaluation is not None else [] ),
            "REPORT.md",
            "progress.jsonl",
        ],
    )
    (args.output / "catalogue.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_jsonl(args.output / "development_assignments.jsonl", record_assignments)
    write_jsonl(args.output / "builders.jsonl", builders)
    if evaluation is not None:
        write_jsonl(args.output / "evaluation_assignments.jsonl", evaluation["assignments"])
    report = [
        "# Algebraic operator-coloured builder catalogue",
        "",
        f"The **{len(builders)} builders** were inferred from **{len(records)} development equations**. Any evaluation set reported below was loaded only after this catalogue was frozen.",
        "",
        "Each builder fixes one rooted coloured skeleton. Variable names are quotiented by permutation; +/* colours remain distinct at arity two or greater, while unary sum and product are identified.",
        "",
        (
            "Every catalogue entry is one fixed builder written `ACBj`. Input dimension is retained only as descriptive equation metadata and does not change or select a builder."
            if args.ignore_input_dimension
            else "Every catalogue entry is input-dimension adaptive and is written `ACBj(input_dimension)`. Each equation supplies exactly its own variable count automatically; input dimensions do not create additional catalogue builders."
        ),
        "",
        (
            "Input dimension is ignored by clustering, builder structure, assignment, and cost. All reported dimension penalties are therefore exactly zero."
            if args.ignore_input_dimension
            else f"A robust dimension-profile distance is available as soft evidence with weight **{args.dimension_weight:g}**. The selection objective is `structural slack + weight × robust dimension distance`. Dimension is never a hard compatibility rule, so a better structural match can override it. Structural slack, dimension penalty, and combined cost are reported separately."
        ),
        *(
            [
                "",
                f"In addition, each builder is instantiated as a dimension-conditioned envelope when external development data provide one. There are **{sum(len(cluster.get('dimension_envelopes', {})) for cluster in clusters)} internal conditioned envelopes**: **{sum(metadata['selected_kind'] == 'observed_prototype' for cluster in clusters for metadata in cluster.get('dimension_envelope_metadata', {}).values())} observed prototypes** and **{sum(metadata['selected_kind'] == 'super_envelope' for cluster in clusters for metadata in cluster.get('dimension_envelope_metadata', {}).values())} synthesized super-envelopes**. They remain branches of the same {len(builders)} builders and are not counted as catalogue entries. The global envelope is retained and used whenever conditioning is incompatible or structurally worse.",
            ]
            if args.dimension_conditioned_envelopes else []
        ),
        *(
            [
                "",
                f"The consistent parameterization contains **{summary['fan_in_groups']} fixed child-pattern fan-in groups** inside those same {len(builders)} skeletons. Input dimension changes **{summary['dimension_informative_fan_in_groups']} groups** through one clipped non-negative affine count rule per group; **{summary['constant_fan_in_groups']} groups** use a constant count because dimension did not reduce required capacity. A multiplicity-one group is an optional 0/1 slot; larger groups repeat only identical copies of the same fixed child subtree. No rule can change an operator colour, substitute a subtree, or change nesting. Every fitted group rule is recorded separately in `progress.jsonl` and `builders.jsonl`.",
            ]
            if args.consistent_fan_in_rules else []
        ),
        "",
        (
            f"Ordinary rooted coloured isomorphism gives **{len(ordinary_representatives)} classes**. Conservative associative-commutative normalization gives **{len(classes)} classes**. Selecting only observed graphs has an exact minimum of **{observed_cover_minimum} envelopes**; therefore the {len(builders)} entries below are synthesized ACU superterms."
            if observed_cover_minimum is not None
            else (
                f"Ordinary rooted coloured isomorphism gives **{len(ordinary_representatives)} classes**. Conservative associative-commutative normalization gives **{len(classes)} classes**. The auxiliary exact observed-envelope cover diagnostic was skipped, so no minimum is claimed. The requested {len(builders)} entries are synthesized ACU superterms."
                if observed_cover["solver_status"] == "skipped"
                else f"Ordinary rooted coloured isomorphism gives **{len(ordinary_representatives)} classes**. Conservative associative-commutative normalization gives **{len(classes)} classes**. The time-bounded observed-envelope cover diagnostic found incumbent **{observed_cover['best_incumbent']}** and lower bound **{observed_cover['lower_bound']}**, without certifying the exact minimum. The requested {len(builders)} entries are synthesized ACU superterms."
            )
        ),
        "",
        "## Build progress",
        "",
        "Detailed machine-readable progress is appended to `progress.jsonl`. It records every stage and every agglomerative merge, including remaining class count, incremental and accumulated weighted slack, and the new envelope pattern.",
        "",
        "| Stage | Result | Elapsed seconds |",
        "|---|---:|---:|",
        f"| Development records loaded | {len(records)} equations | — |",
        f"| Ordinary rooted coloured classification | {len(ordinary_representatives)} classes | {ordinary_seconds:.3f} |",
        f"| Associative-commutative classification | {len(classes)} classes | {ac_seconds:.3f} |",
        (
            f"| Minimum cover using observed envelopes | {observed_cover_minimum} envelopes (certified) | {minimum_cover_seconds:.3f} |"
            if observed_cover_minimum is not None
            else (
                f"| Observed-envelope cover diagnostic | skipped; no minimum claimed | {minimum_cover_seconds:.3f} |"
                if observed_cover["solver_status"] == "skipped"
                else f"| Observed-envelope cover diagnostic | incumbent {observed_cover['best_incumbent']}; lower bound {observed_cover['lower_bound']} (not certified) | {minimum_cover_seconds:.3f} |"
            )
        ),
        f"| Greedy ACU synthesis | {len(representative_terms) - len(builders)} merges to {len(builders)} builders | {synthesis_seconds:.3f} |",
        f"| Development assignment | {len(record_assignments)} equations assigned | {assignment_seconds:.3f} |",
        "",
        "An unmatched branch of a sum is fixed to 0. An unmatched branch of a product is fixed to 1 and receives twice the branch penalty. Same-operator nodes are flattened only across exact identity state edges.",
        f"Of the {len(records)} development equations, **{summary['zero_slack_equations']} require no neutral branches**. The literal neutral-branch mean is **{summary['mean_neutral_branches_per_equation']:.3f}**; weighting product identities twice gives **{summary['mean_weighted_branch_penalty_per_equation']:.3f}**. After also charging unused raw attachments and all neutralized subtree nodes and edges, the structural composite mean is **{summary['mean_weighted_slack_per_equation']:.2f}** and the maximum is **{summary['maximum_weighted_slack_per_equation']}**. The dimension penalty is **{summary['mean_dimension_penalty_per_equation']:.3f}**, giving mean combined selection cost **{summary['mean_combined_selection_cost_per_equation']:.3f}**.",
        "",
        (
            "Here `X` denotes the raw input collection at a node; every displayed +/* symbol is fixed by the builder."
            if args.ignore_input_dimension
            else "Here `X_input_dimension` denotes the adaptive raw input collection; every displayed +/* symbol is fixed by the builder."
        ),
        "",
        *(
            [
                "| Fixed builder | Fixed operator skeleton | Observed input dimensions (metadata) | Equation counts by input dimension | Closest observed equation | Equations | Additive-zero branches | Multiplicative-one branches | Weighted slack |",
                "|---|---|---|---|---|---:|---:|---:|---:|",
                *[
                    f"| {row['template_call']} | `{row['operator_pattern']}` | {row['observed_input_dimensions']} | `{row['equation_count_by_input_dimension']}` | {row['closest_observed_equation']} | {row['equation_count']} | {row['additive_unit_branches']} | {row['multiplicative_unit_branches']} | {row['weighted_slack_total']} |"
                    for row in builders
                ],
            ]
            if args.ignore_input_dimension
            else [
                "| Adaptive builder | Fixed maximum-capacity skeleton | Fan-in groups (dimension / constant) | Observed input dimensions | Equation counts by input dimension | Closest observed equation | Equations | Additive-zero branches | Multiplicative-one branches | Weighted slack |",
                "|---|---|---:|---|---|---|---:|---:|---:|---:|",
                *[
                    f"| {row['template_call']} | `{row['operator_pattern']}` | {row['dimension_informative_fan_in_groups']} / {row['constant_fan_in_groups']} | {row['observed_input_dimensions']} | `{row['equation_count_by_input_dimension']}` | {row['closest_observed_equation']} | {row['equation_count']} | {row['additive_unit_branches']} | {row['multiplicative_unit_branches']} | {row['weighted_slack_total']} |"
                    for row in builders
                ],
            ]
        ),
        "",
        "## Interpretation",
        "",
        "This is a rooted colour-preserving partial isomorphism modulo associativity and commutativity, extended with the neutral elements 0 and 1 and unary sum/product identification. In term-rewriting terminology it is an ACU-envelope relation. It is stricter than generic graph edit distance because +/* substitutions are forbidden at arity two or greater and every deletion has an algebraic interpretation.",
        "",
        (
            f"The {len(builders)}-builder solution is selected by a source-only MILP from observed, pair-merged, and greedy ACU candidates. Its optimum claim, when certified, applies only to that recorded finite candidate pool."
            if args.candidate_milp_refinement
            else f"The {len(builders)}-builder solution is produced by {args.class_weighting} greedy agglomeration. It is a valid exact-coverage construction with reported slack, but it is not a proof that no lower-slack {len(builders)}-builder catalogue exists."
        ),
    ]
    if evaluation is not None:
        evaluation_summary = evaluation["summary"]
        report.extend([
            "",
            "## Held-out evaluation",
            "",
            (
                f"The frozen catalogue covers **{evaluation_summary['covered_equations']}/{evaluation_summary['equations']}** held-out equations. **{evaluation_summary['zero_slack_equations']}** are exact AC matches. Over covered equations, mean structural composite slack is **{evaluation_summary['mean_weighted_slack_over_covered']:.3f}** and maximum structural slack is **{evaluation_summary['maximum_weighted_slack_over_covered']}**. Input dimension is ignored and its penalty is exactly **0**."
                if args.ignore_input_dimension
                else f"The frozen catalogue covers **{evaluation_summary['covered_equations']}/{evaluation_summary['equations']}** held-out equations, with input dimension supplied automatically by each equation. **{evaluation_summary['zero_slack_equations']}** are exact AC matches. Over covered equations, mean structural composite slack is **{evaluation_summary['mean_weighted_slack_over_covered']:.3f}** and maximum structural slack is **{evaluation_summary['maximum_weighted_slack_over_covered']}**. Mean dimension penalty is **{evaluation_summary['mean_dimension_penalty_over_covered']:.3f}** and mean combined selection cost is **{evaluation_summary['mean_combined_selection_cost_over_covered']:.3f}**. The fixed fan-in rules were instantiated for **{evaluation_summary['consistent_fan_in_assignments']}** covered equations; dimension reduced at least one multiplicity below the maximum-capacity skeleton in **{evaluation_summary['dimension_changed_fan_in_assignments']}** assignments."
                if args.consistent_fan_in_rules
                else f"The frozen catalogue covers **{evaluation_summary['covered_equations']}/{evaluation_summary['equations']}** held-out equations, with input dimension supplied automatically by each equation. **{evaluation_summary['zero_slack_equations']}** are exact AC matches. Over covered equations, mean structural composite slack is **{evaluation_summary['mean_weighted_slack_over_covered']:.3f}** and maximum structural slack is **{evaluation_summary['maximum_weighted_slack_over_covered']}**. Mean dimension penalty is **{evaluation_summary['mean_dimension_penalty_over_covered']:.3f}** and mean combined selection cost is **{evaluation_summary['mean_combined_selection_cost_over_covered']:.3f}**. Dimension-conditioned envelopes were selected for **{evaluation_summary['dimension_conditioned_assignments']}** equations: **{evaluation_summary['observed_prototype_assignments']}** used observed prototypes and **{evaluation_summary['super_envelope_assignments']}** used super-envelopes. **{evaluation_summary['global_fallback_assignments']}** used global envelopes."
            ),
            "",
            (
                "No held-out equation or its input dimension is used to add, merge, recolour, resize, or reorder a builder."
                if args.ignore_input_dimension
                else "No held-out equation is used to add, merge, recolour, refit, or reorder a builder. Its input dimension only evaluates the already-frozen count rules."
            ),
        ])
        if evaluation["uncovered"]:
            report.extend([
                "",
                "### Uncovered held-out equations",
                "",
                *[
                    f"- {row['equation']}: `{row['target_operator_pattern']}`"
                    for row in evaluation["uncovered"]
                ],
            ])
    (args.output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    written_artifacts = [
        "catalogue.json",
        "development_assignments.jsonl",
        "builders.jsonl",
        *( ["evaluation_assignments.jsonl"] if evaluation is not None else [] ),
        "REPORT.md",
        "progress.jsonl",
    ]
    progress.emit(
        "artifacts_written",
        artifacts=written_artifacts,
        output_directory=str(args.output.resolve()),
    )
    progress.emit(
        "analysis_completed",
        development_equations=len(records),
        selected_builder_templates=len(builders),
        held_out_equations=(
            evaluation["summary"]["equations"] if evaluation is not None else 0
        ),
        elapsed_seconds=round(time.perf_counter() - analysis_started_at, 6),
    )
    printable = {"development": summary}
    if evaluation is not None:
        printable["evaluation"] = evaluation["summary"]
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
