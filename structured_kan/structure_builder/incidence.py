"""Audit raw incidence and unary placement lost by the Boolean-R catalogue.

This is deliberately a structural stage.  It reuses the frozen external-240
corpus, the held-out Feynman operator-variable specifications, and the exact
ACU child assignment from the previous analyser.  No target expression is
used to design a builder.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DEFAULT_EXTERNAL = PROJECT_ROOT / "dataset" / "external240" / "balanced_external240_source_summary.json"
DEFAULT_FEYNMAN = PROJECT_ROOT / "dataset" / "feynman120" / "selected_operator_variable_specs.json"
DEFAULT_CATALOGUE = PROJECT_ROOT / "artifacts" / "historical_catalogue"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "audit"
DEFAULT_FAILURE_AUDIT = PROJECT_ROOT / "dataset" / "audits" / "general_failure_modes_audit.json"




from . import synthesis as ACU
Term = ACU.Term


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), separators=(",", ":")) + "\n")


def append_event(path: Path, event: str, **payload: Any) -> None:
    row = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "event": event,
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(json.dumps(row, separators=(",", ":")), flush=True)


def edge_route(edge: Mapping[str, Any]) -> str:
    # Every neural edge already has an affine input/output map. Numerical
    # weight, bias, input scale, and input bias therefore do not define a
    # structural unary placement. Only the nonlinear primitive matters.
    return "identity" if edge.get("primitive", "x") == "x" else "unary"


@dataclass(frozen=True)
class RawPort:
    kind: str
    roles: tuple[int, ...]
    route: str

    def payload(self) -> dict[str, Any]:
        return {"kind": self.kind, "roles": list(self.roles), "route": self.route}


@dataclass(frozen=True)
class RichChild:
    route: str
    node: "RichNode"


@dataclass(frozen=True)
class RichNode:
    operator: str
    raw_ports: tuple[RawPort, ...]
    children: tuple[RichChild, ...]
    term: Any


@dataclass(frozen=True)
class BuilderNode:
    operator: str
    raw_attachment: bool
    children: tuple["BuilderNode", ...]
    term: Any
    raw_arity: int = 0


def rich_from_spec(
    spec: Mapping[str, Any], *, collapse_univariate_chains: bool = False
) -> tuple[RichNode, str]:
    by_id = {str(node["id"]): node for node in spec.get("nodes", ())}
    output_sources = [str(value) for value in spec.get("output_sources", ())]
    if len(output_sources) != 1:
        raise ValueError("expected one output source")
    role_ids: dict[tuple[Any, ...], int] = {}

    def source_role_key(source: Mapping[str, Any]) -> tuple[Any, ...] | None:
        indices = tuple(int(value) for value in source.get("variable_indices", ()))
        if not indices:
            return None
        if str(source.get("source_kind")) == "raw" and len(indices) == 1:
            return ("raw", indices[0])
        coefficients = tuple(round(float(value), 12) for value in source.get("coefficients", ()))
        return (
            "affine",
            indices,
            coefficients,
            round(float(source.get("constant", 0.0)), 12),
        )

    def make_raw_port(source: Mapping[str, Any]) -> RawPort | None:
        key = source_role_key(source)
        if key is None:
            # Numeric constants and their rescalings are represented by the
            # affine map/gain/offset already present on every f edge/node.
            return None
        if key not in role_ids:
            role_ids[key] = len(role_ids)
        # A role is itself a learned affine projection. Multivariate affine
        # symbolic sources therefore consume one role port, not a new port
        # category and not one port per participating variable.
        return RawPort("role", (role_ids[key],), edge_route(source.get("edge", {})))

    def merge_ports(ports: Iterable[RawPort], operator: str) -> tuple[RawPort, ...]:
        """Canonicalize repeated uses of one scalar role.

        In the collapsed grammar, any computation depending on one scalar
        coordinate is one learned unary route.  Repeated occurrences at a
        product therefore become a unary route (for example ``x*x`` becomes
        one route able to learn ``x**2``).  The legacy path retains its prior
        incidence behavior for historical artifact reproduction.
        """
        values = list(ports)
        if not collapse_univariate_chains and operator == "*":
            return tuple(sorted(values, key=lambda port: (port.roles, port.route)))
        ports_by_role: dict[int, list[RawPort]] = defaultdict(list)
        for port in values:
            ports_by_role[int(port.roles[0])].append(port)
        result: list[RawPort] = []
        for role, role_ports in sorted(ports_by_role.items()):
            nonlinear = any(port.route == "unary" for port in role_ports)
            if operator == "*" and len(role_ports) > 1:
                nonlinear = True
            result.append(RawPort(
                "role", (role,), "unary" if nonlinear else "identity"
            ))
        return tuple(result)

    @lru_cache(maxsize=None)
    def visit(
        node_id: str,
    ) -> tuple[RichNode | None, frozenset[int], str]:
        if node_id not in by_id:
            digits = "".join(character for character in node_id if character.isdigit())
            role = int(digits) if digits else 0
            port = RawPort("role", (role,), "identity")
            term = Term("+", True, (), raw_arity=1)
            return RichNode("+", (port,), (), term), frozenset((role,)), "identity"

        source_node = by_id[node_id]
        operator = str(source_node["operator"])
        ports: list[RawPort] = [
            port
            for source in source_node.get("raw_sources", ())
            if (port := make_raw_port(source)) is not None
        ]
        dependencies = {int(role) for port in ports for role in port.roles}
        children: list[RichChild] = []
        for state_source in source_node.get("state_sources", ()):
            child, child_dependencies, child_post_route = visit(
                str(state_source["state"])
            )
            if child is None:
                continue
            dependencies.update(child_dependencies)
            route = (
                "unary"
                if child_post_route == "unary"
                or edge_route(state_source.get("edge", {})) == "unary"
                else "identity"
            )
            if collapse_univariate_chains and len(child_dependencies) <= 1:
                if child_dependencies:
                    role = next(iter(child_dependencies))
                    ports.append(RawPort("role", (role,), "unary"))
                continue
            if child.operator == operator and route == "identity":
                ports.extend(child.raw_ports)
                children.extend(child.children)
            else:
                children.append(RichChild(route, child))
        canonical_ports = merge_ports(ports, operator)
        children.sort(key=lambda value: (
            value.node.term.signature(),
            value.route,
            rich_signature(value.node),
        ))
        if not canonical_ports and not children:
            return None, frozenset(dependencies), "identity"
        if collapse_univariate_chains and not canonical_ports and len(children) == 1:
            child = children[0]
            return child.node, frozenset(dependencies), child.route
        term = Term(
            operator,
            bool(canonical_ports),
            tuple(child.node.term for child in children),
            raw_arity=len(canonical_ports),
        )
        return (
            RichNode(operator, canonical_ports, tuple(children), term),
            frozenset(dependencies),
            "identity",
        )

    output_id = output_sources[0]
    root, _dependencies, collapsed_output_route = visit(output_id)
    if root is None:
        raise ValueError("constant-only targets are outside the incidence catalogue")
    output_edges = [
        value for value in spec.get("output_edges", ())
        if str(value.get("source")) == output_id
    ]
    output_route = (
        "unary"
        if collapsed_output_route == "unary"
        or (edge_route(output_edges[0]) if output_edges else "identity") == "unary"
        else "identity"
    )
    if collapse_univariate_chains:
        expected = ACU.term_from_spec(spec, collapse_univariate_chains=True)
        if root.term.signature() != expected.signature():
            raise RuntimeError({
                "reason": "collapsed incidence/topology disagreement",
                "incidence": root.term.notation(),
                "topology": expected.notation(),
            })
    return root, output_route


def rich_signature(node: RichNode) -> str:
    payload = {
        "operator": node.operator,
        "raw": sorted((port.kind, port.roles, port.route) for port in node.raw_ports),
        "children": sorted((child.route, rich_signature(child.node)) for child in node.children),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def builder_from_payload(payload: Mapping[str, Any]) -> BuilderNode:
    operator = str(payload["operator"])
    children = [builder_from_payload(child) for child in payload.get("children", ())]
    children.sort(key=lambda value: value.term.signature())
    raw_arity = int(payload.get("raw_arity", 0))
    term = Term(
        operator,
        bool(payload.get("raw_attachment", False)),
        tuple(child.term for child in children),
        raw_arity=raw_arity,
    )
    return BuilderNode(
        operator,
        bool(payload.get("raw_attachment", False)),
        tuple(children),
        term,
        raw_arity=term.raw_arity,
    )


def capacity_separated_rich_term(node: RichNode) -> Any:
    """Project a target to Boolean raw sites for shared-route matching.

    Actual role arity remains in ``RichNode.raw_ports`` and is checked against
    the materialized core plus reserve after the topology has been aligned.
    """
    return ACU.Term(
        node.operator,
        bool(node.raw_ports),
        tuple(capacity_separated_rich_term(child.node) for child in node.children),
        raw_arity=1 if node.raw_ports else 0,
    )


def capacity_separated_builder_term(node: BuilderNode) -> Any:
    """Project a parametric builder to Boolean raw attachment sites."""
    return ACU.Term(
        node.operator,
        node.raw_attachment,
        tuple(capacity_separated_builder_term(child) for child in node.children),
        raw_arity=1 if node.raw_attachment else 0,
    )


def path_text(path: tuple[int, ...]) -> str:
    return "root" if not path else "root." + ".".join(map(str, path))


def zero_requirement() -> dict[str, int]:
    return {
        "role_identity": 0,
        "role_unary": 0,
    }


def canonical_role_profile(occurrences: Mapping[int, list[str]]) -> list[list[str]]:
    # Renaming inputs cannot change a builder.  Sorting occurrence signatures
    # quotients the profile by arbitrary variable names/indices.
    return sorted((sorted(values) for values in occurrences.values()), key=lambda value: (len(value), value))


def align_case(
    target: RichNode,
    output_route: str,
    builder: BuilderNode,
    raw_arity_policy: str = "strict",
) -> dict[str, Any] | None:
    if raw_arity_policy == "strict":
        target_term = target.term
        builder_term = builder.term
    elif raw_arity_policy == "capacity-separated":
        target_term = capacity_separated_rich_term(target)
        builder_term = capacity_separated_builder_term(builder)
    else:
        raise ValueError(f"unknown raw arity policy {raw_arity_policy!r}")
    if ACU.embedding_slack(target_term, builder_term) is None:
        return None

    node_requirements: dict[str, dict[str, int]] = {}
    state_routes: dict[str, str] = {}
    role_occurrences: dict[int, list[str]] = defaultdict(list)
    raw_port_records: list[dict[str, Any]] = []
    current_raw_capacity = 0
    current_raw_deficit = 0
    current_raw_excess = 0

    def visit(
        target_node: RichNode,
        builder_node: BuilderNode,
        projected_target: Any,
        projected_builder: Any,
        path: tuple[int, ...],
    ) -> None:
        nonlocal current_raw_capacity, current_raw_deficit, current_raw_excess
        selected = ACU.embedding_child_assignment(projected_target, projected_builder)
        if selected is None:
            raise RuntimeError("cached ACU alignment disappeared")
        label = path_text(path)
        requirement = zero_requirement()
        for port_index, port in enumerate(target_node.raw_ports):
            key = f"{port.kind}_{port.route}"
            requirement[key] += 1
            occurrence_label = f"{label}:{port.kind}:{port.route}:{port_index}"
            for role in port.roles:
                role_occurrences[role].append(occurrence_label)
            raw_port_records.append({
                "path": label,
                "kind": port.kind,
                "route": port.route,
                "roles": list(port.roles),
            })
        node_requirements[label] = requirement
        target_port_count = len(target_node.raw_ports)
        capacity = (
            builder_node.raw_arity
            if raw_arity_policy == "strict"
            else int(builder_node.raw_attachment)
        )
        current_raw_capacity += capacity
        current_raw_deficit += max(0, target_port_count - capacity)
        current_raw_excess += max(0, capacity - target_port_count)

        for target_index, builder_index in selected:
            target_child = target_node.children[target_index]
            edge_label = f"{label}->{path_text(path + (builder_index,))}"
            state_routes[edge_label] = target_child.route
            visit(
                target_child.node,
                builder_node.children[builder_index],
                projected_target.children[target_index],
                projected_builder.children[builder_index],
                path + (builder_index,),
            )

    visit(target, builder, target_term, builder_term, ())
    role_profile = canonical_role_profile(role_occurrences)
    canonical_role_order = sorted(
        role_occurrences,
        key=lambda role: (
            len(role_occurrences[role]),
            sorted(role_occurrences[role]),
            role,
        ),
    )
    canonical_role = {role: index for index, role in enumerate(canonical_role_order)}
    canonical_ports = sorted(
        ({
            **port,
            "roles": sorted(canonical_role[role] for role in port["roles"]),
        } for port in raw_port_records),
        key=lambda port: (
            port["path"], port["kind"], port["route"], port["roles"]
        ),
    )
    raw_totals = Counter()
    for requirement in node_requirements.values():
        raw_totals.update(requirement)
    role_occurrence_count = sum(len(values) for values in role_occurrences.values())
    nonconstant_raw_port_count = sum(raw_totals.values())
    return {
        "node_requirements": node_requirements,
        "state_routes": state_routes,
        "output_route": output_route,
        "raw_totals": dict(raw_totals),
        "raw_port_count": nonconstant_raw_port_count,
        "role_occurrence_count": role_occurrence_count,
        "distinct_role_count": len(role_occurrences),
        "role_reuse_excess": role_occurrence_count - len(role_occurrences),
        "reused_role_count": sum(len(values) > 1 for values in role_occurrences.values()),
        "role_profile": role_profile,
        "role_profile_key": json.dumps(role_profile, separators=(",", ":")),
        "canonical_raw_ports": canonical_ports,
        # Retain these legacy JSON keys because the stage-2 materializer reads
        # them; their values are now integer raw-route arities, not Booleans.
        "current_boolean_raw_capacity": current_raw_capacity,
        "current_boolean_raw_deficit": current_raw_deficit,
        "current_boolean_raw_excess": current_raw_excess,
        "current_raw_route_capacity": current_raw_capacity,
        "current_raw_route_deficit": current_raw_deficit,
        "current_raw_route_excess": current_raw_excess,
        "raw_arity_policy": raw_arity_policy,
    }


def percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    alpha = position - lower
    return (1.0 - alpha) * ordered[lower] + alpha * ordered[upper]


def summarize_builder(builder: str, rows: list[dict[str, Any]], builder_node: BuilderNode) -> dict[str, Any]:
    node_paths = sorted({path for row in rows for path in row["incidence"]["node_requirements"]})
    edge_paths = sorted({path for row in rows for path in row["incidence"]["state_routes"]})
    node_capacities: dict[str, dict[str, Any]] = {}
    for path in node_paths:
        requirements: dict[str, list[int]] = defaultdict(list)
        for row in rows:
            values = row["incidence"]["node_requirements"].get(path, zero_requirement())
            for key, value in values.items():
                requirements[key].append(int(value))
        node_capacities[path] = {
            key: {
                "maximum": max(values),
                "mean": sum(values) / len(values),
                "p50": percentile(values, 0.50),
                "p90": percentile(values, 0.90),
                "nonzero_equations": sum(value > 0 for value in values),
            }
            for key, values in sorted(requirements.items())
        }

    state_route_policy: dict[str, dict[str, Any]] = {}
    for path in edge_paths:
        routes = [row["incidence"]["state_routes"].get(path) for row in rows]
        used = [route for route in routes if route is not None]
        state_route_policy[path] = {
            "mode": "learned_unary" if "unary" in used else "native_identity",
            "identity_equations": used.count("identity"),
            "unary_equations": used.count("unary"),
            "unused_equations": routes.count(None),
        }

    # Fixed maximum capacities are intentionally conservative in stage 1.  A
    # later selection stage may split a broad family rather than keeping these
    # maxima and paying excessive slack.
    capacity_by_node = {
        path: {key: int(stats["maximum"]) for key, stats in values.items()}
        for path, values in node_capacities.items()
    }
    typed_capacity_slacks: list[int] = []
    learned_route_slacks: list[int] = []
    for row in rows:
        typed_slack = 0
        for path, capacities in capacity_by_node.items():
            required = row["incidence"]["node_requirements"].get(path, zero_requirement())
            typed_slack += sum(capacities[key] - int(required.get(key, 0)) for key in capacities)
        typed_capacity_slacks.append(typed_slack)
        learned_route_slacks.append(sum(
            policy["mode"] == "learned_unary"
            and row["incidence"]["state_routes"].get(path) == "identity"
            for path, policy in state_route_policy.items()
        ))

    profiles = Counter(row["incidence"]["role_profile_key"] for row in rows)
    return {
        "builder": builder,
        "operator_pattern": builder_node.term.notation(),
        "equation_count": len(rows),
        "node_capacities": node_capacities,
        "state_route_policy": state_route_policy,
        "output_route_mode": (
            "learned_unary"
            if any(row["incidence"]["output_route"] == "unary" for row in rows)
            else "native_identity"
        ),
        "role_capacity": max(row["incidence"]["distinct_role_count"] for row in rows),
        "distinct_role_profiles": len(profiles),
        "top_role_profiles": [
            {"profile": json.loads(key), "equation_count": count}
            for key, count in profiles.most_common(10)
        ],
        "mean_typed_capacity_slack_at_max_envelope": sum(typed_capacity_slacks) / len(rows),
        "maximum_typed_capacity_slack_at_max_envelope": max(typed_capacity_slacks),
        "mean_identity_routes_forced_through_learned_state_edges": sum(learned_route_slacks) / len(rows),
        "mean_current_boolean_raw_deficit": sum(row["incidence"]["current_boolean_raw_deficit"] for row in rows) / len(rows),
        "equations_with_current_boolean_raw_deficit": sum(row["incidence"]["current_boolean_raw_deficit"] > 0 for row in rows),
    }


def geometric_mean(values: Iterable[float]) -> float:
    observed = [max(float(value), 1.0e-300) for value in values]
    return math.exp(sum(math.log(value) for value in observed) / len(observed))


def failure_baseline(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", ())
    if not rows:
        return None
    baseline = geometric_mean(row["test_gnmse"] for row in rows)
    return {
        "equation_count": len(rows),
        "metric": "validation-selected test GNMSE",
        "baseline": baseline,
        "target_improvement": 30.0,
        "target_maximum": baseline / 30.0,
    }


def load_inputs(args: argparse.Namespace) -> tuple[Any, ...]:
    external_payload = json.loads(args.external.read_text(encoding="utf-8"))
    external = list(
        external_payload["selected"]
        if isinstance(external_payload, dict)
        else external_payload
    )
    feynman = json.loads(args.feynman.read_text(encoding="utf-8"))
    builders = read_jsonl(args.catalogue / "builders.jsonl")
    development_path = args.catalogue / "development_assignments.jsonl"
    if not development_path.exists():
        development_path = args.catalogue / "external_assignments.jsonl"
    evaluation_path = args.catalogue / "evaluation_assignments.jsonl"
    if not evaluation_path.exists():
        evaluation_path = args.catalogue / "feynman_assignments.jsonl"
    development = read_jsonl(development_path)
    evaluation = read_jsonl(evaluation_path)
    return external, feynman, builders, development, evaluation


def process_split(
    split: str,
    records: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
    builder_nodes: Mapping[str, BuilderNode],
    structural_score: str = "weighted-slack",
    collapse_univariate_chains: bool = False,
    raw_arity_policy: str = "strict",
) -> list[dict[str, Any]]:
    by_index = {int(row["index"]): row for row in assignments}
    output: list[dict[str, Any]] = []
    for offset, record in enumerate(records, start=1):
        record_index = int(record.get("index", offset))
        equation = str(record["equation"])
        saved_assignment = by_index[record_index]
        rich, output_route = rich_from_spec(
            record["operator_variable_spec"],
            collapse_univariate_chains=collapse_univariate_chains,
        )
        compatible = []
        for builder, builder_node in builder_nodes.items():
            if raw_arity_policy == "strict":
                target_term = rich.term
                builder_term = builder_node.term
            elif raw_arity_policy == "capacity-separated":
                target_term = capacity_separated_rich_term(rich)
                builder_term = capacity_separated_builder_term(builder_node)
            else:
                raise ValueError(f"unknown raw arity policy {raw_arity_policy!r}")
            slack = ACU.embedding_slack(target_term, builder_term)
            if slack is not None:
                score = ACU.structural_score_value(
                    target_term, builder_term, structural_score
                )
                if score is None:
                    raise RuntimeError("embedding and structural score disagree")
                compatible.append((
                    score,
                    (
                        slack.nodes + slack.edges
                        if structural_score == "weighted-slack"
                        else slack.weighted
                    ),
                    builder,
                    slack,
                ))
        compatible.sort(key=lambda value: value[:3])
        # The catalogue builder owns the development assignment.  Re-selecting
        # the lowest-capacity compatible builder here can make a valid builder
        # appear unused and then derive no materialization prior for it.  Keep
        # the saved assignment when it remains compatible; only fall back to a
        # fresh closest match for legacy/incomplete artifacts.
        saved_builder = saved_assignment.get(
            "builder", saved_assignment.get("selected_builder")
        )
        selected = next(
            (value for value in compatible if value[2] == saved_builder),
            compatible[0] if compatible else None,
        )
        row = {
            "split": split,
            "index": record_index,
            "equation": equation,
            "expression": str(record.get("expression", "")),
            "input_dimension": len(record["operator_variable_spec"].get("variables", ())),
            "covered": selected is not None,
            "builder": selected[2] if selected is not None else None,
            "structural_score_mode": structural_score,
            "structural_score": selected[0] if selected is not None else None,
            "structural_slack": selected[3].payload() if selected is not None else None,
            "saved_pre_normalization_builder": saved_builder,
        }
        if row["covered"]:
            incidence = align_case(
                rich,
                output_route,
                builder_nodes[str(row["builder"])],
                raw_arity_policy=raw_arity_policy,
            )
            if incidence is None:
                raise RuntimeError(f"{equation}: saved assignment is no longer compatible")
            row["incidence"] = incidence
        output.append(row)
    return output


def split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    covered = [row for row in rows if row["covered"]]
    return {
        "equations": len(rows),
        "covered": len(covered),
        "uncovered": len(rows) - len(covered),
        "mean_raw_ports": sum(row["incidence"]["raw_port_count"] for row in covered) / len(covered),
        "mean_role_occurrences": sum(row["incidence"]["role_occurrence_count"] for row in covered) / len(covered),
        "mean_role_reuse_excess": sum(row["incidence"]["role_reuse_excess"] for row in covered) / len(covered),
        "equations_with_reused_roles": sum(row["incidence"]["reused_role_count"] > 0 for row in covered),
        "mean_current_boolean_raw_deficit": sum(row["incidence"]["current_boolean_raw_deficit"] for row in covered) / len(covered),
        "equations_with_current_boolean_raw_deficit": sum(row["incidence"]["current_boolean_raw_deficit"] > 0 for row in covered),
        "total_current_boolean_raw_deficit": sum(row["incidence"]["current_boolean_raw_deficit"] for row in covered),
        "raw_route_counts": dict(Counter(
            key.split("_", 1)[1]
            for row in covered
            for key, count in row["incidence"]["raw_totals"].items()
            for _ in range(int(count))
        )),
        "state_route_counts": dict(Counter(
            route
            for row in covered
            for route in row["incidence"]["state_routes"].values()
        )),
    }


def render_report(payload: Mapping[str, Any]) -> str:
    external = payload["external_summary"]
    feynman = payload["feynman_summary"]
    baseline = payload.get("accuracy_target")
    lines = [
        "# Incidence-aware audit of the frozen 18 builders",
        "",
        "The local `f(x)` catalogue direction is discarded. This audit keeps the",
        "full `+`/`*` skeleton and measures raw-input incidence, variable reuse,",
        "and unary-map placement at every ACU-matched node.",
        "",
        "## Accuracy objective",
        "",
    ]
    if baseline:
        lines.extend([
            f"- Fixed audit set: {baseline['equation_count']} equations",
            f"- Current {baseline['metric']}: `{baseline['baseline']:.12g}`",
            f"- 30x target: at most `{baseline['target_maximum']:.12g}`",
        ])
    lines.extend([
        "",
        "## Raw-route arity audit",
        "",
        "| Split | Covered | Mean true raw ports | Mean reuse excess | Cases with reused roles | Mean missing raw ports | Cases missing raw ports |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| External 240 | {external['covered']}/{external['equations']} | "
            f"{external['mean_raw_ports']:.2f} | {external['mean_role_reuse_excess']:.2f} | "
            f"{external['equations_with_reused_roles']} | {external['mean_current_boolean_raw_deficit']:.2f} | "
            f"{external['equations_with_current_boolean_raw_deficit']} |"
        ),
        (
            f"| Feynman 120 | {feynman['covered']}/{feynman['equations']} | "
            f"{feynman['mean_raw_ports']:.2f} | {feynman['mean_role_reuse_excess']:.2f} | "
            f"{feynman['equations_with_reused_roles']} | {feynman['mean_current_boolean_raw_deficit']:.2f} | "
            f"{feynman['equations_with_current_boolean_raw_deficit']} |"
        ),
        "",
        "A deficit means the matched builder supplies fewer independent raw-source",
        "routes than the target requires at the same algebraic node. The rebuilt",
        "arity-aware catalogue has zero such deficit on its assigned source records.",
        "",
        "## Route placement",
        "",
        f"- External raw routes: `{external['raw_route_counts']}`",
        f"- External state routes: `{external['state_route_counts']}`",
        f"- Feynman raw routes: `{feynman['raw_route_counts']}`",
        f"- Feynman state routes: `{feynman['state_route_counts']}`",
        "",
        "Every runtime edge uses one KAN-native map with its affine base already",
        "present. Zero nonlinear coefficients therefore give the affine case",
        "without a separate fit. Shared role nodes encode repeated inputs; numeric",
        "constants and rescaling do not create structural ports.",
        "",
        "## Catalogue design rule",
        "",
        "Each builder has one fixed skeleton, fixed port capacities, and jointly",
        "trained affine/nonlinear edge maps. Input dimension only determines the width",
        "of the affine role bank; it never switches the builder to another",
        "structure. Broad maximum-capacity envelopes are diagnostic only: stage 2",
        "must split or replace families whose typed capacity slack remains high.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--feynman", type=Path, default=DEFAULT_FEYNMAN)
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    parser.add_argument("--failure-audit", type=Path, default=DEFAULT_FAILURE_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--structural-score",
        choices=("weighted-slack", "unused-capacity"),
        default="weighted-slack",
    )
    parser.add_argument(
        "--collapse-univariate-chains",
        action="store_true",
        help="Use the executable topology projection that contracts maximal one-input computations.",
    )
    parser.add_argument(
        "--raw-arity-policy",
        choices=("strict", "capacity-separated"),
        default="strict",
        help="Separate topology compatibility from learned role capacity for Boolean-R catalogues.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    events = args.output / "progress.jsonl"
    events.write_text("", encoding="utf-8")

    append_event(events, "load_started")
    external, feynman, builders, development, evaluation = load_inputs(args)
    builder_nodes = {
        str(row["builder"]): builder_from_payload(row["synthetic_operator_tree"])
        for row in builders
    }
    append_event(
        events,
        "load_completed",
        external_equations=len(external),
        feynman_equations=len(feynman),
        builders=len(builders),
    )

    append_event(events, "external_alignment_started")
    external_rows = process_split(
        "external_development", external, development, builder_nodes,
        args.structural_score,
        args.collapse_univariate_chains,
        args.raw_arity_policy,
    )
    write_jsonl(args.output / "external_equations.jsonl", external_rows)
    append_event(events, "external_alignment_completed", equations=len(external_rows))

    append_event(events, "feynman_alignment_started")
    feynman_rows = process_split(
        "feynman_evaluation", feynman, evaluation, builder_nodes,
        args.structural_score,
        args.collapse_univariate_chains,
        args.raw_arity_policy,
    )
    write_jsonl(args.output / "feynman_equations.jsonl", feynman_rows)
    append_event(events, "feynman_alignment_completed", equations=len(feynman_rows))

    builder_rows: list[dict[str, Any]] = []
    external_by_builder: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in external_rows:
        if row["covered"]:
            external_by_builder[str(row["builder"])].append(row)
    for builder in sorted(builder_nodes):
        rows = external_by_builder[builder]
        if not rows:
            raise RuntimeError(f"{builder} has no external assignments")
        builder_rows.append(summarize_builder(builder, rows, builder_nodes[builder]))
    write_jsonl(args.output / "builder_capacity_audit.jsonl", builder_rows)

    payload = {
        "method": {
            "algebra": "operator-coloured ACU +/* embedding",
            "raw_ports": ["constant", "role", "multivariate affine"],
            "routes": ["native identity", "learned unary"],
            "role_equivalence": "variable names quotiented; repeated occurrences share a role",
            "input_dimension_policy": "projection width only; no dimension-conditioned topology",
            "structural_score": args.structural_score,
            "collapse_univariate_chains": args.collapse_univariate_chains,
            "raw_arity_policy": args.raw_arity_policy,
        },
        "accuracy_target": failure_baseline(args.failure_audit),
        "external_summary": split_summary(external_rows),
        "feynman_summary": split_summary(feynman_rows),
        "builders": builder_rows,
    }
    (args.output / "audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.output / "REPORT.md").write_text(render_report(payload), encoding="utf-8")
    append_event(
        events,
        "audit_completed",
        external_raw_route_deficit=payload["external_summary"]["total_current_boolean_raw_deficit"],
        feynman_raw_route_deficit=payload["feynman_summary"]["total_current_boolean_raw_deficit"],
    )


if __name__ == "__main__":
    main()
