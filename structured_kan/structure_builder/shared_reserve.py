"""Build the runtime-oriented 18-builder incidence parameterization.

Unlike the strict typed-pattern experiment, this catalogue does not treat raw
incidence or unary placement as immutable topology. Each arithmetic node has
sparsely gated role lanes, each lane can reuse an affine role, and every edge
can stay affine or activate a learned unary residual. External patterns are
aggregated into priors inside one fitted model per builder.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import incidence as audit


HERE = Path(__file__).resolve().parent
DEFAULT_AUDIT = HERE / "outputs" / "current18_audit"
DEFAULT_OUTPUT = HERE / "outputs" / "incidence_parametric_external240_18"


def percentile_integer(values: list[int], fraction: float) -> int:
    value = audit.percentile(values, fraction)
    if value is None:
        return 0
    return int(math.ceil(value - 1.0e-12))


def all_builder_paths(root: audit.BuilderNode) -> list[str]:
    output: list[str] = []

    def visit(node: audit.BuilderNode, path: tuple[int, ...]) -> None:
        output.append(audit.path_text(path))
        for index, child in enumerate(node.children):
            visit(child, path + (index,))

    visit(root, ())
    return output


def all_builder_state_paths(root: audit.BuilderNode) -> list[str]:
    output: list[str] = []

    def visit(node: audit.BuilderNode, path: tuple[int, ...]) -> None:
        parent = audit.path_text(path)
        for index, child in enumerate(node.children):
            child_path = path + (index,)
            output.append(f"{parent}->{audit.path_text(child_path)}")
            visit(child, child_path)

    visit(root, ())
    return output


def builder_nodes_by_path(root: audit.BuilderNode) -> dict[str, audit.BuilderNode]:
    output: dict[str, audit.BuilderNode] = {}

    def visit(node: audit.BuilderNode, path: tuple[int, ...]) -> None:
        output[audit.path_text(path)] = node
        for index, child in enumerate(node.children):
            visit(child, path + (index,))

    visit(root, ())
    return output


def node_archetype(node: audit.BuilderNode) -> str:
    return "|".join((
        node.operator,
        "raw" if node.raw_attachment else "state_only",
        "leaf" if not node.children else "internal",
    ))


def global_role_incidence_profile(incidence: Mapping[str, Any]) -> tuple[tuple[str, ...], ...]:
    """Return a variable-name-free multiset of shared-role destination masks."""
    paths_by_role: dict[int, set[str]] = defaultdict(set)
    for port in incidence["canonical_raw_ports"]:
        for role in port["roles"]:
            paths_by_role[int(role)].add(str(port["path"]))
    return tuple(sorted(
        (tuple(sorted(paths)) for paths in paths_by_role.values()),
        key=lambda paths: (-len(paths), paths),
    ))


def role_profile_features(
    profile: tuple[tuple[str, ...], ...],
) -> tuple[int, Counter[str], Counter[tuple[str, str]]]:
    path_counts: Counter[str] = Counter()
    reused_pairs: Counter[tuple[str, str]] = Counter()
    for mask in profile:
        unique_paths = tuple(sorted(set(mask)))
        path_counts.update(unique_paths)
        reused_pairs.update(combinations(unique_paths, 2))
    return len(profile), path_counts, reused_pairs


def role_profile_distance(
    left: tuple[tuple[str, ...], ...],
    right: tuple[tuple[str, ...], ...],
) -> int:
    """Compare role count, per-node incidence, and cross-node reuse."""
    left_roles, left_paths, left_pairs = role_profile_features(left)
    right_roles, right_paths, right_pairs = role_profile_features(right)
    path_error = sum(
        abs(left_paths[key] - right_paths[key])
        for key in left_paths.keys() | right_paths.keys()
    )
    reuse_error = sum(
        abs(left_pairs[key] - right_pairs[key])
        for key in left_pairs.keys() | right_pairs.keys()
    )
    return abs(left_roles - right_roles) + path_error + 2 * reuse_error


def medoid_role_profile(
    incidences: Iterable[Mapping[str, Any]],
) -> tuple[tuple[tuple[str, ...], ...], dict[str, Any]]:
    profiles = [global_role_incidence_profile(incidence) for incidence in incidences]
    counts = Counter(profiles)
    if not counts:
        return (), {"profiles": 0, "distinct_profiles": 0, "mean_distance": 0.0}
    scored = []
    for candidate, frequency in counts.items():
        total = sum(
            role_profile_distance(candidate, observed) * observed_frequency
            for observed, observed_frequency in counts.items()
        )
        scored.append((total, -frequency, len(candidate), candidate))
    total, negative_frequency, _, selected = min(scored)
    return selected, {
        "profiles": len(profiles),
        "distinct_profiles": len(counts),
        "selected_frequency": -negative_frequency,
        "mean_distance": total / len(profiles),
    }


def prefix_role_profile(
    activation_priors: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[str, ...], ...]:
    maximum = max(
        (int(prior["initial_active_lanes"]) for prior in activation_priors.values()),
        default=0,
    )
    masks = []
    for lane in range(maximum):
        masks.append(tuple(sorted(
            path
            for path, prior in activation_priors.items()
            if lane < int(prior["initial_active_lanes"])
        )))
    return tuple(sorted(masks, key=lambda paths: (-len(paths), paths)))


def materialized_capacity(rule: Mapping[str, Any], input_dimension: int) -> int:
    if rule.get("capacity_policy") == "dimension-adaptive":
        dimension = int(input_dimension)
        global_by_dimension = rule["external_global_quantile_by_input_dimension"]
        path_by_dimension = rule["external_path_maximum_by_input_dimension"]
        global_capacity = int(global_by_dimension.get(
            str(dimension), min(dimension, max(1, math.ceil(2.0 * dimension / 3.0)))
        ))
        path_capacity = int(path_by_dimension.get(str(dimension), 0))
        return min(dimension, max(global_capacity, path_capacity))
    capacity = int(rule.get(
        "external_path_capacity",
        max(
            int(rule["external_global_active_node_p85"]),
            int(rule["external_path_maximum"]),
        ),
    ))
    return min(int(input_dimension), capacity) if rule.get("cap_by_input_dimension", True) else capacity


def evaluate(
    records: list[dict[str, Any]],
    builders: Mapping[str, audit.BuilderNode],
    rules: Mapping[str, Mapping[str, Mapping[str, Any]]],
    structural_score: str = "weighted-slack",
    raw_arity_policy: str = "strict",
    collapse_univariate_chains: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset, record in enumerate(records, start=1):
        target, output_route = audit.rich_from_spec(
            record["operator_variable_spec"],
            collapse_univariate_chains=collapse_univariate_chains,
        )
        dimension = len(record["operator_variable_spec"].get("variables", ()))
        candidates = []
        for builder, skeleton in builders.items():
            incidence = audit.align_case(
                target, output_route, skeleton,
                raw_arity_policy=raw_arity_policy,
            )
            if incidence is None:
                continue
            ordinary_rules = [
                rule for path, rule in rules[builder].items()
                if path != "__shared_pool__"
            ]
            cap_by_input_dimension = all(
                rule.get("cap_by_input_dimension", True)
                for rule in ordinary_rules
            )
            shared_pool = rules[builder].get("__shared_pool__")
            if shared_pool is not None:
                core_by_path = shared_pool.get("core_capacity_by_node")
                if core_by_path is not None:
                    deficit = 0
                    core_excess = 0
                    for path, requirement in incidence["node_requirements"].items():
                        required = sum(int(value) for value in requirement.values())
                        core = int(core_by_path.get(path, 0))
                        if cap_by_input_dimension:
                            core = min(dimension, core)
                        deficit += max(0, required - core)
                        core_excess += max(0, core - required)
                    reserve = int(shared_pool["reserve_capacity"])
                    compatible = deficit <= reserve
                    port_slack = core_excess + reserve - deficit if compatible else 0
                else:
                    required = sum(
                        int(value)
                        for requirement in incidence["node_requirements"].values()
                        for value in requirement.values()
                    )
                    capacity = int(shared_pool["capacity"])
                    compatible = required <= capacity
                    port_slack = capacity - required if compatible else 0
            else:
                port_slack = 0
                compatible = True
                for path, requirement in incidence["node_requirements"].items():
                    required = sum(int(value) for value in requirement.values())
                    capacity = materialized_capacity(rules[builder][path], dimension)
                    if required > capacity:
                        compatible = False
                        break
                    port_slack += capacity - required
            if not compatible:
                continue
            target_term = (
                audit.capacity_separated_rich_term(target)
                if raw_arity_policy == "capacity-separated"
                else target.term
            )
            builder_term = (
                audit.capacity_separated_builder_term(skeleton)
                if raw_arity_policy == "capacity-separated"
                else skeleton.term
            )
            structural = audit.ACU.embedding_slack(target_term, builder_term)
            if structural is None:
                raise RuntimeError("alignment and slack disagree")
            score = audit.ACU.structural_score_value(
                target_term, builder_term, structural_score
            )
            if score is None:
                raise RuntimeError("alignment and structural score disagree")
            candidates.append((
                score + port_slack,
                port_slack,
                score,
                structural.weighted,
                builder,
                incidence,
            ))
        candidates.sort(key=lambda value: value[:5])
        winner = candidates[0] if candidates else None
        rows.append({
            "index": int(record.get("index", offset)),
            "equation": str(record["equation"]),
            "expression": str(record.get("expression", "")),
            "input_dimension": dimension,
            "covered": winner is not None,
            "builder": winner[4] if winner else None,
            "structural_score_mode": structural_score,
            "structural_slack": winner[2] if winner else None,
            "legacy_weighted_structural_slack": winner[3] if winner else None,
            "port_capacity_slack": winner[1] if winner else None,
            "combined_slack": winner[0] if winner else None,
        })
    covered = [row for row in rows if row["covered"]]
    summary = {
        "equations": len(rows),
        "covered": len(covered),
        "uncovered": len(rows) - len(covered),
        "mean_structural_slack": sum(row["structural_slack"] for row in covered) / len(covered),
        "mean_port_capacity_slack": sum(row["port_capacity_slack"] for row in covered) / len(covered),
        "mean_combined_slack": sum(row["combined_slack"] for row in covered) / len(covered),
        "maximum_port_capacity_slack": max(row["port_capacity_slack"] for row in covered),
    }
    return rows, summary


def render_report(payload: Mapping[str, Any]) -> str:
    development = payload["development"]
    evaluation = payload["evaluation"]
    capacity = payload["capacity_rule"]
    score_mode = payload.get("structural_score", "weighted-slack")
    score_label = (
        "Mean unused capacity"
        if score_mode == "unused-capacity"
        else "Mean weighted structural slack"
    )
    if capacity["policy"] == "compatible-shared-reserve":
        capacity_description = [
            "Each node keeps its external assigned-path maximum as core capacity.",
            f"A builder-wide shared reserve is the p{100 * capacity['reserve_quantile']:.0f}",
            "of deficits from all structurally compatible external equations.",
            "Reserve nonlinear parameters are tied across destinations; per-edge",
            "affine gates route or neutralize them inside the builder's single fit.",
            (
                "Input dimension consistently caps materialized core width."
                if capacity["input_dimension_changes_materialized_width"]
                else "Core width is fixed and is not capped by input dimension."
            ),
        ]
    else:
        capacity_description = [
            f"Capacity policy: `{capacity['policy']}` with external quantile "
            f"{capacity['external_quantile']:.3g}.",
            "Input dimension controls parameter width but never selects another builder.",
        ]
    return "\n".join([
        "# Parametric incidence catalogue: 18 builders",
        "",
        f"Structural selection score: `{score_mode}`. Only this score differs",
        "from the established shared-reserve catalogue construction.",
        "",
        "Constants and affine rescalings are absent from the structure. Every",
        "nonconstant raw source is one affine role; repeated ports may reuse the",
        "same role. Unary placement is an affine-versus-learned residual gate,",
        "not a fixed primitive label.",
        "",
        *capacity_description,
        "",
        f"| Split | Coverage | {score_label} | Mean active port slack | Combined |",
        "|---|---:|---:|---:|---:|",
        (
            f"| External 240 | {development['covered']}/{development['equations']} | "
            f"{development['mean_structural_slack']:.3f} | {development['mean_port_capacity_slack']:.3f} | "
            f"{development['mean_combined_slack']:.3f} |"
        ),
        (
            f"| Feynman 120 | {evaluation['covered']}/{evaluation['equations']} | "
            f"{evaluation['mean_structural_slack']:.3f} | {evaluation['mean_port_capacity_slack']:.3f} | "
            f"{evaluation['mean_combined_slack']:.3f} |"
        ),
        "",
        "Each builder is fitted exactly once per equation. External incidence",
        "patterns are aggregated into lane-activation and unary-route priors",
        "inside that one model; they never create separately fitted or selected",
        "configurations.",
        "",
        "Static coverage/slack does not establish the 30x accuracy result. That",
        "requires a validation-selected numerical screen on the frozen 50-case",
        "audit with the existing data splits and reporting rule.",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external", type=Path, default=audit.DEFAULT_EXTERNAL)
    parser.add_argument("--feynman", type=Path, default=audit.DEFAULT_FEYNMAN)
    parser.add_argument("--catalogue", type=Path, default=audit.DEFAULT_CATALOGUE)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--global-capacity-quantile", type=float, default=0.85)
    parser.add_argument(
        "--assigned-core-quantile", type=float, default=1.0,
        help=(
            "Robust per-path core capacity quantile over equations assigned to "
            "the builder. The default 1.0 reproduces the historical maximum."
        ),
    )
    parser.add_argument(
        "--capacity-policy",
        choices=(
            "global-floor",
            "raw-sites-global-floor",
            "compatible-path-quantile",
            "dimension-adaptive",
            "archetype-floor",
            "shared-pool",
            "core-plus-shared-reserve",
            "raw-sites-floor-plus-reserve",
            "compatible-shared-reserve",
        ),
        default="global-floor",
    )
    parser.add_argument("--ignore-input-dimension-cap", action="store_true")
    parser.add_argument("--shared-reserve-capacity", type=int, default=1)
    parser.add_argument("--reserve-quantile", type=float, default=0.85)
    parser.add_argument(
        "--prior-population",
        choices=("assigned", "compatible", "compatible-closest"),
        default="assigned",
        help="Population used for activation and state-route priors; capacities remain external-assignment derived.",
    )
    parser.add_argument(
        "--role-incidence-prior",
        choices=("independent-prefix", "compatible-medoid"),
        default="independent-prefix",
    )
    parser.add_argument(
        "--structural-score",
        choices=("weighted-slack", "unused-capacity"),
        default="weighted-slack",
    )
    parser.add_argument(
        "--raw-arity-policy",
        choices=("strict", "capacity-separated"),
        default="strict",
        help=(
            "Topology matching policy. capacity-separated reproduces the v2 "
            "two-stage rule: Boolean raw-site matching first, followed by "
            "dimension-capped core/shared-reserve feasibility."
        ),
    )
    parser.add_argument(
        "--collapse-univariate-chains",
        action="store_true",
        help=(
            "Use the executable topology projection that contracts every maximal "
            "one-input computation into one learned unary edge."
        ),
    )
    args = parser.parse_args()
    if not 0.0 <= args.assigned_core_quantile <= 1.0:
        raise ValueError("--assigned-core-quantile must lie in [0, 1]")
    args.output.mkdir(parents=True, exist_ok=True)
    progress = args.output / "progress.jsonl"
    progress.write_text("", encoding="utf-8")

    audit.append_event(progress, "load_started")
    external_payload = json.loads(args.external.read_text(encoding="utf-8"))
    external_records = list(
        external_payload["selected"]
        if isinstance(external_payload, dict)
        else external_payload
    )
    feynman_records = json.loads(args.feynman.read_text(encoding="utf-8"))
    external_rows = audit.read_jsonl(args.audit / "external_equations.jsonl")
    source_builders = audit.read_jsonl(args.catalogue / "builders.jsonl")
    builders = {
        str(row["builder"]): audit.builder_from_payload(row["synthetic_operator_tree"])
        for row in source_builders
    }
    compatible_by_builder: dict[str, list[dict[str, Any]]] = {
        name: [] for name in builders
    }
    for record in external_records:
        target, output_route = audit.rich_from_spec(
            record["operator_variable_spec"],
            collapse_univariate_chains=args.collapse_univariate_chains,
        )
        dimension = len(record["operator_variable_spec"].get("variables", ()))
        for name, skeleton in builders.items():
            incidence = audit.align_case(
                target, output_route, skeleton,
                raw_arity_policy=args.raw_arity_policy,
            )
            if incidence is not None:
                target_term = (
                    audit.capacity_separated_rich_term(target)
                    if args.raw_arity_policy == "capacity-separated"
                    else target.term
                )
                builder_term = (
                    audit.capacity_separated_builder_term(skeleton)
                    if args.raw_arity_policy == "capacity-separated"
                    else skeleton.term
                )
                structural = audit.ACU.embedding_slack(target_term, builder_term)
                if structural is None:
                    raise RuntimeError("alignment and structural slack disagree")
                compatible_by_builder[name].append({
                    "incidence": incidence,
                    "input_dimension": dimension,
                    "structural_slack": structural.weighted,
                })
    by_builder: dict[str, list[dict[str, Any]]] = defaultdict(list)
    active_node_counts: list[int] = []
    active_node_counts_by_dimension: dict[int, list[int]] = defaultdict(list)
    for row in external_rows:
        if not row["covered"]:
            continue
        by_builder[str(row["builder"])].append(row)
        counts = [
            sum(int(value) for value in requirement.values())
            for requirement in row["incidence"]["node_requirements"].values()
        ]
        active_node_counts.extend(counts)
        active_node_counts_by_dimension[int(row["input_dimension"])].extend(counts)
    global_floor = percentile_integer(active_node_counts, args.global_capacity_quantile)
    global_floor_by_dimension = {
        str(dimension): percentile_integer(values, args.global_capacity_quantile)
        for dimension, values in sorted(active_node_counts_by_dimension.items())
    }
    archetype_counts: dict[str, list[int]] = defaultdict(list)
    for row in external_rows:
        if not row["covered"]:
            continue
        node_by_path = builder_nodes_by_path(builders[str(row["builder"])])
        for path, requirement in row["incidence"]["node_requirements"].items():
            archetype_counts[node_archetype(node_by_path[path])].append(
                sum(int(value) for value in requirement.values())
            )
    archetype_floor = {
        key: percentile_integer(values, args.global_capacity_quantile)
        for key, values in sorted(archetype_counts.items())
    }
    compatible_counts: dict[str, dict[str, list[int]]] = {
        name: {path: [] for path in all_builder_paths(skeleton)}
        for name, skeleton in builders.items()
    }
    if args.capacity_policy == "compatible-path-quantile":
        for name, compatible_rows in compatible_by_builder.items():
            for row in compatible_rows:
                incidence = row["incidence"]
                for path in compatible_counts[name]:
                    requirement = incidence["node_requirements"].get(path, audit.zero_requirement())
                    compatible_counts[name][path].append(sum(int(value) for value in requirement.values()))
    audit.append_event(
        progress,
        "external_capacity_statistic",
        quantile=args.global_capacity_quantile,
        active_nodes=len(active_node_counts),
        capacity_floor=global_floor,
    )

    rules: dict[str, dict[str, dict[str, Any]]] = {}
    builder_payloads: list[dict[str, Any]] = []
    for source in source_builders:
        name = str(source["builder"])
        rows = by_builder[name]
        if args.prior_population == "compatible":
            prior_rows = compatible_by_builder[name]
        elif args.prior_population == "compatible-closest":
            minimum_slack = min(
                int(row["structural_slack"])
                for row in compatible_by_builder[name]
            )
            prior_rows = [
                row for row in compatible_by_builder[name]
                if int(row["structural_slack"]) == minimum_slack
            ]
        else:
            prior_rows = rows
        node_by_path = builder_nodes_by_path(builders[name])
        rules[name] = {}
        activation_priors: dict[str, dict[str, Any]] = {}
        for path in all_builder_paths(builders[name]):
            capacity_counts = [
                sum(int(value) for value in row["incidence"]["node_requirements"].get(path, audit.zero_requirement()).values())
                for row in rows
            ]
            observed_counts = [
                sum(int(value) for value in row["incidence"]["node_requirements"].get(path, audit.zero_requirement()).values())
                for row in prior_rows
            ]
            maximum = max(capacity_counts, default=0)
            assigned_core = percentile_integer(
                capacity_counts, args.assigned_core_quantile
            )
            observed_counts_by_dimension: dict[str, list[int]] = defaultdict(list)
            for row, count in zip(rows, capacity_counts):
                observed_counts_by_dimension[str(int(row["input_dimension"]))].append(count)
            path_maximum_by_dimension = {
                dimension: max(values)
                for dimension, values in sorted(observed_counts_by_dimension.items(), key=lambda item: int(item[0]))
            }
            compatible_quantile = percentile_integer(
                compatible_counts[name][path], args.global_capacity_quantile
            )
            path_capacity = (
                0
                if args.capacity_policy == "dimension-adaptive" else
                max(assigned_core, archetype_floor.get(node_archetype(node_by_path[path]), 0))
                if args.capacity_policy == "archetype-floor" else
                max(assigned_core, compatible_quantile)
                if args.capacity_policy == "compatible-path-quantile" else
                max(
                    global_floor
                    if args.capacity_policy == "global-floor" or node_by_path[path].raw_attachment
                    else 0,
                    assigned_core,
                )
            )
            rules[name][path] = {
                "mode": "input_role_lanes",
                "formula": (
                    ("external compatible-path quantile with assigned-path maximum"
                     if args.capacity_policy == "compatible-path-quantile" else
                     "global active-node quantile with assigned-path maximum")
                ),
                "external_global_active_node_p85": global_floor,
                "external_path_maximum": maximum,
                "external_assigned_core_quantile": args.assigned_core_quantile,
                "external_assigned_core_capacity": assigned_core,
                "external_compatible_path_quantile": compatible_quantile,
                "external_path_capacity": path_capacity,
                "external_archetype": node_archetype(node_by_path[path]),
                "external_archetype_quantile": archetype_floor.get(node_archetype(node_by_path[path]), 0),
                "capacity_policy": args.capacity_policy,
                "external_global_quantile_by_input_dimension": global_floor_by_dimension,
                "external_path_maximum_by_input_dimension": path_maximum_by_dimension,
                "cap_by_input_dimension": not args.ignore_input_dimension_cap,
                "sparse_neutral_gate": True,
            }
            raw_routes = [
                port["route"]
                for row in prior_rows
                for port in row["incidence"]["canonical_raw_ports"]
                if port["path"] == path
            ]
            activation_priors[path] = {
                "external_required_lane_histogram": {
                    str(value): observed_counts.count(value)
                    for value in sorted(set(observed_counts))
                },
                # This is a builder-level external prior, not an equation
                # materialization.  The runtime applies the real input-
                # dimension and path-capacity cap.  Using global_floor here as
                # a fake dimension forced every prior to <= 1 when the chosen
                # global quantile was zero.
                "initial_active_lanes": percentile_integer(observed_counts, 0.50),
                "unary_route_probability_given_active": (
                    raw_routes.count("unary") / len(raw_routes)
                    if raw_routes else 0.0
                ),
                "population": args.prior_population,
                "population_equations": len(prior_rows),
            }
        if args.capacity_policy == "shared-pool":
            pool_counts = [int(row["incidence"]["raw_port_count"]) for row in rows]
            rules[name]["__shared_pool__"] = {
                "mode": "shared_routable_input_role_pool",
                "capacity": max(pool_counts, default=1),
                "external_required_pool_histogram": {
                    str(value): pool_counts.count(value) for value in sorted(set(pool_counts))
                },
                "initial_active_lanes": percentile_integer(pool_counts, 0.50),
            }
        elif args.capacity_policy == "core-plus-shared-reserve":
            core_by_path: dict[str, int] = {}
            for path in all_builder_paths(builders[name]):
                counts = [
                    sum(int(value) for value in row["incidence"]["node_requirements"].get(
                        path, audit.zero_requirement()
                    ).values())
                    for row in rows
                ]
                core_by_path[path] = percentile_integer(counts, args.global_capacity_quantile)
            deficits = []
            for row in rows:
                deficit = 0
                for path, requirement in row["incidence"]["node_requirements"].items():
                    required = sum(int(value) for value in requirement.values())
                    deficit += max(0, required - core_by_path.get(path, 0))
                deficits.append(deficit)
            rules[name]["__shared_pool__"] = {
                "mode": "core_capacity_plus_shared_routable_reserve",
                "core_capacity_by_node": core_by_path,
                "reserve_capacity": max(deficits, default=0),
                "external_required_reserve_histogram": {
                    str(value): deficits.count(value) for value in sorted(set(deficits))
                },
            }
        elif args.capacity_policy == "raw-sites-floor-plus-reserve":
            rules[name]["__shared_pool__"] = {
                "mode": "raw_site_core_plus_shared_routable_reserve",
                "core_capacity_by_node": {
                    path: int(rule["external_path_capacity"])
                    for path, rule in rules[name].items()
                    if path != "__shared_pool__"
                },
                "reserve_capacity": int(args.shared_reserve_capacity),
            }
        elif args.capacity_policy == "compatible-shared-reserve":
            core_by_path = {
                path: int(rule["external_path_capacity"])
                for path, rule in rules[name].items()
                if path != "__shared_pool__"
            }
            compatible_deficits: list[int] = []
            for compatible_row in compatible_by_builder[name]:
                incidence = compatible_row["incidence"]
                dimension = int(compatible_row["input_dimension"])
                deficit = 0
                for path, requirement in incidence["node_requirements"].items():
                    required = sum(int(value) for value in requirement.values())
                    core = min(dimension, core_by_path.get(path, 0))
                    deficit += max(0, required - core)
                compatible_deficits.append(deficit)
            rules[name]["__shared_pool__"] = {
                "mode": "external_compatible_deficit_shared_reserve",
                "core_capacity_by_node": core_by_path,
                "reserve_capacity": percentile_integer(
                    compatible_deficits, args.reserve_quantile
                ),
                "reserve_quantile": args.reserve_quantile,
                "external_compatible_deficit_histogram": {
                    str(value): compatible_deficits.count(value)
                    for value in sorted(set(compatible_deficits))
                },
            }
        medoid_profile, medoid_quality = medoid_role_profile(
            row["incidence"] for row in prior_rows
        )
        representative_incidence = next(
            row["incidence"]
            for row in prior_rows
            if global_role_incidence_profile(row["incidence"]) == medoid_profile
        )
        state_route_priors: dict[str, dict[str, Any]] = {}
        edge_paths = all_builder_state_paths(builders[name])
        for path in edge_paths:
            observed = [
                row["incidence"]["state_routes"].get(path)
                for row in prior_rows
            ]
            used = [value for value in observed if value is not None]
            state_route_priors[path] = {
                "used_equations": len(used),
                "unused_equations": observed.count(None),
                "unary_route_probability_given_used": (
                    used.count("unary") / len(used) if used else 0.0
                ),
                "population": args.prior_population,
                "medoid_initially_used": (
                    path in representative_incidence["state_routes"]
                ),
            }
        prefix_profile = prefix_role_profile(activation_priors)
        compatible_profiles = [
            global_role_incidence_profile(row["incidence"])
            for row in compatible_by_builder[name]
        ]
        prefix_compatible_distance = (
            sum(role_profile_distance(prefix_profile, profile) for profile in compatible_profiles)
            / len(compatible_profiles)
            if compatible_profiles else 0.0
        )
        medoid_compatible_distance = (
            sum(role_profile_distance(medoid_profile, profile) for profile in compatible_profiles)
            / len(compatible_profiles)
            if compatible_profiles else 0.0
        )
        role_incidence_prior = {
            "mode": args.role_incidence_prior,
            "population": args.prior_population,
            "lanes": [
                {"lane": lane, "active_paths": list(paths)}
                for lane, paths in enumerate(medoid_profile)
            ] if args.role_incidence_prior == "compatible-medoid" else [],
            "medoid_quality": medoid_quality,
            "state_usage_source": (
                "same observed medoid incidence"
                if args.role_incidence_prior == "compatible-medoid"
                else "independent state-route majority"
            ),
            "mean_compatible_distance_independent_prefix": prefix_compatible_distance,
            "mean_compatible_distance_medoid": medoid_compatible_distance,
        }
        shared_pool = rules[name].get("__shared_pool__")
        if shared_pool is not None and args.role_incidence_prior == "compatible-medoid":
            core_by_path = shared_pool.get("core_capacity_by_node", {})
            existing_role_capacity = max(
                max((int(value) for value in core_by_path.values()), default=0)
                + int(shared_pool.get("reserve_capacity", 0)),
                int(shared_pool.get("capacity", 0)),
            )
            shared_pool["global_role_capacity"] = max(
                existing_role_capacity,
                len(medoid_profile),
            )
            shared_pool["global_role_capacity_source"] = (
                "maximum of existing lane capacity and external role-incidence medoid"
            )
        compatible_activation_errors: list[int] = []
        shared_pool = rules[name].get("__shared_pool__", {})
        core_by_path = shared_pool.get("core_capacity_by_node")
        for compatible_row in compatible_by_builder[name]:
            dimension = int(compatible_row["input_dimension"])
            incidence = compatible_row["incidence"]
            for path in all_builder_paths(builders[name]):
                required = sum(int(value) for value in incidence["node_requirements"].get(
                    path, audit.zero_requirement()
                ).values())
                if core_by_path is not None:
                    capacity = min(dimension, int(core_by_path.get(path, 0)))
                else:
                    capacity = materialized_capacity(rules[name][path], dimension)
                initial = min(
                    capacity,
                    int(activation_priors[path]["initial_active_lanes"]),
                )
                compatible_activation_errors.append(abs(required - initial))
        builder_payloads.append({
            "builder": name.replace("ACB", "IAPB"),
            "source_boolean_builder": name,
            "synthetic_operator_tree": source["synthetic_operator_tree"],
            "operator_pattern": source["operator_pattern"],
            "raw_incidence_rule": {
                "port_capacity_by_node": rules[name],
                "role_source": "trainable affine projection of all observed inputs",
                "role_reuse": "lane index identifies one affine role shared across arithmetic nodes",
                "neutral_gate": "0 for sum lane; 1 for product lane",
                "activation_prior_by_node": activation_priors,
                "role_incidence_prior": role_incidence_prior,
            },
            "edge_route_rule": {
                "mode": "KAN native with affine base; zero nonlinear coefficients represent affine-only",
                "constant_and_affine_rescaling_in_structure": False,
                "state_route_prior": state_route_priors,
            },
            "fits_per_equation": 1,
            "prior_population": args.prior_population,
            "role_incidence_prior": args.role_incidence_prior,
            "prior_quality": {
                "compatible_equations": len(compatible_by_builder[name]),
                "compatible_node_observations": len(compatible_activation_errors),
                "mean_initial_active_count_absolute_error": (
                    sum(compatible_activation_errors) / len(compatible_activation_errors)
                    if compatible_activation_errors else 0.0
                ),
            },
            "input_dimension_changes_materialized_width": (
                not args.ignore_input_dimension_cap
            ),
            "external_archetype_quantiles": archetype_floor,
        })
    audit.write_jsonl(args.output / "builders.jsonl", builder_payloads)

    development_rows, development = evaluate(
        external_records, builders, rules, args.structural_score,
        args.raw_arity_policy,
        args.collapse_univariate_chains,
    )
    # Frozen evaluation is intentionally performed only after the builder
    # payloads and external-derived capacity statistic have been written.
    evaluation_rows, evaluation = evaluate(
        feynman_records, builders, rules, args.structural_score,
        args.raw_arity_policy,
        args.collapse_univariate_chains,
    )
    audit.write_jsonl(args.output / "development_assignments.jsonl", development_rows)
    audit.write_jsonl(args.output / "evaluation_assignments.jsonl", evaluation_rows)
    report = {
        "method": "fixed ACU skeleton plus parametric incidence/reuse/unary placement",
        "builder_count": len(builder_payloads),
        "fits_per_builder_per_equation": 1,
        "fits_per_equation": len(builder_payloads),
        "prior_population": args.prior_population,
        "role_incidence_prior": args.role_incidence_prior,
        "structural_score": args.structural_score,
        "raw_arity_policy": args.raw_arity_policy,
        "collapse_univariate_chains": args.collapse_univariate_chains,
        "prior_quality": {
            "compatible_node_observations": sum(
                int(row["prior_quality"]["compatible_node_observations"])
                for row in builder_payloads
            ),
            "mean_initial_active_count_absolute_error": (
                sum(
                    float(row["prior_quality"]["mean_initial_active_count_absolute_error"])
                    * int(row["prior_quality"]["compatible_node_observations"])
                    for row in builder_payloads
                )
                / max(1, sum(
                    int(row["prior_quality"]["compatible_node_observations"])
                    for row in builder_payloads
                ))
            ),
        },
        "capacity_rule": {
            "policy": args.capacity_policy,
            "external_quantile": args.global_capacity_quantile,
            "assigned_core_quantile": args.assigned_core_quantile,
            "external_global_active_node_p85": global_floor,
            "shared_reserve_capacity": (
                args.shared_reserve_capacity
                if args.capacity_policy == "raw-sites-floor-plus-reserve" else None
            ),
            "reserve_quantile": (
                args.reserve_quantile
                if args.capacity_policy == "compatible-shared-reserve" else None
            ),
            "cap_by_input_dimension": not args.ignore_input_dimension_cap,
            "input_dimension_changes_materialized_width": (
                not args.ignore_input_dimension_cap
            ),
        },
        "development": development,
        "evaluation": evaluation,
        "accuracy_target": audit.failure_baseline(audit.DEFAULT_FAILURE_AUDIT),
        "builders": builder_payloads,
    }
    (args.output / "catalogue.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "REPORT.md").write_text(render_report(report), encoding="utf-8")
    audit.append_event(progress, "completed", development=development, evaluation=evaluation)


if __name__ == "__main__":
    main()
