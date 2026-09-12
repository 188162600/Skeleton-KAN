"""Canonical executable-edge projection, relocated without rule changes."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
import json

@dataclass(frozen=True)
class Term:
    operator: str
    raw_attachment: bool
    children: tuple["Term", ...]
    # Number of independent scalar role routes attached at this node.  Older
    # payloads stored only a Boolean attachment, so zero remains a backwards-
    # compatible spelling for one route whenever raw_attachment is true.
    raw_arity: int = 0

    def __post_init__(self) -> None:
        normalized_raw_arity = max(1, int(self.raw_arity)) if self.raw_attachment else 0
        if normalized_raw_arity != self.raw_arity:
            object.__setattr__(self, "raw_arity", normalized_raw_arity)
        # A one-input sum and a one-input product are the same unary map.  Use
        # one canonical colour so exact classification does not split them.
        if self.arity == 1 and self.operator != "+":
            object.__setattr__(self, "operator", "+")

    @property
    def arity(self) -> int:
        return self.raw_arity + len(self.children)

    @property
    def nodes(self) -> int:
        return 1 + sum(child.nodes for child in self.children)

    @property
    def edges(self) -> int:
        return len(self.children) + sum(child.edges for child in self.children)

    def payload(self) -> dict[str, Any]:
        payload = {
            "operator": self.operator,
            "raw_attachment": self.raw_attachment,
            "children": [child.payload() for child in self.children],
        }
        # Keep historical signatures unchanged for the ordinary Boolean case.
        if self.raw_arity > 1:
            payload["raw_arity"] = self.raw_arity
        return payload

    def signature(self) -> str:
        # Route multiplicity is part of the executable interaction structure.
        # Omitting it made sum_phi([R]) and prod_phi([R,R]) the same class and
        # produced false exact coverage and zero-cost M/U matches.
        topology = {
            "operator": self.operator,
            "raw_attachment": self.raw_attachment,
            "raw_arity": self.raw_arity,
            "children": [json.loads(child.signature()) for child in self.children],
        }
        return json.dumps(topology, sort_keys=True, separators=(",", ":"))

    def notation(self, raw_symbol: str = "X") -> str:
        """Render the executable one-list notation used in the manuscript.

        ``sum`` and ``prod`` remain word operators.  The Greek ``phi`` suffix
        marks the learned edge-map family, and every node receives exactly one
        list. ``sum(X)`` denotes one reusable affine input-role projection.
        Multiple independent roles are written as repeated ``sum(X)`` entries;
        they remain separate inputs to separate receiving edge maps.
        """
        operator_words = {
            "+": "sum",
            "sum": "sum",
            "sum_direct": "sum",
            "sum_phi": "sum",
            "*": "prod",
            "product": "prod",
            "product_direct": "prod",
            "product_phi": "prod",
        }
        try:
            operator = operator_words[self.operator]
        except KeyError as error:
            raise ValueError(f"unsupported catalogue operator {self.operator!r}") from error
        child_arguments = [child.notation(raw_symbol) for child in self.children]
        shared_roles = (
            [f"sum({raw_symbol})"] * self.raw_arity if self.raw_attachment else []
        )
        arguments = shared_roles + child_arguments
        return f"{operator}_φ([{','.join(arguments)}])"


def is_identity_edge(edge: dict[str, Any]) -> bool:
    return (
        edge.get("primitive") == "x"
        and abs(float(edge.get("weight", 1.0)) - 1.0) < 1e-12
        and abs(float(edge.get("bias", 0.0))) < 1e-12
        and abs(float(edge.get("input_scale", 1.0)) - 1.0) < 1e-12
        and abs(float(edge.get("input_bias", 0.0))) < 1e-12
    )


def term_from_spec(
    spec: dict[str, Any], *, collapse_univariate_chains: bool = True
) -> Term:
    """Project an operator-variable specification to the catalogue grammar.

    The executable projection regards a maximal one-input computation as one
    learned univariate edge map.  ``collapse_univariate_chains=False`` is kept
    only for auditing superseded legacy artifacts.  The canonical projection:

    * constant-only padding records do not create an ``X`` attachment; and
    * an arithmetic node with one constructed input and no ``X`` attachment is
      contracted into that input.

    The returned :class:`Term`, its notation, catalogue synthesis, embedding,
    and Hungarian routines are otherwise the original analyser's code paths.
    """
    by_id = {str(node["id"]): node for node in spec.get("nodes", ())}
    output_sources = [str(source) for source in spec.get("output_sources", ())]
    if len(output_sources) != 1:
        raise ValueError("expected one arithmetic output source")

    def source_token(source: dict[str, Any], node_id: str, index: int) -> tuple[Any, ...] | None:
        indices = tuple(int(value) for value in source.get("variable_indices", ()))
        if not indices:
            return None
        if str(source.get("source_kind", "raw")) == "raw" and len(indices) == 1:
            return ("raw", indices[0])
        return (
            "affine",
            indices,
            tuple(round(float(value), 12) for value in source.get("coefficients", ())),
            round(float(source.get("constant", 0.0)), 12),
        )

    @lru_cache(maxsize=None)
    def visit(
        node_id: str,
    ) -> tuple[
        str,
        Term,
        frozenset[tuple[Any, ...]],
        frozenset[tuple[Any, ...]],
        str,
    ]:
        # A source such as ``i0`` denotes a direct raw/unary output with no
        # explicit arithmetic gather.  In the uniform one-list grammar this is
        # the one-input additive form sum_φ([sum(X)]), so no separate identity
        # operator is needed.
        if node_id not in by_id:
            token = ("direct", node_id)
            tokens = frozenset((token,))
            return "+", Term("+", True, (), raw_arity=1), tokens, tokens, "identity"
        node = by_id[node_id]
        operator = str(node["operator"])
        raw_sources = tuple(node.get("raw_sources", ()))
        direct_tokens = {
            token
            for index, source in enumerate(raw_sources)
            if (token := source_token(source, node_id, index)) is not None
        }
        attached_tokens = set(direct_tokens)
        dependency_tokens = set(direct_tokens)
        raw_attachment = bool(direct_tokens) if collapse_univariate_chains else bool(raw_sources)
        children: list[Term] = []
        only_retained_attached: frozenset[tuple[Any, ...]] = frozenset()
        only_retained_route = "identity"
        for source in node.get("state_sources", ()):
            (
                child_source_operator,
                child,
                child_dependencies,
                child_attached,
                child_post_route,
            ) = visit(str(source["state"]))
            dependency_tokens.update(child_dependencies)
            # A whole state depending on only one scalar coordinate is already
            # a univariate function.  The destination sum_phi/prod_phi edge can
            # learn that complete map, so inserting a child sum_phi node is an
            # executable-topology error.
            if collapse_univariate_chains and len(child_dependencies) <= 1:
                attached_tokens.update(child_dependencies)
                raw_attachment = bool(attached_tokens)
                continue
            source_edge = dict(source.get("edge", {}))
            route = (
                "unary"
                if child_post_route == "unary"
                or (
                    source_edge.get("primitive", "x") != "x"
                    if collapse_univariate_chains
                    else not is_identity_edge(source_edge)
                )
                else "identity"
            )
            if child_source_operator == operator and route == "identity":
                raw_attachment = raw_attachment or child.raw_attachment
                if collapse_univariate_chains:
                    attached_tokens.update(child_attached)
                children.extend(child.children)
            else:
                children.append(child)
                only_retained_attached = child_attached
                only_retained_route = route
        children.sort(key=Term.signature)
        if collapse_univariate_chains and not raw_attachment and len(children) == 1:
            child = children[0]
            # The current arithmetic node disappears from the projected term.
            # Its ancestors must therefore see the retained child's colour,
            # not the colour of this contracted-away node; otherwise an
            # identity edge above can trigger a false AC flattening decision.
            return (
                child.operator,
                child,
                frozenset(dependency_tokens),
                only_retained_attached,
                only_retained_route,
            )
        raw_arity = len(attached_tokens) if collapse_univariate_chains else 0
        value = Term(operator, raw_attachment, tuple(children), raw_arity=raw_arity)
        return (
            operator,
            value,
            frozenset(dependency_tokens),
            frozenset(attached_tokens),
            "identity",
        )

    return visit(output_sources[0])[1]

