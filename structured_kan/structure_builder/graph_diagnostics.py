from __future__ import annotations
from collections import defaultdict
from typing import Any,Mapping
import networkx as nx
NODE_MATCH=nx.algorithms.isomorphism.categorical_node_match("color","")
def add_arithmetic_nodes(graph: nx.DiGraph, spec: Mapping[str, Any]) -> None:
    for node in spec.get("nodes", ()):
        node_id = str(node["id"])
        graph.add_node(node_id, color=f"op:{node['operator']}", category="arithmetic")
    graph.add_node("output", color="output", category="output")


def add_output_edges(graph: nx.DiGraph, spec: Mapping[str, Any]) -> None:
    """Connect outputs, including direct raw/unary outputs with no gather."""
    for source in spec.get("output_sources", ()):
        source_id = str(source)
        if source_id not in graph:
            graph.add_node(source_id, color="raw-output", category="raw-source")
        graph.add_edge(source_id, "output")


def rooted_core_graph(spec: Mapping[str, Any]) -> nx.DiGraph:
    """Contract state-link subdivisions while retaining one raw pool per node."""
    graph = nx.DiGraph(view="rooted-core")
    add_arithmetic_nodes(graph, spec)
    for node in spec.get("nodes", ()):
        target = str(node["id"])
        if node.get("raw_sources"):
            raw_id = f"{target}:raw-pool"
            graph.add_node(raw_id, color="raw-pool", category="raw-source")
            graph.add_edge(raw_id, target)
        for source in node.get("state_sources", ()):
            graph.add_edge(str(source["state"]), target)
    add_output_edges(graph, spec)
    return graph


def exact_isomorphism_classes(graphs: list[nx.DiGraph]) -> tuple[list[int], list[int]]:
    """WL pre-bucketing followed by exact colored-DAG isomorphism."""
    buckets: dict[str, list[int]] = defaultdict(list)
    for index, graph in enumerate(graphs):
        digest = nx.weisfeiler_lehman_graph_hash(graph, node_attr="color", iterations=6)
        buckets[digest].append(index)
    labels = [-1] * len(graphs)
    representatives: list[int] = []
    for bucket in buckets.values():
        local_classes: list[int] = []
        for graph_index in bucket:
            for class_index in local_classes:
                representative = representatives[class_index]
                if nx.is_isomorphic(graphs[graph_index], graphs[representative], node_match=NODE_MATCH):
                    labels[graph_index] = class_index
                    break
            else:
                labels[graph_index] = len(representatives)
                representatives.append(graph_index)
                local_classes.append(labels[graph_index])
    return labels, representatives
