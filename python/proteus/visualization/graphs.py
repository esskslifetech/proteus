"""
Network/graph visualizations for pathways, PPIs, and gene regulatory networks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from proteus.schema import SimulationReport, VariantEvent

COLORS = {
    "viable": "#2ecc71",
    "non_viable": "#e74c3c",
    "uncertain": "#f39c12",
    "reference": "#3498db",
    "mutated": "#e67e22",
    "reverted": "#9b59b6",
    "background": "#fafafa",
    "text": "#2c3e50",
    "edge": "#bdc3c7",
    "highlight": "#e74c3c",
}


def _get_pathway_data(report: SimulationReport) -> dict[str, Any]:
    """Extract pathway data from L3 layer results."""
    if hasattr(report, "layer_results") and report.layer_results:
        l3 = report.layer_results.get("L3_pathway", {})
        if isinstance(l3, dict):
            return l3
    return {}


def _get_ppi_data(report: SimulationReport) -> dict[str, Any]:
    """Extract PPI data from L3 layer results."""
    if hasattr(report, "layer_results") and report.layer_results:
        l3 = report.layer_results.get("L3_pathway", {})
        if isinstance(l3, dict) and "ppi_network" in l3:
            return l3["ppi_network"]
    return {}


def _get_grn_data(report: SimulationReport) -> dict[str, Any]:
    """Extract gene regulatory network from L4 layer results."""
    if hasattr(report, "layer_results") and report.layer_results:
        l4 = report.layer_results.get("L4_tissue", {})
        if isinstance(l4, dict) and "grn" in l4:
            return l4["grn"]
    return {}


def plot_pathway_graph(
    report: SimulationReport,
    output_path: Optional[Path] = None,
    figsize: tuple[int, int] = (14, 10),
    layout: str = "spring",
    max_nodes: int = 100,
) -> plt.Figure:
    """Visualize metabolic/signaling pathway as a directed graph."""
    pathway_data = _get_pathway_data(report)

    # Build graph
    G = nx.DiGraph()

    if pathway_data and "nodes" in pathway_data and "edges" in pathway_data:
        nodes = pathway_data["nodes"][:max_nodes]
        edges = pathway_data["edges"][:max_nodes * 2]

        for node in nodes:
            if isinstance(node, dict):
                G.add_node(node["id"], **node)
            else:
                G.add_node(str(node))

        for edge in edges:
            if isinstance(edge, dict):
                G.add_edge(edge["source"], edge["target"], **edge)
            elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                G.add_edge(str(edge[0]), str(edge[1]))

    else:
        # Fallback: create a demo pathway graph
        _add_demo_pathway(G)

    fig, ax = plt.subplots(figsize=figsize)

    # Layout
    if layout == "spring":
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "shell":
        pos = nx.shell_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    # Node colors by type / mutation impact
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        data = G.nodes[node]
        ntype = data.get("type", "gene")
        impact = data.get("mutation_impact", 0)

        if ntype == "metabolite":
            base_color = COLORS["reference"]
        elif ntype == "enzyme":
            base_color = COLORS["mutated"]
        elif ntype == "transcription_factor":
            base_color = COLORS["reverted"]
        else:
            base_color = COLORS["text"]

        # Blend with impact color
        if impact > 0.5:
            node_colors.append(COLORS["highlight"])
            node_sizes.append(500 + impact * 500)
        elif impact > 0.2:
            node_colors.append(COLORS["uncertain"])
            node_sizes.append(300 + impact * 300)
        else:
            node_colors.append(base_color)
            node_sizes.append(300)

    # Edge widths by flux / weight
    edge_weights = [G[u][v].get("flux", 1) for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    edge_widths = [0.5 + 3 * w / max_w for w in edge_weights]

    # Draw
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color=COLORS["edge"],
                          arrows=True, arrowsize=15, arrowstyle="->", ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                          edgecolors="white", linewidths=1.5, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax)

    # Highlight mutated nodes
    mutated_nodes = [n for n in G.nodes() if G.nodes[n].get("mutated", False)]
    if mutated_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=mutated_nodes, node_color=COLORS["highlight"],
                              node_size=[node_sizes[list(G.nodes()).index(n)] for n in mutated_nodes],
                              edgecolors="black", linewidths=2, ax=ax)

    ax.set_title(f"Pathway Network — {report.simulation_id}", fontsize=16, pad=20)
    ax.axis("off")

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=COLORS["reference"], label="Metabolite", edgecolor="white"),
        Patch(facecolor=COLORS["mutated"], label="Enzyme", edgecolor="white"),
        Patch(facecolor=COLORS["reverted"], label="TF", edgecolor="white"),
        Patch(facecolor=COLORS["highlight"], label="High Mutation Impact", edgecolor="black"),
        Line2D([0], [0], color=COLORS["edge"], lw=2, label="Flux (width ∝ value)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=COLORS["background"])
    return fig


def _add_demo_pathway(G: nx.DiGraph):
    """Add a demo glycolysis-like pathway."""
    nodes = [
        ("Glucose", {"type": "metabolite"}),
        ("Hexokinase", {"type": "enzyme"}),
        ("G6P", {"type": "metabolite"}),
        ("PFK", {"type": "enzyme"}),
        ("F1,6BP", {"type": "metabolite"}),
        ("Aldolase", {"type": "enzyme"}),
        ("GAP", {"type": "metabolite"}),
        ("GAPDH", {"type": "enzyme"}),
        ("1,3BPG", {"type": "metabolite"}),
        ("PGK", {"type": "enzyme"}),
        ("3PG", {"type": "metabolite"}),
        ("PK", {"type": "enzyme"}),
        ("Pyruvate", {"type": "metabolite"}),
        ("LDH", {"type": "enzyme"}),
        ("Lactate", {"type": "metabolite"}),
    ]
    for n, d in nodes:
        G.add_node(n, **d)

    edges = [
        ("Glucose", "Hexokinase", {"flux": 10}),
        ("Hexokinase", "G6P", {"flux": 10}),
        ("G6P", "PFK", {"flux": 9}),
        ("PFK", "F1,6BP", {"flux": 9}),
        ("F1,6BP", "Aldolase", {"flux": 9}),
        ("Aldolase", "GAP", {"flux": 9}),
        ("GAP", "GAPDH", {"flux": 8}),
        ("GAPDH", "1,3BPG", {"flux": 8}),
        ("1,3BPG", "PGK", {"flux": 8}),
        ("PGK", "3PG", {"flux": 8}),
        ("3PG", "PK", {"flux": 7}),
        ("PK", "Pyruvate", {"flux": 7}),
        ("Pyruvate", "LDH", {"flux": 6}),
        ("LDH", "Lactate", {"flux": 6}),
    ]
    for s, t, d in edges:
        G.add_edge(s, t, **d)


def plot_ppi_network(
    report: SimulationReport,
    output_path: Optional[Path] = None,
    figsize: tuple[int, int] = (14, 10),
    layout: str = "spring",
    max_nodes: int = 150,
) -> plt.Figure:
    """Protein-Protein Interaction network with mutation highlights."""
    ppi_data = _get_ppi_data(report)

    G = nx.Graph()

    if ppi_data and "nodes" in ppi_data and "edges" in ppi_data:
        nodes = ppi_data["nodes"][:max_nodes]
        edges = ppi_data["edges"][:max_nodes * 3]

        for node in nodes:
            if isinstance(node, dict):
                G.add_node(node["id"], **node)
            else:
                G.add_node(str(node))

        for edge in edges:
            if isinstance(edge, dict):
                G.add_edge(edge["source"], edge["target"], **edge)
            elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                G.add_edge(str(edge[0]), str(edge[1]))

    else:
        _add_demo_ppi(G)

    fig, ax = plt.subplots(figsize=figsize)

    if layout == "spring":
        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "fruchterman_reingold":
        pos = nx.fruchterman_reingold_layout(G, seed=42)
    else:
        pos = nx.spring_layout(G, seed=42)

    # Community detection for coloring
    try:
        communities = nx.community.greedy_modularity_communities(G)
        community_map = {}
        for i, comm in enumerate(communities):
            for node in comm:
                community_map[node] = i
    except Exception:
        community_map = {n: 0 for n in G.nodes()}

    # Node attributes
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        data = G.nodes[node]
        impact = data.get("mutation_impact", 0)
        is_mutated = data.get("mutated", False)

        if is_mutated:
            node_colors.append(COLORS["highlight"])
            node_sizes.append(600)
        elif impact > 0.3:
            node_colors.append(COLORS["uncertain"])
            node_sizes.append(400)
        else:
            # Color by community
            cmap = plt.cm.Set3
            node_colors.append(cmap(community_map.get(node, 0) % 12))
            node_sizes.append(200 + G.degree(node) * 20)

    # Edge weights by confidence
    edge_weights = [G[u][v].get("confidence", 0.5) for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    edge_widths = [0.3 + 2 * w / max_w for w in edge_weights]

    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.4, edge_color=COLORS["edge"], ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                          edgecolors="white", linewidths=1, ax=ax)

    # Labels for high-degree / mutated nodes only
    labels = {n: n for n in G.nodes() if G.degree(n) > 5 or G.nodes[n].get("mutated", False)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, font_weight="bold", ax=ax)

    ax.set_title(f"PPI Network — {report.simulation_id}", fontsize=16, pad=20)
    ax.axis("off")

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=COLORS["highlight"], label="Mutated Protein", edgecolor="black"),
        Patch(facecolor=COLORS["uncertain"], label="High Impact", edgecolor="white"),
        Patch(facecolor=plt.cm.Set3(0), label="Community 1", edgecolor="white"),
        Patch(facecolor=plt.cm.Set3(1), label="Community 2", edgecolor="white"),
        Line2D([0], [0], color=COLORS["edge"], lw=2, label="Interaction (width ∝ confidence)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=COLORS["background"])
    return fig


def _add_demo_ppi(G: nx.Graph):
    """Add a demo PPI network."""
    import random
    random.seed(42)
    np.random.seed(42)

    # Create a scale-free-like network
    proteins = [f"PROT_{i:03d}" for i in range(50)]
    for p in proteins:
        G.add_node(p, mutated=(p in ["PROT_005", "PROT_012", "PROT_028"]),
                  mutation_impact=np.random.beta(0.5, 5))

    # Preferential attachment
    for i, p in enumerate(proteins):
        if i > 0:
            targets = np.random.choice(proteins[:i], size=min(3, i), replace=False,
                                     p=np.array([G.degree(q)+1 for q in proteins[:i]]) / sum(G.degree(q)+1 for q in proteins[:i]))
            for t in targets:
                G.add_edge(p, t, confidence=np.random.beta(2, 2))


def plot_gene_regulatory_network(
    report: SimulationReport,
    output_path: Optional[Path] = None,
    figsize: tuple[int, int] = (14, 10),
    layout: str = "hierarchical",
) -> plt.Figure:
    """Gene Regulatory Network (TF → gene) with expression changes."""
    grn_data = _get_grn_data(report)

    G = nx.DiGraph()

    if grn_data and "nodes" in grn_data and "edges" in grn_data:
        for node in grn_data["nodes"]:
            if isinstance(node, dict):
                G.add_node(node["id"], **node)
            else:
                G.add_node(str(node))

        for edge in grn_data["edges"]:
            if isinstance(edge, dict):
                G.add_edge(edge["source"], edge["target"], **edge)
            elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                G.add_edge(str(edge[0]), str(edge[1]))

    else:
        _add_demo_grn(G)

    fig, ax = plt.subplots(figsize=figsize)

    # Hierarchical layout (TFs on top)
    if layout == "hierarchical":
        pos = _hierarchical_layout(G)
    elif layout == "spring":
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    else:
        pos = nx.spring_layout(G, seed=42)

    # Node colors by type and expression change
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        data = G.nodes[node]
        ntype = data.get("type", "gene")
        log2fc = data.get("log2fc", 0)

        if ntype == "transcription_factor":
            base_color = COLORS["reverted"]
            size = 500
        else:
            base_color = COLORS["reference"]
            size = 300

        if log2fc > 1:
            node_colors.append(COLORS["mutated"])
            node_sizes.append(size * 1.5)
        elif log2fc < -1:
            node_colors.append(COLORS["viable"])
            node_sizes.append(size * 1.2)
        else:
            node_colors.append(base_color)
            node_sizes.append(size)

    # Edge weights by regulatory strength
    edge_weights = [abs(G[u][v].get("weight", 1)) for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1
    edge_widths = [0.5 + 3 * w / max_w for w in edge_weights]

    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color=COLORS["edge"],
                          arrows=True, arrowsize=18, arrowstyle="->", ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                          edgecolors="white", linewidths=1.5, ax=ax)

    # Labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax)

    ax.set_title(f"Gene Regulatory Network — {report.simulation_id}", fontsize=16, pad=20)
    ax.axis("off")

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=COLORS["reverted"], label="Transcription Factor", edgecolor="white"),
        Patch(facecolor=COLORS["reference"], label="Target Gene", edgecolor="white"),
        Patch(facecolor=COLORS["mutated"], label="Upregulated (log2FC > 1)", edgecolor="white"),
        Patch(facecolor=COLORS["viable"], label="Downregulated (log2FC < -1)", edgecolor="white"),
        Line2D([0], [0], color=COLORS["edge"], lw=2, label="Regulation (width ∝ strength)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=COLORS["background"])
    return fig


def _hierarchical_layout(G: nx.DiGraph) -> dict:
    """Hierarchical layout for GRN: TFs on top, targets below."""
    tfs = [n for n in G.nodes() if G.nodes[n].get("type") == "transcription_factor"]
    targets = [n for n in G.nodes() if n not in tfs]

    pos = {}
    # TFs on top row
    for i, tf in enumerate(tfs):
        pos[tf] = (i * 2 - len(tfs), 2)

    # Targets on bottom row
    for i, t in enumerate(targets):
        pos[t] = (i * 1.5 - len(targets) * 0.75, 0)

    return pos


def _add_demo_grn(G: nx.DiGraph):
    """Add a demo GRN."""
    tfs = ["TP53", "MYC", "FOXO3", "NFKB1"]
    targets = ["CDKN1A", "BAX", "MDM2", "CCND1", "BCL2", "CCNE1", "E2F1", "RB1"]

    for tf in tfs:
        G.add_node(tf, type="transcription_factor", log2fc=np.random.normal(0, 0.5))

    for t in targets:
        G.add_node(t, type="gene", log2fc=np.random.normal(0, 1))

    edges = [
        ("TP53", "CDKN1A", {"weight": 0.9}),
        ("TP53", "BAX", {"weight": 0.8}),
        ("TP53", "MDM2", {"weight": 0.7}),
        ("MYC", "CCND1", {"weight": 0.9}),
        ("MYC", "BCL2", {"weight": 0.6}),
        ("FOXO3", "BCL2", {"weight": -0.7}),
        ("NFKB1", "BCL2", {"weight": 0.8}),
        ("NFKB1", "CCNE1", {"weight": 0.5}),
        ("E2F1", "CCNE1", {"weight": 0.9}),
        ("RB1", "E2F1", {"weight": -0.8}),
    ]
    for s, t, d in edges:
        G.add_edge(s, t, **d)


def plot_mutation_impact_graph(
    report: SimulationReport,
    output_path: Optional[Path] = None,
    figsize: tuple[int, int] = (14, 10),
) -> plt.Figure:
    """Graph showing mutation propagation from variant → protein → pathway → tissue → viability."""
    fig, ax = plt.subplots(figsize=figsize)

    # Build a layered DAG representing the simulation pipeline
    G = nx.DiGraph()

    layers = [
        ("L0", "Variant", [v.id for v in report.applied_variants]),
        ("L1", "Sequence\nSemantics", ["Codon Change", "Splicing", "Conservation", "Pathogenicity"]),
        ("L2", "Protein\nStructure", ["ΔΔG", "Stability", "Binding", "Allostery"]),
        ("L3", "Pathway\nActivity", ["Flux Change", "Metabolites", "Signaling", "PPI"]),
        ("L4", "Tissue\nFunction", ["Expression", "Cell Type", "Morphology", "Physiology"]),
        ("L5", "Viability", [f"Score: {report.viability.composite:.3f}", f"Verdict: {report.viability.verdict}"]),
    ]

    layer_positions = {}
    node_colors = []
    node_labels = {}

    for layer_idx, (layer_id, layer_name, items) in enumerate(layers):
        y = 5 - layer_idx  # Top to bottom
        for item_idx, item in enumerate(items):
            node_id = f"{layer_id}_{item_idx}"
            x = item_idx * 3 - (len(items) - 1) * 1.5
            pos = (x, y)
            layer_positions[node_id] = pos
            G.add_node(node_id)

            # Color by layer
            if layer_idx == 0:
                node_colors.append(COLORS["mutated"])
            elif layer_idx == len(layers) - 1:
                node_colors.append(_verdict_color(str(report.viability.verdict)))
            else:
                node_colors.append(plt.cm.viridis(layer_idx / (len(layers) - 1)))

            node_labels[node_id] = item

        # Connect to next layer
        if layer_idx < len(layers) - 1:
            next_items = layers[layer_idx + 1][2]
            for i, item in enumerate(items):
                for j in range(len(next_items)):
                    if np.random.random() < 0.7:  # Sparse connections
                        G.add_edge(f"{layer_id}_{i}", f"{layers[layer_idx+1][0]}_{j}")

    # Draw
    nx.draw_networkx_edges(G, layer_positions, width=1.5, alpha=0.5, edge_color=COLORS["edge"],
                          arrows=True, arrowsize=15, arrowstyle="->", ax=ax)
    nx.draw_networkx_nodes(G, layer_positions, node_color=node_colors, node_size=2000,
                          edgecolors="white", linewidths=2, ax=ax)
    nx.draw_networkx_labels(G, layer_positions, labels=node_labels, font_size=9, font_weight="bold", ax=ax)

    # Layer labels on the left
    for layer_idx, (layer_id, layer_name, _) in enumerate(layers):
        y = 5 - layer_idx
        ax.text(-8, y, layer_name, ha="right", va="center", fontsize=12, fontweight="bold",
               bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["background"], edgecolor=COLORS["text"]))

    ax.set_title(f"Mutation Impact Propagation — {report.simulation_id}", fontsize=16, pad=20)
    ax.axis("off")
    ax.set_xlim(-10, 8)
    ax.set_ylim(-1, 6)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=COLORS["background"])
    return fig