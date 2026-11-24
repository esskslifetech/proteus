"""
Animation generation for mutation effect visualization.
Creates frame-by-frame sequences and converts to GIF.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

from proteus.schema import SimulationReport


def animate_mutation_effect(
    report: SimulationReport,
    output_path: Optional[Path] = None,
    n_frames: int = 16,
    fps: int = 8,
    figsize: tuple[int, int] = (8, 5),
    dpi: int = 80,
) -> Path:
    """Create an animation showing the mutation effect propagating through layers.

    Each frame shows a different stage of propagation:
    - Frame 0: Baseline state
    - Middle frames: Mutation spreading through L0â†’L1â†’L2â†’L3â†’L4â†’L5
    - Last frame: Final state with viability verdict
    """
    output_path = Path(output_path) if output_path else Path("mutation_effect.gif")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    ax_grid = axes.flatten()

    # Clean subplots
    for ax in ax_grid:
        ax.clear()

    # Color palette
    colors = {
        "baseline": "#3498db",
        "propagating": "#e67e22",
        "affected": "#e74c3c",
        "recovered": "#2ecc71",
        "neutral": "#ecf0f1",
    }

    # Architecture: L0=Sequence, L1=Variant, L2=Protein, L3=Pathway, L4=Tissue, L5=Viability
    layer_names = ["L0\nSequence", "L1\nVariant", "L2\nProtein", "L3\nPathway", "L4\nTissue", "L5\nViability"]
    baseline_scores = [1.0] * 6  # perfect baseline
    mutated_scores = list(report.viability.layer_scores.values()) if report.viability.layer_scores else [0.9] * 6
    while len(mutated_scores) < 6:
        mutated_scores.append(0.9)

    # Pre-compute score trajectories for all frames
    n_layers = 6
    score_trajectories = []
    for l in range(n_layers):
        spread_delay = l / n_layers
        trajectory = []
        for f in range(n_frames):
            p = f / max(n_frames - 1, 1)
            lp = max(0, min(1, (p - spread_delay) / (1 - spread_delay)))
            s = baseline_scores[l] * (1 - lp) + mutated_scores[min(l, len(mutated_scores) - 1)] * lp
            trajectory.append(s)
        score_trajectories.append(trajectory)

    def init():
        for i, ax in enumerate(ax_grid):
            ax.clear()
            ax.set_facecolor(colors["neutral"])

        # Panel 1: Layer scores over time - show frame 0 data
        ax_grid[0].set_title("Layer Scores Over Time", fontsize=12)
        ax_grid[0].set_xlim(0, n_frames)
        ax_grid[0].set_ylim(0, 1.1)
        ax_grid[0].set_xlabel("Frame")
        ax_grid[0].set_ylabel("Score")
        ax_grid[0].axhline(0.8, color="red", linestyle="--", alpha=0.3, label="Viability Threshold")
        layer_colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_layers))
        for l in range(n_layers):
            # Draw frame 0 point
            ax_grid[0].plot(0, score_trajectories[l][0], 'o', color=layer_colors[l], markersize=6, label=layer_names[l].replace('\n', ' '))
        ax_grid[0].legend(loc="lower left", fontsize=7, ncol=2)

        ax_grid[1].set_title("Mutation Impact by Layer", fontsize=12)
        ax_grid[1].set_xlim(-0.5, 5.5)
        ax_grid[1].set_ylim(0, 1)

        ax_grid[2].set_title("Energy Landscape", fontsize=12)
        ax_grid[2].set_xlim(0, 10)
        ax_grid[2].set_ylim(0, 10)

        ax_grid[3].set_title("Viability Gate", fontsize=12)
        ax_grid[3].set_xlim(0, 1)
        ax_grid[3].set_ylim(0, 1)
        ax_grid[3].axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        return ax_grid

    def update(frame: int):
        progress = frame / max(n_frames - 1, 1)

        # Clear all axes
        for ax in ax_grid:
            ax.clear()

        layer_colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_layers))

        # --- Panel 1: Layer scores over time ---
        ax_grid[0].set_facecolor("#fafafa")
        ax_grid[0].set_title("Layer Scores Over Time", fontsize=12)
        ax_grid[0].set_xlim(0, n_frames)
        ax_grid[0].set_ylim(0, 1.1)
        ax_grid[0].set_xlabel("Frame")
        ax_grid[0].set_ylabel("Score")
        ax_grid[0].axhline(0.8, color="red", linestyle="--", alpha=0.3, label="Threshold")

        # Draw lines for each layer up to current frame
        for l in range(n_layers):
            ax_grid[0].plot(range(frame + 1), score_trajectories[l][:frame+1], 
                          color=layer_colors[l], linewidth=2, label=layer_names[l].replace('\n', ' '))
            # Current frame marker
            ax_grid[0].plot(frame, score_trajectories[l][frame], 'o', color=layer_colors[l], markersize=6)
        ax_grid[0].legend(loc="lower left", fontsize=7, ncol=2)

        # --- Panel 2: Mutation propagation heatmap ---
        ax_grid[1].set_facecolor("#fafafa")
        ax_grid[1].set_title("Mutation Impact by Layer", fontsize=12)
        ax_grid[1].set_xlim(-0.5, 5.5)
        ax_grid[1].set_ylim(0, 1)

        impact_values = []
        for l in range(6):
            spread_delay = l / n_layers
            lp = max(0, min(1, (progress - spread_delay) / (1 - spread_delay)))
            impact_values.append(lp)

        # Draw bars
        for i, impact in enumerate(impact_values):
            color = colors["baseline"] if impact < 0.1 else (
                colors["affected"] if impact > 0.7 else colors["propagating"]
            )
            ax_grid[1].barh(0.5, 0.8, left=i * 0.9 + impact * 0.1, height=0.3,
                          color=color, alpha=0.8 + 0.2 * (1 - impact), edgecolor="white")
            ax_grid[1].barh(0.5, impact * 0.8, left=i * 0.9, height=0.3,
                          color=color, alpha=0.7, edgecolor="black")

        ax_grid[1].set_yticks([])
        for i in range(6):
            ax_grid[1].text(i, -0.15, layer_names[i].replace('\n', ''), ha="center", fontsize=7, rotation=45)

        # --- Panel 3: Energy landscape ---
        ax_grid[2].set_facecolor("#fafafa")
        ax_grid[2].set_title("Energy Well Convergence", fontsize=12)
        ax_grid[2].set_xlim(0, 10)
        ax_grid[2].set_ylim(0, 10)

        # Draw a parabolic energy well that narrows over time
        x = np.linspace(0, 10, 100)
        center = 5.0 + 2 * np.sin(progress * np.pi) * np.exp(-progress)
        width = 1.5 + (1 - progress) * 2
        y_baseline = 2 + 3 * ((x - 5) ** 2) / 4
        y_mutated = 2 + 3 * ((x - center) ** 2) / (width * width)

        ax_grid[2].plot(x, y_baseline, color="#3498db", linestyle="--", alpha=0.3, label="Baseline")
        ax_grid[2].fill_between(x, 0, y_mutated, alpha=0.5 * progress, color="#e74c3c")
        ax_grid[2].plot(x, y_mutated, color="#e74c3c", linewidth=2)

        # Mark energy minimum
        min_x = center
        min_y = 2
        ax_grid[2].plot(min_x, min_y, "o", color="#e74c3c", markersize=12,
                       markeredgecolor="white", markeredgewidth=2)
        ax_grid[2].text(min_x, min_y + 0.5, f"Î”E={min_y:.1f}", ha="center", fontsize=9,
                       bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        ax_grid[2].set_xlabel("Conformation")
        ax_grid[2].set_ylabel("Energy")
        ax_grid[2].legend(loc="upper left", fontsize=8)

        # --- Panel 4: Viability gate ---
        ax_grid[3].set_facecolor("#fafafa")
        ax_grid[3].set_title("Viability Gate", fontsize=12)
        ax_grid[3].set_xlim(0, 1)
        ax_grid[3].set_ylim(0, 1)

        composite = report.viability.composite
        # Gate ramps up to final composite score
        gate_score = composite * progress
        verdict = report.viability.verdict
        verdict_str = verdict.name if hasattr(verdict, "name") else str(verdict)

        # Draw gate indicator
        color = "#2ecc71" if gate_score >= 0.8 else ("#e74c3c" if gate_score < 0.5 else "#f39c12")
        gate_rect = ax_grid[3].barh(0.5, gate_score, height=0.6, left=0,
                                   color=color, alpha=0.8, edgecolor="black", linewidth=2)

        ax_grid[3].axvline(0.8, color="red", linestyle="--", linewidth=2, alpha=0.5)
        ax_grid[3].text(0.8, 0.9, "0.8", ha="center", fontsize=9, color="red")
        ax_grid[3].text(gate_score / 2, 0.5, f"{gate_score:.3f}", ha="center", va="center",
                       fontsize=16, fontweight="bold", color="white" if gate_score > 0.4 else "black")
        ax_grid[3].text(0.5, 0.1, verdict_str, ha="center", fontsize=12, fontweight="bold",
                       color=color)

        ax_grid[3].set_yticks([])
        ax_grid[3].set_xticks([])
        ax_grid[3].set_xlabel("Score", fontsize=10)

        # Frame counter
        fig.text(0.02, 0.02, f"Frame {frame}/{n_frames}", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        return ax_grid

    fig.suptitle(f"Mutation Effect â€” {report.simulation_id}", fontsize=16, y=0.98)
    plt.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.08, wspace=0.3, hspace=0.3)

    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                        interval=1000 / fps, blit=False)

    if output_path.suffix.lower() == ".gif":
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)
    elif output_path.suffix.lower() == ".mp4":
        anim.save(output_path, writer="ffmpeg", fps=fps, extra_args=["-vcodec", "libx264"])
    else:
        anim.save(output_path, writer=PillowWriter(fps=fps))

    plt.close(fig)
    return output_path


def animate_layer_propagation(
    report: SimulationReport,
    output_path: Optional[Path] = None,
    n_frames: int = 16,
    fps: int = 8,
    figsize: tuple[int, int] = (9, 6),
    dpi: int = 80,
) -> Path:
    """Create an animation showing signal propagation through biological layers.

    Visualizes the cascade from variant injection through all 6 simulation layers.
    """
    output_path = Path(output_path) if output_path else Path("layer_propagation.gif")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Architecture: L0=Sequence, L1=Variant, L2=Protein, L3=Pathway, L4=Tissue, L5=Viability
    layer_names = ["L0\nSequence", "L1\nVariant", "L2\nProtein", "L3\nPathway", "L4\nTissue", "L5\nViability"]
    n_nodes_per_layer = [1, 3, 4, 4, 3, 1]
    all_positions = {}
    all_edges = []

    # Build network layout with more spacing to avoid overlapping ellipses
    for layer_idx, (name, n_nodes) in enumerate(zip(layer_names, n_nodes_per_layer)):
        for j in range(n_nodes):
            y = -layer_idx * 2.2  # Increased vertical spacing
            x = (j - (n_nodes - 1) / 2) * 1.8  # Increased horizontal spacing
            all_positions[f"L{layer_idx}_{j}"] = [x, y]

        # Connect to next layer
        if layer_idx < 5:
            next_n = n_nodes_per_layer[layer_idx + 1]
            for j in range(n_nodes):
                for k in range(next_n):
                    all_edges.append((f"L{layer_idx}_{j}", f"L{layer_idx+1}_{k}"))

    def init():
        ax.clear()
        ax.set_facecolor("#fafafa")
        ax.set_xlim(-5, 5)
        ax.set_ylim(2, -13)
        ax.axis("off")
        _draw_network(ax, all_positions, all_edges, progress=0)
        return ax

    def update(frame: int):
        progress = frame / max(n_frames - 1, 1)
        ax.clear()
        ax.set_facecolor("#fafafa")
        ax.set_xlim(-5, 5)
        ax.set_ylim(2, -13)
        ax.axis("off")
        _draw_network(ax, all_positions, all_edges, progress=progress)

        # Title with progress
        ax.set_title(f"Layer Propagation â€” Frame {frame}/{n_frames} "
                    f"(Progress: {progress:.0%}) â€” {report.simulation_id}",
                    fontsize=14, pad=10)

        return ax

    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                        interval=1000 / fps, blit=False)

    if output_path.suffix.lower() == ".gif":
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)
    else:
        anim.save(output_path, writer=PillowWriter(fps=fps))

    plt.close(fig)
    return output_path


def _draw_network(ax, positions, edges, progress: float):
    """Draw the layer network with propagation visualization."""
    from matplotlib.patches import FancyArrowPatch, Circle

    layer_names = ["L0\nSequence", "L1\nVariant", "L2\nProtein", "L3\nPathway", "L4\nTissue", "L5\nViability"]
    n_nodes_per_layer = [1, 3, 4, 4, 3, 1]

    # Draw edges
    for src, tgt in edges:
        if src in positions and tgt in positions:
            sp = np.array(positions[src])
            tp = np.array(positions[tgt])
            # Determine which layer this edge belongs to
            src_layer = int(src.split("_")[0][1])
            edge_delay = src_layer / 6
            edge_progress = max(0, min(1, (progress - edge_delay) / (1 - edge_delay)))

            if edge_progress > 0:
                color = "#e67e22" if edge_progress < 0.5 else "#e74c3c"
                ax.plot([sp[0], tp[0]], [sp[1], tp[1]], color=color,
                       linewidth=1.5 * edge_progress, alpha=0.5, zorder=1)
                # Draw signal dot
                signal_pos = sp + (tp - sp) * edge_progress
                ax.plot(signal_pos[0], signal_pos[1], "o", color="#e74c3c",
                       markersize=3 * edge_progress, zorder=2, alpha=0.7)

    # Draw nodes - at frame 0, all nodes are baseline blue
    for layer_idx, name in enumerate(layer_names):
        n_nodes = n_nodes_per_layer[layer_idx]
        for j in range(n_nodes):
            node_id = f"L{layer_idx}_{j}"
            if node_id in positions:
                pos = np.array(positions[node_id])
                # Node activation based on progress
                node_delay = layer_idx / 6
                activation = max(0, min(1, (progress - node_delay) / (1 - node_delay)))

                # Node color transitions from blue (baseline) to red (affected)
                # At frame 0 (progress=0), activation=0 for all -> uniform blue
                r = 0.2 + 0.7 * activation
                g = 0.6 * (1 - activation)
                b = 0.9 * (1 - activation)
                color = (r, g, b)

                # Fixed node size - no pulsing to avoid overlapping
                size = 300
                circle = Circle(pos, np.sqrt(size / np.pi) * 0.12,
                              facecolor=color, edgecolor="white", linewidth=2, zorder=3)
                ax.add_patch(circle)

                # Label
                ax.text(pos[0], pos[1], f"{layer_idx}.{j}", ha="center", va="center",
                       fontsize=7, fontweight="bold", zorder=4)

        # Layer label (adjusted for new y positions)
        ax.text(-4.5, -layer_idx * 2.2, name, ha="right", va="center", fontsize=10,
               fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="#ecf0f1",
                                           edgecolor="#bdc3c7", alpha=0.9))


def create_gif(
    report: SimulationReport,
    output_dir: Path,
    prefix: str = "animation",
    n_frames: int = 16,
    fps: int = 8,
) -> dict[str, Path]:
    """Generate all animation GIFs for a report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # 1. Mutation effect animation
    paths["mutation_effect"] = output_dir / f"{prefix}_mutation_effect.gif"
    animate_mutation_effect(report, paths["mutation_effect"], n_frames=n_frames, fps=fps)

    # 2. Layer propagation animation
    paths["layer_propagation"] = output_dir / f"{prefix}_layer_propagation.gif"
    animate_layer_propagation(report, paths["layer_propagation"], n_frames=n_frames, fps=fps)

    return paths
