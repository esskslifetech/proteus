"""
Static plot generation for simulation results.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

from proteus.schema import SimulationReport, LayerDiff, VariantEvent, ViabilityScore


# Style configuration
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
COLORS = {
    "viable": "#2ecc71",
    "non_viable": "#e74c3c",
    "uncertain": "#f39c12",
    "reference": "#3498db",
    "mutated": "#e67e22",
    "reverted": "#9b59b6",
    "background": "#fafafa",
    "text": "#2c3e50",
}


def _verdict_color(verdict: str) -> str:
    v = verdict.lower()
    if v == "viable":
        return COLORS["viable"]
    if v == "non_viable":
        return COLORS["non_viable"]
    return COLORS["uncertain"]


def _significance_color(sig: str) -> str:
    s = sig.lower()
    if "high" in s:
        return COLORS["non_viable"]
    if "moderate" in s:
        return COLORS["uncertain"]
    return COLORS["viable"]


def plot_viability_radar(
    report: SimulationReport,
    output_path: Optional[Path] = None,
    figsize: tuple[int, int] = (8, 8),
) -> plt.Figure:
    """Radar/spider chart of layer viability scores."""
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection="polar"))

    if not report.viability.layer_scores:
        ax.text(0.5, 0.5, "No layer scores available", ha="center", va="center", transform=ax.transAxes)
        return fig

    layers = list(report.viability.layer_scores.keys())
    scores = [report.viability.layer_scores.get(l, 0.0) for l in layers]

    # Close the polygon
    angles = np.linspace(0, 2 * np.pi, len(layers), endpoint=False).tolist()
    angles += angles[:1]
    scores += scores[:1]

    # Reference perfect circle
    ax.plot(angles, [1.0] * len(angles), "k--", alpha=0.3, label="Perfect (1.0)")
    ax.fill(angles, [1.0] * len(angles), alpha=0.05, color="gray")

    # Mutated scores
    ax.plot(angles, scores, "o-", linewidth=2, color=COLORS["mutated"], label="Mutated")
    ax.fill(angles, scores, alpha=0.25, color=COLORS["mutated"])

    # Baseline scores (if available)
    if hasattr(report, "baseline_layer_scores") and report.baseline_layer_scores:
        base_scores = [report.baseline_layer_scores.get(l, 0.0) for l in layers]
        base_scores += base_scores[:1]
        ax.plot(angles, base_scores, "s--", linewidth=1.5, color=COLORS["reference"], label="Baseline")
        ax.fill(angles, base_scores, alpha=0.1, color=COLORS["reference"])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([l.replace("_", " ").title() for l in layers], fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)

    verdict = report.viability.verdict.name if hasattr(report.viability.verdict, "name") else str(report.viability.verdict)
    ax.set_title(
        f"Viability Radar — {verdict} (Composite: {report.viability.composite:.3f})",
        fontsize=14,
        pad=20,
        color=_verdict_color(verdict),
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=COLORS["background"])
    return fig


def plot_layer_diffs_heatmap(
    report: SimulationReport,
    output_path: Optional[Path] = None,
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Heatmap of layer diffs (delta and relative change).
    
    Shows absolute deltas and significance. When baseline values are 0 (no variants
    in reference), relative change is undefined so we show absolute deltas only.
    """
    if not report.diffs:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No layer diffs available", ha="center", va="center")
        return fig

    # Extract metrics from LayerDiff.changed dict
    layer_metrics = {}
    for d in report.diffs:
        layer_metrics.setdefault(d.layer, {})
        for metric, entry in d.changed.items():
            layer_metrics[d.layer][metric] = {
                "delta": entry.magnitude,
                "significance": entry.significance,
                "before": entry.before,
                "after": entry.after,
            }

    if not layer_metrics:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No changed metrics in diffs", ha="center", va="center")
        return fig

    layers = sorted(layer_metrics.keys())
    all_metrics = sorted(set().union(*[set(m.keys()) for m in layer_metrics.values()]))

    delta_matrix = np.full((len(layers), len(all_metrics)), np.nan)
    sig_matrix = np.full((len(layers), len(all_metrics)), "", dtype=object)

    for i, layer in enumerate(layers):
        for j, metric in enumerate(all_metrics):
            if metric in layer_metrics[layer]:
                m = layer_metrics[layer][metric]
                delta_matrix[i, j] = m["delta"]
                sig_matrix[i, j] = m["significance"]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Delta heatmap with significance annotations
    vmax_delta = np.nanmax(np.abs(delta_matrix)) if np.any(~np.isnan(delta_matrix)) else 1
    im = ax.imshow(delta_matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax_delta, vmax=vmax_delta)
    ax.set_xticks(range(len(all_metrics)))
    ax.set_xticklabels(all_metrics, rotation=45, ha="right")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([l.replace("_", " ").title() for l in layers])
    ax.set_title("Absolute Delta (Mutated - Reference)")
    plt.colorbar(im, ax=ax, label="Delta")

    # Annotate significance
    for i in range(len(layers)):
        for j in range(len(all_metrics)):
            if not np.isnan(delta_matrix[i, j]):
                sig = sig_matrix[i, j]
                color = "white" if abs(delta_matrix[i, j]) > np.nanmax(np.abs(delta_matrix)) * 0.5 else "black"
                ax.text(j, i, sig[0].upper() if sig else "?", ha="center", va="center", fontsize=8, color=color, fontweight="bold")

    fig.suptitle(f"Layer Diffs — {report.simulation_id}", fontsize=14)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=COLORS["background"])
    return fig


def plot_variant_effects(
    report: SimulationReport,
    output_path: Optional[Path] = None,
    figsize: tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Visualize variant positions and predicted effects across the genome."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[1, 3], sharex=True)

    if not report.applied_variants:
        ax1.text(0.5, 0.5, "No variants applied", ha="center", va="center", transform=ax1.transAxes)
        ax2.text(0.5, 0.5, "No variants applied", ha="center", va="center", transform=ax2.transAxes)
        return fig

    # Extract variant info
    variants = report.applied_variants
    positions = [(v.start + v.end) / 2 for v in variants]
    types = [v.type.name if hasattr(v.type, "name") else str(v.type) for v in variants]
    refs = [v.ref_allele for v in variants]
    alts = [v.alt_allele for v in variants]

    # Top: variant type markers
    type_colors = {
        "SNV": "#e74c3c", "MNV": "#c0392b",
        "INSERTION": "#2ecc71", "DELETION": "#e67e22",
        "DELINS": "#9b59b6", "INVERSION": "#3498db",
        "DUPLICATION": "#f39c12", "CNV": "#1abc9c",
    }
    for pos, vtype in zip(positions, types):
        ax1.axvline(pos, color=type_colors.get(vtype, "#95a5a6"), alpha=0.7, linewidth=2)
        ax1.plot(pos, 0.5, "v", color=type_colors.get(vtype, "#95a5a6"), markersize=10)

    ax1.set_ylim(0, 1)
    ax1.set_yticks([])
    ax1.set_title("Variant Positions & Types", fontsize=12)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=t) for t, c in type_colors.items() if t in types]
    ax1.legend(handles=legend_elements, loc="upper right", ncol=4, fontsize=8)

    # Bottom: effect scores (if available in layer_results)
    if hasattr(report, "layer_results") and report.layer_results:
        l1 = report.layer_results.get("L1_variant_semantics", {})
        if isinstance(l1, dict) and "variant_effects" in l1:
            effects = l1["variant_effects"]
            if isinstance(effects, list) and len(effects) == len(variants):
                scores = [e.get("pathogenicity_score", 0.5) for e in effects]
                bars = ax2.bar(positions, scores, width=np.diff(positions).min() if len(positions) > 1 else 1,
                              color=[type_colors.get(t, "#95a5a6") for t in types], alpha=0.7, edgecolor="black")
                ax2.set_ylabel("Pathogenicity Score")
                ax2.set_ylim(0, 1.1)
                ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
                ax2.axhline(0.8, color="red", linestyle="--", alpha=0.5, label="High risk threshold")
                ax2.legend()

    ax2.set_xlabel("Genomic Position (1-based)")
    ax2.set_title("Predicted Variant Effects (L1)", fontsize=12)

    fig.suptitle(f"Variant Effects — {report.simulation_id}", fontsize=14)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=COLORS["background"])
    return fig


def plot_energy_landscape(
    report: SimulationReport,
    output_path: Optional[Path] = None,
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Protein folding energy landscape visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    if hasattr(report, "layer_results") and report.layer_results:
        l2 = report.layer_results.get("L2_protein", {})
        if isinstance(l2, dict):
            ref_terms = l2.get("reference_energy_terms", {})
            mut_terms = l2.get("mutated_energy_terms", {})

            if ref_terms and mut_terms:
                terms = sorted(set(ref_terms.keys()) | set(mut_terms.keys()))
                ref_vals = [ref_terms.get(t, 0) for t in terms]
                mut_vals = [mut_terms.get(t, 0) for t in terms]

                x = np.arange(len(terms))
                width = 0.35
                ax1.bar(x - width/2, ref_vals, width, label="Reference", color=COLORS["reference"], alpha=0.8)
                ax1.bar(x + width/2, mut_vals, width, label="Mutated", color=COLORS["mutated"], alpha=0.8)
                ax1.set_xticks(x)
                ax1.set_xticklabels(terms, rotation=45, ha="right", fontsize=9)
                ax1.set_ylabel("Energy (REU)")
                ax1.set_title("Rosetta Energy Terms")
                ax1.legend()

                # DDG distribution if available
                ddg = l2.get("ddg_distribution", {})
                if ddg:
                    ax2.bar(["Mean ΔΔG"], [ddg.get("mean", 0)], yerr=[[ddg.get("std", 0)]],
                           color=COLORS["mutated"] if ddg.get("mean", 0) > 0 else COLORS["reference"],
                           alpha=0.7, capsize=5, edgecolor="black")
                    ax2.axhline(0, color="black", linewidth=0.5)
                    ax2.axhline(1.0, color="red", linestyle="--", alpha=0.5, label="Destabilizing threshold")
                    ax2.axhline(-1.0, color="green", linestyle="--", alpha=0.5, label="Stabilizing threshold")
                    ax2.set_ylabel("ΔΔG (kcal/mol)")
                    ax2.set_title("Folding Free Energy Change")
                    ax2.legend()

    else:
        ax1.text(0.5, 0.5, "L2 protein data not available", ha="center", va="center", transform=ax1.transAxes)
        ax2.text(0.5, 0.5, "L2 protein data not available", ha="center", va="center", transform=ax2.transAxes)

    fig.suptitle(f"Energy Landscape — {report.simulation_id}", fontsize=14)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=COLORS["background"])
    return fig


def plot_conservation_profile(
    report: SimulationReport,
    output_path: Optional[Path] = None,
    figsize: tuple[int, int] = (12, 4),
) -> plt.Figure:
    """Sequence conservation profile across the genome with variant positions."""
    fig, ax = plt.subplots(figsize=figsize)

    # Simulated conservation scores (would come from MSA in real implementation)
    if hasattr(report, "layer_results") and report.layer_results:
        l1 = report.layer_results.get("L1_variant_semantics", {})
        if isinstance(l1, dict) and "conservation_profile" in l1:
            profile = l1["conservation_profile"]
            if isinstance(profile, dict) and "positions" in profile and "scores" in profile:
                pos = profile["positions"]
                scores = profile["scores"]
                ax.plot(pos, scores, color=COLORS["reference"], linewidth=1, alpha=0.8)
                ax.fill_between(pos, 0, scores, alpha=0.2, color=COLORS["reference"])

                # Mark variants
                for v in report.applied_variants:
                    ax.axvline(v.start, color=COLORS["mutated"], linestyle="--", alpha=0.7)
                    ax.plot(v.start, np.interp(v.start, pos, scores), "o",
                           color=COLORS["mutated"], markersize=8, zorder=5)

    ax.set_xlabel("Genomic Position")
    ax.set_ylabel("Conservation Score (PhyloP-like)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Conservation Profile — {report.simulation_id}")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=COLORS["background"])
    return fig


def plot_validation_summary(
    report: SimulationReport,
    output_path: Optional[Path] = None,
    figsize: tuple[int, int] = (8, 5),
) -> plt.Figure:
    """Bar chart of validation check results."""
    fig, ax = plt.subplots(figsize=figsize)

    if not report.validation_results:
        ax.text(0.5, 0.5, "No validation results", ha="center", va="center")
        return fig

    checks = list(report.validation_results.keys())
    statuses = list(report.validation_results.values())

    colors = [_verdict_color(s) for s in statuses]
    bars = ax.barh(range(len(checks)), [1] * len(checks), color=colors, edgecolor="black", height=0.6)

    for i, (check, status) in enumerate(zip(checks, statuses)):
        ax.text(0.5, i, f"{check}: {status}", ha="center", va="center", fontweight="bold", fontsize=11,
               color="white" if status.lower() in ("fail", "non_viable") else "black")

    ax.set_yticks(range(len(checks)))
    ax.set_yticklabels([])
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title(f"Validation Summary — {report.simulation_id}", fontsize=14)
    ax.invert_yaxis()  # V1 at top

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["viable"], label="Pass", edgecolor="black"),
        Patch(facecolor=COLORS["non_viable"], label="Fail", edgecolor="black"),
        Patch(facecolor=COLORS["uncertain"], label="Warn/Uncertain", edgecolor="black"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=COLORS["background"])
    return fig


def create_dashboard(
    report: SimulationReport,
    output_dir: Path,
    prefix: str = "dashboard",
) -> dict[str, Path]:
    """Generate a complete multi-panel dashboard for the simulation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # 1. Viability radar
    paths["radar"] = output_dir / f"{prefix}_viability_radar.png"
    plot_viability_radar(report, paths["radar"])
    plt.close()

    # 2. Layer diffs heatmap
    paths["heatmap"] = output_dir / f"{prefix}_layer_diffs.png"
    plot_layer_diffs_heatmap(report, paths["heatmap"])
    plt.close()

    # 3. Variant effects
    paths["variants"] = output_dir / f"{prefix}_variant_effects.png"
    plot_variant_effects(report, paths["variants"])
    plt.close()

    # 4. Energy landscape
    paths["energy"] = output_dir / f"{prefix}_energy_landscape.png"
    plot_energy_landscape(report, paths["energy"])
    plt.close()

    # 5. Conservation profile
    paths["conservation"] = output_dir / f"{prefix}_conservation.png"
    plot_conservation_profile(report, paths["conservation"])
    plt.close()

    # 6. Validation summary
    paths["validation"] = output_dir / f"{prefix}_validation.png"
    plot_validation_summary(report, paths["validation"])
    plt.close()

    # 7. Composite dashboard figure
    paths["composite"] = output_dir / f"{prefix}_composite.png"
    _create_composite_dashboard(report, paths["composite"])
    plt.close()

    return paths


def _create_composite_dashboard(report: SimulationReport, output_path: Path) -> plt.Figure:
    """Create a single large composite figure with all panels."""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3)

    # Top row: Viability radar (spans 2 cols)
    ax1 = fig.add_subplot(gs[0, :2], projection="polar")
    _plot_viability_radar_ax(ax1, report)

    # Top right: Validation summary (spans 2 cols)
    ax2 = fig.add_subplot(gs[0, 2:])
    _plot_validation_summary_ax(ax2, report)

    # Second row: Layer diffs heatmap (spans full width)
    ax3 = fig.add_subplot(gs[1, :])
    _plot_layer_diffs_heatmap_ax(ax3, report)

    # Third row: Energy landscape (left) + Conservation (right)
    ax4 = fig.add_subplot(gs[2, :2])
    _plot_energy_landscape_ax(ax4, report)

    ax5 = fig.add_subplot(gs[2, 2:])
    _plot_conservation_profile_ax(ax5, report)

    # Bottom row: Variant effects (spans full width)
    ax6 = fig.add_subplot(gs[3, :])
    _plot_variant_effects_ax(ax6, report)

    fig.suptitle(f"Proteus Simulation Dashboard — {report.simulation_id}\n"
                 f"Verdict: {report.viability.verdict.name if hasattr(report.viability.verdict, 'name') else report.viability.verdict} "
                 f"(Composite: {report.viability.composite:.3f})",
                 fontsize=18, y=0.98)

    fig.savefig(output_path, dpi=120, bbox_inches="tight", facecolor=COLORS["background"])
    return fig


def _plot_viability_radar_ax(ax, report):
    if not report.viability.layer_scores:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    layers = list(report.viability.layer_scores.keys())
    scores = [report.viability.layer_scores.get(l, 0) for l in layers]
    angles = np.linspace(0, 2*np.pi, len(layers), endpoint=False).tolist()
    angles += angles[:1]
    scores += scores[:1]
    ax.plot(angles, scores, "o-", color=COLORS["mutated"], linewidth=2)
    ax.fill(angles, scores, alpha=0.25, color=COLORS["mutated"])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([l.replace("_", " ").title() for l in layers], fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_title("Viability by Layer", fontsize=11, pad=10)


def _plot_validation_summary_ax(ax, report):
    if not report.validation_results:
        return
    checks = list(report.validation_results.keys())
    statuses = list(report.validation_results.values())
    colors = [_verdict_color(s) for s in statuses]
    for i, (check, status, color) in enumerate(zip(checks, statuses, colors)):
        ax.barh(i, 1, color=color, edgecolor="black", height=0.5)
        ax.text(0.5, i, f"{check}: {status}", ha="center", va="center", fontsize=9,
               fontweight="bold", color="white" if status.lower() in ("fail", "non_viable") else "black")
    ax.set_yticks(range(len(checks)))
    ax.set_yticklabels([])
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.invert_yaxis()
    ax.set_title("Validation Checks", fontsize=11)


def _plot_layer_diffs_heatmap_ax(ax, report):
    if not report.diffs:
        ax.text(0.5, 0.5, "No diffs", ha="center", va="center")
        return

    layer_metrics = {}
    for d in report.diffs:
        layer_metrics.setdefault(d.layer, {})
        for metric, entry in d.changed.items():
            layer_metrics[d.layer][metric] = entry.magnitude

    if not layer_metrics:
        ax.text(0.5, 0.5, "No metrics", ha="center", va="center")
        return

    layers = sorted(layer_metrics.keys())
    metrics = sorted(set().union(*[set(m.keys()) for m in layer_metrics.values()]))
    mat = np.full((len(layers), len(metrics)), np.nan)
    for i, layer in enumerate(layers):
        for j, metric in enumerate(metrics):
            if metric in layer_metrics[layer]:
                mat[i, j] = layer_metrics[layer][metric] * 100

    vmax = np.nanmax(np.abs(mat)) if np.any(~np.isnan(mat)) else 1
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([l.replace("_", " ").title() for l in layers], fontsize=8)
    ax.set_title("Layer Diffs (% Change)", fontsize=11)
    plt.colorbar(im, ax=ax, label="%", shrink=0.6)


def _plot_energy_landscape_ax(ax, report):
    if hasattr(report, "layer_results") and report.layer_results:
        l2 = report.layer_results.get("L2_protein", {})
        if isinstance(l2, dict):
            ref = l2.get("reference_energy_terms", {})
            mut = l2.get("mutated_energy_terms", {})
            if ref and mut:
                terms = sorted(set(ref.keys()) | set(mut.keys()))
                x = np.arange(len(terms))
                w = 0.3
                ax.bar(x - w/2, [ref.get(t, 0) for t in terms], w, label="Ref", color=COLORS["reference"], alpha=0.8)
                ax.bar(x + w/2, [mut.get(t, 0) for t in terms], w, label="Mut", color=COLORS["mutated"], alpha=0.8)
                ax.set_xticks(x)
                ax.set_xticklabels(terms, rotation=45, ha="right", fontsize=7)
                ax.legend(fontsize=8)
                ax.set_title("Energy Terms (L2)", fontsize=11)
                return
    ax.text(0.5, 0.5, "L2 data N/A", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Energy Terms (L2)", fontsize=11)


def _plot_conservation_profile_ax(ax, report):
    if hasattr(report, "layer_results") and report.layer_results:
        l1 = report.layer_results.get("L1_variant_semantics", {})
        if isinstance(l1, dict) and "conservation_profile" in l1:
            prof = l1["conservation_profile"]
            if isinstance(prof, dict) and "positions" in prof and "scores" in prof:
                pos, scores = prof["positions"], prof["scores"]
                ax.plot(pos, scores, color=COLORS["reference"], linewidth=1)
                ax.fill_between(pos, 0, scores, alpha=0.2, color=COLORS["reference"])
                for v in report.applied_variants:
                    ax.axvline(v.start, color=COLORS["mutated"], ls="--", alpha=0.7)
                ax.set_title("Conservation Profile (L1)", fontsize=11)
                return
    ax.text(0.5, 0.5, "L1 conservation N/A", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Conservation Profile (L1)", fontsize=11)


def _plot_variant_effects_ax(ax, report):
    if not report.applied_variants:
        ax.text(0.5, 0.5, "No variants", ha="center", va="center")
        return
    pos = [(v.start + v.end)/2 for v in report.applied_variants]
    types = [v.type.name if hasattr(v.type, "name") else str(v.type) for v in report.applied_variants]
    type_colors = {"SNV": "#e74c3c", "INSERTION": "#2ecc71", "DELETION": "#e67e22",
                   "DELINS": "#9b59b6", "INVERSION": "#3498db", "DUPLICATION": "#f39c12"}
    for p, t in zip(pos, types):
        ax.axvline(p, color=type_colors.get(t, "#95a5a6"), alpha=0.5)
        ax.plot(p, 0.5, "v", color=type_colors.get(t, "#95a5a6"), ms=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Genomic Position (1-based)")
    ax.set_title("Applied Variants", fontsize=11)