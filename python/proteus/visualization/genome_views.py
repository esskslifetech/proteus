"""
Genome- and molecule-centric visualizations for Proteus.

These figures show the actual inputs and intermediate objects used by the
pipeline: the reference DNA sequence with injected lesions, a DNA double helix
view, sequence-derived codon translation, the layer cascade with the engine
used at each step, the reference/model inputs, and the glycolysis map with the
expression values that drove the pathway layer.

All functions degrade gracefully to a placeholder when optional layer payloads
are absent (for example when a report is loaded from a serialized artifact).
"""

from __future__ import annotations

import difflib
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

# ---------------------------------------------------------------------------
# Palette / constants
# ---------------------------------------------------------------------------

COLORS = {
    "background": "#f7f9fc",
    "panel": "#ffffff",
    "text": "#1b1f24",
    "muted": "#5b6673",
    "border": "#d0d7de",
    "reference": "#3498db",
    "mutated": "#e67e22",
    "changed": "#e63946",
    "insert": "#9b59b6",
    "delete": "#d97706",
    "pass": "#2ecc71",
    "warn": "#f1c40f",
    "fail": "#e74c3c",
    "helix_bg": "#070b18",
    "helix_left": "#4cc9f0",
    "helix_right": "#f72585",
    "lesion": "#ffe66d",
}

BASE_COLORS = {
    "A": "#2ecc71",
    "T": "#e74c3c",
    "G": "#f1c40f",
    "C": "#3498db",
    "U": "#9b59b6",
    "N": "#95a5a6",
    "-": "#ffffff",
}

CODON_TABLE: dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

AA_NAMES = {
    "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys",
    "Q": "Gln", "E": "Glu", "G": "Gly", "H": "His", "I": "Ile",
    "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe", "P": "Pro",
    "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val",
    "*": "Ter",
}

VIEW_TITLES = {
    "genome_browser": "Genome browser: reference sequence vs injected mutations",
    "dna_helix": "DNA double helix with lesion positions",
    "central_dogma": "Sequence-derived DNA → mRNA → protein consequences",
    "molecular_cascade": "Which layer used which engine, and what it produced",
    "inputs_provenance": "Reference genome, model hooks, and compute engines",
    "glycolysis_map": "Glycolysis map: expression values used by the pathway layer",
}


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _layer(report: Any, name: str) -> dict[str, Any]:
    layers = getattr(report, "layer_results", {}) or {}
    data = layers.get(name, {}) if isinstance(layers, dict) else {}
    return data if isinstance(data, dict) else {}


def _variants(report: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for v in getattr(report, "applied_variants", []) or []:
        vtype = getattr(v, "type", "")
        type_name = str(getattr(vtype, "name", vtype))
        annotations = getattr(v, "annotations", {}) or {}
        out.append({
            "id": str(getattr(v, "id", "")),
            "type": type_name,
            "contig": str(getattr(v, "contig", "")),
            "start": int(getattr(v, "start", 0) or 0),
            "end": int(getattr(v, "end", 0) or 0),
            "ref": str(getattr(v, "ref_allele", "") or "").upper(),
            "alt": str(getattr(v, "alt_allele", "") or "").upper(),
            "gene": str(annotations.get("gene_id") or annotations.get("gene") or "unannotated"),
        })
    return out


def _previews(report: Any) -> dict[str, Any]:
    previews = _layer(report, "L0").get("sequence_previews")
    return previews if isinstance(previews, dict) else {}


def _l1_details(report: Any) -> dict[str, dict[str, Any]]:
    details = _layer(report, "L1").get("variant_details") or []
    return {str(d.get("id")): d for d in details if isinstance(d, dict)}


def _l2_details(report: Any) -> dict[str, dict[str, Any]]:
    effects = _layer(report, "L2").get("effects") or []
    return {str(e.get("id")): e for e in effects if isinstance(e, dict)}


def _first_window(report: Any) -> tuple[str, dict[str, Any]]:
    for contig, preview in _previews(report).items():
        windows = preview.get("windows") or []
        if windows:
            return contig, windows[0]
    return "", {}


def _clean_sequence(text: Any) -> str:
    return "".join(ch for ch in str(text or "").upper() if ch in "ACGTUN")


def _alignment(reference: str, mutated: str) -> list[dict[str, Any]]:
    """Align two short sequences and return display columns with positions."""
    columns: list[dict[str, Any]] = []
    matcher = difflib.SequenceMatcher(None, reference, mutated, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                columns.append({
                    "ref": reference[i1 + offset],
                    "alt": mutated[j1 + offset],
                    "changed": False,
                    "kind": "match",
                })
        elif tag == "replace":
            length = max(i2 - i1, j2 - j1)
            for offset in range(length):
                ref_base = reference[i1 + offset] if i1 + offset < i2 else ""
                alt_base = mutated[j1 + offset] if j1 + offset < j2 else ""
                columns.append({
                    "ref": ref_base,
                    "alt": alt_base,
                    "changed": True,
                    "kind": "replace",
                })
        elif tag == "delete":
            for offset in range(i2 - i1):
                columns.append({
                    "ref": reference[i1 + offset],
                    "alt": "",
                    "changed": True,
                    "kind": "delete",
                })
        elif tag == "insert":
            for offset in range(j2 - j1):
                columns.append({
                    "ref": "",
                    "alt": mutated[j1 + offset],
                    "changed": True,
                    "kind": "insert",
                })

    ref_pos = 0
    alt_pos = 0
    for column in columns:
        if column["ref"]:
            ref_pos += 1
            column["ref_pos"] = ref_pos
        else:
            column["ref_pos"] = None
        if column["alt"]:
            alt_pos += 1
            column["alt_pos"] = alt_pos
        else:
            column["alt_pos"] = None
    return columns


def _variant_column(columns: list[dict[str, Any]], variant: dict[str, Any]) -> int | None:
    start = variant["start"]
    if variant["type"] == "INSERTION":
        for index, column in enumerate(columns):
            if column.get("alt_pos") == start + 1:
                return index
        return None
    for index, column in enumerate(columns):
        if column.get("ref_pos") == start:
            return index
    for index, column in enumerate(columns):
        if column.get("alt_pos") == start:
            return index
    return None


def _codon_for_variant(
    sequence: str,
    variant: dict[str, Any],
    frame: int = 0,
) -> dict[str, Any] | None:
    """Return the reference/alternate codon and translation for a SNV."""
    if variant["type"] != "SNV" or not sequence:
        return None
    if len(variant["ref"]) != 1 or len(variant["alt"]) != 1:
        return None
    pos0 = variant["start"] - 1 - frame
    if pos0 < 0 or pos0 >= len(sequence):
        return None
    codon_start = (pos0 // 3) * 3
    codon = sequence[codon_start:codon_start + 3]
    if len(codon) != 3:
        return None
    offset = pos0 - codon_start
    if codon[offset] != variant["ref"]:
        # The injected ref allele is authoritative; use it if the window
        # starts at a different phase.
        codon = codon[:offset] + variant["ref"] + codon[offset + 1:]
    alt_codon = codon[:offset] + variant["alt"] + codon[offset + 1:]
    return {
        "codon": codon,
        "alt_codon": alt_codon,
        "offset": offset,
        "frame": frame,
        "codon_start_1based": codon_start + frame + 1,
        "from_aa": CODON_TABLE.get(codon.replace("U", "T"), "?"),
        "to_aa": CODON_TABLE.get(alt_codon.replace("U", "T"), "?"),
    }


def _placeholder(output_path: Path | None, title: str, message: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=COLORS["background"])
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=20, fontweight="bold",
            color=COLORS["text"], transform=ax.transAxes)
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=12,
            color=COLORS["muted"], wrap=True, transform=ax.transAxes)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    return fig


def _save(fig: plt.Figure, output_path: Path | None, dpi: int = 135) -> None:
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())


# ---------------------------------------------------------------------------
# 1. Genome browser
# ---------------------------------------------------------------------------

def plot_genome_browser(report: Any, output_path: Path | None = None) -> plt.Figure:
    """Show the reference/mutant DNA sequence with lesions and annotations."""
    previews = _previews(report)
    variants = _variants(report)
    if not previews or not variants:
        return _placeholder(output_path, "Genome browser",
                            "No L0 sequence preview or variants are attached to this report.")

    contig, window = _first_window(report)
    reference = _clean_sequence(window.get("baseline", ""))
    mutated = _clean_sequence(window.get("mutated", ""))
    if not reference:
        return _placeholder(output_path, "Genome browser",
                            "The L0 layer did not provide a readable sequence window.")

    columns = _alignment(reference, mutated)
    # Keep the figure readable: crop to a window around the first lesion when
    # the preview is long.
    max_columns = 96
    if len(columns) > max_columns:
        focus = next((i for i, c in enumerate(columns) if c["changed"]), 0)
        start = max(0, focus - max_columns // 2)
        columns = columns[start:start + max_columns]

    n = len(columns)
    contig_length = int(previews.get(contig, {}).get("baseline_length") or len(reference))
    window_start = int(window.get("start_1based", 1))
    synthetic = contig_length < 1000

    fig = plt.figure(figsize=(16, 10.5), facecolor=COLORS["background"])
    grid = fig.add_gridspec(3, 1, height_ratios=[1.0, 2.5, 1.45], hspace=0.35,
                            left=0.07, right=0.97, top=0.86, bottom=0.06)

    # --- contig ideogram ---------------------------------------------------
    ax0 = fig.add_subplot(grid[0])
    ax0.set_facecolor(COLORS["background"])
    ax0.set_xlim(0, max(contig_length, 1))
    ax0.set_ylim(-0.35, 1.45)
    ax0.add_patch(Rectangle((0, 0), contig_length, 0.22, facecolor="#dbe4ee",
                           edgecolor="#9aa7b4", linewidth=1.0))
    ax0.plot([0, contig_length], [0.11, 0.11], color="#9aa7b4", linewidth=0.8, alpha=0.6)
    for tick in sorted({1, contig_length, max(1, contig_length // 2)}):
        ax0.plot([tick, tick], [-0.06, 0.28], color="#6b7785", linewidth=0.8)
        ax0.text(tick, -0.16, f"{tick:,} bp", ha="center", va="top",
                 fontsize=8, color=COLORS["muted"])
    variant_colors = {"SNV": COLORS["changed"], "INSERTION": COLORS["insert"],
                      "DELETION": COLORS["delete"], "MNV": "#8e44ad"}
    for i, variant in enumerate(variants):
        x = variant["start"]
        color = variant_colors.get(variant["type"], COLORS["mutated"])
        ax0.plot([x, x], [0.22, 0.9], color=color, linewidth=1.2, alpha=0.7)
        ax0.scatter([x], [0.92], s=90, color=color, edgecolor="white", linewidth=1.2, zorder=5)
        label = f"{variant['gene']} · {variant['type']}"
        y = 1.02 + (i % 2) * 0.17
        ax0.annotate(label, xy=(x, 0.92), xytext=(x, y), ha="center", va="bottom",
                     fontsize=8.5, color=COLORS["text"],
                     arrowprops=dict(arrowstyle="-", color=color, linewidth=0.9, alpha=0.8))
    ax0.text(0, 1.34, f"Contig ideogram — {contig} ({contig_length:,} bp)",
             fontsize=12, fontweight="bold", color=COLORS["text"])
    ax0.axis("off")

    # --- sequence alignment ------------------------------------------------
    ax1 = fig.add_subplot(grid[1])
    ax1.set_facecolor(COLORS["background"])
    ax1.set_xlim(-1.0, n + 0.5)
    ax1.set_ylim(-0.8, 1.75)
    ax1.axis("off")

    ref_y, alt_y = 0.98, 0.18
    cell_h = 0.38
    for index, column in enumerate(columns):
        for base, y in ((column["ref"], ref_y), (column["alt"], alt_y)):
            if base:
                ax1.add_patch(Rectangle((index - 0.44, y - cell_h / 2), 0.88, cell_h,
                                        facecolor=BASE_COLORS.get(base, "#bdc3c7"),
                                        edgecolor="white", linewidth=0.8, alpha=0.95))
                ax1.text(index, y, base, ha="center", va="center", fontsize=12.5,
                         fontweight="bold", color="white")
            else:
                ax1.add_patch(Rectangle((index - 0.44, y - cell_h / 2), 0.88, cell_h,
                                        facecolor="none", edgecolor=COLORS["border"],
                                        linewidth=1.0, linestyle=(0, (3, 2))))
                ax1.text(index, y, "–", ha="center", va="center", fontsize=12,
                         color=COLORS["muted"])
        if column["changed"]:
            ax1.add_patch(Rectangle((index - 0.47, alt_y - cell_h / 2 - 0.03),
                                    0.94, (ref_y - alt_y) + cell_h + 0.06,
                                    facecolor="none", edgecolor=COLORS["changed"],
                                    linewidth=2.0, zorder=6))
            ax1.plot([index, index], [alt_y + cell_h / 2 + 0.02, ref_y - cell_h / 2 - 0.02],
                     color=COLORS["changed"], linewidth=1.6, alpha=0.8, zorder=4)

    ax1.text(-0.92, ref_y, "REF", ha="right", va="center", fontsize=9.5,
             fontweight="bold", color=COLORS["reference"])
    ax1.text(-0.92, alt_y, "MUT", ha="right", va="center", fontsize=9.5,
             fontweight="bold", color=COLORS["mutated"])
    ax1.text(0, 1.52, "Aligned DNA sequence (5' → 3')", fontsize=12,
             fontweight="bold", color=COLORS["text"])
    ax1.text(0, 1.36, f"Window {window_start:,}–{window_start + len(reference) - 1:,} "
             f"· outlines mark differences · red connector = injected lesion",
             fontsize=9, color=COLORS["muted"])

    # 1-based position ruler
    step = 1 if n <= 32 else max(1, n // 12)
    for index in range(0, n, step):
        ref_pos = columns[index].get("ref_pos")
        pos = window_start + ref_pos - 1 if ref_pos else window_start
        ax1.text(index, -0.42, str(pos), ha="center", va="center", fontsize=6.5,
                 color=COLORS["muted"], rotation=0 if n <= 32 else 45)
    ax1.text(0, -0.72, "1-based genomic coordinate", fontsize=8, color=COLORS["muted"])

    # Lesion callouts above the sequence.
    for i, variant in enumerate(variants):
        index = _variant_column(columns, variant)
        if index is None:
            continue
        y = 1.14 + (i % 2) * 0.14
        color = variant_colors.get(variant["type"], COLORS["mutated"])
        change = f"{variant['ref'] or '-'}→{variant['alt'] or '-'}"
        ax1.annotate(f"{variant['id']} · {variant['gene']} {change}",
                     xy=(index, ref_y + cell_h / 2 + 0.05), xytext=(index, y + 0.10),
                     ha="center", va="bottom", fontsize=8, color=color,
                     arrowprops=dict(arrowstyle="->", color=color, linewidth=1.0))

    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor=BASE_COLORS["A"], edgecolor="white", label="A"),
        Rectangle((0, 0), 1, 1, facecolor=BASE_COLORS["T"], edgecolor="white", label="T"),
        Rectangle((0, 0), 1, 1, facecolor=BASE_COLORS["G"], edgecolor="white", label="G"),
        Rectangle((0, 0), 1, 1, facecolor=BASE_COLORS["C"], edgecolor="white", label="C"),
        Line2D([0], [0], marker="s", markerfacecolor="none", markeredgecolor=COLORS["changed"],
               markeredgewidth=2, markersize=10, linestyle="None", label="changed"),
    ]
    ax1.legend(handles=legend_handles, loc="upper right", ncol=5, frameon=False,
               fontsize=8, handlelength=0.9, columnspacing=0.8, bbox_to_anchor=(1.0, 1.08))

    # --- variant cards -----------------------------------------------------
    ax2 = fig.add_subplot(grid[2])
    ax2.set_facecolor(COLORS["background"])
    ax2.axis("off")
    ax2.set_xlim(0, max(len(variants), 1))
    ax2.set_ylim(0, 1)
    l1 = _l1_details(report)
    l2 = _l2_details(report)
    for i, variant in enumerate(variants[:5]):
        detail = l1.get(variant["id"], {})
        effect = l2.get(variant["id"], {})
        color = variant_colors.get(variant["type"], COLORS["mutated"])
        ax2.add_patch(FancyBboxPatch((i + 0.04, 0.08), 0.92, 0.82,
                                     boxstyle="round,pad=0.01,rounding_size=0.03",
                                     facecolor=COLORS["panel"], edgecolor=color, linewidth=1.6))
        ax2.text(i + 0.12, 0.78, f"{variant['id']}  ·  {variant['gene']}",
                 fontsize=10, fontweight="bold", color=COLORS["text"], va="top")
        change = f"{variant['ref'] or '-'} → {variant['alt'] or '-'}"
        ax2.text(i + 0.12, 0.63, f"{variant['type']}  {variant['contig']}:{variant['start']}  {change}",
                 fontsize=8.5, color=COLORS["muted"], va="top")
        consequence = detail.get("consequence") or "not annotated"
        pathogenicity = detail.get("pathogenicity_score")
        path_text = f"pathogenicity {pathogenicity:.2f}" if isinstance(pathogenicity, (int, float)) else "pathogenicity n/a"
        ax2.text(i + 0.12, 0.44, f"L1: {consequence}", fontsize=8.2,
                 color=COLORS["text"], va="top")
        ax2.text(i + 0.12, 0.33, path_text, fontsize=7.8, color=COLORS["muted"], va="top")
        ddg = effect.get("delta_delta_g")
        ddg_text = f"L2 ΔΔG {ddg:.2f} kcal/mol" if isinstance(ddg, (int, float)) else "L2 ΔΔG n/a"
        ax2.text(i + 0.12, 0.20, ddg_text, fontsize=7.8,
                 color=COLORS["text"] if isinstance(ddg, (int, float)) else COLORS["muted"], va="top")

    fig.suptitle("Genome used in this run — sequence, lesions, and annotations",
                 fontsize=20, fontweight="bold", color=COLORS["text"], y=0.975)
    ref_uri = str(getattr(report, "reference_genome", "") or
                  getattr(getattr(getattr(report, "baseline", None), "reference", None), "uri", "") or
                  "unknown reference")
    note = f"Reference: {ref_uri}"
    if synthetic:
        note += "   ·   12-bp-scale synthetic test contig — NOT GRCh38"
    fig.text(0.5, 0.925, note, ha="center", va="center", fontsize=10,
             color=COLORS["fail"] if synthetic else COLORS["muted"])

    _save(fig, output_path)
    return fig


# ---------------------------------------------------------------------------
# 2. DNA helix
# ---------------------------------------------------------------------------

def plot_dna_helix(report: Any, output_path: Path | None = None) -> plt.Figure:
    """Render a stylized double helix with injected lesions highlighted."""
    contig, window = _first_window(report)
    reference = _clean_sequence(window.get("baseline", ""))
    variants = _variants(report)
    if not reference:
        return _placeholder(output_path, "DNA double helix",
                            "No readable L0 sequence window is attached to this report.")

    if len(reference) > 72:
        reference = reference[:72]
    n = len(reference)
    changed = set()
    for variant in variants:
        if variant["type"] == "SNV":
            changed.add(variant["start"] - 1)
        else:
            changed.add(variant["start"] - 1)

    fig = plt.figure(figsize=(16, 8.4), facecolor=COLORS["helix_bg"])
    ax = fig.add_axes([0.04, 0.10, 0.92, 0.80])
    ax.set_facecolor(COLORS["helix_bg"])

    x = np.linspace(0.8, 14.2, n) if n > 1 else np.array([0.8])
    turns = 2.2 if n <= 24 else 4.0
    phase = np.linspace(0, turns * 2 * math.pi, n)
    amplitude = 1.0
    y_left = amplitude * np.sin(phase)
    y_right = -amplitude * np.sin(phase)

    # Backbone glow + core.
    for ys, color in ((y_left, COLORS["helix_left"]), (y_right, COLORS["helix_right"])):
        for width, alpha in ((8.0, 0.08), (4.0, 0.18), (1.6, 0.9)):
            ax.plot(x, ys, color=color, linewidth=width, alpha=alpha,
                    solid_capstyle="round", zorder=3)

    # Base-pair rungs.
    for i, base in enumerate(reference):
        is_changed = i in changed
        rung_color = COLORS["changed"] if is_changed else BASE_COLORS.get(base, "#bdc3c7")
        ax.plot([x[i], x[i]], [y_left[i], y_right[i]],
                color=rung_color, linewidth=4.2 if is_changed else 1.7,
                alpha=0.98 if is_changed else 0.55, zorder=2)
        ax.scatter([x[i], x[i]], [y_left[i], y_right[i]],
                   s=42 if is_changed else 14,
                   color=COLORS["lesion"] if is_changed else BASE_COLORS.get(base, "#bdc3c7"),
                   edgecolor=COLORS["changed"] if is_changed else "none",
                   linewidth=1.0, zorder=4)
        if n <= 32:
            ax.text(x[i], y_left[i] + 0.13, base, ha="center", va="bottom",
                    fontsize=8, color=COLORS["helix_left"], alpha=0.9, zorder=5)

    # Lesion callouts.
    for i, variant in enumerate(variants):
        index = variant["start"] - 1
        if index < 0 or index >= n:
            continue
        y = max(y_left[index], y_right[index], 0.0)
        ax.annotate(
            f"{variant['gene']}  {variant['ref'] or '-'}→{variant['alt'] or '-'}  @ {variant['contig']}:{variant['start']}",
            xy=(x[index], y), xytext=(x[index], 1.55 + (i % 3) * 0.26),
            ha="center", va="bottom", fontsize=9, color=COLORS["lesion"],
            arrowprops=dict(arrowstyle="->", color=COLORS["lesion"], linewidth=1.3,
                            connectionstyle="arc3,rad=0.15"),
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#121a2e",
                      edgecolor=COLORS["changed"], linewidth=1.0),
            zorder=8,
        )

    ax.set_xlim(0, 15)
    ax.set_ylim(-1.85, 2.65)
    ax.axis("off")
    ax.set_title("DNA double helix — injected lesions highlighted",
                 fontsize=20, fontweight="bold", color="white", pad=18)
    ax.text(0.5, 1.015, f"{contig or 'contig'} · {n} bases shown · "
            "red rungs and gold nodes are altered positions",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=10,
            color="#9fb3c8")

    handles = [
        Line2D([0], [0], color=COLORS["helix_left"], lw=3, label="strand 1"),
        Line2D([0], [0], color=COLORS["helix_right"], lw=3, label="strand 2"),
        Line2D([0], [0], color=COLORS["changed"], lw=3, label="altered base pair"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.14),
              ncol=3, frameon=False, fontsize=9, labelcolor="#c9d6e3")

    # Sequence strip below the helix.
    ax.text(0.5, -0.22, "  ".join(reference), transform=ax.transAxes,
            ha="center", va="center", fontsize=10, family="monospace",
            color="#c9d6e3")

    _save(fig, output_path, dpi=140)
    return fig


# ---------------------------------------------------------------------------
# 3. Central dogma
# ---------------------------------------------------------------------------

def plot_central_dogma(report: Any, output_path: Path | None = None) -> plt.Figure:
    """Show sequence-derived codon changes for each injected variant."""
    contig, window = _first_window(report)
    sequence = _clean_sequence(window.get("baseline", ""))
    file_variants = _variants(report)
    if not sequence or not file_variants:
        return _placeholder(output_path, "Central dogma",
                            "No L0 sequence window or variants are attached to this report.")

    variants = file_variants[:4]
    l1 = _l1_details(report)
    nrows = len(variants)
    fig = plt.figure(figsize=(16, 2.55 + nrows * 2.05), facecolor=COLORS["background"])
    grid = fig.add_gridspec(nrows, 4, width_ratios=[1.05, 1.35, 1.35, 1.55],
                            hspace=0.55, wspace=0.25,
                            left=0.04, right=0.97, top=0.84, bottom=0.07)

    column_titles = ["VARIANT", "DNA CODON", "mRNA CODON", "PROTEIN"]
    for col, title in enumerate(column_titles):
        x = {0: 0.03, 1: 0.28, 2: 0.52, 3: 0.76}[col]
        fig.text(x, 0.89, title, fontsize=11, fontweight="bold",
                 color=COLORS["reference"])
    fig.text(0.5, 0.945, "Central dogma: sequence-derived codon changes",
             ha="center", va="center", fontsize=20, fontweight="bold", color=COLORS["text"])
    fig.text(0.5, 0.915, "Codons are translated directly from the L0 reference window; "
             "L1 consequence and pathogenicity are surrogate annotations.",
             ha="center", va="center", fontsize=9.5, color=COLORS["muted"])

    for row, variant in enumerate(variants):
        detail = l1.get(variant["id"], {})
        info = _codon_for_variant(sequence, variant)
        color = COLORS["changed"] if variant["type"] == "SNV" else COLORS["insert"]

        # Col 0: variant identity.
        ax0 = fig.add_subplot(grid[row, 0])
        ax0.axis("off")
        ax0.add_patch(FancyBboxPatch((0.02, 0.10), 0.94, 0.80,
                                     boxstyle="round,pad=0.02,rounding_size=0.04",
                                     facecolor=COLORS["panel"], edgecolor=color, linewidth=1.5,
                                     transform=ax0.transAxes))
        ax0.text(0.10, 0.72, variant["id"], fontsize=11, fontweight="bold",
                 color=COLORS["text"], transform=ax0.transAxes)
        ax0.text(0.10, 0.52, variant["gene"], fontsize=10, color=color,
                 transform=ax0.transAxes)
        ax0.text(0.10, 0.34, f"{variant['type']} · {variant['contig']}:{variant['start']}",
                 fontsize=8, color=COLORS["muted"], transform=ax0.transAxes)
        consequence = str(detail.get("consequence") or "annotated")
        ax0.text(0.10, 0.19, consequence.replace("_", " "), fontsize=7.8,
                 color=COLORS["muted"], transform=ax0.transAxes)

        # Col 1: DNA.
        ax1 = fig.add_subplot(grid[row, 1])
        ax1.axis("off")
        if info:
            codon = info["codon"]
            alt_codon = info["alt_codon"]
            offset = info["offset"]
            _draw_codon(ax1, codon, offset, color, label=f"reference")
            _draw_codon(ax1, alt_codon, offset, color, y=0.30, label="mutant")
            ax1.text(0.5, 0.90, f"codon {info['codon_start_1based']}–"
                     f"{info['codon_start_1based'] + 2}", ha="center", va="center",
                     fontsize=8, color=COLORS["muted"], transform=ax1.transAxes)
        elif variant["type"] == "INSERTION":
            ax1.text(0.5, 0.55, f"+{variant['alt']} insertion", ha="center", va="center",
                     fontsize=10, color=COLORS["insert"], transform=ax1.transAxes)
            ax1.text(0.5, 0.35, "frameshift / not codon-resolved", ha="center", va="center",
                     fontsize=8, color=COLORS["muted"], transform=ax1.transAxes)
        else:
            ax1.text(0.5, 0.5, "not codon-resolved", ha="center", va="center",
                     fontsize=10, color=COLORS["muted"], transform=ax1.transAxes)

        # Col 2: mRNA.
        ax2 = fig.add_subplot(grid[row, 2])
        ax2.axis("off")
        if info:
            _draw_codon(ax2, info["codon"].replace("T", "U"), info["offset"], color,
                        label="reference")
            _draw_codon(ax2, info["alt_codon"].replace("T", "U"), info["offset"], color,
                        y=0.30, label="mutant")
            ax2.text(0.5, 0.90, "transcribed (T → U)", ha="center", va="center",
                     fontsize=8, color=COLORS["muted"], transform=ax2.transAxes)
        else:
            ax2.text(0.5, 0.5, "transcript unresolved", ha="center", va="center",
                     fontsize=10, color=COLORS["muted"], transform=ax2.transAxes)

        # Col 3: protein.
        ax3 = fig.add_subplot(grid[row, 3])
        ax3.axis("off")
        if info:
            from_aa = AA_NAMES.get(info["from_aa"], "?")
            to_aa = AA_NAMES.get(info["to_aa"], "?")
            protein_pos = (info["codon_start_1based"] + info["offset"] + 2) // 3
            ax3.text(0.05, 0.68, f"{from_aa}", fontsize=18, fontweight="bold",
                     color=COLORS["muted"], transform=ax3.transAxes)
            ax3.annotate("", xy=(0.72, 0.76), xytext=(0.38, 0.76),
                         xycoords="axes fraction", textcoords="axes fraction",
                         arrowprops=dict(arrowstyle="-|>", color=color, linewidth=2.0))
            ax3.text(0.78, 0.68, f"{to_aa}", fontsize=18, fontweight="bold",
                     color=color, transform=ax3.transAxes)
            ax3.text(0.05, 0.40, f"p.{from_aa}{protein_pos}{to_aa}", fontsize=10,
                     family="monospace", color=COLORS["text"], transform=ax3.transAxes)
            effect = "missense" if info["from_aa"] != info["to_aa"] and info["to_aa"] != "*" else (
                "synonymous" if info["from_aa"] == info["to_aa"] else "stop gained")
            ax3.text(0.05, 0.24, effect, fontsize=9, color=COLORS["muted"],
                     transform=ax3.transAxes)
        else:
            ax3.text(0.05, 0.55, "frameshift" if variant["type"] in {"INSERTION", "DELETION"} else "n/a",
                     fontsize=15, fontweight="bold", color=COLORS["insert"],
                     transform=ax3.transAxes)
            ax3.text(0.05, 0.35, "downstream protein consequence", fontsize=8,
                     color=COLORS["muted"], transform=ax3.transAxes)

    _save(fig, output_path, dpi=135)
    return fig


def _draw_codon(ax, codon: str, offset: int, color: str, y: float = 0.62,
                label: str = "") -> None:
    """Draw a three-base codon with the altered position outlined."""
    if len(codon) != 3:
        return
    cell = 0.24
    start = 0.5 - (3 * cell + 0.06) / 2
    for i, base in enumerate(codon):
        x = start + i * (cell + 0.03)
        changed = i == offset
        ax.add_patch(Rectangle((x, y - 0.10), cell, 0.20,
                               facecolor=BASE_COLORS.get(base.replace("U", "T"), "#bdc3c7"),
                               edgecolor=COLORS["changed"] if changed else "white",
                               linewidth=2.4 if changed else 0.8,
                               transform=ax.transAxes, zorder=3))
        ax.text(x + cell / 2, y, base, ha="center", va="center", fontsize=13,
                fontweight="bold", color="white", transform=ax.transAxes, zorder=4)
    ax.text(start - 0.02, y, label, ha="right", va="center", fontsize=7.5,
            color=COLORS["muted"], transform=ax.transAxes)


# ---------------------------------------------------------------------------
# 4. Molecular cascade
# ---------------------------------------------------------------------------

def plot_molecular_cascade(report: Any, output_path: Path | None = None) -> plt.Figure:
    """Show the L0–L5 cascade, the engine used, and the key metric from each."""
    l0 = _layer(report, "L0")
    l1 = _layer(report, "L1")
    l2 = _layer(report, "L2")
    l3 = _layer(report, "L3")
    l4 = _layer(report, "L4")
    viability = getattr(report, "viability", None)
    variants = _variants(report)

    l1_details = l1.get("variant_details") or []
    l2_effects = l2.get("effects") or []
    l3_pathway = l3.get("pathway", {}) if isinstance(l3.get("pathway"), dict) else {}
    l3_agg = l3.get("aggregate", {}) if isinstance(l3.get("aggregate"), dict) else {}
    l4_agg = l4.get("aggregate", {}) if isinstance(l4.get("aggregate"), dict) else {}
    verdict = str(getattr(getattr(viability, "verdict", None), "name", "UNSPECIFIED"))
    composite = float(getattr(viability, "composite", 0.0) or 0.0)
    survival = getattr(viability, "survival_window_h", None)
    ponr = getattr(viability, "point_of_no_return_h", None)
    hard_gates = getattr(viability, "hard_gate_failures", []) or []
    events = getattr(viability, "irreversible_events", []) or []

    l0_contig_length = 0
    for preview in _previews(report).values():
        l0_contig_length += int(preview.get("baseline_length") or 0)
    l1_max_path = max((d.get("pathogenicity_score") or 0.0 for d in l1_details), default=0.0)
    l2_mean = max((e.get("delta_delta_g") or 0.0 for e in l2_effects), default=0.0)
    l3_biomass = l3_agg.get("biomass_flux", 0.0)
    l3_status = l3_pathway.get("status", "n/a")
    l3_critical = l3_agg.get("critical_reactions_affected", 0)
    l4_systemic = l4_agg.get("systemic_score", 0.0)
    l4_atp = l4_agg.get("mean_tissue_atp", 0.0)

    survival_text = f"{survival:.2f} h" if isinstance(survival, (int, float)) else "n/a"
    ponr_text = f"{ponr:.2f} h" if isinstance(ponr, (int, float)) else "n/a"
    l3b_metrics = (
        f"survival window {survival_text}\n"
        f"point of no return {ponr_text} · events {len(events)}"
    )

    statuses = {
        "L0": ("pass", "sequence"),
        "L1": ("warn", "annotation"),
        "L2": ("warn", "stability"),
        "L3": ("fail" if (isinstance(l3_biomass, (int, float)) and l3_biomass <= 0.01) else "pass", "metabolism"),
        "L3b": ("fail" if events else "pass", "dynamics"),
        "L4": ("warn", "physiology"),
        "L5": ("fail" if verdict == "NON_VIABLE" else "pass", "gate"),
    }

    rows = [
        ("L0", "Rust core — event-sourced genome",
         f"{len(variants)} variants · {l0_contig_length or '?'} bp reference\n"
         f"revert exact: {str((getattr(report, 'revert_hysteresis', {}) or {}).get('genome_revert_exact', 'n/a'))}"),
        ("L1", "AlphaMissense / ESM surrogate",
         f"{len(l1_details)} consequences · max pathogenicity {l1_max_path:.2f}\n"
         f"annotation: gene IDs from variant records"),
        ("L2", "ESMFold / FoldX-style ΔΔG kernel",
         f"{len(l2_effects)} protein effects · max ΔΔG {l2_mean:.2f} kcal/mol\n"
         f"domains and active-site disruption evaluated"),
        ("L3", "Human-GEM FBA + CVODE ODE",
         f"biomass {float(l3_biomass):.4f} · status {l3_status}\n"
         f"critical reactions affected: {l3_critical}"),
        ("L3b", "Time-dynamic ODE (mRNA/protein turnover)", l3b_metrics),
        ("L4", "PBPK / tissue physiology + neuro proxies",
         f"systemic score {float(l4_systemic):.2f} · mean tissue ATP {float(l4_atp):.2f}\n"
         f"neuro proxies are correlates, not consciousness"),
        ("L5", "Weighted viability gate",
         f"verdict {verdict} · composite {composite:.3f}\n"
         f"hard gates: {', '.join(hard_gates) if hard_gates else 'none'}"),
    ]

    fig = plt.figure(figsize=(16, 11.5), facecolor=COLORS["background"])
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 2.05], wspace=0.06,
                            left=0.04, right=0.97, top=0.88, bottom=0.05)
    ax_side = fig.add_subplot(grid[0])
    ax_main = fig.add_subplot(grid[1])
    ax_side.axis("off")
    ax_main.axis("off")

    fig.suptitle("Molecular cascade — what was used at each layer",
                 fontsize=21, fontweight="bold", color=COLORS["text"], y=0.965)
    fig.text(0.5, 0.925, "Each box reports the engine and the actual metric carried forward. "
             "Red boxes are failed or failed-gate layers.",
             ha="center", va="center", fontsize=10, color=COLORS["muted"])

    # Sidebar: reference and engine provenance.
    ax_side.add_patch(FancyBboxPatch((0.02, 0.02), 0.94, 0.94,
                                     boxstyle="round,pad=0.02,rounding_size=0.03",
                                     facecolor=COLORS["panel"], edgecolor=COLORS["border"],
                                     transform=ax_side.transAxes))
    ax_side.text(0.09, 0.92, "Reference & run inputs", fontsize=13,
                 fontweight="bold", color=COLORS["text"], transform=ax_side.transAxes)
    ref_uri = str(getattr(report, "reference_genome", "") or "unknown")
    sidebar_items = [
        ("Reference FASTA", ref_uri),
        ("Sequence length", f"{l0_contig_length:,} bp" if l0_contig_length else "n/a"),
        ("Gene model (GTF)", "not loaded → annotation surrogate"),
        ("Ortholog map", "Corvus cornix core-gene subset"),
        ("Rust core", "proteus_core (exact genome hash)"),
        ("C++ kernels", "proteus_kernels / CVODE + SIMD"),
        ("Evidence chains", f"{len(getattr(report, 'evidence', []) or [])} hash-linked records"),
        ("Report formats", "HTML · PDF · LaTeX · JSON-LD · RO-Crate"),
    ]
    y = 0.84
    for label, value in sidebar_items:
        ax_side.text(0.09, y, label, fontsize=8.5, fontweight="bold",
                     color=COLORS["reference"], transform=ax_side.transAxes)
        ax_side.text(0.09, y - 0.035, value, fontsize=7.8, color=COLORS["text"],
                     transform=ax_side.transAxes, wrap=True)
        y -= 0.098
    ax_side.text(0.09, 0.06, "Limitation: deterministic surrogates are\n"
                 "used where trained weights are absent.",
                 fontsize=7.6, color=COLORS["muted"], transform=ax_side.transAxes)

    # Main cascade.
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)
    nrows = len(rows)
    top = 0.94
    box_h = min(0.105, 0.86 / nrows)
    gap = (0.90 - nrows * box_h) / max(nrows - 1, 1)
    for i, (layer, engine, metrics) in enumerate(rows):
        y0 = top - i * (box_h + gap) - box_h
        status, _ = statuses[layer]
        border = {"pass": COLORS["pass"], "warn": COLORS["warn"], "fail": COLORS["fail"]}[status]
        fill = {"pass": "#eafaf0", "warn": "#fffbe6", "fail": "#fff0ef"}[status]
        ax_main.add_patch(FancyBboxPatch((0.02, y0), 0.96, box_h,
                                         boxstyle="round,pad=0.009,rounding_size=0.02",
                                         facecolor=fill, edgecolor=border, linewidth=1.6,
                                         transform=ax_main.transAxes))
        ax_main.add_patch(FancyBboxPatch((0.035, y0 + 0.012), 0.075, box_h - 0.024,
                                         boxstyle="round,pad=0.006,rounding_size=0.015",
                                         facecolor=border, edgecolor="none",
                                         transform=ax_main.transAxes))
        ax_main.text(0.072, y0 + box_h / 2, layer, ha="center", va="center",
                     fontsize=8.5, fontweight="bold", color="white",
                     transform=ax_main.transAxes)
        ax_main.text(0.125, y0 + box_h * 0.70, engine, fontsize=9.5, fontweight="bold",
                     color=COLORS["text"], transform=ax_main.transAxes)
        ax_main.text(0.125, y0 + box_h * 0.37, metrics, fontsize=7.9,
                     color=COLORS["muted"], transform=ax_main.transAxes, linespacing=1.25)
        if i < nrows - 1:
            ax_main.annotate("", xy=(0.5, y0 - 0.003), xytext=(0.5, y0 + gap + 0.003),
                             xycoords="axes fraction", textcoords="axes fraction",
                             arrowprops=dict(arrowstyle="-|>", color=COLORS["border"],
                                             linewidth=2.0, shrinkA=0, shrinkB=0))

    _save(fig, output_path, dpi=135)
    return fig


# ---------------------------------------------------------------------------
# 5. Inputs provenance
# ---------------------------------------------------------------------------

def plot_inputs_provenance(report: Any, output_path: Path | None = None) -> plt.Figure:
    """Show reference inputs, model hooks, and compute engines in one panel."""
    l0 = _layer(report, "L0")
    variants = _variants(report)
    contig_length = sum(
        int(p.get("baseline_length") or 0)
        for p in (_previews(report).values() if isinstance(_previews(report), dict) else [])
    )

    model_rows: list[tuple[str, str, str]] = []
    try:
        from proteus.models.model_registry import get_model_registry

        registry = get_model_registry()
        for config in registry.list_available():
            path = Path(str(config.path)) if config.path else None
            has_weights = bool(path and path.exists())
            if has_weights:
                status, color = "weights present", COLORS["pass"]
            else:
                status, color = "surrogate (no local weights)", COLORS["warn"]
            model_rows.append((config.name, f"{config.type.value} · v{config.version}", status))
    except Exception:
        model_rows = [
            ("AlphaMissense", "variant effect", "surrogate (no local weights)"),
            ("ESM2-650M", "variant effect", "surrogate (no local weights)"),
            ("FoldX", "protein stability", "surrogate (no local weights)"),
            ("ESMFold", "protein stability", "surrogate (no local weights)"),
            ("Human-GEM", "metabolic flux", "surrogate (no local weights)"),
            ("PBPK-Human", "systemic PBPK", "surrogate (no local weights)"),
        ]

    try:
        from proteus.cpp_adapter import cpp_status

        cpp = cpp_status()
        cpp_text = "available" if cpp.get("available") else "unavailable (Python fallback)"
        cpp_color = COLORS["pass"] if cpp.get("available") else COLORS["fail"]
    except Exception:
        cpp_text, cpp_color = "unknown", COLORS["muted"]

    rust_text = "active (exact revert)" if l0 else "not reported"
    fig = plt.figure(figsize=(16, 9), facecolor=COLORS["background"])
    ax = fig.add_axes([0.03, 0.05, 0.94, 0.80])
    ax.axis("off")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1)

    fig.suptitle("What this simulation actually used",
                 fontsize=21, fontweight="bold", color=COLORS["text"], y=0.965)
    fig.text(0.5, 0.925, "Genomic inputs, trained-model hooks, and compute engines. "
             "Yellow items are deterministic surrogates in this build.",
             ha="center", va="center", fontsize=10, color=COLORS["muted"])

    columns = [
        ("GENOMIC INPUTS", COLORS["reference"], [
            ("Reference FASTA", str(getattr(report, "reference_genome", "") or "unknown"), COLORS["muted"]),
            ("Contig length", f"{contig_length:,} bp" if contig_length else "n/a", COLORS["muted"]),
            ("Variants injected", str(len(variants)), COLORS["mutated"]),
            ("Gene model (GTF)", "not loaded — positional/annotation surrogate", COLORS["warn"]),
            ("Ortholog mapping", "Corvus cornix core-gene subset", COLORS["warn"]),
            ("Coordinates", "1-based inclusive", COLORS["muted"]),
        ]),
        ("TRAINED-MODEL HOOKS", COLORS["mutated"], [
            (name, kind, status) for name, kind, status in model_rows[:6]
        ]),
        ("COMPUTE & EVIDENCE", COLORS["revert"] if "revert" in COLORS else COLORS["pass"], [
            ("Rust core", "event-sourced genome state", rust_text),
            ("C++ kernels", "CVODE / SIMD / ΔΔG", cpp_text),
            ("Python", "orchestration + reports", "active"),
            ("Validation", "V1–V7 runtime checks", "see report"),
            ("Evidence", "hash-linked chain", f"{len(getattr(report, 'evidence', []) or [])} records"),
            ("Artifacts", "multi-format bundle", "HTML · PDF · TeX · JSON-LD · RO-Crate"),
        ]),
    ]

    for col, (title, color, rows) in enumerate(columns):
        x0 = col * 1.0 + 0.025
        ax.add_patch(FancyBboxPatch((x0, 0.02), 0.95, 0.94,
                                    boxstyle="round,pad=0.01,rounding_size=0.02",
                                    facecolor=COLORS["panel"], edgecolor=color, linewidth=1.5))
        ax.add_patch(FancyBboxPatch((x0 + 0.025, 0.84), 0.90, 0.075,
                                    boxstyle="round,pad=0.005,rounding_size=0.015",
                                    facecolor=color, edgecolor="none"))
        ax.text(x0 + 0.47, 0.877, title, ha="center", va="center", fontsize=10.5,
                fontweight="bold", color="white")
        y = 0.775
        for label, value, status in rows:
            ax.text(x0 + 0.05, y, label, fontsize=8.4, fontweight="bold",
                    color=COLORS["text"], va="top")
            ax.text(x0 + 0.05, y - 0.042, value, fontsize=7.6,
                    color=COLORS["muted"], va="top", wrap=True)
            if status in {COLORS["pass"], COLORS["warn"], COLORS["fail"], COLORS["muted"]}:
                ax.add_patch(Circle((x0 + 0.885, y - 0.008), 0.014, color=status,
                                    transform=ax.transData))
            y -= 0.135
        ax.text(x0 + 0.05, 0.07, "status dot: green = available · yellow = surrogate · red = missing",
                fontsize=6.8, color=COLORS["muted"])

    _save(fig, output_path, dpi=135)
    return fig


# ---------------------------------------------------------------------------
# 6. Glycolysis map
# ---------------------------------------------------------------------------

GLYCOLYSIS_ENZYMES = [
    ("HK", ["HK1", "HK2", "GCK"], "Glucose → G6P"),
    ("GPI", ["GPI"], "G6P → F6P"),
    ("PFK", ["PFKM", "PFKL", "PFKP"], "F6P → F1,6BP"),
    ("ALDO", ["ALDOA", "ALDOB", "ALDOC"], "F1,6BP → GAP"),
    ("TPI", ["TPI1"], "GAP ↔ DHAP"),
    ("GAPDH", ["GAPDH"], "GAP → 1,3BPG"),
    ("PGK", ["PGK1"], "1,3BPG → 3PG"),
    ("PGAM", ["PGAM1"], "3PG → 2PG"),
    ("ENO", ["ENO1", "ENO2", "ENO3"], "2PG → PEP"),
    ("PK", ["PKM", "PKLR"], "PEP → pyruvate"),
    ("LDH", ["LDHA", "LDHB"], "pyruvate → lactate"),
]


def plot_glycolysis_map(report: Any, output_path: Path | None = None) -> plt.Figure:
    """Glycolysis map annotated with the expression values used by L3."""
    l3 = _layer(report, "L3")
    grn = l3.get("grn", {}) if isinstance(l3.get("grn"), dict) else {}
    expression = grn.get("expression", {}) if isinstance(grn.get("expression"), dict) else {}
    pathway = l3.get("pathway", {}) if isinstance(l3.get("pathway"), dict) else {}
    aggregate = l3.get("aggregate", {}) if isinstance(l3.get("aggregate"), dict) else {}

    if not expression and not pathway:
        return _placeholder(output_path, "Glycolysis map",
                            "No L3 pathway or GRN payload is attached to this report.")

    modeled_values = []
    for _, genes, _ in GLYCOLYSIS_ENZYMES:
        vals = [float(expression[g]) for g in genes if g in expression]
        if vals:
            modeled_values.append(sum(vals))
    max_value = max(modeled_values) if modeled_values else 1.0

    fig = plt.figure(figsize=(16, 9.2), facecolor=COLORS["background"])
    ax = fig.add_axes([0.04, 0.07, 0.70, 0.78])
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(-1.0, 3.5)

    fig.suptitle("Glycolysis — the pathway state the model actually used",
                 fontsize=20, fontweight="bold", color=COLORS["text"], y=0.965)
    fig.text(0.5, 0.925, "Box colour and value come from the L3 GRN expression vector. "
             "Zero-expression nodes are the injected knockouts.",
             ha="center", va="center", fontsize=10, color=COLORS["muted"])

    # Coordinates: serpentine layout.
    top_y, bottom_y = 2.6, 0.75
    top_x = np.linspace(0.9, 8.7, 6)
    bottom_x = np.linspace(8.7, 1.9, 5)
    positions = []
    for i in range(6):
        positions.append((float(top_x[i]), top_y))
    for i in range(5):
        positions.append((float(bottom_x[i]), bottom_y))

    for i, ((short, genes, reaction), (x, y)) in enumerate(zip(GLYCOLYSIS_ENZYMES, positions)):
        vals = [float(expression[g]) for g in genes if g in expression]
        value = sum(vals) if vals else None
        if value is None:
            color = "#eef1f5"
            text_color = COLORS["muted"]
            value_text = "not modeled"
        elif value <= 1e-9:
            color = "#ffd6d3"
            text_color = COLORS["fail"]
            value_text = "KO / 0.0"
        else:
            frac = min(1.0, value / max_value) if max_value else 0.0
            color = plt.cm.Greens(0.22 + 0.58 * frac)
            text_color = COLORS["text"]
            value_text = f"{value:.1f}"

        ax.add_patch(FancyBboxPatch((x - 0.62, y - 0.34), 1.24, 0.68,
                                    boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor=color, edgecolor=COLORS["border"], linewidth=1.2))
        ax.text(x, y + 0.13, short, ha="center", va="center", fontsize=11,
                fontweight="bold", color=text_color)
        ax.text(x, y - 0.08, value_text, ha="center", va="center", fontsize=7.8,
                color=text_color)
        ax.text(x, y - 0.24, reaction, ha="center", va="center", fontsize=6.2,
                color=COLORS["muted"])

        # Products / cofactors.
        if short == "HK":
            ax.text(x, y + 0.50, "ATP → ADP", ha="center", fontsize=6.8, color=COLORS["reference"])
        if short == "PFK":
            ax.text(x, y + 0.50, "ATP → ADP", ha="center", fontsize=6.8, color=COLORS["reference"])
        if short == "GAPDH":
            ax.text(x, y + 0.50, "NAD+ → NADH", ha="center", fontsize=6.8, color=COLORS["reference"])
        if short == "PGK":
            ax.text(x, y + 0.50, "ADP → ATP", ha="center", fontsize=6.8, color=COLORS["pass"])
        if short == "PK":
            ax.text(x, y + 0.50, "ADP → ATP", ha="center", fontsize=6.8, color=COLORS["pass"])
        if short == "LDH":
            ax.text(x, y + 0.50, "NADH → NAD+", ha="center", fontsize=6.8, color=COLORS["reference"])

        if i < len(positions) - 1:
            nx, ny = positions[i + 1]
            ax.annotate("", xy=(nx - 0.66 if nx > x else nx + 0.66, ny if ny == y else ny + 0.34),
                        xytext=(x + 0.66 if nx > x else x - 0.66, y if ny == y else y - 0.34),
                        arrowprops=dict(arrowstyle="-|>", color=COLORS["border"],
                                        linewidth=1.6, shrinkA=0, shrinkB=0))

    # Side panel: L3 summary.
    ax_side = fig.add_axes([0.76, 0.10, 0.22, 0.73])
    ax_side.axis("off")
    ax_side.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                                     boxstyle="round,pad=0.02,rounding_size=0.03",
                                     facecolor=COLORS["panel"], edgecolor=COLORS["border"],
                                     linewidth=1.2, transform=ax_side.transAxes))
    ax_side.text(0.08, 0.93, "L3 pathway readout", fontsize=12.5, fontweight="bold",
                 color=COLORS["text"], transform=ax_side.transAxes)
    status = str(pathway.get("status", "n/a"))
    status_color = COLORS["fail"] if status == "infeasible" else COLORS["pass"]
    summary = [
        ("Model", str(pathway.get("model_id", "n/a"))),
        ("FBA status", status),
        ("Biomass flux", f"{float(aggregate.get('biomass_flux', 0.0)):.4f}"),
        ("ATP pool", f"{float(aggregate.get('atp_pool', 0.0)):.2f}"),
        ("ADP pool", f"{float(aggregate.get('adp_pool', 0.0)):.2f}"),
        ("Critical reactions", str(aggregate.get("critical_reactions_affected", "n/a"))),
        ("ODE engine", str(aggregate.get("ode_status", "n/a"))),
    ]
    y = 0.84
    for label, value in summary:
        ax_side.text(0.08, y, label, fontsize=8.2, fontweight="bold",
                     color=COLORS["reference"], transform=ax_side.transAxes)
        ax_side.text(0.08, y - 0.035, value, fontsize=8.0,
                     color=status_color if label == "FBA status" else COLORS["text"],
                     transform=ax_side.transAxes)
        y -= 0.11

    ax_side.text(0.08, 0.08, "Red boxes have expression ≈ 0\n"
                 "because they were knocked out.",
                 fontsize=7.4, color=COLORS["muted"], transform=ax_side.transAxes)

    _save(fig, output_path, dpi=135)
    return fig


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def create_genome_views(report: Any, output_dir: Path | str) -> dict[str, Path]:
    """Generate all genome/molecule-centric figures and return their paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "genome_browser": output_dir / "genome_dna_browser.png",
        "dna_helix": output_dir / "genome_dna_helix.png",
        "central_dogma": output_dir / "genome_central_dogma.png",
        "molecular_cascade": output_dir / "genome_molecular_cascade.png",
        "inputs_provenance": output_dir / "genome_inputs_provenance.png",
        "glycolysis_map": output_dir / "genome_glycolysis_map.png",
    }
    generators = {
        "genome_browser": plot_genome_browser,
        "dna_helix": plot_dna_helix,
        "central_dogma": plot_central_dogma,
        "molecular_cascade": plot_molecular_cascade,
        "inputs_provenance": plot_inputs_provenance,
        "glycolysis_map": plot_glycolysis_map,
    }
    for key, generator in generators.items():
        fig = None
        try:
            fig = generator(report, paths[key])
        finally:
            if fig is not None:
                plt.close(fig)
    plt.close("all")
    return paths
