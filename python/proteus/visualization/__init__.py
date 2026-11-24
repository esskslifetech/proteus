"""
Visualization module for Proteus: multi-image simulation outputs including
static plots, interactive diagrams, heatmaps, pathway graphs, and audio sonification.
"""

from __future__ import annotations

from .plots import (
    plot_viability_radar,
    plot_layer_diffs_heatmap,
    plot_variant_effects,
    plot_energy_landscape,
    plot_conservation_profile,
    plot_validation_summary,
    create_dashboard,
)
from .graphs import (
    plot_pathway_graph,
    plot_ppi_network,
    plot_gene_regulatory_network,
    plot_mutation_impact_graph,
)
from .sonification import (
    sonify_genome,
    sonify_mutation,
    sonify_pathway,
    save_audio,
    create_comparison_audio,
)
from .animation import (
    animate_mutation_effect,
    animate_layer_propagation,
    create_gif,
)
from .genome_views import (
    plot_genome_browser,
    plot_dna_helix,
    plot_central_dogma,
    plot_molecular_cascade,
    plot_inputs_provenance,
    plot_glycolysis_map,
    create_genome_views,
    VIEW_TITLES,
)

__all__ = [
    # Static plots
    "plot_viability_radar",
    "plot_layer_diffs_heatmap",
    "plot_variant_effects",
    "plot_energy_landscape",
    "plot_conservation_profile",
    "plot_validation_summary",
    "create_dashboard",
    # Genome / molecule visualizations
    "plot_genome_browser",
    "plot_dna_helix",
    "plot_central_dogma",
    "plot_molecular_cascade",
    "plot_inputs_provenance",
    "plot_glycolysis_map",
    "create_genome_views",
    "VIEW_TITLES",
    # Graph visualizations
    "plot_pathway_graph",
    "plot_ppi_network",
    "plot_gene_regulatory_network",
    "plot_mutation_impact_graph",
    # Audio sonification
    "sonify_genome",
    "sonify_mutation",
    "sonify_pathway",
    "save_audio",
    "create_comparison_audio",
    # Animations
    "animate_mutation_effect",
    "animate_layer_propagation",
    "create_gif",
]