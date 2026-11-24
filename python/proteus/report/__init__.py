"""
Report generation: HTML, PDF, LaTeX, JSON-LD, RO-Crate evidence bundles.

Every format is generated from the same :class:`SimulationReport` object.  The
HTML report is the canonical human-readable artifact; PDF is produced from the
same HTML through WeasyPrint when the optional ``reports`` extra is installed.
All other formats are dependency-free fallbacks so a run always emits a
complete evidence bundle.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    """Best-effort recursive conversion of schema objects to JSON-safe values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "name") and hasattr(value, "value"):  # Enum
        return value.name
    if hasattr(value, "__dict__"):
        return {k: _jsonable(v) for k, v in vars(value).items() if not k.startswith("_")}
    return str(value)


def _hash_string(content_hash: Any) -> str:
    if content_hash is None:
        return ""
    algorithm = getattr(content_hash, "algorithm", "") or ""
    digest = getattr(content_hash, "digest", b"") or b""
    if isinstance(digest, bytes):
        digest = digest.hex()
    if algorithm and digest:
        return f"{algorithm}:{digest}"
    return str(digest or "")


def _variant_type_name(variant: Any) -> str:
    vtype = getattr(variant, "type", "")
    return str(getattr(vtype, "name", vtype))


@dataclass
class ReportContext:
    """Flattened, template-friendly view over a SimulationReport."""

    simulation_id: str
    generated_at: str
    timestamp_ns: int
    reference_uri: str
    reference_hash: str
    state_baseline_hash: str
    state_mutated_hash: str
    state_reverted_hash: str
    provenance: str
    variants: list[dict[str, Any]]
    viability: dict[str, Any]
    diffs: list[dict[str, Any]]
    validation: list[dict[str, Any]]
    validation_compact: dict[str, str]
    evidence: list[dict[str, Any]]
    layers: dict[str, Any]
    limitations: list[str]
    environment: dict[str, str]
    revert_hysteresis: dict[str, Any]
    visualizations: list[dict[str, Any]]


def _load_layers(report: Any) -> dict[str, Any]:
    """Pull layer payloads attached to the report by the pipeline (if any)."""
    layers = getattr(report, "layer_results", None)
    if isinstance(layers, dict):
        return layers
    # The pipeline may attach rich validation details; layers are optional.
    return {}


def build_context(report: Any) -> ReportContext:
    """Normalize a SimulationReport into a ReportContext."""
    baseline = getattr(report, "baseline", None)
    mutated = getattr(report, "mutated", None)
    reverted = getattr(report, "reverted", None)

    def state_hash(state: Any) -> str:
        if state is None:
            return ""
        try:
            return state.hash()
        except Exception:
            return ""

    variants = []
    for v in getattr(report, "applied_variants", []) or []:
        variants.append({
            "id": getattr(v, "id", ""),
            "type": _variant_type_name(v),
            "contig": getattr(v, "contig", ""),
            "start": getattr(v, "start", 0),
            "end": getattr(v, "end", 0),
            "ref": getattr(v, "ref_allele", ""),
            "alt": getattr(v, "alt_allele", ""),
            "source": getattr(v, "source", ""),
            "annotations": _jsonable(getattr(v, "annotations", {}) or {}),
        })

    viability_obj = getattr(report, "viability", None)
    verdict = getattr(viability_obj, "verdict", None)
    viability = {
        "composite": float(getattr(viability_obj, "composite", 0.0) or 0.0),
        "verdict": str(getattr(verdict, "name", verdict or "UNSPECIFIED")),
        "confidence": float(getattr(viability_obj, "confidence", 0.0) or 0.0),
        "layer_scores": _jsonable(getattr(viability_obj, "layer_scores", {}) or {}),
        "weights": _jsonable(getattr(viability_obj, "weights", {}) or {}),
        "failed_gates": _jsonable(getattr(viability_obj, "failed_gates", []) or []),
        "survival_window_h": getattr(viability_obj, "survival_window_h", None),
        "point_of_no_return_h": getattr(viability_obj, "point_of_no_return_h", None),
        "irreversible_events": _jsonable(getattr(viability_obj, "irreversible_events", []) or []),
        "swap_gate_results": _jsonable(getattr(viability_obj, "swap_gate_results", {}) or {}),
        "hard_gate_failures": _jsonable(getattr(viability_obj, "hard_gate_failures", []) or []),
    }

    diffs = []
    for diff in getattr(report, "diffs", []) or []:
        changed = getattr(diff, "changed", {}) or {}
        diffs.append({
            "layer": getattr(diff, "layer", ""),
            "added_count": len(getattr(diff, "added", {}) or {}),
            "removed_count": len(getattr(diff, "removed", {}) or {}),
            "changed_count": len(changed),
            "added": _jsonable(getattr(diff, "added", {}) or {}),
            "removed": _jsonable(getattr(diff, "removed", {}) or {}),
            "changed": _jsonable(changed),
        })

    validation_details = getattr(report, "validation_details", None) or {}
    validation = []
    for name, compact in sorted((getattr(report, "validation_results", None) or {}).items()):
        detail = validation_details.get(name, {})
        validation.append({
            "name": name,
            "status": compact,
            "message": detail.get("message", ""),
            "details": _jsonable(detail.get("details", {})),
        })

    evidence = []
    for chain in getattr(report, "evidence", []) or []:
        confidence = getattr(chain, "confidence", None)
        evidence.append({
            "claim": getattr(chain, "claim", ""),
            "tool": getattr(chain, "tool", ""),
            "params": _jsonable(getattr(chain, "params", {}) or {}),
            "confidence": str(getattr(confidence, "name", confidence or "")),
            "validation_passes": _jsonable(getattr(chain, "validation_passes", []) or []),
            "input_hash": _hash_string(getattr(chain, "input_hash", None)),
            "output_hash": _hash_string(getattr(chain, "output_hash", None)),
            "upstream_ref": getattr(chain, "upstream_ref", ""),
        })

    layers = _load_layers(report)
    limitations = getattr(report, "limitations", None)
    if not limitations:
        limitations = [
            "This simulation is a mechanistic model, not a clinical or experimental result.",
            "Variant effect predictors are deterministic surrogates, not trained ML models, "
            "until the optional ML extras are wired in.",
            "Neurophysiological proxies (EEG power, PCI, GNW activation, Phi approximations) "
            "do NOT model consciousness or subjective experience. They are reported only as "
            "named, citable physiological correlates, with the limitations stated in the report.",
            "SCALE: The test genome is a 12 bp synthetic contig, NOT the human genome. "
            "All 'human genome' references in this report refer to this toy model.",
            "GENE ANNOTATION: The test contig has no gene model (gene_id: None). "
            "missense_variant predictions are made without transcript context and are NOT "
            "biologically validated. A real transcript model (GTF/GFF) is required for "
            "credible variant annotation.",
            "NEURO PROXY DEPENDENCIES: All neuro proxies (PCI, GNW, Phi, LZ, EEG bands) "
            "are derived from ~2 degrees of freedom (a single variant hash). They are "
            "perfectly correlated (r ≈ 1.0) and do NOT represent independent measurements. "
            "They are surrogates for demonstration only.",
        ]

    reference = getattr(baseline, "reference", None)
    provenance = _hash_string(getattr(report, "provenance_manifest", None))

    # Static figures embedded in the HTML/PDF report. Paths are relative to
    # the report directory; RO-Crate generation can override the prefix.
    viz_prefix = str(getattr(report, "_viz_path_prefix", "") or "").strip("/")
    visualizations: list[dict[str, Any]] = []
    for viz in getattr(report, "visualizations", []) or []:
        if not isinstance(viz, dict):
            continue
        rel_path = str(viz.get("path", "")).replace("\\", "/")
        if not rel_path:
            continue
        if viz_prefix:
            rel_path = f"{viz_prefix}/{rel_path}"
        visualizations.append({
            "key": str(viz.get("key", "")),
            "path": rel_path,
            "title": str(viz.get("title", "")),
            "caption": str(viz.get("caption", "")),
        })

    return ReportContext(
        simulation_id=getattr(report, "simulation_id", ""),
        generated_at=datetime.now(timezone.utc).isoformat(),
        timestamp_ns=int(getattr(report, "timestamp_ns", 0) or 0),
        reference_uri=getattr(reference, "uri", "") or "",
        reference_hash=_hash_string(getattr(reference, "hash", None)),
        state_baseline_hash=state_hash(baseline),
        state_mutated_hash=state_hash(mutated),
        state_reverted_hash=state_hash(reverted),
        provenance=provenance,
        variants=variants,
        viability=viability,
        diffs=diffs,
        validation=validation,
        validation_compact=dict(getattr(report, "validation_results", {}) or {}),
        evidence=evidence,
        layers=layers,
        limitations=[str(x) for x in limitations],
        environment={
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        revert_hysteresis=_jsonable(getattr(report, "revert_hysteresis", {}) or {}),
        visualizations=visualizations,
    )


_TEMPLATE_NAME = "report.html.jinja"


class ReportGenerator:
    """Generates multi-format reports with evidence chains."""

    def __init__(
        self,
        output_dir: Path | str,
        template_dir: Path | str | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.template_dir = Path(template_dir) if template_dir else Path(__file__).parent / "templates"

    # ------------------------------------------------------------------ HTML
    def _render_html(self, context: ReportContext) -> str:
        template_path = self.template_dir / _TEMPLATE_NAME
        try:
            from jinja2 import Environment, FileSystemLoader, select_autoescape

            env = Environment(
                loader=FileSystemLoader(str(self.template_dir)),
                autoescape=select_autoescape(["html", "xml"]),
                trim_blocks=True,
                lstrip_blocks=True,
            )
            template = env.get_template(_TEMPLATE_NAME)
            return template.render(**vars(context))
        except Exception:
            # Minimal dependency-free fallback if Jinja2 or the template is absent.
            rows = "".join(
                f"<tr><td>{v['id']}</td><td>{v['type']}</td><td>{v['contig']}</td>"
                f"<td>{v['start']}</td><td>{v['ref']}&rarr;{v['alt']}</td></tr>"
                for v in context.variants
            )
            checks = "".join(
                f"<li>{c['name']}: <strong>{c['status']}</strong> {c['message']}</li>"
                for c in context.validation
            )
            return (
                "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                "<title>Proteus Report</title></head><body>"
                f"<h1>Proteus Simulation Report</h1>"
                f"<p>Run: {context.simulation_id}</p>"
                f"<p>Verdict: {context.viability['verdict']} "
                f"({context.viability['composite']:.4f})</p>"
                f"<table border='1'><tr><th>ID</th><th>Type</th><th>Contig</th>"
                f"<th>Start</th><th>Change</th></tr>{rows}</table>"
                f"<h2>Validation</h2><ul>{checks}</ul>"
                "</body></html>"
            )

    def generate_html(self, report: Any, output_path: Path | None = None) -> Path:
        """Generate the HTML report using the Jinja2 template."""
        output_path = Path(output_path) if output_path else self.output_dir / "report.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        context = build_context(report)
        output_path.write_text(self._render_html(context), encoding="utf-8")
        return output_path

    # ------------------------------------------------------------------- PDF
    def generate_pdf(self, report: Any, output_path: Path | None = None) -> Path:
        """Generate a PDF from the report using reportlab."""
        output_path = Path(output_path) if output_path else self.output_dir / "report.pdf"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch, cm
            from reportlab.lib.colors import HexColor
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                PageBreak, KeepTogether, ListFlowable, ListItem
            )
            from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
            from reportlab.lib import colors

            doc = SimpleDocTemplate(
                str(output_path),
                pagesize=A4,
                rightMargin=2*cm,
                leftMargin=2*cm,
                topMargin=2*cm,
                bottomMargin=2*cm,
            )
            styles = getSampleStyleSheet()
            styles.add(ParagraphStyle(name='CustomTitle', fontSize=20, leading=24, alignment=TA_CENTER, spaceAfter=12, textColor=HexColor('#1a1a2e')))
            styles.add(ParagraphStyle(name='CustomHeading', fontSize=14, leading=18, spaceBefore=12, spaceAfter=6, textColor=HexColor('#16213e')))
            styles.add(ParagraphStyle(name='CustomBody', fontSize=10, leading=13, spaceAfter=4, alignment=TA_JUSTIFY))
            styles.add(ParagraphStyle(name='CustomCode', fontSize=9, leading=11, fontName='Courier', backColor=HexColor('#f0f0f0'), borderPadding=4))

            ctx = build_context(report)
            story = []

            # Title
            story.append(Paragraph(f"Proteus Simulation Report", styles['CustomTitle']))
            story.append(Spacer(1, 6))
            story.append(Paragraph(f"Simulation ID: {ctx.simulation_id}", styles['CustomBody']))
            story.append(Paragraph(f"Generated: {ctx.generated_at}", styles['CustomBody']))
            story.append(Paragraph(f"Provenance: {ctx.provenance}", styles['CustomBody']))
            story.append(Spacer(1, 12))

            # Variants
            if ctx.variants:
                story.append(Paragraph("Injected Variants", styles['CustomHeading']))
                data = [['Type', 'Contig', 'Start', 'End', 'Ref', 'Alt', 'ID']]
                for v in ctx.variants:
                    data.append([
                        _variant_type_name(v),
                        str(v.get('contig', '')),
                        str(v.get('start', '')),
                        str(v.get('end', '')),
                        str(v.get('ref', '')),
                        str(v.get('alt', '')),
                        str(v.get('id', ''))[:12],
                    ])
                t = Table(data, repeatRows=1)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#16213e')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                story.append(t)
                story.append(Spacer(1, 12))

            # Viability
            if ctx.viability:
                story.append(Paragraph("Viability Assessment", styles['CustomHeading']))
                data = [['Metric', 'Value']]
                for k, v in ctx.viability.items():
                    data.append([str(k).replace('_', ' ').title(), str(v)])
                t = Table(data)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#16213e')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ]))
                story.append(t)
                story.append(Spacer(1, 12))

            # Diffs
            if ctx.diffs:
                story.append(Paragraph("Layer Diffs", styles['CustomHeading']))
                data = [['Layer', 'Metric', 'Ref', 'Mut', 'Delta', 'Rel %', 'Significance']]
                for d in ctx.diffs:
                    data.append([
                        str(d.get('layer', '')),
                        str(d.get('metric', '')),
                        str(d.get('reference', '')),
                        str(d.get('mutated', '')),
                        f"{float(d.get('delta', 0)):.4g}",
                        f"{float(d.get('rel', 0))*100:.2f}%",
                        str(d.get('significance', '')),
                    ])
                t = Table(data, repeatRows=1)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#16213e')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 7),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                story.append(t)
                story.append(Spacer(1, 12))

            # Validations
            if ctx.validation:
                story.append(Paragraph("Validation Results", styles['CustomHeading']))
                data = [['Check', 'Status', 'Details']]
                for v in ctx.validation:
                    data.append([
                        str(v.get('check', '')),
                        str(v.get('status', '')),
                        str(v.get('details', ''))[:80],
                    ])
                t = Table(data)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#16213e')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ]))
                story.append(t)
                story.append(Spacer(1, 12))

            # Layers summary
            if ctx.layers:
                story.append(Paragraph("Layer Results", styles['CustomHeading']))
                for layer_name, layer_data in ctx.layers.items():
                    story.append(Paragraph(f"  {layer_name}", styles['CustomBody']))
                    if isinstance(layer_data, dict):
                        for k, v in list(layer_data.items())[:10]:
                            story.append(Paragraph(f"    {k}: {v}", styles['CustomBody']))
                story.append(Spacer(1, 12))

            # Embedded molecular/static figures
            if ctx.visualizations:
                story.append(PageBreak())
                story.append(Paragraph("Molecular Visualizations", styles['CustomHeading']))
                story.append(Paragraph(
                    "Static figures generated from this run. Full-resolution files are "
                    "in the visualizations/ directory.", styles['CustomBody']
                ))
                story.append(Spacer(1, 8))
                # Keep the PDF focused: genome/molecule figures + the two most
                # information-dense dashboards.
                selected = [
                    v for v in ctx.visualizations
                    if v["key"].startswith("genome_")
                    or v["key"] in {"composite", "energy", "variants"}
                ]
                for viz in selected:
                    image_path = (Path(self.output_dir) / viz["path"]).resolve()
                    if not image_path.is_file() or image_path.suffix.lower() != ".png":
                        continue
                    try:
                        from reportlab.lib.utils import ImageReader
                        from reportlab.platypus import Image as RLImage

                        iw, ih = ImageReader(str(image_path)).getSize()
                        max_w = 17 * cm
                        max_h = 19 * cm
                        scale = min(max_w / iw, max_h / ih)
                        story.append(RLImage(str(image_path), width=iw * scale, height=ih * scale))
                        story.append(Spacer(1, 4))
                        story.append(Paragraph(viz["title"], styles['CustomBody']))
                        story.append(Spacer(1, 10))
                    except Exception:
                        continue

            # Limitations
            if ctx.limitations:
                story.append(Paragraph("Limitations & Caveats", styles['CustomHeading']))
                for lim in ctx.limitations:
                    story.append(Paragraph(f"• {lim}", styles['CustomBody']))
                story.append(Spacer(1, 12))

            # Environment
            if ctx.environment:
                story.append(Paragraph("Environment", styles['CustomHeading']))
                data = [['Key', 'Value']]
                for k, v in ctx.environment.items():
                    data.append([k, v])
                t = Table(data)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#16213e')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ]))
                story.append(t)

            doc.build(story)
            return output_path
        except Exception as e:
            # Fallback: write a minimal PDF stub
            if not output_path.exists():
                output_path.write_bytes(
                    b"%PDF-1.4\n% Proteus report PDF not rendered: "
                    + str(e).encode('utf-8', errors='replace') + b"\n"
                )
            return output_path

    # ----------------------------------------------------------------- LaTeX
    def generate_latex(self, report: Any, output_path: Path | None = None) -> Path:
        """Generate a standalone LaTeX document."""
        output_path = Path(output_path) if output_path else self.output_dir / "report.tex"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ctx = build_context(report)

        def esc(text: Any) -> str:
            return (
                str(text)
                .replace("\\", r"\textbackslash{}")
                .replace("&", r"\&")
                .replace("%", r"\%")
                .replace("$", r"\$")
                .replace("#", r"\#")
                .replace("_", r"\_")
                .replace("{", r"\{")
                .replace("}", r"\}")
                .replace("~", r"\textasciitilde{}")
                .replace("^", r"\textasciicircum{}")
            )

        lines = [
            r"\documentclass[11pt]{article}",
            r"\usepackage[margin=1in]{geometry}",
            r"\usepackage{longtable}",
            r"\usepackage{graphicx}",
            r"\usepackage{hyperref}",
            r"\title{Proteus Simulation Report}",
            rf"\date{{{esc(ctx.generated_at)}}}",
            r"\begin{document}",
            r"\maketitle",
            r"\section*{Run summary}",
            rf"\textbf{{Simulation ID:}} \texttt{{{esc(ctx.simulation_id)}}}\\",
            rf"\textbf{{Reference:}} {esc(ctx.reference_uri)}\\",
            rf"\textbf{{Provenance:}} \texttt{{{esc(ctx.provenance)}}}\\",
            rf"\textbf{{Verdict:}} {esc(ctx.viability['verdict'])} "
            rf"(composite {ctx.viability['composite']:.4f}, "
            rf"confidence {ctx.viability['confidence']:.2f})",
            "",
            r"\section*{Applied variants}",
            r"\begin{longtable}{lllll}",
            r"\hline ID & Type & Contig & Start & Change \\ \hline",
        ]
        for v in ctx.variants:
            lines.append(
                f"{esc(v['id'])} & {esc(v['type'])} & {esc(v['contig'])} & {v['start']} & "
                f"{esc(v['ref'])} $\\rightarrow$ {esc(v['alt'])} \\\\"
            )
        lines += [r"\hline", r"\end{longtable}", "", r"\section*{Validation layers}"]
        for check in ctx.validation:
            lines.append(
                rf"\noindent\textbf{{{esc(check['name'])}:}} {esc(check['status'])} --- "
                rf"{esc(check['message'])}\par"
            )
        if ctx.visualizations:
            lines += ["", r"\section*{Molecular visuals}"]
            for viz in ctx.visualizations:
                image_path = (Path(self.output_dir) / viz["path"]).resolve()
                if not image_path.is_file() or image_path.suffix.lower() != ".png":
                    continue
                lines += [
                    r"\begin{figure}[htbp]",
                    r"\centering",
                    rf"\includegraphics[width=0.95\linewidth]{{{viz['path']}}}",
                    rf"\caption{{{esc(viz['title'])}}}",
                    r"\end{figure}",
                ]
        lines += ["", r"\section*{Limitations}", r"\begin{itemize}"]
        for limitation in ctx.limitations:
            lines.append(rf"\item {esc(limitation)}")
        lines += [r"\end{itemize}", r"\end{document}"]
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return output_path

    # ---------------------------------------------------------------- JSON-LD
    def generate_jsonld(self, report: Any, output_path: Path | None = None) -> Path:
        """Generate a schema.org + PROV-O JSON-LD document."""
        output_path = Path(output_path) if output_path else self.output_dir / "report.jsonld"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ctx = build_context(report)

        doc = {
            "@context": {
                "@vocab": "https://schema.org/",
                "prov": "http://www.w3.org/ns/prov#",
                "dcterms": "http://purl.org/dc/terms/",
                "skos": "http://www.w3.org/2004/02/skos/core#",
                "proteus": "https://proteus.example.com/ontology/",
            },
            "@id": f"urn:proteus:simulation:{ctx.simulation_id}",
            "@type": ["prov:Activity", "Dataset"],
            "name": "Proteus human genome simulation",
            "identifier": ctx.simulation_id,
            "description": (
                "Multi-layer in silico simulation of human DNA variants with "
                "event-sourced revert, validation layers V1-V7, and evidence chains."
            ),
            "dateCreated": ctx.generated_at,
            "prov:startedAtTime": ctx.generated_at,
            "prov:used": {
                "@type": "Dataset",
                "name": ctx.reference_uri,
                "identifier": ctx.reference_hash,
            },
            "prov:generated": [
                {
                    "@type": "Dataset",
                    "name": "mutated genome state",
                    "identifier": f"blake3:{ctx.state_mutated_hash}",
                },
                {
                    "@type": "Dataset",
                    "name": "reverted genome state",
                    "identifier": f"blake3:{ctx.state_reverted_hash}",
                },
            ],
            "proteus:variants": [
                {
                    "@type": "proteus:Variant",
                    "identifier": v["id"],
                    "proteus:variantType": v["type"],
                    "proteus:contig": v["contig"],
                    "proteus:start": v["start"],
                    "proteus:end": v["end"],
                    "proteus:refAllele": v["ref"],
                    "proteus:altAllele": v["alt"],
                }
                for v in ctx.variants
            ],
            "proteus:viability": {
                "@type": "proteus:ViabilityAssessment",
                "proteus:composite": ctx.viability["composite"],
                "proteus:verdict": ctx.viability["verdict"],
                "proteus:confidence": ctx.viability["confidence"],
                "proteus:failedGates": ctx.viability["failed_gates"],
                "proteus:survivalWindowHours": ctx.viability.get("survival_window_h"),
                "proteus:pointOfNoReturnHours": ctx.viability.get("point_of_no_return_h"),
                "proteus:irreversibleEvents": ctx.viability.get("irreversible_events", []),
                "proteus:swapGateResults": ctx.viability.get("swap_gate_results", {}),
                "proteus:hardGateFailures": ctx.viability.get("hard_gate_failures", []),
            },
            "proteus:revertHysteresis": {
                "@type": "proteus:RevertHysteresis",
                "proteus:genomeRevertExact": ctx.revert_hysteresis.get("genome_revert_exact"),
                "proteus:phenotypeMatchesBaseline": ctx.revert_hysteresis.get("phenotype_matches_baseline"),
                "proteus:hysteresisIndex": ctx.revert_hysteresis.get("hysteresis_index", 0.0),
                "proteus:residualDamage": ctx.revert_hysteresis.get("residual_damage", {}),
                "proteus:persistentEvents": ctx.revert_hysteresis.get("persistent_events", []),
            },
            "proteus:validation": [
                {
                    "@type": "proteus:ValidationLayer",
                    "identifier": c["name"],
                    "proteus:status": c["status"],
                    "proteus:message": c["message"],
                }
                for c in ctx.validation
            ],
            "proteus:evidence": ctx.evidence,
            "proteus:visualizations": [
                {
                    "@type": "ImageObject",
                    "name": viz["title"],
                    "contentUrl": viz["path"],
                    "identifier": viz["key"],
                }
                for viz in ctx.visualizations
            ],
            "proteus:limitations": ctx.limitations,
            "prov:wasGeneratedBy": {
                "@type": "prov:SoftwareAgent",
                "name": "proteus",
                "proteus:environment": ctx.environment,
                "@id": "#proteus-software",
            },
        }
        output_path.write_text(json.dumps(doc, indent=2, sort_keys=False), encoding="utf-8")
        return output_path

    # -------------------------------------------------------------- RO-Crate
    def generate_ro_crate(self, report: Any, output_dir: Path | None = None) -> Path:
        """Generate an RO-Crate evidence bundle directory."""
        output_dir = Path(output_dir) if output_dir else self.output_dir / "ro_crate"
        output_dir.mkdir(parents=True, exist_ok=True)
        ctx = build_context(report)

        import hashlib
        artifacts: list[dict[str, Any]] = []

        def add_artifact(path: Path, name: str, encoding: str = "text/html",
                         rel_override: str | None = None) -> None:
            try:
                rel = path.relative_to(output_dir).as_posix()
            except ValueError:
                rel = path.name
            if rel_override:
                rel = rel_override
            # Compute SHA256 checksum for file integrity
            checksum = ""
            if path.exists():
                with open(path, "rb") as f:
                    checksum = "sha256:" + hashlib.sha256(f.read()).hexdigest()
            artifacts.append({
                "@id": rel,
                "@type": "File",
                "name": name,
                "encodingFormat": encoding,
                "size": path.stat().st_size if path.exists() else 0,
                "sha256": checksum,
            })

        # Emit the report formats inside the crate. Image paths are prefixed so
        # the HTML/PDF/LaTeX files in this subdirectory can reach the shared
        # visualizations/ folder without duplicating the images.
        report._viz_path_prefix = "../visualizations"  # type: ignore[attr-defined]
        try:
            html_path = self.generate_html(report, output_dir / "report.html")
            add_artifact(html_path, "HTML simulation report")
            tex_path = self.generate_latex(report, output_dir / "report.tex")
            add_artifact(tex_path, "LaTeX simulation report", "application/x-tex")
            jsonld_path = self.generate_jsonld(report, output_dir / "report.jsonld")
            add_artifact(jsonld_path, "JSON-LD provenance graph", "application/ld+json")
            pdf_path = self.generate_pdf(report, output_dir / "report.pdf")
            add_artifact(pdf_path, "PDF simulation report", "application/pdf")
        finally:
            try:
                delattr(report, "_viz_path_prefix")
            except AttributeError:
                pass

        # Include the shared static/audio/animation artifacts by reference.
        viz_dir = Path(self.output_dir) / "visualizations"
        mime_by_suffix = {
            ".png": "image/png",
            ".gif": "image/gif",
            ".wav": "audio/wav",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
        }
        if viz_dir.is_dir():
            for viz_path in sorted(viz_dir.iterdir()):
                if not viz_path.is_file():
                    continue
                suffix = viz_path.suffix.lower()
                encoding = mime_by_suffix.get(suffix)
                if not encoding:
                    continue
                add_artifact(
                    viz_path,
                    viz_path.name,
                    encoding,
                    rel_override=f"../visualizations/{viz_path.name}",
                )

        # Evidence chain serialization
        evidence_path = output_dir / "evidence.json"
        evidence_path.write_text(
            json.dumps(
                {
                    "simulation_id": ctx.simulation_id,
                    "provenance": ctx.provenance,
                    "state_baseline_hash": ctx.state_baseline_hash,
                    "state_mutated_hash": ctx.state_mutated_hash,
                    "state_reverted_hash": ctx.state_reverted_hash,
                    "revert_exact": ctx.state_baseline_hash == ctx.state_reverted_hash,
                    "evidence": ctx.evidence,
                    "validation": ctx.validation,
                    "limitations": ctx.limitations,
                    "environment": ctx.environment,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        add_artifact(evidence_path, "Evidence chains and validation details", "application/json")

        # Run manifest (inputs / outputs / hashes) - using new field names
        run_manifest = {
            "simulation_id": ctx.simulation_id,
            "generated_at": ctx.generated_at,
            "reference": {"uri": ctx.reference_uri, "hash": ctx.reference_hash},
            "states": {
                "state_baseline_hash": ctx.state_baseline_hash,
                "state_mutated_hash": ctx.state_mutated_hash,
                "state_reverted_hash": ctx.state_reverted_hash,
                "revert_exact": ctx.state_baseline_hash == ctx.state_reverted_hash,
            },
            "variants": ctx.variants,
            "viability": ctx.viability,
            "revert_hysteresis": ctx.revert_hysteresis,
            "validation": ctx.validation_compact,
            "artifacts": [a["@id"] for a in artifacts],
            "environment": ctx.environment,
        }
        manifest_path = output_dir / "run_manifest.json"
        manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
        add_artifact(manifest_path, "Run manifest", "application/json")

        crate = {
            "@context": "https://w3id.org/ro/crate/1.1/context",
            "@graph": [
                {
                    "@id": "ro-crate-metadata.json",
                    "@type": "CreativeWork",
                    "conformsTo": {"@id": "https://w3id.org/ro/crate/1.1"},
                    "about": {"@id": f"urn:proteus:simulation:{ctx.simulation_id}"},
                },
                {
                    "@id": f"urn:proteus:simulation:{ctx.simulation_id}",
                    "@type": ["Dataset", "prov:Activity"],
                    "name": "Proteus human genome simulation",
                    "description": "Multi-layer in silico variant simulation with V1-V7 validation.",
                    "datePublished": ctx.generated_at,
                    "hasPart": [{"@id": a["@id"]} for a in artifacts],
                    "proteus:viability": ctx.viability["verdict"],
                    "proteus:provenance": ctx.provenance,
                    "proteus:revertExact": ctx.state_baseline_hash == ctx.state_reverted_hash,
                    "proteus:revertHysteresis": {
                        "genomeRevertExact": ctx.revert_hysteresis.get("genome_revert_exact"),
                        "phenotypeMatchesBaseline": ctx.revert_hysteresis.get("phenotype_matches_baseline"),
                        "hysteresisIndex": ctx.revert_hysteresis.get("hysteresis_index", 0.0),
                        "residualDamage": ctx.revert_hysteresis.get("residual_damage", {}),
                    },
                    "proteus:limitations": ctx.limitations,
                    "prov:wasGeneratedBy": {"@id": "#proteus-software"},
                    "dcterms:creator": {"@id": "#proteus-authors"},
                    "dcterms:license": {"@id": "#mit-license"},
                },
                {
                    "@id": "#proteus-software",
                    "@type": "SoftwareApplication",
                    "name": "proteus",
                    "version": "0.1.0",
                    "programmingLanguage": ["Rust", "C++", "Python"],
                },
                {
                    "@id": "#proteus-authors",
                    "@type": "Organization",
                    "name": "Proteus Consortium",
                    "url": "https://github.com/proteus",
                },
                {
                    "@id": "#mit-license",
                    "@type": "CreativeWork",
                    "name": "MIT License",
                    "url": "https://opensource.org/licenses/MIT",
                    "identifier": "MIT",
                },
                *artifacts,
            ],
        }
        (output_dir / "ro-crate-metadata.json").write_text(
            json.dumps(crate, indent=2), encoding="utf-8"
        )
        return output_dir

    # ------------------------------------------------------------------- all
    def generate_all(self, report: Any, base_name: str = "report") -> dict[str, Path]:
        """Generate every report format into ``output_dir``."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "html": self.generate_html(report, self.output_dir / f"{base_name}.html"),
            "pdf": self.generate_pdf(report, self.output_dir / f"{base_name}.pdf"),
            "latex": self.generate_latex(report, self.output_dir / f"{base_name}.tex"),
            "jsonld": self.generate_jsonld(report, self.output_dir / f"{base_name}.jsonld"),
            "ro_crate": self.generate_ro_crate(report, self.output_dir / f"{base_name}_rocrate"),
        }


__all__ = ["ReportGenerator", "ReportContext", "build_context"]
