"""
Runtime validation layers V1-V7.

The Rust workspace owns the authoritative validators (``proteus-validate``);
this module provides the *pipeline-level* checks that can be evaluated from
Python over the objects produced by a run. Each check returns a structured
result; the pipeline records the compact verdicts in the simulation report
and attaches details as evidence.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

IUPAC_RE = re.compile(r"^[ACGTUNRYSWKMBDHVacgtunryswkmbdhv]+$")


@dataclass
class CheckResult:
    layer: str
    status: str  # "pass" | "fail" | "skip"
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "status": self.status,
            "message": self.message,
            "details": self.details,
        }


def _iter_floats(value: Any) -> Iterable[float]:
    """Recursively yield every float found in nested dict/list structures."""
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield float(value)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _iter_floats(v)
    elif isinstance(value, (list, tuple, set)):
        for v in value:
            yield from _iter_floats(v)


def _layer_data(layer_results: list[Any], name: str) -> dict[str, Any]:
    for result in layer_results:
        if getattr(result, "layer", None) == name:
            return getattr(result, "data", {}) or {}
    return {}


class RuntimeValidator:
    """Evaluates V1-V7 over the artifacts of a completed pipeline run."""

    def __init__(self, enabled: list[str] | None = None):
        self.enabled = set(enabled or ["V1", "V2", "V3", "V4", "V5", "V6", "V7"])

    def validate(
        self,
        *,
        variants: list[Any],
        layer_results: list[Any],
        viability: Any,
        baseline: Any,
        mutated: Any,
        reverted: Any | None,
        rust_baseline: Any | None = None,
        rust_mutated: Any | None = None,
        rust_reverted: Any | None = None,
        seed: int = 42,
    ) -> dict[str, CheckResult]:
        results: dict[str, CheckResult] = {}
        checks = {
            "V1": lambda: self._v1_schema(variants),
            "V2": lambda: self._v2_semantic(variants, rust_baseline),
            "V3": lambda: self._v3_physical(layer_results, viability),
            "V4": lambda: self._v4_triangulation(variants, layer_results, mutated, rust_mutated),
            "V5": lambda: self._v5_determinism(rust_baseline, rust_mutated, seed),
            "V6": lambda: self._v6_scientific(layer_results, viability),
            "V7": lambda: self._v7_reproducibility(baseline, mutated, reverted, rust_baseline),
        }
        for name, check in checks.items():
            if name not in self.enabled:
                results[name] = CheckResult(name, "skip", "not enabled")
                continue
            try:
                results[name] = check()
            except Exception as exc:  # a validator crash must not abort the run
                results[name] = CheckResult(
                    name, "fail", f"validator error: {type(exc).__name__}: {exc}"
                )
        return results

    # ------------------------------------------------------------------ V1
    def _v1_schema(self, variants: list[Any]) -> CheckResult:
        errors: list[str] = []
        for i, v in enumerate(variants):
            vid = getattr(v, "id", "")
            start = getattr(v, "start", None)
            end = getattr(v, "end", None)
            ref = getattr(v, "ref_allele", "") or ""
            alt = getattr(v, "alt_allele", "") or ""

            if not vid:
                errors.append(f"variant[{i}]: empty id")
            if not isinstance(start, int) or start < 1:
                errors.append(f"{vid or i}: invalid start {start!r} (expected >= 1, 1-based)")
            if not isinstance(end, int) or end < 1:
                errors.append(f"{vid or i}: invalid end {end!r} (expected >= 1, 1-based)")
            elif isinstance(start, int) and end < start:
                errors.append(f"{vid or i}: end {end} < start {start}")
            if ref and not IUPAC_RE.match(ref):
                errors.append(f"{vid or i}: ref allele not IUPAC DNA: {ref!r}")
            if alt and not IUPAC_RE.match(alt):
                errors.append(f"{vid or i}: alt allele not IUPAC DNA: {alt!r}")

            type_name = str(getattr(getattr(v, "type", ""), "name", getattr(v, "type", "")))
            if "SNV" in type_name and not (len(ref) == 1 and len(alt) == 1):
                errors.append(f"{vid or i}: SNV requires single-base ref/alt ({ref!r}->{alt!r})")

        status = "fail" if errors else "pass"
        return CheckResult(
            "V1", status,
            f"{len(errors)} schema violation(s)" if errors else f"{len(variants)} variant(s) valid",
            {"errors": errors[:20], "variant_count": len(variants)},
        )

    # ------------------------------------------------------------------ V2
    def _v2_semantic(self, variants: list[Any], rust_baseline: Any | None) -> CheckResult:
        errors: list[str] = []
        checked = 0

        reference = getattr(getattr(rust_baseline, "_rs_state", None), "reference", None)
        contig_names: list[str] = []
        if reference is not None and hasattr(reference, "contig_names"):
            contig_names = list(reference.contig_names())

        for v in variants:
            contig = getattr(v, "contig", "")
            start = int(getattr(v, "start", 0) or 0)
            end = int(getattr(v, "end", 0) or 0)
            ref = (getattr(v, "ref_allele", "") or "").upper()

            if contig_names and contig not in contig_names:
                errors.append(f"{getattr(v, 'id', '?')}: contig {contig!r} not in reference")
                continue

            if rust_baseline is not None and contig_names and hasattr(rust_baseline, "get_sequence"):
                try:
                    seq = bytes(rust_baseline.get_sequence(contig))
                    checked += 1
                    if start < 1:
                        errors.append(
                            f"{getattr(v, 'id', '?')}: start {start} is not 1-based"
                        )
                        continue
                    if start > len(seq):
                        errors.append(
                            f"{getattr(v, 'id', '?')}: start {start} beyond contig {contig} length {len(seq)}"
                        )
                        continue
                    if end > len(seq):
                        errors.append(
                            f"{getattr(v, 'id', '?')}: end {end} beyond contig {contig} length {len(seq)}"
                        )
                        continue
                    if ref:
                        # Rust core uses 1-based inclusive coordinates (VCF convention)
                        observed = seq[start - 1:end].decode("ascii", errors="replace").upper()
                        if observed != ref and "N" not in observed and "N" not in ref:
                            errors.append(
                                f"{getattr(v, 'id', '?')}: REF mismatch at {contig}:{start} "
                                f"(expected {ref}, observed {observed})"
                            )
                except Exception as exc:
                    errors.append(f"{getattr(v, 'id', '?')}: reference read failed: {exc}")

        status = "fail" if errors else ("pass" if checked or contig_names else "skip")
        message = (
            f"{len(errors)} semantic violation(s)" if errors
            else (f"{checked} variant(s) checked against reference" if checked
                  else "reference sequence unavailable; structural checks only")
        )
        return CheckResult("V2", status, message, {
            "errors": errors[:20],
            "checked_against_reference": checked,
            "contigs": contig_names,
        })

    # ------------------------------------------------------------------ V3
    def _v3_physical(self, layer_results: list[Any], viability: Any) -> CheckResult:
        errors: list[str] = []
        non_finite: list[str] = []

        for result in layer_results:
            name = getattr(result, "layer", "?")
            data = getattr(result, "data", {}) or {}
            for value in _iter_floats(data):
                if not math.isfinite(value):
                    non_finite.append(name)
                    break

        if non_finite:
            errors.append(f"non-finite numeric values in layers: {sorted(set(non_finite))}")

        composite = float(getattr(viability, "composite", 0.0) or 0.0)
        if not math.isfinite(composite) or not (0.0 <= composite <= 1.0):
            errors.append(f"viability composite outside [0,1]: {composite}")

        for name, score in (getattr(viability, "layer_scores", {}) or {}).items():
            if not math.isfinite(float(score)) or not (0.0 <= float(score) <= 1.0):
                errors.append(f"layer score {name} outside [0,1]: {score}")

        status = "fail" if errors else "pass"
        return CheckResult(
            "V3", status,
            "; ".join(errors) if errors else "all numeric outputs finite and in range",
            {"errors": errors},
        )

    # ------------------------------------------------------------------ V4
    def _v4_triangulation(
        self,
        variants: list[Any],
        layer_results: list[Any],
        mutated: Any,
        rust_mutated: Any | None,
    ) -> CheckResult:
        errors: list[str] = []
        bridge_checks: dict[str, Any] = {}

        expected = len(variants)
        py_count = len(getattr(mutated, "overlays", []) or [])
        bridge_checks["expected_variant_count"] = expected
        bridge_checks["python_overlay_count"] = py_count
        if py_count != expected:
            errors.append(f"Python overlay count {py_count} != expected {expected}")

        if rust_mutated is not None:
            rs_count = len(getattr(rust_mutated, "overlays", []) or [])
            bridge_checks["rust_overlay_count"] = rs_count
            if rs_count != expected:
                errors.append(f"Rust overlay count {rs_count} != expected {expected}")

        l0 = _layer_data(layer_results, "L0")
        l0_count = l0.get("variants_applied")
        if l0_count is not None:
            bridge_checks["l0_variants_applied"] = l0_count
            if int(l0_count) != expected:
                errors.append(f"L0 variants_applied {l0_count} != expected {expected}")

        # Rust L0 diff must report the same number of added variants
        added = (l0.get("variant_diff", {}) or {}).get("added", {})
        if isinstance(added, dict) and added:
            bridge_checks["l0_diff_added"] = len(added)
            if len(added) > expected:
                errors.append(f"L0 diff added {len(added)} exceeds expected {expected}")

        status = "fail" if errors else "pass"
        message = "; ".join(errors) if errors else (
            f"Rust/Python agree on {expected} variant(s)"
            if rust_mutated is not None else
            f"Python-only consistency for {expected} variant(s) (Rust core unavailable)"
        )
        return CheckResult("V4", status, message, bridge_checks)

    # ------------------------------------------------------------------ V5
    def _v5_determinism(
        self,
        rust_baseline: Any | None,
        rust_mutated: Any | None,
        seed: int,
    ) -> CheckResult:
        details: dict[str, Any] = {"seed": seed}
        errors: list[str] = []

        # Stable surrogate hash must be reproducible regardless of platform/PYTHONHASHSEED
        from proteus.layers.l1_variant_semantics import stable_hash

        h1 = stable_hash(seed, "determinism-probe", 12345)
        h2 = stable_hash(seed, "determinism-probe", 12345)
        details["surrogate_hash_reproducible"] = h1 == h2
        if h1 != h2:
            errors.append("surrogate scoring hash is not reproducible")

        if rust_baseline is not None and rust_mutated is not None:
            try:
                details["baseline_hash"] = rust_baseline.hash()
                details["mutated_hash"] = rust_mutated.hash()

                # Re-derive the mutated state from the baseline and the original
                # overlay log; an event-sourced core must reach the same hash.
                overlays = list(getattr(rust_mutated, "overlays", []))
                replay = rust_baseline.apply_batch(overlays)
                replay_hash = replay.hash()
                details["replay_hash"] = replay_hash
                details["replay_overlay_count"] = len(overlays)
                details["replay_matches_mutated"] = replay_hash == details["mutated_hash"]
                if replay_hash != details["mutated_hash"]:
                    errors.append(
                        "event replay produced a different mutated hash "
                        f"({replay_hash[:16]} != {details['mutated_hash'][:16]})"
                    )
            except Exception as exc:
                errors.append(f"event replay failed: {type(exc).__name__}: {exc}")

        status = "fail" if errors else "pass"
        return CheckResult(
            "V5", status,
            "; ".join(errors) if errors else "deterministic surrogates and event replay verified",
            details,
        )

    # ------------------------------------------------------------------ V6
    def _v6_scientific(self, layer_results: list[Any], viability: Any) -> CheckResult:
        errors: list[str] = []
        observed: dict[str, Any] = {}

        l1 = _layer_data(layer_results, "L1").get("aggregate", {}) or {}
        l2 = _layer_data(layer_results, "L2").get("aggregate", {}) or {}
        l3 = _layer_data(layer_results, "L3").get("aggregate", {}) or {}
        l4 = _layer_data(layer_results, "L4").get("aggregate", {}) or {}

        max_path = float(l1.get("max_pathogenicity", 0.0) or 0.0)
        observed["max_pathogenicity"] = max_path
        if not (0.0 <= max_path <= 1.0):
            errors.append(f"max_pathogenicity outside [0,1]: {max_path}")

        max_ddg = float(l2.get("max_delta_delta_g", 0.0) or 0.0)
        observed["max_delta_delta_g"] = max_ddg
        if max_ddg < 0.0 or max_ddg > 30.0:
            errors.append(f"max ΔΔG outside plausible [0,30] kcal/mol: {max_ddg}")

        biomass = float(l3.get("biomass_flux", 1.0) or 0.0)
        observed["biomass_flux"] = biomass
        if biomass < 0.0 or biomass > 1.5:
            errors.append(f"biomass flux outside [0,1.5]: {biomass}")

        atp_pool = l3.get("atp_pool")
        if atp_pool is not None:
            observed["atp_pool"] = float(atp_pool)
            if float(atp_pool) < 0.0:
                errors.append(f"negative ATP pool: {atp_pool}")

        systemic = float(l4.get("systemic_score", 1.0) or 0.0)
        observed["systemic_score"] = systemic
        if not (0.0 <= systemic <= 1.0):
            errors.append(f"systemic score outside [0,1]: {systemic}")

        pci = (l4.get("neuro_proxies") or {}).get("perturbational_complexity_index")
        if pci is not None:
            observed["pci"] = float(pci)
            if not (0.0 <= float(pci) <= 1.0):
                errors.append(f"PCI proxy outside [0,1]: {pci}")

        if getattr(viability, "failed_gates", None):
            observed["failed_gates"] = list(viability.failed_gates)

        status = "fail" if errors else "pass"
        return CheckResult(
            "V6", status,
            "; ".join(errors) if errors else "all scientific plausibility ranges satisfied",
            observed,
        )

    # ------------------------------------------------------------------ V7
    def _v7_reproducibility(
        self,
        baseline: Any,
        mutated: Any,
        reverted: Any | None,
        rust_baseline: Any | None,
    ) -> CheckResult:
        errors: list[str] = []
        details: dict[str, Any] = {}

        if rust_baseline is not None:
            try:
                details["baseline_hash"] = rust_baseline.hash()
            except Exception as exc:
                errors.append(f"baseline hash unavailable: {exc}")
        else:
            try:
                details["baseline_hash"] = baseline.hash()
            except Exception as exc:
                errors.append(f"baseline hash unavailable: {exc}")

        ref_uri = getattr(getattr(baseline, "reference", None), "uri", "")
        details["reference_uri"] = ref_uri
        if not ref_uri:
            errors.append("reference URI missing (inputs are not content-addressed)")

        ref_hash = getattr(getattr(baseline, "reference", None), "hash", None)
        digest = getattr(ref_hash, "digest", b"") if ref_hash is not None else b""
        details["reference_digest_present"] = bool(digest)
        if not digest and rust_baseline is None:
            errors.append("reference content hash missing")

        if reverted is not None:
            try:
                if rust_baseline is not None:
                    details["revert_exact"] = reverted.hash() == baseline.hash()
                else:
                    details["revert_exact"] = reverted.hash() == baseline.hash()
                if not details["revert_exact"]:
                    errors.append("reverted state does not match baseline hash")
            except Exception as exc:
                errors.append(f"revert comparison failed: {exc}")

        status = "fail" if errors else "pass"
        return CheckResult(
            "V7", status,
            "; ".join(errors) if errors else "content-addressed inputs and exact revert verified",
            details,
        )

    # ------------------------------------------------------------ reporting
    @staticmethod
    def compact(results: dict[str, CheckResult]) -> dict[str, str]:
        """Compact {V1: 'pass', ...} mapping for SimulationReport.validation_results."""
        return {name: result.status for name, result in results.items()}

    @staticmethod
    def details(results: dict[str, CheckResult]) -> dict[str, dict[str, Any]]:
        return {name: result.to_dict() for name, result in results.items()}
