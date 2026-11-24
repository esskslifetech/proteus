# Proteus

**Human genome mutation simulation, viability analysis, and auditable evidence bundles.**

Proteus is a polyglot (Rust / C++ / Python) system that simulates DNA mutations
across biological layers. It loads a reference genome, injects variants,
simulates baseline and mutated states, evaluates time-dynamic viability, reverts
the genome exactly, and produces a per-layer diff plus a hash-linked evidence
bundle.

## Core loop

```text
load reference → inject variant(s) → simulate L1–L4 → time-dynamic viability
→ revert → per-layer diff → report + evidence bundle
```

## What is modeled

| Layer | Scope | Engine |
|-------|-------|--------|
| **L0** | DNA sequence edits, exact revert | Rust event-sourced core |
| **L1** | Coding / splicing / regulatory consequence | Deterministic surrogate |
| **L2** | Protein stability (ΔΔG), domains | C++ kernels / Python fallback |
| **L3** | Metabolism, GRN, ODE dynamics | C++ CVODE / RK4 fallback |
| **L3b** | mRNA/protein turnover, irreversible damage, hysteresis | Python / C++ ODE |
| **L4** | Tissue and organ physiology, neuro proxies | Compartment model |
| **L5** | Weighted viability gate | Evidence fusion |

## Key properties

- **Exact genome revert** — event-sourced state restoration with content hashing.
- **Hysteresis** — irreversible phenotypic damage persists after the genome is restored.
- **Validation V1–V7** — schema, semantics, physical bounds, triangulation, determinism, scientific plausibility, and reproducibility.
- **Auditable outputs** — HTML, PDF, LaTeX, JSON-LD, and RO-Crate, with DNA/genome visualizations.
- **Trained-model hooks** — structured drop-in points for AlphaMissense/ESM, FoldX/ESMFold, Human-GEM, and PBPK models.

## Limitations

Variant predictors are deterministic surrogates structured for replacement by
trained models. Neurophysiological proxies are measurable correlates, **not**
consciousness or subjective experience.

## Links

- [Source code](https://github.com/esskslifetech/proteus)
- [Polyglot architecture ADR](adr/001-polyglot-architecture.md)
- [README](https://github.com/esskslifetech/proteus#readme)

## License

Apache-2.0
