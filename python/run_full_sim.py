import asyncio
import json
import os
from proteus.pipeline import run_simulation
from proteus.schema import PipelineConfig, VariantEvent

# Create output directory in proteus root
os.makedirs('../output', exist_ok=True)

# Configure with multiple variant types to exercise all layers
# Test genome: ATCGATCGATCG (1-based positions: 1=A,2=T,3=C,4=G,5=A,6=T,7=C,8=G,9=A,10=T,11=C,12=G)
config = PipelineConfig(
    reference_genome_uri='test_genome.fasta',
    variants=[
        # SNV at position 10 (T->A)
        VariantEvent(id='v1_snv', type=VariantEvent.Type.SNV, contig='chr1', start=10, end=10, ref_allele='T', alt_allele='A', annotations={'gene_id': 'GAPDH'}, timestamp_ns=0, source='demo'),
        # SNV at position 2 (T->C)
        VariantEvent(id='v2_snv', type=VariantEvent.Type.SNV, contig='chr1', start=2, end=2, ref_allele='T', alt_allele='C', annotations={'gene_id': 'PGK1'}, timestamp_ns=0, source='demo'),
        # Insertion at position 8 (after G at pos 8, insert A)
        VariantEvent(id='v3_ins', type=VariantEvent.Type.INSERTION, contig='chr1', start=8, end=8, ref_allele='', alt_allele='A', annotations={'gene_id': 'PKM'}, timestamp_ns=0, source='demo'),
    ],
    output_dir='../output',
    produce_report=True,
)

result = asyncio.run(run_simulation(config))
r = result.report

print('='*60)
print('SIMULATION COMPLETE')
print('='*60)
print(f'Run ID: {result.run_id}')
print(f'Success: {result.success}')
print(f'Error: {result.error}')
print()
print('VIABILITY:')
print(f'  Verdict: {r.viability.verdict.name}')
print(f'  Composite: {r.viability.composite:.4f}')
print(f'  Layer Scores: {r.viability.layer_scores}')
print(f'  Failed Gates: {r.viability.failed_gates}')
print()
print('LAYER RESULTS:')
for k, v in r.layer_results.items():
    print(f'  {k}: {len(v)} keys')
    for key, val in v.items():
        if isinstance(val, (int, float, str, bool)):
            print(f'    {key}: {val}')
        elif isinstance(val, list):
            print(f'    {key}: list[{len(val)}]')
        elif isinstance(val, dict):
            print(f'    {key}: dict[{len(val)}]')
        else:
            print(f'    {key}: {type(val).__name__}')
print()
print(f'DIFFS: {len(r.diffs)} layers')
for d in r.diffs:
    print(f'  {d.layer}: {len(d.changed)} changed, {len(d.added)} added, {len(d.removed)} removed')
print()
print(f'EVIDENCE CHAINS: {len(r.evidence)}')
for ec in r.evidence:
    inp = ec.input_hash.digest.hex()[:16] if ec.input_hash and ec.input_hash.digest else 'none'
    out = ec.output_hash.digest.hex()[:16] if ec.output_hash and ec.output_hash.digest else 'none'
    conf = ec.confidence.name if ec.confidence else 'UNSPECIFIED'
    print(f'  {ec.tool}: confidence={conf}, upstream={ec.upstream_ref}, input={inp}..., output={out}...')
print()
print(f'VALIDATIONS: {len(r.validation_results)}')
for k,v in r.validation_results.items():
    print(f'  {k}: {v}')
print()
print('HEADLINE OUTPUTS:')
print(f'  Survival window: {r.survival_window_h} h')
print(f'  Point of no return: {r.point_of_no_return_event}')
print(f'  Irreversible events: {len(r.irreversible_events)}')
hyst = getattr(r, 'revert_hysteresis', {}) or {}
print(f'  Genome revert exact: {hyst.get("genome_revert_exact")}')
print(f'  Phenotype matches baseline: {hyst.get("phenotype_matches_baseline")}')
print(f'  Hysteresis index: {hyst.get("hysteresis_index")}')
print(f'  Residual damage: {hyst.get("residual_damage")}')
print()
lim = getattr(r, 'limitations', [])
print(f'LIMITATIONS: {len(lim)}')
print()
print('ARTIFACTS:')
for k,v in result.artifacts.items():
    if hasattr(v, 'digest'):
        print(f'  {k}: {v.digest.hex() if v.digest else "empty"}')
    else:
        print(f'  {k}: {v}')
print()
print('OUTPUT DIRECTORY:')
for root, dirs, files in os.walk('../output'):
    level = root.replace('./output', '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for f in files:
        print(f'{subindent}{f}')