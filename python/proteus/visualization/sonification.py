"""
Audio sonification of simulation results.
Converts genomic/phenotypic data into audible waveforms.

Uses scipy.signal to synthesize audio; no external audio libraries required.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path
from typing import Optional

import numpy as np

from proteus.schema import SimulationReport, VariantEvent

SAMPLE_RATE = 44100  # CD quality
DEFAULT_DURATION = 2.0  # seconds


def _freq_from_position(pos: int, scale: str = "chromatic", octave: int = 4) -> float:
    """Map genomic position to a musical frequency (440Hz = A4)."""
    # 12-tone equal temperament
    n = pos % 12
    oct = (pos // 12) % 7 + octave
    # A4 = 440Hz = 69 in MIDI
    midi_note = 12 * (oct + 1) + n
    freq = 440.0 * 2 ** ((midi_note - 69) / 12)
    return np.clip(freq, 80, 8000)  # audible range


def _note_from_value(value: float, scale: list[float] | None = None) -> float:
    """Map a normalized value [0,1] to a frequency from a scale."""
    if scale is None:
        # Major pentatonic scale (C4 to C6)
        scale = [261.63, 293.66, 329.63, 392.00, 440.00,   # C4-D4-E4-G4-A4
                 523.25, 587.33, 659.25, 783.99, 880.00,  # C5-D5-E5-G5-A5
                 1046.50]                                   # C6
    idx = int(value * (len(scale) - 1))
    return scale[idx]


def sonify_genome(
    sequence: str,
    duration: float = 3.0,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Convert a DNA sequence to audio using nucleotide-to-frequency mapping.

    A=261.6Hz (C4), T=293.7Hz (D4), G=329.6Hz (E4), C=392.0Hz (G4)
    """
    nuc_freqs = {"A": 261.63, "T": 293.66, "G": 329.63, "C": 392.00}
    t_per_note = duration / max(len(sequence), 1)
    t = np.arange(0, int(duration * sample_rate), dtype=np.float64)

    audio = np.zeros_like(t, dtype=np.float64)
    for i, nuc in enumerate(sequence.upper()):
        freq = nuc_freqs.get(nuc, 440.0)
        start = int(i * t_per_note * sample_rate)
        end = min(start + int(t_per_note * sample_rate), len(t))
        if end > start:
            seg = np.linspace(start * 1.0 / sample_rate, (end - 1) * 1.0 / sample_rate, end - start)
            audio[start:end] = 0.3 * np.sin(2 * np.pi * freq * seg)

    # Envelope
    env = np.ones_like(audio)
    fade_samples = min(int(0.1 * sample_rate), len(audio) // 4)
    if fade_samples > 0:
        env[:fade_samples] = np.linspace(0, 1, fade_samples)
        env[-fade_samples:] = np.linspace(1, 0, fade_samples)
    audio *= env

    return audio


def sonify_mutation(
    report: SimulationReport,
    variant: VariantEvent,
    duration: float = 1.5,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Create a before/after audio comparison for a single variant.

    First half: reference nucleotide tone
    Second half: alternate nucleotide tone (with slight dissonance for pathogenic variants)
    """
    t = np.arange(0, int(duration * sample_rate), dtype=np.float64)
    audio = np.zeros_like(t, dtype=np.float64)
    mid = len(t) // 2

    ref_freq = {"A": 261.63, "T": 293.66, "G": 329.63, "C": 392.00}.get(variant.ref_allele, 440.0)
    alt_freq = {"A": 261.63, "T": 293.66, "G": 329.63, "C": 392.00}.get(variant.alt_allele, 440.0)

    # Reference (first half)
    seg1 = np.linspace(0, duration / 2, mid)
    audio[:mid] = 0.3 * np.sin(2 * np.pi * ref_freq * seg1)

    # Alternate (second half) - add slight vibrato if pathogenic
    seg2 = np.linspace(duration / 2, duration, len(t) - mid)
    audio[mid:] = 0.3 * np.sin(2 * np.pi * alt_freq * seg2)

    # If pathogenic score > 0.5, add a minor 2nd dissonance
    if hasattr(report, "layer_results") and report.layer_results:
        l1 = report.layer_results.get("L1_variant_semantics", {})
        if isinstance(l1, dict) and l1.get("pathogenicity_score", 0) > 0.5:
            dissonant_freq = alt_freq * (16 / 15)  # minor second above
            audio[mid:] += 0.05 * np.sin(2 * np.pi * dissonant_freq * seg2)

    # Envelope
    fade_samples = min(int(0.05 * sample_rate), len(audio) // 8)
    if fade_samples > 0:
        env = np.ones_like(audio)
        env[:fade_samples] = np.linspace(0, 1, fade_samples)
        env[-fade_samples:] = np.linspace(1, 0, fade_samples)
        audio *= env

    return audio


def sonify_pathway(
    report: SimulationReport,
    layer: str = "L3",
    duration: float = 4.0,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Convert pathway activity changes into an audio melody.

    Each node in the pathway is a note; flux values map to pitch.
    Higher flux = higher pitch. Disrupted edges create dissonance.
    """
    pathway_data = {}
    if hasattr(report, "layer_results") and report.layer_results:
        pathway_data = report.layer_results.get(layer, {})

    t = np.arange(0, int(duration * sample_rate), dtype=np.float64)
    audio = np.zeros_like(t, dtype=np.float64)

    if pathway_data and "fluxes" in pathway_data:
        fluxes = pathway_data["fluxes"]
        if isinstance(fluxes, list) and len(fluxes) > 0:
            # Normalize fluxes to [0, 1]
            flux_arr = np.array(fluxes)
            norm = (flux_arr - flux_arr.min()) / (flux_arr.max() - flux_arr.min() + 1e-10)

            # Pentatonic scale
            scale = [261.63, 293.66, 329.63, 392.00, 440.00, 523.25, 587.33, 659.25]
            t_per_note = duration / len(norm)

            for i, n in enumerate(norm):
                freq = scale[int(n * (len(scale) - 1))]
                start = int(i * t_per_note * sample_rate)
                end = min(start + int(t_per_note * sample_rate), len(t))
                if end > start:
                    seg = np.linspace(start / sample_rate, (end - 1) / sample_rate, end - start)
                    audio[start:end] = 0.2 * np.sin(2 * np.pi * freq * seg)

    # Fallback: simple melody based on viability score
    if not np.any(audio):
        score = report.viability.composite
        base_freq = 261.63 * 2 ** (score * 2)  # Map score to C4-C6 range
        melody = [base_freq, base_freq * 1.25, base_freq * 1.5, base_freq * 2,
                  base_freq * 1.5, base_freq * 1.25, base_freq]
        t_per_note = duration / len(melody)
        for i, freq in enumerate(melody):
            start = int(i * t_per_note * sample_rate)
            end = min(start + int(t_per_note * sample_rate), len(t))
            if end > start:
                seg = np.linspace(start / sample_rate, (end - 1) / sample_rate, end - start)
                audio[start:end] = 0.2 * np.sin(2 * np.pi * freq * seg)

    # Envelope
    fade_samples = min(int(0.1 * sample_rate), len(audio) // 4)
    if fade_samples > 0:
        env = np.ones_like(audio)
        env[:fade_samples] = np.linspace(0, 1, fade_samples)
        env[-fade_samples:] = np.linspace(1, 0, fade_samples)
        audio *= env

    return audio


def save_audio(
    audio: np.ndarray,
    output_path: Path,
    sample_rate: int = SAMPLE_RATE,
    channels: int = 1,
) -> Path:
    """Save a numpy audio array as a WAV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalize to [-1, 1]
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.95  # headroom

    # Convert to 16-bit PCM
    audio_int16 = np.int16(audio * 32767)

    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())

    return output_path


def create_comparison_audio(
    report: SimulationReport,
    output_dir: Path,
    prefix: str = "sonification",
) -> dict[str, Path]:
    """Generate all sonification audio files for a report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # 1. Genome sonification (if sequence available)
    seq = ""
    if hasattr(report, "layer_results") and report.layer_results:
        l1 = report.layer_results.get("L1_variant_semantics", {})
        if isinstance(l1, dict) and "sequence_context" in l1:
            seq = l1["sequence_context"]
    if seq:
        paths["genome"] = output_dir / f"{prefix}_genome.wav"
        audio = sonify_genome(seq[:200])  # limit to first 200 bp
        save_audio(audio, paths["genome"])
    else:
        # Default: use test genome
        paths["genome"] = output_dir / f"{prefix}_genome.wav"
        audio = sonify_genome("ATCGATCGATCG" * 10)
        save_audio(audio, paths["genome"])

    # 2. Variant sonifications
    for i, v in enumerate(report.applied_variants):
        paths[f"variant_{i}"] = output_dir / f"{prefix}_variant_{v.id}.wav"
        audio = sonify_mutation(report, v)
        save_audio(audio, paths[f"variant_{i}"])

    # 3. Pathway sonification
    paths["pathway"] = output_dir / f"{prefix}_pathway.wav"
    audio = sonify_pathway(report)
    save_audio(audio, paths["pathway"])

    # 4. Combined piece
    paths["combined"] = output_dir / f"{prefix}_combined.wav"
    all_audio = []
    all_audio.append(sonify_genome("ATCGATCGATCG" * 5, duration=2.0))
    if report.applied_variants:
        all_audio.append(sonify_mutation(report, report.applied_variants[0], duration=2.0))
    all_audio.append(sonify_pathway(report, duration=2.0))

    combined = np.concatenate(all_audio)
    save_audio(combined, paths["combined"])

    return paths