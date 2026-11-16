#!/usr/bin/env python3
"""
SQUINT Difficult Whole-Genome MSA Benchmark:
  Case 1A (C1A_HIV1_GroupM_PanSubtype_34): 34 full-length HIV-1 Group M references (17 lineages, 8.6-9.8 kb).
  Case 1B (C1B_Lentivirus_HIV1_HIV2_SIV_18): 18 primate lentiviruses (HIV-1/SIVcpz + HIV-2/SIVsmm/mac, 8.9-10.5 kb).
  Case 2A (C2A_Vertebrate_mtDNA_RawRefSeq_24): 24 complete RefSeq vertebrate mitogenomes (16.2-17.8 kb, raw orientation).
  Case 2B (C2B_Vertebrate_mtDNA_Canonical_trnF_24): Same 24 vertebrate mitogenomes canonically rotated to tRNA-Phe = bp 1.
  Case 3  (C3_Viral_vs_Mito_Decoy_12): 6 lentiviruses + 6 vertebrate mitogenomes in a single unaligned cohort.

Evaluates runtime, peak RSS, .fa/.hdb lossless convert + squint subset --region roundtrips,
reference core accuracy (C1A), blind DNA-mode reading-frame codon-phase concordance,
gene/locus Cross_Feature_FDR, lineage-specific vpx/vpu false homology, avian ND6 rearrangement FDR,
viral-vs-mitochondrial cross-kingdom false merges, and DNA HorizonBubbleReport segmentation.

Zero empirical substitution matrices used anywhere.
Outputs:
  - benchmarks/difficult_genomes_results.csv
  - benchmarks/difficult_genomes_summary.json
  - benchmarks/difficult_genomes_plot.png (+ copy in artifacts directory)
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq

BENCH_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = BENCH_DIR.parent
sys.path.insert(0, str(BENCH_DIR))

from run_easy_msa_benchmark import extract_ref_structures, parse_fasta, score_alignment  # noqa: E402
from run_scaling_benchmark import (  # noqa: E402
    parse_squint_components_fasta,
    restore_aliased_fasta,
    write_aliased_fasta,
)

SQUINT = os.environ.get("SQUINT_BIN", str(ROOT_DIR / "target" / "release" / "squint"))
MAFFT = shutil.which("mafft") or os.environ.get("MAFFT_BIN", "mafft")
MUSCLE = shutil.which("muscle") or os.environ.get("MUSCLE_BIN", "muscle")
CLUSTALO = shutil.which("clustalo") or os.environ.get("CLUSTALO_BIN", "clustalo")

HIV_DIR = pathlib.Path(
    os.environ.get("HIV_DATA_DIR", str(BENCH_DIR / "external_data" / "hiv_data"))
)
CACHE_DIR = BENCH_DIR / "external_data" / "difficult_genomes"
WORK_DIR = CACHE_DIR / "_work"
ARTIFACT_DIR = (
    pathlib.Path(os.environ["SQUINT_ARTIFACT_DIR"])
    if os.environ.get("SQUINT_ARTIFACT_DIR")
    else None
)

OUT_CSV = BENCH_DIR / "difficult_genomes_results.csv"
OUT_JSON = BENCH_DIR / "difficult_genomes_summary.json"
OUT_PLOT = BENCH_DIR / "difficult_genomes_plot.png"

PER_RUN_TIMEOUT_S = 120.0

HIV18_ACCESSIONS = [
    "K03455.1",
    "AB253421.1",
    "U52953.1",
    "K03454.1",
    "L20587.1",
    "AY169812.1",
    "AJ006022.1",
    "AY532635.1",
    "GU111555.1",
    "HQ179987.1",
    "AF103818.1",
    "X52154.1",
    "NC_001722.1",
    "M15390.1",
    "X61240.1",
    "U27200.1",
    "M33262.1",
    "X14307.1",
]
HIV1_CLADE = set(HIV18_ACCESSIONS[:12])
HIV2_CLADE = set(HIV18_ACCESSIONS[12:])

MTDNA24_ACCESSIONS = [
    "NC_012920.1",
    "NC_001643.1",
    "NC_005089.1",
    "NC_006853.1",
    "NC_001601.1",
    "NC_000891.1",
    "NC_001323.1",
    "NC_007897.1",
    "NC_002785.1",
    "NC_001922.1",
    "NC_000886.1",
    "NC_010972.2",
    "NC_006839.1",
    "NC_002805.1",
    "NC_005797.1",
    "NC_002333.2",
    "NC_004299.1",
    "NC_001960.1",
    "NC_004387.1",
    "NC_001804.1",
    "NC_001950.1",
    "NC_022415.1",
    "NC_001626.1",
    "NC_002807.1",
]
BIRD_MTDNA = {"NC_001323.1", "NC_007897.1", "NC_002785.1"}

DECOY_LENTIVIRUS = ["K03455.1", "U52953.1", "AJ006022.1", "AF103818.1", "NC_001722.1", "M33262.1"]
DECOY_MTDNA = [
    "NC_012920.1",
    "NC_001323.1",
    "NC_002805.1",
    "NC_002333.2",
    "NC_001804.1",
    "NC_001626.1",
]


def fetch_genbank_cached(accs: list[str], out_path: pathlib.Path) -> list[Any]:
    """Fetch GenBank records politely in batches <= 8 if not already cached locally."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not out_path.exists() or out_path.stat().st_size < 10_000:
        chunks: list[str] = []
        for i in range(0, len(accs), 8):
            batch = ",".join(accs[i : i + 8])
            url = (
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
                f"?db=nuccore&id={batch}&rettype=gbwithparts&retmode=text"
            )
            chunks.append(urllib.request.urlopen(url, timeout=45).read().decode("utf-8"))
            time.sleep(0.5)
        out_path.write_text("".join(chunks))
    recs = list(SeqIO.parse(out_path, "genbank"))
    assert len(recs) == len(accs), f"Expected {len(accs)} records in {out_path}, got {len(recs)}"
    return recs


def run_timed_cmd(cmd: list[str], timeout_s: float = PER_RUN_TIMEOUT_S, stdout_path: pathlib.Path | None = None) -> dict[str, Any]:
    """Run command under `/usr/bin/time -l`, enforcing wall timeout cleanly."""
    full_cmd = ["/usr/bin/time", "-l", *cmd]
    out_fh = open(stdout_path, "w") if stdout_path is not None else subprocess.DEVNULL
    proc: subprocess.Popen[str] | None = None
    t0 = time.perf_counter()
    try:
        proc = subprocess.Popen(
            full_cmd,
            stdout=out_fh,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        _, stderr_txt = proc.communicate(timeout=timeout_s)
        elapsed_s = time.perf_counter() - t0
    except subprocess.TimeoutExpired:
        elapsed_s = time.perf_counter() - t0
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.communicate(timeout=5)
            except Exception:
                pass
        return {
            "status": "TIMEOUT",
            "elapsed_s": round(elapsed_s, 4),
            "peak_rss_mib": None,
            "stderr": "",
        }
    finally:
        if out_fh is not subprocess.DEVNULL:
            out_fh.close()

    m_rss = re.search(r"(\d+)\s+maximum resident set size", stderr_txt)
    peak_rss_mib = (int(m_rss.group(1)) / (1024.0 * 1024.0)) if m_rss else None
    if proc.returncode != 0:
        return {
            "status": f"ERROR_{proc.returncode}",
            "elapsed_s": round(elapsed_s, 4),
            "peak_rss_mib": round(peak_rss_mib, 2) if peak_rss_mib is not None else None,
            "stderr": stderr_txt[-500:],
        }
    return {
        "status": "OK",
        "elapsed_s": round(elapsed_s, 4),
        "peak_rss_mib": round(peak_rss_mib, 2) if peak_rss_mib is not None else None,
        "stderr": stderr_txt,
    }


def _clean_nt_seq(seq_str: str) -> str:
    return "".join(ch for ch in seq_str.upper() if ch.isalpha())


# ---------------------------------------------------------------------------
# Case 1A: LANL HIV-1 Group M Pan-Subtype (34 genomes)
# ---------------------------------------------------------------------------
def build_c1a_dataset() -> dict[str, Any]:
    ref_aln_all = parse_fasta((HIV_DIR / "lanl_ref_2023.fasta").read_text())
    with open(HIV_DIR / "lanl_ref_2023_metadata.csv") as fh:
        meta = list(csv.DictReader(fh))
    hxb2_map = json.loads((HIV_DIR / "hxb2_gene_domain_map.json").read_text())

    by_tag: dict[str, list[str]] = {}
    for r in meta:
        tag = r["lanl_tag"]
        if not tag[0].isdigit():
            by_tag.setdefault(tag, []).append(r["seq_id"])

    selected_ids: list[str] = []
    hxb2_id = "Ref.B.FR.83.HXB2_LAI_IIIB_BRU.K03455"
    for tag in sorted(by_tag.keys()):
        cands = by_tag[tag]
        if tag == "B":
            others = [s for s in cands if s != hxb2_id]
            others.sort(key=lambda s: (-len(ref_aln_all[s].replace("-", "")), s))
            selected_ids.extend([hxb2_id, others[0]])
        else:
            cands.sort(key=lambda s: (-len(ref_aln_all[s].replace("-", "")), s))
            selected_ids.extend(cands[:2])

    ref_sub = {sid: ref_aln_all[sid].upper() for sid in selected_ids}
    orig_seqs = {sid: aln.replace("-", "") for sid, aln in ref_sub.items()}

    # Build exact 1-based HXB2 coordinate -> LANL ref column index
    hxb2_aln = ref_sub[hxb2_id]
    hxb2_to_col: dict[int, int] = {}
    hpos = 0
    for col, ch in enumerate(hxb2_aln):
        if ch != "-":
            hpos += 1
            hxb2_to_col[hpos] = col
    assert hpos == 9719

    def cols_for_intervals(intervals: list[tuple[int, int]]) -> set[int]:
        out: set[int] = set()
        for a, b in intervals:
            for p in range(a, b + 1):
                if p in hxb2_to_col:
                    out.add(hxb2_to_col[p])
        return out

    # Conserved core regions mapped from HXB2 map
    core_cols = cols_for_intervals(
        [
            (790, 2292),   # gag
            (2085, 5096),  # pol
            (5041, 5619),  # vif
            (5559, 5850),  # vpr
            (6225, 6611),  # env.gp120_C1
            (7758, 8795),  # env.gp41
        ]
    )
    ltr5_cols = cols_for_intervals([(1, 634)])
    ltr3_cols = cols_for_intervals([(9086, 9719)])
    v1v2_v4_cols = cols_for_intervals([(6612, 6812), (7377, 7478)])

    # Non-circular codon phase on single-frame gag-CA (1186..1878) and pol-RT..IN (2550..5040)
    col_phase: dict[int, int] = {}
    for p in range(1186, 1879):
        if p in hxb2_to_col:
            col_phase[hxb2_to_col[p]] = (p - 1186) % 3
    for p in range(2550, 5041):
        if p in hxb2_to_col:
            col_phase[hxb2_to_col[p]] = (p - 2550) % 3

    # Restricted reference alignment on conserved core columns ONLY (for Pair_Prec/Q/F1/TC)
    ref_core_aln: dict[str, list[str]] = {sid: [] for sid in selected_ids}
    # Map (sid, seq_pos) -> (region_tag, phase_or_none)
    pos_region: dict[str, list[str]] = {sid: [""] * len(orig_seqs[sid]) for sid in selected_ids}
    pos_core_phase: dict[str, list[int | None]] = {
        sid: [None] * len(orig_seqs[sid]) for sid in selected_ids
    }
    aln_len = len(hxb2_aln)
    cur_pos = {sid: 0 for sid in selected_ids}
    for c in range(aln_len):
        is_core = c in core_cols
        tag = (
            "5LTR"
            if c in ltr5_cols
            else ("3LTR" if c in ltr3_cols else ("HYPERVAR" if c in v1v2_v4_cols else ("CORE" if is_core else "OTHER")))
        )
        ph = col_phase.get(c)
        for sid in selected_ids:
            ch = ref_sub[sid][c]
            if ch != "-":
                sp = cur_pos[sid]
                pos_region[sid][sp] = tag
                if ph is not None:
                    pos_core_phase[sid][sp] = ph
                cur_pos[sid] += 1
            if is_core:
                ref_core_aln[sid].append(ch)

    ref_core_seqs = {sid: "".join(chars) for sid, chars in ref_core_aln.items()}
    # Build reference homologous pairs cleanly ONLY on core columns (`CORE_REF`)
    ref_core_info = extract_ref_structures(ref_core_seqs, False)
    core_orig_seqs = {sid: aln.replace("-", "") for sid, aln in ref_core_seqs.items()}
    # Remap original full sequence positions -> core-only local index for scoring Core_Pair_Prec/Q/F1/TC
    full_to_core_pos: dict[str, list[int | None]] = {}
    for sid in selected_ids:
        mp: list[int | None] = [None] * len(orig_seqs[sid])
        p_full = 0
        p_core = 0
        for c in range(aln_len):
            ch = ref_sub[sid][c]
            if ch != "-":
                if c in core_cols:
                    mp[p_full] = p_core
                    p_core += 1
                p_full += 1
        assert p_full == len(orig_seqs[sid])
        assert p_core == len(core_orig_seqs[sid])
        full_to_core_pos[sid] = mp

    return {
        "dataset": "C1A_HIV1_GroupM_PanSubtype_34",
        "mode": "dna",
        "orig_seqs": orig_seqs,
        "core_orig_seqs": core_orig_seqs,
        "ref_core_info": ref_core_info,
        "full_to_core_pos": full_to_core_pos,
        "pos_region": pos_region,
        "pos_core_phase": pos_core_phase,
        "hxb2_id": hxb2_id,
        "hxb2_map": hxb2_map,
    }


def score_c1a(comps: list[dict[str, str]], ds: dict[str, Any]) -> dict[str, Any]:
    """Score C1A alignment across conserved core + LTR + codon phase + V1-V2/V4 hypervar loops."""
    orig_seqs = ds["orig_seqs"]
    full_to_core = ds["full_to_core_pos"]
    pos_region = ds["pos_region"]
    pos_phase = ds["pos_core_phase"]

    # Extract sub-alignments on CORE residues only for exact LANL Core_Pair_Prec / Recall / F1 / TC
    core_comps: list[dict[str, str]] = []
    # Also compute LTR_Cross_Err, Core_Phase_Conc, HyperVar pair density in one linear pass!
    ltr5_aligned_pairs = 0
    ltr5_false_pairs = 0
    phase_total = 0
    phase_match = 0
    hyper_pairs = 0
    other_core_pairs = 0

    for comp in comps:
        sids = sorted(comp.keys())
        m = len(sids)
        if m == 0:
            continue
        aln_l = len(next(iter(comp.values())))
        # Validate lossless roundtrip
        for sid in sids:
            assert comp[sid].replace("-", "") == orig_seqs[sid], f"Lossless mismatch on {sid}"

        cur_p = [0] * m
        for c in range(aln_l):
            pres: list[tuple[int, str, int]] = []
            for idx, sid in enumerate(sids):
                ch = comp[sid][c]
                if ch != "-":
                    p = cur_p[idx]
                    pres.append((idx, sid, p))
                    cur_p[idx] = p + 1
            if len(pres) < 2:
                continue

            # Column histogram fast-path vs mixed column exact pair loop
            reg0 = pos_region[pres[0][1]][pres[0][2]]
            all_same_reg = all(pos_region[sid][p] == reg0 for _, sid, p in pres)
            k_pres = len(pres)
            n_pairs = k_pres * (k_pres - 1) // 2

            if all_same_reg:
                if reg0 == "5LTR":
                    ltr5_aligned_pairs += n_pairs
                elif reg0 == "HYPERVAR":
                    hyper_pairs += n_pairs
                elif reg0 == "CORE":
                    other_core_pairs += n_pairs
                # Fast-path phase check if any residues carry CORE phase
                ph_cnt = [0, 0, 0]
                for _, sid, p in pres:
                    ph = pos_phase[sid][p]
                    if ph is not None:
                        ph_cnt[ph] += 1
                tot_ph = ph_cnt[0] + ph_cnt[1] + ph_cnt[2]
                if tot_ph >= 2:
                    phase_total += tot_ph * (tot_ph - 1) // 2
                    phase_match += sum(v * (v - 1) // 2 for v in ph_cnt)
            else:
                # Exact pair loop for heterogeneous columns
                for i in range(k_pres):
                    _, sa, pa = pres[i]
                    ra = pos_region[sa][pa]
                    pha = pos_phase[sa][pa]
                    for j in range(i + 1, k_pres):
                        _, sb, pb = pres[j]
                        rb = pos_region[sb][pb]
                        phb = pos_phase[sb][pb]
                        if ra == "5LTR" or rb == "5LTR":
                            ltr5_aligned_pairs += 1
                            if ra != rb:
                                ltr5_false_pairs += 1
                        if ra == "HYPERVAR" and rb == "HYPERVAR":
                            hyper_pairs += 1
                        elif ra == "CORE" and rb == "CORE":
                            other_core_pairs += 1
                        if pha is not None and phb is not None:
                            phase_total += 1
                            if pha == phb:
                                phase_match += 1

        # Build aligned core sub-rows preserving shared columns of CORE bases
        core_col_flags: list[bool] = []
        tmp_p = [0] * m
        for c in range(aln_l):
            has_core_base = False
            for idx, sid in enumerate(sids):
                if comp[sid][c] != "-":
                    if full_to_core[sid][tmp_p[idx]] is not None:
                        has_core_base = True
                    tmp_p[idx] += 1
            core_col_flags.append(has_core_base)

        sub_dict: dict[str, list[str]] = {sid: [] for sid in sids}
        tmp_p = [0] * m
        for c, keep in enumerate(core_col_flags):
            if not keep:
                # advance non-core bases if any
                for idx, sid in enumerate(sids):
                    if comp[sid][c] != "-":
                        tmp_p[idx] += 1
                continue
            for idx, sid in enumerate(sids):
                ch = comp[sid][c]
                if ch != "-":
                    if full_to_core[sid][tmp_p[idx]] is not None:
                        sub_dict[sid].append(ch)
                    else:
                        sub_dict[sid].append("-")
                    tmp_p[idx] += 1
                else:
                    sub_dict[sid].append("-")
        core_comps.append({sid: "".join(chars) for sid, chars in sub_dict.items()})

    sc_core = score_alignment(ds["core_orig_seqs"], ds["ref_core_info"], core_comps, None)

    # Theoretical max hypervar pair count across sequences that actually cover V1V2/V4
    seq_hyper_cnt = [sum(1 for r in pos_region[sid] if r == "HYPERVAR") for sid in orig_seqs]
    seq_core_cnt = [sum(1 for r in pos_region[sid] if r == "CORE") for sid in orig_seqs]
    mean_h = sum(seq_hyper_cnt) / len(seq_hyper_cnt)
    n_pairs_tax = len(orig_seqs) * (len(orig_seqs) - 1) // 2

    return {
        "Core_Pair_Prec": round(sc_core["Pair_Prec"], 4),
        "Core_Pair_Recall_Q": round(sc_core["Pair_Recall_Q"], 4),
        "Core_Pair_F1": round(sc_core["Pair_F1"], 4),
        "Core_Exact_Col_TC": round(sc_core["Exact_Col_TC"], 4),
        "Core_Phase_Conc": round(phase_match / phase_total, 4) if phase_total > 0 else None,
        "LTR_Cross_Err": round(ltr5_false_pairs / ltr5_aligned_pairs, 4) if ltr5_aligned_pairs > 0 else 0.0,
        "HyperVar_V1V2_V4_PairsPerPair": round(hyper_pairs / max(1, n_pairs_tax * mean_h), 4),
        "LTR5_FalsePairs": ltr5_false_pairs,
        "LTR5_TotalPairs": ltr5_aligned_pairs,
    }


# ---------------------------------------------------------------------------
# Case 1B: Lentivirus Cross-Species (18 GenBank records)
# ---------------------------------------------------------------------------
def _canonicalize_lentivirus_features(rec: Any, hxb2_map: dict[str, Any]) -> tuple[list[frozenset[str]], list[dict[str, int]]]:
    """Parse canonical lentivirus gene/LTR labels and exact CDS reading-frame phases per nucleotide."""
    L = len(rec.seq)
    labels: list[set[str]] = [set() for _ in range(L)]
    phases: list[dict[str, int]] = [{} for _ in range(L)]

    def add_coding(gene_name: str, parts: list[tuple[int, int]], init_offset: int) -> None:
        cum = init_offset
        for s, e in parts:
            s_c = max(0, s)
            e_c = min(L, e)
            for p in range(s_c, e_c):
                labels[p].add(gene_name)
                phases[p][gene_name] = (cum + (p - s)) % 3
            cum += max(0, e - s)

    for f in rec.features:
        if f.type not in ("CDS", "LTR", "misc_feature"):
            continue
        gene_q = " ".join(f.qualifiers.get("gene", [])).lower()
        prod_q = " ".join(f.qualifiers.get("product", [])).lower()
        note_q = " ".join(f.qualifiers.get("note", [])).lower()
        txt = f"{gene_q} {prod_q} {note_q}"
        if f.type == "misc_feature":
            span = sum(int(pt.end) - int(pt.start) for pt in f.location.parts)
            if span < 1000 or "polymerase polyprotein" not in txt:
                continue
        parts = [(int(pt.start), int(pt.end)) for pt in f.location.parts if pt.strand != -1]
        if not parts:
            continue
        cs = int(f.qualifiers.get("codon_start", ["1"])[0]) - 1

        if f.type == "LTR":
            for s, e in parts:
                nm = "5LTR" if s < L // 2 else "3LTR"
                for p in range(s, min(L, e)):
                    labels[p].add(nm)
            continue

        # Special 2-part gag-pol ribosomal slip fusion
        if "gag-pol" in txt or ("gag" in txt and "pol" in txt and "fusion" in txt):
            if len(parts) >= 2:
                add_coding("gag", [parts[0]], -cs)
                len0 = parts[0][1] - parts[0][0]
                add_coding("pol", parts[1:], len0 - cs)
            else:
                add_coding("gag", parts, -cs)
            continue

        if re.search(r"\bgag\b", txt):
            add_coding("gag", parts, -cs)
        elif re.search(r"\bpol\b|polymerase polyprotein", txt):
            add_coding("pol", parts, -cs)
        elif re.search(r"\bvif\b|\bsor\b", txt):
            add_coding("vif", parts, -cs)
        elif re.search(r"\bvpx\b", txt):
            add_coding("vpx", parts, -cs)
        elif re.search(r"\bvpr\b|r \(orf\)", txt):
            add_coding("vpr", parts, -cs)
        elif re.search(r"\btat\b", txt):
            add_coding("tat", parts, -cs)
        elif re.search(r"\brev\b|\btrs\b", txt):
            add_coding("rev", parts, -cs)
        elif re.search(r"\bvpu\b|viral protein u", txt):
            add_coding("vpu", parts, -cs)
        elif re.search(r"\benv\b|envelope", txt):
            add_coding("env", parts, -cs)
        elif re.search(r"\bnef\b|27k protein", txt):
            add_coding("nef", parts, -cs)

    # Supplement HXB2 (K03455.1) with official vpu + 5LTR + 3LTR intervals missing from historical GenBank record
    if rec.id == "K03455.1":
        for g in hxb2_map["genes"]:
            if g["id"] in ("5LTR", "vpu", "3LTR"):
                s0 = g["start"] - 1
                e0 = g["end"]
                if g["id"] == "vpu":
                    add_coding("vpu", [(s0, e0)], 0)
                else:
                    for p in range(s0, min(L, e0)):
                        labels[p].add(g["id"])

    return [frozenset(s) for s in labels], phases


def build_c1b_dataset() -> dict[str, Any]:
    recs = fetch_genbank_cached(HIV18_ACCESSIONS, CACHE_DIR / "lentivirus_18.gb")
    hxb2_map = json.loads((HIV_DIR / "hxb2_gene_domain_map.json").read_text())

    orig_seqs: dict[str, str] = {}
    labels_by_id: dict[str, list[frozenset[str]]] = {}
    phases_by_id: dict[str, list[dict[str, int]]] = {}
    pol_len_by_id: dict[str, int] = {}

    for rec in recs:
        sid = rec.id
        seq_str = _clean_nt_seq(str(rec.seq))
        orig_seqs[sid] = seq_str
        lbls, phs = _canonicalize_lentivirus_features(rec, hxb2_map)
        assert len(lbls) == len(seq_str)
        labels_by_id[sid] = lbls
        phases_by_id[sid] = phs
        pol_len_by_id[sid] = sum(1 for fs in lbls if "pol" in fs)

        # Startup biological assertion on pol & gag start codons where complete
        if sid in ("AB253421.1", "K03455.1"):
            gag_0 = next(i for i, fs in enumerate(lbls) if "gag" in fs and phs[i].get("gag") == 0)
            assert seq_str[gag_0 : gag_0 + 3] == "ATG", f"{sid} gag start != ATG: {seq_str[gag_0:gag_0+3]}"
        if sid in ("NC_001722.1", "M15390.1"):
            assert any("vpx" in fs for fs in lbls), f"{sid} missing vpx"
        if sid in ("K03455.1", "L20587.1", "AJ006022.1"):
            assert any("vpu" in fs for fs in lbls), f"{sid} missing vpu"

    return {
        "dataset": "C1B_Lentivirus_HIV1_HIV2_SIV_18",
        "mode": "dna",
        "orig_seqs": orig_seqs,
        "labels": labels_by_id,
        "phases": phases_by_id,
        "pol_len": pol_len_by_id,
        "clade1": HIV1_CLADE,
        "clade2": HIV2_CLADE,
    }


def score_c1b(comps: list[dict[str, str]], ds: dict[str, Any]) -> dict[str, Any]:
    """Score non-circular GenBank lentivirus metrics (Cross_Gene_FDR, vpx/vpu false align, Pol phase)."""
    orig_seqs = ds["orig_seqs"]
    labels = ds["labels"]
    phases = ds["phases"]
    clade1 = ds["clade1"]
    clade2 = ds["clade2"]

    ann_pairs = 0
    cross_gene_pairs = 0

    # Pure lineage-specific vpx/vpu bases (excluding overlap with shared neighbors vif/vpr/env)
    vpx_only_total_cross_opportunities = 0
    vpx_only_false_aligned = 0
    vpu_only_total_cross_opportunities = 0
    vpu_only_false_aligned = 0

    # Cross-clade pol alignment & exact reading-frame phase concordance
    cross_pol_pairs = 0
    cross_pol_phase_ok = 0

    # Count available pure vpx-only / vpu-only bases against opposite clade
    n_c1 = len([s for s in orig_seqs if s in clade1])
    n_c2 = len([s for s in orig_seqs if s in clade2])
    vpx_only_bases = sum(
        1 for sid in orig_seqs if sid in clade2 for fs in labels[sid] if fs == frozenset({"vpx"})
    )
    vpu_only_bases = sum(
        1 for sid in orig_seqs if sid in clade1 for fs in labels[sid] if fs == frozenset({"vpu"})
    )
    vpx_only_total_cross_opportunities = vpx_only_bases
    vpu_only_total_cross_opportunities = vpu_only_bases

    vpx_aligned_to_c1: set[tuple[str, int]] = set()
    vpu_aligned_to_c2: set[tuple[str, int]] = set()
    cross_pol_max_pairs = sum(
        min(ds["pol_len"][s1], ds["pol_len"][s2])
        for s1 in clade1
        for s2 in clade2
        if ds["pol_len"][s1] >= 1500 and ds["pol_len"][s2] >= 1500
    )

    for comp in comps:
        sids = sorted(comp.keys())
        m = len(sids)
        if m == 0:
            continue
        aln_l = len(next(iter(comp.values())))
        for sid in sids:
            assert comp[sid].replace("-", "") == orig_seqs[sid]

        cur_p = [0] * m
        for c in range(aln_l):
            pres: list[tuple[str, int, frozenset[str], dict[str, int]]] = []
            for idx, sid in enumerate(sids):
                ch = comp[sid][c]
                if ch != "-":
                    p = cur_p[idx]
                    pres.append((sid, p, labels[sid][p], phases[sid][p]))
                    cur_p[idx] = p + 1
            k_p = len(pres)
            if k_p < 2:
                continue

            # Column fast-path: identical non-empty singleton gene set
            fs0 = pres[0][2]
            if len(fs0) == 1 and all(x[2] == fs0 for x in pres):
                g_only = next(iter(fs0))
                n_pairs = k_p * (k_p - 1) // 2
                ann_pairs += n_pairs
                # Cross-clade check on vpx/vpu (impossible if singleton unless cross-clade aligned to nothing? no, clade1 has no vpx!)
                if g_only == "pol":
                    c1_ph = [0, 0, 0]
                    c2_ph = [0, 0, 0]
                    for sid, _, _, phd in pres:
                        ph = phd.get("pol")
                        if ph is not None:
                            if sid in clade1:
                                c1_ph[ph] += 1
                            elif sid in clade2:
                                c2_ph[ph] += 1
                    tc1 = sum(c1_ph)
                    tc2 = sum(c2_ph)
                    if tc1 > 0 and tc2 > 0:
                        cross_pol_pairs += tc1 * tc2
                        cross_pol_phase_ok += c1_ph[0] * c2_ph[0] + c1_ph[1] * c2_ph[1] + c1_ph[2] * c2_ph[2]
                continue

            # Exact pairwise loop for heterogeneous / multi-gene columns
            for i in range(k_p):
                sa, pa, fsa, pha = pres[i]
                ca = 1 if sa in clade1 else 2
                for j in range(i + 1, k_p):
                    sb, pb, fsb, phb = pres[j]
                    cb = 1 if sb in clade1 else 2
                    if fsa and fsb:
                        ann_pairs += 1
                        if fsa.isdisjoint(fsb):
                            cross_gene_pairs += 1
                    if ca != cb:
                        if fsa == frozenset({"vpx"}):
                            vpx_aligned_to_c1.add((sa, pa))
                        elif fsb == frozenset({"vpx"}):
                            vpx_aligned_to_c1.add((sb, pb))
                        if fsa == frozenset({"vpu"}):
                            vpu_aligned_to_c2.add((sa, pa))
                        elif fsb == frozenset({"vpu"}):
                            vpu_aligned_to_c2.add((sb, pb))
                        if "pol" in fsa and "pol" in fsb:
                            cross_pol_pairs += 1
                            if pha.get("pol") is not None and pha.get("pol") == phb.get("pol"):
                                cross_pol_phase_ok += 1

    vpx_only_false_aligned = len(vpx_aligned_to_c1)
    vpu_only_false_aligned = len(vpu_aligned_to_c2)
    denom_acc = vpx_only_total_cross_opportunities + vpu_only_total_cross_opportunities
    return {
        "Cross_Gene_FDR": round(cross_gene_pairs / ann_pairs, 5) if ann_pairs > 0 else 0.0,
        "Vpx_Vpu_False_Align_Rate": round((vpx_only_false_aligned + vpu_only_false_aligned) / max(1, denom_acc), 4),
        "Vpx_False_nt": f"{vpx_only_false_aligned}/{vpx_only_total_cross_opportunities}",
        "Vpu_False_nt": f"{vpu_only_false_aligned}/{vpu_only_total_cross_opportunities}",
        "CrossClade_Pol_Cov": round(cross_pol_pairs / max(1, cross_pol_max_pairs), 4),
        "CrossClade_Pol_Phase_Conc": round(cross_pol_phase_ok / cross_pol_pairs, 4) if cross_pol_pairs > 0 else None,
        "Cross_Gene_Pairs": cross_gene_pairs,
        "Annotated_Pairs": ann_pairs,
    }


# ---------------------------------------------------------------------------
# Case 2A & 2B: Vertebrate Mitochondrial Genomes (24 complete RefSeq)
# ---------------------------------------------------------------------------
def _canonicalize_mt_feature_name(ftype: str, f: Any) -> str | None:
    q = f.qualifiers
    gene_s = " ".join(q.get("gene", [])).strip()
    prod_s = " ".join(q.get("product", [])).strip()
    note_s = " ".join(q.get("note", []) + q.get("standard_name", [])).strip()
    if ftype == "CDS":
        g_up = gene_s.upper()
        if g_up in {
            "ND1", "ND2", "COX1", "COX2", "ATP8", "ATP6", "COX3",
            "ND3", "ND4L", "ND4", "ND5", "ND6", "CYTB",
        }:
            return g_up
        return None
    if ftype == "rRNA":
        low = f"{gene_s} {prod_s}".lower()
        if "12s" in low or "s-rrna" in low or "rnr1" in low:
            return "rRNA_12S"
        if "16s" in low or "l-rrna" in low or "rnr2" in low:
            return "rRNA_16S"
        return None
    if ftype == "tRNA":
        m = re.search(r"tRNA-([A-Za-z]{3})", f"{prod_s} {gene_s}")
        if m:
            return f"tRNA-{m.group(1).capitalize()}"
        # Map 1-letter TRNF/trnF when product lacks tRNA-Xxx
        m2 = re.match(r"^(?:MT-)?TRN([A-Z])", gene_s.upper())
        aa3 = {
            "F": "Phe", "V": "Val", "L": "Leu", "I": "Ile", "Q": "Gln", "M": "Met",
            "W": "Trp", "A": "Ala", "N": "Asn", "C": "Cys", "Y": "Tyr", "S": "Ser",
            "D": "Asp", "K": "Lys", "G": "Gly", "R": "Arg", "H": "His", "E": "Glu",
            "T": "Thr", "P": "Pro",
        }
        if m2 and m2.group(1) in aa3:
            return f"tRNA-{aa3[m2.group(1)]}"
        return None
    if ftype == "D-loop":
        return "D-loop"
    if ftype == "misc_feature":
        low = f"{note_s} {gene_s}".lower()
        if "control region" in low or "d-loop" in low:
            return "D-loop"
    return None


def _parse_single_mt_record(rec: Any, rotate_to_trnf: bool) -> dict[str, Any]:
    L = len(rec.seq)
    seq_str = _clean_nt_seq(str(rec.seq))
    assert len(seq_str) == L
    labels: list[set[str]] = [set() for _ in range(L)]
    phases: list[dict[str, int]] = [{} for _ in range(L)]
    phe_start: int | None = None

    for f in rec.features:
        canon = _canonicalize_mt_feature_name(f.type, f)
        if canon is None:
            continue
        if canon == "tRNA-Phe" and phe_start is None:
            phe_start = int(f.location.parts[0].start)

        if f.type == "CDS":
            cs = int(f.qualifiers.get("codon_start", ["1"])[0]) - 1
            cum = -cs
            for pt in f.location.parts:
                s, e, st = int(pt.start), int(pt.end), pt.strand
                span = e - s
                if st == -1:
                    for p in range(s, e):
                        labels[p].add(canon)
                        rel = cum + (e - 1 - p)
                        if rel >= 0:
                            phases[p][canon] = rel % 3
                else:
                    for p in range(s, e):
                        labels[p].add(canon)
                        rel = cum + (p - s)
                        if rel >= 0:
                            phases[p][canon] = rel % 3
                cum += span
        else:
            for pt in f.location.parts:
                for p in range(int(pt.start), int(pt.end)):
                    labels[p].add(canon)

    assert phe_start is not None, f"Missing tRNA-Phe in {rec.id}"

    # Verify COX1 (+1) and ND6 (-1) start codons on Human reference NC_012920.1
    if rec.id == "NC_012920.1":
        cox1_0 = [i for i, fs in enumerate(labels) if "COX1" in fs and phases[i].get("COX1") == 0][0]
        assert seq_str[cox1_0 : cox1_0 + 3] == "ATG"
        nd6_end = max(i for i, fs in enumerate(labels) if "ND6" in fs)
        assert phases[nd6_end]["ND6"] == 0
        assert str(Seq(seq_str[nd6_end - 2 : nd6_end + 1]).reverse_complement()) == "ATG"

    if rotate_to_trnf:
        seq_str = seq_str[phe_start:] + seq_str[:phe_start]
        labels = labels[phe_start:] + labels[:phe_start]
        phases = phases[phe_start:] + phases[:phe_start]
        assert "tRNA-Phe" in labels[0], f"Rotation check failed on {rec.id}"

    f_labels = [frozenset(s) for s in labels]
    cds_len_map: dict[str, int] = {}
    trna_count = 0
    for fs in f_labels:
        for lbl in fs:
            if lbl.startswith("tRNA-"):
                trna_count += 1
            elif not lbl.startswith("rRNA_") and lbl != "D-loop":
                cds_len_map[lbl] = cds_len_map.get(lbl, 0) + 1

    return {
        "seq": seq_str,
        "labels": f_labels,
        "phases": phases,
        "cds_len": cds_len_map,
        "trna_len": trna_count,
        "phe_start": phe_start,
    }


def build_c2_dataset(rotate_to_trnf: bool) -> dict[str, Any]:
    recs = fetch_genbank_cached(MTDNA24_ACCESSIONS, CACHE_DIR / "vertebrate_mtdna_24.gb")
    ds_name = (
        "C2B_Vertebrate_mtDNA_Canonical_trnF_24"
        if rotate_to_trnf
        else "C2A_Vertebrate_mtDNA_RawRefSeq_24"
    )
    orig_seqs: dict[str, str] = {}
    labels_by_id: dict[str, list[frozenset[str]]] = {}
    phases_by_id: dict[str, list[dict[str, int]]] = {}
    cds_len_by_id: dict[str, dict[str, int]] = {}
    trna_len_by_id: dict[str, int] = {}

    for rec in recs:
        parsed = _parse_single_mt_record(rec, rotate_to_trnf=rotate_to_trnf)
        orig_seqs[rec.id] = parsed["seq"]
        labels_by_id[rec.id] = parsed["labels"]
        phases_by_id[rec.id] = parsed["phases"]
        cds_len_by_id[rec.id] = parsed["cds_len"]
        trna_len_by_id[rec.id] = parsed["trna_len"]

    return {
        "dataset": ds_name,
        "mode": "dna",
        "orig_seqs": orig_seqs,
        "labels": labels_by_id,
        "phases": phases_by_id,
        "cds_len": cds_len_by_id,
        "trna_len": trna_len_by_id,
    }


def score_c2(comps: list[dict[str, str]], ds: dict[str, Any]) -> dict[str, Any]:
    """Score non-circular GenBank mtDNA metrics (Cross_Feature_FDR, 13-CDS & ND6 phase, tRNA, Avian ND6)."""
    orig_seqs = ds["orig_seqs"]
    labels = ds["labels"]
    phases = ds["phases"]
    cds_len = ds["cds_len"]
    trna_len = ds["trna_len"]

    ann_pairs = 0
    cross_feat_pairs = 0

    same_cds_pairs = 0
    same_cds_phase_ok = 0
    nd6_pairs = 0
    nd6_phase_ok = 0

    trna_same_pairs = 0
    trna_any_pairs = 0

    avian_nd6_trne_total = 0
    avian_nd6_trne_false = 0

    dloop_crossclass_pairs = 0
    bird_set = BIRD_MTDNA
    all_ids = sorted(orig_seqs.keys())

    cds_genes = {"ND1", "ND2", "COX1", "COX2", "ATP8", "ATP6", "COX3", "ND3", "ND4L", "ND4", "ND5", "ND6", "CYTB"}
    max_cds_pairs = 0
    for i in range(len(all_ids)):
        for j in range(i + 1, len(all_ids)):
            si, sj = all_ids[i], all_ids[j]
            for g in cds_genes:
                max_cds_pairs += min(cds_len[si].get(g, 0), cds_len[sj].get(g, 0))
    max_trna_pairs = sum(
        min(trna_len[all_ids[i]], trna_len[all_ids[j]])
        for i in range(len(all_ids))
        for j in range(i + 1, len(all_ids))
    )

    for comp in comps:
        sids = sorted(comp.keys())
        m = len(sids)
        if m == 0:
            continue
        aln_l = len(next(iter(comp.values())))
        for sid in sids:
            assert comp[sid].replace("-", "") == orig_seqs[sid]

        cur_p = [0] * m
        for c in range(aln_l):
            pres: list[tuple[str, int, frozenset[str], dict[str, int]]] = []
            for idx, sid in enumerate(sids):
                ch = comp[sid][c]
                if ch != "-":
                    p = cur_p[idx]
                    pres.append((sid, p, labels[sid][p], phases[sid][p]))
                    cur_p[idx] = p + 1
            k_p = len(pres)
            if k_p < 2:
                continue

            fs0 = pres[0][2]
            if len(fs0) == 1 and all(x[2] == fs0 for x in pres):
                lbl = next(iter(fs0))
                n_pairs = k_p * (k_p - 1) // 2
                ann_pairs += n_pairs
                if lbl in cds_genes:
                    same_cds_pairs += n_pairs
                    ph_c = [0, 0, 0]
                    for _, _, _, phd in pres:
                        ph = phd.get(lbl)
                        if ph is not None:
                            ph_c[ph] += 1
                    tot_ph = ph_c[0] + ph_c[1] + ph_c[2]
                    if tot_ph >= 2:
                        ok_ph = sum(v * (v - 1) // 2 for v in ph_c)
                        same_cds_phase_ok += ok_ph
                        if lbl == "ND6":
                            nd6_pairs += tot_ph * (tot_ph - 1) // 2
                            nd6_phase_ok += ok_ph
                elif lbl.startswith("tRNA-"):
                    trna_same_pairs += n_pairs
                    trna_any_pairs += n_pairs
                elif lbl == "D-loop":
                    n_bird = sum(1 for sid, _, _, _ in pres if sid in bird_set)
                    dloop_crossclass_pairs += n_bird * (k_p - n_bird)
                continue

            for i in range(k_p):
                sa, pa, fsa, pha = pres[i]
                is_ba = sa in bird_set
                for j in range(i + 1, k_p):
                    sb, pb, fsb, phb = pres[j]
                    is_bb = sb in bird_set
                    if fsa and fsb:
                        ann_pairs += 1
                        shared = fsa & fsb
                        if not shared:
                            cross_feat_pairs += 1
                        else:
                            for g in shared:
                                if g in cds_genes:
                                    same_cds_pairs += 1
                                    p1, p2 = pha.get(g), phb.get(g)
                                    if p1 is not None and p2 is not None and p1 == p2:
                                        same_cds_phase_ok += 1
                                    if g == "ND6":
                                        nd6_pairs += 1
                                        if p1 is not None and p2 is not None and p1 == p2:
                                            nd6_phase_ok += 1
                                elif g.startswith("tRNA-"):
                                    trna_same_pairs += 1
                                elif g == "D-loop" and is_ba != is_bb:
                                    dloop_crossclass_pairs += 1

                        if any(x.startswith("tRNA-") for x in fsa) or any(x.startswith("tRNA-") for x in fsb):
                            trna_any_pairs += 1

                        # Avian ND6 / tRNA-Glu translocation false-alignment check
                        if is_ba != is_bb:
                            bird_fs = fsa if is_ba else fsb
                            non_fs = fsb if is_ba else fsa
                            if ("ND6" in bird_fs or "tRNA-Glu" in bird_fs):
                                avian_nd6_trne_total += 1
                                if bird_fs.isdisjoint(non_fs):
                                    avian_nd6_trne_false += 1

    return {
        "Cross_Feature_FDR": round(cross_feat_pairs / ann_pairs, 5) if ann_pairs > 0 else 0.0,
        "CDS_Homolog_Cov": round(same_cds_pairs / max(1, max_cds_pairs), 4),
        "CDS_Phase_Conc": round(same_cds_phase_ok / same_cds_pairs, 4) if same_cds_pairs > 0 else None,
        "ND6_Phase_Conc": round(nd6_phase_ok / nd6_pairs, 4) if nd6_pairs > 0 else None,
        "tRNA_Prec": round(trna_same_pairs / trna_any_pairs, 4) if trna_any_pairs > 0 else None,
        "tRNA_Homolog_Cov": round(trna_same_pairs / max(1, max_trna_pairs), 4),
        "Avian_ND6_FDR": round(avian_nd6_trne_false / avian_nd6_trne_total, 4) if avian_nd6_trne_total > 0 else 0.0,
        "DLoop_BirdNonBird_Pairs": dloop_crossclass_pairs,
        "Cross_Feature_Pairs": cross_feat_pairs,
        "Annotated_Pairs": ann_pairs,
    }


# ---------------------------------------------------------------------------
# Case 3: Mixed Cross-Kingdom Decoy Cohort (6 Lentiviruses + 6 Vertebrate mtDNAs)
# ---------------------------------------------------------------------------
def build_c3_dataset(c1b_ds: dict[str, Any], c2b_ds: dict[str, Any]) -> dict[str, Any]:
    orig_seqs: dict[str, str] = {}
    kingdom: dict[str, str] = {}
    labels: dict[str, list[frozenset[str]]] = {}
    phases: dict[str, list[dict[str, int]]] = {}

    for sid in DECOY_LENTIVIRUS:
        orig_seqs[sid] = c1b_ds["orig_seqs"][sid]
        kingdom[sid] = "VIR"
        labels[sid] = [frozenset(f"VIR:{x}" for x in fs) for fs in c1b_ds["labels"][sid]]
        phases[sid] = [{f"VIR:{k}": v for k, v in d.items()} for d in c1b_ds["phases"][sid]]

    for sid in DECOY_MTDNA:
        orig_seqs[sid] = c2b_ds["orig_seqs"][sid]
        kingdom[sid] = "MITO"
        labels[sid] = [frozenset(f"MITO:{x}" for x in fs) for fs in c2b_ds["labels"][sid]]
        phases[sid] = [{f"MITO:{k}": v for k, v in d.items()} for d in c2b_ds["phases"][sid]]

    return {
        "dataset": "C3_Viral_vs_Mito_Decoy_12",
        "mode": "dna",
        "orig_seqs": orig_seqs,
        "kingdom": kingdom,
        "labels": labels,
        "phases": phases,
    }


def score_c3(comps: list[dict[str, str]], ds: dict[str, Any]) -> dict[str, Any]:
    orig_seqs = ds["orig_seqs"]
    kingdom = ds["kingdom"]
    labels = ds["labels"]

    total_pairs = 0
    viral_mito_pairs = 0
    ann_pairs = 0
    cross_gene_pairs = 0

    for comp in comps:
        sids = sorted(comp.keys())
        m = len(sids)
        if m < 2:
            continue
        aln_l = len(next(iter(comp.values())))
        for sid in sids:
            assert comp[sid].replace("-", "") == orig_seqs[sid]
        cur_p = [0] * m
        has_v = any(kingdom[s] == "VIR" for s in sids)
        has_m = any(kingdom[s] == "MITO" for s in sids)

        for c in range(aln_l):
            pres: list[tuple[str, int]] = []
            for idx, sid in enumerate(sids):
                if comp[sid][c] != "-":
                    pres.append((sid, cur_p[idx]))
                    cur_p[idx] += 1
            k_p = len(pres)
            if k_p < 2:
                continue
            n_p = k_p * (k_p - 1) // 2
            total_pairs += n_p
            if has_v and has_m:
                nv = sum(1 for sid, _ in pres if kingdom[sid] == "VIR")
                nm = k_p - nv
                if nv > 0 and nm > 0:
                    viral_mito_pairs += nv * nm
            fs0 = labels[pres[0][0]][pres[0][1]]
            if len(fs0) == 1 and all(labels[sid][p] == fs0 for sid, p in pres):
                ann_pairs += n_p
            else:
                for i in range(k_p):
                    sa, pa = pres[i]
                    fsa = labels[sa][pa]
                    if not fsa:
                        continue
                    for j in range(i + 1, k_p):
                        sb, pb = pres[j]
                        fsb = labels[sb][pb]
                        if fsb:
                            ann_pairs += 1
                            if fsa.isdisjoint(fsb):
                                cross_gene_pairs += 1

    return {
        "Viral_Mito_Cross_Pairs": viral_mito_pairs,
        "Cross_Kingdom_FDR": round(viral_mito_pairs / total_pairs, 5) if total_pairs > 0 else 0.0,
        "Cross_Feature_FDR": round(cross_gene_pairs / ann_pairs, 5) if ann_pairs > 0 else 0.0,
        "Total_Aligned_Pairs": total_pairs,
    }


def assert_fast_hist_matches_bruteforce(ds_c2: dict[str, Any]) -> None:
    """Verify exact equivalence between column histogram fast-path and brute-force pairs on synthetic MSA."""
    sids = ["NC_012920.1", "NC_001323.1", "NC_002333.2"]
    comps = [{s: ds_c2["orig_seqs"][s][:1500] for s in sids}]
    res_fast = score_c2(comps, {"orig_seqs": {s: ds_c2["orig_seqs"][s][:1500] for s in sids},
                                "labels": {s: ds_c2["labels"][s][:1500] for s in sids},
                                "phases": {s: ds_c2["phases"][s][:1500] for s in sids},
                                "cds_len": ds_c2["cds_len"],
                                "trna_len": ds_c2["trna_len"]})
    # Brute-force reference calculation over the exact same alignment
    ann = 0
    cf = 0
    for c in range(1500):
        for i in range(3):
            for j in range(i + 1, 3):
                fa = ds_c2["labels"][sids[i]][c]
                fb = ds_c2["labels"][sids[j]][c]
                if fa and fb:
                    ann += 1
                    if fa.isdisjoint(fb):
                        cf += 1
    assert abs(res_fast["Cross_Feature_FDR"] - round(cf / ann, 5)) < 1e-9
    assert res_fast["Cross_Feature_Pairs"] == cf
    assert res_fast["Annotated_Pairs"] == ann


# ---------------------------------------------------------------------------
# Tool Execution & Format Verification
# ---------------------------------------------------------------------------
def run_all_tools_on_dataset(
    ds: dict[str, Any],
    scorer_fn: Any,
    tools: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    name = ds["dataset"]
    orig_seqs: dict[str, str] = ds["orig_seqs"]
    n_taxa = len(orig_seqs)
    min_l = min(len(v) for v in orig_seqs.values())
    max_l = max(len(v) for v in orig_seqs.values())

    in_squint = WORK_DIR / f"{name}.input.fa"
    with open(in_squint, "w") as fh:
        for sid in orig_seqs:
            fh.write(f">{sid}\n{orig_seqs[sid]}\n")

    in_aliased = WORK_DIR / f"{name}.aliased.fa"
    alias_map = write_aliased_fasta(orig_seqs, in_aliased)

    rows: list[dict[str, Any]] = []
    squint_ledger: dict[str, Any] | None = None

    for tool in tools:
        row: dict[str, Any] = {
            "dataset": name,
            "N": n_taxa,
            "L_range": f"{min_l}-{max_l}",
            "tool": tool,
            "status": "OK",
            "elapsed_s": None,
            "peak_rss_mib": None,
            "output_kib": None,
            "components_K": 1,
            "horizons": 0,
        }

        if tool == "squint_fa":
            out_fa = WORK_DIR / f"{name}.squint.fa"
            out_json = WORK_DIR / f"{name}.ledger.json"
            res = run_timed_cmd([SQUINT, str(in_squint), "-o", str(out_fa), "-j", str(out_json), "--mode", ds["mode"]])
            row["status"] = res["status"]
            row["elapsed_s"] = res["elapsed_s"]
            row["peak_rss_mib"] = res["peak_rss_mib"]
            if res["status"] == "OK":
                row["output_kib"] = round(out_fa.stat().st_size / 1024.0, 2)
                m_k = re.search(r"\bcomponents=(\d+)\b", res["stderr"])
                row["components_K"] = int(m_k.group(1)) if m_k else 1
                squint_ledger = json.loads(out_json.read_text())
                row["horizons"] = sum(len(c.get("horizon_bubbles", [])) for c in squint_ledger["components"])
                comps = parse_squint_components_fasta(out_fa)
                row.update(scorer_fn(comps, ds))
        elif tool == "squint_hdb":
            out_hdb = WORK_DIR / f"{name}.squint.hdb"
            rt_fa = WORK_DIR / f"{name}.squint_hdb.rt.fa"
            res = run_timed_cmd([SQUINT, str(in_squint), "-o", str(out_hdb), "--mode", ds["mode"]])
            row["status"] = res["status"]
            row["elapsed_s"] = res["elapsed_s"]
            row["peak_rss_mib"] = res["peak_rss_mib"]
            if res["status"] == "OK":
                row["output_kib"] = round(out_hdb.stat().st_size / 1024.0, 2)
                m_k = re.search(r"\bcomponents=(\d+)\b", res["stderr"])
                row["components_K"] = int(m_k.group(1)) if m_k else 1
                subprocess.run([SQUINT, "convert", str(out_hdb), "-o", str(rt_fa)], capture_output=True, text=True, check=True)
                comps = parse_squint_components_fasta(rt_fa)
                row.update(scorer_fn(comps, ds))
                if squint_ledger is not None:
                    row["horizons"] = sum(len(c.get("horizon_bubbles", [])) for c in squint_ledger["components"])
        else:
            out_aliased = WORK_DIR / f"{name}.{tool}.aliased.fa"
            if tool == "mafft_auto":
                res = run_timed_cmd([MAFFT, "--auto", "--thread", "-1", "--quiet", str(in_aliased)], stdout_path=out_aliased)
            elif tool == "mafft_linsi":
                res = run_timed_cmd(
                    [MAFFT, "--localpair", "--maxiterate", "1000", "--thread", "-1", "--quiet", str(in_aliased)],
                    stdout_path=out_aliased,
                )
            elif tool == "muscle":
                res = run_timed_cmd([MUSCLE, "-in", str(in_aliased), "-out", str(out_aliased), "-quiet"])
            elif tool == "clustalo":
                res = run_timed_cmd([CLUSTALO, "-i", str(in_aliased), "-o", str(out_aliased), "--threads=1", "--force"])
            else:
                raise ValueError(tool)

            row["status"] = res["status"]
            row["elapsed_s"] = res["elapsed_s"]
            row["peak_rss_mib"] = res["peak_rss_mib"]
            if res["status"] == "OK":
                row["output_kib"] = round(out_aliased.stat().st_size / 1024.0, 2)
                aln = {k: v.upper() for k, v in restore_aliased_fasta(out_aliased, alias_map).items()}
                row.update(scorer_fn([aln], ds))

        rows.append(row)
        print(
            f"  [{name:34s}] {tool:12s} | {row['status']:7s} | "
            f"t={str(row['elapsed_s']):>7s}s | rss={str(row['peak_rss_mib']):>7s} MiB | "
            f"K={row['components_K']} | H={row['horizons']}",
            flush=True,
        )

    return rows, squint_ledger


def run_region_subset_checks() -> list[dict[str, Any]]:
    """Verify `squint subset --region` surgical gene extraction on C1A pol-RT and C2B COX1."""
    checks: list[dict[str, Any]] = []
    targets = [
        ("C1A_HXB2_polRT", "C1A_HIV1_GroupM_PanSubtype_34", "Ref.B.FR.83.HXB2_LAI_IIIB_BRU.K03455:2550-3569"),
        ("C2B_Human_COX1", "C2B_Vertebrate_mtDNA_Canonical_trnF_24", "NC_012920.1:5328-6869"),
    ]
    for label, ds_name, region in targets:
        hdb_path = WORK_DIR / f"{ds_name}.squint.hdb"
        out_fa = WORK_DIR / f"{label}.subset.fa"
        t0 = time.perf_counter()
        proc = subprocess.run(
            [SQUINT, "subset", str(hdb_path), "--region", region, "-o", str(out_fa)],
            capture_output=True,
            text=True,
            check=True,
        )
        dt_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        sub_aln = parse_fasta(out_fa.read_text())
        n_s = len(sub_aln)
        l_s = len(next(iter(sub_aln.values()))) if sub_aln else 0
        checks.append(
            {
                "label": label,
                "region": region,
                "subset_ms": dt_ms,
                "num_seqs": n_s,
                "subset_cols": l_s,
                "stderr": proc.stderr.strip(),
            }
        )
        print(f"  [subset check] {label}: region={region} -> {n_s} seqs x {l_s} cols in {dt_ms} ms", flush=True)
    return checks


# ---------------------------------------------------------------------------
# Visualization (2x2 Figure)
# ---------------------------------------------------------------------------
def build_figure(
    all_rows: list[dict[str, Any]],
    c1b_ledger: dict[str, Any] | None,
    c2b_ledger: dict[str, Any] | None,
    c1b_ds: dict[str, Any],
    c2b_ds: dict[str, Any],
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.0,
        }
    )

    tools_order = ["squint_fa", "mafft_auto", "mafft_linsi", "muscle", "clustalo"]
    tool_labels = {
        "squint_fa": "SQUINT",
        "mafft_auto": "MAFFT --auto",
        "mafft_linsi": "MAFFT L-INS-i",
        "muscle": "MUSCLE 3.8",
        "clustalo": "ClustalO",
    }
    tool_colors = {
        "squint_fa": "#1b6ca8",
        "mafft_auto": "#2a9d8f",
        "mafft_linsi": "#e9c46a",
        "muscle": "#f4a261",
        "clustalo": "#e76f51",
    }

    by_key = {(r["dataset"], r["tool"]): r for r in all_rows}
    ds_list = [
        ("C1A_HIV1_GroupM_PanSubtype_34", "C1A\nHIV-1 (34)"),
        ("C1B_Lentivirus_HIV1_HIV2_SIV_18", "C1B\nLenti (18)"),
        ("C2A_Vertebrate_mtDNA_RawRefSeq_24", "C2A\nmt Raw (24)"),
        ("C2B_Vertebrate_mtDNA_Canonical_trnF_24", "C2B\nmt Rot (24)"),
        ("C3_Viral_vs_Mito_Decoy_12", "C3\nDecoy (12)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.8), dpi=200)
    plt.subplots_adjust(hspace=0.38, wspace=0.25)

    # Panel A: Wall-clock Runtime (s, log scale)
    ax = axes[0, 0]
    x = np.arange(len(ds_list))
    w = 0.155
    for ti, t in enumerate(tools_order):
        vals = []
        statuses = []
        for ds_k, _ in ds_list:
            r = by_key.get((ds_k, t))
            if r and r["status"] == "OK":
                vals.append(max(0.05, float(r["elapsed_s"])))
                statuses.append("OK")
            elif r and r["status"] == "TIMEOUT":
                vals.append(PER_RUN_TIMEOUT_S)
                statuses.append("TO")
            else:
                vals.append(0.0)
                statuses.append("NA")
        xpos = x + (ti - 2.0) * w
        ax.bar(xpos, vals, width=w * 0.92, color=tool_colors[t], label=tool_labels[t], zorder=3)
        for bx, val, st in zip(xpos, vals, statuses):
            if st == "TO":
                ax.text(bx, val * 1.08, "TO", ha="center", va="bottom", fontsize=6.5, color="#b00020", fontweight="bold")
    ax.axhline(PER_RUN_TIMEOUT_S, color="#b00020", linestyle="--", linewidth=0.9, zorder=2)
    ax.set_yscale("log")
    ax.set_ylim(0.1, 700.0)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in ds_list], fontsize=8.5)
    ax.set_ylabel("Wall-clock runtime (s, log scale)")
    ax.set_title("A. Whole-Genome Runtime (TO = 120s Timeout)")
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
    ax.legend(loc="upper left", ncol=3, fontsize=7.5, framealpha=0.92)

    # Panel B: False-Homology Hazards Across Scenarios (%)
    ax = axes[0, 1]
    hazards = [
        ("C1A_HIV1_GroupM_PanSubtype_34", "LTR_Cross_Err", "C1A: 5' vs 3' LTR\nCross-Align"),
        ("C1B_Lentivirus_HIV1_HIV2_SIV_18", "Vpx_Vpu_False_Align_Rate", "C1B: vpx/vpu\nFalse Align"),
        ("C2B_Vertebrate_mtDNA_Canonical_trnF_24", "Avian_ND6_FDR", "C2B: Avian ND6\nRearr. FDR"),
        ("C3_Viral_vs_Mito_Decoy_12", "Cross_Kingdom_FDR", "C3: Viral-vs-Mito\nCross FDR"),
    ]
    xh = np.arange(len(hazards))
    for ti, t in enumerate(tools_order):
        hvals = []
        hmiss = []
        for ds_k, field, _ in hazards:
            r = by_key.get((ds_k, t))
            if r and r["status"] == "OK" and r.get(field) is not None:
                hvals.append(100.0 * float(r[field]))
                hmiss.append(False)
            else:
                hvals.append(0.0)
                hmiss.append(True)
        xpos = xh + (ti - 2.0) * w
        ax.bar(xpos, hvals, width=w * 0.92, color=tool_colors[t], label=tool_labels[t], zorder=3)
        for bx, val, ms in zip(xpos, hvals, hmiss):
            if ms:
                ax.text(bx, 1.5, "TO", ha="center", va="bottom", fontsize=6.5, color="#b00020", rotation=90)
            elif val > 0.1:
                ax.text(bx, val + 1.0, f"{val:.1f}" if val < 10 else f"{val:.0f}", ha="center", va="bottom", fontsize=6.5)
            elif t == "squint_fa":
                ax.text(bx, 1.0, f"{val:.1f}%", ha="center", va="bottom", fontsize=6.5, color="#1b6ca8", fontweight="bold")
    ax.set_xticks(xh)
    ax.set_xticklabels([lbl for _, _, lbl in hazards], fontsize=8.5)
    ax.set_ylabel("False homology / cross-align (%)")
    ax.set_ylim(0, 112)
    ax.set_title("B. Non-Circular Decoy, Accessory-Gene & Rearrangement Hazards")
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)

    # Panel C: Reading-Frame Codon-Phase Concordance (Blind DNA Mode)
    ax = axes[1, 0]
    phase_cats = [
        ("C1A_HIV1_GroupM_PanSubtype_34", "Core_Phase_Conc", "C1A: HIV-1 M\ngag/pol Core"),
        ("C1B_Lentivirus_HIV1_HIV2_SIV_18", "CrossClade_Pol_Phase_Conc", "C1B: HIV1<->HIV2\npol"),
        ("C2B_Vertebrate_mtDNA_Canonical_trnF_24", "CDS_Phase_Conc", "C2B: 13 mtDNA\nCDSs (500 Myr)"),
        ("C2B_Vertebrate_mtDNA_Canonical_trnF_24", "ND6_Phase_Conc", "C2B: Light-Strand\nND6"),
    ]
    xp = np.arange(len(phase_cats))
    for ti, t in enumerate(tools_order):
        pvals = []
        pmiss = []
        for ds_k, field, _ in phase_cats:
            r = by_key.get((ds_k, t))
            if r and r["status"] == "OK" and r.get(field) is not None:
                pvals.append(100.0 * float(r[field]))
                pmiss.append(False)
            else:
                pvals.append(0.0)
                pmiss.append(True)
        xpos = xp + (ti - 2.0) * w
        ax.bar(xpos, pvals, width=w * 0.92, color=tool_colors[t], zorder=3)
        for bx, val, ms in zip(xpos, pvals, pmiss):
            if ms:
                ax.text(bx, 2.0, "TO", ha="center", va="bottom", fontsize=6.5, color="#b00020", rotation=90)
    ax.axhline(33.33, color="#666666", linestyle="--", linewidth=0.9)
    ax.text(2.16, 35.0, "Random null\n(33.3%)", fontsize=6.8, color="#444444", ha="center", va="bottom", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.0))
    ax.set_xticks(xp)
    ax.set_xticklabels([lbl for _, _, lbl in phase_cats], fontsize=8.5)
    ax.set_ylabel("Same-frame codon phase concordance (%)")
    ax.set_ylim(0, 108)
    ax.set_title("C. Blind DNA Reading-Frame Phase Concordance")
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)

    # Panel D: SQUINT Genome-Wide Scale Anatomy along Human mtDNA (C2B)
    ax = axes[1, 1]
    if c2b_ledger is not None and c2b_ledger["components"]:
        comp = c2b_ledger["components"][0]
        ref_id = "NC_012920.1"
        if ref_id in dict(comp["aligned_fasta"]):
            aln_s = dict(comp["aligned_fasta"])[ref_id]
            cols_meta = comp["columns"]
            seq_len = len(c2b_ds["orig_seqs"][ref_id])
            win = 400
            bins = list(range(0, seq_len, win))
            g_frac, c_frac, p_frac = [], [], []
            scales_by_pos = []
            for ci, ch in enumerate(aln_s):
                if ch != "-":
                    scales_by_pos.append(cols_meta[ci]["scale"])
            for b0 in bins:
                sl = scales_by_pos[b0 : b0 + win]
                tot = max(1, len(sl))
                g_frac.append(100.0 * sum(1 for x in sl if x == "GlobalFine") / tot)
                c_frac.append(100.0 * sum(1 for x in sl if x == "CladeLocalFine") / tot)
                p_frac.append(100.0 * sum(1 for x in sl if x == "LineagePrivate") / tot)
            xk = np.array(bins) / 1000.0
            ax.stackplot(
                xk,
                g_frac,
                c_frac,
                p_frac,
                labels=["GlobalFine core", "CladeLocalFine bubble", "LineagePrivate"],
                colors=["#1b6ca8", "#f4a261", "#d3d3d3"],
                alpha=0.88,
            )
            lbls = c2b_ds["labels"][ref_id]
            for locus, short, y_pos in [
                ("rRNA_16S", "16S", 102.0),
                ("COX1", "COX1", 102.0),
                ("ATP8", "ATP8", 102.0),
                ("ND5", "ND5", 102.0),
                ("ND6", "ND6", 108.5),
                ("CYTB", "CYTB", 102.0),
                ("D-loop", "D-loop", 108.5),
            ]:
                idxs = [i for i, fs in enumerate(lbls) if locus in fs]
                if idxs:
                    mid = 0.5 * (idxs[0] + idxs[-1]) / 1000.0
                    ax.text(mid, y_pos, short, ha="center", va="bottom", fontsize=7.0, fontweight="bold", color="#222222")
            ax.set_xlim(0, seq_len / 1000.0)
            ax.set_ylim(0, 118)
            ax.set_xlabel("Human mtDNA position rotated to tRNA-Phe = 0 (kb)")
            ax.set_ylabel("SQUINT column scale share (%)")
            ax.set_title("D. SQUINT Multi-Scale Segmentation Along Vertebrate mtDNA (C2B)")
            ax.legend(loc="lower left", framealpha=0.92)
            ax.grid(axis="x", linestyle=":", alpha=0.4)

    fig.suptitle(
        "SQUINT Difficult Whole-Genome MSA Benchmarks (Full HIV/Lentiviruses & Vertebrate Mitogenomes)",
        fontsize=12.0,
        fontweight="bold",
        y=0.98,
    )
    plt.savefig(OUT_PLOT, bbox_inches="tight", dpi=200)
    if ARTIFACT_DIR is not None:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(ARTIFACT_DIR / "difficult_genomes_plot.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def main() -> None:
    if "--replot" in sys.argv:
        summary = json.loads(OUT_JSON.read_text())
        ds_c1b = build_c1b_dataset()
        ds_c2b = build_c2_dataset(rotate_to_trnf=True)
        lg_c1b = json.loads((WORK_DIR / "C1B_Lentivirus_HIV1_HIV2_SIV_18.ledger.json").read_text())
        lg_c2b = json.loads((WORK_DIR / "C2B_Vertebrate_mtDNA_Canonical_trnF_24.ledger.json").read_text())
        build_figure(summary["rows"], lg_c1b, lg_c2b, ds_c1b, ds_c2b)
        print("Replot complete.")
        return

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Building & Validating Difficult Whole-Genome Cohorts ===", flush=True)
    ds_c1a = build_c1a_dataset()
    ds_c1b = build_c1b_dataset()
    ds_c2a = build_c2_dataset(rotate_to_trnf=False)
    ds_c2b = build_c2_dataset(rotate_to_trnf=True)
    ds_c3 = build_c3_dataset(ds_c1b, ds_c2b)
    assert_fast_hist_matches_bruteforce(ds_c2b)
    print("Startup GenBank feature/phase/rotation & O(N*L) scorer assertions PASSED.", flush=True)

    all_tools = ["squint_fa", "squint_hdb", "mafft_auto", "mafft_linsi", "muscle", "clustalo"]
    all_rows: list[dict[str, Any]] = []
    ledgers: dict[str, Any] = {}

    for ds, fn in [
        (ds_c1a, score_c1a),
        (ds_c1b, score_c1b),
        (ds_c2a, score_c2),
        (ds_c2b, score_c2),
        (ds_c3, score_c3),
    ]:
        print(f"\n--- Running {ds['dataset']} (N={len(ds['orig_seqs'])}) ---", flush=True)
        rows, ledger = run_all_tools_on_dataset(ds, fn, all_tools)
        all_rows.extend(rows)
        if ledger is not None:
            ledgers[ds["dataset"]] = ledger

    print("\n--- Verifying `squint subset --region` on .hdb archives ---", flush=True)
    subset_checks = run_region_subset_checks()

    # Write flat CSV
    fieldnames: list[str] = []
    seen_fields: set[str] = set()
    for r in all_rows:
        for k in r.keys():
            if k not in seen_fields:
                seen_fields.add(k)
                fieldnames.append(k)
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    summary = {
        "rows": all_rows,
        "subset_checks": subset_checks,
        "horizon_bubble_summaries": {
            ds_name: {
                "num_bubbles": sum(len(c.get("horizon_bubbles", [])) for c in lg["components"]),
                "sample_bubbles": [
                    {
                        "step": hb["tree_merge_step"],
                        "clade_sizes": [hb["left_clade_size"], hb["right_clade_size"]],
                        "col_spans": [hb["left_col_span"], hb["right_col_span"]],
                        "class": hb["dominant_biophys_class"],
                        "gain_nats": round(hb["macro_biophys_gain_nats"], 2),
                    }
                    for hb in lg["components"][0].get("horizon_bubbles", [])[:8]
                ]
                if lg.get("components")
                else [],
                "net_cohort_gain_nats": round(lg["net_cohort_gain_nats"], 2),
                "compression_pct": round(100.0 * lg["net_cohort_gain_nats"] / max(1.0, lg["raw_null_cohort_nats"]), 2),
            }
            for ds_name, lg in ledgers.items()
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2))

    build_figure(
        all_rows,
        ledgers.get("C1B_Lentivirus_HIV1_HIV2_SIV_18"),
        ledgers.get("C2B_Vertebrate_mtDNA_Canonical_trnF_24"),
        ds_c1b,
        ds_c2b,
    )
    print(f"\nSaved CSV -> {OUT_CSV}\nSaved JSON -> {OUT_JSON}\nSaved Plot -> {OUT_PLOT}", flush=True)


if __name__ == "__main__":
    main()
