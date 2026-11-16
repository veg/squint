#!/usr/bin/env python3
"""Parametric simulator (IQ-TREE 2.4.0 AliSim / INDELible-mode) tau-sweep
phase-transition experiment across 5 generative regimes (132 total N=8 cohorts):

  - M1_Homog_AA_Poisson   : Equidistant 8-taxon star, Poisson AA + POW{1.7/25} indels (L=220)
  - M2_Homog_AA_LG        : Equidistant 8-taxon star, LG AA + POW{1.7/25} indels (L=220)
  - M3_Homog_DNA_JC       : Equidistant 8-taxon star, JC DNA + POW{1.7/25} indels (L=300)
  - M4_StructFold_AA_LG_C60: Matched-rate & matched-indel structured fold:
                             5 contiguous LG+C60 cores (130 AA, no indels, r=0.55) +
                             4 LG loops (90 AA, r=1.65, indel=0.04444, POW{1.7/12})
  - M5_HierTree_AA_Poisson: Balanced 8-taxon binary tree (((T1,T2),(T3,T4)),((T5,T6),(T7,T8))),
                             root-spanning divergence tau = 6*b, Poisson AA + POW{1.7/25} indels

Evaluates AliSim's omniscient ground-truth 'oracle' alongside SQUINT, MAFFT L-INS-i,
MUSCLE, and Clustal Omega on:
  1. Pair Precision / FDR_pair / Recall Q / Exact Column TC against the AliSim oracle
  2. True Oracle Identity vs Apparent Column Identity mirage
  3. Aligner-agnostic Universal Lossless Compression (Delta L_KT_nats over K=20 or K=4)

Outputs:
  - benchmarks/tau_phase_transition_results.csv
  - benchmarks/tau_phase_transition_summary.json
  - benchmarks/tau_phase_transition_plot.png (+ copy in artifact dir)
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import tarfile
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
SQUINT_BIN = REPO_ROOT / "target" / "release" / "squint"
IQTREE_BIN = shutil.which("iqtree2") or os.environ.get("IQTREE2_BIN", "iqtree2")
MAFFT_BIN = shutil.which("mafft") or os.environ.get("MAFFT_BIN", "mafft")
MUSCLE_BIN = shutil.which("muscle") or os.environ.get("MUSCLE_BIN", "muscle")
CLUSTALO_BIN = shutil.which("clustalo") or os.environ.get("CLUSTALO_BIN", "clustalo")

OUT_CSV = REPO_ROOT / "benchmarks" / "tau_phase_transition_results.csv"
OUT_JSON = REPO_ROOT / "benchmarks" / "tau_phase_transition_summary.json"
OUT_PLOT = REPO_ROOT / "benchmarks" / "tau_phase_transition_plot.png"
ARTIFACT_PLOT = (
    Path(os.environ["SQUINT_ARTIFACT_DIR"]) / "tau_phase_transition_plot.png"
    if os.environ.get("SQUINT_ARTIFACT_DIR")
    else None
)

TOOLS = ("oracle", "squint", "mafft", "muscle", "clustalo")
ALIGNERS = ("squint", "mafft", "muscle", "clustalo")
SEEDS = (101, 202, 303)

AA_ALPH = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {c: i for i, c in enumerate(AA_ALPH)}
DNA_ALPH = "ACGT"
DNA_IDX = {c: i for i, c in enumerate(DNA_ALPH)}

# Structured fold architecture matching M2's exact protein-wide length (220),
# mean substitution rate (1.000 * tau), and mean indel rate (0.030 * tau):
#   5 * 26 = 130 AA core at R_core = 0.55 (0 indels)
#   22 + 23 + 23 + 22 = 90 AA loops at R_loop = (220 - 130*0.55)/90 = 1.65
#   loop indel rate = (220 * 0.03) / (90 * 1.65) = 0.044444444...
STRUCT_ARCH: list[tuple[str, int]] = [
    ("core", 26),
    ("loop", 22),
    ("core", 26),
    ("loop", 23),
    ("core", 26),
    ("loop", 23),
    ("core", 26),
    ("loop", 22),
    ("core", 26),
]
R_CORE = 0.55
R_LOOP = (220.0 - 130.0 * R_CORE) / 90.0
INDEL_LOOP = 0.03 * (220.0 / 90.0) / R_LOOP

MODEL_SPECS: list[dict[str, Any]] = [
    {
        "model": "M1_Homog_AA_Poisson",
        "seqtype": "AA",
        "squint_mode": "protein",
        "taus": [0.2, 0.5, 0.8, 1.1, 1.5, 2.0, 2.6, 3.5, 4.8],
    },
    {
        "model": "M2_Homog_AA_LG",
        "seqtype": "AA",
        "squint_mode": "protein",
        "taus": [0.2, 0.6, 1.0, 1.5, 2.0, 2.6, 3.3, 4.2, 5.5],
    },
    {
        "model": "M3_Homog_DNA_JC",
        "seqtype": "DNA",
        "squint_mode": "dna",
        "taus": [0.15, 0.35, 0.55, 0.80, 1.05, 1.35, 1.70, 2.20, 3.00],
    },
    {
        "model": "M4_StructFold_AA_LG_C60",
        "seqtype": "AA",
        "squint_mode": "protein",
        "taus": [0.2, 0.6, 1.0, 1.5, 2.0, 2.6, 3.3, 4.2, 5.5],
    },
    {
        "model": "M5_HierTree_AA_Poisson",
        "seqtype": "AA",
        "squint_mode": "protein",
        "taus": [0.6, 1.2, 2.0, 3.0, 4.2, 5.7, 7.8, 10.8],
    },
]


def elias_rissanen_nats(n: int) -> float:
    if n == 0:
        return math.log(2.0)
    p = 0.5 * math.log(2.0) * (1.0 / math.log(n + 1.0) - 1.0 / math.log(n + 2.0))
    return -math.log(p)


def logsumexp(vals: list[float]) -> float:
    m = max(vals)
    return m + math.log(sum(math.exp(v - m) for v in vals))


def dm_ln_p(counts: list[int], k_dim: int, alpha0: float) -> float:
    n = sum(counts)
    if n == 0:
        return 0.0
    a = alpha0 / k_dim
    lp = math.lgamma(alpha0) - math.lgamma(n + alpha0)
    for c in counts:
        if c > 0:
            lp += math.lgamma(c + a) - math.lgamma(a)
    return lp


def col_universal_mix_ln_p(counts: list[int], k_dim: int) -> float:
    alphas = (0.20, 1.00, 5.00)
    ln_w = -math.log(len(alphas))
    return logsumexp([dm_ln_p(counts, k_dim, a0) + ln_w for a0 in alphas])


def row_markov_gap_nats(bits: list[int]) -> float:
    if not bits:
        return 0.0
    nats = math.log(2.0)
    cnt = [[0.5, 0.5], [0.5, 0.5]]
    prev = bits[0]
    for b in bits[1:]:
        tot = cnt[prev][0] + cnt[prev][1]
        nats += -math.log(cnt[prev][b] / tot)
        cnt[prev][b] += 1.0
        prev = b
    return nats


def partition_header_nats(n_seqs: int, comp_sizes: list[int]) -> float:
    c_num = len(comp_sizes)
    if n_seqs <= 1:
        return 0.0
    if n_seqs == 2:
        return math.log(2.0)
    if c_num == 1 or c_num == n_seqs:
        return math.log(3.0)
    return math.log(3.0) + math.log(n_seqs - 2) - dm_ln_p(comp_sizes, c_num, 0.5 * c_num)


def verify_kraft_identities() -> dict[str, float]:
    col_sum = 0.0
    for x in range(4**5):
        tmp = x
        cnt = [0, 0, 0, 0]
        for _ in range(5):
            cnt[tmp & 3] += 1
            tmp >>= 2
        col_sum += math.exp(col_universal_mix_ln_p(cnt, 4))

    row_sum = 0.0
    for x in range(256):
        bits = [(x >> k) & 1 for k in range(8)]
        row_sum += math.exp(-row_markov_gap_nats(bits))

    n_max = 200_000
    er_sum = sum(math.exp(-elias_rissanen_nats(k)) for k in range(n_max + 1)) + (
        0.5 * math.log(2.0) / math.log(n_max + 2.0)
    )

    assert abs(col_sum - 1.0) < 1e-12, col_sum
    assert abs(row_sum - 1.0) < 1e-12, row_sum
    assert abs(er_sum - 1.0) < 1e-12, er_sum
    return {
        "column_dm_mixture_kraft_sum_K4_n5": col_sum,
        "row_markov_kt_kraft_sum_W8": row_sum,
        "elias_rissanen_kraft_sum": er_sum,
    }


def parse_fasta(text: str) -> dict[str, str]:
    seqs: dict[str, str] = {}
    cur: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith(">"):
            if cur is not None:
                seqs[cur] = "".join(buf)
            cur = s[1:].split()[0]
            buf = []
        else:
            buf.append(s)
    if cur is not None:
        seqs[cur] = "".join(buf)
    return seqs


def score_kt_mdl(
    orig_seqs: dict[str, str],
    components: list[dict[str, str]],
    seqtype: str,
) -> dict[str, float]:
    tok_idx = AA_IDX if seqtype == "AA" else DNA_IDX
    k_dim = len(tok_idx)

    l0_nats = 0.0
    total_units = 0
    for seq in orig_seqs.values():
        s_clean = list(seq.upper())
        total_units += len(s_clean)
        cnt = [0] * k_dim
        for ch in s_clean:
            idx = tok_idx.get(ch)
            if idx is not None:
                cnt[idx] += 1
        l0_nats += elias_rissanen_nats(len(s_clean)) - dm_ln_p(cnt, k_dim, 0.5 * k_dim)

    n_seqs = len(orig_seqs)
    comp_nats = partition_header_nats(n_seqs, [len(c) for c in components])
    total_cols = 0

    for comp in components:
        m = len(comp)
        if m == 1:
            sid = next(iter(comp.keys()))
            raw = [ch for ch in comp[sid].upper() if ch not in ".-!"]
            cnt = [0] * k_dim
            for ch in raw:
                idx = tok_idx.get(ch)
                if idx is not None:
                    cnt[idx] += 1
            comp_nats += elias_rissanen_nats(len(raw)) - dm_ln_p(cnt, k_dim, 0.5 * k_dim)
            total_cols = max(total_cols, len(raw))
            continue

        sids = sorted(comp.keys())
        rows = {sid: list(comp[sid].upper()) for sid in sids}
        w_cols = len(rows[sids[0]])
        total_cols += w_cols
        comp_nats += elias_rissanen_nats(w_cols)

        for sid in sids:
            bits = [0 if ch in ".-!" else 1 for ch in rows[sid]]
            comp_nats += row_markov_gap_nats(bits)

        priv_streams: dict[str, list[str]] = {sid: [] for sid in sids}
        for j in range(w_cols):
            pres = [(sid, rows[sid][j]) for sid in sids if rows[sid][j] not in ".-!"]
            nj = len(pres)
            if nj == 1:
                priv_streams[pres[0][0]].append(pres[0][1])
            elif nj >= 2:
                cnt = [0] * k_dim
                for _, ch in pres:
                    idx = tok_idx.get(ch)
                    if idx is not None:
                        cnt[idx] += 1
                comp_nats += -col_universal_mix_ln_p(cnt, k_dim)

        for p_chars in priv_streams.values():
            if p_chars:
                cnt = [0] * k_dim
                for ch in p_chars:
                    idx = tok_idx.get(ch)
                    if idx is not None:
                        cnt[idx] += 1
                comp_nats += -dm_ln_p(cnt, k_dim, 0.5 * k_dim)

    delta_l = l0_nats - comp_nats
    return {
        "L0_nats": l0_nats,
        "L_msa_nats": comp_nats,
        "delta_L_KT_nats": delta_l,
        "total_units": float(total_units),
        "total_cols": float(total_cols),
    }


def extract_oracle_structures(
    ref_seqs: dict[str, str],
    order: list[str],
    core_col_mask: list[bool] | None,
) -> dict[str, Any]:
    rmap = {k: ref_seqs[k].upper() for k in order}
    lr = len(rmap[order[0]])
    pos = {k: 0 for k in order}
    ref_pairs: set[tuple[int, int, int, int]] = set()
    ref_cols: list[tuple[tuple[int, int], ...]] = []
    core_res: set[tuple[int, int]] = set()
    core_pairs: set[tuple[int, int, int, int]] = set()

    for c in range(lr):
        col: list[tuple[int, int]] = []
        is_c = True if core_col_mask is None else core_col_mask[c]
        for ti, k in enumerate(order):
            ch = rmap[k][c]
            if ch not in ".-":
                p = pos[k]
                pos[k] += 1
                col.append((ti, p))
                if is_c:
                    core_res.add((ti, p))
        if len(col) >= 2:
            ref_cols.append(tuple(col))
            for a in range(len(col)):
                ta, pa = col[a]
                for b in range(a + 1, len(col)):
                    tb, pb = col[b]
                    ref_pairs.add((ta, pa, tb, pb))
                    if is_c:
                        core_pairs.add((ta, pa, tb, pb))

    return {
        "order": order,
        "k_to_ti": {k: ti for ti, k in enumerate(order)},
        "ref_pairs": ref_pairs,
        "ref_cols": set(ref_cols),
        "core_res": core_res,
        "core_pairs": core_pairs,
    }


def evaluate_alignment(
    oracle_info: dict[str, Any],
    comps: list[dict[str, str]],
) -> dict[str, Any]:
    k_to_ti = oracle_info["k_to_ti"]
    ref_pairs = oracle_info["ref_pairs"]
    ref_cols = oracle_info["ref_cols"]
    core_res = oracle_info["core_res"]
    core_pairs = oracle_info["core_pairs"]

    test_pairs: set[tuple[int, int, int, int]] = set()
    test_core_pairs: set[tuple[int, int, int, int]] = set()
    test_cols: set[tuple[tuple[int, int], ...]] = set()
    ident_p = 0
    cross_root_emitted = 0
    cross_root_fp = 0

    for comp in comps:
        if len(comp) < 2:
            continue
        ckeys = list(comp.keys())
        lt = len(comp[ckeys[0]])
        tpos = {k: 0 for k in ckeys}
        for c in range(lt):
            items: list[tuple[int, int, str]] = []
            for k in ckeys:
                ch = comp[k][c].upper()
                if ch not in "-.!":
                    items.append((k_to_ti[k], tpos[k], ch))
                    tpos[k] += 1
            if len(items) < 2:
                continue
            items.sort(key=lambda x: x[0])
            col_key = tuple((ti, p) for ti, p, _ in items)
            test_cols.add(col_key)
            for a in range(len(items)):
                ta, pa, cha = items[a]
                for b in range(a + 1, len(items)):
                    tb, pb, chb = items[b]
                    pair = (ta, pa, tb, pb)
                    test_pairs.add(pair)
                    if cha == chb:
                        ident_p += 1
                    if pair in core_res or (
                        (ta, pa) in core_res and (tb, pb) in core_res
                    ):
                        test_core_pairs.add(pair)
                    # Cross-root clade check for M5: {0,1,2,3} vs {4,5,6,7}
                    if (ta < 4) != (tb < 4):
                        cross_root_emitted += 1
                        if pair not in ref_pairs:
                            cross_root_fp += 1

    emitted_n = len(test_pairs)
    oracle_n = len(ref_pairs)
    tp = len(ref_pairs & test_pairs)
    fp = emitted_n - tp

    prec = tp / emitted_n if emitted_n > 0 else 1.0
    fdr = fp / emitted_n if emitted_n > 0 else 0.0
    rec = tp / oracle_n if oracle_n > 0 else 1.0
    f1 = 2.0 * tp / (emitted_n + oracle_n) if (emitted_n + oracle_n) > 0 else 1.0
    tc = len(ref_cols & test_cols) / max(1, len(ref_cols))
    app_id = (100.0 * ident_p / emitted_n) if emitted_n > 0 else None

    core_emitted = len(test_core_pairs)
    core_tp = len(core_pairs & test_core_pairs)
    core_prec = core_tp / core_emitted if core_emitted > 0 else 1.0
    core_rec = core_tp / max(1, len(core_pairs))

    return {
        "emitted_pairs": emitted_n,
        "oracle_pairs": oracle_n,
        "tp_pairs": tp,
        "fp_pairs": fp,
        "Pair_Prec": prec,
        "FDR_pair": fdr,
        "Pair_Recall_Q": rec,
        "Pair_F1": f1,
        "Exact_Col_TC": tc,
        "Apparent_Pct_ID": app_id,
        "Core_Pair_Prec": core_prec,
        "Core_Pair_Recall_Q": core_rec,
        "CrossRoot_Emitted_Pairs": cross_root_emitted,
        "CrossRoot_FP_Pairs": cross_root_fp,
    }


def simulate_cohort(
    model: str, tau: float, seed: int, tmp: Path
) -> tuple[dict[str, str], dict[str, str], list[str], list[bool] | None]:
    order = [f"T{i}" for i in range(1, 9)]

    if model in ("M1_Homog_AA_Poisson", "M2_Homog_AA_LG", "M3_Homog_DNA_JC"):
        bl = tau / 2.0
        nwk = tmp / "star.nwk"
        nwk.write_text("(" + ",".join(f"{t}:{bl:.6f}" for t in order) + ");\n")
        pref = tmp / "sim"
        if model == "M1_Homog_AA_Poisson":
            sqt, mod, length = "AA", "Poisson", 220
        elif model == "M2_Homog_AA_LG":
            sqt, mod, length = "AA", "LG", 220
        else:
            sqt, mod, length = "DNA", "JC", 300
        subprocess.run(
            [
                IQTREE_BIN,
                "--alisim",
                str(pref),
                "--seqtype",
                sqt,
                "-m",
                mod,
                "-t",
                str(nwk),
                "--length",
                str(length),
                "--indel",
                "0.03,0.03",
                "--indel-size",
                "POW{1.7/25},POW{1.7/25}",
                "--seed",
                str(seed),
                "-af",
                "fasta",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        ref = parse_fasta((tmp / "sim.fa").read_text())
        orig = {k: v.replace("-", "").upper() for k, v in ref.items()}
        return ref, orig, order, None

    if model == "M4_StructFold_AA_LG_C60":
        bl_core = R_CORE * (tau / 2.0)
        bl_loop = R_LOOP * (tau / 2.0)
        nwk_c = tmp / "c.nwk"
        nwk_l = tmp / "l.nwk"
        nwk_c.write_text("(" + ",".join(f"{t}:{bl_core:.6f}" for t in order) + ");\n")
        nwk_l.write_text("(" + ",".join(f"{t}:{bl_loop:.6f}" for t in order) + ");\n")

        acc: dict[str, list[str]] = {t: [] for t in order}
        core_mask: list[bool] = []
        for b_idx, (kind, blen) in enumerate(STRUCT_ARCH):
            pref = tmp / f"blk_{b_idx}"
            if kind == "core":
                cmd = [
                    IQTREE_BIN,
                    "--alisim",
                    str(pref),
                    "--seqtype",
                    "AA",
                    "-m",
                    "LG+C60",
                    "-t",
                    str(nwk_c),
                    "--length",
                    str(blen),
                    "--seed",
                    str(seed * 100 + b_idx),
                    "-af",
                    "fasta",
                ]
            else:
                cmd = [
                    IQTREE_BIN,
                    "--alisim",
                    str(pref),
                    "--seqtype",
                    "AA",
                    "-m",
                    "LG",
                    "-t",
                    str(nwk_l),
                    "--length",
                    str(blen),
                    "--indel",
                    f"{INDEL_LOOP:.5f},{INDEL_LOOP:.5f}",
                    "--indel-size",
                    "POW{1.7/12},POW{1.7/12}",
                    "--seed",
                    str(seed * 100 + b_idx),
                    "-af",
                    "fasta",
                ]
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            blk_map = parse_fasta((tmp / f"blk_{b_idx}.fa").read_text())
            w_blk = len(blk_map[order[0]])
            core_mask.extend([kind == "core"] * w_blk)
            for t in order:
                acc[t].append(blk_map[t])

        ref = {t: "".join(acc[t]) for t in order}
        orig = {t: v.replace("-", "").upper() for t, v in ref.items()}
        return ref, orig, order, core_mask

    if model == "M5_HierTree_AA_Poisson":
        b = tau / 6.0
        nwk = tmp / "hier.nwk"
        nwk.write_text(
            f"(((T1:{b:.6f},T2:{b:.6f}):{b:.6f},(T3:{b:.6f},T4:{b:.6f}):{b:.6f}):{b:.6f},"
            f"((T5:{b:.6f},T6:{b:.6f}):{b:.6f},(T7:{b:.6f},T8:{b:.6f}):{b:.6f}):{b:.6f});\n"
        )
        pref = tmp / "hsim"
        subprocess.run(
            [
                IQTREE_BIN,
                "--alisim",
                str(pref),
                "--seqtype",
                "AA",
                "-m",
                "Poisson",
                "-t",
                str(nwk),
                "--length",
                "220",
                "--indel",
                "0.02,0.02",
                "--indel-size",
                "POW{1.7/25},POW{1.7/25}",
                "--seed",
                str(seed),
                "-af",
                "fasta",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        ref = parse_fasta((tmp / "hsim.fa").read_text())
        orig = {k: v.replace("-", "").upper() for k, v in ref.items()}
        return ref, orig, order, None

    raise ValueError(f"Unknown model: {model}")


def run_aligner(
    tool: str,
    squint_mode: str,
    ug_path: Path,
    tmp: Path,
) -> tuple[list[dict[str, str]], float, dict[str, Any]]:
    meta = {"num_horizon_bubbles": 0, "mean_macro_biophys_gain_nats": None}
    t0 = time.perf_counter()

    if tool == "squint":
        sq_js = tmp / "sq_ledger.json"
        subprocess.run(
            [str(SQUINT_BIN), str(ug_path), "-j", str(sq_js), "--mode", squint_mode],
            capture_output=True,
            text=True,
            check=True,
        )
        rt = time.perf_counter() - t0
        led = json.loads(sq_js.read_text())
        comps = [{sid: aln for sid, aln in c["aligned_fasta"]} for c in led["components"]]
        hb_gains = [
            float(hb["macro_biophys_gain_nats"])
            for c in led["components"]
            for hb in c.get("horizon_bubbles", [])
        ]
        meta["num_horizon_bubbles"] = len(hb_gains)
        if hb_gains:
            meta["mean_macro_biophys_gain_nats"] = statistics.mean(hb_gains)
        return comps, rt, meta

    if tool == "mafft":
        proc = subprocess.run(
            [MAFFT_BIN, "--localpair", "--maxiterate", "1000", "--quiet", str(ug_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        return [parse_fasta(proc.stdout)], time.perf_counter() - t0, meta

    if tool == "muscle":
        out_fa = tmp / "mu.fa"
        subprocess.run(
            [MUSCLE_BIN, "-in", str(ug_path), "-out", str(out_fa), "-quiet"],
            capture_output=True,
            text=True,
            check=True,
        )
        return [parse_fasta(out_fa.read_text())], time.perf_counter() - t0, meta

    if tool == "clustalo":
        out_fa = tmp / "cl.fa"
        subprocess.run(
            [CLUSTALO_BIN, "-i", str(ug_path), "-o", str(out_fa), "--threads=1", "--force"],
            capture_output=True,
            text=True,
            check=True,
        )
        return [parse_fasta(out_fa.read_text())], time.perf_counter() - t0, meta

    raise ValueError(tool)


def agg_stats(vals: list[float | None]) -> dict[str, float | int | None]:
    clean = [float(v) for v in vals if v is not None]
    if not clean:
        return {"mean": None, "sd": None, "n": 0}
    return {
        "mean": statistics.mean(clean),
        "sd": statistics.stdev(clean) if len(clean) >= 2 else 0.0,
        "n": len(clean),
    }


def make_summary(rows: list[dict[str, Any]], kraft_audit: dict[str, float]) -> dict[str, Any]:
    grouped: dict[tuple[str, float, str], list[dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault((r["model"], r["tau"], r["tool"]), []).append(r)

    by_model: dict[str, list[dict[str, Any]]] = {}
    for spec in MODEL_SPECS:
        mod = spec["model"]
        pts: list[dict[str, Any]] = []
        for tau in spec["taus"]:
            pt: dict[str, Any] = {"tau": tau, "tools": {}}
            for tool in TOOLS:
                grp = grouped[(mod, tau, tool)]
                pt["tools"][tool] = {
                    "num_components": agg_stats([x["num_components"] for x in grp]),
                    "Pair_Prec": agg_stats([x["Pair_Prec"] for x in grp]),
                    "FDR_pair": agg_stats([x["FDR_pair"] for x in grp]),
                    "Pair_Recall_Q": agg_stats([x["Pair_Recall_Q"] for x in grp]),
                    "Pair_F1": agg_stats([x["Pair_F1"] for x in grp]),
                    "Exact_Col_TC": agg_stats([x["Exact_Col_TC"] for x in grp]),
                    "Apparent_Pct_ID": agg_stats([x["Apparent_Pct_ID"] for x in grp]),
                    "delta_L_KT_nats": agg_stats([x["delta_L_KT_nats"] for x in grp]),
                    "Core_Pair_Prec": agg_stats([x["Core_Pair_Prec"] for x in grp]),
                    "Core_Pair_Recall_Q": agg_stats([x["Core_Pair_Recall_Q"] for x in grp]),
                    "CrossRoot_Emitted_Pairs": agg_stats(
                        [x["CrossRoot_Emitted_Pairs"] for x in grp]
                    ),
                    "CrossRoot_FP_Pairs": agg_stats([x["CrossRoot_FP_Pairs"] for x in grp]),
                    "num_horizon_bubbles": agg_stats([x["num_horizon_bubbles"] for x in grp]),
                    "runtime_s": agg_stats([x["runtime_s"] for x in grp]),
                }
            pts.append(pt)
        by_model[mod] = pts

    return {
        "kraft_self_check": kraft_audit,
        "seeds": list(SEEDS),
        "n_cohorts_total": sum(len(s["taus"]) * len(SEEDS) for s in MODEL_SPECS),
        "by_model": by_model,
    }


def generate_plot(summary: dict[str, Any]) -> None:
    by_mod = summary["by_model"]
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.6), dpi=180)
    plt.rcParams.update({"font.size": 9.5})

    def series(mod: str, tool: str, metric: str) -> tuple[list[float], list[float], list[float]]:
        taus, ms, sds = [], [], []
        for pt in by_mod[mod]:
            st = pt["tools"][tool][metric]
            if st["mean"] is not None:
                taus.append(pt["tau"])
                ms.append(st["mean"])
                sds.append(st["sd"] or 0.0)
        return taus, ms, sds

    # Panel A: M2 Homog LG vs M4 Structured LG+C60
    ax = axes[0, 0]
    for mod, lbl_pref, c_sq, c_mf in [
        ("M4_StructFold_AA_LG_C60", "Structured LG+C60", "#0f766e", "#2563eb"),
        ("M2_Homog_AA_LG", "Homog LG", "#dc2626", "#f97316"),
    ]:
        t, m, s = series(mod, "squint", "Pair_Prec")
        ax.plot(t, m, "-", marker="o", color=c_sq, lw=2.1, ms=4.5, label=f"SQUINT {lbl_pref} Prec")
        ax.fill_between(
            t,
            [max(0.0, a - b) for a, b in zip(m, s)],
            [min(1.02, a + b) for a, b in zip(m, s)],
            color=c_sq,
            alpha=0.12,
        )
        t_q, m_q, _ = series(mod, "squint", "Pair_Recall_Q")
        ax.plot(t_q, m_q, "--", marker="o", color=c_sq, lw=1.4, ms=3.5, alpha=0.8)

        tm, mm, sm = series(mod, "mafft", "Pair_Prec")
        ax.plot(tm, mm, "-", marker="s", color=c_mf, lw=2.0, ms=4.2, label=f"MAFFT {lbl_pref} Prec")
        ax.fill_between(
            tm,
            [max(0.0, a - b) for a, b in zip(mm, sm)],
            [min(1.02, a + b) for a, b in zip(mm, sm)],
            color=c_mf,
            alpha=0.12,
        )
    ax.set_title("A. Homogeneous LG Cliff vs Matched-Rate Structured LG+C60", fontweight="bold")
    ax.set_xlabel(r"Pairwise Branch Divergence $\tau$ (mean subst/site)")
    ax.set_ylabel("All-Residue Pair Precision (solid) / Recall Q (dashed)")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7.8, loc="lower left")

    # Panel B: Apparent Identity Mirage vs True Identity (M1 Poisson AA & M3 JC DNA)
    ax = axes[0, 1]
    t1_o, m1_o, _ = series("M1_Homog_AA_Poisson", "oracle", "Apparent_Pct_ID")
    t1_m, m1_m, _ = series("M1_Homog_AA_Poisson", "mafft", "Apparent_Pct_ID")
    t1_p, m1_p, _ = series("M1_Homog_AA_Poisson", "mafft", "Pair_Prec")
    t3_o, m3_o, _ = series("M3_Homog_DNA_JC", "oracle", "Apparent_Pct_ID")
    t3_m, m3_m, _ = series("M3_Homog_DNA_JC", "mafft", "Apparent_Pct_ID")
    t3_p, m3_p, _ = series("M3_Homog_DNA_JC", "mafft", "Pair_Prec")

    ax.plot(t1_o, m1_o, ":", marker=".", color="#1e293b", lw=1.8, label="AA Poisson True Oracle % ID")
    ax.plot(t1_m, m1_m, "-", marker="s", color="#2563eb", lw=2.0, ms=4.2, label="AA Poisson MAFFT Apparent % ID")
    ax.plot(t1_p, [100.0 * v for v in m1_p], "--", color="#2563eb", lw=1.4, label="AA MAFFT True Precision x100")

    ax.plot(t3_o, m3_o, ":", marker=".", color="#7c2d12", lw=1.8, label="DNA JC True Oracle % ID")
    ax.plot(t3_m, m3_m, "-", marker="^", color="#d97706", lw=2.0, ms=4.5, label="DNA JC MAFFT Apparent % ID")
    ax.plot(t3_p, [100.0 * v for v in m3_p], "--", color="#d97706", lw=1.4, label="DNA MAFFT True Precision x100")

    ax.set_title("B. Gap-Packing Twilight Mirage (Apparent % ID vs True Precision)", fontweight="bold")
    ax.set_xlabel(r"Pairwise Branch Divergence $\tau$ (subst/site)")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(-2, 102)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7.3, loc="upper right")

    # Panel C: Information-Theoretic Compression Transition on M2_Homog_AA_LG
    ax = axes[1, 0]
    for tool, col, mk, lbl in [
        ("oracle", "#111827", "o", r"AliSim Oracle $\Delta L_{\mathrm{KT}}$"),
        ("squint", "#0f766e", "s", r"SQUINT $\Delta L_{\mathrm{KT}}$"),
        ("mafft", "#f97316", "^", r"MAFFT $\Delta L_{\mathrm{KT}}$"),
    ]:
        t, m, _ = series("M2_Homog_AA_LG", tool, "delta_L_KT_nats")
        ax.plot(t, m, "-", marker=mk, color=col, lw=2.0, ms=4.5, label=lbl)
    ax.axhline(0.0, color="#991b1b", ls="--", lw=1.2, alpha=0.8)
    ax.set_title(r"C. Information-Theoretic Cliff on Homogeneous LG ($\Delta L_{\mathrm{KT}}$ vs $K$)", fontweight="bold")
    ax.set_xlabel(r"Pairwise Branch Divergence $\tau$ (subst/site)")
    ax.set_ylabel(r"Universal Lossless Gain $\Delta L_{\mathrm{KT}}$ (nats)")
    ax.grid(True, alpha=0.25)

    ax_r = ax.twinx()
    tk, mk, sk = series("M2_Homog_AA_LG", "squint", "num_components")
    ax_r.plot(tk, mk, "-.", marker="D", color="#6d28d9", lw=2.0, ms=4.5, label="SQUINT Components K")
    ax_r.set_ylabel("SQUINT Components K (1 -> 8 singletons)", color="#6d28d9")
    ax_r.set_ylim(0.5, 8.6)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7.6, loc="center right")

    # Panel D: M5 Balanced Binary Tree Scale-by-Scale Fission
    ax = axes[1, 1]
    tk5, mk5, _ = series("M5_HierTree_AA_Poisson", "squint", "num_components")
    ax.plot(tk5, mk5, "-", marker="D", color="#6d28d9", lw=2.2, ms=5.0, label="SQUINT Components K (1->2->4->8)")
    ax.set_yticks([1, 2, 4, 6, 8])
    ax.set_ylim(0.5, 8.6)
    ax.set_xlabel(r"Root-Spanning Divergence $\tau = 6b$ (cousin=$2\tau/3$, cherry=$\tau/3$)")
    ax.set_ylabel("SQUINT Components K", color="#6d28d9")
    ax.set_title("D. Hierarchical Tree Scale Fission & Pair Precision", fontweight="bold")
    ax.grid(True, alpha=0.25)

    ax5_r = ax.twinx()
    for tool, col, ls, lbl in [
        ("squint", "#0f766e", "-", "SQUINT Prec"),
        ("mafft", "#2563eb", "--", "MAFFT Prec"),
        ("muscle", "#dc2626", ":", "MUSCLE Prec"),
        ("clustalo", "#d97706", "-.", "ClustalO Prec"),
    ]:
        tp, mp, _ = series("M5_HierTree_AA_Poisson", tool, "Pair_Prec")
        ax5_r.plot(tp, mp, ls, color=col, lw=1.9, label=lbl)
    ax5_r.set_ylabel("Pair Precision vs Oracle")
    ax5_r.set_ylim(-0.03, 1.05)
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax5_r.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=7.5, loc="center right")

    plt.tight_layout()
    fig.savefig(OUT_PLOT, dpi=180)
    plt.close(fig)
    if ARTIFACT_PLOT is not None:
        ARTIFACT_PLOT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(OUT_PLOT, ARTIFACT_PLOT)


def main() -> int:
    t0_all = time.perf_counter()
    kraft_audit = verify_kraft_identities()
    total_cohorts = sum(len(s["taus"]) * len(SEEDS) for s in MODEL_SPECS)
    rows: list[dict[str, Any]] = []
    idx = 0

    for spec in MODEL_SPECS:
        mod = spec["model"]
        sqt = spec["seqtype"]
        smode = spec["squint_mode"]
        for tau in spec["taus"]:
            for seed in SEEDS:
                idx += 1
                with tempfile.TemporaryDirectory(prefix=f"tau_{mod}_{tau}_{seed}_") as td:
                    tmp = Path(td)
                    ref_seqs, orig_seqs, order, core_mask = simulate_cohort(mod, tau, seed, tmp)
                    oracle_info = extract_oracle_structures(ref_seqs, order, core_mask)
                    ug_path = tmp / "unaligned.fa"
                    ug_path.write_text("".join(f">{k}\n{orig_seqs[k]}\n" for k in order))

                    # 1. Score Oracle tape under universal KT(1/2) lossless codec
                    ev_o = evaluate_alignment(oracle_info, [ref_seqs])
                    mdl_o = score_kt_mdl(orig_seqs, [ref_seqs], sqt)
                    rows.append(
                        {
                            "model": mod,
                            "seqtype": sqt,
                            "tau": tau,
                            "seed": seed,
                            "tool": "oracle",
                            "runtime_s": 0.0,
                            "num_components": 1,
                            "total_cols": int(mdl_o["total_cols"]),
                            "emitted_pairs": ev_o["emitted_pairs"],
                            "oracle_pairs": ev_o["oracle_pairs"],
                            "tp_pairs": ev_o["tp_pairs"],
                            "fp_pairs": 0,
                            "Pair_Prec": 1.0,
                            "FDR_pair": 0.0,
                            "Pair_Recall_Q": 1.0,
                            "Pair_F1": 1.0,
                            "Exact_Col_TC": 1.0,
                            "Apparent_Pct_ID": ev_o["Apparent_Pct_ID"],
                            "Core_Pair_Prec": 1.0,
                            "Core_Pair_Recall_Q": 1.0,
                            "CrossRoot_Emitted_Pairs": ev_o["CrossRoot_Emitted_Pairs"],
                            "CrossRoot_FP_Pairs": 0,
                            "L0_nats": round(mdl_o["L0_nats"], 3),
                            "L_msa_nats": round(mdl_o["L_msa_nats"], 3),
                            "delta_L_KT_nats": round(mdl_o["delta_L_KT_nats"], 3),
                            "num_horizon_bubbles": 0,
                            "mean_macro_biophys_gain_nats": None,
                        }
                    )

                    sq_row = None
                    mf_row = None
                    for tool in ALIGNERS:
                        comps, rt, meta = run_aligner(tool, smode, ug_path, tmp)
                        ev = evaluate_alignment(oracle_info, comps)
                        mdl = score_kt_mdl(orig_seqs, comps, sqt)
                        row = {
                            "model": mod,
                            "seqtype": sqt,
                            "tau": tau,
                            "seed": seed,
                            "tool": tool,
                            "runtime_s": round(rt, 4),
                            "num_components": len(comps),
                            "total_cols": int(mdl["total_cols"]),
                            "emitted_pairs": ev["emitted_pairs"],
                            "oracle_pairs": ev["oracle_pairs"],
                            "tp_pairs": ev["tp_pairs"],
                            "fp_pairs": ev["fp_pairs"],
                            "Pair_Prec": round(ev["Pair_Prec"], 6),
                            "FDR_pair": round(ev["FDR_pair"], 6),
                            "Pair_Recall_Q": round(ev["Pair_Recall_Q"], 6),
                            "Pair_F1": round(ev["Pair_F1"], 6),
                            "Exact_Col_TC": round(ev["Exact_Col_TC"], 6),
                            "Apparent_Pct_ID": (
                                round(ev["Apparent_Pct_ID"], 4)
                                if ev["Apparent_Pct_ID"] is not None
                                else None
                            ),
                            "Core_Pair_Prec": round(ev["Core_Pair_Prec"], 6),
                            "Core_Pair_Recall_Q": round(ev["Core_Pair_Recall_Q"], 6),
                            "CrossRoot_Emitted_Pairs": ev["CrossRoot_Emitted_Pairs"],
                            "CrossRoot_FP_Pairs": ev["CrossRoot_FP_Pairs"],
                            "L0_nats": round(mdl["L0_nats"], 3),
                            "L_msa_nats": round(mdl["L_msa_nats"], 3),
                            "delta_L_KT_nats": round(mdl["delta_L_KT_nats"], 3),
                            "num_horizon_bubbles": meta["num_horizon_bubbles"],
                            "mean_macro_biophys_gain_nats": (
                                round(meta["mean_macro_biophys_gain_nats"], 3)
                                if meta["mean_macro_biophys_gain_nats"] is not None
                                else None
                            ),
                        }
                        rows.append(row)
                        if tool == "squint":
                            sq_row = row
                        elif tool == "mafft":
                            mf_row = row

                    true_id_s = (
                        f"{ev_o['Apparent_Pct_ID']:4.1f}%"
                        if ev_o["Apparent_Pct_ID"] is not None
                        else " n/a "
                    )
                    mf_id_s = (
                        f"{mf_row['Apparent_Pct_ID']:4.1f}%"
                        if mf_row["Apparent_Pct_ID"] is not None
                        else " n/a "
                    )
                    print(
                        f"[{idx:03d}/{total_cohorts}] {mod:25s} tau={tau:4.2f} s={seed} "
                        f"| Oracle: ID={true_id_s} dL={mdl_o['delta_L_KT_nats']:+7.1f} "
                        f"| SQ: K={sq_row['num_components']} Prec={sq_row['Pair_Prec']:.3f} Q={sq_row['Pair_Recall_Q']:.3f} dL={sq_row['delta_L_KT_nats']:+6.1f} "
                        f"| MF: Prec={mf_row['Pair_Prec']:.3f} Q={mf_row['Pair_Recall_Q']:.3f} appID={mf_id_s} dL={mf_row['delta_L_KT_nats']:+7.1f}",
                        flush=True,
                    )

    fieldnames = list(rows[0].keys())
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    summary = make_summary(rows, kraft_audit)
    OUT_JSON.write_text(json.dumps(summary, indent=2))
    generate_plot(summary)

    print(f"\nCompleted {total_cohorts} AliSim cohorts in {time.perf_counter() - t0_all:.1f}s")
    print(f"Wrote {OUT_CSV}, {OUT_JSON}, and {OUT_PLOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
