#!/usr/bin/env python3
"""Parameter-free, non-circular MSA benchmark evaluating:
  Axis 1: Epistemic calibration & decoy specificity via column-free FDR_pair,
          SeqPair_Recall, and SeqPair_FPR across zero-homology, entrapment, and multi-family cohorts.
  Axis 2: Aligner-agnostic, Kraft-tight Universal Lossless MDL compression (Delta L_KT)
          across Protein (K=20) and Codon (K=61 / K=64) alphabets with zero empirical matrices.
  Axis 3 (qualitative/structural anatomy): Top hallucinated 'conserved' motif blocks forced
          by MAFFT L-INS-i, MUSCLE, and Clustal Omega on zero-homology unrelated proteins.

Outputs:
  - benchmarks/noncircular_msa_results.csv
  - benchmarks/noncircular_msa_summary.json
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
HYPHY_DATA = Path(
    os.environ.get("HYPHY_DATA_DIR", str(REPO_ROOT / "benchmarks" / "external_data" / "hyphy_data"))
)
ARCHIVE_PATH = REPO_ROOT / "benchmarks" / "external_data" / "bench.tar.gz"
SQUINT_BIN = REPO_ROOT / "target" / "release" / "squint"
OUT_CSV = REPO_ROOT / "benchmarks" / "noncircular_msa_results.csv"
OUT_JSON = REPO_ROOT / "benchmarks" / "noncircular_msa_summary.json"

AA_ALPH = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {c: i for i, c in enumerate(AA_ALPH)}

# Standard 6 biophysical groups matching non-parametric physicochemical partition
AA_GROUPS = {
    "A": 0, "V": 0, "I": 0, "L": 0, "M": 0,  # Hydrophobic (aliphatic)
    "F": 1, "Y": 1, "W": 1,                    # Aromatic
    "K": 2, "R": 2, "H": 2,                    # Basic (+)
    "D": 3, "E": 3,                            # Acidic (-)
    "S": 4, "T": 4, "N": 4, "Q": 4,            # Polar uncharged
    "G": 5, "P": 5, "C": 5,                    # Special
}

GC = {
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
SENSE_CODONS = sorted([c for c, aa in GC.items() if aa != "*"])
SENSE_IDX = {c: i for i, c in enumerate(SENSE_CODONS)}
ALL_CODONS = sorted(GC.keys())
ALL_CODON_IDX = {c: i for i, c in enumerate(ALL_CODONS)}

TOOLS = ("squint", "mafft", "muscle", "clustalo")


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
    """Exact 3-scale symmetric Dirichlet-Multinomial mixture with uniform base 1/K."""
    alphas = (0.20, 1.00, 5.00)
    ln_w = -math.log(len(alphas))
    return logsumexp([dm_ln_p(counts, k_dim, a0) + ln_w for a0 in alphas])


def row_markov_gap_nats(bits: list[int]) -> float:
    """Exact causal 2-state Krichevsky-Trofimov code for gap/presence binary rows."""
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


def verify_kraft_identities() -> dict[str, float]:
    # 1. Column universal mixture: sum over all K^n columns (K=4, n=5 => 1024 strings)
    col_sum = 0.0
    for x in range(4**5):
        tmp = x
        cnt = [0, 0, 0, 0]
        for _ in range(5):
            cnt[tmp & 3] += 1
            tmp >>= 2
        col_sum += math.exp(col_universal_mix_ln_p(cnt, 4))

    # 2. Row Markov gap code: sum over all 2^8 = 256 binary strings
    row_sum = 0.0
    for x in range(256):
        bits = [(x >> k) & 1 for k in range(8)]
        row_sum += math.exp(-row_markov_gap_nats(bits))

    # 3. Elias-Rissanen prefix code + exact analytic tail
    n_max = 200_000
    er_partial = sum(math.exp(-elias_rissanen_nats(k)) for k in range(n_max + 1))
    er_tail = 0.5 * math.log(2.0) / math.log(n_max + 2.0)
    er_sum = er_partial + er_tail

    assert abs(col_sum - 1.0) < 1e-12, f"col_sum={col_sum}"
    assert abs(row_sum - 1.0) < 1e-12, f"row_sum={row_sum}"
    assert abs(er_sum - 1.0) < 1e-12, f"er_sum={er_sum}"
    return {
        "column_dm_mixture_kraft_sum_K4_n5": col_sum,
        "row_markov_kt_kraft_sum_W8": row_sum,
        "elias_rissanen_kraft_sum": er_sum,
    }


def partition_header_nats(n_seqs: int, comp_sizes: list[int]) -> float:
    """Exact Kraft-valid prior over sequence partition into C components."""
    c_num = len(comp_sizes)
    if n_seqs <= 1:
        return 0.0
    if n_seqs == 2:
        return math.log(2.0)
    if c_num == 1 or c_num == n_seqs:
        return math.log(3.0)
    return math.log(3.0) + math.log(n_seqs - 2) - dm_ln_p(comp_sizes, c_num, 0.5 * c_num)


def tokenize_seq(seq: str, mode: str) -> list[str]:
    s = seq.upper().replace("U", "T")
    if mode == "protein":
        return [ch for ch in s if ch not in "-."]
    assert len(s) % 3 == 0, f"Coding length {len(s)} not divisible by 3"
    return [s[k : k + 3] for k in range(0, len(s), 3) if s[k : k + 3] not in ("---", "...")]


def tokenize_aligned_row(row: str, mode: str) -> list[str | None]:
    s = row.upper().replace("U", "T")
    if mode == "protein":
        return [None if ch in "-." else ch for ch in s]
    assert len(s) % 3 == 0, f"Aligned coding row len {len(s)} not divisible by 3"
    out: list[str | None] = []
    for k in range(0, len(s), 3):
        tri = s[k : k + 3]
        out.append(None if ("-" in tri or "." in tri) else tri)
    return out


def score_universal_mdl(
    orig_seqs: dict[str, str],
    components: list[dict[str, str]],
    mode: str,
) -> dict[str, Any]:
    """Compute aligner-agnostic KT + symmetric Dirichlet-Multinomial mixture MDL."""
    if mode == "protein":
        tok_idx = AA_IDX
        k_dim = 20
    else:
        has_stop = any(
            tri not in SENSE_IDX
            for s in orig_seqs.values()
            for tri in tokenize_seq(s, "codon")
        )
        tok_idx = ALL_CODON_IDX if has_stop else SENSE_IDX
        k_dim = len(tok_idx)

    # Single-sequence null KT description length L0
    l0_nats = 0.0
    total_units = 0
    for seq in orig_seqs.values():
        toks = tokenize_seq(seq, mode)
        total_units += len(toks)
        cnt = [0] * k_dim
        for tk in toks:
            idx = tok_idx.get(tk)
            if idx is not None:
                cnt[idx] += 1
        l0_nats += elias_rissanen_nats(len(toks)) - dm_ln_p(cnt, k_dim, 0.5 * k_dim)

    n_seqs = len(orig_seqs)
    comp_nats = partition_header_nats(n_seqs, [len(c) for c in components])
    emitted_pairs = 0
    cols_ge2 = 0
    gap_opens = 0
    exact_cons_cols = 0
    biophys_cons_cols = 0

    for comp in components:
        m = len(comp)
        if m == 1:
            seq = next(iter(comp.values()))
            toks = tokenize_seq(seq, mode)
            cnt = [0] * k_dim
            for tk in toks:
                idx = tok_idx.get(tk)
                if idx is not None:
                    cnt[idx] += 1
            comp_nats += elias_rissanen_nats(len(toks)) - dm_ln_p(cnt, k_dim, 0.5 * k_dim)
            continue

        sids = sorted(comp.keys())
        rows_tok = {sid: tokenize_aligned_row(comp[sid], mode) for sid in sids}
        w_cols = len(rows_tok[sids[0]])
        for sid in sids:
            assert len(rows_tok[sid]) == w_cols

        comp_nats += elias_rissanen_nats(w_cols)
        priv_streams: dict[str, list[str]] = {sid: [] for sid in sids}

        for sid in sids:
            bits = [0 if tk is None else 1 for tk in rows_tok[sid]]
            comp_nats += row_markov_gap_nats(bits)
            gap_opens += sum(
                1 for k in range(len(bits)) if bits[k] == 0 and (k == 0 or bits[k - 1] == 1)
            )

        for j in range(w_cols):
            pres = [(sid, rows_tok[sid][j]) for sid in sids if rows_tok[sid][j] is not None]
            nj = len(pres)
            if nj == 1:
                priv_streams[pres[0][0]].append(pres[0][1])  # type: ignore[arg-type]
            elif nj >= 2:
                cols_ge2 += 1
                emitted_pairs += nj * (nj - 1) // 2
                cnt = [0] * k_dim
                g_cnt = [0] * 6
                for _, tk in pres:
                    assert tk is not None
                    idx = tok_idx.get(tk)
                    if idx is not None:
                        cnt[idx] += 1
                    aa = tk if mode == "protein" else GC.get(tk, "X")
                    grp = AA_GROUPS.get(aa)
                    if grp is not None:
                        g_cnt[grp] += 1
                comp_nats += -col_universal_mix_ln_p(cnt, k_dim)
                if nj >= max(3, (m + 1) // 2):
                    if max(cnt) >= (nj + 1) // 2:
                        exact_cons_cols += 1
                    if max(g_cnt) >= math.ceil(0.70 * nj):
                        biophys_cons_cols += 1

        for sid, p_toks in priv_streams.items():
            if p_toks:
                cnt = [0] * k_dim
                for tk in p_toks:
                    idx = tok_idx.get(tk)
                    if idx is not None:
                        cnt[idx] += 1
                comp_nats += -dm_ln_p(cnt, k_dim, 0.5 * k_dim)

    delta_l = l0_nats - comp_nats
    bits_per_unit = (delta_l / math.log(2.0)) / max(1, total_units)
    return {
        "k_dim": k_dim,
        "total_units": total_units,
        "L0_nats": l0_nats,
        "L_msa_nats": comp_nats,
        "delta_L_nats": delta_l,
        "bits_per_unit_saved": bits_per_unit,
        "num_components": len(components),
        "emitted_pairs": emitted_pairs,
        "cols_ge2": cols_ge2,
        "gap_opens": gap_opens,
        "exact_cons_cols": exact_cons_cols,
        "biophys_cons_cols": biophys_cons_cols,
    }


def score_family_calibration(
    fam_of: dict[str, str],
    components: list[dict[str, str]],
    mode: str,
) -> dict[str, Any]:
    """Column-free pairwise homology precision/FDR & sequence-pair ROC against true family tags."""
    sids_all = sorted(fam_of.keys())
    true_res_pairs = 0
    false_res_pairs = 0
    coaligned_seq_pairs: set[tuple[str, str]] = set()

    for comp in components:
        if len(comp) < 2:
            continue
        sids = sorted(comp.keys())
        rows_tok = {sid: tokenize_aligned_row(comp[sid], mode) for sid in sids}
        w_cols = len(rows_tok[sids[0]])
        for j in range(w_cols):
            pres = [sid for sid in sids if rows_tok[sid][j] is not None]
            k = len(pres)
            if k < 2:
                continue
            for a_i in range(k):
                sa = pres[a_i]
                fa = fam_of[sa]
                for b_i in range(a_i + 1, k):
                    sb = pres[b_i]
                    pair = (sa, sb) if sa < sb else (sb, sa)
                    coaligned_seq_pairs.add(pair)
                    if fa == fam_of[sb]:
                        true_res_pairs += 1
                    else:
                        false_res_pairs += 1

    tot_emitted = true_res_pairs + false_res_pairs
    fdr_pair = (false_res_pairs / tot_emitted) if tot_emitted > 0 else 0.0

    # Sequence-pair level sensitivity and FPR
    true_hom_sp = 0
    true_nonhom_sp = 0
    pred_tp_sp = 0
    pred_fp_sp = 0
    for i in range(len(sids_all)):
        sa = sids_all[i]
        for j in range(i + 1, len(sids_all)):
            sb = sids_all[j]
            pair = (sa, sb)
            is_hom = fam_of[sa] == fam_of[sb]
            is_pred = pair in coaligned_seq_pairs
            if is_hom:
                true_hom_sp += 1
                if is_pred:
                    pred_tp_sp += 1
            else:
                true_nonhom_sp += 1
                if is_pred:
                    pred_fp_sp += 1

    sp_recall = (pred_tp_sp / true_hom_sp) if true_hom_sp > 0 else None
    sp_fpr = (pred_fp_sp / true_nonhom_sp) if true_nonhom_sp > 0 else None

    return {
        "true_fam_res_pairs": true_res_pairs,
        "false_fam_res_pairs": false_res_pairs,
        "FDR_pair": fdr_pair,
        "seqpair_tp": pred_tp_sp,
        "seqpair_true_hom": true_hom_sp,
        "seqpair_fp": pred_fp_sp,
        "seqpair_true_nonhom": true_nonhom_sp,
        "SeqPair_Recall": sp_recall,
        "SeqPair_FPR": sp_fpr,
    }


def parse_fasta(txt: str) -> dict[str, str]:
    out: dict[str, str] = {}
    cur: str | None = None
    buf: list[str] = []
    for ln in txt.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith(">"):
            if cur is not None:
                out[cur] = "".join(buf)
            cur = s[1:].split()[0]
            buf = []
        else:
            buf.append(s)
    if cur is not None:
        out[cur] = "".join(buf)
    return out


def read_hyphy_clean_cds(nex_path: Path, max_n: int | None = None) -> list[tuple[str, str, str]]:
    txt = nex_path.read_text(errors="replace")
    in_mat = False
    seqs: dict[str, str] = {}
    order: list[str] = []
    for ln in txt.splitlines():
        s = ln.strip()
        if not in_mat:
            if s.upper().startswith("MATRIX"):
                in_mat = True
            continue
        if s == ";":
            break
        end = s.endswith(";")
        if end:
            s = s[:-1].strip()
        if not s or s.startswith("["):
            if end:
                break
            continue
        parts = s.split(None, 1)
        if len(parts) == 2:
            tid = parts[0].strip("'\"")
            if tid not in seqs:
                order.append(tid)
            seqs[tid] = seqs.get(tid, "") + re.sub(r"\s+", "", parts[1]).upper()
        if end:
            break

    out: list[tuple[str, str, str]] = []
    seen_seqs: set[str] = set()
    for tid in order:
        ug = re.sub(r"[^ACGT]", "", seqs[tid])
        if len(ug) >= 240 and len(ug) % 3 == 0:
            aas = [GC.get(ug[i : i + 3], "*") for i in range(0, len(ug), 3)]
            if aas and aas[-1] == "*":
                aas = aas[:-1]
                ug = ug[:-3]
            if "*" not in aas and len(aas) >= 80 and ug not in seen_seqs:
                seen_seqs.add(ug)
                out.append((tid, ug, "".join(aas)))
                if max_n is not None and len(out) >= max_n:
                    break
    return out


def reverse_codons(nt_seq: str) -> str:
    cods = [nt_seq[i : i + 3] for i in range(0, len(nt_seq), 3)]
    cods.reverse()
    return "".join(cods)


def codon_to_aa_for_ext_aligner(nt_seq: str) -> tuple[str, list[str]]:
    cods = [nt_seq[i : i + 3].upper().replace("U", "T") for i in range(0, len(nt_seq), 3)]
    aas = []
    for c in cods:
        aa = GC.get(c, "X")
        aas.append("X" if aa == "*" else aa)
    return "".join(aas), cods


def backtranslate_aa_alignment(aln_aa: dict[str, str], orig_codons: dict[str, list[str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for sid, row in aln_aa.items():
        cods = orig_codons[sid]
        idx = 0
        pieces: list[str] = []
        for ch in row:
            if ch in "-.":
                pieces.append("---")
            else:
                pieces.append(cods[idx])
                idx += 1
        assert idx == len(cods), f"Backtranslate count mismatch on {sid}: {idx} != {len(cods)}"
        out[sid] = "".join(pieces)
    return out


def run_aligner(tool: str, seqs: dict[str, str], mode: str) -> tuple[list[dict[str, str]], float]:
    with tempfile.TemporaryDirectory(prefix=f"noncirc_{tool}_") as td:
        tmp = Path(td)
        inp = tmp / "in.fa"
        if tool == "squint":
            inp.write_text("".join(f">{k}\n{v}\n" for k, v in seqs.items()))
            out_fa = tmp / "out.fa"
            out_js = tmp / "out.json"
            t0 = time.perf_counter()
            subprocess.run(
                [str(SQUINT_BIN), str(inp), "-o", str(out_fa), "-j", str(out_js), "--mode", mode, "-T", "4"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            rt = time.perf_counter() - t0
            led = json.loads(out_js.read_text())
            comps = [{sid: aln for sid, aln in c["aligned_fasta"]} for c in led["components"]]
            return comps, rt

        # External protein/codon-AA tools
        if mode == "codon":
            aa_in: dict[str, str] = {}
            cod_map: dict[str, list[str]] = {}
            for k, v in seqs.items():
                aa_s, cods = codon_to_aa_for_ext_aligner(v)
                aa_in[k] = aa_s
                cod_map[k] = cods
            inp.write_text("".join(f">{k}\n{v}\n" for k, v in aa_in.items()))
        else:
            inp.write_text("".join(f">{k}\n{v}\n" for k, v in seqs.items()))

        t0 = time.perf_counter()
        if tool == "mafft":
            proc = subprocess.run(
                ["mafft", "--quiet", "--thread", "4", "--localpair", "--maxiterate", "1000", str(inp)],
                capture_output=True,
                text=True,
                check=True,
            )
            raw_aln = parse_fasta(proc.stdout)
        elif tool == "muscle":
            out_fa = tmp / "mu.fa"
            subprocess.run(["muscle", "-in", str(inp), "-out", str(out_fa), "-quiet"], check=True)
            raw_aln = parse_fasta(out_fa.read_text())
        elif tool == "clustalo":
            out_fa = tmp / "cl.fa"
            subprocess.run(["clustalo", "-i", str(inp), "-o", str(out_fa), "--threads=1", "--force"], check=True)
            raw_aln = parse_fasta(out_fa.read_text())
        else:
            raise ValueError(tool)
        rt = time.perf_counter() - t0

        if mode == "codon":
            return [backtranslate_aa_alignment(raw_aln, cod_map)], rt
        return [raw_aln], rt


def top_hallucinated_blocks(aln: dict[str, str], min_span: int = 8) -> dict[str, Any]:
    """Find best spurious 'conserved' alignment slices forced across unrelated proteins."""
    sids = sorted(aln.keys())
    W = len(aln[sids[0]])
    n = len(sids)
    col_stats: list[tuple[int, int, str, int, int, str]] = []
    for j in range(W):
        chars = [aln[s][j] for s in sids]
        pres = [c for c in chars if c not in "-."]
        nj = len(pres)
        if nj < max(3, (n + 1) // 2):
            col_stats.append((j, nj, "-", 0, 0, "-"))
            continue
        mc_aa, mc_cnt = Counter(pres).most_common(1)[0]
        g_cnt = Counter(AA_GROUPS.get(c, -1) for c in pres if c in AA_GROUPS)
        bg_grp, bg_cnt = g_cnt.most_common(1)[0] if g_cnt else (-1, 0)
        g_names = ["AVILM", "FYW", "KRH", "DE", "STNQ", "GPC"]
        col_stats.append((j, nj, mc_aa, mc_cnt, bg_cnt, g_names[bg_grp] if bg_grp >= 0 else "?"))

    # Sliding window of width 25 cols maximizing combined identity + biophysical concentration
    win = min(30, W)
    best_score = -1.0
    best_lo = 0
    for lo in range(0, max(1, W - win + 1)):
        sc = sum(
            (st[3] + 0.5 * st[4])
            for st in col_stats[lo : lo + win]
            if st[1] >= max(3, (n + 1) // 2)
        )
        if sc > best_score:
            best_score = sc
            best_lo = lo
    hi = min(W, best_lo + win)
    slice_rows = {sid: aln[sid][best_lo:hi] for sid in sids}
    # Highlight columns inside window where >= 40% share exact AA or >= 60% share biophys group
    markers = []
    for st in col_stats[best_lo:hi]:
        _, nj, _, mc_cnt, bg_cnt, _ = st
        if nj >= max(3, (n + 1) // 2) and mc_cnt * 2 >= nj:
            markers.append("*")
        elif nj >= max(3, (n + 1) // 2) and bg_cnt >= math.ceil(0.65 * nj):
            markers.append("+")
        else:
            markers.append(" ")
    return {
        "window": [best_lo, hi],
        "marker_line": "".join(markers),
        "rows": slice_rows,
        "mean_occupancy": statistics.mean(
            sum(1 for sid in sids if aln[sid][j] not in "-.") / n for j in range(best_lo, hi)
        ),
    }


def build_cohorts() -> list[dict[str, Any]]:
    cohorts: list[dict[str, Any]] = []

    cross_sources = [
        ("Globin", HYPHY_DATA / "bglobin.nex"),
        ("IgVHH", HYPHY_DATA / "camelid.nex"),
        ("Lysozyme", HYPHY_DATA / "lysozyme.nex"),
        ("SpermLysin", HYPHY_DATA / "lysin.nex"),
        ("HIV1Vif", HYPHY_DATA / "HIVvif.nex"),
        ("ADH", HYPHY_DATA / "adh.nex"),
        ("AdoR_GPCR", HYPHY_DATA / "adora3.nex"),
        ("RuBisCO", HYPHY_DATA / "rbcL.nex"),
        ("FluH1N1HA", HYPHY_DATA / "IAV-human-H1N1-HA.nex"),
        ("PTSTransp", HYPHY_DATA / "Bacterial_PTS_trehalose_transporter_suIII.nex"),
    ]
    cross_recs = []
    for lbl, p in cross_sources:
        tid, nt, aa = read_hyphy_clean_cds(p, max_n=1)[0]
        cross_recs.append((f"{lbl}_{tid}", nt, aa))

    # Regime A: Zero Reasonable Homology (6 cohorts)
    cohorts.append({
        "cohort": "ZeroHom_10_CrossKingdom_AA",
        "regime": "RegimeA_ZeroHomology",
        "mode": "protein",
        "seqs": {sid: aa for sid, _, aa in cross_recs},
        "fam_of": {sid: f"SOLO_{i}" for i, (sid, _, _) in enumerate(cross_recs)},
        "capture_slice": True,
    })
    cohorts.append({
        "cohort": "ZeroHom_10_CrossKingdom_Codon",
        "regime": "RegimeA_ZeroHomology",
        "mode": "codon",
        "seqs": {sid: nt for sid, nt, _ in cross_recs},
        "fam_of": {sid: f"SOLO_{i}" for i, (sid, _, _) in enumerate(cross_recs)},
        "capture_slice": False,
    })
    compact7 = [(sid, aa) for sid, _, aa in cross_recs if len(aa) <= 260][:7]
    cohorts.append({
        "cohort": "ZeroHom_7_SingleDomain_AA",
        "regime": "RegimeA_ZeroHomology",
        "mode": "protein",
        "seqs": {sid: aa for sid, aa in compact7},
        "fam_of": {sid: f"SOLO_{i}" for i, (sid, _) in enumerate(compact7)},
        "capture_slice": True,
    })

    # Distinct SCOP fold singletons from sabre twilight sets (100..160 aa)
    with tarfile.open(ARCHIVE_PATH, "r:gz") as tf:
        sab_ids = [
            ln.strip()
            for ln in tf.extractfile("bench1.0/sabre/info/ids.txt").read().decode().splitlines()
            if ln.strip().startswith("twi_")
        ]
        scop_reps: list[tuple[str, str]] = []
        for did in sab_ids:
            raw = tf.extractfile(f"bench1.0/sabre/in/{did}").read().decode()
            parsed = parse_fasta(raw)
            cands = [(k, v) for k, v in parsed.items() if 100 <= len(v) <= 160 and set(v).issubset(set(AA_ALPH))]
            if cands:
                # Choose median-length representative
                cands.sort(key=lambda kv: len(kv[1]))
                sk, sv = cands[len(cands) // 2]
                scop_reps.append((f"{did}_{sk}", sv))
                if len(scop_reps) >= 20:
                    break

        cohorts.append({
            "cohort": "ZeroHom_10_SCOP_Folds_Set1",
            "regime": "RegimeA_ZeroHomology",
            "mode": "protein",
            "seqs": {k: v for k, v in scop_reps[:10]},
            "fam_of": {k: f"SCOP1_{i}" for i, (k, _) in enumerate(scop_reps[:10])},
            "capture_slice": False,
        })
        cohorts.append({
            "cohort": "ZeroHom_10_SCOP_Folds_Set2",
            "regime": "RegimeA_ZeroHomology",
            "mode": "protein",
            "seqs": {k: v for k, v in scop_reps[10:20]},
            "fam_of": {k: f"SCOP2_{i}" for i, (k, _) in enumerate(scop_reps[10:20])},
            "capture_slice": False,
        })

        bb11028 = list(parse_fasta(tf.extractfile("bench1.0/bali3pdb/in/BB11028").read().decode()).items())
        bb30002 = list(parse_fasta(tf.extractfile("bench1.0/bali3pdb/in/BB30002").read().decode()).items())
        bb11005 = parse_fasta(tf.extractfile("bench1.0/bali3pdb/in/BB11005").read().decode())
        twi114 = parse_fasta(tf.extractfile("bench1.0/sabre/in/twi_114").read().decode())

    cohorts.append({
        "cohort": "ZeroHom_10_Reversed_CrossKingdom_AA",
        "regime": "RegimeA_ZeroHomology",
        "mode": "protein",
        "seqs": {f"REV_{sid}": aa[::-1] for sid, _, aa in cross_recs},
        "fam_of": {f"REV_{sid}": f"RSOLO_{i}" for i, (sid, _, _) in enumerate(cross_recs)},
        "capture_slice": False,
    })

    # Regime B: Decoy Entrapment & Multi-Family Mixtures (8 cohorts)
    bg_8 = read_hyphy_clean_cds(HYPHY_DATA / "bglobin.nex", max_n=8)
    ly_8 = read_hyphy_clean_cds(HYPHY_DATA / "lysin.nex", max_n=8)

    def add_twin_entrap(name: str, items: list[tuple[str, str]], mode: str, tag: str) -> None:
        seqs_d: dict[str, str] = {}
        fam_d: dict[str, str] = {}
        for k, s in items:
            fk = f"FWD_{k}"
            rk = f"REV_{k}"
            seqs_d[fk] = s
            seqs_d[rk] = s[::-1] if mode == "protein" else reverse_codons(s)
            fam_d[fk] = f"{tag}_FWD"
            fam_d[rk] = f"{tag}_REV"
        cohorts.append({
            "cohort": name,
            "regime": "RegimeB_Entrapment_MultiFam",
            "mode": mode,
            "seqs": seqs_d,
            "fam_of": fam_d,
            "capture_slice": False,
        })

    def add_solo_decoys(
        name: str,
        items: list[tuple[str, str]],
        mode: str,
        tag: str,
        skip_prefix: str | None,
        num_decoys: int,
        max_aa_len: int | None = None,
    ) -> None:
        seqs_d: dict[str, str] = {}
        fam_d: dict[str, str] = {}
        for k, s in items:
            fk = f"{tag}_{k}"
            seqs_d[fk] = s
            fam_d[fk] = tag
        decs = []
        for sid, nt, aa in cross_recs:
            if skip_prefix and sid.startswith(skip_prefix):
                continue
            if max_aa_len is not None and len(aa) > max_aa_len:
                continue
            decs.append((sid, nt, aa))
            if len(decs) >= num_decoys:
                break
        for i, (sid, nt, aa) in enumerate(decs):
            dk = f"REV_DECOY_{sid}"
            seqs_d[dk] = aa[::-1] if mode == "protein" else reverse_codons(nt)
            fam_d[dk] = f"SOLO_DECOY_{i}"
        cohorts.append({
            "cohort": name,
            "regime": "RegimeB_Entrapment_MultiFam",
            "mode": mode,
            "seqs": seqs_d,
            "fam_of": fam_d,
            "capture_slice": False,
        })

    add_twin_entrap("Entrap_TwinPol_bglobin_AA", [(k, aa) for k, _, aa in bg_8], "protein", "BGLOBIN")
    add_twin_entrap("Entrap_TwinPol_lysin_Codon", [(k, nt) for k, nt, _ in ly_8], "codon", "LYSIN")
    add_twin_entrap("Entrap_TwinPol_BB11028_AA", bb11028[:8], "protein", "BB11028")
    add_solo_decoys("Entrap_SoloDecoys_bglobin_AA", [(k, aa) for k, _, aa in bg_8], "protein", "BGLOBIN", "Globin_", 8)
    add_solo_decoys("Entrap_SoloDecoys_lysin_Codon", [(k, nt) for k, nt, _ in ly_8], "codon", "LYSIN", "SpermLysin_", 8)
    add_solo_decoys("Entrap_SoloDecoys_BB30002_AA", bb30002[:4], "protein", "BB30002", None, 6, max_aa_len=300)

    mixed_neg = parse_fasta((REPO_ROOT / "benchmarks" / "mixed_negative_control.fa").read_text())
    mixed_fam = {}
    for k in mixed_neg:
        if k.startswith("REV_"):
            mixed_fam[k] = f"DECOY_{k}"
        elif k.startswith("KPC_"):
            mixed_fam[k] = "KPC"
        elif k.startswith("BGLOBIN_"):
            mixed_fam[k] = "BGLOBIN"
        elif k.startswith("LYSIN_"):
            mixed_fam[k] = "LYSIN"
        elif k.startswith("FWD_TP53_"):
            mixed_fam[k] = "TP53"
        elif k.startswith("FWD_COXI_"):
            mixed_fam[k] = "COXI"
    cohorts.append({
        "cohort": "MultiFam_MixedNegCtrl_Codon",
        "regime": "RegimeB_Entrapment_MultiFam",
        "mode": "codon",
        "seqs": mixed_neg,
        "fam_of": mixed_fam,
        "capture_slice": False,
    })

    mf6_sources = [
        ("BGLOB", HYPHY_DATA / "bglobin.nex"),
        ("IGVHH", HYPHY_DATA / "camelid.nex"),
        ("LYSOZ", HYPHY_DATA / "lysozyme.nex"),
        ("LYSIN", HYPHY_DATA / "lysin.nex"),
        ("VIF", HYPHY_DATA / "HIVvif.nex"),
        ("ADH", HYPHY_DATA / "adh.nex"),
    ]
    mf6_seqs: dict[str, str] = {}
    mf6_fam: dict[str, str] = {}
    for ftag, path in mf6_sources:
        for tid, _, aa in read_hyphy_clean_cds(path, max_n=5):
            sid = f"{ftag}_{tid}"
            mf6_seqs[sid] = aa
            mf6_fam[sid] = ftag
    cohorts.append({
        "cohort": "MultiFam_6HyPhy_30Seqs_AA",
        "regime": "RegimeB_Entrapment_MultiFam",
        "mode": "protein",
        "seqs": mf6_seqs,
        "fam_of": mf6_fam,
        "capture_slice": False,
    })

    # Regime C: Real Uncurated Homologous Cohorts Across Divergence (8 cohorts)
    blakpc = parse_fasta((REPO_ROOT / "benchmarks" / "blaKPC_homolog.fa").read_text())
    hom_specs = [
        ("Hom_blaKPC_Codon", "codon", blakpc),
        ("Hom_bglobin_AA", "protein", {tid: aa for tid, _, aa in read_hyphy_clean_cds(HYPHY_DATA / "bglobin.nex")}),
        ("Hom_lysozyme_AA", "protein", {tid: aa for tid, _, aa in read_hyphy_clean_cds(HYPHY_DATA / "lysozyme.nex")}),
        ("Hom_lysin_AA", "protein", {tid: aa for tid, _, aa in read_hyphy_clean_cds(HYPHY_DATA / "lysin.nex")}),
        ("Hom_camelid_AA", "protein", {tid: aa for tid, _, aa in read_hyphy_clean_cds(HYPHY_DATA / "camelid.nex", max_n=25)}),
        ("Hom_HIVvif_AA", "protein", {tid: aa for tid, _, aa in read_hyphy_clean_cds(HYPHY_DATA / "HIVvif.nex", max_n=24)}),
        ("Hom_BB11005_Twilight_AA", "protein", bb11005),
        ("Hom_twi114_ExtremeTwilight_AA", "protein", twi114),
    ]
    for name, mode, seqs_d in hom_specs:
        cohorts.append({
            "cohort": name,
            "regime": "RegimeC_RealHomologs",
            "mode": mode,
            "seqs": seqs_d,
            "fam_of": {k: name for k in seqs_d},
            "capture_slice": False,
        })

    return cohorts


def main() -> None:
    kraft = verify_kraft_identities()
    print("[Kraft Verification]", json.dumps(kraft))

    cohorts = build_cohorts()
    print(f"Running {len(cohorts)} authentic non-circular benchmark cohorts across {TOOLS}...")

    rows: list[dict[str, Any]] = []
    slices: dict[str, Any] = {}

    for c_spec in cohorts:
        cname = c_spec["cohort"]
        regime = c_spec["regime"]
        mode = c_spec["mode"]
        seqs = c_spec["seqs"]
        fam_of = c_spec["fam_of"]
        n_fam = len(set(fam_of.values()))

        print(f"\n=== {cname} ({regime}, N={len(seqs)}, true_groups={n_fam}, mode={mode}) ===")
        for tool in TOOLS:
            comps, rt = run_aligner(tool, seqs, mode)
            mdl = score_universal_mdl(seqs, comps, mode)
            cal = score_family_calibration(fam_of, comps, mode)
            row = {
                "cohort": cname,
                "regime": regime,
                "mode": mode,
                "n_seqs": len(seqs),
                "true_groups": n_fam,
                "tool": tool,
                "runtime_s": rt,
                **mdl,
                **cal,
            }
            rows.append(row)
            fdr_str = f"{cal['FDR_pair']:.4f}"
            rec_str = f"{cal['SeqPair_Recall']:.3f}" if cal["SeqPair_Recall"] is not None else "n/a"
            fpr_str = f"{cal['SeqPair_FPR']:.3f}" if cal["SeqPair_FPR"] is not None else "n/a"
            print(
                f"  {tool:9s} | C={mdl['num_components']:2d} | L0={mdl['L0_nats']:7.1f} -> L_msa={mdl['L_msa_nats']:7.1f} "
                f"(dL={mdl['delta_L_nats']:+8.1f} nats, {mdl['bits_per_unit_saved']:+6.3f} b/u) | "
                f"pairs={mdl['emitted_pairs']:5d} (false={cal['false_fam_res_pairs']:5d}, FDR={fdr_str}) | "
                f"Rec={rec_str} FPR={fpr_str} | cons={mdl['exact_cons_cols']}/{mdl['biophys_cons_cols']} gaps={mdl['gap_opens']}"
            )
            if c_spec["capture_slice"]:
                slices.setdefault(cname, {})[tool] = (
                    top_hallucinated_blocks(comps[0]) if len(comps) == 1 and len(comps[0]) > 1 else {"abstained_components": len(comps)}
                )

    fieldnames = list(rows[0].keys())
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    summary: dict[str, Any] = {"kraft_checks": kraft, "by_regime": {}, "slices": slices}
    for reg in ("RegimeA_ZeroHomology", "RegimeB_Entrapment_MultiFam", "RegimeC_RealHomologs"):
        reg_s: dict[str, Any] = {}
        for tool in TOOLS:
            tr = [r for r in rows if r["regime"] == reg and r["tool"] == tool]
            tot_tp = sum(r["true_fam_res_pairs"] for r in tr)
            tot_fp = sum(r["false_fam_res_pairs"] for r in tr)
            tot_emit = tot_tp + tot_fp
            tot_l0 = sum(r["L0_nats"] for r in tr)
            tot_lm = sum(r["L_msa_nats"] for r in tr)
            sp_fp_tot = sum(r["seqpair_fp"] for r in tr)
            sp_nh_tot = sum(r["seqpair_true_nonhom"] for r in tr)
            sp_tp_tot = sum(r["seqpair_tp"] for r in tr)
            sp_h_tot = sum(r["seqpair_true_hom"] for r in tr)
            reg_s[tool] = {
                "cohorts": len(tr),
                "pooled_FDR_pair": (tot_fp / tot_emit) if tot_emit > 0 else 0.0,
                "total_emitted_pairs": tot_emit,
                "total_false_pairs": tot_fp,
                "pooled_SeqPair_FPR": (sp_fp_tot / sp_nh_tot) if sp_nh_tot > 0 else None,
                "pooled_SeqPair_Recall": (sp_tp_tot / sp_h_tot) if sp_h_tot > 0 else None,
                "total_L0_nats": tot_l0,
                "total_L_msa_nats": tot_lm,
                "total_delta_L_nats": tot_l0 - tot_lm,
                "total_exact_cons_cols": sum(r["exact_cons_cols"] for r in tr),
                "total_biophys_cons_cols": sum(r["biophys_cons_cols"] for r in tr),
                "total_gap_opens": sum(r["gap_opens"] for r in tr),
            }
        summary["by_regime"][reg] = reg_s

    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {OUT_CSV} and {OUT_JSON}")
    print("\n=== POOLED REGIME SUMMARY ===")
    print(json.dumps(summary["by_regime"], indent=2))


if __name__ == "__main__":
    main()
