#!/usr/bin/env python3
"""'Slam-Dunk Easy' MSA benchmark suite comparing SQUINT against MAFFT L-INS-i,
MUSCLE, and Clustal Omega across 29 high-identity cohorts in 5 regimes:

  - E1: Ungapped conserved coding genes (--mode codon, evaluated at nt coordinates)
  - E2: Clean biological indels in coding genes (--mode codon, evaluated at nt coordinates)
  - E3: High-identity 3D structural proteins from OXBench (--mode protein, core & all pairs)
  - E4: High-identity 4-letter nucleotide alignments (--mode dna, evaluated at nt coordinates)
  - E5: Clinical amplicon & stepping-stone fragment tiling (--mode codon & --mode dna)

All 4 aligners receive identical unaligned FASTA inputs and are scored identically on both:
  1. Reference MSA concordance (Pair Precision, Pair Recall Q, Pair F1, Exact Column TC)
  2. Aligner-neutral, Kraft-tight Universal Lossless Compression (Delta L_KT_nats over K=4 or K=20).

Outputs:
  - benchmarks/easy_msa_results.csv
  - benchmarks/easy_msa_summary.json
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
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
OUT_CSV = REPO_ROOT / "benchmarks" / "easy_msa_results.csv"
OUT_JSON = REPO_ROOT / "benchmarks" / "easy_msa_summary.json"

MAFFT_BIN = shutil.which("mafft") or os.environ.get("MAFFT_BIN", "mafft")
MUSCLE_BIN = shutil.which("muscle") or os.environ.get("MUSCLE_BIN", "muscle")
CLUSTALO_BIN = shutil.which("clustalo") or os.environ.get("CLUSTALO_BIN", "clustalo")

TOOLS = ("squint", "mafft", "muscle", "clustalo")
DNA_ALPH = "ACGT"
DNA_IDX = {c: i for i, c in enumerate(DNA_ALPH)}
DNA_IUPAC_EXT = 11  # R, Y, S, W, K, M, B, D, H, V, N
AA_ALPH = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {c: i for i, c in enumerate(AA_ALPH)}


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

    # Ambiguity indicator + canonical K=4 + explicit A=11 IUPAC code sum over n=3 strings
    # For fixed length n=3, total alphabet size = 4 + 11 = 15, sum over all 15^3 strings == 1.0
    ambig_str_sum = 0.0
    for s0 in range(15):
        for s1 in range(15):
            for s2 in range(15):
                cnt4 = [0, 0, 0, 0]
                na = 0
                for sym in (s0, s1, s2):
                    if sym < 4:
                        cnt4[sym] += 1
                    else:
                        na += 1
                nc = 3 - na
                lp = (
                    dm_ln_p([nc, na], 2, 1.0)
                    + col_universal_mix_ln_p(cnt4, 4)
                    - na * math.log(DNA_IUPAC_EXT)
                )
                ambig_str_sum += math.exp(lp)

    n_max = 200_000
    er_sum = sum(math.exp(-elias_rissanen_nats(k)) for k in range(n_max + 1)) + (
        0.5 * math.log(2.0) / math.log(n_max + 2.0)
    )

    assert abs(col_sum - 1.0) < 1e-12, col_sum
    assert abs(row_sum - 1.0) < 1e-12, row_sum
    assert abs(ambig_str_sum - 1.0) < 1e-11, ambig_str_sum
    assert abs(er_sum - 1.0) < 1e-12, er_sum
    return {
        "column_dm_mixture_kraft_sum_K4_n5": col_sum,
        "row_markov_kt_kraft_sum_W8": row_sum,
        "iupac_escaped_K4_string_kraft_sum_n3": ambig_str_sum,
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


def parse_nex(path: Path) -> list[tuple[str, str]]:
    txt = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"MATRIX\s+(.*?)\s*;", txt, re.S | re.I)
    if not m:
        raise RuntimeError(f"Could not find MATRIX block in {path}")
    seqs: dict[str, list[str]] = {}
    order: list[str] = []
    for line in m.group(1).splitlines():
        s = line.strip()
        if not s or s.startswith("["):
            continue
        parts = s.split()
        if len(parts) >= 2:
            name = parts[0].strip("'\"")
            seq_chunk = "".join(parts[1:]).upper().replace("U", "T").replace("?", "N")
            if name not in seqs:
                order.append(name)
                seqs[name] = []
            seqs[name].append(seq_chunk)
    return [(k, "".join(seqs[k])) for k in order]


def compute_pairwise_identity(ref_recs: list[tuple[str, str]], uppercase_only: bool) -> float:
    rows = [s for _, s in ref_recs]
    n = len(rows)
    w = len(rows[0])
    total = 0
    ident = 0
    for i in range(n):
        ri = rows[i]
        for j in range(i + 1, n):
            rj = rows[j]
            for c in range(w):
                a = ri[c]
                b = rj[c]
                if uppercase_only:
                    if a.isalpha() and b.isalpha() and a.isupper() and b.isupper():
                        total += 1
                        if a == b:
                            ident += 1
                else:
                    if a not in "-.?" and b not in "-.?":
                        total += 1
                        if a.upper() == b.upper():
                            ident += 1
    return 100.0 * ident / max(1, total)


def banded_nw_map(orig: str, ug: str) -> list[int]:
    a = orig.upper().replace("U", "T")
    b = ug.upper().replace("U", "T")
    n, m = len(a), len(b)
    if m == 0:
        return []
    if m == n and a == b:
        return list(range(n))

    band = abs(n - m) + 28
    neg_inf = -10**9
    gap = -2

    dp_rows: list[dict[int, int]] = []
    bt_rows: list[dict[int, int]] = []
    row0: dict[int, int] = {}
    bt0: dict[int, int] = {}
    for i in range(min(n, band) + 1):
        row0[i] = i * gap
        bt0[i] = 1 if i > 0 else 0
    dp_rows.append(row0)
    bt_rows.append(bt0)

    ratio = n / m
    for j in range(1, m + 1):
        center = int(round(j * ratio))
        lo = max(0, min(j, center) - band)
        hi = min(n, max(j, center) + band)
        prev = dp_rows[-1]
        cur: dict[int, int] = {}
        bt: dict[int, int] = {}
        bj = b[j - 1]
        for i in range(lo, hi + 1):
            best = neg_inf
            best_op = -1
            if i > 0 and (i - 1) in prev:
                score = prev[i - 1] + (2 if a[i - 1] == bj else -1)
                if score > best:
                    best = score
                    best_op = 0
            if i in prev:
                score = prev[i] + gap
                if score > best:
                    best = score
                    best_op = 2
            if (i - 1) in cur:
                score = cur[i - 1] + gap
                if score > best:
                    best = score
                    best_op = 1
            cur[i] = best
            bt[i] = best_op
        dp_rows.append(cur)
        bt_rows.append(bt)

    if n not in dp_rows[m]:
        raise RuntimeError(f"banded_nw_map failed to reach ({n}, {m}) with band={band}")

    mapping = [-1] * m
    i, j = n, m
    while j > 0:
        op = bt_rows[j][i]
        if op == 0:
            mapping[j - 1] = i - 1
            i -= 1
            j -= 1
        elif op == 1:
            i -= 1
        elif op == 2:
            j -= 1
        else:
            raise RuntimeError(f"invalid backtrace state at ({i}, {j})")
    return mapping


def map_subseq_to_orig(orig: str, ug: str, fs_events: list[dict[str, Any]]) -> list[int]:
    """Map ungapped emitted residue positions back to 0-based indices in orig."""
    if len(ug) == len(orig):
        return list(range(len(orig)))

    orig_u = orig.upper().replace("U", "T")
    ug_u = ug.upper().replace("U", "T")

    if fs_events:
        excluded: set[int] = set()
        for ev in fs_events:
            p0 = int(ev["nt_pos"])
            ps = int(ev["phase_shift"])
            for k in range(p0, p0 + ps):
                excluded.add(k)
        rem = [i for i in range(len(orig)) if i not in excluded]
        for lead in (0, 1, 2):
            if len(rem) >= lead + len(ug_u):
                cand = "".join(orig_u[i] for i in rem[lead : lead + len(ug_u)])
                if cand == ug_u:
                    return rem[lead : lead + len(ug_u)]

    for lead in (0, 1, 2):
        if len(orig_u) >= lead + len(ug_u) and orig_u[lead : lead + len(ug_u)] == ug_u:
            return list(range(lead, lead + len(ug_u)))

    return banded_nw_map(orig, ug)


def extract_ref_structures(
    ref_seqs: dict[str, str],
    uppercase_core_only: bool,
) -> dict[str, Any]:
    sids = sorted(ref_seqs.keys())
    l_ref = len(next(iter(ref_seqs.values())))
    ug_len = {s: sum(1 for ch in ref_seqs[s] if ch not in ".-") for s in sids}

    core_rc_of = {s: [-1] * ug_len[s] for s in sids}
    is_core = {s: [False] * ug_len[s] for s in sids}
    core_col_sizes: list[int] = []
    n_core_pairs = 0

    all_rc_of = {s: [-1] * ug_len[s] for s in sids}
    all_col_sizes: list[int] = []
    n_all_pairs = 0

    pos = {s: 0 for s in sids}
    for c in range(l_ref):
        core_pos: list[tuple[str, int]] = []
        all_pos: list[tuple[str, int]] = []
        for s in sids:
            ch = ref_seqs[s][c]
            if ch not in ".-":
                p = pos[s]
                pos[s] += 1
                all_pos.append((s, p))
                if not uppercase_core_only or ch.isupper():
                    is_core[s][p] = True
                    core_pos.append((s, p))
        ka = len(all_pos)
        if ka >= 2:
            a_id = len(all_col_sizes)
            all_col_sizes.append(ka)
            n_all_pairs += ka * (ka - 1) // 2
            for s, p in all_pos:
                all_rc_of[s][p] = a_id
        kc = len(core_pos)
        if kc >= 2:
            c_id = len(core_col_sizes)
            core_col_sizes.append(kc)
            n_core_pairs += kc * (kc - 1) // 2
            for s, p in core_pos:
                core_rc_of[s][p] = c_id

    return {
        "core_rc_of": core_rc_of,
        "is_core": is_core,
        "core_col_sizes": core_col_sizes,
        "n_core_pairs": n_core_pairs,
        "all_rc_of": all_rc_of,
        "all_col_sizes": all_col_sizes,
        "n_all_pairs": n_all_pairs,
    }


def score_alignment(
    orig_seqs: dict[str, str],
    ref_info: dict[str, Any],
    components_alns: list[dict[str, str]],
    fs_by_seq: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    core_rc_of = ref_info["core_rc_of"]
    is_core = ref_info["is_core"]
    core_col_sizes = ref_info["core_col_sizes"]
    n_core_pairs = ref_info["n_core_pairs"]

    all_rc_of = ref_info["all_rc_of"]
    all_col_sizes = ref_info["all_col_sizes"]
    n_all_pairs = ref_info["n_all_pairs"]

    seen_sids: set[str] = set()
    tp_core = 0
    n_test_both_core = 0
    matched_core_rc: set[int] = set()

    tp_all = 0
    n_test_all_pairs = 0
    matched_all_rc: set[int] = set()

    excised_by_seq: dict[str, int] = {}

    for aln in components_alns:
        if not aln:
            continue
        asids = sorted(aln.keys())
        for s in asids:
            if s in seen_sids:
                raise RuntimeError(f"Duplicate sequence {s} across components")
            seen_sids.add(s)

        la = len(next(iter(aln.values())))
        pos_map: dict[str, list[int]] = {}
        apos = {s: 0 for s in asids}
        for s in asids:
            if len(aln[s]) != la:
                raise RuntimeError(f"Jagged aligned sequence {s}: {len(aln[s])} != {la}")
            ug = re.sub(r"[.\-!]", "", aln[s])
            excised_by_seq[s] = len(orig_seqs[s]) - len(ug)
            evs = fs_by_seq.get(s, []) if fs_by_seq else []
            pos_map[s] = map_subseq_to_orig(orig_seqs[s], ug, evs)

        if len(asids) < 2:
            continue

        for c in range(la):
            col_items: list[tuple[str, int]] = []
            kc = 0
            for s in asids:
                ch = aln[s][c]
                if ch not in ".-!":
                    p = pos_map[s][apos[s]]
                    apos[s] += 1
                    col_items.append((s, p))
                    if is_core[s][p]:
                        kc += 1
            kt = len(col_items)
            if kt >= 2:
                n_test_all_pairs += kt * (kt - 1) // 2
                n_test_both_core += kc * (kc - 1) // 2

                c_counts: Counter[int] = Counter()
                a_counts: Counter[int] = Counter()
                for s, p in col_items:
                    cr_id = core_rc_of[s][p]
                    if cr_id != -1:
                        c_counts[cr_id] += 1
                    ar_id = all_rc_of[s][p]
                    if ar_id != -1:
                        a_counts[ar_id] += 1

                for r_id, cnt in c_counts.items():
                    if cnt >= 2:
                        tp_core += cnt * (cnt - 1) // 2
                    if cnt == kc and kc == core_col_sizes[r_id]:
                        matched_core_rc.add(r_id)

                for r_id, cnt in a_counts.items():
                    if cnt >= 2:
                        tp_all += cnt * (cnt - 1) // 2
                    if cnt == kt and kt == all_col_sizes[r_id]:
                        matched_all_rc.add(r_id)

    if seen_sids != set(orig_seqs.keys()):
        raise RuntimeError(f"Missing sequences: expected {len(orig_seqs)}, got {len(seen_sids)}")

    q_core = tp_core / max(1, n_core_pairs)
    prec_core = tp_core / max(1, n_test_both_core) if n_test_both_core > 0 else 1.0
    f1_core = (
        2.0 * tp_core / (n_core_pairs + n_test_both_core)
        if (n_core_pairs + n_test_both_core) > 0
        else 1.0
    )
    tc_core = len(matched_core_rc) / max(1, len(core_col_sizes))

    q_all = tp_all / max(1, n_all_pairs)
    prec_all = tp_all / max(1, n_test_all_pairs) if n_test_all_pairs > 0 else 1.0
    f1_all = (
        2.0 * tp_all / (n_all_pairs + n_test_all_pairs)
        if (n_all_pairs + n_test_all_pairs) > 0
        else 1.0
    )
    tc_all = len(matched_all_rc) / max(1, len(all_col_sizes))

    return {
        "Pair_Prec": prec_core,
        "Pair_Recall_Q": q_core,
        "Pair_F1": f1_core,
        "Exact_Col_TC": tc_core,
        "All_Pair_Prec": prec_all,
        "All_Pair_Q": q_all,
        "All_Pair_F1": f1_all,
        "All_Exact_Col_TC": tc_all,
        "tp_core": tp_core,
        "n_core_pairs": n_core_pairs,
        "n_test_both_core": n_test_both_core,
        "tc_matched": len(matched_core_rc),
        "n_core_cols": len(core_col_sizes),
        "excised_by_seq": excised_by_seq,
    }


def score_easy_kt_mdl(
    orig_seqs: dict[str, str],
    components: list[dict[str, str]],
    eval_alphabet: str,
    excised_by_seq: dict[str, int],
) -> dict[str, Any]:
    """Aligner-neutral, Kraft-tight Universal Lossless Compression on identical K=4 or K=20 string space.
    For nucleotide cohorts containing IUPAC ambiguities, uses exact binary KT(1/2) canonical-vs-ambig
    split + canonical K=4 KT(1/2) + uniform IUPAC code (ln 11) in both L0 and L_msa.
    """
    if eval_alphabet == "protein":
        tok_idx = AA_IDX
        k_dim = 20
        ambig_escape_nats = math.log(6.0)  # B, Z, X, U, O, *
    else:
        tok_idx = DNA_IDX
        k_dim = 4
        ambig_escape_nats = math.log(float(DNA_IUPAC_EXT))

    has_any_ambig = any(
        ch not in tok_idx
        for s in orig_seqs.values()
        for ch in s.upper().replace("U", "T")
    )

    def encode_stream_nats(chars: list[str], is_column: bool) -> float:
        cnt = [0] * k_dim
        na = 0
        for ch in chars:
            idx = tok_idx.get(ch)
            if idx is not None:
                cnt[idx] += 1
            else:
                na += 1
        nc = len(chars) - na
        nats = -col_universal_mix_ln_p(cnt, k_dim) if is_column else -dm_ln_p(cnt, k_dim, 0.5 * k_dim)
        if has_any_ambig and len(chars) > 0:
            nats += -dm_ln_p([nc, na], 2, 1.0) + na * ambig_escape_nats
        return nats

    l0_nats = 0.0
    total_units = 0
    for seq in orig_seqs.values():
        s_clean = list(seq.upper().replace("U", "T"))
        total_units += len(s_clean)
        l0_nats += elias_rissanen_nats(len(s_clean)) + encode_stream_nats(s_clean, is_column=False)

    n_seqs = len(orig_seqs)
    comp_nats = partition_header_nats(n_seqs, [len(c) for c in components])
    gap_opens = 0
    total_cols = 0

    for comp in components:
        m = len(comp)
        if m == 1:
            sid = next(iter(comp.keys()))
            raw = [ch for ch in comp[sid].upper().replace("U", "T") if ch not in ".-!"]
            comp_nats += elias_rissanen_nats(len(raw)) + encode_stream_nats(raw, is_column=False)
            total_cols = max(total_cols, len(raw))
            continue

        sids = sorted(comp.keys())
        rows = {sid: list(comp[sid].upper().replace("U", "T")) for sid in sids}
        w_cols = len(rows[sids[0]])
        total_cols += w_cols
        comp_nats += elias_rissanen_nats(w_cols)

        for sid in sids:
            bits = [0 if ch in ".-!" else 1 for ch in rows[sid]]
            comp_nats += row_markov_gap_nats(bits)
            gap_opens += sum(
                1 for k in range(len(bits)) if bits[k] == 0 and (k == 0 or bits[k - 1] == 1)
            )

        priv_streams: dict[str, list[str]] = {sid: [] for sid in sids}
        for j in range(w_cols):
            pres = [(sid, rows[sid][j]) for sid in sids if rows[sid][j] not in ".-!"]
            nj = len(pres)
            if nj == 1:
                priv_streams[pres[0][0]].append(pres[0][1])
            elif nj >= 2:
                comp_nats += encode_stream_nats([ch for _, ch in pres], is_column=True)

        for sid, p_chars in priv_streams.items():
            if p_chars:
                comp_nats += encode_stream_nats(p_chars, is_column=False)

    # Losslessly charge any trimmed/excised nucleotides (e.g., partial-codon ends or repaired frameshifts)
    total_excised = 0
    for sid, r in excised_by_seq.items():
        if r > 0:
            total_excised += r
            comp_nats += r * (math.log(float(k_dim)) + elias_rissanen_nats(len(orig_seqs[sid])))

    delta_l = l0_nats - comp_nats
    bits_per_unit = (delta_l / math.log(2.0)) / max(1, total_units)
    return {
        "k_dim": k_dim,
        "total_units": total_units,
        "total_cols": total_cols,
        "gap_opens": gap_opens,
        "total_excised": total_excised,
        "L0_nats": l0_nats,
        "L_msa_nats": comp_nats,
        "delta_L_KT_nats": delta_l,
        "bits_per_unit_saved": bits_per_unit,
    }


def build_cohorts() -> list[dict[str, Any]]:
    cohorts: list[dict[str, Any]] = []

    def add_nex_cohort(
        regime: str,
        cid: str,
        fn: str,
        squint_mode: str,
        slice_n: int | None = None,
    ) -> None:
        recs = parse_nex(HYPHY_DATA / fn)
        if slice_n is not None:
            recs = recs[:slice_n]
        pid = compute_pairwise_identity(recs, uppercase_only=False)
        ref_seqs = {k: v for k, v in recs}
        orig_seqs = {k: re.sub(r"[.\-]", "", v).upper() for k, v in recs}
        cohorts.append(
            {
                "regime": regime,
                "cohort_id": cid,
                "source": fn + (f"[:{slice_n}]" if slice_n else ""),
                "squint_mode": squint_mode,
                "eval_alphabet": "dna",
                "uppercase_core_only": False,
                "mean_pct_id": pid,
                "nseqs": len(recs),
                "ref_len": len(recs[0][1]),
                "ref_seqs": ref_seqs,
                "orig_seqs": orig_seqs,
                "order": [k for k, _ in recs],
            }
        )

    # E1: Ungapped Conserved Coding Genes (--mode codon)
    add_nex_cohort("E1_Ungapped_Coding", "E1_Lysozyme_19", "lysozyme.nex", "codon")
    add_nex_cohort("E1_Ungapped_Coding", "E1_ENCenv_23", "ENCenv.nex", "codon")
    add_nex_cohort("E1_Ungapped_Coding", "E1_Drosophila_Adh_23", "adh.nex", "codon")
    add_nex_cohort("E1_Ungapped_Coding", "E1_Mammal_ADORA3_67_IUPAC", "adora3.nex", "codon")
    add_nex_cohort("E1_Ungapped_Coding", "E1_HIV1_Vif_29", "HIVvif.nex", "codon")
    add_nex_cohort(
        "E1_Ungapped_Coding",
        "E1_Bacterial_PTS_16",
        "Bacterial_PTS_trehalose_transporter_suIII.nex",
        "codon",
    )

    # E2: Clean Biological Indels in Coding Genes (--mode codon)
    add_nex_cohort("E2_CleanIndels_Coding", "E2_SARSCoV2_Spike_VOC30", "SARS-CoV-2-spike.nex", "codon", slice_n=30)
    add_nex_cohort("E2_CleanIndels_Coding", "E2_Vertebrate_Rhodopsin_38", "yokoyama.rh1.cds.mod.1-990.nex", "codon")
    add_nex_cohort("E2_CleanIndels_Coding", "E2_HepatitisD_33", "HepatitisD.nex", "codon")
    add_nex_cohort("E2_CleanIndels_Coding", "E2_Mammal_vWF_62_IUPAC", "vwf.nex", "codon")
    add_nex_cohort("E2_CleanIndels_Coding", "E2_Mammal_RBP3_54", "rbp3.nex", "codon")

    # E3: High-Identity Structural Proteins (OXBench, --mode protein)
    ox_ids = [
        ("E3_OX_86s7", "__86s7"),
        ("E3_OX_620s7", "_620s7"),
        ("E3_OX_12s45", "_12s45"),
        ("E3_OX_22s35", "_22s35"),
        ("E3_OX_22t44", "_22t44"),
        ("E3_OX_12s74", "_12s74"),
        ("E3_OX_12s55", "_12s55"),
        ("E3_OX_136s16", "136s16"),
    ]
    with tarfile.open(ARCHIVE_PATH, "r:gz") as tf:
        for cid, ox_id in ox_ids:
            ref_txt = tf.extractfile(f"bench1.0/ox/ref/{ox_id}").read().decode("utf-8", errors="replace")
            in_txt = tf.extractfile(f"bench1.0/ox/in/{ox_id}").read().decode("utf-8", errors="replace")
            ref_dict = parse_fasta(ref_txt)
            in_dict = {k: v.upper() for k, v in parse_fasta(in_txt).items()}
            order = list(ref_dict.keys())
            for k in order:
                assert re.sub(r"[.\-]", "", ref_dict[k]).upper() == in_dict[k], (ox_id, k)
            pid = compute_pairwise_identity([(k, ref_dict[k]) for k in order], uppercase_only=True)
            cohorts.append(
                {
                    "regime": "E3_OXBench_Protein",
                    "cohort_id": cid,
                    "source": f"ox/{ox_id}",
                    "squint_mode": "protein",
                    "eval_alphabet": "protein",
                    "uppercase_core_only": True,
                    "mean_pct_id": pid,
                    "nseqs": len(order),
                    "ref_len": len(ref_dict[order[0]]),
                    "ref_seqs": ref_dict,
                    "orig_seqs": in_dict,
                    "order": order,
                }
            )

    # E4: High-Identity 4-Letter Nucleotide Alignments (--mode dna)
    add_nex_cohort("E4_Nucleotide_DNA", "E4_DNA_Lysozyme_19", "lysozyme.nex", "dna")
    add_nex_cohort("E4_Nucleotide_DNA", "E4_DNA_ENCenv_23", "ENCenv.nex", "dna")
    add_nex_cohort("E4_Nucleotide_DNA", "E4_DNA_Adh_23", "adh.nex", "dna")
    add_nex_cohort("E4_Nucleotide_DNA", "E4_DNA_Rhodopsin_38", "yokoyama.rh1.cds.mod.1-990.nex", "dna")
    add_nex_cohort("E4_Nucleotide_DNA", "E4_DNA_SARSCoV2_Spike_30", "SARS-CoV-2-spike.nex", "dna", slice_n=30)
    add_nex_cohort(
        "E4_Nucleotide_DNA",
        "E4_DNA_Bacterial_PTS_16",
        "Bacterial_PTS_trehalose_transporter_suIII.nex",
        "dna",
    )

    # E5: Clinical Amplicon & Partial-Fragment Tiling
    lyso = parse_nex(HYPHY_DATA / "lysozyme.nex")
    enc = parse_nex(HYPHY_DATA / "ENCenv.nex")

    def make_disjoint_halves(
        cid: str, src_fn: str, recs: list[tuple[str, str]], squint_mode: str
    ) -> None:
        l_seq = len(recs[0][1])
        half = l_seq // 2
        ref_seqs: dict[str, str] = {}
        orig_seqs: dict[str, str] = {}
        order: list[str] = []
        for idx, (k, s) in enumerate(recs):
            if idx % 3 == 0:
                ms = s
            elif idx % 3 == 1:
                ms = s[:half] + "-" * (l_seq - half)
            else:
                ms = "-" * half + s[half:]
            order.append(k)
            ref_seqs[k] = ms
            orig_seqs[k] = ms.replace("-", "").upper()
        pid = compute_pairwise_identity([(k, ref_seqs[k]) for k in order], uppercase_only=False)
        cohorts.append(
            {
                "regime": "E5_Amplicon_Tiling",
                "cohort_id": cid,
                "source": f"{src_fn} (full/5p/3p disjoint halves)",
                "squint_mode": squint_mode,
                "eval_alphabet": "dna",
                "uppercase_core_only": False,
                "mean_pct_id": pid,
                "nseqs": len(order),
                "ref_len": l_seq,
                "ref_seqs": ref_seqs,
                "orig_seqs": orig_seqs,
                "order": order,
            }
        )

    make_disjoint_halves("E5_Lysozyme_DisjointHalves_Codon", "lysozyme.nex", lyso, "codon")
    make_disjoint_halves("E5_ENCenv_DisjointHalves_Codon", "ENCenv.nex", enc, "codon")

    for cid, smode in [
        ("E5_ENCenv_3Tile_SteppingStone_Codon", "codon"),
        ("E5_ENCenv_3Tile_SteppingStone_DNA", "dna"),
    ]:
        l_seq = len(enc[0][1])
        ref_seqs = {}
        orig_seqs = {}
        order = []
        for idx, (k, s) in enumerate(enc):
            if idx % 3 == 0:
                ms = s[:630] + "-" * (l_seq - 630)
            elif idx % 3 == 1:
                ms = "-" * 420 + s[420:1080] + "-" * (l_seq - 1080)
            else:
                ms = "-" * 870 + s[870:]
            order.append(k)
            ref_seqs[k] = ms
            orig_seqs[k] = ms.replace("-", "").upper()
        pid = compute_pairwise_identity([(k, ref_seqs[k]) for k in order], uppercase_only=False)
        cohorts.append(
            {
                "regime": "E5_Amplicon_Tiling",
                "cohort_id": cid,
                "source": "ENCenv.nex (pure 3-tile stepping stone, 0 full-length)",
                "squint_mode": smode,
                "eval_alphabet": "dna",
                "uppercase_core_only": False,
                "mean_pct_id": pid,
                "nseqs": len(order),
                "ref_len": l_seq,
                "ref_seqs": ref_seqs,
                "orig_seqs": orig_seqs,
                "order": order,
            }
        )

    assert len(cohorts) == 29, f"Expected 29 cohorts, got {len(cohorts)}"
    return cohorts


def run_tool_on_cohort(
    tool: str,
    cohort: dict[str, Any],
    tmp: Path,
) -> tuple[list[dict[str, str]], float, dict[str, list[dict[str, Any]]], dict[str, Any]]:
    inp = tmp / "input.fa"
    if not inp.exists():
        inp.write_text("".join(f">{k}\n{cohort['orig_seqs'][k]}\n" for k in cohort["order"]))

    fs_by_seq: dict[str, list[dict[str, Any]]] = {}
    meta: dict[str, Any] = {
        "num_global_fine_cols": None,
        "num_clade_local_cols": None,
        "num_lineage_private_cols": None,
        "num_horizon_bubbles": None,
        "num_frameshifts": 0,
    }

    t0 = time.perf_counter()
    if tool == "squint":
        out_fa = tmp / "sq_out.fa"
        out_js = tmp / "sq_ledger.json"
        subprocess.run(
            [
                str(SQUINT_BIN),
                str(inp),
                "-o",
                str(out_fa),
                "-j",
                str(out_js),
                "--mode",
                cohort["squint_mode"],
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        rt = time.perf_counter() - t0
        led = json.loads(out_js.read_text())
        comps = [{sid: aln for sid, aln in c["aligned_fasta"]} for c in led["components"]]
        for c in led["components"]:
            for ev in c.get("frameshifts_repaired", []):
                fs_by_seq.setdefault(ev["seq_id"], []).append(ev)
        if len(led["components"]) == 1:
            c0 = led["components"][0]
            meta["num_global_fine_cols"] = c0["num_global_fine_cols"]
            meta["num_clade_local_cols"] = c0["num_clade_local_cols"]
            meta["num_lineage_private_cols"] = c0["num_lineage_private_cols"]
            meta["num_horizon_bubbles"] = len(c0.get("horizon_bubbles", []))
        else:
            meta["num_global_fine_cols"] = sum(c["num_global_fine_cols"] for c in led["components"])
            meta["num_clade_local_cols"] = sum(c["num_clade_local_cols"] for c in led["components"])
            meta["num_lineage_private_cols"] = sum(
                c["num_lineage_private_cols"] for c in led["components"]
            )
            meta["num_horizon_bubbles"] = sum(
                len(c.get("horizon_bubbles", [])) for c in led["components"]
            )
        meta["num_frameshifts"] = sum(
            len(c.get("frameshifts_repaired", [])) for c in led["components"]
        )
        return comps, rt, fs_by_seq, meta

    ext_inp = tmp / "ext_input.fa"
    alias_to_orig = {f"SEQ_{i:04d}": k for i, k in enumerate(cohort["order"])}
    if not ext_inp.exists():
        ext_inp.write_text(
            "".join(
                f">SEQ_{i:04d}\n{cohort['orig_seqs'][k]}\n"
                for i, k in enumerate(cohort["order"])
            )
        )

    def unalias(raw_map: dict[str, str]) -> dict[str, str]:
        return {alias_to_orig[ak]: v for ak, v in raw_map.items()}

    t0 = time.perf_counter()
    if tool == "mafft":
        proc = subprocess.run(
            [MAFFT_BIN, "--localpair", "--maxiterate", "1000", "--quiet", str(ext_inp)],
            capture_output=True,
            text=True,
            check=True,
        )
        rt = time.perf_counter() - t0
        return [unalias(parse_fasta(proc.stdout))], rt, fs_by_seq, meta

    if tool == "muscle":
        out_fa = tmp / "mu_out.fa"
        subprocess.run(
            [MUSCLE_BIN, "-in", str(ext_inp), "-out", str(out_fa), "-quiet"],
            capture_output=True,
            text=True,
            check=True,
        )
        rt = time.perf_counter() - t0
        return [unalias(parse_fasta(out_fa.read_text()))], rt, fs_by_seq, meta

    if tool == "clustalo":
        out_fa = tmp / "cl_out.fa"
        subprocess.run(
            [CLUSTALO_BIN, "-i", str(ext_inp), "-o", str(out_fa), "--threads=1", "--force"],
            capture_output=True,
            text=True,
            check=True,
        )
        rt = time.perf_counter() - t0
        return [unalias(parse_fasta(out_fa.read_text()))], rt, fs_by_seq, meta

    raise ValueError(f"Unknown tool: {tool}")


def summarize_results(rows: list[dict[str, Any]], kraft_audit: dict[str, float]) -> dict[str, Any]:
    regimes: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        regimes.setdefault(r["regime"], []).append(r)

    regime_summary: dict[str, Any] = {}
    for reg_name, reg_rows in regimes.items():
        by_tool: dict[str, list[dict[str, Any]]] = {}
        for r in reg_rows:
            by_tool.setdefault(r["tool"], []).append(r)
        t_sum: dict[str, Any] = {}
        for tool in TOOLS:
            tr = by_tool[tool]
            t_sum[tool] = {
                "n_cohorts": len(tr),
                "mean_Pair_Prec": statistics.mean(x["Pair_Prec"] for x in tr),
                "mean_Pair_Recall_Q": statistics.mean(x["Pair_Recall_Q"] for x in tr),
                "mean_Pair_F1": statistics.mean(x["Pair_F1"] for x in tr),
                "mean_Exact_Col_TC": statistics.mean(x["Exact_Col_TC"] for x in tr),
                "mean_All_Pair_Prec": statistics.mean(x["All_Pair_Prec"] for x in tr),
                "mean_All_Pair_Q": statistics.mean(x["All_Pair_Q"] for x in tr),
                "all_single_component": all(x["num_components"] == 1 for x in tr),
                "max_components": max(x["num_components"] for x in tr),
                "pooled_delta_L_KT_nats": sum(x["delta_L_KT_nats"] for x in tr),
                "mean_bits_per_unit_saved": statistics.mean(x["bits_per_unit_saved"] for x in tr),
                "total_gap_opens": sum(x["gap_opens"] for x in tr),
                "total_runtime_s": sum(x["runtime_s"] for x in tr),
                "mean_runtime_s": statistics.mean(x["runtime_s"] for x in tr),
            }
        regime_summary[reg_name] = t_sum

    overall: dict[str, Any] = {}
    for tool in TOOLS:
        tr = [r for r in rows if r["tool"] == tool]
        overall[tool] = {
            "n_cohorts": len(tr),
            "mean_Pair_Prec": statistics.mean(x["Pair_Prec"] for x in tr),
            "mean_Pair_Recall_Q": statistics.mean(x["Pair_Recall_Q"] for x in tr),
            "mean_Pair_F1": statistics.mean(x["Pair_F1"] for x in tr),
            "mean_Exact_Col_TC": statistics.mean(x["Exact_Col_TC"] for x in tr),
            "all_single_component": all(x["num_components"] == 1 for x in tr),
            "pooled_delta_L_KT_nats": sum(x["delta_L_KT_nats"] for x in tr),
            "total_gap_opens": sum(x["gap_opens"] for x in tr),
            "total_runtime_s": sum(x["runtime_s"] for x in tr),
        }

    return {
        "kraft_self_check": kraft_audit,
        "n_cohorts_total": len(rows) // len(TOOLS),
        "by_regime": regime_summary,
        "overall_29_cohorts": overall,
    }


def main() -> int:
    t_start = time.perf_counter()
    kraft_audit = verify_kraft_identities()
    cohorts = build_cohorts()
    rows: list[dict[str, Any]] = []

    for idx, cohort in enumerate(cohorts, 1):
        ref_info = extract_ref_structures(cohort["ref_seqs"], cohort["uppercase_core_only"])
        print(
            f"[{idx:02d}/{len(cohorts)}] {cohort['regime']:22s} | {cohort['cohort_id']:36s} "
            f"(N={cohort['nseqs']:2d}, ID={cohort['mean_pct_id']:5.1f}%, mode={cohort['squint_mode']})",
            flush=True,
        )
        with tempfile.TemporaryDirectory(prefix=f"easy_{cohort['cohort_id']}_") as td:
            tmp = Path(td)
            for tool in TOOLS:
                comps, rt, fs_by_seq, meta = run_tool_on_cohort(tool, cohort, tmp)
                ev = score_alignment(cohort["orig_seqs"], ref_info, comps, fs_by_seq)
                mdl = score_easy_kt_mdl(
                    cohort["orig_seqs"], comps, cohort["eval_alphabet"], ev["excised_by_seq"]
                )
                row = {
                    "regime": cohort["regime"],
                    "cohort_id": cohort["cohort_id"],
                    "source": cohort["source"],
                    "squint_mode": cohort["squint_mode"],
                    "eval_alphabet": cohort["eval_alphabet"],
                    "nseqs": cohort["nseqs"],
                    "ref_len": cohort["ref_len"],
                    "mean_pct_id": round(cohort["mean_pct_id"], 2),
                    "tool": tool,
                    "runtime_s": round(rt, 4),
                    "num_components": len(comps),
                    "total_cols": mdl["total_cols"],
                    "Pair_Prec": round(ev["Pair_Prec"], 6),
                    "Pair_Recall_Q": round(ev["Pair_Recall_Q"], 6),
                    "Pair_F1": round(ev["Pair_F1"], 6),
                    "Exact_Col_TC": round(ev["Exact_Col_TC"], 6),
                    "All_Pair_Prec": round(ev["All_Pair_Prec"], 6),
                    "All_Pair_Q": round(ev["All_Pair_Q"], 6),
                    "All_Pair_F1": round(ev["All_Pair_F1"], 6),
                    "All_Exact_Col_TC": round(ev["All_Exact_Col_TC"], 6),
                    "L0_nats": round(mdl["L0_nats"], 3),
                    "L_msa_nats": round(mdl["L_msa_nats"], 3),
                    "delta_L_KT_nats": round(mdl["delta_L_KT_nats"], 3),
                    "bits_per_unit_saved": round(mdl["bits_per_unit_saved"], 5),
                    "gap_opens": mdl["gap_opens"],
                    "total_excised": mdl["total_excised"],
                    "num_frameshifts": meta["num_frameshifts"],
                    "num_global_fine_cols": meta["num_global_fine_cols"],
                    "num_clade_local_cols": meta["num_clade_local_cols"],
                    "num_lineage_private_cols": meta["num_lineage_private_cols"],
                    "num_horizon_bubbles": meta["num_horizon_bubbles"],
                }
                rows.append(row)
                print(
                    f"    {tool:8s}: K={len(comps)} cols={mdl['total_cols']:4d} | "
                    f"Prec={ev['Pair_Prec']:.4f} Q={ev['Pair_Recall_Q']:.4f} "
                    f"F1={ev['Pair_F1']:.4f} TC={ev['Exact_Col_TC']:.4f} | "
                    f"dL={mdl['delta_L_KT_nats']:9.1f} | gaps={mdl['gap_opens']:4d} | {rt:5.2f}s",
                    flush=True,
                )

    fieldnames = list(rows[0].keys())
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    summary = summarize_results(rows, kraft_audit)
    OUT_JSON.write_text(json.dumps(summary, indent=2))

    print("\n=== REGIME SUMMARY ===")
    for reg_name, t_sum in summary["by_regime"].items():
        print(f"\n[{reg_name}]")
        for tool in TOOLS:
            d = t_sum[tool]
            print(
                f"  {tool:8s}: Prec={d['mean_Pair_Prec']:.4f} | Q={d['mean_Pair_Recall_Q']:.4f} | "
                f"F1={d['mean_Pair_F1']:.4f} | TC={d['mean_Exact_Col_TC']:.4f} | "
                f"single_K={d['all_single_component']} | dL_KT={d['pooled_delta_L_KT_nats']:10.1f} | "
                f"gaps={d['total_gap_opens']:5d} | rt={d['total_runtime_s']:6.2f}s"
            )

    print("\n=== OVERALL 29 EASY COHORTS ===")
    for tool in TOOLS:
        d = summary["overall_29_cohorts"][tool]
        print(
            f"  {tool:8s}: Prec={d['mean_Pair_Prec']:.4f} | Q={d['mean_Pair_Recall_Q']:.4f} | "
            f"F1={d['mean_Pair_F1']:.4f} | TC={d['mean_Exact_Col_TC']:.4f} | "
            f"single_K={d['all_single_component']} | dL_KT={d['pooled_delta_L_KT_nats']:10.1f} | "
            f"gaps={d['total_gap_opens']:5d} | rt={d['total_runtime_s']:6.2f}s"
        )
    print(f"\nWrote {OUT_CSV} and {OUT_JSON} in {time.perf_counter() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
