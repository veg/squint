#!/usr/bin/env python3
"""Standard MSA benchmark harness comparing SQUINT against MAFFT, MUSCLE, and Clustal Omega.

Reads datasets directly from benchmarks/external_data/bench.tar.gz without unpacking to disk:
  - bali3pdb   (180 datasets, protein)
  - bali3_full (84 full-length BB11/BB30/BB50 datasets from bali3, protein)
  - sabre_twi  (108 SABmark Twilight twi_* datasets from sabre, protein)
  - bali2dnaf  (141 frameshifted coding DNA datasets, codon mode for squint)

Outputs:
  - benchmarks/standard_msa_per_dataset.csv
  - benchmarks/standard_msa_summary.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
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
ARCHIVE_PATH = REPO_ROOT / "benchmarks" / "external_data" / "bench.tar.gz"
SQUINT_BIN = REPO_ROOT / "target" / "release" / "squint"
DEFAULT_CSV = REPO_ROOT / "benchmarks" / "standard_msa_per_dataset.csv"
DEFAULT_JSON = REPO_ROOT / "benchmarks" / "standard_msa_summary.json"

TOOLS = ("squint", "mafft", "muscle", "clustalo")
TIMEOUT_S = 600.0


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


def parse_info_map(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        s = line.strip()
        if not s or ";" not in s:
            continue
        k, v = s.split(";", 1)
        out[k.strip()] = v.strip()
    return out


def pct_id_bin(pct: float) -> str:
    if pct < 15.0:
        return "<15%"
    if pct < 20.0:
        return "15-20%"
    if pct < 30.0:
        return "20-30%"
    if pct < 40.0:
        return "30-40%"
    return ">=40%"


def dataset_category(suite: str, did: str) -> str:
    if suite in ("bali3pdb", "bali3_full"):
        return did[:4]
    if suite == "sabre_twi":
        return "twi"
    return "bali2dnaf"


def banded_nw_map(orig: str, ug: str) -> list[int]:
    """Map each index j in ug to monotonic index i in orig via banded global alignment."""
    a = orig.upper().replace("U", "T")
    b = ug.upper().replace("U", "T")
    n, m = len(a), len(b)
    if m == 0:
        return []
    if m == n and a == b:
        return list(range(n))

    band = abs(n - m) + 24
    neg_inf = -10**9
    gap = -2

    # Store traceback over band [max(0, j + offset_lo) .. min(n, j + offset_hi)]
    # Centered around i = j * n / m when m > 0, plus generous margin.
    dp_rows: list[dict[int, int]] = []
    bt_rows: list[dict[int, int]] = []  # 0: diag, 1: up (skip orig[i-1]), 2: left (skip ug[j-1])

    row0: dict[int, int] = {}
    bt0: dict[int, int] = {}
    hi0 = min(n, band)
    for i in range(hi0 + 1):
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

    for j in range(m):
        if mapping[j] < 0 or mapping[j] >= n:
            raise RuntimeError(f"unmapped ug residue at j={j} (n={n}, m={m})")
        if j > 0 and mapping[j] <= mapping[j - 1]:
            raise RuntimeError(f"non-monotonic mapping at j={j}: {mapping[j-1]} -> {mapping[j]}")
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
        if len(rem) >= len(ug) and "".join(orig_u[i] for i in rem[: len(ug)]) == ug_u:
            return rem[: len(ug)]

    if len(orig) >= len(ug) and orig_u[: len(ug)] == ug_u:
        return list(range(len(ug)))

    return banded_nw_map(orig, ug)


def extract_ref_structures(
    ref_seqs: dict[str, str], is_bali2dnaf: bool
) -> tuple[
    dict[str, list[int]],
    dict[str, list[bool]],
    list[int],
    int,
    dict[str, Any] | None,
]:
    sids = sorted(ref_seqs.keys())
    Lref = len(next(iter(ref_seqs.values())))
    ug_len = {s: sum(1 for ch in ref_seqs[s] if ch not in ".-!") for s in sids}
    rc_of = {s: [-1] * ug_len[s] for s in sids}
    is_core = {s: [False] * ug_len[s] for s in sids}
    ref_col_sizes: list[int] = []
    n_ref_pairs = 0
    pos = {s: 0 for s in sids}
    bang_cols: list[int] = []

    for c in range(Lref):
        upper_pos: list[tuple[str, int]] = []
        has_bang = False
        for s in sids:
            ch = ref_seqs[s][c]
            if ch == "!":
                has_bang = True
            if ch not in ".-!":
                p = pos[s]
                pos[s] += 1
                if ch.isupper():
                    is_core[s][p] = True
                    upper_pos.append((s, p))
        if has_bang:
            bang_cols.append(c)
        k = len(upper_pos)
        if k >= 2:
            r_id = len(ref_col_sizes)
            ref_col_sizes.append(k)
            n_ref_pairs += k * (k - 1) // 2
            for s, p in upper_pos:
                rc_of[s][p] = r_id

    fs_truth: dict[str, Any] | None = None
    if is_bali2dnaf:
        fs_cands = [s for s in sids if not any(ref_seqs[s][c] == "!" for c in bang_cols)]
        if len(fs_cands) != 1 or len(bang_cols) not in (1, 2):
            raise RuntimeError(f"unexpected bali2dnaf truth: bang_cols={bang_cols}, fs_cands={fs_cands}")
        fs_seq = fs_cands[0]
        true_nt_pos = sum(1 for ch in ref_seqs[fs_seq][: bang_cols[0]] if ch not in ".-!")
        fs_truth = {
            "true_fs_seq": fs_seq,
            "true_fs_nt_pos": true_nt_pos,
            "true_phase_shift": len(bang_cols),
        }

    return rc_of, is_core, ref_col_sizes, n_ref_pairs, fs_truth


def score_alignment(
    orig_seqs: dict[str, str],
    rc_of: dict[str, list[int]],
    is_core: dict[str, list[bool]],
    ref_col_sizes: list[int],
    n_ref_pairs: int,
    components_alns: list[dict[str, str]],
    fs_by_seq: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    seen_sids: set[str] = set()
    tp_core = 0
    n_test_both_core = 0
    n_test_all_pairs = 0
    matched_rc: set[int] = set()

    for aln in components_alns:
        if not aln:
            continue
        asids = sorted(aln.keys())
        for s in asids:
            if s in seen_sids:
                raise RuntimeError(f"duplicate sequence {s} across components")
            seen_sids.add(s)

        La = len(next(iter(aln.values())))
        pos_map: dict[str, list[int]] = {}
        apos = {s: 0 for s in asids}
        for s in asids:
            if len(aln[s]) != La:
                raise RuntimeError(f"jagged aligned sequence {s}: {len(aln[s])} != {La}")
            ug = re.sub(r"[.\-!]", "", aln[s])
            evs = fs_by_seq.get(s, []) if fs_by_seq else []
            pos_map[s] = map_subseq_to_orig(orig_seqs[s], ug, evs)

        if len(asids) < 2:
            continue

        for c in range(La):
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
                counts: Counter[int] = Counter()
                for s, p in col_items:
                    r_id = rc_of[s][p]
                    if r_id != -1:
                        counts[r_id] += 1
                for r_id, cnt in counts.items():
                    if cnt >= 2:
                        tp_core += cnt * (cnt - 1) // 2
                    if cnt == kt and kt == ref_col_sizes[r_id]:
                        matched_rc.add(r_id)

    if seen_sids != set(orig_seqs.keys()):
        raise RuntimeError(f"missing sequences in output: expected {len(orig_seqs)}, got {len(seen_sids)}")

    n_ref_core_cols = len(ref_col_sizes)
    q = tp_core / n_ref_pairs if n_ref_pairs > 0 else None
    prec_core = tp_core / n_test_both_core if n_test_both_core > 0 else None
    f1_core = (
        (2.0 * tp_core / (n_ref_pairs + n_test_both_core))
        if (n_ref_pairs + n_test_both_core) > 0
        else None
    )
    tc = len(matched_rc) / n_ref_core_cols if n_ref_core_cols > 0 else None
    over_aln = (n_test_all_pairs - tp_core) / n_test_all_pairs if n_test_all_pairs > 0 else None

    return {
        "tp_core": tp_core,
        "n_ref_pairs": n_ref_pairs,
        "n_test_both_core": n_test_both_core,
        "n_test_all_pairs": n_test_all_pairs,
        "tc_matched": len(matched_rc),
        "n_ref_core_cols": n_ref_core_cols,
        "Q": q,
        "Prec_core": prec_core,
        "F1_core": f1_core,
        "TC": tc,
        "OverAln": over_aln,
    }


def run_dataset_task(task: tuple[Any, ...]) -> list[dict[str, Any]]:
    (suite, did, mode, category, nseqs, max_len, pct_id, raw_in_bytes, ref_text) = task
    orig_seqs = parse_fasta(raw_in_bytes.decode("utf-8", errors="replace"))
    ref_seqs = parse_fasta(ref_text)
    rc_of, is_core, ref_col_sizes, n_ref_pairs, fs_truth = extract_ref_structures(
        ref_seqs, is_bali2dnaf=(suite == "bali2dnaf")
    )

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix=f"msa_{suite}_{did}_") as tmpdir:
        tmp = Path(tmpdir)
        inp_path = tmp / f"{did}.fa"
        inp_path.write_bytes(raw_in_bytes)

        for tool in TOOLS:
            base: dict[str, Any] = {
                "suite": suite,
                "dataset_id": did,
                "category": category,
                "mode": mode,
                "nseqs": nseqs,
                "max_len": max_len,
                "pct_id": pct_id,
                "pct_id_bin": pct_id_bin(pct_id),
                "tool": tool,
                "status": "ok",
                "error": "",
                "runtime_s": None,
                "tp_core": 0,
                "n_ref_pairs": n_ref_pairs,
                "n_test_both_core": 0,
                "n_test_all_pairs": 0,
                "tc_matched": 0,
                "n_ref_core_cols": len(ref_col_sizes),
                "Q": None,
                "Prec_core": None,
                "F1_core": None,
                "TC": None,
                "OverAln": None,
                "num_components": 1,
                "unified": 1,
                "num_global_fine_cols": None,
                "num_clade_local_cols": None,
                "num_lineage_private_cols": None,
                "num_horizon_bubbles": None,
                "net_cohort_gain_nats": None,
                "num_fs_repaired": None,
                "true_fs_seq": fs_truth["true_fs_seq"] if fs_truth else "",
                "true_fs_nt_pos": fs_truth["true_fs_nt_pos"] if fs_truth else None,
                "true_phase_shift": fs_truth["true_phase_shift"] if fs_truth else None,
                "fs_tp_seq": None,
                "fs_exact_seq_phase": None,
                "fs_fp_events": None,
                "fs_nt_err": None,
            }

            t0 = time.perf_counter()
            try:
                if tool == "squint":
                    out_fa = tmp / "squint.fa"
                    out_json = tmp / "squint.json"
                    cmd = [
                        str(SQUINT_BIN),
                        str(inp_path),
                        "-o",
                        str(out_fa),
                        "-j",
                        str(out_json),
                        "--mode",
                        mode,
                        "-T",
                        "1",
                    ]
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
                    elapsed = time.perf_counter() - t0
                    base["runtime_s"] = elapsed
                    if proc.returncode != 0:
                        base["status"] = "error"
                        base["error"] = f"rc={proc.returncode}: {proc.stderr.strip()[:240]}"
                        rows.append(base)
                        continue
                    ledger = json.loads(out_json.read_text())
                    comps = ledger["components"]
                    comp_alns = [{sid: aln for sid, aln in c["aligned_fasta"]} for c in comps]
                    fs_by_seq: dict[str, list[dict[str, Any]]] = {}
                    all_fs: list[dict[str, Any]] = []
                    for c in comps:
                        for ev in c.get("frameshifts_repaired", []):
                            fs_by_seq.setdefault(ev["seq_id"], []).append(ev)
                            all_fs.append(ev)

                    scores = score_alignment(
                        orig_seqs,
                        rc_of,
                        is_core,
                        ref_col_sizes,
                        n_ref_pairs,
                        comp_alns,
                        fs_by_seq=fs_by_seq,
                    )
                    base.update(scores)
                    k_comp = int(ledger["num_homology_components"])
                    base["num_components"] = k_comp
                    base["unified"] = 1 if k_comp == 1 else 0
                    base["num_global_fine_cols"] = sum(int(c["num_global_fine_cols"]) for c in comps)
                    base["num_clade_local_cols"] = sum(int(c["num_clade_local_cols"]) for c in comps)
                    base["num_lineage_private_cols"] = sum(
                        int(c["num_lineage_private_cols"]) for c in comps
                    )
                    base["num_horizon_bubbles"] = sum(len(c.get("horizon_bubbles", [])) for c in comps)
                    base["net_cohort_gain_nats"] = float(ledger["net_cohort_gain_nats"])
                    base["num_fs_repaired"] = len(all_fs)

                    if fs_truth is not None:
                        t_seq = fs_truth["true_fs_seq"]
                        t_pos = int(fs_truth["true_fs_nt_pos"])
                        t_phase = int(fs_truth["true_phase_shift"])
                        hits = [ev for ev in all_fs if ev["seq_id"] == t_seq]
                        base["fs_tp_seq"] = 1 if hits else 0
                        base["fs_exact_seq_phase"] = (
                            1 if any(int(ev["phase_shift"]) == t_phase for ev in hits) else 0
                        )
                        base["fs_fp_events"] = len(all_fs) - (1 if hits else 0)
                        base["fs_nt_err"] = (
                            min(abs(int(ev["nt_pos"]) - t_pos) for ev in hits) if hits else None
                        )

                elif tool == "mafft":
                    cmd = ["mafft", "--quiet", "--thread", "1", "--localpair", "--maxiterate", "1000", str(inp_path)]
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
                    base["runtime_s"] = time.perf_counter() - t0
                    if proc.returncode != 0:
                        base["status"] = "error"
                        base["error"] = f"rc={proc.returncode}: {proc.stderr.strip()[:240]}"
                        rows.append(base)
                        continue
                    aln = parse_fasta(proc.stdout)
                    base.update(
                        score_alignment(
                            orig_seqs, rc_of, is_core, ref_col_sizes, n_ref_pairs, [aln]
                        )
                    )

                elif tool == "muscle":
                    out_fa = tmp / "muscle.fa"
                    cmd = ["muscle", "-in", str(inp_path), "-out", str(out_fa), "-quiet"]
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
                    base["runtime_s"] = time.perf_counter() - t0
                    if proc.returncode != 0:
                        base["status"] = "error"
                        base["error"] = f"rc={proc.returncode}: {proc.stderr.strip()[:240]}"
                        rows.append(base)
                        continue
                    aln = parse_fasta(out_fa.read_text())
                    base.update(
                        score_alignment(
                            orig_seqs, rc_of, is_core, ref_col_sizes, n_ref_pairs, [aln]
                        )
                    )

                elif tool == "clustalo":
                    out_fa = tmp / "clustalo.fa"
                    cmd = ["clustalo", "-i", str(inp_path), "-o", str(out_fa), "--threads=1", "--force"]
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
                    base["runtime_s"] = time.perf_counter() - t0
                    if proc.returncode != 0:
                        base["status"] = "error"
                        base["error"] = f"rc={proc.returncode}: {proc.stderr.strip()[:240]}"
                        rows.append(base)
                        continue
                    aln = parse_fasta(out_fa.read_text())
                    base.update(
                        score_alignment(
                            orig_seqs, rc_of, is_core, ref_col_sizes, n_ref_pairs, [aln]
                        )
                    )

            except subprocess.TimeoutExpired:
                base["runtime_s"] = time.perf_counter() - t0
                base["status"] = "timeout"
                base["error"] = f"exceeded {TIMEOUT_S}s"
            except Exception as exc:
                base["runtime_s"] = time.perf_counter() - t0
                base["status"] = "error"
                base["error"] = str(exc)[:240]

            rows.append(base)

    return rows


def load_tasks(limit: int | None = None) -> list[tuple[Any, ...]]:
    specs = [
        ("bali3pdb", "bali3pdb", "protein", None),
        ("bali3_full", "bali3", "protein", ("BB11", "BB30", "BB50")),
        ("sabre_twi", "sabre", "protein", ("twi_",)),
        ("bali2dnaf", "bali2dnaf", "codon", None),
    ]
    tasks: list[tuple[Any, ...]] = []
    with tarfile.open(ARCHIVE_PATH, "r:gz") as tf:
        for suite, arc_dir, mode, prefixes in specs:
            ids_raw = tf.extractfile(f"bench1.0/{arc_dir}/info/ids.txt").read().decode()
            ids = [ln.strip() for ln in ids_raw.splitlines() if ln.strip()]
            if prefixes is not None:
                ids = [i for i in ids if i.startswith(prefixes)]
            nseqs_map = parse_info_map(tf.extractfile(f"bench1.0/{arc_dir}/info/nseqs.txt").read().decode())
            pct_map = parse_info_map(tf.extractfile(f"bench1.0/{arc_dir}/info/pctids.txt").read().decode())

            for did in ids:
                raw_in_bytes = tf.extractfile(f"bench1.0/{arc_dir}/in/{did}").read()
                ref_text = tf.extractfile(f"bench1.0/{arc_dir}/ref/{did}").read().decode()
                orig = parse_fasta(raw_in_bytes.decode("utf-8", errors="replace"))
                nseqs = int(nseqs_map[did])
                max_len = max(len(v) for v in orig.values())
                pct_id = float(pct_map[did])
                cat = dataset_category(suite, did)
                tasks.append(
                    (suite, did, mode, cat, nseqs, max_len, pct_id, raw_in_bytes, ref_text)
                )

    # LPT ordering: longest processing time (proxied by quadratic effective DP cost) first.
    tasks.sort(
        key=lambda t: (
            t[4] * ((t[5] // 3 if t[2] == "codon" else t[5]) ** 2),
            t[4],
            t[5],
        ),
        reverse=True,
    )
    if limit is not None:
        tasks = tasks[:limit]
    return tasks


def safe_mean(vals: list[float]) -> float | None:
    return statistics.mean(vals) if vals else None


def safe_median(vals: list[float]) -> float | None:
    return statistics.median(vals) if vals else None


def quantile(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    return s[lo] * (hi - pos) + s[hi] * (pos - lo)


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for tool in TOOLS:
        trows = [r for r in rows if r["tool"] == tool]
        ok_rows = [r for r in trows if r["status"] == "ok"]
        q_vals = [float(r["Q"]) for r in ok_rows if r["Q"] is not None]
        prec_vals = [float(r["Prec_core"]) for r in ok_rows if r["Prec_core"] is not None]
        f1_vals = [float(r["F1_core"]) for r in ok_rows if r["F1_core"] is not None]
        tc_vals = [float(r["TC"]) for r in ok_rows if r["TC"] is not None]
        over_vals = [float(r["OverAln"]) for r in ok_rows if r["OverAln"] is not None]
        rt_vals = [float(r["runtime_s"]) for r in ok_rows if r["runtime_s"] is not None]

        sum_tp = sum(int(r["tp_core"]) for r in ok_rows)
        sum_ref = sum(int(r["n_ref_pairs"]) for r in ok_rows)
        sum_tbc = sum(int(r["n_test_both_core"]) for r in ok_rows)
        sum_tall = sum(int(r["n_test_all_pairs"]) for r in ok_rows)

        d: dict[str, Any] = {
            "n_datasets": len(trows),
            "n_ok": len(ok_rows),
            "n_fail": len(trows) - len(ok_rows),
            "n_eval_prec_core": len(prec_vals),
            "pct_eval_prec_core": (100.0 * len(prec_vals) / len(ok_rows)) if ok_rows else None,
            "n_eval_over_aln": len(over_vals),
            "mean_Q": safe_mean(q_vals),
            "median_Q": safe_median(q_vals),
            "pooled_Q": (sum_tp / sum_ref) if sum_ref > 0 else None,
            "pooled_Prec_core": (sum_tp / sum_tbc) if sum_tbc > 0 else None,
            "mean_Prec_core_cond": safe_mean(prec_vals),
            "median_Prec_core_cond": safe_median(prec_vals),
            "mean_F1_core": safe_mean(f1_vals),
            "pooled_F1_core": (2.0 * sum_tp / (sum_ref + sum_tbc)) if (sum_ref + sum_tbc) > 0 else None,
            "mean_TC": safe_mean(tc_vals),
            "pooled_OverAln": ((sum_tall - sum_tp) / sum_tall) if sum_tall > 0 else None,
            "mean_OverAln_cond": safe_mean(over_vals),
            "mean_runtime_s": safe_mean(rt_vals),
            "median_runtime_s": safe_median(rt_vals),
            "max_runtime_s": max(rt_vals) if rt_vals else None,
            "sum_runtime_s": sum(rt_vals),
        }

        if tool == "squint":
            unified_vals = [int(r["unified"]) for r in ok_rows]
            comp_vals = [int(r["num_components"]) for r in ok_rows]
            gain_vals = [float(r["net_cohort_gain_nats"]) for r in ok_rows if r["net_cohort_gain_nats"] is not None]
            bub_vals = [int(r["num_horizon_bubbles"]) for r in ok_rows if r["num_horizon_bubbles"] is not None]
            gcols = [int(r["num_global_fine_cols"]) for r in ok_rows if r["num_global_fine_cols"] is not None]
            ccols = [int(r["num_clade_local_cols"]) for r in ok_rows if r["num_clade_local_cols"] is not None]
            lcols = [int(r["num_lineage_private_cols"]) for r in ok_rows if r["num_lineage_private_cols"] is not None]
            d.update(
                {
                    "n_unified_K1": sum(unified_vals),
                    "pct_unified_K1": (100.0 * sum(unified_vals) / len(unified_vals)) if unified_vals else None,
                    "mean_components": safe_mean([float(x) for x in comp_vals]),
                    "max_components": max(comp_vals) if comp_vals else None,
                    "mean_net_gain_nats": safe_mean(gain_vals),
                    "sum_horizon_bubbles": sum(bub_vals),
                    "mean_global_fine_cols": safe_mean([float(x) for x in gcols]),
                    "mean_clade_local_cols": safe_mean([float(x) for x in ccols]),
                    "mean_lineage_private_cols": safe_mean([float(x) for x in lcols]),
                }
            )
        out[tool] = d
    return out


def summarize_bali2dnaf_frameshifts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hrows = [r for r in rows if r["suite"] == "bali2dnaf" and r["tool"] == "squint" and r["status"] == "ok"]
    n = len(hrows)
    if n == 0:
        return {"n_datasets": 0}
    tp_seq = sum(int(r["fs_tp_seq"]) for r in hrows)
    exact_phase = sum(int(r["fs_exact_seq_phase"]) for r in hrows)
    pred_events = sum(int(r["num_fs_repaired"]) for r in hrows)
    fp_events = sum(int(r["fs_fp_events"]) for r in hrows)
    zero_rep = sum(1 for r in hrows if int(r["num_fs_repaired"]) == 0)
    errs = [float(r["fs_nt_err"]) for r in hrows if r["fs_nt_err"] is not None]

    det_dids = {r["dataset_id"] for r in hrows if int(r["fs_tp_seq"]) == 1}
    miss_dids = {r["dataset_id"] for r in hrows if int(r["fs_tp_seq"]) == 0}
    det_rows = [r for r in rows if r["suite"] == "bali2dnaf" and r["dataset_id"] in det_dids]
    miss_rows = [r for r in rows if r["suite"] == "bali2dnaf" and r["dataset_id"] in miss_dids]

    by_phase: dict[str, Any] = {}
    for p in (1, 2):
        prows = [r for r in hrows if int(r["true_phase_shift"]) == p]
        p_tp = sum(int(r["fs_tp_seq"]) for r in prows)
        p_ex = sum(int(r["fs_exact_seq_phase"]) for r in prows)
        p_errs = [float(r["fs_nt_err"]) for r in prows if r["fs_nt_err"] is not None]
        by_phase[str(p)] = {
            "n": len(prows),
            "tp_seq": p_tp,
            "seq_recall": p_tp / len(prows) if prows else None,
            "exact_seq_phase": p_ex,
            "exact_seq_phase_rate": p_ex / len(prows) if prows else None,
            "median_nt_err": safe_median(p_errs),
        }

    return {
        "n_datasets": n,
        "fs_detected_seq_count": tp_seq,
        "fs_seq_recall": tp_seq / n,
        "fs_exact_seq_phase_count": exact_phase,
        "fs_exact_seq_phase_rate": exact_phase / n,
        "fs_total_pred_events": pred_events,
        "fs_fp_events": fp_events,
        "fs_event_precision": (tp_seq / pred_events) if pred_events > 0 else None,
        "fs_zero_repaired_count": zero_rep,
        "fs_nt_err_n": len(errs),
        "fs_nt_err_mean": safe_mean(errs),
        "fs_nt_err_median": safe_median(errs),
        "fs_nt_err_p25": quantile(errs, 0.25),
        "fs_nt_err_p75": quantile(errs, 0.75),
        "fs_nt_err_max": max(errs) if errs else None,
        "fs_within_0nt": sum(1 for e in errs if e == 0),
        "fs_within_3nt": sum(1 for e in errs if e <= 3),
        "fs_within_15nt": sum(1 for e in errs if e <= 15),
        "fs_within_30nt": sum(1 for e in errs if e <= 30),
        "by_true_phase_shift": by_phase,
        "subset_fs_detected": summarize_group(det_rows),
        "subset_fs_missed": summarize_group(miss_rows),
    }


def build_summary(rows: list[dict[str, Any]], wall_s: float) -> dict[str, Any]:
    suites = ("bali3pdb", "bali3_full", "sabre_twi", "bali2dnaf")
    pct_bins = ("<15%", "15-20%", "20-30%", "30-40%", ">=40%")
    prot_rows = [r for r in rows if r["suite"] in ("bali3pdb", "bali3_full", "sabre_twi")]

    summary: dict[str, Any] = {
        "meta": {
            "total_datasets": len({(r["suite"], r["dataset_id"]) for r in rows}),
            "total_runs": len(rows),
            "failed_or_timeout_runs": sum(1 for r in rows if r["status"] != "ok"),
            "wall_time_s": wall_s,
            "timeout_limit_s": TIMEOUT_S,
        },
        "by_suite": {s: summarize_group([r for r in rows if r["suite"] == s]) for s in suites},
        "by_category_bali3pdb": {
            cat: summarize_group([r for r in rows if r["suite"] == "bali3pdb" and r["category"] == cat])
            for cat in ("BB11", "BB12", "BB20", "BB30", "BB40", "BB50")
        },
        "by_category_bali3_full": {
            cat: summarize_group([r for r in rows if r["suite"] == "bali3_full" and r["category"] == cat])
            for cat in ("BB11", "BB30", "BB50")
        },
        "by_pct_id_protein_all": {
            b: summarize_group([r for r in prot_rows if r["pct_id_bin"] == b]) for b in pct_bins
        },
        "by_pct_id_per_suite": {
            s: {b: summarize_group([r for r in rows if r["suite"] == s and r["pct_id_bin"] == b]) for b in pct_bins}
            for s in suites
        },
        "bali2dnaf_frameshift_audit": summarize_bali2dnaf_frameshifts(rows),
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    args = parser.parse_args()

    t_start = time.perf_counter()
    tasks = load_tasks(limit=args.limit)
    print(f"Loaded {len(tasks)} datasets; running {len(tasks) * len(TOOLS)} MSA invocations across {args.workers} workers.", flush=True)

    all_rows: list[dict[str, Any]] = []
    done = 0
    failures = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_dataset_task, t) for t in tasks]
        for fut in concurrent.futures.as_completed(futures):
            ds_rows = fut.result()
            all_rows.extend(ds_rows)
            done += 1
            for r in ds_rows:
                if r["status"] != "ok":
                    failures += 1
                    print(
                        f"  [FAIL] {r['suite']}/{r['dataset_id']} tool={r['tool']} status={r['status']} err={r['error']}",
                        flush=True,
                    )
            if done % 25 == 0 or done == len(tasks):
                elapsed = time.perf_counter() - t_start
                print(
                    f"  Progress: {done}/{len(tasks)} datasets ({done * 4}/{len(tasks) * 4} runs) | elapsed={elapsed:.1f}s | failures={failures}",
                    flush=True,
                )

    suite_order = {"bali3pdb": 0, "bali3_full": 1, "sabre_twi": 2, "bali2dnaf": 3}
    tool_order = {t: i for i, t in enumerate(TOOLS)}
    all_rows.sort(key=lambda r: (suite_order[r["suite"]], r["dataset_id"], tool_order[r["tool"]]))

    fieldnames = list(all_rows[0].keys())
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    wall_s = time.perf_counter() - t_start
    summary = build_summary(all_rows, wall_s)
    args.json_out.write_text(json.dumps(summary, indent=2))
    print(f"Completed {len(tasks)} datasets in {wall_s:.2f}s (failures={failures}).", flush=True)
    print(f"Wrote {args.csv_out} and {args.json_out}.", flush=True)
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
