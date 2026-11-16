#!/usr/bin/env python3
"""
SQUINT Scaling & Speed Benchmark (Suite A: 6 HyPhy empirical codon cohorts;
Suite B: IQ-TREE 2 AliSim N = 100..5,000 phylogeny scaling).

Measures true wall-clock runtime, peak RSS (via /usr/bin/time -l on macOS),
output size (.fa vs .hdb), exact combinatoric accuracy (Pair Prec, Recall Q,
F1, Exact Column TC), homology component count K, and repaired frameshifts.
"""

from __future__ import annotations

import csv
import json
import math
import os
import pathlib
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BENCH_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = BENCH_DIR.parent
sys.path.insert(0, str(BENCH_DIR))

from run_easy_msa_benchmark import (  # noqa: E402
    extract_ref_structures,
    parse_fasta,
    parse_nex,
    score_alignment,
)

SQUINT = os.environ.get("SQUINT_BIN", str(ROOT_DIR / "target" / "release" / "squint"))
IQTREE2 = shutil.which("iqtree2") or os.environ.get("IQTREE2_BIN", "iqtree2")
MAFFT = shutil.which("mafft") or os.environ.get("MAFFT_BIN", "mafft")
MUSCLE = shutil.which("muscle") or os.environ.get("MUSCLE_BIN", "muscle")
CLUSTALO = shutil.which("clustalo") or os.environ.get("CLUSTALO_BIN", "clustalo")
HYPHY_DIR = pathlib.Path(
    os.environ.get("HYPHY_DATA_DIR", str(BENCH_DIR / "external_data" / "hyphy_data"))
)
ARTIFACT_DIR = (
    pathlib.Path(os.environ["SQUINT_ARTIFACT_DIR"])
    if os.environ.get("SQUINT_ARTIFACT_DIR")
    else None
)

PER_RUN_TIMEOUT_S = 90.0
SCALE_NS = [100, 250, 500, 1000, 2500, 5000]

SUITE_A_COHORTS: list[tuple[str, float]] = [
    ("camelid", 2.27),
    ("InfluenzaA", 9.22),
    ("HIV_RT", 12.20),
    ("SARS-CoV-2-spike", 18.38),
    ("rbcL", 25.36),
    ("IAV-human-H1N1-HA", 26.55),
]


def assert_scorer_matches_bruteforce() -> None:
    """Verify O(N * L) score_alignment equals brute-force all-pairs enumeration."""
    ref_seqs = {
        "A01": "ACDEF-GHIKLM",
        "A02": "ACDEFGGHIK-M",
        "A03": "AC-EFGGHIKLM",
        "A04": "ACDEFGGHI-LM",
        "A05": "ACDEF-GHIKLM",
        "A06": "ACDE--GHIKLM",
        "A07": "BCDEFGGHIKLM",
        "B01": "MNOPQR-STVW-",
        "B02": "MNOPQRSTVW--",
        "B03": "MN-PQRSTVWY-",
        "B04": "MNOP-RSTVWY-",
        "B05": "MNOPQRSTVWY-",
    }
    test_comp_a = {
        "A01": "ACDEFGHIKLM-",
        "A02": "ACDEFGGHIKM-",
        "A03": "AC-EFGGHIKLM",
        "A04": "ACDEFGGHI-LM",
        "A05": "ACDEF-GHIKLM",
        "A06": "ACDE--GHIKLM",
        "A07": "BCDEFGGHIKLM",
    }
    test_comp_b = {
        "B01": "MNOPQR-STVW-",
        "B02": "MNOPQRSTVW--",
        "B03": "MN-PQRSTVWY-",
        "B04": "MNOP-RSTVWY-",
        "B05": "MNOPQRSTVWY-",
    }
    orig_seqs = {k: v.replace("-", "") for k, v in ref_seqs.items()}
    ref_info = extract_ref_structures(ref_seqs, False)
    fast = score_alignment(orig_seqs, ref_info, [test_comp_a, test_comp_b], None)

    # Brute-force reference pairs and columns
    sids = sorted(ref_seqs.keys())
    pos_r = {s: 0 for s in sids}
    ref_pairs: set[tuple[str, int, str, int]] = set()
    ref_cols: set[tuple[tuple[str, int], ...]] = set()
    ref_l = len(next(iter(ref_seqs.values())))
    for c in range(ref_l):
        col: list[tuple[str, int]] = []
        for s in sids:
            if ref_seqs[s][c] != "-":
                col.append((s, pos_r[s]))
                pos_r[s] += 1
        if len(col) >= 2:
            ref_cols.add(tuple(col))
            for i in range(len(col)):
                for j in range(i + 1, len(col)):
                    ref_pairs.add((col[i][0], col[i][1], col[j][0], col[j][1]))

    test_pairs: set[tuple[str, int, str, int]] = set()
    test_cols: set[tuple[tuple[str, int], ...]] = set()
    for aln in [test_comp_a, test_comp_b]:
        asids = sorted(aln.keys())
        pos_t = {s: 0 for s in asids}
        la = len(next(iter(aln.values())))
        for c in range(la):
            col_t: list[tuple[str, int]] = []
            for s in asids:
                if aln[s][c] != "-":
                    col_t.append((s, pos_t[s]))
                    pos_t[s] += 1
            if len(col_t) >= 2:
                test_cols.add(tuple(col_t))
                for i in range(len(col_t)):
                    for j in range(i + 1, len(col_t)):
                        test_pairs.add((col_t[i][0], col_t[i][1], col_t[j][0], col_t[j][1]))

    tp = len(ref_pairs & test_pairs)
    bf_prec = tp / len(test_pairs)
    bf_q = tp / len(ref_pairs)
    bf_f1 = 2.0 * tp / (len(ref_pairs) + len(test_pairs))
    bf_tc = len(ref_cols & test_cols) / len(ref_cols)
    assert abs(fast["Pair_Prec"] - bf_prec) < 1e-15
    assert abs(fast["Pair_Recall_Q"] - bf_q) < 1e-15
    assert abs(fast["Pair_F1"] - bf_f1) < 1e-15
    assert abs(fast["Exact_Col_TC"] - bf_tc) < 1e-15


def run_timed_cmd(
    cmd: list[str],
    timeout_s: float = PER_RUN_TIMEOUT_S,
    stdout_path: pathlib.Path | None = None,
) -> dict[str, Any]:
    """Execute command under `/usr/bin/time -l`, capturing wall time and peak RSS."""
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
            "peak_footprint_mib": None,
            "stderr": "",
        }
    finally:
        if out_fh is not subprocess.DEVNULL:
            out_fh.close()

    m_rss = re.search(r"(\d+)\s+maximum resident set size", stderr_txt)
    peak_rss_mib = (int(m_rss.group(1)) / (1024.0 * 1024.0)) if m_rss else None
    m_fp = re.search(r"(\d+)\s+peak memory footprint", stderr_txt)
    peak_fp_mib = (int(m_fp.group(1)) / (1024.0 * 1024.0)) if m_fp else None
    if proc.returncode != 0:
        return {
            "status": f"ERROR_{proc.returncode}",
            "elapsed_s": round(elapsed_s, 4),
            "peak_rss_mib": round(peak_rss_mib, 2) if peak_rss_mib is not None else None,
            "peak_footprint_mib": round(peak_fp_mib, 2) if peak_fp_mib is not None else None,
            "stderr": stderr_txt[-400:],
        }
    return {
        "status": "OK",
        "elapsed_s": round(elapsed_s, 4),
        "peak_rss_mib": round(peak_rss_mib, 2) if peak_rss_mib is not None else None,
        "peak_footprint_mib": round(peak_fp_mib, 2) if peak_fp_mib is not None else None,
        "stderr": stderr_txt,
    }


def parse_squint_components_fasta(path: pathlib.Path) -> list[dict[str, str]]:
    """Parse SQUINT FASTA output, splitting `>ID component=K` headers cleanly."""
    by_comp: dict[int, dict[str, str]] = {}
    cur_comp = 0
    cur_id: str | None = None
    buf: list[str] = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith(">"):
            if cur_id is not None:
                by_comp.setdefault(cur_comp, {})[cur_id] = "".join(buf)
            hdr = s[1:].strip()
            parts = hdr.split()
            cur_id = parts[0]
            m_c = re.search(r"\bcomponent=(\d+)\b", hdr)
            cur_comp = int(m_c.group(1)) if m_c else 0
            buf = []
        else:
            buf.append(s)
    if cur_id is not None:
        by_comp.setdefault(cur_comp, {})[cur_id] = "".join(buf)
    return [by_comp[k] for k in sorted(by_comp.keys())]


def parse_squint_frameshifts(stderr_txt: str) -> tuple[dict[str, list[dict[str, Any]]], int, int]:
    """Extract repaired frameshift coords and reported component count K from SQUINT stderr."""
    fs_by_seq: dict[str, list[dict[str, Any]]] = {}
    fs_count = 0
    for m in re.finditer(
        r"Repaired Frameshift in '([^']+)': nt=(\d+) .* phase=\+(\d+)", stderr_txt
    ):
        sid = m.group(1)
        fs_by_seq.setdefault(sid, []).append(
            {"nt_pos": int(m.group(2)), "phase_shift": int(m.group(3))}
        )
        fs_count += 1
    m_k = re.search(r"\bcomponents=(\d+)\b", stderr_txt)
    k_comps = int(m_k.group(1)) if m_k else 1
    return fs_by_seq, fs_count, k_comps


def write_aliased_fasta(orig_seqs: dict[str, str], out_path: pathlib.Path) -> dict[str, str]:
    """Write FASTA with short `SEQ_{i:05d}` headers to avoid ClustalO 127-char limit."""
    alias_to_orig: dict[str, str] = {}
    with open(out_path, "w") as fh:
        for i, sid in enumerate(sorted(orig_seqs.keys())):
            alias = f"SEQ_{i:05d}"
            alias_to_orig[alias] = sid
            fh.write(f">{alias}\n{orig_seqs[sid]}\n")
    return alias_to_orig


def restore_aliased_fasta(path: pathlib.Path, alias_to_orig: dict[str, str]) -> dict[str, str]:
    raw = parse_fasta(path.read_text())
    return {alias_to_orig[k]: v for k, v in raw.items()}


def eval_tool_on_cohort(
    suite: str,
    dataset: str,
    n_taxa: int,
    ref_len: int,
    tool: str,
    in_squint: pathlib.Path,
    in_aliased: pathlib.Path,
    alias_to_orig: dict[str, str],
    orig_seqs: dict[str, str],
    ref_info: dict[str, Any],
    mode: str,
    work_dir: pathlib.Path,
    baseline_s: float | None = None,
) -> dict[str, Any]:
    """Run a single aligner on one dataset and return a structured result row."""
    row: dict[str, Any] = {
        "suite": suite,
        "dataset": dataset,
        "N": n_taxa,
        "ref_L": ref_len,
        "tool": tool,
        "status": "OK",
        "elapsed_s": None,
        "peak_rss_mib": None,
        "peak_footprint_mib": None,
        "output_kib": None,
        "components_K": 1,
        "frameshifts": 0,
        "Pair_Prec": None,
        "Pair_Recall_Q": None,
        "Pair_F1": None,
        "Exact_Col_TC": None,
        "baseline_s": baseline_s,
    }

    if tool in ("squint_codon_fa", "squint_dna_fa", "squint_fa"):
        out_fa = work_dir / f"{dataset}.{tool}.fa"
        res = run_timed_cmd([SQUINT, str(in_squint), "-o", str(out_fa), "--mode", mode])
        row["status"] = res["status"]
        row["elapsed_s"] = res["elapsed_s"]
        row["peak_rss_mib"] = res["peak_rss_mib"]
        row["peak_footprint_mib"] = res["peak_footprint_mib"]
        if res["status"] != "OK":
            return row
        row["output_kib"] = round(out_fa.stat().st_size / 1024.0, 2)
        fs_by_seq, fs_cnt, k_comps = parse_squint_frameshifts(res["stderr"])
        comps = parse_squint_components_fasta(out_fa)
        sc = score_alignment(orig_seqs, ref_info, comps, fs_by_seq if mode == "codon" else None)
        row["components_K"] = k_comps
        row["frameshifts"] = fs_cnt
        row["Pair_Prec"] = round(sc["Pair_Prec"], 4)
        row["Pair_Recall_Q"] = round(sc["Pair_Recall_Q"], 4)
        row["Pair_F1"] = round(sc["Pair_F1"], 4)
        row["Exact_Col_TC"] = round(sc["Exact_Col_TC"], 4)
        return row

    if tool in ("squint_codon_hdb", "squint_hdb"):
        out_hdb = work_dir / f"{dataset}.{tool}.hdb"
        res = run_timed_cmd([SQUINT, str(in_squint), "-o", str(out_hdb), "--mode", mode])
        row["status"] = res["status"]
        row["elapsed_s"] = res["elapsed_s"]
        row["peak_rss_mib"] = res["peak_rss_mib"]
        row["peak_footprint_mib"] = res["peak_footprint_mib"]
        if res["status"] != "OK":
            return row
        row["output_kib"] = round(out_hdb.stat().st_size / 1024.0, 2)
        fs_by_seq, fs_cnt, k_comps = parse_squint_frameshifts(res["stderr"])
        rt_fa = work_dir / f"{dataset}.{tool}.rt.fa"
        subprocess.run(
            [SQUINT, "convert", str(out_hdb), "-o", str(rt_fa)],
            capture_output=True,
            text=True,
            check=True,
        )
        comps = parse_squint_components_fasta(rt_fa)
        sc = score_alignment(orig_seqs, ref_info, comps, fs_by_seq if mode == "codon" else None)
        row["components_K"] = k_comps
        row["frameshifts"] = fs_cnt
        row["Pair_Prec"] = round(sc["Pair_Prec"], 4)
        row["Pair_Recall_Q"] = round(sc["Pair_Recall_Q"], 4)
        row["Pair_F1"] = round(sc["Pair_F1"], 4)
        row["Exact_Col_TC"] = round(sc["Exact_Col_TC"], 4)
        return row

    out_aliased = work_dir / f"{dataset}.{tool}.aliased.fa"
    if tool == "mafft_auto":
        res = run_timed_cmd(
            [MAFFT, "--auto", "--thread", "-1", "--quiet", str(in_aliased)],
            stdout_path=out_aliased,
        )
    elif tool == "mafft_linsi":
        res = run_timed_cmd(
            [
                MAFFT,
                "--localpair",
                "--maxiterate",
                "1000",
                "--thread",
                "-1",
                "--quiet",
                str(in_aliased),
            ],
            stdout_path=out_aliased,
        )
    elif tool == "muscle":
        res = run_timed_cmd(
            [MUSCLE, "-in", str(in_aliased), "-out", str(out_aliased), "-quiet"]
        )
    elif tool == "clustalo":
        res = run_timed_cmd(
            [
                CLUSTALO,
                "-i",
                str(in_aliased),
                "-o",
                str(out_aliased),
                "--threads=1",
                "--force",
            ]
        )
    else:
        raise ValueError(f"Unknown tool: {tool}")

    row["status"] = res["status"]
    row["elapsed_s"] = res["elapsed_s"]
    row["peak_rss_mib"] = res["peak_rss_mib"]
    row["peak_footprint_mib"] = res["peak_footprint_mib"]
    if res["status"] != "OK":
        return row
    row["output_kib"] = round(out_aliased.stat().st_size / 1024.0, 2)
    aln = restore_aliased_fasta(out_aliased, alias_to_orig)
    sc = score_alignment(orig_seqs, ref_info, [aln], None)
    row["Pair_Prec"] = round(sc["Pair_Prec"], 4)
    row["Pair_Recall_Q"] = round(sc["Pair_Recall_Q"], 4)
    row["Pair_F1"] = round(sc["Pair_F1"], 4)
    row["Exact_Col_TC"] = round(sc["Exact_Col_TC"], 4)
    return row


def make_yule_newick(n: int, target_height: float, seed: int, prefix: str = "T") -> str:
    """Deterministic near-ultrametric Yule phylogeny scaled to exact root height."""
    rng = random.Random(seed)
    nodes = [(f"{prefix}{i:05d}", 0.0) for i in range(n)]
    times = [0.0]
    cur_t = 0.0
    for k in range(n, 1, -1):
        cur_t += rng.expovariate(k)
        times.append(cur_t)
    scale = target_height / max(cur_t, 1e-12)
    cur_nodes = list(nodes)
    for step in range(1, n):
        h = times[step] * scale
        i = rng.randrange(len(cur_nodes))
        n1, h1 = cur_nodes.pop(i)
        j = rng.randrange(len(cur_nodes))
        n2, h2 = cur_nodes.pop(j)
        b1 = max(1e-6, (h - h1) * rng.uniform(0.85, 1.15))
        b2 = max(1e-6, (h - h2) * rng.uniform(0.85, 1.15))
        cur_nodes.append((f"({n1}:{b1:.6f},{n2}:{b2:.6f})", h))
    return cur_nodes[0][0] + ";"


def make_s2_newick(n: int, seed: int) -> str:
    """Deterministic 4-clade divergent phylogeny at root height 0.24."""
    sizes = [n // 4 + (1 if c < n % 4 else 0) for c in range(4)]
    off = 0
    clades: list[str] = []
    for c, sz in enumerate(sizes):
        sub = make_yule_newick(sz, 0.08, seed + 100 + c, prefix="X")[:-1]
        for local_i in range(sz - 1, -1, -1):
            sub = sub.replace(f"X{local_i:05d}", f"T{off + local_i:05d}")
        off += sz
        clades.append(sub)
    return (
        f"(({clades[0]}:0.080000,{clades[1]}:0.080000):0.080000,"
        f"({clades[2]}:0.080000,{clades[3]}:0.080000):0.080000);"
    )


def generate_alisim_dataset(
    scenario: str,
    n_taxa: int,
    seed: int,
    work_dir: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path, dict[str, str], dict[str, str], dict[str, Any], int]:
    """Generate aligned ground-truth and unaligned FASTA via IQ-TREE 2 AliSim."""
    if scenario == "S1_Surveillance_Clade_AA300":
        nwk = make_yule_newick(n_taxa, 0.04, seed)
        length = 300
    elif scenario == "S2_Divergent_MultiClade_AA250":
        nwk = make_s2_newick(n_taxa, seed)
        length = 250
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    tree_path = work_dir / f"{scenario}_N{n_taxa}.nwk"
    tree_path.write_text(nwk)
    pref = work_dir / f"{scenario}_N{n_taxa}_sim"
    subprocess.run(
        [
            IQTREE2,
            "--alisim",
            str(pref),
            "-t",
            str(tree_path),
            "--seqtype",
            "AA",
            "-m",
            "LG+G4",
            "--indel",
            "0.03,0.03",
            "--indel-size",
            "POW{1.7/20},POW{1.7/20}",
            "--length",
            str(length),
            "-af",
            "fasta",
            "--seed",
            str(seed),
            "--keep-ident",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    ref_seqs = parse_fasta((work_dir / f"{scenario}_N{n_taxa}_sim.fa").read_text())
    ref_len = len(next(iter(ref_seqs.values())))
    orig_seqs = {k: re.sub(r"[.\-?!]", "", v) for k, v in ref_seqs.items()}
    in_fa = work_dir / f"{scenario}_N{n_taxa}.unaligned.fa"
    with open(in_fa, "w") as fh:
        for sid in sorted(orig_seqs.keys()):
            fh.write(f">{sid}\n{orig_seqs[sid]}\n")
    ref_info = extract_ref_structures(ref_seqs, False)
    return in_fa, tree_path, ref_seqs, orig_seqs, ref_info, ref_len


def build_plot(rows: list[dict[str, Any]], plot_path: pathlib.Path) -> None:
    """Create 2x2 scaling benchmark visualization from measured CSV rows."""
    colors = {
        "squint_fa": "#0072B2",
        "squint_hdb": "#56B4E9",
        "mafft_auto": "#009E73",
        "mafft_linsi": "#E69F00",
        "muscle": "#D55E00",
        "clustalo": "#CC79A7",
    }
    labels = {
        "squint_fa": "SQUINT (.fa)",
        "squint_hdb": "SQUINT (.hdb)",
        "mafft_auto": "MAFFT --auto",
        "mafft_linsi": "MAFFT L-INS-i",
        "muscle": "MUSCLE v3.8",
        "clustalo": "ClustalO (1T)",
    }
    markers = {
        "squint_fa": "o",
        "squint_hdb": "s",
        "mafft_auto": "^",
        "mafft_linsi": "v",
        "muscle": "D",
        "clustalo": "P",
    }

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), dpi=160)

    # Panel A & B: log-log runtime scaling on S1 and S2
    for ax, scen, title in [
        (axes[0, 0], "S1_Surveillance_Clade_AA300", "(A) S1 Tight Clade (L=300 AA, h=0.04): Runtime"),
        (axes[0, 1], "S2_Divergent_MultiClade_AA250", "(B) S2 4-Clade Div. (L=250 AA, h=0.24): Runtime"),
    ]:
        for tool in ["squint_fa", "squint_hdb", "mafft_auto", "mafft_linsi", "muscle", "clustalo"]:
            sub = [
                r
                for r in rows
                if r["suite"] == "SuiteB"
                and r["dataset"] == scen
                and r["tool"] == tool
                and r["status"] == "OK"
            ]
            sub.sort(key=lambda r: int(r["N"]))
            if sub:
                xs = [int(r["N"]) for r in sub]
                ys = [float(r["elapsed_s"]) for r in sub]
                ax.loglog(
                    xs,
                    ys,
                    marker=markers[tool],
                    color=colors[tool],
                    linewidth=2.0 if tool.startswith("squint") else 1.5,
                    label=labels[tool],
                )
            to_rows = [
                r
                for r in rows
                if r["suite"] == "SuiteB"
                and r["dataset"] == scen
                and r["tool"] == tool
                and r["status"] == "TIMEOUT"
            ]
            if to_rows:
                ax.scatter(
                    [int(to_rows[0]["N"])],
                    [PER_RUN_TIMEOUT_S],
                    marker="x",
                    s=90,
                    color=colors[tool],
                    zorder=5,
                )
        ax.axhline(PER_RUN_TIMEOUT_S, color="#777777", linestyle="--", linewidth=1.0, label="90s Timeout")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(SCALE_NS)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("Number of Taxa N")
        ax.set_ylabel("Wall-Clock Runtime (s)")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, which="both", alpha=0.25)
        if scen == "S1_Surveillance_Clade_AA300":
            ax.legend(fontsize=8, loc="upper left", ncol=2)

    # Panel C: Peak RSS vs N on S1 & S2 (plus N=5000 size comparison note)
    ax_c = axes[1, 0]
    for tool in ["squint_fa", "squint_hdb", "mafft_auto", "muscle", "clustalo"]:
        for scen, ls, tag in [
            ("S1_Surveillance_Clade_AA300", "-", "S1"),
            ("S2_Divergent_MultiClade_AA250", "--", "S2"),
        ]:
            sub = [
                r
                for r in rows
                if r["suite"] == "SuiteB"
                and r["dataset"] == scen
                and r["tool"] == tool
                and r["status"] == "OK"
                and r["peak_rss_mib"] is not None
            ]
            sub.sort(key=lambda r: int(r["N"]))
            if sub:
                xs = [int(r["N"]) for r in sub]
                ys = [float(r["peak_rss_mib"]) for r in sub]
                lbl = f"{labels[tool]} ({tag})" if tool in ("squint_fa", "squint_hdb") or tag == "S1" else None
                ax_c.plot(
                    xs,
                    ys,
                    marker=markers[tool],
                    linestyle=ls,
                    color=colors[tool],
                    linewidth=1.8 if tool.startswith("squint") else 1.2,
                    alpha=0.95 if tag == "S1" else 0.75,
                    label=lbl,
                )
    ax_c.set_xscale("log")
    ax_c.set_xticks(SCALE_NS)
    ax_c.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_c.set_xlabel("Number of Taxa N")
    ax_c.set_ylabel("Peak RSS Memory (MiB)")
    ax_c.set_title("(C) Peak RSS Memory Scaling (S1 solid, S2 dashed)", fontsize=11, fontweight="bold")
    ax_c.grid(True, which="both", alpha=0.25)
    ax_c.legend(fontsize=7.5, loc="upper left", ncol=2)

    # Annotate N=5000 S1 .fa vs .hdb disk size inside Panel C
    fa_5k = next(
        (
            r
            for r in rows
            if r["suite"] == "SuiteB"
            and r["dataset"] == "S1_Surveillance_Clade_AA300"
            and int(r["N"]) == 5000
            and r["tool"] == "squint_fa"
            and r["status"] == "OK"
        ),
        None,
    )
    hdb_5k = next(
        (
            r
            for r in rows
            if r["suite"] == "SuiteB"
            and r["dataset"] == "S1_Surveillance_Clade_AA300"
            and int(r["N"]) == 5000
            and r["tool"] == "squint_hdb"
            and r["status"] == "OK"
        ),
        None,
    )
    if fa_5k and hdb_5k and fa_5k["output_kib"] and hdb_5k["output_kib"]:
        fa_mib = float(fa_5k["output_kib"]) / 1024.0
        hdb_mib = float(hdb_5k["output_kib"]) / 1024.0
        if hdb_mib >= fa_mib:
            size_note = f"{hdb_mib / fa_mib:.1f}x larger (full DB)"
        else:
            size_note = f"{fa_mib / hdb_mib:.1f}x smaller"
        ax_c.text(
            0.97,
            0.06,
            f"S1 N=5,000 size:\n.fa={fa_mib:.2f} MiB\n.hdb={hdb_mib:.2f} MiB ({size_note})",
            transform=ax_c.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F4F8FB", "edgecolor": "#0072B2", "alpha": 0.9},
        )

    # Panel D: Suite B accuracy (Pair F1 solid, Exact Col TC dashed) vs N on S2 + Suite A speedup summary
    ax_d = axes[1, 1]
    for tool in ["squint_fa", "mafft_auto", "mafft_linsi", "muscle", "clustalo"]:
        sub = [
            r
            for r in rows
            if r["suite"] == "SuiteB"
            and r["dataset"] == "S2_Divergent_MultiClade_AA250"
            and r["tool"] == tool
            and r["status"] == "OK"
            and r["Pair_F1"] is not None
        ]
        sub.sort(key=lambda r: int(r["N"]))
        if sub:
            xs = [int(r["N"]) for r in sub]
            f1s = [float(r["Pair_F1"]) for r in sub]
            tcs = [float(r["Exact_Col_TC"]) for r in sub]
            ax_d.plot(
                xs,
                f1s,
                marker=markers[tool],
                linestyle="-",
                color=colors[tool],
                linewidth=2.0 if tool == "squint_fa" else 1.4,
                label=f"{labels[tool]} (F1)",
            )
            ax_d.plot(
                xs,
                tcs,
                marker=markers[tool],
                linestyle="--",
                color=colors[tool],
                linewidth=1.5 if tool == "squint_fa" else 1.0,
                alpha=0.75,
                label=f"{labels[tool]} (TC)" if tool == "squint_fa" else None,
            )

    suite_a_tot_base = sum(b for _, b in SUITE_A_COHORTS)
    suite_a_tot_codon = sum(
        float(r["elapsed_s"])
        for r in rows
        if r["suite"] == "SuiteA" and r["tool"] == "squint_codon_fa" and r["status"] == "OK"
    )
    suite_a_tot_dna = sum(
        float(r["elapsed_s"])
        for r in rows
        if r["suite"] == "SuiteA" and r["tool"] == "squint_dna_fa" and r["status"] == "OK"
    )
    if suite_a_tot_codon > 0:
        spd = suite_a_tot_base / suite_a_tot_codon
        ax_d.text(
            0.03,
            0.05,
            f"Suite A (6 HyPhy Cohorts):\nBase={suite_a_tot_base:.1f}s -> Codon={suite_a_tot_codon:.1f}s ({spd:.1f}x)\nSQUINT DNA={suite_a_tot_dna:.1f}s",
            transform=ax_d.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F4F8FB", "edgecolor": "#0072B2", "alpha": 0.9},
        )
    ax_d.set_xscale("log")
    ax_d.set_xticks(SCALE_NS)
    ax_d.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_d.set_ylim(-0.02, 1.04)
    ax_d.set_xlabel("Number of Taxa N")
    ax_d.set_ylabel("Accuracy Score (0–1)")
    ax_d.set_title("(D) S2 Accuracy (Pair F1 solid, Exact TC dashed) & Suite A Speedup", fontsize=11, fontweight="bold")
    ax_d.grid(True, which="both", alpha=0.25)
    ax_d.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(0.01, 0.45), ncol=1)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def main() -> None:
    assert_scorer_matches_bruteforce()
    print("[init] Brute-force combinatoric scorer check passed (< 1e-15).", flush=True)

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as td_str:
        td = pathlib.Path(td_str)

        # Suite A: 6 Empirical HyPhy Codon Cohorts
        suite_a_dir = td / "suite_a"
        suite_a_dir.mkdir(parents=True, exist_ok=True)
        for ds_name, base_s in SUITE_A_COHORTS:
            nex_path = HYPHY_DIR / f"{ds_name}.nex"
            ref_seqs = dict(parse_nex(nex_path))
            ref_len = len(next(iter(ref_seqs.values())))
            orig_seqs = {k: re.sub(r"[.\-?!]", "", v) for k, v in ref_seqs.items()}
            ref_info = extract_ref_structures(ref_seqs, False)

            ug_fa = suite_a_dir / f"{ds_name}.unaligned.fa"
            with open(ug_fa, "w") as fh:
                for sid in sorted(orig_seqs.keys()):
                    fh.write(f">{sid}\n{orig_seqs[sid]}\n")
            aliased_fa = suite_a_dir / f"{ds_name}.aliased.fa"
            alias_to_orig = write_aliased_fasta(orig_seqs, aliased_fa)

            for tool in [
                "squint_codon_fa",
                "squint_codon_hdb",
                "squint_dna_fa",
                "mafft_auto",
                "mafft_linsi",
                "muscle",
                "clustalo",
            ]:
                if tool in ("squint_codon_fa", "squint_codon_hdb"):
                    in_sq = nex_path
                    mode = "codon"
                else:
                    in_sq = ug_fa
                    mode = "dna"
                r = eval_tool_on_cohort(
                    "SuiteA",
                    ds_name,
                    len(orig_seqs),
                    ref_len,
                    tool,
                    in_sq,
                    aliased_fa,
                    alias_to_orig,
                    orig_seqs,
                    ref_info,
                    mode,
                    suite_a_dir,
                    baseline_s=base_s if tool == "squint_codon_fa" else None,
                )
                rows.append(r)
                print(
                    f"[SuiteA] {ds_name:18s} {tool:17s} status={r['status']:8s} "
                    f"t={r['elapsed_s']}s rss={r['peak_rss_mib']}MiB "
                    f"fp={r['peak_footprint_mib']}MiB "
                    f"F1={r['Pair_F1']} TC={r['Exact_Col_TC']} fs={r['frameshifts']}",
                    flush=True,
                )

        # Suite B: IQ-TREE 2 AliSim Large-N Scaling (N = 100..5000)
        suite_b_dir = td / "suite_b"
        suite_b_dir.mkdir(parents=True, exist_ok=True)
        for scen, seed_base in [
            ("S1_Surveillance_Clade_AA300", 1000),
            ("S2_Divergent_MultiClade_AA250", 2000),
        ]:
            timed_out: set[str] = set()
            for n_taxa in SCALE_NS:
                in_fa, _, _, orig_seqs, ref_info, ref_len = generate_alisim_dataset(
                    scen, n_taxa, seed_base + n_taxa, suite_b_dir
                )
                aliased_fa = suite_b_dir / f"{scen}_N{n_taxa}.aliased.fa"
                alias_to_orig = write_aliased_fasta(orig_seqs, aliased_fa)
                for tool in [
                    "squint_fa",
                    "squint_hdb",
                    "mafft_auto",
                    "mafft_linsi",
                    "muscle",
                    "clustalo",
                ]:
                    if tool in timed_out:
                        r = {
                            "suite": "SuiteB",
                            "dataset": scen,
                            "N": n_taxa,
                            "ref_L": ref_len,
                            "tool": tool,
                            "status": "SKIPPED_AFTER_TIMEOUT",
                            "elapsed_s": None,
                            "peak_rss_mib": None,
                            "peak_footprint_mib": None,
                            "output_kib": None,
                            "components_K": None,
                            "frameshifts": None,
                            "Pair_Prec": None,
                            "Pair_Recall_Q": None,
                            "Pair_F1": None,
                            "Exact_Col_TC": None,
                            "baseline_s": None,
                        }
                        rows.append(r)
                        print(
                            f"[SuiteB] {scen:29s} N={n_taxa:4d} {tool:12s} SKIPPED_AFTER_TIMEOUT",
                            flush=True,
                        )
                        continue
                    r = eval_tool_on_cohort(
                        "SuiteB",
                        scen,
                        n_taxa,
                        ref_len,
                        tool,
                        in_fa,
                        aliased_fa,
                        alias_to_orig,
                        orig_seqs,
                        ref_info,
                        "protein",
                        suite_b_dir,
                    )
                    rows.append(r)
                    if r["status"] == "TIMEOUT":
                        timed_out.add(tool)
                    print(
                        f"[SuiteB] {scen:29s} N={n_taxa:4d} ref_L={ref_len:4d} {tool:12s} "
                        f"status={r['status']:8s} t={r['elapsed_s']}s rss={r['peak_rss_mib']}MiB "
                        f"fp={r['peak_footprint_mib']}MiB kib={r['output_kib']} "
                        f"F1={r['Pair_F1']} TC={r['Exact_Col_TC']}",
                        flush=True,
                    )

    # Save CSV
    csv_path = BENCH_DIR / "scaling_benchmark_results.csv"
    fieldnames = [
        "suite",
        "dataset",
        "N",
        "ref_L",
        "tool",
        "status",
        "elapsed_s",
        "peak_rss_mib",
        "peak_footprint_mib",
        "output_kib",
        "components_K",
        "frameshifts",
        "Pair_Prec",
        "Pair_Recall_Q",
        "Pair_F1",
        "Exact_Col_TC",
        "baseline_s",
    ]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Save Summary JSON
    suite_a_tot_base = round(sum(b for _, b in SUITE_A_COHORTS), 4)
    suite_a_tot_codon = round(
        sum(
            float(r["elapsed_s"])
            for r in rows
            if r["suite"] == "SuiteA" and r["tool"] == "squint_codon_fa" and r["status"] == "OK"
        ),
        4,
    )
    summary = {
        "suite_a_total_baseline_s": suite_a_tot_base,
        "suite_a_total_squint_codon_fa_s": suite_a_tot_codon,
        "suite_a_speedup": round(suite_a_tot_base / max(suite_a_tot_codon, 1e-6), 2),
        "rows": rows,
    }
    json_path = BENCH_DIR / "scaling_benchmark_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    # Build and copy Plot PNG
    plot_path = BENCH_DIR / "scaling_benchmark_plot.png"
    build_plot(rows, plot_path)
    if ARTIFACT_DIR is not None:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(plot_path, ARTIFACT_DIR / "scaling_benchmark_plot.png")
    print(f"[done] Wrote {csv_path}, {json_path}, and {plot_path}")


if __name__ == "__main__":
    main()
