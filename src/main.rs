use std::env;
use std::fs;
use std::path::Path;
use std::process;
use std::time::Instant;

use squint::alphabet::{parse_sequences_str, SeqMode};
use squint::evaluate_against_reference_msa;
use squint::multiseq::analyze_unaligned_cohort;
use squint::storage::{
    inspect_archive_summary, is_hdb_file, read_archive_auto, subset_archive, subset_hdb_file,
    write_flat_fasta_file, write_hdb_file, write_hnex_file, HierarchicalArchive, SubsetSpec,
};

fn print_usage() {
    eprintln!(
        "SQUINT v0.1.0 — Scale-Quotiented Unaligned-Interval Network Transducer\n\
         Usage:\n  \
           squint <input.fa|input.nex> [-o out.fa|out.hnex|out.hdb] [--hnex out.hnex] [--hdb out.hdb] [-j ledger.json] [--mode auto|codon|protein|dna] [-T threads] [--eval-ref]\n  \
           squint convert <in.hnex|in.hdb> -o <out.hdb|out.hnex|out.fa>\n  \
           squint subset  <in.hdb|in.hnex> -o <out.hnex|out.hdb|out.fa> [--component id] [--clade node] [--taxa id1,id2] [--region seq:start-end] [--scale all|global-core]\n  \
           squint inspect <in.hdb|in.hnex>\n"
    );
}

fn write_archive_by_ext(out_path: &Path, arch: &HierarchicalArchive) -> Result<(), String> {
    let ext = out_path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "hdb" | "sdb" | "sqlite" | "db" => write_hdb_file(out_path, arch),
        "hnex" | "snex" | "nex" | "nexus" => write_hnex_file(out_path, arch),
        _ => write_flat_fasta_file(out_path, arch),
    }
}

fn parse_region_spec(s: &str) -> Result<(String, usize, usize), String> {
    let (seq_name, range_s) = s
        .rsplit_once(':')
        .ok_or_else(|| format!("Expected '--region <seq>:<start>-<end>', got '{s}'"))?;
    let (a_s, b_s) = range_s
        .split_once('-')
        .ok_or_else(|| format!("Expected '<start>-<end>' in '--region {s}'"))?;
    let a: usize = a_s
        .parse()
        .map_err(|e| format!("Invalid start coordinate in '{s}': {e}"))?;
    let b: usize = b_s
        .parse()
        .map_err(|e| format!("Invalid end coordinate in '{s}': {e}"))?;
    if a == 0 || b < a {
        return Err(format!("Region must be 1-based inclusive with start <= end, got '{s}'"));
    }
    Ok((seq_name.to_string(), a, b))
}

fn cmd_convert(args: &[String]) {
    let mut in_path: Option<String> = None;
    let mut out_path: Option<String> = None;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                i += 1;
                out_path = args.get(i).cloned();
            }
            "-h" | "--help" => {
                print_usage();
                return;
            }
            arg if !arg.starts_with('-') && in_path.is_none() => {
                in_path = Some(arg.to_string());
            }
            other => {
                eprintln!("Unrecognized convert argument: {other}");
                print_usage();
                process::exit(1);
            }
        }
        i += 1;
    }
    let (Some(ip), Some(op)) = (in_path, out_path) else {
        eprintln!("Usage: squint convert <in.hnex|in.hdb> -o <out.hdb|out.hnex|out.fa>");
        process::exit(1);
    };
    let arch = match read_archive_auto(Path::new(&ip)) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Failed reading '{ip}': {e}");
            process::exit(1);
        }
    };
    if let Err(e) = write_archive_by_ext(Path::new(&op), &arch) {
        eprintln!("Failed writing '{op}': {e}");
        process::exit(1);
    }
    eprintln!(
        "[squint convert] {} -> {} (seqs={}, components={})",
        ip,
        op,
        arch.seqs.len(),
        arch.components.len()
    );
}

fn cmd_subset(args: &[String]) {
    let mut in_path: Option<String> = None;
    let mut out_path: Option<String> = None;
    let mut spec = SubsetSpec::default();
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "-o" | "--output" => {
                i += 1;
                out_path = args.get(i).cloned();
            }
            "--component" => {
                i += 1;
                let v = args.get(i).unwrap_or_else(|| {
                    eprintln!("Missing value for --component");
                    process::exit(1);
                });
                spec.component_id = Some(v.parse::<usize>().unwrap_or_else(|e| {
                    eprintln!("Invalid --component '{v}': {e}");
                    process::exit(1);
                }));
            }
            "--clade" => {
                i += 1;
                spec.clade = args.get(i).cloned();
            }
            "--taxa" => {
                i += 1;
                let v = args.get(i).unwrap_or_else(|| {
                    eprintln!("Missing value for --taxa");
                    process::exit(1);
                });
                spec.taxa = Some(
                    v.split(',')
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(str::to_string)
                        .collect(),
                );
            }
            "--region" => {
                i += 1;
                let v = args.get(i).unwrap_or_else(|| {
                    eprintln!("Missing value for --region");
                    process::exit(1);
                });
                spec.region = Some(parse_region_spec(v).unwrap_or_else(|e| {
                    eprintln!("{e}");
                    process::exit(1);
                }));
            }
            "--scale" => {
                i += 1;
                let v = args.get(i).map(|s| s.as_str()).unwrap_or("");
                match v {
                    "all" => spec.global_core_only = false,
                    "global-core" => spec.global_core_only = true,
                    other => {
                        eprintln!("Invalid --scale '{other}'. Expected all|global-core.");
                        process::exit(1);
                    }
                }
            }
            "-h" | "--help" => {
                print_usage();
                return;
            }
            arg if !arg.starts_with('-') && in_path.is_none() => {
                in_path = Some(arg.to_string());
            }
            other => {
                eprintln!("Unrecognized subset argument: {other}");
                print_usage();
                process::exit(1);
            }
        }
        i += 1;
    }
    let (Some(ip), Some(op)) = (in_path, out_path) else {
        eprintln!(
            "Usage: squint subset <in.hnex|in.hdb> -o <out.hnex|out.hdb|out.fa> [--component id] [--clade node] [--taxa csv] [--region seq:start-end] [--scale all|global-core]"
        );
        process::exit(1);
    };
    let in_p = Path::new(&ip);
    let sub_arch = if is_hdb_file(in_p) {
        subset_hdb_file(in_p, &spec)
    } else {
        read_archive_auto(in_p).and_then(|a| subset_archive(&a, &spec))
    };
    let arch = match sub_arch {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Subset failed for '{ip}': {e}");
            process::exit(1);
        }
    };
    if let Err(e) = write_archive_by_ext(Path::new(&op), &arch) {
        eprintln!("Failed writing '{op}': {e}");
        process::exit(1);
    }
    eprintln!(
        "[squint subset] {} -> {} (seqs={}, components={}, cols={})",
        ip,
        op,
        arch.seqs.len(),
        arch.components.len(),
        arch.components.iter().map(|c| c.num_columns).sum::<usize>(),
    );
}

fn cmd_inspect(args: &[String]) {
    let Some(ip) = args.first().filter(|a| !a.starts_with('-')) else {
        eprintln!("Usage: squint inspect <in.hdb|in.hnex>");
        process::exit(1);
    };
    match read_archive_auto(Path::new(ip)) {
        Ok(arch) => print!("{}", inspect_archive_summary(&arch)),
        Err(e) => {
            eprintln!("Inspect failed for '{ip}': {e}");
            process::exit(1);
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }

    match args[1].as_str() {
        "convert" => {
            cmd_convert(&args[2..]);
            return;
        }
        "subset" => {
            cmd_subset(&args[2..]);
            return;
        }
        "inspect" => {
            cmd_inspect(&args[2..]);
            return;
        }
        _ => {}
    }

    let mut input_path: Option<String> = None;
    let mut out_path: Option<String> = None;
    let mut out_hnex: Option<String> = None;
    let mut out_hdb: Option<String> = None;
    let mut out_json: Option<String> = None;
    let mut mode_opt: Option<SeqMode> = None;
    let mut threads: Option<usize> = None;
    let mut eval_ref = false;

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print_usage();
                return;
            }
            "-o" | "--output" => {
                i += 1;
                out_path = args.get(i).cloned();
            }
            "--hnex" => {
                i += 1;
                out_hnex = args.get(i).cloned();
            }
            "--hdb" => {
                i += 1;
                out_hdb = args.get(i).cloned();
            }
            "-j" | "--json" => {
                i += 1;
                out_json = args.get(i).cloned();
            }
            "--mode" => {
                i += 1;
                if let Some(m) = args.get(i) {
                    mode_opt = match m.to_ascii_lowercase().as_str() {
                        "auto" => None,
                        "codon" => Some(SeqMode::Codon),
                        "protein" | "aa" => Some(SeqMode::Protein),
                        "dna" | "nt" => Some(SeqMode::Dna),
                        other => {
                            eprintln!("Unknown mode '{}'. Expected auto|codon|protein|dna.", other);
                            process::exit(1);
                        }
                    };
                }
            }
            "-T" | "--threads" => {
                i += 1;
                if let Some(t) = args.get(i) {
                    threads = t.parse::<usize>().ok();
                }
            }
            "--eval-ref" => {
                eval_ref = true;
            }
            arg if !arg.starts_with('-') && input_path.is_none() => {
                input_path = Some(arg.to_string());
            }
            other => {
                eprintln!("Unrecognized argument: {}", other);
                print_usage();
                process::exit(1);
            }
        }
        i += 1;
    }

    let Some(in_path) = input_path else {
        print_usage();
        process::exit(1);
    };

    if let Some(t) = threads {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(t.max(1))
            .build_global();
    }

    let content = match fs::read_to_string(&in_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to read '{}': {}", in_path, e);
            process::exit(1);
        }
    };

    let t0 = Instant::now();
    let (raw_seqs, ref_aligned) = parse_sequences_str(&content);
    drop(content);
    let ref_aligned = if eval_ref { ref_aligned } else { Vec::new() };
    if raw_seqs.is_empty() {
        eprintln!("No sequences found in '{}'.", in_path);
        process::exit(1);
    }

    let ledger = analyze_unaligned_cohort(&raw_seqs, mode_opt);
    drop(raw_seqs);
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let comp_ratio = if ledger.raw_null_cohort_nats > 0.0 {
        100.0 * ledger.net_cohort_gain_nats / ledger.raw_null_cohort_nats
    } else {
        0.0
    };

    eprintln!(
        "[squint] N={} | mode={:?} | time={:.1} ms | L0={:.1} nats -> L_prov={:.1} nats (saved {:+.1} nats, {:.1}%) | components={}",
        ledger.num_sequences,
        ledger.mode,
        elapsed_ms,
        ledger.raw_null_cohort_nats,
        ledger.provenance_cohort_nats,
        ledger.net_cohort_gain_nats,
        comp_ratio,
        ledger.num_homology_components,
    );

    for comp in &ledger.components {
        eprintln!(
            "  Component #{} ({} seqs): cols={} [GlobalFine={}, CladeLocal={}, Private={}] | mean_P_core={:.4} | horizons={} | frameshifts={}",
            comp.component_id,
            comp.sequence_ids.len(),
            comp.num_columns,
            comp.num_global_fine_cols,
            comp.num_clade_local_cols,
            comp.num_lineage_private_cols,
            comp.mean_core_posterior,
            comp.horizon_bubbles.len(),
            comp.frameshifts_repaired.len(),
        );
        for hb in &comp.horizon_bubbles {
            eprintln!(
                "    -> Coarse Horizon Bubble (step #{}): {}x{} clades, {} vs {} local cols | class={} | Macro-PRIME={:+.2} nats | dHydro={:+.2} | dCharge={:+.2}",
                hb.tree_merge_step,
                hb.left_clade_size,
                hb.right_clade_size,
                hb.left_col_span,
                hb.right_col_span,
                hb.dominant_biophys_class.as_deref().unwrap_or("n/a"),
                hb.macro_biophys_gain_nats,
                hb.delta_hydropathy,
                hb.delta_charge,
            );
        }
        for fs in &comp.frameshifts_repaired {
            eprintln!(
                "    -> Repaired Frameshift in '{}': nt={} (~codon {}), phase=+{:}, gain={:+.1} nats",
                fs.seq_id, fs.nt_pos, fs.approx_codon_pos, fs.phase_shift, fs.mdl_gain_nats
            );
        }
    }

    if eval_ref && ledger.components.len() == 1 {
        if let Some((prec, rec, f1, tc)) = evaluate_against_reference_msa(
            &ref_aligned,
            &ledger.components[0].aligned_fasta,
            ledger.mode,
            &ledger.components[0].frameshifts_repaired,
        ) {
            eprintln!(
                "  [Reference MSA Audit (blind from unaligned)] Pair Precision={:.4} | Recall={:.4} | F1={:.4} | Exact Column TC={:.4}",
                prec, rec, f1, tc
            );
        }
    }

    let mut arch_opt: Option<HierarchicalArchive> = None;
    let need_arch = out_hnex.is_some()
        || out_hdb.is_some()
        || out_path.as_deref().map_or(false, |p| {
            let low = p.to_ascii_lowercase();
            low.ends_with(".hnex")
                || low.ends_with(".snex")
                || low.ends_with(".nex")
                || low.ends_with(".nexus")
                || low.ends_with(".hdb")
                || low.ends_with(".sdb")
                || low.ends_with(".sqlite")
                || low.ends_with(".db")
        });
    if need_arch {
        arch_opt = Some(HierarchicalArchive::from_ledger(&ledger));
    }

    if let Some(f_path) = out_path {
        let low = f_path.to_ascii_lowercase();
        if low.ends_with(".hnex")
            || low.ends_with(".snex")
            || low.ends_with(".nex")
            || low.ends_with(".nexus")
        {
            if let Err(e) = write_hnex_file(Path::new(&f_path), arch_opt.as_ref().unwrap()) {
                eprintln!("{e}");
                process::exit(1);
            }
        } else if low.ends_with(".hdb")
            || low.ends_with(".sdb")
            || low.ends_with(".sqlite")
            || low.ends_with(".db")
        {
            if let Err(e) = write_hdb_file(Path::new(&f_path), arch_opt.as_ref().unwrap()) {
                eprintln!("{e}");
                process::exit(1);
            }
        } else {
            use std::io::{BufWriter, Write};
            let file = match fs::File::create(&f_path) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("Failed to create FASTA '{}': {}", f_path, e);
                    process::exit(1);
                }
            };
            let mut w = BufWriter::with_capacity(262_144, file);
            let multi = ledger.components.len() > 1;
            for comp in &ledger.components {
                for (id, seq) in &comp.aligned_fasta {
                    let res = if multi {
                        writeln!(w, ">{} component={}\n{}", id, comp.component_id, seq)
                    } else {
                        writeln!(w, ">{}\n{}", id, seq)
                    };
                    if let Err(e) = res {
                        eprintln!("Failed to write FASTA '{}': {}", f_path, e);
                        process::exit(1);
                    }
                }
            }
            if let Err(e) = w.flush() {
                eprintln!("Failed to flush FASTA '{}': {}", f_path, e);
                process::exit(1);
            }
        }
    }

    if let Some(ref hnex_path) = out_hnex {
        if let Err(e) = write_hnex_file(Path::new(hnex_path), arch_opt.as_ref().unwrap()) {
            eprintln!("{e}");
            process::exit(1);
        }
    }

    if let Some(ref hdb_path) = out_hdb {
        if let Err(e) = write_hdb_file(Path::new(hdb_path), arch_opt.as_ref().unwrap()) {
            eprintln!("{e}");
            process::exit(1);
        }
    }

    if let Some(j_path) = out_json {
        let j = serde_json::to_string_pretty(&ledger).unwrap();
        if let Err(e) = fs::write(&j_path, j) {
            eprintln!("Failed to write JSON '{}': {}", j_path, e);
            process::exit(1);
        }
    }
}
