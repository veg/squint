pub mod alphabet;
pub mod mdl;
pub mod models;
pub mod multiseq;
pub mod storage;
pub mod transducer;

#[cfg(test)]
mod tests {
    use crate::alphabet::{
        parse_sequences_str, EncodedSeq, GeneticCode, RawSeq, SeqMode, AA_CHARS,
    };
    use crate::mdl::{elias_rissanen_nats, logsumexp2, SingleSeqNullProfile, NEG_INF};
    use crate::models::{
        build_mechanistic_aa_matrix, build_telescoping_codon_matrix, column_niche_mixture_ln_prob,
        default_background_pi,
    };
    use crate::multiseq::analyze_unaligned_cohort;
    use crate::transducer::{
        build_copula_ratio_matrix, coarse_regime_for, compute_ln_path_partition,
        compute_ln_path_partition_corridor, evaluate_pair_homology, exact_marginal_mass_up_to_len,
        get_global_regime_tables, precompute_local_coarse_tilts, run_forward_backward_regime,
        run_forward_backward_regime_corridor, CorridorBounds, PreparedSeq, SegmentKind,
        TransducerRegime,
    };

    #[test]
    fn test_elias_rissanen_kraft_tight() {
        let mut sum = 0.0f64;
        let n_max = 200_000usize;
        for n in 0..=n_max {
            sum += (-elias_rissanen_nats(n)).exp();
        }
        // Exact analytical tail for n > n_max: 0.5 * ln(2) / ln(n_max + 2)
        let tail = 0.5 * std::f64::consts::LN_2 / ((n_max + 2) as f64).ln();
        let total = sum + tail;
        assert!(
            (total - 1.0).abs() < 1e-13,
            "Elias-Rissanen Kraft sum deviated from 1.0: {}",
            total
        );
    }

    #[test]
    fn test_single_seq_null_m0_kraft_and_repeat_deflation() {
        let gc = GeneticCode::standard();
        let bg = default_background_pi(SeqMode::Codon, &gc);
        // 15 varied codons followed by 35 tandem CAG (Gln) repeats
        let raw_dna = format!(
            "ATGGCCCTGTGGATGCGCCTCCTGCCCCTGCTGGCGCTGCTGGCC{}",
            "CAGCAGCAG".repeat(12)
        );
        let raw = RawSeq {
            id: "polyq_test".into(),
            raw_bytes: raw_dna.into_bytes(),
        };
        let enc = EncodedSeq::from_raw(&raw, SeqMode::Codon, &gc);
        let prof = SingleSeqNullProfile::compute(&enc.tokens, 61, &bg);

        // 1. Verify exact normalization \sum_{a=0..60} M_0(a | X_{<i}) == 1.0 at every site i
        for i in 0..enc.len() {
            let mut s = 0.0;
            for a in 0..61 {
                s += prof.m0_prob(i, a);
            }
            assert!(
                (s - 1.0).abs() < 1e-13,
                "M_0 distribution at pos {} summed to {}",
                i,
                s
            );
        }

        // 2. Verify repeat deflation inside the CAG tract: late CAG codons must cost < 0.25 nats (vs ~4.11 nats uniform!)
        let last_ln_p = prof.site_ln_m0[enc.len() - 1];
        assert!(
            -last_ln_p < 0.25,
            "Expected poly-Q repeat surprise < 0.25 nats, got {}",
            -last_ln_p
        );
        assert!(
            prof.site_repeat_weight[enc.len() - 1] > 0.85,
            "Expected high repeat weight in poly-Q tract, got {}",
            prof.site_repeat_weight[enc.len() - 1]
        );
    }

    #[test]
    fn test_telescoping_codon_matrix_kraft_and_saturation() {
        let gc = GeneticCode::standard();
        // Extreme 90% GC3 biased target codon distribution
        let mut pi_gc3 = vec![0.0f64; 61];
        for s in 0..61 {
            let t = gc.sense_to_triplet[s];
            let b2 = t & 3;
            pi_gc3[s] = if b2 == 1 || b2 == 2 { 9.0 } else { 1.0 };
        }
        let tot: f64 = pi_gc3.iter().sum();
        for v in &mut pi_gc3 {
            *v /= tot;
        }

        for &(ta, ts) in &[(0.0, 0.0), (0.05, 0.2), (0.8, 3.0), (200.0, 200.0)] {
            let mat = build_telescoping_codon_matrix(ta, ts, &pi_gc3, &gc);
            for c1 in 0..61 {
                let rsum: f64 = mat[c1 * 61..(c1 + 1) * 61].iter().sum();
                assert!(
                    (rsum - 1.0).abs() < 1e-13,
                    "Row {} at ({}, {}) summed to {}",
                    c1,
                    ta,
                    ts,
                    rsum
                );
            }
            if ta == 0.0 && ts == 0.0 {
                for c1 in 0..61 {
                    assert!((mat[c1 * 61 + c1] - 1.0).abs() < 1e-12);
                }
            }
            if ta >= 200.0 && ts >= 200.0 {
                // At deep saturation, every row must converge to target pi_gc3 so log-odds -> 0!
                for c1 in 0..61 {
                    for c2 in 0..61 {
                        assert!(
                            (mat[c1 * 61 + c2] - pi_gc3[c2]).abs() < 1e-3,
                            "Saturation mismatch at ({}, {}): {} vs {}",
                            c1,
                            c2,
                            mat[c1 * 61 + c2],
                            pi_gc3[c2]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_transducer_exact_kraft_sum_over_all_strings() {
        let gc = GeneticCode::standard();
        let bg = default_background_pi(SeqMode::Dna, &gc);
        let raw_x = RawSeq {
            id: "X".into(),
            raw_bytes: b"ACG".to_vec(),
        };
        let seq_x = EncodedSeq::from_raw(&raw_x, SeqMode::Dna, &gc);
        let prof_x = SingleSeqNullProfile::compute(&seq_x.tokens, 4, &bg);
        let reg = TransducerRegime {
            t_primary: 0.25,
            t_syn: 0.25,
        };
        let cop_mat = build_copula_ratio_matrix(SeqMode::Dna, reg, &bg, &gc);
        let cop_coarse =
            build_copula_ratio_matrix(SeqMode::Dna, coarse_regime_for(SeqMode::Dna, reg), &bg, &gc);
        let local_coarse = precompute_local_coarse_tilts(&seq_x, &cop_coarse, 4);

        // 1. Exact analytical marginal mass propagation up to L_Y = 300: must equal 1.0000000000000!
        for &lx_test in &[1usize, 3, 12] {
            let exact_mass = exact_marginal_mass_up_to_len(lx_test, reg, 0.8, 300);
            assert!(
                (exact_mass - 1.0).abs() < 1e-12,
                "Automaton Kraft mass for Lx={} deviated from 1.0: {}",
                lx_test,
                exact_mass
            );
        }

        // 2. Explicit brute-force string enumeration over ALL 21,845 DNA strings Y of length 0..=7
        // under causal non-stationary M_0(Y_{<j}), confirming:
        //   (a) exact match to global analytical mass at L_Y = 7, AND
        //   (b) exact pointwise Length-Copula Kraft sum == 1.000000000000 for EVERY individual L_Y in 1..=7!
        let expected_mass_7 = exact_marginal_mass_up_to_len(seq_x.len(), reg, 0.8, 7);
        let mut ln_total = NEG_INF;
        let letters = [b'A', b'C', b'G', b'T'];
        for ly in 0..=7usize {
            let num_strs = 1usize << (2 * ly);
            let mut ln_len_slice = NEG_INF;
            for mask in 0..num_strs {
                let mut bytes = Vec::with_capacity(ly);
                for pos in 0..ly {
                    let code = (mask >> (2 * pos)) & 3;
                    bytes.push(letters[code]);
                }
                let raw_y = RawSeq {
                    id: "Y".into(),
                    raw_bytes: bytes,
                };
                let seq_y = EncodedSeq::from_raw(&raw_y, SeqMode::Dna, &gc);
                let prof_y = SingleSeqNullProfile::compute(&seq_y.tokens, 4, &bg);
                let fb = run_forward_backward_regime(
                    &seq_x,
                    &seq_y,
                    &prof_x,
                    &prof_y,
                    &local_coarse,
                    &cop_mat,
                    &gc,
                    reg,
                    0.8,
                    true,
                );
                let fb_fast = run_forward_backward_regime(
                    &seq_x,
                    &seq_y,
                    &prof_x,
                    &prof_y,
                    &local_coarse,
                    &cop_mat,
                    &gc,
                    reg,
                    0.8,
                    false,
                );
                assert!(
                    fb_fast.ln_prob_y_given_x == fb.ln_prob_y_given_x
                        || (fb_fast.ln_prob_y_given_x - fb.ln_prob_y_given_x).abs() < 1e-13,
                    "2-row rolling Forward diverged from full FB: {} vs {}",
                    fb_fast.ln_prob_y_given_x,
                    fb.ln_prob_y_given_x
                );
                ln_total = logsumexp2(ln_total, fb.ln_prob_y_given_x);
                ln_len_slice = logsumexp2(ln_len_slice, fb.ln_prob_y_given_x);
                for &p in &fb.post_fine {
                    assert!(
                        p >= -1e-9 && p <= 1.0 + 1e-9,
                        "Posterior fine out of [0,1]: {}",
                        p
                    );
                }
            }
            if ly >= 1 {
                let ln_part = compute_ln_path_partition(seq_x.len(), ly, reg, 0.8);
                let slice_kraft = (ln_len_slice - ln_part).exp();
                assert!(
                    (slice_kraft - 1.0).abs() < 1e-11,
                    "Length-Copula Kraft sum for L_Y={} deviated from 1.0: {}",
                    ly,
                    slice_kraft
                );
            }
        }
        let prob_mass = ln_total.exp();
        assert!(
            (prob_mass - expected_mass_7).abs() < 1e-11 && prob_mass <= 1.0,
            "Enumerated string Kraft sum ({}) mismatched exact analytical mass ({})!",
            prob_mass,
            expected_mass_7
        );
    }

    #[test]
    fn test_easy_identical_snp_and_clean_indel() {
        let gc = GeneticCode::standard();
        let cds_a = "ATGGAGGAGCCGCAGTCAGATCCTAGCGTCGAGCCCCCTCTGAGTCAGGAAACATTTTCAGACCTATGGAAACTACTTCCTGAAAACAACGTTCTGTCCCCCTTGCCGTCCCAAGCAATGGATGATTTGATGCTGTCCCCGGACGATATTGAACAATGGTTCACTGAAGACCCAGGTCCAGATGAAGCTCCCAGAATGCCAGAGGCTGCTCCCCCCGTGGCCCCTGCACCAGCAGCTCCTACACCGGCGGCCCCTGCACCAGCCCCCTCCTGGCCCCTGTCATCTTCTGTCCCTTCCCAGAAAACCTACCAGGGCAGCTACGGTTTCCGTCTGGGCTTCTTGCATTCTGGGACAGCCAAGTCTGTGACTTGCACGTACTCCCCTGCCCTCAACAAGATGTTTTGCCAACTGGCCAAGACCTGCCCTGTGCAGCTGTGGGTTGATTCCACACCCCCGCCCGGCACCCGCGTCCGCGCCATGGCCATCTACAAGCAGTCACAGCACATGACGGAGGTTGTGAGGCGCTGCCCCCACCATGAGCGCTGCTCAGATAGCGATGGTCTGGCCCCTCCTCAGCATCTTATCCGAGTGGAAGGAAATTTGCGTGTGGAGTATTTGGATGACAGAAACACTTTTCGACATAGTGTGGTGGTGCCCTATGAGCCGCCTGAGGTTGGCTCTGACTGTACCACCATCCACTACAACTACATGTGTAACAGTTCCTGCATGGGCGGCATGAACCGGAGGCCCATCCTCACCATCATCACACTGGAAGACTCCAGTGGTAATCTACTGGGACGGAACAGCTTTGAGGTGCGTGTTTGTGCCTGTCCTGGGAGAGACCGGCGCACAGAGGAAGAGAATCTCCGCAAGAAAGGGGAGCCTCACCACGAGCTGCCCCCAGGGAGCACTAAGCGAGCACTGCCCAACAACACCAGCTCCTCTCCCCAGCCAAAGAAGAAACCACTGGATGGAGAATATTTCACCCTTCAGATCCGTGGGCGTGAGCGCTTCGAGATGTTCCGAGAGCTGAATGAGGCCTTGGAACTCAAGGATGCCCAGGCTGGGAAGGAGCCAGGGGGGAGCAGGGCTCACTCCAGCCACCTGAAGTCCAAAAAGGGTCAGTCTACCTCCCGCCATAAAAAACTCATGTTCAAGACAGAAGGGCCTGACTCAGACTGA";
        // Create cds_b with a 3-codon (9-bp) clean deletion at codon 40 and 4 point mutations
        let mut bytes_b = cds_a.as_bytes().to_vec();
        // Point mutations
        bytes_b[12] = b'A';
        bytes_b[60] = b'T';
        bytes_b[300] = b'C';
        // Delete 9 bp (codons 40..43: bytes 120..129)
        bytes_b.drain(120..129);

        let raw_a = RawSeq {
            id: "TP53_WT".into(),
            raw_bytes: cds_a.as_bytes().to_vec(),
        };
        let raw_b = RawSeq {
            id: "TP53_del3".into(),
            raw_bytes: bytes_b,
        };
        let sa = EncodedSeq::from_raw(&raw_a, SeqMode::Codon, &gc);
        let sb = EncodedSeq::from_raw(&raw_b, SeqMode::Codon, &gc);
        let res = evaluate_pair_homology(&sa, &sb, &gc, true);

        assert!(
            res.mdl_gain_nats > 800.0,
            "Expected massive MDL homology gain on clean orthologs, got {}",
            res.mdl_gain_nats
        );
        assert!((res.homology_posterior - 1.0).abs() < 1e-12);

        // Check exact 1-to-1 recovery upstream (i == j for i < 40) and downstream (i == j + 3 for i >= 43)!
        let mut map = std::collections::HashMap::new();
        for e in &res.fine_edges {
            if e.posterior > 0.80 {
                map.insert(e.idx_x, e.idx_y);
            }
        }
        let mut exact_hits = 0usize;
        let mut expected = 0usize;
        for i in 0..sa.len() {
            if i < 39 {
                expected += 1;
                if map.get(&i) == Some(&i) {
                    exact_hits += 1;
                }
            } else if i >= 44 {
                expected += 1;
                if map.get(&i) == Some(&(i - 3)) {
                    exact_hits += 1;
                }
            }
        }
        let acc = exact_hits as f64 / expected as f64;
        assert!(
            acc > 0.99,
            "Expected >99% exact codon accuracy around clean 3-codon indel, got {:.4}",
            acc
        );
    }

    #[test]
    fn test_frameshift_detection_and_repair() {
        let cds_wt = "ATGGAGGAGCCGCAGTCAGATCCTAGCGTCGAGCCCCCTCTGAGTCAGGAAACATTTTCAGACCTATGGAAACTACTTCCTGAAAACAACGTTCTGTCCCCCTTGCCGTCCCAAGCAATGGATGATTTGATGCTGTCCCCGGACGATATTGAACAATGGTTCACTGAAGACCCAGGTCCAGATGAAGCTCCCAGAATGCCAGAGGCTGCTCCCCCCGTGGCCCCTGCACCAGCAGCTCCTACACCGGCGGCCCCTGCACCAGCCCCCTCCTGGCCCCTGTCATCTTCTGTCCCTTCCCAGAAAACCTACCAGGGCAGCTACGGTTTCCGTCTGGGCTTCTTGCATTCTGGGACAGCCAAGTCTGTGACTTGCACGTACTCCCCTGCCCTCAACAAGATGTTTTGCCAACTGGCCAAGACCTGCCCTGTGCAGCTGTGGGTTGATTCCACACCCCCGCCCGGCACCCGCGTCCGCGCCATGGCCATCTACAAGCAGTCACAGCACATGACGGAGGTTGTGAGGCGCTGCCCCCACCATGAGCGCTGCTCAGATAGCGATGGTCTGGCCCCTCCTCAGCATCTTATCCGAGTGGAAGGAAATTTGCGTGTGGAGTATTTGGATGACAGAAACACTTTTCGACATAGTGTGGTGGTGCCCTATGAGCCGCCTGAGGTTGGCTCTGACTGTACCACCATCCACTACAACTACATGTGTAACAGTTCCTGCATGGGCGGCATGAACCGGAGGCCCATCCTCACCATCATCACACTGGAAGACTCCAGTGGTAATCTACTGGGACGGAACAGCTTTGAGGTGCGTGTTTGTGCCTGTCCTGGGAGAGACCGGCGCACAGAGGAAGAGAATCTCCGCAAGAAAGGGGAGCCTCACCACGAGCTGCCCCCAGGGAGCACTAAGCGAGCACTGCCCAACAACACCAGCTCCTCTCCCCAGCCAAAGAAGAAACCACTGGATGGAGAATATTTCACCCTTCAGATCCGTGGGCGTGAGCGCTTCGAGATGTTCCGAGAGCTGAATGAGGCCTTGGAACTCAAGGATGCCCAGGCTGGGAAGGAGCCAGGGGGGAGCAGGGCTCACTCCAGCCACCTGAAGTCCAAAAAGGGTCAGTCTACCTCCCGCCATAAAAAACTCATGTTCAAGACAGAAGGGCCTGACTCAGACTGA";
        // Insert a stray single 'A' nucleotide at byte 301 (+1 bp frameshift!)
        let mut fs_bytes = cds_wt.as_bytes().to_vec();
        fs_bytes.insert(301, b'A');

        let raw_wt = RawSeq {
            id: "WT".into(),
            raw_bytes: cds_wt.as_bytes().to_vec(),
        };
        let raw_fs = RawSeq {
            id: "FS_plus1".into(),
            raw_bytes: fs_bytes,
        };
        let ledger = analyze_unaligned_cohort(&[raw_wt, raw_fs], Some(SeqMode::Codon));
        assert_eq!(ledger.num_homology_components, 1);
        let comp = &ledger.components[0];
        assert!(
            !comp.frameshifts_repaired.is_empty(),
            "Expected automatic detection and repair of +1 bp frameshift!"
        );
        let ev = &comp.frameshifts_repaired[0];
        assert!(
            (ev.nt_pos as isize - 301).abs() <= 6,
            "Expected frameshift near nt 301, got {}",
            ev.nt_pos
        );
        assert!(
            comp.num_global_fine_cols >= 385,
            "Expected >=385 recovered global fine codons after frameshift repair, got {}",
            comp.num_global_fine_cols
        );
    }

    #[test]
    fn test_negative_controls_reversed_gc3_polyq_and_contaminant() {
        let gc = GeneticCode::standard();
        let tp53_aa = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD";
        let rev_aa: String = tp53_aa.chars().rev().collect();

        let s_fwd = EncodedSeq::from_raw(
            &RawSeq {
                id: "TP53_fwd".into(),
                raw_bytes: tp53_aa.as_bytes().to_vec(),
            },
            SeqMode::Protein,
            &gc,
        );
        let s_rev = EncodedSeq::from_raw(
            &RawSeq {
                id: "TP53_rev".into(),
                raw_bytes: rev_aa.into_bytes(),
            },
            SeqMode::Protein,
            &gc,
        );

        // 1. Reversed sequence (exact same length, AA composition, and proline-rich region!) MUST give <= 0 MDL gain!
        let rev_res = evaluate_pair_homology(&s_fwd, &s_rev, &gc, false);
        assert!(
            rev_res.mdl_gain_nats < 0.0,
            "Reversed sequence false positive! Expected < 0 nats, got {:.3}",
            rev_res.mdl_gain_nats
        );

        // 2. Two unrelated Poly-Q + different flanking sequences MUST give <= 0 MDL gain!
        let pq1 = format!("MKTLLIAAVVFF{}{}", "QQQ".repeat(15), "WYYFFCCVVAAGGLLII");
        let pq2 = format!("RHKDESTNQGPC{}{}", "QQQ".repeat(15), "DDDEEEKKKRRRHHHPP");
        let spq1 = EncodedSeq::from_raw(
            &RawSeq {
                id: "PQ1".into(),
                raw_bytes: pq1.into_bytes(),
            },
            SeqMode::Protein,
            &gc,
        );
        let spq2 = EncodedSeq::from_raw(
            &RawSeq {
                id: "PQ2".into(),
                raw_bytes: pq2.into_bytes(),
            },
            SeqMode::Protein,
            &gc,
        );
        let pq_res = evaluate_pair_homology(&spq1, &spq2, &gc, false);
        assert!(
            pq_res.mdl_gain_nats < 0.0,
            "Poly-Q low-complexity false positive! Expected < 0 nats, got {:.3}",
            pq_res.mdl_gain_nats
        );

        // 3. Two unrelated 100%-GC3 coding genes MUST give <= 0 MDL gain!
        // Gene 1: hydrophobic-rich with C/G 3rd positions; Gene 2: charged/polar with C/G 3rd positions
        let gc3_gene1 = "CTGATCGTGGCCCTGTTCGTGATCCTGGCCGTGCTGATCGGCCTGTTCGTGATCCTGGCC".repeat(4);
        let gc3_gene2 = "GACGAGGACAAGAAGCGCGAGCACGACCAGAACAGCTCCGACGAGGACAAGAAGCGCGAG".repeat(4);
        let sg1 = EncodedSeq::from_raw(
            &RawSeq {
                id: "GC3_1".into(),
                raw_bytes: gc3_gene1.into_bytes(),
            },
            SeqMode::Codon,
            &gc,
        );
        let sg2 = EncodedSeq::from_raw(
            &RawSeq {
                id: "GC3_2".into(),
                raw_bytes: gc3_gene2.into_bytes(),
            },
            SeqMode::Codon,
            &gc,
        );
        let gc3_res = evaluate_pair_homology(&sg1, &sg2, &gc, false);
        assert!(
            gc3_res.mdl_gain_nats < 0.0,
            "GC3 compositional false positive! Expected < 0 nats, got {:.3}",
            gc3_res.mdl_gain_nats
        );

        // 4. Multi-sequence cohort with 2 TP53 homologs + 1 reversed non-homolog MUST split into 2 components!
        let mut tp53_mut = tp53_aa.as_bytes().to_vec();
        tp53_mut[5] = b'A';
        tp53_mut[50] = b'D';
        let cohort = vec![
            RawSeq {
                id: "TP53_1".into(),
                raw_bytes: tp53_aa.as_bytes().to_vec(),
            },
            RawSeq {
                id: "TP53_2".into(),
                raw_bytes: tp53_mut,
            },
            RawSeq {
                id: "Contaminant_Rev".into(),
                raw_bytes: tp53_aa.chars().rev().collect::<String>().into_bytes(),
            },
        ];
        let ledger = analyze_unaligned_cohort(&cohort, Some(SeqMode::Protein));
        assert_eq!(
            ledger.num_homology_components, 2,
            "Expected contaminant non-homolog to split cleanly into a 2nd component!"
        );
    }

    #[test]
    fn test_clade_local_homology_horizon_and_coarse_bubble() {
        // Construct two 2-taxon clades sharing conserved N- and C-terminal globular cores (45 aa each)
        // flanking a completely turned-over acidic/proline IDR loop (24 aa in Clade A vs 18 aa in Clade B):
        // Within Clade A, A1 and A2 share 1-to-1 loop homology;
        // Within Clade B, B1 and B2 share 1-to-1 loop homology;
        // Between Clade A and Clade B, the loop has 0% 1-to-1 positional homology (completely different length & order),
        // yet conserved Acidic(DE)/Proline biophysical character!
        let n_core = "MKTWCGHCKALAPEYAKAAGKLKAEGSEIRLAKVDATEESDLAQQY";
        let c_core = "FVTGYHPTILFFKNGELNRPVKYGQRSVDALIGFIKNHQGLPAVCW";

        let loop_a1 = "DDSSNNDDSSTTDDSSEENNDDSQ"; // 24 aa Acidic + Polar (non-collinear with Clade B)
        let loop_a2 = "DDSSNNDDSSTTDDSSEENNDDSN";
        let loop_b1 = "GGEEDDPPGGEEDDPPGGE"; // 19 aa Acidic + Pro/Gly
        let loop_b2 = "GGEEDDPPGGEEDDPPGGD";

        let fasta = format!(
            ">A1\n{}{}{}\n>A2\n{}{}{}\n>B1\n{}{}{}\n>B2\n{}{}{}\n",
            n_core, loop_a1, c_core,
            n_core, loop_a2, c_core,
            n_core, loop_b1, c_core,
            n_core, loop_b2, c_core,
        );
        let (raws, _) = parse_sequences_str(&fasta);
        let ledger = analyze_unaligned_cohort(&raws, Some(SeqMode::Protein));
        assert_eq!(ledger.num_homology_components, 1);
        let comp = &ledger.components[0];

        // Core N + Core C = 46 + 46 = 92 GlobalFine columns!
        assert!(
            comp.num_global_fine_cols >= 90 && comp.num_global_fine_cols <= 96,
            "Expected ~92 GlobalFine core columns, got {} (clade_local={})",
            comp.num_global_fine_cols,
            comp.num_clade_local_cols
        );
        // Clade A loop (24) + Clade B loop (19) should stay CladeLocalFine (~43 columns) instead of falsely packed!
        assert!(
            comp.num_clade_local_cols >= 35,
            "Expected clade-local loop columns to stay unmerged across the homology horizon, got {}",
            comp.num_clade_local_cols
        );
        assert!(
            !comp.horizon_bubbles.is_empty(),
            "Expected HorizonBubbleReport across the clade stem!"
        );
        let hb = &comp.horizon_bubbles[0];
        assert!(
            hb.dominant_biophys_class
                .as_deref()
                .unwrap_or("")
                .contains("Acidic"),
            "Expected Macro-PRIME dominant class Acidic(DE) on coarse bubble, got {:?}",
            hb.dominant_biophys_class
        );

        // Verify 100% lossless character round-trip on every sequence!
        for (orig, (_, aln)) in raws.iter().zip(comp.aligned_fasta.iter()) {
            let stripped: Vec<u8> = aln.bytes().filter(|&b| b != b'-').collect();
            assert_eq!(orig.raw_bytes, stripped, "Lossless round-trip violated!");
        }
        let _ = (loop_a1, loop_a2, loop_b1, loop_b2);
    }

    #[test]
    fn test_pairwise_coarse_bubble_detection() {
        let gc = GeneticCode::standard();
        let n_core = "MKTWCGHCKALAPEYAKAAGKLKAEGSEIRLAKVDATEESDLAQQY";
        let c_core = "FVTGYHPTILFFKNGELNRPVKYGQRSVDALIGFIKNHQGLPAVCW";
        let s1 = format!("{}{}{}", n_core, "DSDSEESDDSEDESEDSSEEDESD", c_core);
        let s2 = format!("{}{}{}", n_core, "EGEGEEGDDEGEGEEGDDE", c_core);

        let e1 = EncodedSeq::from_raw(
            &RawSeq {
                id: "S1".into(),
                raw_bytes: s1.into_bytes(),
            },
            SeqMode::Protein,
            &gc,
        );
        let e2 = EncodedSeq::from_raw(
            &RawSeq {
                id: "S2".into(),
                raw_bytes: s2.into_bytes(),
            },
            SeqMode::Protein,
            &gc,
        );
        let res = evaluate_pair_homology(&e1, &e2, &gc, true);
        let has_bubble = res
            .segments
            .iter()
            .any(|seg| seg.kind == SegmentKind::CoarseBubble);
        assert!(
            has_bubble,
            "Expected pairwise segmenter to identify interior CoarseBubble, got {:?}",
            res.segments
        );
    }

    #[test]
    fn test_cr_nexus_iupac_and_mito_tga_roundtrip() {
        // 1. Classic MacOS `\r`-only NEXUS with IUPAC codes (`AAY`, `AGR`, `TAN`) AND mitochondrial `TGA = Trp` codons!
        let seq1 = "ATGTTCATTAATCGTTGATTATTCTCAACCAATCACAAAGATATCGGAACCCTCTATCTACTATTCGGAGCCTGAGCGGGAATAGTGGGTACTGCACTAAGTATTTTAATTCGAGCAGAATTAGGTCAACCAGGTGCACTTTTAGGAGATGACCAAATTTACAATGTTATCGTAACTGCCCATGCTTTTGTT";
        let seq2 = "ATGTTCATTAATAAYTGATTATTCTCAACCAATCACAAAGATATCGGAACCCTCTATCTAAGRTTCGGAGCCTGAGCGGGAATAGTGGGTACTGCACTAAGTATTTTAATTCGAGCAGAATTAGGTCAACCAGGTGCACTTTTAGGAGATGACCAAATTTANAATGTTATCGTAACTGCCCATGCTTTTGTT";
        let cr_nexus = format!(
            "#NEXUS\r\rBEGIN CHARACTERS;\rDIMENSIONS NCHAR = 192;\rFORMAT DATATYPE=DNA GAP=- MISSING=?;\rMATRIX\r'Taxon_WT' {}\r'Taxon_IUPAC' {}\r;\rEND;\r",
            seq1, seq2
        );
        let (raws, alns) = parse_sequences_str(&cr_nexus);
        assert_eq!(raws.len(), 2, "Failed to parse \\r-delimited NEXUS!");
        assert_eq!(alns.len(), 2);

        let gc_mito = GeneticCode::for_cohort(&raws);
        assert!(
            gc_mito.mito_tga_trp && gc_mito.num_sense == 62,
            "Expected automatic 62-sense mitochondrial TGA=Trp detection!"
        );
        let bg62 = default_background_pi(SeqMode::Codon, &gc_mito);
        let mat62 = build_telescoping_codon_matrix(0.2, 0.8, &bg62, &gc_mito);
        for c1 in 0..62 {
            let rsum: f64 = mat62[c1 * 62..(c1 + 1) * 62].iter().sum();
            assert!((rsum - 1.0).abs() < 1e-13, "62-codon row sum = {}", rsum);
        }

        let ledger = analyze_unaligned_cohort(&raws, None);
        assert_eq!(ledger.mode, SeqMode::Codon);
        assert_eq!(ledger.num_homology_components, 1);
        let comp = &ledger.components[0];
        assert!(
            comp.frameshifts_repaired.is_empty(),
            "IUPAC or mitochondrial TGA codons must NOT trigger bogus frameshifts! got {:?}",
            comp.frameshifts_repaired
        );
        assert_eq!(comp.num_global_fine_cols, 64);

        // Verify 100% verbatim lossless preservation of IUPAC `AAY`, `AGR`, `TAN` and `TGA`!
        for (orig, (_, aln)) in raws.iter().zip(comp.aligned_fasta.iter()) {
            let stripped: Vec<u8> = aln.bytes().filter(|&b| b != b'-').collect();
            assert_eq!(orig.raw_bytes, stripped, "Verbatim IUPAC/TGA round-trip failed!");
        }
    }

    #[test]
    fn test_corridor_exact_kraft_and_full_grid_equivalence() {
        let gc = GeneticCode::standard();
        let bg = default_background_pi(SeqMode::Dna, &gc);
        let tables = get_global_regime_tables(SeqMode::Dna, &gc);
        let reg_idx = 0usize;
        let reg = tables.regimes[reg_idx];
        let cop_mat = &tables.copulas[reg_idx];
        let ln_r_base = &tables.ln_r_bases[reg_idx];

        let raw_x = RawSeq {
            id: "X".into(),
            raw_bytes: b"ACG".to_vec(),
        };
        let seq_x = EncodedSeq::from_raw(&raw_x, SeqMode::Dna, &gc);
        let px = PreparedSeq::new(seq_x.clone(), &gc);
        let prof_x = SingleSeqNullProfile::compute(&seq_x.tokens, 4, &bg);
        let cop_coarse =
            build_copula_ratio_matrix(SeqMode::Dna, coarse_regime_for(SeqMode::Dna, reg), &bg, &gc);
        let local_coarse = precompute_local_coarse_tilts(&seq_x, &cop_coarse, 4);

        // 1. CorridorBounds::full must agree with run_forward_backward_regime to <= 1e-12
        for y_bytes in [b"".as_slice(), b"A", b"ACG", b"ATGCG", b"TTTTTTT"] {
            let raw_y = RawSeq {
                id: "Y".into(),
                raw_bytes: y_bytes.to_vec(),
            };
            let seq_y = EncodedSeq::from_raw(&raw_y, SeqMode::Dna, &gc);
            let py = PreparedSeq::new(seq_y.clone(), &gc);
            let prof_y = SingleSeqNullProfile::compute(&seq_y.tokens, 4, &bg);
            let full_corr = CorridorBounds::full(seq_x.len(), seq_y.len());

            if seq_y.len() >= 1 {
                let p_full = compute_ln_path_partition(seq_x.len(), seq_y.len(), reg, 0.8);
                let p_corr = compute_ln_path_partition_corridor(
                    seq_x.len(),
                    seq_y.len(),
                    &full_corr,
                    reg,
                    0.8,
                );
                assert!(
                    (p_full - p_corr).abs() < 1e-12,
                    "Full partition mismatch for ly={}: {} vs {}",
                    seq_y.len(),
                    p_full,
                    p_corr
                );
            }

            let fb_dense = run_forward_backward_regime(
                &seq_x,
                &seq_y,
                &prof_x,
                &prof_y,
                &local_coarse,
                cop_mat,
                &gc,
                reg,
                0.8,
                true,
            );
            let fb_corr = run_forward_backward_regime_corridor(
                &px,
                &py,
                &full_corr,
                px.get_local_coarse(reg_idx, &tables),
                cop_mat,
                ln_r_base,
                None,
                reg,
                0.8,
                true,
            );
            assert!(
                fb_dense.ln_prob_y_given_x == fb_corr.ln_prob_y_given_x
                    || (fb_dense.ln_prob_y_given_x - fb_corr.ln_prob_y_given_x).abs() < 1e-12,
                "CorridorBounds::full log-prob mismatch for Y={}: {} vs {}",
                String::from_utf8_lossy(y_bytes),
                fb_dense.ln_prob_y_given_x,
                fb_corr.ln_prob_y_given_x
            );
            assert_eq!(fb_dense.post_fine.len(), fb_corr.post_fine_banded.len());
            for (k, (&a, &b)) in fb_dense
                .post_fine
                .iter()
                .zip(fb_corr.post_fine_banded.iter())
                .enumerate()
            {
                assert!(
                    (a - b).abs() < 1e-11,
                    "post_fine mismatch at {} for Y={}: {} vs {}",
                    k,
                    String::from_utf8_lossy(y_bytes),
                    a,
                    b
                );
            }
            for (k, (&a, &b)) in fb_dense.post_bx.iter().zip(fb_corr.post_bx.iter()).enumerate() {
                assert!((a - b).abs() < 1e-11, "post_bx mismatch at {}: {} vs {}", k, a, b);
            }
            for (k, (&a, &b)) in fb_dense.post_by.iter().zip(fb_corr.post_by.iter()).enumerate() {
                assert!((a - b).abs() < 1e-11, "post_by mismatch at {}: {} vs {}", k, a, b);
            }
        }

        // 2. Brute-force enumeration over ALL 21,845 DNA strings Y of length 0..=7
        // under fixed Y-independent CorridorBounds::diagonal_band(3, ly, 1)
        let letters = [b'A', b'C', b'G', b'T'];
        let mut ln_total_corr = NEG_INF;
        let expected_mass_7 = exact_marginal_mass_up_to_len(seq_x.len(), reg, 0.8, 7);

        for ly in 0..=7usize {
            let num_strs = 1usize << (2 * ly);
            let band = CorridorBounds::diagonal_band(seq_x.len(), ly, 1);
            let mut ln_len_slice_corr = NEG_INF;

            for mask in 0..num_strs {
                let mut bytes = Vec::with_capacity(ly);
                for pos in 0..ly {
                    bytes.push(letters[(mask >> (2 * pos)) & 3]);
                }
                let raw_y = RawSeq {
                    id: "Y".into(),
                    raw_bytes: bytes,
                };
                let seq_y = EncodedSeq::from_raw(&raw_y, SeqMode::Dna, &gc);
                let py = PreparedSeq::new(seq_y, &gc);

                let fb_post = run_forward_backward_regime_corridor(
                    &px,
                    &py,
                    &band,
                    px.get_local_coarse(reg_idx, &tables),
                    cop_mat,
                    ln_r_base,
                    None,
                    reg,
                    0.8,
                    true,
                );
                let fb_fast = run_forward_backward_regime_corridor(
                    &px,
                    &py,
                    &band,
                    px.get_local_coarse(reg_idx, &tables),
                    cop_mat,
                    ln_r_base,
                    None,
                    reg,
                    0.8,
                    false,
                );
                assert!(
                    fb_post.ln_prob_y_given_x == fb_fast.ln_prob_y_given_x
                        || (fb_post.ln_prob_y_given_x - fb_fast.ln_prob_y_given_x).abs() < 1e-13,
                    "2-row rolling corridor Forward diverged for ly={}: {} vs {}",
                    ly,
                    fb_post.ln_prob_y_given_x,
                    fb_fast.ln_prob_y_given_x
                );
                ln_len_slice_corr = logsumexp2(ln_len_slice_corr, fb_post.ln_prob_y_given_x);
                ln_total_corr = logsumexp2(ln_total_corr, fb_post.ln_prob_y_given_x);
            }

            if ly >= 1 {
                let ln_part_corr =
                    compute_ln_path_partition_corridor(seq_x.len(), ly, &band, reg, 0.8);
                let slice_kraft_corr = (ln_len_slice_corr - ln_part_corr).exp();
                assert!(
                    (slice_kraft_corr - 1.0).abs() < 1e-11,
                    "Corridor pointwise Kraft sum for L_Y={} deviated from 1.0: {}",
                    ly,
                    slice_kraft_corr
                );

                let ln_part_full = compute_ln_path_partition(seq_x.len(), ly, reg, 0.8);
                let slice_full_norm = (ln_len_slice_corr - ln_part_full).exp();
                assert!(
                    slice_full_norm <= 1.0 + 1e-11,
                    "Full-partition normalized corridor mass exceeded 1.0 at L_Y={}: {}",
                    ly,
                    slice_full_norm
                );
            }
        }
        assert!(
            ln_total_corr.exp() <= expected_mass_7 + 1e-11,
            "Total corridor mass {} exceeded full mass {}",
            ln_total_corr.exp(),
            expected_mass_7
        );
    }

    #[test]
    fn test_sparse_large_cohort_accuracy_and_contaminant_isolation() {
        // Deterministic PRNG helper for realistic non-repetitive protein generation
        let mut seed: u64 = 0x5EED_CAFE_BABE_1234;
        fn next_u64(s: &mut u64) -> u64 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *s
        }
        fn gen_aa_seq(len: usize, s: &mut u64) -> Vec<u8> {
            (0..len)
                .map(|_| AA_CHARS[(next_u64(s) >> 32) as usize % 20])
                .collect()
        }
        fn mutate_aa(base: &[u8], rate_pct: usize, s: &mut u64) -> Vec<u8> {
            base.iter()
                .map(|&aa| {
                    if (next_u64(s) >> 24) as usize % 100 < rate_pct {
                        let cand = AA_CHARS[(next_u64(s) >> 32) as usize % 20];
                        if cand == aa {
                            AA_CHARS[((next_u64(s) >> 16) as usize % 19 + 1) % 20]
                        } else {
                            cand
                        }
                    } else {
                        aa
                    }
                })
                .collect()
        }

        let anc_core = gen_aa_seq(90, &mut seed);
        let clade_a_proto = mutate_aa(&anc_core, 10, &mut seed);
        let clade_b_proto = mutate_aa(&anc_core, 12, &mut seed);

        let mut raw_seqs: Vec<RawSeq> = Vec::with_capacity(36);
        let mut expected_homologs: Vec<String> = Vec::with_capacity(32);

        // 16 Clade A sequences
        for i in 0..16 {
            let id = format!("CladeA_{}", i);
            let bytes = mutate_aa(&clade_a_proto, 6, &mut seed);
            expected_homologs.push(id.clone());
            raw_seqs.push(RawSeq {
                id,
                raw_bytes: bytes,
            });
        }

        // 16 Clade B sequences: full-length, N-fragments, C-fragments, clean interior del, clean interior ins
        let ins_block = gen_aa_seq(6, &mut seed);
        for i in 0..16 {
            let id = format!("CladeB_{}", i);
            let base = mutate_aa(&clade_b_proto, 6, &mut seed);
            let bytes = match i {
                0 | 1 => base[..60].to_vec(),                      // N-terminal fragment (67%)
                2 | 3 => base[30..].to_vec(),                      // C-terminal fragment (67%)
                4 | 5 => {
                    // Clean 6-AA interior deletion
                    let mut v = base[..40].to_vec();
                    v.extend_from_slice(&base[46..]);
                    v
                }
                6 | 7 => {
                    // Clean 6-AA interior insertion
                    let mut v = base[..45].to_vec();
                    v.extend_from_slice(&ins_block);
                    v.extend_from_slice(&base[45..]);
                    v
                }
                _ => base,
            };
            expected_homologs.push(id.clone());
            raw_seqs.push(RawSeq {
                id,
                raw_bytes: bytes,
            });
        }

        // 2 reversed controls (mirror of CladeA_0, and mirror of an independent non-homologous protein)
        let mut rev_a = raw_seqs[0].raw_bytes.clone();
        rev_a.reverse();
        raw_seqs.push(RawSeq {
            id: "Control_RevA".into(),
            raw_bytes: rev_a,
        });
        let mut rev_indep = gen_aa_seq(90, &mut seed);
        rev_indep.reverse();
        raw_seqs.push(RawSeq {
            id: "Control_RevIndep".into(),
            raw_bytes: rev_indep,
        });

        // 2 completely unrelated non-homologous proteins
        raw_seqs.push(RawSeq {
            id: "Control_Unrel1".into(),
            raw_bytes: gen_aa_seq(88, &mut seed),
        });
        raw_seqs.push(RawSeq {
            id: "Control_Unrel2".into(),
            raw_bytes: gen_aa_seq(92, &mut seed),
        });

        let ledger = analyze_unaligned_cohort(&raw_seqs, Some(SeqMode::Protein));
        assert_eq!(ledger.num_sequences, 36);
        assert_eq!(
            ledger.num_homology_components, 5,
            "Expected exact 5 components (32 homologs + 4 isolated controls), got {}",
            ledger.num_homology_components
        );

        let main_comp = ledger
            .components
            .iter()
            .find(|c| c.sequence_ids.len() == 32)
            .expect("Expected main component of size 32");

        for hid in &expected_homologs {
            assert!(
                main_comp.sequence_ids.contains(hid),
                "Missing homologous sequence {} from main component!",
                hid
            );
        }
        for ctrl_id in [
            "Control_RevA",
            "Control_RevIndep",
            "Control_Unrel1",
            "Control_Unrel2",
        ] {
            assert!(
                !main_comp.sequence_ids.contains(&ctrl_id.to_string()),
                "False merge of control {} into main component!",
                ctrl_id
            );
            let singleton = ledger
                .components
                .iter()
                .find(|c| c.sequence_ids == vec![ctrl_id.to_string()])
                .expect("Control must remain isolated as singleton component");
            assert_eq!(singleton.num_global_fine_cols, 0);
        }

        // Lossless round-trip across ALL components
        let raw_by_id: std::collections::HashMap<&str, &[u8]> = raw_seqs
            .iter()
            .map(|r| (r.id.as_str(), r.raw_bytes.as_slice()))
            .collect();
        for comp in &ledger.components {
            for (id, aln) in &comp.aligned_fasta {
                let stripped: Vec<u8> = aln.bytes().filter(|&b| b != b'-').collect();
                assert_eq!(
                    stripped.as_slice(),
                    raw_by_id[id.as_str()],
                    "Lossless round-trip failed for {}",
                    id
                );
            }
        }

        assert!(
            main_comp.num_global_fine_cols >= 75,
            "Expected full GlobalFine core recovery (>= 75 cols of 90 AA core), got {}",
            main_comp.num_global_fine_cols
        );
        assert!(
            main_comp.mean_core_posterior >= 0.90,
            "Expected high mean_core_posterior >= 0.90, got {}",
            main_comp.mean_core_posterior
        );
    }

    #[test]
    fn test_mechanistic_aa_partition_of_unity_detailed_balance_and_saturation() {
        let aa_idx = |ch: u8| AA_CHARS.iter().position(|&c| c == ch).unwrap();
        let raw_pi = [
            0.12, 0.015, 0.07, 0.08, 0.03, 0.09, 0.02, 0.06, 0.07, 0.10, 0.02, 0.04, 0.05, 0.04,
            0.05, 0.06, 0.05, 0.07, 0.01, 0.025,
        ];
        let s_pi: f64 = raw_pi.iter().sum();
        let pi: Vec<f64> = raw_pi.iter().map(|x| x / s_pi).collect();

        for &t in &[0.0, 0.25, 0.65, 1.20, 200.0] {
            let p = build_mechanistic_aa_matrix(t, &pi);
            for a in 0..20 {
                let row_sum: f64 = p[a * 20..(a + 1) * 20].iter().sum();
                assert!(
                    (row_sum - 1.0).abs() < 1e-13,
                    "Row sum at t={t} row={a} was {row_sum}"
                );
                for b in 0..20 {
                    let db_err = (pi[a] * p[a * 20 + b] - pi[b] * p[b * 20 + a]).abs();
                    assert!(
                        db_err < 1e-13,
                        "Detailed balance err {db_err} at t={t} ({a},{b})"
                    );
                    if t == 0.0 {
                        let target = if a == b { 1.0 } else { 0.0 };
                        assert!((p[a * 20 + b] - target).abs() < 1e-13);
                    }
                    if t >= 200.0 {
                        assert!(
                            (p[a * 20 + b] - pi[b]).abs() < 1e-9,
                            "Saturation err at t={t} ({a},{b}): got {} vs pi {}",
                            p[a * 20 + b],
                            pi[b]
                        );
                    }
                }
            }
        }

        let p_twi = build_mechanistic_aa_matrix(1.20, &pi);
        for &(ca, cb) in &[
            (b'I', b'V'),
            (b'F', b'Y'),
            (b'Y', b'W'),
            (b'K', b'R'),
            (b'D', b'E'),
            (b'N', b'Q'),
        ] {
            let a = aa_idx(ca);
            let b = aa_idx(cb);
            let lo = (p_twi[a * 20 + b] / pi[b]).ln();
            assert!(
                lo > 0.2,
                "Expected twilight conservative log-odds > 0.2 for {}{}, got {lo}",
                ca as char,
                cb as char
            );
        }
        for &(ca, cb) in &[(b'I', b'D'), (b'L', b'K'), (b'W', b'G'), (b'V', b'E')] {
            let a = aa_idx(ca);
            let b = aa_idx(cb);
            let lo = (p_twi[a * 20 + b] / pi[b]).ln();
            assert!(
                lo < -1.0,
                "Expected clash log-odds < -1.0 for {}{}, got {lo}",
                ca as char,
                cb as char
            );
        }
    }

    #[test]
    fn test_cohort_biophysical_dirichlet_column_mdl() {
        let counts_of = |res: &[u8]| -> [f64; 20] {
            let mut c = [0.0f64; 20];
            for &r in res {
                let idx = AA_CHARS.iter().position(|&x| x == r).unwrap();
                c[idx] += 1.0;
            }
            c
        };
        let delta_of = |a: &[u8], b: &[u8]| -> f64 {
            let ca = counts_of(a);
            let cb = counts_of(b);
            let mut cj = [0.0f64; 20];
            for k in 0..20 {
                cj[k] = ca[k] + cb[k];
            }
            column_niche_mixture_ln_prob(SeqMode::Protein, &cj)
                - column_niche_mixture_ln_prob(SeqMode::Protein, &ca)
                - column_niche_mixture_ln_prob(SeqMode::Protein, &cb)
        };

        let d_hyd = delta_of(b"IILIM", b"VVVVM");
        assert!(
            d_hyd > 1.0,
            "Expected hydrophobic twilight delta_col > 1.0, got {d_hyd}"
        );
        let d_arom = delta_of(b"FFYF", b"YYWY");
        assert!(
            d_arom > 1.0,
            "Expected aromatic twilight delta_col > 1.0, got {d_arom}"
        );
        let d_clash = delta_of(b"ILVIM", b"DEKRD");
        assert!(
            d_clash < -4.0,
            "Expected cross-niche clash delta_col < -4.0, got {d_clash}"
        );
    }

    fn build_two_clade_fixture_ledger() -> crate::multiseq::CohortHomologyLedger {
        use crate::multiseq::{
            AnnotatedColumn, CohortHomologyLedger, ColumnScale, HomologyComponentReport,
            HorizonBubbleReport, TreeNodeRecord,
        };
        // Topology: ((A1:0.12,A2:0.18)N0:0.45,(B1:0.22,B2:0.28)N1:0.55)N2;
        let tree_nodes = vec![
            TreeNodeRecord {
                node_id: 0,
                is_leaf: true,
                label: "A1".to_string(),
                left_node_id: None,
                right_node_id: None,
                len_left: 0.0,
                len_right: 0.0,
                local_seq_idx: Some(0),
            },
            TreeNodeRecord {
                node_id: 1,
                is_leaf: true,
                label: "A2".to_string(),
                left_node_id: None,
                right_node_id: None,
                len_left: 0.0,
                len_right: 0.0,
                local_seq_idx: Some(1),
            },
            TreeNodeRecord {
                node_id: 2,
                is_leaf: true,
                label: "B1".to_string(),
                left_node_id: None,
                right_node_id: None,
                len_left: 0.0,
                len_right: 0.0,
                local_seq_idx: Some(2),
            },
            TreeNodeRecord {
                node_id: 3,
                is_leaf: true,
                label: "B2".to_string(),
                left_node_id: None,
                right_node_id: None,
                len_left: 0.0,
                len_right: 0.0,
                local_seq_idx: Some(3),
            },
            TreeNodeRecord {
                node_id: 4,
                is_leaf: false,
                label: "N0".to_string(),
                left_node_id: Some(0),
                right_node_id: Some(1),
                len_left: 0.12,
                len_right: 0.18,
                local_seq_idx: None,
            },
            TreeNodeRecord {
                node_id: 5,
                is_leaf: false,
                label: "N1".to_string(),
                left_node_id: Some(2),
                right_node_id: Some(3),
                len_left: 0.22,
                len_right: 0.28,
                local_seq_idx: None,
            },
            TreeNodeRecord {
                node_id: 6,
                is_leaf: false,
                label: "N2".to_string(),
                left_node_id: Some(4),
                right_node_id: Some(5),
                len_left: 0.45,
                len_right: 0.55,
                local_seq_idx: None,
            },
        ];

        // Columns (total 34):
        //  0..10: GlobalFine (owner=6, all 4 present)
        // 10..12: GlobalFine internal deletion in A2 (owner=6, A1,B1,B2 present)
        // 12..14: CladeLocalFine N0 loop part 1 (owner=4, A1,A2 present)
        // 14..15: CladeLocalFine N0 loop gap in A2 (owner=4, A1 present; Wait: keep occ=1 or make N0 owner=4)
        // 15..18: CladeLocalFine N0 loop part 2 (owner=4, A1,A2 present)
        // 18..24: CladeLocalFine N1 loop (owner=5, B1,B2 present)
        // 24..34: GlobalFine right anchor (owner=6, all 4 present)
        let row_a1 = "ACDEFGHIKLMNPQRSTVWY------ACDEFGHIKL";
        let row_a2 = "ACDEFGHIKL--PQ-STVWY------ACDEFGHIKL";
        let row_b1 = "ACDEFGHIKLMN--------DEFGHIACDEFGHIKL";
        let row_b2 = "ACDEFGHIKLMN--------DEFGHIACDEFGHIKL";
        assert_eq!(row_a1.len(), 36);
        assert_eq!(row_a2.len(), 36);
        assert_eq!(row_b1.len(), 36);
        assert_eq!(row_b2.len(), 36);
        let strip_gaps = |s: &str| s.chars().filter(|&c| c != '-').collect::<String>();

        let mut cols = Vec::with_capacity(36);
        for c in 0..36usize {
            let (scale, owner, occ, p_f, p_b) = if c < 10 {
                (ColumnScale::GlobalFine, 6, 4, 0.99, 0.01)
            } else if c < 12 {
                (ColumnScale::GlobalFine, 6, 3, 0.95, 0.02)
            } else if c < 14 {
                (ColumnScale::CladeLocalFine, 4, 2, 0.90, 0.72)
            } else if c < 15 {
                (ColumnScale::CladeLocalFine, 4, 1, 0.85, 0.75)
            } else if c < 20 {
                (ColumnScale::CladeLocalFine, 4, 2, 0.90, 0.72)
            } else if c < 26 {
                (ColumnScale::CladeLocalFine, 5, 2, 0.88, 0.74)
            } else {
                (ColumnScale::GlobalFine, 6, 4, 0.99, 0.01)
            };
            cols.push(AnnotatedColumn {
                col_index: c,
                scale,
                owner_node_id: owner,
                num_present: occ,
                mean_fine_posterior: p_f,
                coarse_bubble_posterior: p_b,
            });
        }

        // Unaligned lengths:
        // A1: 10 + 2 + 8 + 10 = 30 (loop is tokens 12..20, start=12 len=8)
        // A2: 10 + 0 + 7 + 10 = 27 (loop is tokens 10..17, start=10 len=7)
        // B1: 10 + 2 + 6 + 10 = 28 (loop is tokens 12..18, start=12 len=6)
        // B2: 28
        let bubble = HorizonBubbleReport {
            tree_merge_step: 3,
            split_node_id: 6,
            left_node_id: 4,
            right_node_id: 5,
            left_clade_size: 2,
            right_clade_size: 2,
            left_col_span: 8,
            right_col_span: 6,
            left_intervals: vec![(0, 12, 8), (1, 10, 7)],
            right_intervals: vec![(2, 12, 6), (3, 12, 6)],
            dominant_biophys_class: Some("Polar_SerThr".to_string()),
            macro_biophys_gain_nats: 2.34,
            delta_hydropathy: -0.45,
            delta_charge: 0.12,
        };

        CohortHomologyLedger {
            mode: SeqMode::Protein,
            mito_tga_trp: false,
            num_sequences: 4,
            raw_null_cohort_nats: 320.0,
            provenance_cohort_nats: 180.0,
            net_cohort_gain_nats: 140.0,
            num_homology_components: 1,
            components: vec![HomologyComponentReport {
                component_id: 0,
                sequence_ids: vec!["A1".into(), "A2".into(), "B1".into(), "B2".into()],
                newick_tree: "((A1:0.12000,A2:0.18000)N0:0.45000,(B1:0.22000,B2:0.28000)N1:0.55000)N2;".to_string(),
                root_node_id: 6,
                tree_nodes,
                unaligned_seqs: vec![
                    ("A1".into(), strip_gaps(row_a1)),
                    ("A2".into(), strip_gaps(row_a2)),
                    ("B1".into(), strip_gaps(row_b1)),
                    ("B2".into(), strip_gaps(row_b2)),
                ],
                num_columns: 36,
                num_global_fine_cols: 22,
                num_clade_local_cols: 14,
                num_lineage_private_cols: 0,
                mean_core_posterior: 0.985,
                horizon_bubbles: vec![bubble],
                frameshifts_repaired: Vec::new(),
                aligned_fasta: vec![
                    ("A1".into(), row_a1.into()),
                    ("A2".into(), row_a2.into()),
                    ("B1".into(), row_b1.into()),
                    ("B2".into(), row_b2.into()),
                ],
                columns: cols,
            }],
            pairwise_summary: Vec::new(),
        }
    }

    #[test]
    fn test_hnex_hdb_lossless_roundtrip_and_horizon_gap_semantics() {
        use crate::storage::{
            read_hdb_file, read_hnex_file, render_component_matrix, write_hdb_file,
            write_hnex_file, write_hnex_str, HierarchicalArchive,
        };

        let ledger = build_two_clade_fixture_ledger();
        let arch = HierarchicalArchive::from_ledger(&ledger);
        assert!(arch.components[0].blocks.len() < 36);

        let hnex0 = write_hnex_str(&arch);
        assert!(hnex0.contains("FORMAT DATATYPE=PROTEIN GAP=- MISSING=.;"));

        let mat0 = render_component_matrix(&arch.components[0], &arch.seqs, arch.mode, true);
        // Check Clade A deletion emits '-' and Clade A/B horizon emits '.'
        assert_eq!(mat0[0].1, "ACDEFGHIKLMNPQRSTVWY......ACDEFGHIKL");
        assert_eq!(mat0[1].1, "ACDEFGHIKL--PQ-STVWY......ACDEFGHIKL");
        assert_eq!(mat0[2].1, "ACDEFGHIKLMN........DEFGHIACDEFGHIKL");
        assert_eq!(mat0[3].1, "ACDEFGHIKLMN........DEFGHIACDEFGHIKL");

        let tmp_dir = std::env::temp_dir();
        let pid = std::process::id();
        let p_hnex1 = tmp_dir.join(format!("squint_test_{pid}_1.hnex"));
        let p_hdb = tmp_dir.join(format!("squint_test_{pid}.hdb"));
        let p_hnex2 = tmp_dir.join(format!("squint_test_{pid}_2.hnex"));

        write_hnex_file(&p_hnex1, &arch).unwrap();
        let arch_from_hnex = read_hnex_file(&p_hnex1).unwrap();
        write_hdb_file(&p_hdb, &arch_from_hnex).unwrap();
        let arch_from_hdb = read_hdb_file(&p_hdb).unwrap();
        let hnex2 = write_hnex_str(&arch_from_hdb);
        write_hnex_file(&p_hnex2, &arch_from_hdb).unwrap();

        let _ = std::fs::remove_file(&p_hnex1);
        let _ = std::fs::remove_file(&p_hdb);
        let _ = std::fs::remove_file(&p_hnex2);

        assert_eq!(hnex0, hnex2, "Lossless .hnex -> .hdb -> .hnex mismatch");
        assert_eq!(arch.seqs.len(), arch_from_hdb.seqs.len());
        assert_eq!(
            arch.components[0].blocks.len(),
            arch_from_hdb.components[0].blocks.len()
        );
        assert_eq!(
            arch.components[0].horizon_bubbles.len(),
            arch_from_hdb.components[0].horizon_bubbles.len()
        );
    }

    #[test]
    fn test_hdb_to_hnex_clade_and_spatial_region_subsetting() {
        use crate::multiseq::ColumnScale;
        use crate::storage::{
            read_hdb_file, render_component_matrix, subset_hdb_file, write_hdb_file,
            write_hnex_file, read_hnex_file, HierarchicalArchive, SubsetSpec,
        };

        let ledger = build_two_clade_fixture_ledger();
        let arch = HierarchicalArchive::from_ledger(&ledger);
        let tmp_dir = std::env::temp_dir();
        let pid = std::process::id();
        let p_hdb = tmp_dir.join(format!("squint_sub_{pid}.hdb"));
        write_hdb_file(&p_hdb, &arch).unwrap();

        // (a) Clade N0 subsetting: drops B1/B2 & N1 loop, promotes N0 loop to GlobalFine!
        let sub_clade = subset_hdb_file(
            &p_hdb,
            &SubsetSpec {
                clade: Some("N0".to_string()),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(sub_clade.seqs.len(), 2);
        let c0 = &sub_clade.components[0];
        assert_eq!(c0.num_columns, 30); // 36 - 6 N1-only loop cols
        assert_eq!(c0.num_clade_local_cols, 0);
        // Check patristic distance A1-A2 == 0.12 + 0.18 = 0.30
        let root = &c0.tree_nodes[c0.root_node_id];
        assert!(((root.len_left + root.len_right) - 0.30).abs() < 1e-9);
        assert!(c0.blocks.iter().any(|b| b.scale == ColumnScale::GlobalFine));
        let mat_clade = render_component_matrix(c0, &sub_clade.seqs, sub_clade.mode, true);
        assert!(!mat_clade[0].1.contains('.') && !mat_clade[1].1.contains('.'));

        // (b) Taxa subsetting {A1, B1, B2}: N0 contracts away preserving d(A1, root) == 0.12 + 0.45 == 0.57
        let sub_taxa = subset_hdb_file(
            &p_hdb,
            &SubsetSpec {
                taxa: Some(vec!["A1".into(), "B1".into(), "B2".into()]),
                ..Default::default()
            },
        )
        .unwrap();
        let ct = &sub_taxa.components[0];
        let rt = &ct.tree_nodes[ct.root_node_id];
        assert!((rt.len_left - 0.57).abs() < 1e-9, "got {}", rt.len_left);
        assert!((rt.len_right - 0.55).abs() < 1e-9, "got {}", rt.len_right);

        // (c) Spatial --region A1:9-22 bracketing the Horizon Bubble via SQLite indexes!
        let sub_reg = subset_hdb_file(
            &p_hdb,
            &SubsetSpec {
                region: Some(("A1".to_string(), 9, 22)),
                ..Default::default()
            },
        )
        .unwrap();
        let seq_a1 = sub_reg.seqs.iter().find(|s| s.name == "A1").unwrap();
        assert_eq!(seq_a1.orig_offset, 8);
        assert!(seq_a1.seq_len < 30);
        assert_eq!(sub_reg.components[0].horizon_bubbles.len(), 1);
        let p_reg_hnex = tmp_dir.join(format!("squint_reg_{pid}.hnex"));
        let p_reg_hdb = tmp_dir.join(format!("squint_reg_{pid}.hdb"));
        write_hnex_file(&p_reg_hnex, &sub_reg).unwrap();
        let reg_reread = read_hnex_file(&p_reg_hnex).unwrap();
        write_hdb_file(&p_reg_hdb, &reg_reread).unwrap();
        let reg_hdb = read_hdb_file(&p_reg_hdb).unwrap();
        assert_eq!(reg_hdb.seqs[0].orig_offset, seq_a1.orig_offset);
        assert_eq!(reg_hdb.components[0].num_columns, sub_reg.components[0].num_columns);

        let _ = std::fs::remove_file(&p_hdb);
        let _ = std::fs::remove_file(&p_reg_hnex);
        let _ = std::fs::remove_file(&p_reg_hdb);
    }

    #[test]
    fn test_multi_component_and_codon_frameshift_archive_roundtrip() {
        use crate::storage::{
            read_hnex_file, subset_hdb_file, write_hdb_file, write_hnex_file, write_hnex_str,
            HierarchicalArchive, SubsetSpec,
        };

        let fam1_a = "ATGAAGGCCTTCAACTTCAACTTCAACAAGTCCAAGGCCTTCAACGTGAAGACCTACCTGGACTACTACCTGGACTACAAGGACTACCTGGAC";
        let fam1_b = "ATGAAGGCCTTCAACTTCAACTTCAACAAGTCCAAGGCCTTCAAACGTGAAGACCTACCTGGACTACTACCTGGACTACAAGGACTACCTGGAC";
        let fam1_c = "ATGAAGGCCTTCAACTTCAACTTCAACAAGTCCAAGGCCTTCAACGTGAAGACCTACCTGGACTACTACCTGGACTACAAGGACTACCTGGAC";
        let fam2_a = "GGTCCCTGGTGGTACTGCCAGATCAGGATGCACTGGCCCTACTACTGGTGGATCTGCCAGGGTCCCTGGTGGTACTGCCAGATC";
        let fam2_b = "GGTCCCTGGTGGTACTGCCAGATCAGGATGCACTGGCCCTACTACTGGTGGATCTGCCAGGGTCCCTGGTGGTACTGCCAGATC";
        let raw = vec![
            RawSeq { id: "F1_ref1".into(), raw_bytes: fam1_a.as_bytes().to_vec() },
            RawSeq { id: "F1_fs".into(), raw_bytes: fam1_b.as_bytes().to_vec() },
            RawSeq { id: "F1_ref2".into(), raw_bytes: fam1_c.as_bytes().to_vec() },
            RawSeq { id: "F2_ref1".into(), raw_bytes: fam2_a.as_bytes().to_vec() },
            RawSeq { id: "F2_ref2".into(), raw_bytes: fam2_b.as_bytes().to_vec() },
        ];

        let ledger = analyze_unaligned_cohort(&raw, Some(SeqMode::Codon));
        assert_eq!(ledger.num_homology_components, 2);

        let arch = HierarchicalArchive::from_ledger(&ledger);
        let tmp_dir = std::env::temp_dir();
        let pid = std::process::id();
        let p_hdb = tmp_dir.join(format!("squint_multi_{pid}.hdb"));
        let p_hnex = tmp_dir.join(format!("squint_multi_{pid}_c0.hnex"));
        write_hdb_file(&p_hdb, &arch).unwrap();

        let fs_comp_id = arch
            .components
            .iter()
            .position(|c| c.seq_ids.iter().any(|&sid| arch.seqs[sid].name == "F1_fs"))
            .unwrap();
        let sub_c0 = subset_hdb_file(
            &p_hdb,
            &SubsetSpec {
                component_id: Some(fs_comp_id),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(sub_c0.components.len(), 1);
        assert_eq!(sub_c0.mode, SeqMode::Codon);
        assert!(!sub_c0.components[0].frameshifts_repaired.is_empty());

        let hnex_c0 = write_hnex_str(&sub_c0);
        assert!(hnex_c0.contains("FORMAT DATATYPE=DNA GAP=- MISSING=.;"));
        write_hnex_file(&p_hnex, &sub_c0).unwrap();
        let reread_c0 = read_hnex_file(&p_hnex).unwrap();
        assert_eq!(
            reread_c0.components[0].frameshifts_repaired.len(),
            sub_c0.components[0].frameshifts_repaired.len()
        );
        assert_eq!(reread_c0.mito_tga_trp, sub_c0.mito_tga_trp);

        let _ = std::fs::remove_file(&p_hdb);
        let _ = std::fs::remove_file(&p_hnex);
    }

    #[test]
    fn test_asymmetric_domain_extension_corridor_recovery() {
        let mut seed: u64 = 0xA571_9900_0001_2345;
        fn next_u64(s: &mut u64) -> u64 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *s
        }
        fn gen_aa(len: usize, s: &mut u64) -> Vec<u8> {
            (0..len)
                .map(|_| AA_CHARS[(next_u64(s) >> 32) as usize % 20])
                .collect()
        }
        fn mutate_aa(base: &[u8], rate_pct: usize, s: &mut u64) -> Vec<u8> {
            base.iter()
                .map(|&aa| {
                    if (next_u64(s) >> 24) as usize % 100 < rate_pct {
                        let cand = AA_CHARS[(next_u64(s) >> 32) as usize % 20];
                        if cand == aa {
                            AA_CHARS[((next_u64(s) >> 16) as usize % 19 + 1) % 20]
                        } else {
                            cand
                        }
                    } else {
                        aa
                    }
                })
                .collect()
        }

        let core = gen_aa(85, &mut seed);
        let flank_n1 = gen_aa(300, &mut seed);
        let flank_c1 = gen_aa(280, &mut seed);
        let flank_n2 = gen_aa(290, &mut seed);
        let flank_c2 = gen_aa(270, &mut seed);

        // Include a 2-aa insertion right before the first exact 4-mer anchor inside the core!
        let mut s2 = flank_n1;
        let core2 = mutate_aa(&core, 10, &mut seed);
        s2.extend_from_slice(&core2[..2]);
        s2.extend_from_slice(b"PG");
        s2.extend_from_slice(&core2[2..]);
        s2.extend_from_slice(&flank_c1);

        let mut s3 = flank_n2;
        s3.extend_from_slice(&mutate_aa(&core, 10, &mut seed));
        s3.extend_from_slice(&flank_c2);

        let raw = vec![
            RawSeq {
                id: "Short_Domain".into(),
                raw_bytes: core,
            },
            RawSeq {
                id: "Long_Ext_A".into(),
                raw_bytes: s2,
            },
            RawSeq {
                id: "Long_Ext_B".into(),
                raw_bytes: s3,
            },
        ];

        let ledger = analyze_unaligned_cohort(&raw, Some(SeqMode::Protein));
        assert_eq!(
            ledger.num_homology_components, 1,
            "Asymmetric domain-extension homologs should unite into 1 component"
        );
        let comp = &ledger.components[0];
        assert!(
            comp.num_global_fine_cols >= 70,
            "Expected >= 70 GlobalFine core columns across asymmetric extensions, got {}",
            comp.num_global_fine_cols
        );
    }

    #[test]
    fn test_progressive_merge_rare_leaf_indels_preserve_core_and_bubbles() {
        let mut seed: u64 = 0x51A7_BEEF_9988_7766;
        fn next_u64(s: &mut u64) -> u64 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *s
        }
        fn mutate_aa(base: &[u8], rate_pct: usize, s: &mut u64) -> Vec<u8> {
            base.iter()
                .map(|&aa| {
                    if (next_u64(s) >> 24) as usize % 100 < rate_pct {
                        let cand = AA_CHARS[(next_u64(s) >> 32) as usize % 20];
                        if cand == aa {
                            AA_CHARS[((next_u64(s) >> 16) as usize % 19 + 1) % 20]
                        } else {
                            cand
                        }
                    } else {
                        aa
                    }
                })
                .collect()
        }

        let n_core = b"MKTWCGHCKALAPEYAKAAGKLKAEGSEIRLAKVDATEESDLAQQY";
        let c_core = b"FVTGYHPTILFFKNGELNRPVKYGQRSVDALIGFIKNHQGLPAVCW";
        let loop_a = b"DDSSNNDDSSTTDDSSEENNDDSQ"; // 24 aa Acidic/Polar
        let loop_b = b"GGEEDDPPGGEEDDPPGGE"; // 19 aa Acidic/Special

        let mut raw = Vec::new();
        for k in 0..4usize {
            let mut nc = mutate_aa(n_core, 4, &mut seed);
            // Scattered single-leaf private insertion near core middle on leaf A1
            if k == 1 {
                nc.insert(20, b'Q');
            } else if k == 2 {
                nc.insert(34, b'S');
            }
            let la = mutate_aa(loop_a, 4, &mut seed);
            let mut cc = mutate_aa(c_core, 4, &mut seed);
            if k == 3 {
                cc.insert(15, b'A');
            }
            let mut seq = Vec::new();
            seq.extend_from_slice(&nc);
            seq.extend_from_slice(&la);
            seq.extend_from_slice(&cc);
            raw.push(RawSeq {
                id: format!("CladeA_{k}"),
                raw_bytes: seq,
            });
        }
        for k in 0..4usize {
            let mut nc = mutate_aa(n_core, 6, &mut seed);
            // Nearby single-leaf private insertion in CladeB leaf B1 at core pos 21
            if k == 1 {
                nc.insert(21, b'N');
            } else if k == 0 {
                nc.insert(10, b'T');
            }
            let lb = mutate_aa(loop_b, 4, &mut seed);
            let mut cc = mutate_aa(c_core, 6, &mut seed);
            if k == 2 {
                cc.insert(16, b'G');
            }
            let mut seq = Vec::new();
            seq.extend_from_slice(&nc);
            seq.extend_from_slice(&lb);
            seq.extend_from_slice(&cc);
            raw.push(RawSeq {
                id: format!("CladeB_{k}"),
                raw_bytes: seq,
            });
        }

        let ledger = analyze_unaligned_cohort(&raw, Some(SeqMode::Protein));
        assert_eq!(ledger.num_homology_components, 1);
        let comp = &ledger.components[0];
        assert!(
            comp.num_global_fine_cols >= 84,
            "Rare leaf indels must not collapse GlobalFine core blocks! got {} (clade_local={})",
            comp.num_global_fine_cols,
            comp.num_clade_local_cols
        );
        assert!(
            !comp.horizon_bubbles.is_empty(),
            "Genuine clade-wide disordered loop must still emit a HorizonBubbleReport!"
        );
    }

    #[test]
    fn test_stealth_div3_frameshift_with_terminal_extensions() {
        let gc = GeneticCode::standard();
        let mut seed: u64 = 0xC0D0_F51F_0000_0042;
        fn next_u64(s: &mut u64) -> u64 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *s
        }

        // Stop-free codon pool (only A/C/G or C/G/T safe codons so +1/-1 shifted frame has 0 stop codons)
        let domain_codons = [
            "ATG", "AAG", "GAC", "GCC", "GAG", "CAG", "AAC", "ACC", "AGG", "GGC", "ATC", "CTG",
            "GTG", "TTC", "TAC", "TCC",
        ];
        let ext_codons = ["AAG", "GAC", "GCC", "GAG", "CAG", "AAC", "ACC", "GGC", "CTG", "GTG"];
        let mut anc_codons = Vec::with_capacity(90);
        for c in 0..90 {
            if (30..45).contains(&c) {
                // Inside compensated frameshift block, use strictly stop-free {A,C,G} codons
                let acg_only = ["AAG", "GAC", "GCC", "GAG", "CAG", "AAC", "ACC", "GGC"];
                let idx = (next_u64(&mut seed) >> 32) as usize % acg_only.len();
                anc_codons.push(acg_only[idx]);
            } else {
                let idx = (next_u64(&mut seed) >> 32) as usize % domain_codons.len();
                anc_codons.push(domain_codons[idx]);
            }
        }
        let mut ext_nt = String::new();
        for _ in 0..150 {
            let idx = (next_u64(&mut seed) >> 32) as usize % ext_codons.len();
            ext_nt.push_str(ext_codons[idx]);
        }

        // Build Seq1, Seq2 (90 codons clean), Seq3 (240 codons clean w/ C-term extension),
        // Seq4 (240 codons w/ +1 at codon 30 and -1 at codon 45 -> nt_len % 3 == 0, 0 stops!)
        let nt_dom: String = anc_codons.concat();
        let mut nt_dom_syn = anc_codons.clone();
        // Minor synonymous tweak on Seq2 so it isn't identical bytes
        nt_dom_syn[5] = "AAA";

        let mut nt_fs = String::new();
        for (c, &cod) in anc_codons.iter().enumerate() {
            if c == 30 {
                nt_fs.push('C'); // +1 insertion
            }
            if c == 45 {
                nt_fs.push_str(&cod[1..]); // -1 deletion -> restores mod-3 parity!
            } else {
                nt_fs.push_str(cod);
            }
        }
        nt_fs.push_str(&ext_nt);
        assert_eq!(nt_fs.len() % 3, 0);

        let enc_fs = EncodedSeq::from_raw(
            &RawSeq {
                id: "Seq4_stealth_fs".into(),
                raw_bytes: nt_fs.as_bytes().to_vec(),
            },
            SeqMode::Codon,
            &gc,
        );
        assert!(
            !enc_fs.has_unambiguous_internal_stop,
            "Stealth frameshift fixture should have zero internal stop codons"
        );

        let mut nt_s3 = nt_dom.clone();
        nt_s3.push_str(&ext_nt);

        let raw = vec![
            RawSeq {
                id: "Seq1_clean".into(),
                raw_bytes: nt_dom.into_bytes(),
            },
            RawSeq {
                id: "Seq2_clean".into(),
                raw_bytes: nt_dom_syn.concat().into_bytes(),
            },
            RawSeq {
                id: "Seq3_clean_ext".into(),
                raw_bytes: nt_s3.into_bytes(),
            },
            RawSeq {
                id: "Seq4_stealth_fs".into(),
                raw_bytes: nt_fs.into_bytes(),
            },
        ];

        let ledger = analyze_unaligned_cohort(&raw, Some(SeqMode::Codon));
        assert_eq!(ledger.num_homology_components, 1);
        let comp = &ledger.components[0];
        let fs_seq4: Vec<_> = comp
            .frameshifts_repaired
            .iter()
            .filter(|ev| ev.seq_id == "Seq4_stealth_fs")
            .collect();
        assert!(
            fs_seq4.len() >= 2,
            "Expected compensated +1/-1 stealth frameshift repair on Seq4_stealth_fs, got {:?}",
            comp.frameshifts_repaired
        );
        for clean_id in ["Seq1_clean", "Seq2_clean", "Seq3_clean_ext"] {
            assert!(
                !comp
                    .frameshifts_repaired
                    .iter()
                    .any(|ev| ev.seq_id == clean_id),
                "False frameshift repair triggered on clean sequence {}!",
                clean_id
            );
        }
    }

    #[test]
    fn test_bionj_fragment_rescue_and_topology_free_refinement() {
        use crate::multiseq::ColumnScale;
        use crate::storage::{
            read_hdb_file, read_hnex_file, render_component_matrix, subset_hdb_file,
            write_hdb_file, write_hnex_file, write_hnex_str, HierarchicalArchive, SubsetSpec,
        };

        let mut seed: u64 = 0xB10A_7090_F4A6_0021;
        fn next_u64(s: &mut u64) -> u64 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *s
        }
        fn gen_aa(len: usize, s: &mut u64) -> Vec<u8> {
            (0..len)
                .map(|_| AA_CHARS[(next_u64(s) >> 32) as usize % 20])
                .collect()
        }
        fn mutate_aa(base: &[u8], rate_pct: usize, s: &mut u64) -> Vec<u8> {
            base.iter()
                .map(|&aa| {
                    if (next_u64(s) >> 24) as usize % 100 < rate_pct {
                        let cand = AA_CHARS[(next_u64(s) >> 32) as usize % 20];
                        if cand == aa {
                            AA_CHARS[((next_u64(s) >> 16) as usize % 19 + 1) % 20]
                        } else {
                            cand
                        }
                    } else {
                        aa
                    }
                })
                .collect()
        }

        // 135 aa conserved structural ancestor + N/C flanks
        let anc_core = gen_aa(135, &mut seed);
        let fam_a_core = mutate_aa(&anc_core, 10, &mut seed);
        let fam_b_core = mutate_aa(&anc_core, 30, &mut seed);
        let flank_n = gen_aa(30, &mut seed);
        let flank_c = gen_aa(30, &mut seed);

        // Domain-truncated ~35% central fragment of Family A (`anc_core` slice ~48 aa vs ~195 aa full)
        let frag_a = mutate_aa(&fam_a_core[42..90], 8, &mut seed);

        let make_full = |core: &[u8], s: &mut u64| -> Vec<u8> {
            let mut out = mutate_aa(&flank_n, 15, s);
            out.extend_from_slice(core);
            out.extend_from_slice(&mutate_aa(&flank_c, 15, s));
            out
        };

        // Negative control: unrelated synthetic reversed/permuted non-homolog
        let mut rev_seq = gen_aa(140, &mut seed);
        rev_seq.reverse();

        let raw = vec![
            RawSeq { id: "A1".into(), raw_bytes: make_full(&mutate_aa(&fam_a_core, 8, &mut seed), &mut seed) },
            RawSeq { id: "A2".into(), raw_bytes: make_full(&mutate_aa(&fam_a_core, 10, &mut seed), &mut seed) },
            RawSeq { id: "A3".into(), raw_bytes: make_full(&mutate_aa(&fam_a_core, 12, &mut seed), &mut seed) },
            RawSeq { id: "A_frag".into(), raw_bytes: frag_a },
            RawSeq { id: "B1".into(), raw_bytes: make_full(&mutate_aa(&fam_b_core, 10, &mut seed), &mut seed) },
            RawSeq { id: "B2".into(), raw_bytes: make_full(&mutate_aa(&fam_b_core, 12, &mut seed), &mut seed) },
            RawSeq { id: "REV_CTRL".into(), raw_bytes: rev_seq },
        ];

        let ledger = analyze_unaligned_cohort(&raw, Some(SeqMode::Protein));
        assert_eq!(
            ledger.num_homology_components, 2,
            "Expected REV_CTRL isolated and all 6 homologs (including 35% A_frag) unified"
        );

        let hom_comp = ledger
            .components
            .iter()
            .find(|c| c.sequence_ids.len() == 6)
            .expect("Expected 6-sequence homologous component");
        let rev_comp = ledger
            .components
            .iter()
            .find(|c| c.sequence_ids.len() == 1)
            .expect("Expected singleton REV_CTRL component");
        assert_eq!(rev_comp.sequence_ids[0], "REV_CTRL");
        assert!(
            hom_comp.num_global_fine_cols >= 100,
            "Expected >= 100 GlobalFine columns across full core, got {}",
            hom_comp.num_global_fine_cols
        );

        // Specifically verify `A_frag` participates directly in GlobalFine columns with the other 5 homologs
        let frag_row_idx = hom_comp
            .aligned_fasta
            .iter()
            .position(|(id, _)| id == "A_frag")
            .unwrap();
        let frag_bytes = hom_comp.aligned_fasta[frag_row_idx].1.as_bytes();
        let frag_global_fine_cols = hom_comp
            .columns
            .iter()
            .filter(|col| {
                col.scale == ColumnScale::GlobalFine
                    && col.num_present >= 6
                    && frag_bytes[col.col_index] != b'-'
            })
            .count();
        assert!(
            frag_global_fine_cols >= 35,
            "Expected >= 35 6-way GlobalFine columns containing A_frag, got {}",
            frag_global_fine_cols
        );

        // Verify .hnex <-> .hdb round-trip & subsetting preserve exact LCA invariants on refined core
        let arch = HierarchicalArchive::from_ledger(&ledger);
        let hnex0 = write_hnex_str(&arch);
        let tmp_dir = std::env::temp_dir();
        let pid = std::process::id();
        let p_hnex = tmp_dir.join(format!("squint_p2_{pid}.hnex"));
        let p_hdb = tmp_dir.join(format!("squint_p2_{pid}.hdb"));

        write_hnex_file(&p_hnex, &arch).unwrap();
        let arch_reread = read_hnex_file(&p_hnex).unwrap();
        write_hdb_file(&p_hdb, &arch_reread).unwrap();
        let arch_hdb = read_hdb_file(&p_hdb).unwrap();
        assert_eq!(hnex0, write_hnex_str(&arch_hdb));

        let sub = subset_hdb_file(
            &p_hdb,
            &SubsetSpec {
                taxa: Some(vec!["A_frag".into(), "A1".into(), "B1".into()]),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(sub.components.len(), 1);
        let mat = render_component_matrix(&sub.components[0], &sub.seqs, sub.mode, true);
        assert_eq!(mat.len(), 3);
        for (name, row) in mat {
            let clean: String = row.chars().filter(|&c| c != '-' && c != '.').collect();
            let orig = raw.iter().find(|r| r.id == name).unwrap();
            assert_eq!(clean.as_bytes(), orig.raw_bytes.as_slice());
        }

        let _ = std::fs::remove_file(&p_hnex);
        let _ = std::fs::remove_file(&p_hdb);
    }

    #[test]
    fn test_linear_corridor_matches_logspace_across_modes_and_lengths() {
        let gc = GeneticCode::standard();
        let mut seed: u64 = 0x1122_3344_5566_7788;
        let mut next_u64 = || -> u64 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed
        };

        let nt_chars = [b'A', b'C', b'G', b'T'];
        let mut sense_codons: Vec<[u8; 3]> = Vec::new();
        for &a in &nt_chars {
            for &b in &nt_chars {
                for &c in &nt_chars {
                    let tri = [a, b, c];
                    if tri != *b"TAA" && tri != *b"TAG" && tri != *b"TGA" {
                        sense_codons.push(tri);
                    }
                }
            }
        }

        for (mode, len_x, len_y, reg_idx, pml, mutate_pct) in [
            (SeqMode::Dna, 220usize, 240usize, 1usize, 0.8f64, 0usize),
            (SeqMode::Dna, 260usize, 250usize, 4usize, 350.0f64, 25usize),
            (SeqMode::Protein, 650usize, 635usize, 2usize, 350.0f64, 0usize),
            (SeqMode::Protein, 420usize, 440usize, 5usize, 0.8f64, 30usize),
            (SeqMode::Codon, 180usize, 185usize, 1usize, 350.0f64, 0usize),
            (SeqMode::Codon, 160usize, 150usize, 7usize, 0.8f64, 35usize),
        ] {
            let (raw_x_bytes, raw_y_bytes) = match mode {
                SeqMode::Dna => {
                    let x: Vec<u8> = (0..len_x)
                        .map(|_| nt_chars[(next_u64() >> 32) as usize % 4])
                        .collect();
                    let mut y: Vec<u8> = (0..len_y)
                        .map(|k| {
                            let base = x[k.min(len_x - 1)];
                            if (next_u64() >> 24) as usize % 100 < mutate_pct {
                                nt_chars[(next_u64() >> 32) as usize % 4]
                            } else {
                                base
                            }
                        })
                        .collect();
                    if mutate_pct == 0 && len_y > len_x {
                        for k in len_x..len_y {
                            y[k] = nt_chars[(next_u64() >> 32) as usize % 4];
                        }
                    }
                    (x, y)
                }
                SeqMode::Protein => {
                    let x: Vec<u8> = (0..len_x)
                        .map(|_| AA_CHARS[(next_u64() >> 32) as usize % 20])
                        .collect();
                    let y: Vec<u8> = (0..len_y)
                        .map(|k| {
                            if k < len_x && (next_u64() >> 24) as usize % 100 >= mutate_pct {
                                x[k]
                            } else {
                                AA_CHARS[(next_u64() >> 32) as usize % 20]
                            }
                        })
                        .collect();
                    (x, y)
                }
                SeqMode::Codon => {
                    let cx: Vec<[u8; 3]> = (0..len_x)
                        .map(|_| sense_codons[(next_u64() >> 32) as usize % sense_codons.len()])
                        .collect();
                    let cy: Vec<[u8; 3]> = (0..len_y)
                        .map(|k| {
                            if k < len_x && (next_u64() >> 24) as usize % 100 >= mutate_pct {
                                cx[k]
                            } else {
                                sense_codons[(next_u64() >> 32) as usize % sense_codons.len()]
                            }
                        })
                        .collect();
                    (cx.concat(), cy.concat())
                }
            };

            let seq_x = EncodedSeq::from_raw(
                &RawSeq {
                    id: "X".into(),
                    raw_bytes: raw_x_bytes,
                },
                mode,
                &gc,
            );
            let seq_y = EncodedSeq::from_raw(
                &RawSeq {
                    id: "Y".into(),
                    raw_bytes: raw_y_bytes,
                },
                mode,
                &gc,
            );
            let bg = default_background_pi(mode, &gc);
            let prof_x = SingleSeqNullProfile::compute(&seq_x.tokens, seq_x.alphabet_size(), &bg);
            let prof_y = SingleSeqNullProfile::compute(&seq_y.tokens, seq_y.alphabet_size(), &bg);
            let px = PreparedSeq::new(seq_x.clone(), &gc);
            let py = PreparedSeq::new(seq_y.clone(), &gc);
            let tables = get_global_regime_tables(mode, &gc);
            let reg = tables.regimes[reg_idx];
            let full_corr = CorridorBounds::full(seq_x.len(), seq_y.len());

            let p_part = compute_ln_path_partition(seq_x.len(), seq_y.len(), reg, pml);
            let p_part_corr = compute_ln_path_partition_corridor(
                seq_x.len(),
                seq_y.len(),
                &full_corr,
                reg,
                pml,
            );
            assert!(
                (p_part - p_part_corr).abs() < 1e-10,
                "Path partition mismatch for {:?} ({}, {}): {} vs {}",
                mode,
                len_x,
                len_y,
                p_part,
                p_part_corr
            );

            let fb_dense = run_forward_backward_regime(
                &seq_x,
                &seq_y,
                &prof_x,
                &prof_y,
                px.get_local_coarse(reg_idx, &tables),
                &tables.copulas[reg_idx],
                &gc,
                reg,
                pml,
                true,
            );
            let fb_corr = run_forward_backward_regime_corridor(
                &px,
                &py,
                &full_corr,
                px.get_local_coarse(reg_idx, &tables),
                &tables.copulas[reg_idx],
                &tables.ln_r_bases[reg_idx],
                None,
                reg,
                pml,
                true,
            );
            let fb_corr_fast = run_forward_backward_regime_corridor(
                &px,
                &py,
                &full_corr,
                px.get_local_coarse(reg_idx, &tables),
                &tables.copulas[reg_idx],
                &tables.ln_r_bases[reg_idx],
                None,
                reg,
                pml,
                false,
            );

            assert!(
                (fb_dense.ln_prob_y_given_x - fb_corr.ln_prob_y_given_x).abs() < 1e-9,
                "Corridor FB log-prob mismatch for {:?} ({}, {}, pml={}): {} vs {}",
                mode,
                len_x,
                len_y,
                pml,
                fb_dense.ln_prob_y_given_x,
                fb_corr.ln_prob_y_given_x
            );
            assert!(
                (fb_corr.ln_prob_y_given_x - fb_corr_fast.ln_prob_y_given_x).abs() < 1e-12,
                "Fast vs full linear corridor log-prob mismatch for {:?}: {} vs {}",
                mode,
                fb_corr.ln_prob_y_given_x,
                fb_corr_fast.ln_prob_y_given_x
            );

            let max_fine_diff = fb_dense
                .post_fine
                .iter()
                .zip(fb_corr.post_fine_banded.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            assert!(
                max_fine_diff < 1e-9,
                "post_fine max diff {} exceeded 1e-9 for {:?}",
                max_fine_diff,
                mode
            );
            let max_bx_diff = fb_dense
                .post_bx
                .iter()
                .zip(fb_corr.post_bx.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            assert!(
                max_bx_diff < 1e-9,
                "post_bx max diff {} exceeded 1e-9 for {:?}",
                max_bx_diff,
                mode
            );
            let max_by_diff = fb_dense
                .post_by
                .iter()
                .zip(fb_corr.post_by.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            assert!(
                max_by_diff < 1e-9,
                "post_by max diff {} exceeded 1e-9 for {:?}",
                max_by_diff,
                mode
            );
        }
    }

    #[test]
    fn test_ambiguous_iupac_and_nnn_marginalization_kraft_and_alignment() {
        let gc = GeneticCode::standard();
        let mut seed: u64 = 0x9988_7766_5544_3322;
        let mut next_u64 = || -> u64 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed
        };

        // 1. Dense vs corridor < 1e-10 parity under IUPAC ambiguities in Dna, Protein, and Codon
        let nt_exact = [b'A', b'C', b'G', b'T'];
        let nt_iupac = [b'N', b'R', b'Y', b'W'];
        let aa_iupac = [b'X', b'B', b'Z', b'J'];
        let mut sense_codons: Vec<[u8; 3]> = Vec::new();
        for &a in &nt_exact {
            for &b in &nt_exact {
                for &c in &nt_exact {
                    let tri = [a, b, c];
                    if tri != *b"TAA" && tri != *b"TAG" && tri != *b"TGA" {
                        sense_codons.push(tri);
                    }
                }
            }
        }
        let codon_iupac: [[u8; 3]; 4] = [*b"NNN", *b"GGN", *b"GNN", *b"AYR"];

        let dna_x: Vec<u8> = (0..75)
            .map(|k| {
                if k % 11 == 3 {
                    nt_iupac[(next_u64() >> 32) as usize % nt_iupac.len()]
                } else {
                    nt_exact[(next_u64() >> 32) as usize % 4]
                }
            })
            .collect();
        let mut dna_y = dna_x.clone();
        for k in (5..70).step_by(9) {
            dna_y[k] = if k % 2 == 0 { b'N' } else { nt_exact[(next_u64() >> 32) as usize % 4] };
        }

        let prot_x: Vec<u8> = (0..60)
            .map(|k| {
                if k % 10 == 2 {
                    aa_iupac[(next_u64() >> 32) as usize % aa_iupac.len()]
                } else {
                    AA_CHARS[(next_u64() >> 32) as usize % 20]
                }
            })
            .collect();
        let mut prot_y = prot_x.clone();
        for k in (4..56).step_by(8) {
            prot_y[k] = if k % 16 == 4 { b'X' } else { AA_CHARS[(next_u64() >> 32) as usize % 20] };
        }

        let cod_x: Vec<u8> = (0..50)
            .flat_map(|k| {
                if k % 9 == 4 {
                    codon_iupac[(next_u64() >> 32) as usize % codon_iupac.len()]
                } else {
                    sense_codons[(next_u64() >> 32) as usize % sense_codons.len()]
                }
            })
            .collect();
        let mut cod_y = cod_x.clone();
        for k in [12usize, 13, 14, 30, 31] {
            let tri = codon_iupac[k % codon_iupac.len()];
            cod_y[k * 3..k * 3 + 3].copy_from_slice(&tri);
        }

        for (mode, rx, ry, reg_idx, pml) in [
            (SeqMode::Dna, dna_x, dna_y, 2usize, 180.0f64),
            (SeqMode::Protein, prot_x, prot_y, 3usize, 240.0f64),
            (SeqMode::Codon, cod_x, cod_y, 2usize, 200.0f64),
        ] {
            let seq_x = EncodedSeq::from_raw(&RawSeq { id: "AX".into(), raw_bytes: rx }, mode, &gc);
            let seq_y = EncodedSeq::from_raw(&RawSeq { id: "AY".into(), raw_bytes: ry }, mode, &gc);
            let px = PreparedSeq::new(seq_x.clone(), &gc);
            let py = PreparedSeq::new(seq_y.clone(), &gc);
            let tables = get_global_regime_tables(mode, &gc);
            let reg = tables.regimes[reg_idx];
            let full_corr = CorridorBounds::full(seq_x.len(), seq_y.len());

            let fb_dense = run_forward_backward_regime(
                &seq_x,
                &seq_y,
                &px.prof,
                &py.prof,
                px.get_local_coarse(reg_idx, &tables),
                &tables.copulas[reg_idx],
                &gc,
                reg,
                pml,
                true,
            );
            let fb_corr = run_forward_backward_regime_corridor(
                &px,
                &py,
                &full_corr,
                px.get_local_coarse(reg_idx, &tables),
                &tables.copulas[reg_idx],
                &tables.ln_r_bases[reg_idx],
                None,
                reg,
                pml,
                true,
            );

            assert!(
                (fb_dense.ln_prob_y_given_x - fb_corr.ln_prob_y_given_x).abs() < 1e-10,
                "IUPAC dense vs corridor log-prob mismatch for {:?}: {} vs {}",
                mode,
                fb_dense.ln_prob_y_given_x,
                fb_corr.ln_prob_y_given_x
            );
            let max_fine_diff = fb_dense
                .post_fine
                .iter()
                .zip(fb_corr.post_fine_banded.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            assert!(
                max_fine_diff < 1e-10,
                "IUPAC post_fine diff {} exceeded 1e-10 for {:?}",
                max_fine_diff,
                mode
            );
            let max_bx_diff = fb_dense
                .post_bx
                .iter()
                .zip(fb_corr.post_bx.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            assert!(
                max_bx_diff < 1e-10,
                "IUPAC post_bx diff {} exceeded 1e-10 for {:?}",
                max_bx_diff,
                mode
            );
            let max_by_diff = fb_dense
                .post_by
                .iter()
                .zip(fb_corr.post_by.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            assert!(
                max_by_diff < 1e-10,
                "IUPAC post_by diff {} exceeded 1e-10 for {:?}",
                max_by_diff,
                mode
            );
        }

        // 2. All-wildcard sequences yield zero SingleSeqNullProfile ln_m0 and <= 1e-9 MDL homology gain
        for (mode, wild_bytes) in [
            (SeqMode::Dna, vec![b'N'; 60]),
            (SeqMode::Protein, vec![b'X'; 60]),
            (SeqMode::Codon, b"NNN".repeat(60)),
        ] {
            let wild_seq =
                EncodedSeq::from_raw(&RawSeq { id: "WILD".into(), raw_bytes: wild_bytes }, mode, &gc);
            let bg = default_background_pi(mode, &gc);
            let prof = SingleSeqNullProfile::compute_with_masks(
                &wild_seq.tokens,
                &wild_seq.ambig_masks,
                wild_seq.alphabet_size(),
                &bg,
            );
            assert!(
                prof.payload_nats.abs() < 1e-12
                    && prof.site_ln_m0.iter().all(|&v| v.abs() < 1e-12),
                "Expected all-wildcard payload_nats == 0 for {:?}, got {}",
                mode,
                prof.payload_nats
            );
            if mode == SeqMode::Codon {
                let real_cods: Vec<u8> = (0..60)
                    .flat_map(|_| sense_codons[(next_u64() >> 32) as usize % sense_codons.len()])
                    .collect();
                let real_seq =
                    EncodedSeq::from_raw(&RawSeq { id: "REAL".into(), raw_bytes: real_cods }, mode, &gc);
                let pair = evaluate_pair_homology(&real_seq, &wild_seq, &gc, false);
                assert!(
                    pair.mdl_gain_nats <= 1e-9,
                    "Expected real vs all-NNN gain <= 1e-9, got {}",
                    pair.mdl_gain_nats
                );
            }
        }

        // 3. 6-sequence homologous coding cohort with internal 15-codon & 3'-terminal 40-codon NNN spans
        let anc: Vec<[u8; 3]> = (0..120)
            .map(|_| sense_codons[(next_u64() >> 32) as usize % sense_codons.len()])
            .collect();
        let mut make_child = |pct: usize| -> Vec<u8> {
            anc.iter()
                .flat_map(|&c| {
                    if (next_u64() >> 24) as usize % 100 < pct {
                        sense_codons[(next_u64() >> 32) as usize % sense_codons.len()]
                    } else {
                        c
                    }
                })
                .collect()
        };
        let s0 = make_child(8);
        let s1 = make_child(9);
        let mut s2 = make_child(10);
        let mut s3 = make_child(10);
        let mut s4 = make_child(11);
        let mut s5 = make_child(11);
        for k in 40..55 {
            s2[k * 3..k * 3 + 3].copy_from_slice(b"NNN");
            s3[k * 3..k * 3 + 3].copy_from_slice(b"NNN");
        }
        for k in 80..120 {
            s4[k * 3..k * 3 + 3].copy_from_slice(b"NNN");
            s5[k * 3..k * 3 + 3].copy_from_slice(b"NNN");
        }

        let raw = vec![
            RawSeq { id: "S0".into(), raw_bytes: s0 },
            RawSeq { id: "S1".into(), raw_bytes: s1 },
            RawSeq { id: "S2_INT".into(), raw_bytes: s2 },
            RawSeq { id: "S3_INT".into(), raw_bytes: s3 },
            RawSeq { id: "S4_TAIL".into(), raw_bytes: s4 },
            RawSeq { id: "S5_TAIL".into(), raw_bytes: s5 },
        ];
        let ledger = analyze_unaligned_cohort(&raw, Some(SeqMode::Codon));
        assert_eq!(ledger.num_homology_components, 1);
        let comp = &ledger.components[0];
        assert_eq!(
            comp.num_global_fine_cols, 120,
            "Expected 120 GlobalFine cols without C-terminal duplication, got {}",
            comp.num_global_fine_cols
        );
        assert_eq!(
            comp.num_lineage_private_cols, 0,
            "Expected 0 LineagePrivate cols from NNN blocks, got {}",
            comp.num_lineage_private_cols
        );
        assert!(
            comp.horizon_bubbles.is_empty(),
            "Expected 0 horizon bubbles from NNN blocks, got {}",
            comp.horizon_bubbles.len()
        );
        for (id, aln) in &comp.aligned_fasta {
            let clean: String = aln.chars().filter(|&c| c != '-' && c != '.').collect();
            let orig = raw.iter().find(|r| r.id == *id).unwrap();
            assert_eq!(
                clean.as_bytes(),
                orig.raw_bytes.as_slice(),
                "Verbatim round-trip failed for {}",
                id
            );
        }
    }

    #[test]
    fn test_compensatory_paired_frameshift_detection_and_repair() {
        let gc = GeneticCode::standard();
        let mut seed: u64 = 0x51A7_E9C0_D2B3_0415;
        let mut next_u64 = || -> u64 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed
        };

        let nt_chars = [b'A', b'C', b'G', b'T'];
        let is_stop = |tri: &[u8]| tri == b"TAA" || tri == b"TAG" || tri == b"TGA";
        let mut sense_codons: Vec<[u8; 3]> = Vec::new();
        for &a in &nt_chars {
            for &b in &nt_chars {
                for &c in &nt_chars {
                    let tri = [a, b, c];
                    if !is_stop(&tri) {
                        sense_codons.push(tri);
                    }
                }
            }
        }

        // 90-codon clean ancestor with a stop-free shifted window across codons 34..42
        let mut anc: Vec<[u8; 3]> = (0..90)
            .map(|_| sense_codons[(next_u64() >> 32) as usize % sense_codons.len()])
            .collect();
        // Fixed GC-rich codons in 34..42 guarantee no stop codons in shifted frame or after +/-1 nt shift
        let window_codons: [[u8; 3]; 8] = [
            *b"GCC", *b"ATC", *b"GCG", *b"CCC", *b"GGC", *b"ACC", *b"CCG", *b"ACG",
        ];
        for (off, &wc) in window_codons.iter().enumerate() {
            anc[34 + off] = wc;
        }

        let mut mutate_hom = |pct_nt: usize| -> Vec<u8> {
            let mut out = Vec::with_capacity(270);
            for &cod in &anc {
                let mut cur = cod;
                for _attempt in 0..8 {
                    let mut cand = cod;
                    for p in 0..3 {
                        if (next_u64() >> 24) as usize % 100 < pct_nt {
                            cand[p] = nt_chars[(next_u64() >> 32) as usize % 4];
                        }
                    }
                    if !is_stop(&cand) {
                        cur = cand;
                        break;
                    }
                }
                out.extend_from_slice(&cur);
            }
            out
        };

        let s0 = mutate_hom(3);
        let s1 = mutate_hom(3);
        let s2 = mutate_hom(3);
        let s3 = mutate_hom(3);
        let s4 = mutate_hom(3);

        // FS_COMP: derived from ancestor with -1 nt deletion at codon 35 (+1) and +1 nt insertion at codon 40
        // => net len = 270 nt (len % 3 == 0) and 0 internal stops!
        let mut fs_nt: Vec<u8> = anc.iter().flat_map(|c| c.iter().copied()).collect();
        let del_pos = 35 * 3 + 1;
        fs_nt.remove(del_pos);
        // After deleting 1 nt at 106, old codon 40's end (122) moved to 121; insert 'G' there -> net shift 0
        let ins_pos = 40 * 3 + 1;
        fs_nt.insert(ins_pos, b'G');
        assert_eq!(fs_nt.len(), 270);
        let fs_check = EncodedSeq::from_raw(
            &RawSeq { id: "FS_COMP".into(), raw_bytes: fs_nt.clone() },
            SeqMode::Codon,
            &gc,
        );
        assert!(
            !fs_check.has_unambiguous_internal_stop,
            "Test precondition failed: FS_COMP must have 0 internal stops so Pass B handles it"
        );

        let raw = vec![
            RawSeq { id: "C0".into(), raw_bytes: s0 },
            RawSeq { id: "C1".into(), raw_bytes: s1 },
            RawSeq { id: "C2".into(), raw_bytes: s2 },
            RawSeq { id: "C3".into(), raw_bytes: s3 },
            RawSeq { id: "C4".into(), raw_bytes: s4 },
            RawSeq { id: "FS_COMP".into(), raw_bytes: fs_nt.clone() },
        ];

        let ledger = analyze_unaligned_cohort(&raw, Some(SeqMode::Codon));
        assert_eq!(ledger.num_homology_components, 1);
        let comp = &ledger.components[0];
        assert_eq!(
            comp.frameshifts_repaired.len(),
            2,
            "Expected exactly 2 compensated frameshift events on FS_COMP, got {:?}",
            comp.frameshifts_repaired
        );
        assert!(
            comp.frameshifts_repaired.iter().all(|e| e.seq_id == "FS_COMP"),
            "Expected frameshift repairs ONLY on FS_COMP (0 FPs), got {:?}",
            comp.frameshifts_repaired
        );

        let mut keep = vec![true; fs_nt.len()];
        for ev in &comp.frameshifts_repaired {
            for idx in ev.nt_pos..(ev.nt_pos + ev.phase_shift as usize).min(fs_nt.len()) {
                keep[idx] = false;
            }
        }
        let expected_repaired: Vec<u8> = fs_nt
            .iter()
            .zip(keep.iter())
            .filter_map(|(&b, &k)| if k { Some(b) } else { None })
            .collect();
        let fs_aln = comp
            .aligned_fasta
            .iter()
            .find(|(id, _)| id == "FS_COMP")
            .expect("FS_COMP missing from aligned_fasta");
        let fs_ungapped: Vec<u8> = fs_aln
            .1
            .bytes()
            .filter(|&b| b != b'-' && b != b'.')
            .collect();
        assert_eq!(
            fs_ungapped, expected_repaired,
            "Repaired FS_COMP sequence did not match strict excised subsequence"
        );
        assert!(
            comp.num_global_fine_cols >= 88,
            "Expected >= 88 GlobalFine cols after paired frameshift repair, got {}",
            comp.num_global_fine_cols
        );
    }

    #[test]
    fn test_cross_kingdom_zero_homology_forest_header_calibration() {
        let mut seed: u64 = 0xD3C0_11AA_55BB_7799;
        let mut next_u64 = || -> u64 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed
        };

        let pools: [&[u8]; 8] = [
            b"ACDEFGHIKLMNPQRSTVWY",                     // Uniform 20
            b"AILMVAILMVFWPSTQG",                        // Hydrophobic-rich
            b"STNQSTNQDEKRAGPVL",                        // Polar-rich
            b"DEDEDEDEKRAVLIMST",                        // Acidic-rich
            b"KRHKRHKRSTNQGPVLA",                        // Basic-rich
            b"FWYFWYAILMVGPSCDE",                        // Aromatic/hydrophobic
            b"GPGPGPSASTVAILMKR",                        // Loop/turn Pro-Gly enriched
            b"VLIAMFYWCDEKRHNQST",                       // Distinct uniform-shuffled
        ];
        let lens: [usize; 8] = [108, 124, 138, 152, 168, 184, 200, 218];

        let mut prot_raw = Vec::with_capacity(8);
        let mut cod_raw = Vec::with_capacity(8);

        // Build distinct synonymous codon lists per AA (standard genetic code)
        let aa_to_codons: std::collections::BTreeMap<u8, Vec<[u8; 3]>> = {
            let gc = GeneticCode::standard();
            let mut m: std::collections::BTreeMap<u8, Vec<[u8; 3]>> =
                std::collections::BTreeMap::new();
            for s in 0..gc.num_sense {
                let aa_ch = AA_CHARS[gc.sense_to_aa[s] as usize];
                let cs = gc.codon_to_string(s as u8);
                let cb = cs.as_bytes();
                m.entry(aa_ch).or_default().push([cb[0], cb[1], cb[2]]);
            }
            m
        };

        for k in 0..8 {
            let pool = pools[k];
            let l = lens[k];
            let mut aa_bytes = Vec::with_capacity(l);
            let mut nt_bytes = Vec::with_capacity(l * 3);
            for _ in 0..l {
                let aa = pool[(next_u64() >> 32) as usize % pool.len()];
                aa_bytes.push(aa);
                let cands = &aa_to_codons[&aa];
                let cod = cands[(next_u64() >> 24) as usize % cands.len()];
                nt_bytes.extend_from_slice(&cod);
            }
            prot_raw.push(RawSeq {
                id: format!("UNREL_PROT_{}", k),
                raw_bytes: aa_bytes,
            });
            cod_raw.push(RawSeq {
                id: format!("UNREL_COD_{}", k),
                raw_bytes: nt_bytes,
            });
        }

        let prot_ledger = analyze_unaligned_cohort(&prot_raw, Some(SeqMode::Protein));
        assert_eq!(
            prot_ledger.num_homology_components, 8,
            "Expected 8 singleton components in unrelated Protein cohort, got {} (components={:?})",
            prot_ledger.num_homology_components,
            prot_ledger
                .components
                .iter()
                .map(|c| c.sequence_ids.clone())
                .collect::<Vec<_>>()
        );

        let cod_ledger = analyze_unaligned_cohort(&cod_raw, Some(SeqMode::Codon));
        assert_eq!(
            cod_ledger.num_homology_components, 8,
            "Expected 8 singleton components in unrelated Codon cohort, got {} (components={:?})",
            cod_ledger.num_homology_components,
            cod_ledger
                .components
                .iter()
                .map(|c| c.sequence_ids.clone())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_dna_mode_wobble_and_corridor_no_pinch() {
        // 4 homologous 1,500-nt coding sequences (~78% NT ID driven by 3rd-codon wobble + moderate
        // divergence), plus an off-diagonal shared 9-mer (|Delta d| = 55) inside a wobble-heavy stretch.
        // Verifies that wobble-invariant 9-of-13 seeds + corridor c_drift=0.08 prevent off-diagonal
        // corridor pinching and recover all 1,500 colinear columns without spurious gap blocks.
        let mut seed: u64 = 0x517A_99C0_D04A_1357;
        let mut next_u64 = || -> u64 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed
        };

        const BASES: [u8; 4] = [b'A', b'C', b'G', b'T'];
        let n_codons = 500usize;
        let len_nt = n_codons * 3;
        let mut anc = Vec::with_capacity(len_nt);
        for _ in 0..len_nt {
            anc.push(BASES[(next_u64() >> 32) as usize % 4]);
        }

        let mut seqs = vec![anc.clone(); 4];
        for t in 0..4 {
            for c in 0..n_codons {
                for pos in 0..3 {
                    let idx = c * 3 + pos;
                    let r = (next_u64() >> 32) as usize % 100;
                    if pos == 2 {
                        // Heavy synonymous/wobble turnover (~50% substitution at 3rd codon pos)
                        if r < 50 {
                            // Prefer transition A<->G (0<->2), C<->T (1<->3)
                            let cur = seqs[t][idx];
                            seqs[t][idx] = match cur {
                                b'A' => b'G',
                                b'G' => b'A',
                                b'C' => b'T',
                                b'T' => b'C',
                                _ => b'A',
                            };
                        }
                    } else {
                        // Sparse nonsynonymous changes (~6% per 1st/2nd codon position)
                        if r < 6 {
                            let step = 1 + ((next_u64() >> 24) as usize % 3);
                            let cur_i = BASES.iter().position(|&b| b == seqs[t][idx]).unwrap();
                            seqs[t][idx] = BASES[(cur_i + step) % 4];
                        }
                    }
                }
            }
        }

        // Plant an identical off-diagonal 9-mer in a short diverged subwindow between two taxa
        // (|Delta d| = 55) while keeping the rest of the cohort homologous.
        let motif = b"ATGCGTACA";
        seqs[0][700..709].copy_from_slice(motif);
        seqs[1][755..764].copy_from_slice(motif);
        // Slightly diversify local wobble around 700..770 between seq 0 and seq 1 without adding indels
        for idx in (692..770).step_by(3) {
            if idx < 700 || idx >= 709 {
                seqs[0][idx] = if seqs[2][idx] == b'A' { b'G' } else { b'A' };
            }
            if idx < 755 || idx >= 764 {
                seqs[1][idx] = if seqs[3][idx] == b'C' { b'T' } else { b'C' };
            }
        }

        let raw: Vec<RawSeq> = seqs
            .into_iter()
            .enumerate()
            .map(|(i, b)| RawSeq {
                id: format!("WOBBLE_SEQ_{}", i),
                raw_bytes: b,
            })
            .collect();

        let ledger = analyze_unaligned_cohort(&raw, Some(SeqMode::Dna));
        assert_eq!(
            ledger.num_homology_components, 1,
            "Expected 1 homology component in DNA wobble test, got {}",
            ledger.num_homology_components
        );
        let comp = &ledger.components[0];
        assert!(
            comp.num_columns <= len_nt + 6,
            "Expected colinear ~1500 DNA columns without corridor pinch, got {} (GlobalFine={}, CladeLocal={}, Private={})",
            comp.num_columns,
            comp.num_global_fine_cols,
            comp.num_clade_local_cols,
            comp.num_lineage_private_cols
        );
        assert!(
            comp.num_global_fine_cols >= len_nt - 30,
            "Expected >= 1470 GlobalFine cols in DNA wobble test, got {}",
            comp.num_global_fine_cols
        );
    }

    #[test]
    fn test_large_cohort_skeleton_collapse_and_grafting_exactness() {
        use crate::storage::{
            read_hdb_file, read_hnex_file, render_component_matrix, write_hdb_file,
            write_hnex_file, HierarchicalArchive,
        };
        use std::collections::HashMap;
        use std::time::{Instant, SystemTime, UNIX_EPOCH};

        let mut seed: u64 = 0x9E37_79B9_CAFE_1337;
        fn next_u64(s: &mut u64) -> u64 {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *s
        }

        let gc = GeneticCode::standard();
        let sense_codons: Vec<[u8; 3]> = (0..64u8)
            .filter_map(|c| {
                if gc.triplet_to_sense[c as usize].is_none() {
                    None
                } else {
                    let bases = [b'A', b'C', b'G', b'T'];
                    Some([
                        bases[(c / 16) as usize],
                        bases[((c / 4) % 4) as usize],
                        bases[(c % 4) as usize],
                    ])
                }
            })
            .collect();

        fn gen_codons(n_cod: usize, sense_codons: &[[u8; 3]], seed: &mut u64) -> Vec<u8> {
            let mut v = Vec::with_capacity(n_cod * 3);
            for _ in 0..n_cod {
                let c = sense_codons[(next_u64(seed) >> 32) as usize % sense_codons.len()];
                v.extend_from_slice(&c);
            }
            v
        }

        let anc = gen_codons(120, &sense_codons, &mut seed);

        // Subfamily A prototype (~10% synonymous/AA diversity from anc)
        let mut sub_a = anc.clone();
        for c in 0..120 {
            if (next_u64(&mut seed) >> 32) as usize % 10 == 0 {
                let nc = sense_codons[(next_u64(&mut seed) >> 24) as usize % sense_codons.len()];
                sub_a[c * 3..c * 3 + 3].copy_from_slice(&nc);
            }
        }

        // Subfamily B prototype: 4-codon deletion (48..52) + deeper divergence (~18% codon diff)
        let mut sub_b = Vec::with_capacity(116 * 3);
        for c in 0..120 {
            if (48..52).contains(&c) {
                continue;
            }
            let src = &anc[c * 3..c * 3 + 3];
            if (next_u64(&mut seed) >> 32) as usize % 6 == 0 {
                let nc = sense_codons[(next_u64(&mut seed) >> 24) as usize % sense_codons.len()];
                sub_b.extend_from_slice(&nc);
            } else {
                sub_b.extend_from_slice(src);
            }
        }

        let mut raw_seqs: Vec<RawSeq> = Vec::with_capacity(260);
        // 140 Subfamily A members (including 5'-partial and 3'-partial colinear fragments)
        for i in 0..140 {
            let mut s = sub_a.clone();
            for c in 0..120 {
                if (next_u64(&mut seed) >> 32) as usize % 50 == 0 {
                    let nc = sense_codons[(next_u64(&mut seed) >> 20) as usize % sense_codons.len()];
                    s[c * 3..c * 3 + 3].copy_from_slice(&nc);
                }
            }
            let raw_bytes = if (10..20).contains(&i) {
                s[..(75 * 3)].to_vec()
            } else if (20..30).contains(&i) {
                s[(40 * 3)..].to_vec()
            } else {
                s
            };
            raw_seqs.push(RawSeq {
                id: format!("SubA_{:03}", i),
                raw_bytes,
            });
        }

        // 118 Subfamily B members
        for i in 0..118 {
            let mut s = sub_b.clone();
            for c in 0..116 {
                if (next_u64(&mut seed) >> 32) as usize % 45 == 0 {
                    let nc = sense_codons[(next_u64(&mut seed) >> 20) as usize % sense_codons.len()];
                    s[c * 3..c * 3 + 3].copy_from_slice(&nc);
                }
            }
            raw_seqs.push(RawSeq {
                id: format!("SubB_{:03}", i),
                raw_bytes: s,
            });
        }

        // 1 frameshifted Subfamily A sequence (+1 nt insertion at nt 90)
        let mut fs_seq = Vec::with_capacity(sub_a.len() + 1);
        fs_seq.extend_from_slice(&sub_a[..90]);
        fs_seq.push(b'A');
        fs_seq.extend_from_slice(&sub_a[90..]);
        raw_seqs.push(RawSeq {
            id: "SubA_FS".to_string(),
            raw_bytes: fs_seq,
        });

        // 1 unrelated random decoy singleton
        let decoy = gen_codons(120, &sense_codons, &mut seed);
        raw_seqs.push(RawSeq {
            id: "Decoy_Singleton".to_string(),
            raw_bytes: decoy,
        });

        let t0 = Instant::now();
        let ledger = analyze_unaligned_cohort(&raw_seqs, Some(SeqMode::Codon));
        let elapsed = t0.elapsed().as_secs_f64();

        assert!(
            elapsed < 5.0,
            "Expected large N=260 skeleton-collapsed run < 5.0s, took {:.2}s",
            elapsed
        );
        assert_eq!(
            ledger.num_homology_components, 2,
            "Expected 2 components (259 homologs + 1 decoy), got {}",
            ledger.num_homology_components
        );

        let big_comp = ledger
            .components
            .iter()
            .find(|c| c.sequence_ids.len() == 259)
            .expect("missing 259-sequence homology component");
        let small_comp = ledger
            .components
            .iter()
            .find(|c| c.sequence_ids.len() == 1)
            .expect("missing 1-sequence decoy component");
        assert_eq!(small_comp.sequence_ids[0], "Decoy_Singleton");
        assert_eq!(big_comp.frameshifts_repaired.len(), 1);
        assert_eq!(big_comp.frameshifts_repaired[0].seq_id, "SubA_FS");
        assert_eq!(big_comp.tree_nodes.len(), 2 * 259 - 1);
        assert!(ledger.net_cohort_gain_nats > 50_000.0);

        // Verify exact sequence round-trip on all aligned rows
        let unaligned_map: HashMap<&str, &str> = big_comp
            .unaligned_seqs
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        for (id, aln) in &big_comp.aligned_fasta {
            let degapped: String = aln.chars().filter(|&c| c != '-' && c != '.').collect();
            assert_eq!(
                degapped, unaligned_map[id.as_str()],
                "Mismatch after ungapping for {}",
                id
            );
        }

        // Verify .hnex and .hdb serialization + lossless render round-trip
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let hnex_path = std::env::temp_dir().join(format!("squint_skel_{stamp}.hnex"));
        let hdb_path = std::env::temp_dir().join(format!("squint_skel_{stamp}.hdb"));
        let arch = HierarchicalArchive::from_ledger(&ledger);
        write_hnex_file(&hnex_path, &arch).expect("write hnex");
        write_hdb_file(&hdb_path, &arch).expect("write hdb");
        let arch_hnex = read_hnex_file(&hnex_path).expect("read hnex");
        let arch_hdb = read_hdb_file(&hdb_path).expect("read hdb");
        assert_eq!(arch_hnex.components[0].num_columns, arch_hdb.components[0].num_columns);
        let mat_hdb = render_component_matrix(
            &arch_hdb.components[0],
            &arch_hdb.seqs,
            arch_hdb.mode,
            false,
        );
        assert_eq!(mat_hdb.len(), big_comp.aligned_fasta.len());
        assert_eq!(mat_hdb, big_comp.aligned_fasta);
        let _ = std::fs::remove_file(&hnex_path);
        let _ = std::fs::remove_file(&hdb_path);
    }

    #[test]
    fn test_fast_impulse_path_partition_matches_2d_exact() {
        use crate::transducer::{
            compute_ln_path_partition_vec, compute_ln_path_partition_vec_2d_exact,
            TransducerRegime, PRIOR_MEAN_LEN,
        };

        let regimes = [
            TransducerRegime {
                t_primary: 0.02,
                t_syn: 0.02,
            },
            TransducerRegime {
                t_primary: 0.32,
                t_syn: 0.32,
            },
            TransducerRegime {
                t_primary: 2.00,
                t_syn: 2.00,
            },
        ];
        let max_ly = 850usize;
        for &reg in &regimes {
            for &lx in &[1usize, 5, 65, 180, 720] {
                let exact = compute_ln_path_partition_vec_2d_exact(lx, max_ly, reg, PRIOR_MEAN_LEN);
                let fast = compute_ln_path_partition_vec(lx, max_ly, reg, PRIOR_MEAN_LEN);
                assert_eq!(exact.len(), max_ly + 1);
                assert_eq!(fast.len(), max_ly + 1);
                let mut max_diff = 0.0f64;
                let mut sum_p = 0.0f64;
                for ly in 1..=max_ly {
                    let d = (fast[ly] - exact[ly]).abs();
                    if d > max_diff {
                        max_diff = d;
                    }
                    sum_p += fast[ly].exp();
                }
                assert!(
                    max_diff < 1e-11,
                    "Impulse path partition mismatch for lx={}, t={}: max_diff={:.3e}",
                    lx,
                    reg.t_primary,
                    max_diff
                );
                if lx <= 180 {
                    assert!(
                        sum_p > 0.5 && sum_p <= 1.0 + 1e-6,
                        "Kraft path partition sum violated for lx={}, sum={}",
                        lx,
                        sum_p
                    );
                }
            }
            // Verify tail normalization when max_ly >> lx
            let v1 = compute_ln_path_partition_vec(70, 4_000, reg, PRIOR_MEAN_LEN);
            let s1: f64 = v1[1..].iter().map(|&x| x.exp()).sum();
            assert!(
                (s1 - 1.0).abs() < 1e-4,
                "Expected normalized tail sum ~1.0 for lx=70, max_ly=4000, got {}",
                s1
            );
        }
    }

    #[test]
    fn test_long_dna_corridor_chaining_and_dna_horizon_bubbles() {
        let mut seed: u64 = 0xD4A0_517A_99C1_0423;
        let mut next_u64 = || -> u64 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed
        };

        const BASES: [u8; 4] = [b'A', b'C', b'G', b'T'];
        let mut core_5 = Vec::with_capacity(1_000);
        let mut core_3 = Vec::with_capacity(1_000);
        for _ in 0..1_000 {
            core_5.push(BASES[(next_u64() >> 32) as usize % 4]);
            core_3.push(BASES[(next_u64() >> 32) as usize % 4]);
        }

        // Shared off-diagonal 12-mer chaff block inserted into AT-rich loops to ensure chaining ignores chaff
        let chaff_motif = b"ATGCTAGCTAGC";
        let mut make_at_loop = |tag: u8| -> Vec<u8> {
            let at_heavy = [b'A', b'A', b'A', b'T', b'T', b'T', b'G', b'C'];
            let mut v = Vec::with_capacity(400);
            for idx in 0..400 {
                if (120..132).contains(&idx) {
                    v.push(chaff_motif[idx - 120]);
                } else {
                    let b = at_heavy[((next_u64() >> 32) as usize + tag as usize) % at_heavy.len()];
                    v.push(b);
                }
            }
            v
        };
        let loop_a = make_at_loop(0);
        let loop_b = make_at_loop(3);

        let mut raw_seqs = Vec::new();
        for &(name, clade_loop, rate) in &[
            ("CladeA_1", &loop_a, 0.02f64),
            ("CladeA_2", &loop_a, 0.02f64),
            ("CladeB_1", &loop_b, 0.03f64),
            ("CladeB_2", &loop_b, 0.03f64),
        ] {
            let mut s = Vec::with_capacity(2_400);
            let mut mutate_append = |src: &[u8], out: &mut Vec<u8>, r: f64| {
                for (pos, &b) in src.iter().enumerate() {
                    let u = ((next_u64() >> 11) as f64) / ((1u64 << 53) as f64);
                    // Inject chaff motif at a non-homologous off-diagonal site inside core_5 on CladeA vs core_3 on CladeB
                    if name.starts_with("CladeA") && pos >= 200 && pos < 212 && out.len() < 1000 {
                        out.push(chaff_motif[pos - 200]);
                    } else if u < r {
                        let off = ((next_u64() >> 32) as usize % 3) + 1;
                        let bi = BASES.iter().position(|&x| x == b).unwrap_or(0);
                        out.push(BASES[(bi + off) % 4]);
                    } else {
                        out.push(b);
                    }
                }
            };
            mutate_append(&core_5, &mut s, rate);
            mutate_append(clade_loop, &mut s, rate * 0.5);
            mutate_append(&core_3, &mut s, rate);
            raw_seqs.push(RawSeq {
                id: name.to_string(),
                raw_bytes: s,
            });
        }

        let ledger = analyze_unaligned_cohort(&raw_seqs, Some(SeqMode::Dna));
        assert_eq!(ledger.num_homology_components, 1);
        let comp = &ledger.components[0];
        assert!(
            comp.num_global_fine_cols >= 1_850 && comp.num_global_fine_cols <= 2_050,
            "Expected ~2000 GlobalFine DNA core columns, got {} (clade_local={})",
            comp.num_global_fine_cols,
            comp.num_clade_local_cols
        );
        assert!(
            comp.num_clade_local_cols >= 650,
            "Expected 400-nt clade-A + 400-nt clade-B AT-rich loops isolated in CladeLocal, got {}",
            comp.num_clade_local_cols
        );
        assert!(
            !comp.horizon_bubbles.is_empty(),
            "Expected non-empty DNA HorizonBubbleReport"
        );
        assert!(
            comp.horizon_bubbles
                .iter()
                .any(|hb| hb.dominant_biophys_class.as_deref() == Some("AT-rich(AT)")),
            "Expected AT-rich(AT) DNA horizon bubble, got {:?}",
            comp.horizon_bubbles
                .iter()
                .map(|hb| hb.dominant_biophys_class.clone())
                .collect::<Vec<_>>()
        );
        for (orig, (_, aln)) in raw_seqs.iter().zip(comp.aligned_fasta.iter()) {
            let degapped: Vec<u8> = aln.bytes().filter(|&b| b != b'-').collect();
            assert_eq!(orig.raw_bytes, degapped);
        }
    }

    #[test]
    fn test_long_profile_banded_5state_equivalence() {
        use std::time::Instant;

        let mut seed: u64 = 0x5B17_2026_1500_ABCD;
        let mut next_u64 = || -> u64 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed
        };

        const BASES: [u8; 4] = [b'A', b'C', b'G', b'T'];
        let core_len = 1_560usize;
        let anc: Vec<u8> = (0..core_len)
            .map(|_| BASES[(next_u64() >> 32) as usize % 4])
            .collect();

        let mut raw_seqs = Vec::new();
        for idx in 0..4usize {
            let mut s = Vec::with_capacity(core_len + 120);
            for (pos, &b) in anc.iter().enumerate() {
                if pos == 750 && idx >= 2 {
                    // Shared 48-nt insertion in lineage B (exercises ml, mr > 1_200 banded 5-state DP)
                    for k in 0..48usize {
                        s.push(BASES[(k * 3 + 1) % 4]);
                    }
                }
                let u = ((next_u64() >> 11) as f64) / ((1u64 << 53) as f64);
                if u < 0.03 {
                    let bi = BASES.iter().position(|&x| x == b).unwrap();
                    s.push(BASES[(bi + 1 + (idx % 3)) % 4]);
                } else {
                    s.push(b);
                }
            }
            raw_seqs.push(RawSeq {
                id: format!("Seq_{}", idx),
                raw_bytes: s,
            });
        }

        let t0 = Instant::now();
        let ledger = analyze_unaligned_cohort(&raw_seqs, Some(SeqMode::Dna));
        let elapsed = t0.elapsed().as_secs_f64();
        assert_eq!(ledger.num_homology_components, 1);
        let comp = &ledger.components[0];
        assert!(
            elapsed < 5.0,
            "Expected >1500-nt banded 5-state alignment to finish quickly (<5s), took {:.2}s",
            elapsed
        );
        let full_occ = comp.columns.iter().filter(|c| c.num_present == 4).count();
        assert!(
            full_occ >= 1_540 && comp.num_columns >= 1_595 && comp.num_columns <= 1_620,
            "Expected ~1560 4-taxon core columns and ~1608 total cols (with 48-nt indel), got full_occ={}, num_columns={}, global_fine={}",
            full_occ,
            comp.num_columns,
            comp.num_global_fine_cols
        );
        for (orig, (_, aln)) in raw_seqs.iter().zip(comp.aligned_fasta.iter()) {
            let degapped: Vec<u8> = aln.bytes().filter(|&b| b != b'-').collect();
            assert_eq!(orig.raw_bytes, degapped);
        }
    }
}
