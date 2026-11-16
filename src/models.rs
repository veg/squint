use std::sync::OnceLock;

use crate::alphabet::{GeneticCode, SeqMode, AA_INTRINSIC_PROPS};
use crate::mdl::{exchangeable_dirichlet_frac_ln_prob, logsumexp_slice};

/// Precomputed alignment-free biophysical & genetic-code primitives for the 20 amino acids
/// (`ACDEFGHIKLMNPQRSTVWY`) and nucleotide column Dirichlet mixture urns.
struct IntrinsicAaPhysics {
    z: [[f64; 5]; 20],
    mut_acc: [[f64; 20]; 20],
    psi: [[f64; 20]; 8],
    col_urns_aa: [[f64; 20]; 9],
    col_urns_dna: [[f64; 4]; 5],
}

fn aa_physics() -> &'static IntrinsicAaPhysics {
    static PHYSICS: OnceLock<IntrinsicAaPhysics> = OnceLock::new();
    PHYSICS.get_or_init(|| {
        let gc_std = GeneticCode::standard();

        // 1. Standardize each of the 5 free amino acid physicochemical coordinates
        let mut z = [[0.0f64; 5]; 20];
        for k in 0..5 {
            let mu = (0..20).map(|a| AA_INTRINSIC_PROPS[a][k]).sum::<f64>() / 20.0;
            let var = (0..20)
                .map(|a| {
                    let d = AA_INTRINSIC_PROPS[a][k] - mu;
                    d * d
                })
                .sum::<f64>()
                / 20.0;
            let sd = var.sqrt().max(1e-12);
            for a in 0..20 {
                z[a][k] = (AA_INTRINSIC_PROPS[a][k] - mu) / sd;
            }
        }

        // 2. Combinatorial 64-codon genetic-code mutational accessibility
        let mut mut_acc = [[0.0f64; 20]; 20];
        for a in 0..20 {
            let senses_a = &gc_std.aa_to_senses[a];
            for b in 0..20 {
                if a == b {
                    continue;
                }
                let senses_b = &gc_std.aa_to_senses[b];
                let mut best_w = 0.0f64;
                let mut tot_w = 0.0f64;
                for &c1 in senses_a {
                    let c1 = c1 as usize;
                    for &c2 in senses_b {
                        let c2 = c2 as usize;
                        let diffs = gc_std.nt_diffs[c1][c2];
                        let w = if diffs == 1 && gc_std.is_single_transition[c1][c2] {
                            2.0
                        } else if diffs == 1 {
                            1.0
                        } else if diffs == 2 {
                            0.25
                        } else {
                            0.06
                        };
                        if w > best_w {
                            best_w = w;
                        }
                        tot_w += w;
                    }
                }
                let n_pairs = (senses_a.len() * senses_b.len()).max(1) as f64;
                mut_acc[a][b] = 0.5 * best_w + 0.5 * (tot_w / n_pairs);
            }
        }

        // 3. Eight Partition-of-Unity Biophysical Site-Niche Memberships (sum_m psi[m][a] == 1.0)
        let mut psi = [[0.0f64; 20]; 8];
        for a in 0..20 {
            let is_p = if a == 12 { 1.0 } else { 0.0 };
            let is_g = if a == 5 { 1.0 } else { 0.0 };
            let is_c = if a == 1 { 1.0 } else { 0.0 };
            let is_pgc = is_p + is_g + is_c;
            let chg = AA_INTRINSIC_PROPS[a][3];
            let abs_chg = chg.abs();
            let logits = [
                2.2 * z[a][0] - 1.2 * z[a][1] - 2.0 * abs_chg - 0.8 * z[a][4] - 2.0 * is_pgc,
                2.6 * z[a][4] + 0.8 * z[a][2] - 1.5 * abs_chg,
                -2.2 * z[a][2] - 1.5 * abs_chg - 1.2 * z[a][4] - 1.5 * is_c,
                1.2 * z[a][1] - 0.6 * z[a][0] - 2.2 * abs_chg - 0.8 * z[a][4] - 1.5 * is_pgc,
                3.2 * chg.max(0.0) + 0.6 * z[a][1] - 1.0 * z[a][0],
                3.4 * (-chg).max(0.0) + 0.6 * z[a][1] - 1.0 * z[a][0],
                3.5 * is_pgc,
                0.3,
            ];
            let max_l = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut sum_e = 0.0f64;
            let mut exps = [0.0f64; 8];
            for m in 0..8 {
                let e = (logits[m] - max_l).exp();
                exps[m] = e;
                sum_e += e;
            }
            for m in 0..8 {
                psi[m][a] = exps[m] / sum_e;
            }
        }

        // 4. Nine Biophysical Dirichlet-Mixture Column Urns
        let pi0 = default_background_pi(SeqMode::Protein, &gc_std);
        let mut col_urns_aa = [[0.0f64; 20]; 9];
        for a in 0..20 {
            col_urns_aa[0][a] = 0.5 * pi0[a];
        }
        for m in 0..8 {
            let w_m: f64 = (0..20).map(|a| pi0[a] * psi[m][a]).sum();
            let inv_w = 1.0 / w_m.max(1e-15);
            for a in 0..20 {
                let pi_m = pi0[a] * psi[m][a] * inv_w;
                col_urns_aa[1 + m][a] = 18.0 * (0.92 * pi_m + 0.08 * pi0[a]);
            }
        }

        // 5. Five Nucleotide Dirichlet-Mixture Column Urns
        let col_urns_dna = [
            [0.10; 4],
            [2.0; 4],
            [4.0, 1.0, 1.0, 4.0],
            [1.0, 4.0, 4.0, 1.0],
            [3.5, 1.5, 3.5, 1.5],
        ];

        IntrinsicAaPhysics {
            z,
            mut_acc,
            psi,
            col_urns_aa,
            col_urns_dna,
        }
    })
}

/// Default background frequencies for each mode derived from genetic-code combinatorics.
pub fn default_background_pi(mode: SeqMode, gc: &GeneticCode) -> Vec<f64> {
    match mode {
        SeqMode::Dna => vec![0.25; 4],
        SeqMode::Codon => {
            let n_sense = gc.num_sense;
            vec![1.0 / (n_sense as f64); n_sense]
        }
        SeqMode::Protein => {
            let mut pi = vec![0.0; 20];
            let mut sum = 0.0;
            for aa in 0..20 {
                let w = gc.aa_to_senses[aa].len() as f64 + 0.5;
                pi[aa] = w;
                sum += w;
            }
            for x in &mut pi {
                *x /= sum;
            }
            pi
        }
    }
}

/// High-precision Scaling-and-Squaring Taylor-12 matrix exponential P = exp(Q * t)
/// for a continuous-time Markov rate matrix Q (rows summing to 0), enforcing exact
/// non-negativity and row-stochastic normalization \sum_b P_{ab} == 1.0.
pub fn matrix_exp_stochastic(q: &[f64], dim: usize, t: f64) -> Vec<f64> {
    if t <= 0.0 {
        let mut id = vec![0.0; dim * dim];
        for i in 0..dim {
            id[i * dim + i] = 1.0;
        }
        return id;
    }

    // Determine scaling power 2^s so ||Q * t / 2^s||_inf <= 0.25
    let mut max_diag = 0.0f64;
    for i in 0..dim {
        let d = -q[i * dim + i];
        if d > max_diag {
            max_diag = d;
        }
    }
    let norm = max_diag * t;
    let mut s = 0u32;
    let mut scaled = norm;
    while scaled > 0.25 && s < 30 {
        scaled *= 0.5;
        s += 1;
    }
    let scale = 1.0 / ((1u64 << s) as f64);

    // M = Q * (t * scale)
    let mut m = vec![0.0; dim * dim];
    for idx in 0..(dim * dim) {
        m[idx] = q[idx] * t * scale;
    }

    // 12-term Taylor series: res = I + M + M^2/2! + ... + M^12/12!
    let mut res = vec![0.0; dim * dim];
    let mut term = vec![0.0; dim * dim];
    let mut next_term = vec![0.0; dim * dim];
    for i in 0..dim {
        res[i * dim + i] = 1.0;
        term[i * dim + i] = 1.0;
    }

    for k in 1..=12 {
        let inv_k = 1.0 / (k as f64);
        next_term.fill(0.0);
        for i in 0..dim {
            let row_off = i * dim;
            for l in 0..dim {
                let t_il = term[row_off + l];
                if t_il != 0.0 {
                    let m_off = l * dim;
                    for j in 0..dim {
                        next_term[row_off + j] += t_il * m[m_off + j];
                    }
                }
            }
        }
        for idx in 0..(dim * dim) {
            term[idx] = next_term[idx] * inv_k;
            res[idx] += term[idx];
        }
    }

    // Repeated squaring s times: res = res^(2^s)
    for _ in 0..s {
        next_term.fill(0.0);
        for i in 0..dim {
            let row_off = i * dim;
            for l in 0..dim {
                let r_il = res[row_off + l];
                let l_off = l * dim;
                for j in 0..dim {
                    next_term[row_off + j] += r_il * res[l_off + j];
                }
            }
        }
        res.copy_from_slice(&next_term);
    }

    // Enforce exact row-stochastic conservation to machine epsilon
    for i in 0..dim {
        let row = &mut res[i * dim..(i + 1) * dim];
        let mut rsum = 0.0;
        for val in row.iter_mut() {
            if *val < 1e-300 {
                *val = 1e-300;
            }
            rsum += *val;
        }
        for val in row.iter_mut() {
            *val /= rsum;
        }
    }

    res
}

/// Build Partition-of-Unity Biophysical Niche + Genetic-Code Mutational Accessibility +
/// Structural Turnover 20x20 transition matrix P_AA(b | a; t_aa, pi_target), normalized so
/// average substitution rate wrt pi_target is 1.0 before applying t_aa.
pub fn build_mechanistic_aa_matrix(t_aa: f64, pi_target: &[f64]) -> [f64; 400] {
    assert_eq!(pi_target.len(), 20);
    if t_aa <= 0.0 {
        let mut eye = [0.0f64; 400];
        for i in 0..20 {
            eye[i * 20 + i] = 1.0;
        }
        return eye;
    }

    let phys = aa_physics();
    const NICHE_RATES: [f64; 8] = [0.65, 0.55, 0.75, 1.15, 0.95, 0.95, 0.30, 2.20];
    const GAMMA_TURNOVER: f64 = 0.30;

    let mut q_mats = [[0.0f64; 400]; 8];
    let mut init_flux = 0.0f64;

    for m in 0..8 {
        let w_m: f64 = (0..20).map(|a| pi_target[a] * phys.psi[m][a]).sum();
        let inv_w = 1.0 / w_m.max(1e-15);
        let mut pi_m = [0.0f64; 20];
        for a in 0..20 {
            pi_m[a] = pi_target[a] * phys.psi[m][a] * inv_w;
        }

        let q_m = &mut q_mats[m];
        let mut flux_m = 0.0f64;
        for a in 0..20 {
            let mut rsum = 0.0f64;
            for b in 0..20 {
                if a != b {
                    let cov_pen = if m == 6
                        && ((a == 12) != (b == 12) || (a == 5) != (b == 5) || (a == 1) != (b == 1))
                    {
                        0.05
                    } else {
                        1.0
                    };
                    let rate = phys.mut_acc[a][b] * cov_pen * pi_m[b];
                    q_m[a * 20 + b] = rate;
                    rsum += rate;
                }
            }
            q_m[a * 20 + a] = -rsum;
            flux_m += pi_m[a] * rsum;
        }
        if flux_m > 0.0 {
            let scale = NICHE_RATES[m] / flux_m;
            for v in q_m.iter_mut() {
                *v *= scale;
            }
        }
        init_flux += w_m * NICHE_RATES[m];
    }

    let pi_sq: f64 = pi_target.iter().map(|x| x * x).sum();
    let tot_flux = init_flux + GAMMA_TURNOVER * (1.0 - pi_sq);
    let tau = t_aa / tot_flux.max(1e-12);
    let surv = (-GAMMA_TURNOVER * tau).exp();

    let mut p_niche = [0.0f64; 400];
    for m in 0..8 {
        let exp_m = matrix_exp_stochastic(&q_mats[m], 20, tau);
        for a in 0..20 {
            let w_ma = phys.psi[m][a];
            for b in 0..20 {
                p_niche[a * 20 + b] += w_ma * exp_m[a * 20 + b];
            }
        }
    }

    let mut p_out = [0.0f64; 400];
    for a in 0..20 {
        let mut rsum = 0.0f64;
        for b in 0..20 {
            let val = surv * p_niche[a * 20 + b] + (1.0 - surv) * pi_target[b];
            p_out[a * 20 + b] = val;
            rsum += val;
        }
        if rsum > 0.0 {
            for b in 0..20 {
                p_out[a * 20 + b] /= rsum;
            }
        }
    }
    p_out
}

/// Build dyadic 4-rate-octave purine/pyrimidine ring-conservation 4x4 nucleotide transition
/// matrix P_NT(b | a; t_nt, pi_target) with transition/transversion ratio 2.0.
pub fn build_purine_pyrimidine_nt_matrix(t_nt: f64, pi_target: &[f64]) -> [f64; 16] {
    assert_eq!(pi_target.len(), 4);
    if t_nt <= 0.0 {
        let mut eye = [0.0f64; 16];
        for i in 0..4 {
            eye[i * 4 + i] = 1.0;
        }
        return eye;
    }
    const DYADIC4_RATES: [f64; 4] = [4.0 / 15.0, 8.0 / 15.0, 16.0 / 15.0, 32.0 / 15.0];
    let kappa = 2.0;
    let mut q = [0.0f64; 16];
    let mut mean_rate = 0.0;
    for i in 0..4 {
        let mut rsum = 0.0;
        for j in 0..4 {
            if i != j {
                let is_ti = (i ^ j) == 2;
                let rate = if is_ti { kappa } else { 1.0 } * pi_target[j];
                q[i * 4 + j] = rate;
                rsum += rate;
            }
        }
        q[i * 4 + i] = -rsum;
        mean_rate += pi_target[i] * rsum;
    }
    if mean_rate > 0.0 {
        let inv = 1.0 / mean_rate;
        for val in &mut q {
            *val *= inv;
        }
    }
    let mut p_avg = [0.0f64; 16];
    for &r in &DYADIC4_RATES {
        let p_cat = matrix_exp_stochastic(&q, 4, t_nt * r);
        for idx in 0..16 {
            p_avg[idx] += 0.25 * p_cat[idx];
        }
    }
    for i in 0..4 {
        let s: f64 = p_avg[i * 4..(i + 1) * 4].iter().sum();
        for v in &mut p_avg[i * 4..(i + 1) * 4] {
            *v /= s;
        }
    }
    p_avg
}

/// Evaluate column log marginal likelihood under the biophysical Dirichlet-mixture cohort model.
pub fn column_niche_mixture_ln_prob(mode: SeqMode, counts: &[f64]) -> f64 {
    let phys = aa_physics();
    match mode {
        SeqMode::Dna => {
            let ln_u = (phys.col_urns_dna.len() as f64).ln();
            let mut terms = [0.0f64; 5];
            for (u, urn) in phys.col_urns_dna.iter().enumerate() {
                terms[u] = exchangeable_dirichlet_frac_ln_prob(counts, urn) - ln_u;
            }
            logsumexp_slice(&terms)
        }
        SeqMode::Protein | SeqMode::Codon => {
            let ln_u = (phys.col_urns_aa.len() as f64).ln();
            let mut terms = [0.0f64; 9];
            for (u, urn) in phys.col_urns_aa.iter().enumerate() {
                terms[u] = exchangeable_dirichlet_frac_ln_prob(counts, urn) - ln_u;
            }
            logsumexp_slice(&terms)
        }
    }
}

/// Build the 61x61 Telescoping Continuous-Time Codon Transition Matrix:
///   P(c2 | c1; t_aa, t_syn, pi_codon_target) = P_AA(a2 | a1; t_aa) * P_syn(c2 | c1, a2; t_syn)
/// Properties:
/// 1. Exact Kraft normalization: \sum_{c2=0..60} P(c2 | c1) == 1.0 for every c1.
/// 2. At (t_aa, t_syn) -> (0, 0): P(c2 | c1) -> I_61.
/// 3. As t_syn -> inf: P_syn(c2 | c1, a2) -> pi_syn_target(c2 | a2), so GC3/codon-bias skew
///    cancels target M_0 identically to 0.000 nats while biophysical AA homology stays active!
/// 4. As (t_aa, t_syn) -> (inf, inf): P(c2 | c1) -> pi_codon_target(c2) identically!
pub fn build_telescoping_codon_matrix(
    t_aa: f64,
    t_syn: f64,
    pi_codon_target: &[f64],
    gc: &GeneticCode,
) -> Vec<f64> {
    let n_sense = gc.num_sense;
    assert_eq!(pi_codon_target.len(), n_sense);

    // 1. Aggregate target amino acid frequencies pi_aa_target[0..20]
    let mut pi_aa = [0.0f64; 20];
    for s in 0..n_sense {
        pi_aa[gc.sense_to_aa[s] as usize] += pi_codon_target[s];
    }
    let aa_sum: f64 = pi_aa.iter().sum();
    for x in &mut pi_aa {
        *x = (*x / aa_sum).max(1e-8);
    }
    let aa_sum2: f64 = pi_aa.iter().sum();
    for x in &mut pi_aa {
        *x /= aa_sum2;
    }

    // Conditional target synonymous codon probabilities pi_syn(c | aa)
    let mut pi_syn_cond = vec![0.0f64; n_sense];
    for aa in 0..20 {
        let senses = &gc.aa_to_senses[aa];
        let mut sub_sum = 0.0;
        for &s in senses {
            sub_sum += pi_codon_target[s as usize];
        }
        for &s in senses {
            pi_syn_cond[s as usize] = if sub_sum > 0.0 {
                pi_codon_target[s as usize] / sub_sum
            } else {
                1.0 / (senses.len() as f64)
            };
        }
    }

    // 2. AA transition matrix via mechanistic biophysical Markov kernel
    let p_aa = build_mechanistic_aa_matrix(t_aa, &pi_aa);

    // 3. Within-AA synonymous continuous-time generators Q_syn^(aa) for each degenerate AA
    let kappa = 2.0; // Purine/pyrimidine ring-conservation bias on synonymous sites
    let mut p_syn_same_aa = vec![[0.0f64; 6]; n_sense]; // p_syn_same_aa[c1][idx_in_aa]
    for aa in 0..20 {
        let senses = &gc.aa_to_senses[aa];
        let deg = senses.len();
        if deg == 1 {
            p_syn_same_aa[senses[0] as usize][0] = 1.0;
            continue;
        }
        let mut q_sub = vec![0.0f64; deg * deg];
        let mut mean_r = 0.0;
        for i in 0..deg {
            let ci = senses[i] as usize;
            let mut rsum = 0.0;
            for j in 0..deg {
                if i != j {
                    let cj = senses[j] as usize;
                    let diffs = gc.nt_diffs[ci][cj];
                    let exch = match diffs {
                        1 => {
                            if gc.is_single_transition[ci][cj] {
                                kappa
                            } else {
                                1.0
                            }
                        }
                        2 => 0.08, // 2-nt synonymous bridges (Ser TCN<->AGY, Arg CGN<->AGR, Leu CTN<->TTR)
                        _ => 0.01,
                    };
                    let rate = exch * pi_syn_cond[cj];
                    q_sub[i * deg + j] = rate;
                    rsum += rate;
                }
            }
            q_sub[i * deg + i] = -rsum;
            mean_r += pi_syn_cond[ci] * rsum;
        }
        if mean_r > 0.0 {
            for v in &mut q_sub {
                *v /= mean_r;
            }
        }
        let p_sub = matrix_exp_stochastic(&q_sub, deg, t_syn);
        for i in 0..deg {
            let ci = senses[i] as usize;
            for j in 0..deg {
                p_syn_same_aa[ci][j] = p_sub[i * deg + j];
            }
        }
    }

    // 4. Assemble full n_sense x n_sense matrix
    let mut p_codon = vec![0.0f64; n_sense * n_sense];
    let retention = (-t_syn).exp(); // Genetic-code silent-position memory on non-synonymous steps
    for c1 in 0..n_sense {
        let a1 = gc.sense_to_aa[c1] as usize;
        for a2 in 0..20 {
            let prob_aa = p_aa[a1 * 20 + a2];
            let senses_2 = &gc.aa_to_senses[a2];
            if a1 == a2 {
                for (j, &c2) in senses_2.iter().enumerate() {
                    p_codon[c1 * n_sense + (c2 as usize)] = prob_aa * p_syn_same_aa[c1][j];
                }
            } else if senses_2.len() == 1 {
                p_codon[c1 * n_sense + (senses_2[0] as usize)] = prob_aa;
            } else {
                // Weight target codons of a2 by nucleotide proximity at finite t_syn,
                // decaying smoothly to pi_syn_cond(c2 | a2) as t_syn -> inf!
                let mut weights = [0.0f64; 6];
                let mut w_sum = 0.0;
                for (j, &c2) in senses_2.iter().enumerate() {
                    let cj = c2 as usize;
                    let diffs = gc.nt_diffs[c1][cj];
                    let step_favor = match diffs {
                        1 => {
                            if gc.is_single_transition[c1][cj] {
                                4.0
                            } else {
                                1.8
                            }
                        }
                        2 => 0.4,
                        _ => 0.1,
                    };
                    let tilted = pi_syn_cond[cj] * (1.0 + retention * (step_favor - 1.0));
                    weights[j] = tilted.max(1e-12);
                    w_sum += weights[j];
                }
                for (j, &c2) in senses_2.iter().enumerate() {
                    p_codon[c1 * n_sense + (c2 as usize)] = prob_aa * (weights[j] / w_sum);
                }
            }
        }
        // Final row renormalization to machine epsilon
        let row = &mut p_codon[c1 * n_sense..(c1 + 1) * n_sense];
        let rsum: f64 = row.iter().sum();
        for v in row.iter_mut() {
            *v /= rsum;
        }
    }

    p_codon
}

/// Number of latent biophysical attractor channels in Coarse Homology Bubbles (`B(0..5)`).
pub const NUM_BUBBLE_ATTRACTORS: usize = 5;

/// Build 5 Kraft-normalized biophysical attractor distributions over alphabet K (tilts of target `pi_base`):
/// 0: Hydrophobic / Aliphatic-Aromatic core-like segment
/// 1: Polar / Ser-Thr-Asn-Gln glycosylation/mucin-like loop
/// 2: Basic / Positively charged (KRH) tail/NLS
/// 3: Acidic / Negatively charged (DE) transactivation/IDR loop
/// 4: Proline/Glycine-rich flexible disordered linker
pub fn build_coarse_bubble_attractors(
    mode: SeqMode,
    pi_base: &[f64],
    gc: &GeneticCode,
) -> Vec<Vec<f64>> {
    let k_alpha = pi_base.len();
    if mode == SeqMode::Dna {
        // For non-coding DNA: AT-rich, GC-rich, Purine-rich, Pyrimidine-rich, Neutral
        let tilts: [[f64; 4]; 5] = [
            [2.2, 0.5, 0.5, 2.2], // AT-rich enhancer/promoter bubble
            [0.5, 2.2, 2.2, 0.5], // GC-rich CpG bubble
            [2.0, 0.6, 2.0, 0.6], // Purine-rich
            [0.6, 2.0, 0.6, 2.0], // Pyrimidine-rich
            [1.0, 1.0, 1.0, 1.0], // Neutral composition
        ];
        return tilts
            .iter()
            .map(|t| {
                let mut v: Vec<f64> = (0..4).map(|b| (pi_base[b] * t[b]).max(1e-9)).collect();
                let s: f64 = v.iter().sum();
                for x in &mut v {
                    *x /= s;
                }
                v
            })
            .collect();
    }

    let phys = aa_physics();
    let mut attractors = Vec::with_capacity(NUM_BUBBLE_ATTRACTORS);
    for attr_idx in 0..NUM_BUBBLE_ATTRACTORS {
        let mut dist = vec![0.0f64; k_alpha];
        for tok in 0..k_alpha {
            let aa = match mode {
                SeqMode::Protein => tok,
                SeqMode::Codon => gc.sense_to_aa[tok] as usize,
                SeqMode::Dna => unreachable!(),
            };
            let z_a = phys.z[aa];
            let chg = AA_INTRINSIC_PROPS[aa][3];
            let abs_chg = chg.abs();
            let is_pg: f64 = if aa == 12 || aa == 5 { 1.0 } else { 0.0 };
            let mult = match attr_idx {
                0 => (0.9 * z_a[0] + 0.5 * z_a[4]).exp(),
                1 => (0.9 * z_a[1] - 1.2 * abs_chg).exp(),
                2 => (1.5 * chg.max(0.0)).exp(),
                3 => (1.6 * (-chg).max(0.0)).exp(),
                4 => (1.3 * is_pg).exp(),
                _ => 1.0,
            };
            dist[tok] = (pi_base[tok] * mult).max(1e-12);
        }
        let s: f64 = dist.iter().sum();
        for x in &mut dist {
            *x /= s;
        }
        attractors.push(dist);
    }
    attractors
}
