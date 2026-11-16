use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use serde::{Deserialize, Serialize};

use crate::alphabet::{
    EncodedSeq, GeneticCode, SeqMode, AA_PROPERTIES, AA_TO_BIOPHYS, BIOPHYS_CLASS_NAMES,
    NUM_BIOPHYS_CLASSES,
};
use crate::mdl::{
    elias_rissanen_nats, exchangeable_dirichlet_seq_ln_prob, logsumexp2, logsumexp_slice,
    mdl_gain_to_prob, SingleSeqNullProfile, NEG_INF,
};
use crate::models::{
    build_mechanistic_aa_matrix, build_purine_pyrimidine_nt_matrix,
    build_telescoping_codon_matrix, default_background_pi,
};

pub const PRIOR_MEAN_LEN: f64 = 350.0;

#[inline]
pub fn geometric_len_nats(len: usize) -> f64 {
    let eta = PRIOR_MEAN_LEN / (PRIOR_MEAN_LEN + 1.0);
    let one_minus_eta = 1.0 / (PRIOR_MEAN_LEN + 1.0);
    -(one_minus_eta.ln() + (len as f64) * eta.ln())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameshiftEvent {
    pub seq_id: String,
    pub approx_codon_pos: usize,
    pub nt_pos: usize,
    pub phase_shift: u8,
    pub mdl_gain_nats: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SegmentKind {
    FineHomology,
    CoarseBubble,
    NonHomologous,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiScaleSegment {
    pub kind: SegmentKind,
    pub range_x: (usize, usize),
    pub range_y: (usize, usize),
    pub mean_posterior: f64,
    pub dominant_biophys_class: Option<String>,
    pub macro_biophys_gain_nats: Option<f64>,
    pub delta_hydropathy: Option<f64>,
    pub delta_charge: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseHomologyEdge {
    pub idx_x: usize,
    pub idx_y: usize,
    pub posterior: f64,
    #[serde(default)]
    pub cond_posterior: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairHomologyResult {
    pub id_x: String,
    pub id_y: String,
    pub len_x: usize,
    pub len_y: usize,
    pub null_joint_nats: f64,
    pub hom_joint_nats: f64,
    pub mdl_gain_nats: f64,
    pub homology_posterior: f64,
    pub est_t_primary: f64,
    pub est_t_syn: Option<f64>,
    pub est_f_shared: f64,
    pub fine_edges: Vec<SparseHomologyEdge>,
    pub bubble_post_x: Vec<f64>,
    pub bubble_post_y: Vec<f64>,
    pub fine_cov_x: Vec<f64>,
    pub fine_cov_y: Vec<f64>,
    pub segments: Vec<MultiScaleSegment>,
    pub frameshifts: Vec<FrameshiftEvent>,
}

/// Coarse-grained evolutionary regime for `ST_BY` (`CoarseBubble`):
/// Represents unordered segment-level homology where 1-to-1 positional colinearity has scrambled
/// over deeper loop/domain turnover time `t + \Delta t_bubble`.
pub fn coarse_regime_for(mode: SeqMode, reg: TransducerRegime) -> TransducerRegime {
    match mode {
        SeqMode::Protein | SeqMode::Dna => TransducerRegime {
            t_primary: (reg.t_primary + 0.85).min(2.50),
            t_syn: (reg.t_syn + 0.85).min(2.50),
        },
        SeqMode::Codon => TransducerRegime {
            t_primary: (reg.t_primary + 0.85).min(2.50),
            t_syn: (reg.t_syn + 3.00).min(10.0),
        },
    }
}

/// Precompute Renormalization-Group Coarse Copula ratios across the upstream bubble window ending at `i` in `X`:
///   R_coarse(i, z; reg) = (1 / |W_i|) \sum_{u \in W_i} R_{t_coarse}(x_u, z),
/// where `W_i = [i.saturating_sub(span) ..= i]` is the upstream segment consumed by `ST_BX` prior to `ST_BY`.
/// Unifies FineHomology (`|W|=1, t`), CoarseBubble (`|W|=span+1, t_coarse`), and Null (`t -> \infty, R=1`)
/// under a single continuous evolutionary Markov kernel with exact Kraft normalization!
pub fn precompute_local_coarse_tilts(
    seq_x: &EncodedSeq,
    copula_coarse_mat: &[f64],
    half_win: usize,
) -> Vec<f64> {
    let lx = seq_x.len();
    let k_alpha = seq_x.alphabet_size();
    if lx == 0 {
        return Vec::new();
    }
    let full_mask: u64 = if k_alpha >= 64 {
        u64::MAX
    } else {
        (1u64 << k_alpha) - 1
    };
    let span = (2 * half_win).max(4);
    let mut out = vec![1.0f64; lx * k_alpha];
    for i in 0..lx {
        let lo = i.saturating_sub(span);
        let hi = (i + 1).min(lx);
        let dst = &mut out[i * k_alpha..(i + 1) * k_alpha];
        dst.fill(0.0);
        let mut valid_cnt = 0usize;
        for pos in lo..hi {
            let xm = seq_x.ambig_masks[pos];
            if xm == full_mask {
                continue;
            }
            valid_cnt += 1;
            let mc = xm.count_ones();
            if mc == 1 {
                let xu = seq_x.tokens[pos] as usize;
                let r_row = &copula_coarse_mat[xu * k_alpha..(xu + 1) * k_alpha];
                for z in 0..k_alpha {
                    dst[z] += r_row[z];
                }
            } else {
                let inv_m = 1.0 / (mc as f64);
                let mut mm = xm;
                while mm != 0 {
                    let xu = mm.trailing_zeros() as usize;
                    mm &= mm - 1;
                    let r_row = &copula_coarse_mat[xu * k_alpha..(xu + 1) * k_alpha];
                    for z in 0..k_alpha {
                        dst[z] += r_row[z] * inv_m;
                    }
                }
            }
        }
        if valid_cnt == 0 {
            dst.fill(1.0);
        } else {
            let inv_w = 1.0 / (valid_cnt as f64);
            for z in 0..k_alpha {
                dst[z] *= inv_w;
            }
        }
    }
    out
}

#[derive(Debug, Clone, Copy)]
pub struct TransducerRegime {
    pub t_primary: f64,
    pub t_syn: f64,
}

pub fn candidate_regimes(mode: SeqMode) -> Vec<TransducerRegime> {
    match mode {
        SeqMode::Protein | SeqMode::Dna => {
            let ts = [0.005, 0.02, 0.06, 0.15, 0.32, 0.65, 1.20, 2.00];
            ts.iter()
                .map(|&t| TransducerRegime {
                    t_primary: t,
                    t_syn: t,
                })
                .collect()
        }
        SeqMode::Codon => {
            let pairs = [
                (0.002, 0.006),
                (0.008, 0.030),
                (0.025, 0.100),
                (0.060, 0.250),
                (0.060, 0.060),
                (0.140, 0.600),
                (0.140, 0.180),
                (0.300, 1.300),
                (0.300, 0.400),
                (0.600, 2.800),
                (0.600, 0.900),
                (1.150, 5.000),
                (1.900, 7.500),
            ];
            pairs
                .iter()
                .map(|&(ta, ts)| TransducerRegime {
                    t_primary: ta,
                    t_syn: ts,
                })
                .collect()
        }
    }
}

/// Build universal Copula ratio matrix R_t(x, z) = P_univ(z | x; t) / pi_univ(z).
/// Satisfies R_t(x, z) -> 1.0 as t -> inf, so Copula-tilting M_0(z | Y_{<j}) converges
/// identically to M_0(z | Y_{<j}) with zero compositional or lookahead bias!
pub fn build_copula_ratio_matrix(
    mode: SeqMode,
    reg: TransducerRegime,
    univ_pi: &[f64],
    gc: &GeneticCode,
) -> Vec<f64> {
    let k = univ_pi.len();
    let p_mat: Vec<f64> = match mode {
        SeqMode::Protein => build_mechanistic_aa_matrix(reg.t_primary, univ_pi).to_vec(),
        SeqMode::Dna => build_purine_pyrimidine_nt_matrix(reg.t_primary, univ_pi).to_vec(),
        SeqMode::Codon => {
            build_telescoping_codon_matrix(reg.t_primary, reg.t_syn, univ_pi, gc)
        }
    };
    let mut r_mat = vec![1.0f64; k * k];
    for x in 0..k {
        for z in 0..k {
            r_mat[x * k + z] = p_mat[x * k + z] / univ_pi[z].max(1e-15);
        }
    }
    r_mat
}

const N_STATES: usize = 7;
const ST_PRE_Y: usize = 0;
const ST_M: usize = 1;
const ST_D: usize = 2;
const ST_I: usize = 3;
const ST_BX: usize = 4;
const ST_BY: usize = 5;
const ST_POST_Y: usize = 6;

pub struct SingleRegimeFB {
    pub ln_prob_y_given_x: f64,
    pub post_fine: Vec<f64>,
    pub post_bx: Vec<f64>,
    pub post_by: Vec<f64>,
}

#[inline(always)]
fn dot_k_unrolled(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    if n == 4 {
        return (a[0] * b[0] + a[1] * b[1]) + (a[2] * b[2] + a[3] * b[3]);
    }
    if n == 20 {
        let mut s0 = a[0] * b[0] + a[4] * b[4] + a[8] * b[8] + a[12] * b[12] + a[16] * b[16];
        let mut s1 = a[1] * b[1] + a[5] * b[5] + a[9] * b[9] + a[13] * b[13] + a[17] * b[17];
        let mut s2 = a[2] * b[2] + a[6] * b[6] + a[10] * b[10] + a[14] * b[14] + a[18] * b[18];
        let mut s3 = a[3] * b[3] + a[7] * b[7] + a[11] * b[11] + a[15] * b[15] + a[19] * b[19];
        s0 += s2;
        s1 += s3;
        let _ = &mut s2;
        let _ = &mut s3;
        return s0 + s1;
    }
    let mut s0 = 0.0f64;
    let mut s1 = 0.0f64;
    let mut s2 = 0.0f64;
    let mut s3 = 0.0f64;
    let mut i = 0usize;
    while i + 4 <= n {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
        i += 4;
    }
    while i < n {
        s0 += a[i] * b[i];
        i += 1;
    }
    (s0 + s1) + (s2 + s3)
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn eval_ambig_emit_hom(
    mx: u64,
    xi: usize,
    my: u64,
    yj: usize,
    full_mask: u64,
    k_alpha: usize,
    r_mat: &[f64],
    inv_z_sharp: Option<f64>,
    m0_row: &[f64],
    r_eff_buf: &mut [f64],
) -> (f64, Option<f64>) {
    if mx == full_mask || my == full_mask {
        return (1.0, inv_z_sharp);
    }
    if mx.count_ones() == 1 {
        let r_row = &r_mat[xi * k_alpha..(xi + 1) * k_alpha];
        let (inv_z, new_iz) = if let Some(iz) = inv_z_sharp {
            (iz, None)
        } else {
            let dot = dot_k_unrolled(&m0_row[..k_alpha], r_row);
            let iz = 1.0 / dot.max(1e-300);
            (iz, Some(iz))
        };
        if my.count_ones() == 1 {
            return (r_row[yj].max(1e-300) * inv_z, new_iz);
        }
        let mut num = 0.0;
        let mut m0_my = 0.0;
        let mut mm = my;
        while mm != 0 {
            let z = mm.trailing_zeros() as usize;
            mm &= mm - 1;
            let m0_z = m0_row[z];
            m0_my += m0_z;
            num += m0_z * r_row[z];
        }
        return ((num.max(1e-300) / m0_my.max(1e-300)) * inv_z, new_iz);
    }

    let inv_mc = 1.0 / (mx.count_ones() as f64);
    r_eff_buf[..k_alpha].fill(0.0);
    let mut mx_iter = mx;
    while mx_iter != 0 {
        let x = mx_iter.trailing_zeros() as usize;
        mx_iter &= mx_iter - 1;
        let r_row = &r_mat[x * k_alpha..(x + 1) * k_alpha];
        for z in 0..k_alpha {
            r_eff_buf[z] += r_row[z] * inv_mc;
        }
    }
    let dot = dot_k_unrolled(&m0_row[..k_alpha], &r_eff_buf[..k_alpha]);
    let inv_z = 1.0 / dot.max(1e-300);
    if my.count_ones() == 1 {
        return (r_eff_buf[yj].max(1e-300) * inv_z, None);
    }
    let mut num = 0.0;
    let mut m0_my = 0.0;
    let mut mm = my;
    while mm != 0 {
        let z = mm.trailing_zeros() as usize;
        mm &= mm - 1;
        let m0_z = m0_row[z];
        m0_my += m0_z;
        num += m0_z * r_eff_buf[z];
    }
    ((num.max(1e-300) / m0_my.max(1e-300)) * inv_z, None)
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn eval_ambig_emit_coarse(
    my: u64,
    yj: usize,
    full_mask: u64,
    k_alpha: usize,
    c_tilt_row: &[f64],
    ln_c_tilt_row: &[f64],
    rho_bi: f64,
    m0_row: &[f64],
) -> f64 {
    if my == full_mask {
        return 1.0;
    }
    if rho_bi <= 0.25 {
        let z_coarse = dot_k_unrolled(&m0_row[..k_alpha], &c_tilt_row[..k_alpha]);
        if my.count_ones() == 1 {
            return c_tilt_row[yj].max(1e-300) / z_coarse.max(1e-300);
        }
        let mut num = 0.0;
        let mut m0_my = 0.0;
        let mut mm = my;
        while mm != 0 {
            let z = mm.trailing_zeros() as usize;
            mm &= mm - 1;
            let m0_z = m0_row[z];
            m0_my += m0_z;
            num += m0_z * c_tilt_row[z];
        }
        return num.max(1e-300) / (m0_my * z_coarse).max(1e-300);
    }

    let pwr_c = 1.0 - 0.85 * rho_bi;
    let mut z_coarse = 0.0;
    for z in 0..k_alpha {
        z_coarse += m0_row[z] * (pwr_c * ln_c_tilt_row[z]).exp();
    }
    if my.count_ones() == 1 {
        return (pwr_c * ln_c_tilt_row[yj]).exp() / z_coarse.max(1e-300);
    }
    let mut num = 0.0;
    let mut m0_my = 0.0;
    let mut mm = my;
    while mm != 0 {
        let z = mm.trailing_zeros() as usize;
        mm &= mm - 1;
        let m0_z = m0_row[z];
        m0_my += m0_z;
        num += m0_z * (pwr_c * ln_c_tilt_row[z]).exp();
    }
    num.max(1e-300) / (m0_my * z_coarse).max(1e-300)
}

/// Execute exact Copula-Tilted Forward-Backward pass for P_T(Y | X; regime).
/// Strictly causal in Y (uses only M_0(. | Y_{<j}) at step j), guaranteeing:
///   \sum_{L_Y = 0}^\infty \sum_{Y \in \Sigma^{L_Y}} P_T(Y | X; regime) == 1.000000000000000,
/// and for every individual L_Y >= 1:
///   \sum_{Y \in \Sigma^{L_Y}} P_T(Y | X; regime) / P_len(L_Y | L_X; regime) == 1.000000000000000!
pub fn run_forward_backward_regime(
    seq_x: &EncodedSeq,
    seq_y: &EncodedSeq,
    prof_x: &SingleSeqNullProfile,
    prof_y: &SingleSeqNullProfile,
    local_coarse_tilts_x: &[f64],
    copula_ratio_mat: &[f64],
    _gc: &GeneticCode,
    regime: TransducerRegime,
    prior_mean_len: f64,
    compute_posteriors: bool,
) -> SingleRegimeFB {
    let lx = seq_x.len();
    let ly = seq_y.len();
    let k_alpha = seq_x.alphabet_size();
    assert!(lx >= 1, "Transducer requires Lx >= 1");

    let eta_y = prior_mean_len / (prior_mean_len + 1.0);
    let ln_eta_y = eta_y.ln();
    let ln_one_minus_eta_y = (1.0 / (prior_mean_len + 1.0)).ln();

    let (ln_start_x_0, ln_start_x_rest) = if lx == 1 {
        (0.0, NEG_INF)
    } else {
        (0.75f64.ln(), (0.25 / ((lx - 1) as f64)).ln())
    };
    let ln_pre_y_zero = 0.75f64.ln();
    let ln_pre_y_enter = 0.25f64.ln();

    let ln_enter_m = 0.95f64.ln();
    let ln_enter_bx = 0.05f64.ln();

    let t_eff = regime.t_primary;
    let p_gap_total = (0.0008 + 0.038 * (1.0 - (-t_eff).exp())).clamp(0.0008, 0.045);
    let p_m_to_d = 0.5 * p_gap_total;
    let p_m_to_i = 0.5 * p_gap_total;
    let p_m_to_bx = (0.0018 + 0.014 * (1.0 - (-0.8 * t_eff).exp())).clamp(0.0018, 0.018);
    let p_m_to_end = (0.25 / (lx as f64 + 10.0)).clamp(0.0002, 0.02);
    let p_m_to_m = 1.0 - p_m_to_d - p_m_to_i - p_m_to_bx - p_m_to_end;

    let ln_m_to_m = p_m_to_m.ln();
    let ln_m_to_d = p_m_to_d.ln();
    let ln_m_to_i = p_m_to_i.ln();
    let ln_m_to_bx = p_m_to_bx.ln();
    let ln_m_to_end = p_m_to_end.ln();

    let p_indel_ext = 0.65f64;
    let ln_indel_ext = p_indel_ext.ln();
    let ln_indel_exit = (1.0 - p_indel_ext).ln();

    let p_bx_ext = 0.90f64;
    let ln_bx_ext = p_bx_ext.ln();
    let ln_bx_to_by = (1.0 - p_bx_ext).ln();

    let p_by_ext = 0.90f64;
    let ln_by_ext = p_by_ext.ln();
    let ln_by_exit = (1.0 - p_by_ext).ln();

    let ln_coterm_eof = 0.75f64.ln();
    let ln_coterm_post = 0.25f64.ln();

    let ln_early_eof = 0.25f64.ln();
    let ln_early_post = 0.75f64.ln();

    let ln_m0_y = &prof_y.site_ln_m0;

    let mut ln_emit_hom = vec![0.0f64; lx * ly];
    let mut ln_emit_coarse = vec![0.0f64; lx * ly];

    // Quantize causal repeat weights onto 65 exact exponential-family temperature levels q in 0..=64:
    // Because q(i, j) = max(qx[i], qy[j]) depends ONLY on X and Y_{<j} (strictly prior to y_j),
    // every level q defines an exact normalized distribution \sum_{z=0}^{K-1} P_q(z | x_i, Y_{<j}) == 1.000000000000000,
    // while eliminating 99.9% of transcendental `powf` and `ln` calls from the O(Lx * Ly) inner loop!
    const Q_STEPS: usize = 64;
    let qx: Vec<u8> = (0..lx)
        .map(|i| {
            let r = prof_x.ref_repeat_weight[i];
            if r <= 0.05 {
                0
            } else {
                ((r * (Q_STEPS as f64)).round() as u8).clamp(1, Q_STEPS as u8)
            }
        })
        .collect();
    let qy: Vec<u8> = (0..ly)
        .map(|j| {
            let r = prof_y.site_repeat_weight[j];
            if r <= 0.05 {
                0
            } else {
                ((r * (Q_STEPS as f64)).round() as u8).clamp(1, Q_STEPS as u8)
            }
        })
        .collect();

    let mut used_q = [false; Q_STEPS + 1];
    used_q[0] = true;
    for &q in &qx {
        used_q[q as usize] = true;
    }
    for &q in &qy {
        used_q[q as usize] = true;
    }

    let mut ln_r_base = vec![0.0f64; k_alpha * k_alpha];
    for idx in 0..(k_alpha * k_alpha) {
        ln_r_base[idx] = copula_ratio_mat[idx].max(1e-300).ln();
    }

    let has_any_ambig = seq_x.has_ambig || seq_y.has_ambig;
    let full_mask: u64 = if k_alpha >= 64 {
        u64::MAX
    } else {
        (1u64 << k_alpha) - 1
    };

    let mut ln_z_hom_by_q: Vec<Vec<f64>> = vec![Vec::new(); Q_STEPS + 1];
    let mut r_pow_by_q: Vec<Vec<f64>> = if has_any_ambig {
        vec![Vec::new(); Q_STEPS + 1]
    } else {
        Vec::new()
    };
    for q in 0..=Q_STEPS {
        if !used_q[q] {
            continue;
        }
        let pwr = if q == 0 {
            1.0
        } else {
            1.0 - 0.78 * (q as f64 / (Q_STEPS as f64))
        };
        let r_pow: Vec<f64> = if q == 0 {
            copula_ratio_mat.to_vec()
        } else {
            ln_r_base.iter().map(|&lr| (pwr * lr).exp()).collect()
        };
        let mut ln_z = vec![0.0f64; ly * k_alpha];
        for j in 0..ly {
            let m0_row = &prof_y.site_m0_dist[j * k_alpha..(j + 1) * k_alpha];
            for x_tok in 0..k_alpha {
                let r_row = &r_pow[x_tok * k_alpha..(x_tok + 1) * k_alpha];
                let mut dot = 0.0;
                for z in 0..k_alpha {
                    dot += m0_row[z] * r_row[z];
                }
                ln_z[j * k_alpha + x_tok] = dot.max(1e-300).ln();
            }
        }
        ln_z_hom_by_q[q] = ln_z;
        if has_any_ambig {
            r_pow_by_q[q] = r_pow;
        }
    }

    let mut ln_c_tilt = vec![0.0f64; lx * k_alpha];
    for idx in 0..(lx * k_alpha) {
        ln_c_tilt[idx] = local_coarse_tilts_x[idx].max(1e-300).ln();
    }

    if !has_any_ambig {
        for i in 0..lx {
            let xi = seq_x.tokens[i] as usize;
            let qxi = qx[i];
            let rho_x = prof_x.ref_repeat_weight[i];
            let ln_r_row = &ln_r_base[xi * k_alpha..(xi + 1) * k_alpha];
            let c_tilt_row = &local_coarse_tilts_x[i * k_alpha..(i + 1) * k_alpha];
            let ln_c_tilt_row = &ln_c_tilt[i * k_alpha..(i + 1) * k_alpha];
            let hom_out = &mut ln_emit_hom[i * ly..(i + 1) * ly];
            let coarse_out = &mut ln_emit_coarse[i * ly..(i + 1) * ly];

            for j in 0..ly {
                let yj = seq_y.tokens[j] as usize;
                let q = qxi.max(qy[j]) as usize;
                let pwr = if q == 0 {
                    1.0
                } else {
                    1.0 - 0.78 * (q as f64 / (Q_STEPS as f64))
                };
                hom_out[j] =
                    ln_m0_y[j] + pwr * ln_r_row[yj] - ln_z_hom_by_q[q][j * k_alpha + xi];

                let rho_y = prof_y.site_repeat_weight[j];
                let m0_row = &prof_y.site_m0_dist[j * k_alpha..(j + 1) * k_alpha];
                let rho_bi = (rho_x * rho_y).sqrt();
                if rho_bi <= 0.25 {
                    let mut z_coarse = 0.0;
                    for z in 0..k_alpha {
                        z_coarse += m0_row[z] * c_tilt_row[z];
                    }
                    coarse_out[j] = ln_m0_y[j] + ln_c_tilt_row[yj] - z_coarse.max(1e-300).ln();
                } else {
                    let pwr_c = 1.0 - 0.85 * rho_bi;
                    let mut z_coarse = 0.0;
                    for z in 0..k_alpha {
                        z_coarse += m0_row[z] * (pwr_c * ln_c_tilt_row[z]).exp();
                    }
                    coarse_out[j] =
                        ln_m0_y[j] + pwr_c * ln_c_tilt_row[yj] - z_coarse.max(1e-300).ln();
                }
            }
        }
    } else {
        let mut r_eff_buf = vec![0.0f64; k_alpha];
        for i in 0..lx {
            let xi = seq_x.tokens[i] as usize;
            let mx = seq_x.ambig_masks[i];
            let qxi = qx[i];
            let rho_x = prof_x.ref_repeat_weight[i];
            let c_tilt_row = &local_coarse_tilts_x[i * k_alpha..(i + 1) * k_alpha];
            let ln_c_tilt_row = &ln_c_tilt[i * k_alpha..(i + 1) * k_alpha];
            let hom_out = &mut ln_emit_hom[i * ly..(i + 1) * ly];
            let coarse_out = &mut ln_emit_coarse[i * ly..(i + 1) * ly];

            for j in 0..ly {
                let yj = seq_y.tokens[j] as usize;
                let my = seq_y.ambig_masks[j];
                let q = qxi.max(qy[j]) as usize;
                let m0_row = &prof_y.site_m0_dist[j * k_alpha..(j + 1) * k_alpha];
                let inv_z_opt = if mx.count_ones() == 1 {
                    Some((-ln_z_hom_by_q[q][j * k_alpha + xi]).exp())
                } else {
                    None
                };
                let (e_hom, _) = eval_ambig_emit_hom(
                    mx,
                    xi,
                    my,
                    yj,
                    full_mask,
                    k_alpha,
                    &r_pow_by_q[q],
                    inv_z_opt,
                    m0_row,
                    &mut r_eff_buf,
                );
                hom_out[j] = ln_m0_y[j] + e_hom.max(1e-300).ln();

                let rho_y = prof_y.site_repeat_weight[j];
                let rho_bi = (rho_x * rho_y).sqrt();
                let e_coarse = eval_ambig_emit_coarse(
                    my,
                    yj,
                    full_mask,
                    k_alpha,
                    c_tilt_row,
                    ln_c_tilt_row,
                    rho_bi,
                    m0_row,
                );
                coarse_out[j] = ln_m0_y[j] + e_coarse.max(1e-300).ln();
            }
        }
    }

    #[inline]
    fn add_log(slot: &mut f64, val: f64) {
        *slot = logsumexp2(*slot, val);
    }

    // Fast L1-cache 2-row rolling Forward pass when posteriors are not requested (26/30 regime evaluations per pair):
    if !compute_posteriors {
        let row_len = (ly + 1) * N_STATES;
        let mut cur_row = vec![NEG_INF; row_len];
        let mut next_row = vec![NEG_INF; row_len];
        let mut post_y_row = vec![NEG_INF; ly + 1];
        let mut ln_eof_total = NEG_INF;

        for i in 0..=lx {
            next_row.fill(NEG_INF);
            if i < lx {
                let ln_sx = if i == 0 {
                    ln_start_x_0
                } else {
                    ln_start_x_rest
                };
                let ln_ec0 = ln_sx + ln_pre_y_zero;
                if ly >= 1 {
                    add_log(
                        &mut next_row[1 * N_STATES + ST_M],
                        ln_ec0 + ln_enter_m + ln_emit_hom[i * ly],
                    );
                    add_log(
                        &mut cur_row[1 * N_STATES + ST_PRE_Y],
                        ln_sx + ln_pre_y_enter + ln_m0_y[0],
                    );
                }
                add_log(&mut next_row[ST_BX], ln_ec0 + ln_enter_bx);
            } else {
                for j in 0..=ly {
                    cur_row[j * N_STATES + ST_POST_Y] = post_y_row[j];
                }
            }

            for j in 0..=ly {
                if i < lx && j >= 1 {
                    let cur = cur_row[j * N_STATES + ST_PRE_Y];
                    if cur > NEG_INF {
                        if j < ly {
                            add_log(
                                &mut cur_row[(j + 1) * N_STATES + ST_PRE_Y],
                                cur + ln_eta_y + ln_m0_y[j],
                            );
                        }
                        let ln_ec = cur + ln_one_minus_eta_y;
                        if j < ly {
                            add_log(
                                &mut next_row[(j + 1) * N_STATES + ST_M],
                                ln_ec + ln_enter_m + ln_emit_hom[i * ly + j],
                            );
                        }
                        add_log(&mut next_row[j * N_STATES + ST_BX], ln_ec + ln_enter_bx);
                    }
                }

                let cur_m = cur_row[j * N_STATES + ST_M];
                if cur_m > NEG_INF {
                    if i == lx {
                        if j == ly {
                            add_log(&mut ln_eof_total, cur_m + ln_coterm_eof);
                        }
                        if j < ly {
                            add_log(
                                &mut cur_row[(j + 1) * N_STATES + ST_POST_Y],
                                cur_m + ln_coterm_post + ln_m0_y[j],
                            );
                        }
                    } else {
                        let ln_ex = cur_m + ln_m_to_end;
                        if j == ly {
                            add_log(&mut ln_eof_total, ln_ex + ln_early_eof);
                        }
                        if j < ly {
                            add_log(
                                &mut post_y_row[j + 1],
                                ln_ex + ln_early_post + ln_m0_y[j],
                            );
                        }
                        add_log(&mut next_row[j * N_STATES + ST_D], cur_m + ln_m_to_d);
                        add_log(&mut next_row[j * N_STATES + ST_BX], cur_m + ln_m_to_bx);
                        if j < ly {
                            add_log(
                                &mut cur_row[(j + 1) * N_STATES + ST_I],
                                cur_m + ln_m_to_i + ln_m0_y[j],
                            );
                            add_log(
                                &mut next_row[(j + 1) * N_STATES + ST_M],
                                cur_m + ln_m_to_m + ln_emit_hom[i * ly + j],
                            );
                        }
                    }
                }

                let cur_d = cur_row[j * N_STATES + ST_D];
                if cur_d > NEG_INF {
                    if i == lx {
                        if j == ly {
                            add_log(&mut ln_eof_total, cur_d + ln_coterm_eof);
                        }
                        if j < ly {
                            add_log(
                                &mut cur_row[(j + 1) * N_STATES + ST_POST_Y],
                                cur_d + ln_coterm_post + ln_m0_y[j],
                            );
                        }
                    } else {
                        add_log(&mut next_row[j * N_STATES + ST_D], cur_d + ln_indel_ext);
                        if j < ly {
                            add_log(
                                &mut next_row[(j + 1) * N_STATES + ST_M],
                                cur_d + ln_indel_exit + ln_emit_hom[i * ly + j],
                            );
                        }
                    }
                }

                if i >= 1 && i < lx && j >= 1 {
                    let cur_i = cur_row[j * N_STATES + ST_I];
                    if cur_i > NEG_INF && j < ly {
                        add_log(
                            &mut cur_row[(j + 1) * N_STATES + ST_I],
                            cur_i + ln_indel_ext + ln_m0_y[j],
                        );
                        add_log(
                            &mut next_row[(j + 1) * N_STATES + ST_M],
                            cur_i + ln_indel_exit + ln_emit_hom[i * ly + j],
                        );
                    }
                }

                let cur_bx = cur_row[j * N_STATES + ST_BX];
                if cur_bx > NEG_INF {
                    let x_ref = (i - 1).min(lx - 1);
                    if i == lx {
                        if j < ly {
                            add_log(
                                &mut cur_row[(j + 1) * N_STATES + ST_BY],
                                cur_bx + ln_emit_coarse[x_ref * ly + j],
                            );
                        }
                    } else {
                        add_log(&mut next_row[j * N_STATES + ST_BX], cur_bx + ln_bx_ext);
                        if j < ly {
                            add_log(
                                &mut cur_row[(j + 1) * N_STATES + ST_BY],
                                cur_bx + ln_bx_to_by + ln_emit_coarse[x_ref * ly + j],
                            );
                        }
                    }
                }

                let cur_by = cur_row[j * N_STATES + ST_BY];
                if cur_by > NEG_INF {
                    let x_ref = (i - 1).min(lx - 1);
                    if j < ly {
                        add_log(
                            &mut cur_row[(j + 1) * N_STATES + ST_BY],
                            cur_by + ln_by_ext + ln_emit_coarse[x_ref * ly + j],
                        );
                    }
                    if i == lx {
                        let ln_ex = cur_by + ln_by_exit;
                        if j == ly {
                            add_log(&mut ln_eof_total, ln_ex + ln_coterm_eof);
                        }
                        if j < ly {
                            add_log(
                                &mut cur_row[(j + 1) * N_STATES + ST_POST_Y],
                                ln_ex + ln_coterm_post + ln_m0_y[j],
                            );
                        }
                    } else if j < ly {
                        add_log(
                            &mut next_row[(j + 1) * N_STATES + ST_M],
                            cur_by + ln_by_exit + ln_emit_hom[i * ly + j],
                        );
                    }
                }

                if i == lx && j >= 1 {
                    let cur_post = cur_row[j * N_STATES + ST_POST_Y];
                    if cur_post > NEG_INF {
                        if j == ly {
                            add_log(&mut ln_eof_total, cur_post + ln_one_minus_eta_y);
                        }
                        if j < ly {
                            add_log(
                                &mut cur_row[(j + 1) * N_STATES + ST_POST_Y],
                                cur_post + ln_eta_y + ln_m0_y[j],
                            );
                        }
                    }
                }
            }
            std::mem::swap(&mut cur_row, &mut next_row);
        }

        return SingleRegimeFB {
            ln_prob_y_given_x: ln_eof_total,
            post_fine: Vec::new(),
            post_bx: Vec::new(),
            post_by: Vec::new(),
        };
    }

    let stride_j = N_STATES;
    let stride_i = (ly + 1) * N_STATES;
    let mut fwd = vec![NEG_INF; (lx + 1) * (ly + 1) * N_STATES];

    #[inline]
    fn cell(i: usize, j: usize, s: usize, stride_i: usize, stride_j: usize) -> usize {
        i * stride_i + j * stride_j + s
    }

    let mut ln_eof_total = NEG_INF;

    for i0 in 0..lx {
        let ln_sx = if i0 == 0 {
            ln_start_x_0
        } else {
            ln_start_x_rest
        };
        let ln_ec0 = ln_sx + ln_pre_y_zero;
        if ly >= 1 {
            let v_m = ln_ec0 + ln_enter_m + ln_emit_hom[i0 * ly + 0];
            add_log(&mut fwd[cell(i0 + 1, 1, ST_M, stride_i, stride_j)], v_m);
        }
        let v_bx = ln_ec0 + ln_enter_bx;
        add_log(&mut fwd[cell(i0 + 1, 0, ST_BX, stride_i, stride_j)], v_bx);

        if ly >= 1 {
            let v_pre = ln_sx + ln_pre_y_enter + ln_m0_y[0];
            add_log(&mut fwd[cell(i0, 1, ST_PRE_Y, stride_i, stride_j)], v_pre);
        }
    }

    for i in 0..=lx {
        for j in 0..=ly {
            if i < lx && j >= 1 {
                let cur = fwd[cell(i, j, ST_PRE_Y, stride_i, stride_j)];
                if cur > NEG_INF {
                    if j < ly {
                        add_log(
                            &mut fwd[cell(i, j + 1, ST_PRE_Y, stride_i, stride_j)],
                            cur + ln_eta_y + ln_m0_y[j],
                        );
                    }
                    let ln_ec = cur + ln_one_minus_eta_y;
                    if j < ly {
                        add_log(
                            &mut fwd[cell(i + 1, j + 1, ST_M, stride_i, stride_j)],
                            ln_ec + ln_enter_m + ln_emit_hom[i * ly + j],
                        );
                    }
                    add_log(
                        &mut fwd[cell(i + 1, j, ST_BX, stride_i, stride_j)],
                        ln_ec + ln_enter_bx,
                    );
                }
            }

            let cur_m = fwd[cell(i, j, ST_M, stride_i, stride_j)];
            if cur_m > NEG_INF {
                if i == lx {
                    if j == ly {
                        add_log(&mut ln_eof_total, cur_m + ln_coterm_eof);
                    }
                    if j < ly {
                        add_log(
                            &mut fwd[cell(lx, j + 1, ST_POST_Y, stride_i, stride_j)],
                            cur_m + ln_coterm_post + ln_m0_y[j],
                        );
                    }
                } else {
                    let ln_ex = cur_m + ln_m_to_end;
                    if j == ly {
                        add_log(&mut ln_eof_total, ln_ex + ln_early_eof);
                    }
                    if j < ly {
                        add_log(
                            &mut fwd[cell(lx, j + 1, ST_POST_Y, stride_i, stride_j)],
                            ln_ex + ln_early_post + ln_m0_y[j],
                        );
                    }
                    add_log(
                        &mut fwd[cell(i + 1, j, ST_D, stride_i, stride_j)],
                        cur_m + ln_m_to_d,
                    );
                    add_log(
                        &mut fwd[cell(i + 1, j, ST_BX, stride_i, stride_j)],
                        cur_m + ln_m_to_bx,
                    );
                    if j < ly {
                        add_log(
                            &mut fwd[cell(i, j + 1, ST_I, stride_i, stride_j)],
                            cur_m + ln_m_to_i + ln_m0_y[j],
                        );
                        add_log(
                            &mut fwd[cell(i + 1, j + 1, ST_M, stride_i, stride_j)],
                            cur_m + ln_m_to_m + ln_emit_hom[i * ly + j],
                        );
                    }
                }
            }

            let cur_d = fwd[cell(i, j, ST_D, stride_i, stride_j)];
            if cur_d > NEG_INF {
                if i == lx {
                    if j == ly {
                        add_log(&mut ln_eof_total, cur_d + ln_coterm_eof);
                    }
                    if j < ly {
                        add_log(
                            &mut fwd[cell(lx, j + 1, ST_POST_Y, stride_i, stride_j)],
                            cur_d + ln_coterm_post + ln_m0_y[j],
                        );
                    }
                } else {
                    add_log(
                        &mut fwd[cell(i + 1, j, ST_D, stride_i, stride_j)],
                        cur_d + ln_indel_ext,
                    );
                    if j < ly {
                        add_log(
                            &mut fwd[cell(i + 1, j + 1, ST_M, stride_i, stride_j)],
                            cur_d + ln_indel_exit + ln_emit_hom[i * ly + j],
                        );
                    }
                }
            }

            if i >= 1 && i < lx && j >= 1 {
                let cur_i = fwd[cell(i, j, ST_I, stride_i, stride_j)];
                if cur_i > NEG_INF && j < ly {
                    add_log(
                        &mut fwd[cell(i, j + 1, ST_I, stride_i, stride_j)],
                        cur_i + ln_indel_ext + ln_m0_y[j],
                    );
                    add_log(
                        &mut fwd[cell(i + 1, j + 1, ST_M, stride_i, stride_j)],
                        cur_i + ln_indel_exit + ln_emit_hom[i * ly + j],
                    );
                }
            }

            let cur_bx = fwd[cell(i, j, ST_BX, stride_i, stride_j)];
            if cur_bx > NEG_INF {
                let x_ref = (i - 1).min(lx - 1);
                if i == lx {
                    if j < ly {
                        add_log(
                            &mut fwd[cell(lx, j + 1, ST_BY, stride_i, stride_j)],
                            cur_bx + ln_emit_coarse[x_ref * ly + j],
                        );
                    }
                } else {
                    add_log(
                        &mut fwd[cell(i + 1, j, ST_BX, stride_i, stride_j)],
                        cur_bx + ln_bx_ext,
                    );
                    if j < ly {
                        add_log(
                            &mut fwd[cell(i, j + 1, ST_BY, stride_i, stride_j)],
                            cur_bx + ln_bx_to_by + ln_emit_coarse[x_ref * ly + j],
                        );
                    }
                }
            }

            let cur_by = fwd[cell(i, j, ST_BY, stride_i, stride_j)];
            if cur_by > NEG_INF {
                let x_ref = (i - 1).min(lx - 1);
                if j < ly {
                    add_log(
                        &mut fwd[cell(i, j + 1, ST_BY, stride_i, stride_j)],
                        cur_by + ln_by_ext + ln_emit_coarse[x_ref * ly + j],
                    );
                }
                if i == lx {
                    let ln_ex = cur_by + ln_by_exit;
                    if j == ly {
                        add_log(&mut ln_eof_total, ln_ex + ln_coterm_eof);
                    }
                    if j < ly {
                        add_log(
                            &mut fwd[cell(lx, j + 1, ST_POST_Y, stride_i, stride_j)],
                            ln_ex + ln_coterm_post + ln_m0_y[j],
                        );
                    }
                } else if j < ly {
                    add_log(
                        &mut fwd[cell(i + 1, j + 1, ST_M, stride_i, stride_j)],
                        cur_by + ln_by_exit + ln_emit_hom[i * ly + j],
                    );
                }
            }

            if i == lx && j >= 1 {
                let cur_post = fwd[cell(lx, j, ST_POST_Y, stride_i, stride_j)];
                if cur_post > NEG_INF {
                    if j == ly {
                        add_log(&mut ln_eof_total, cur_post + ln_one_minus_eta_y);
                    }
                    if j < ly {
                        add_log(
                            &mut fwd[cell(lx, j + 1, ST_POST_Y, stride_i, stride_j)],
                            cur_post + ln_eta_y + ln_m0_y[j],
                        );
                    }
                }
            }
        }
    }

    if !compute_posteriors || ln_eof_total == NEG_INF {
        return SingleRegimeFB {
            ln_prob_y_given_x: ln_eof_total,
            post_fine: Vec::new(),
            post_bx: Vec::new(),
            post_by: Vec::new(),
        };
    }

    let mut bwd = vec![NEG_INF; (lx + 1) * (ly + 1) * N_STATES];

    for i in (0..=lx).rev() {
        for j in (0..=ly).rev() {
            if i == lx && j >= 1 {
                let mut v = NEG_INF;
                if j == ly {
                    v = logsumexp2(v, ln_one_minus_eta_y);
                }
                if j < ly {
                    v = logsumexp2(
                        v,
                        ln_eta_y + ln_m0_y[j] + bwd[cell(lx, j + 1, ST_POST_Y, stride_i, stride_j)],
                    );
                }
                bwd[cell(lx, j, ST_POST_Y, stride_i, stride_j)] = v;
            }

            if i >= 1 && j >= 1 {
                let x_ref = (i - 1).min(lx - 1);
                let mut v = NEG_INF;
                if j < ly {
                    v = logsumexp2(
                        v,
                        ln_by_ext
                            + ln_emit_coarse[x_ref * ly + j]
                            + bwd[cell(i, j + 1, ST_BY, stride_i, stride_j)],
                    );
                }
                if i == lx {
                    if j == ly {
                        v = logsumexp2(v, ln_by_exit + ln_coterm_eof);
                    }
                    if j < ly {
                        v = logsumexp2(
                            v,
                            ln_by_exit
                                + ln_coterm_post
                                + ln_m0_y[j]
                                + bwd[cell(lx, j + 1, ST_POST_Y, stride_i, stride_j)],
                        );
                    }
                } else if j < ly {
                    v = logsumexp2(
                        v,
                        ln_by_exit
                            + ln_emit_hom[i * ly + j]
                            + bwd[cell(i + 1, j + 1, ST_M, stride_i, stride_j)],
                    );
                }
                bwd[cell(i, j, ST_BY, stride_i, stride_j)] = v;
            }

            if i >= 1 {
                let x_ref = (i - 1).min(lx - 1);
                let mut v = NEG_INF;
                if i == lx {
                    if j < ly {
                        v = logsumexp2(
                            v,
                            ln_emit_coarse[x_ref * ly + j]
                                + bwd[cell(lx, j + 1, ST_BY, stride_i, stride_j)],
                        );
                    }
                } else {
                    v = logsumexp2(
                        v,
                        ln_bx_ext + bwd[cell(i + 1, j, ST_BX, stride_i, stride_j)],
                    );
                    if j < ly {
                        v = logsumexp2(
                            v,
                            ln_bx_to_by
                                + ln_emit_coarse[x_ref * ly + j]
                                + bwd[cell(i, j + 1, ST_BY, stride_i, stride_j)],
                        );
                    }
                }
                bwd[cell(i, j, ST_BX, stride_i, stride_j)] = v;
            }

            if i >= 1 && i < lx && j >= 1 {
                let mut v = NEG_INF;
                if j < ly {
                    v = logsumexp2(
                        v,
                        ln_indel_ext
                            + ln_m0_y[j]
                            + bwd[cell(i, j + 1, ST_I, stride_i, stride_j)],
                    );
                    v = logsumexp2(
                        v,
                        ln_indel_exit
                            + ln_emit_hom[i * ly + j]
                            + bwd[cell(i + 1, j + 1, ST_M, stride_i, stride_j)],
                    );
                }
                bwd[cell(i, j, ST_I, stride_i, stride_j)] = v;
            }

            if i >= 1 {
                let mut v = NEG_INF;
                if i == lx {
                    if j == ly {
                        v = logsumexp2(v, ln_coterm_eof);
                    }
                    if j < ly {
                        v = logsumexp2(
                            v,
                            ln_coterm_post
                                + ln_m0_y[j]
                                + bwd[cell(lx, j + 1, ST_POST_Y, stride_i, stride_j)],
                        );
                    }
                } else {
                    v = logsumexp2(
                        v,
                        ln_indel_ext + bwd[cell(i + 1, j, ST_D, stride_i, stride_j)],
                    );
                    if j < ly {
                        v = logsumexp2(
                            v,
                            ln_indel_exit
                                + ln_emit_hom[i * ly + j]
                                + bwd[cell(i + 1, j + 1, ST_M, stride_i, stride_j)],
                        );
                    }
                }
                bwd[cell(i, j, ST_D, stride_i, stride_j)] = v;
            }

            if i >= 1 && j >= 1 {
                let mut v = NEG_INF;
                if i == lx {
                    if j == ly {
                        v = logsumexp2(v, ln_coterm_eof);
                    }
                    if j < ly {
                        v = logsumexp2(
                            v,
                            ln_coterm_post
                                + ln_m0_y[j]
                                + bwd[cell(lx, j + 1, ST_POST_Y, stride_i, stride_j)],
                        );
                    }
                } else {
                    if j == ly {
                        v = logsumexp2(v, ln_m_to_end + ln_early_eof);
                    }
                    if j < ly {
                        v = logsumexp2(
                            v,
                            ln_m_to_end
                                + ln_early_post
                                + ln_m0_y[j]
                                + bwd[cell(lx, j + 1, ST_POST_Y, stride_i, stride_j)],
                        );
                    }
                    v = logsumexp2(
                        v,
                        ln_m_to_d + bwd[cell(i + 1, j, ST_D, stride_i, stride_j)],
                    );
                    v = logsumexp2(
                        v,
                        ln_m_to_bx + bwd[cell(i + 1, j, ST_BX, stride_i, stride_j)],
                    );
                    if j < ly {
                        v = logsumexp2(
                            v,
                            ln_m_to_i
                                + ln_m0_y[j]
                                + bwd[cell(i, j + 1, ST_I, stride_i, stride_j)],
                        );
                        v = logsumexp2(
                            v,
                            ln_m_to_m
                                + ln_emit_hom[i * ly + j]
                                + bwd[cell(i + 1, j + 1, ST_M, stride_i, stride_j)],
                        );
                    }
                }
                bwd[cell(i, j, ST_M, stride_i, stride_j)] = v;
            }

            if i < lx && j >= 1 {
                let mut v = NEG_INF;
                if j < ly {
                    v = logsumexp2(
                        v,
                        ln_eta_y
                            + ln_m0_y[j]
                            + bwd[cell(i, j + 1, ST_PRE_Y, stride_i, stride_j)],
                    );
                    v = logsumexp2(
                        v,
                        ln_one_minus_eta_y
                            + ln_enter_m
                            + ln_emit_hom[i * ly + j]
                            + bwd[cell(i + 1, j + 1, ST_M, stride_i, stride_j)],
                    );
                }
                v = logsumexp2(
                    v,
                    ln_one_minus_eta_y
                        + ln_enter_bx
                        + bwd[cell(i + 1, j, ST_BX, stride_i, stride_j)],
                );
                bwd[cell(i, j, ST_PRE_Y, stride_i, stride_j)] = v;
            }
        }
    }

    let mut post_fine = vec![0.0f64; lx * ly];
    let mut post_bx = vec![0.0f64; lx];
    let mut post_by = vec![0.0f64; ly];

    for i in 0..lx {
        let ln_sx = if i == 0 {
            ln_start_x_0
        } else {
            ln_start_x_rest
        };
        for j in 0..ly {
            let b_after = bwd[cell(i + 1, j + 1, ST_M, stride_i, stride_j)];
            if b_after > NEG_INF {
                let mut ln_into = NEG_INF;
                if j == 0 {
                    ln_into = logsumexp2(ln_into, ln_sx + ln_pre_y_zero + ln_enter_m);
                } else {
                    let f_pre = fwd[cell(i, j, ST_PRE_Y, stride_i, stride_j)];
                    if f_pre > NEG_INF {
                        ln_into = logsumexp2(ln_into, f_pre + ln_one_minus_eta_y + ln_enter_m);
                    }
                }
                if i >= 1 && j >= 1 {
                    let f_m = fwd[cell(i, j, ST_M, stride_i, stride_j)];
                    if f_m > NEG_INF {
                        ln_into = logsumexp2(ln_into, f_m + ln_m_to_m);
                    }
                    let f_i = fwd[cell(i, j, ST_I, stride_i, stride_j)];
                    if f_i > NEG_INF {
                        ln_into = logsumexp2(ln_into, f_i + ln_indel_exit);
                    }
                    let f_by = fwd[cell(i, j, ST_BY, stride_i, stride_j)];
                    if f_by > NEG_INF {
                        ln_into = logsumexp2(ln_into, f_by + ln_by_exit);
                    }
                }
                if i >= 1 {
                    let f_d = fwd[cell(i, j, ST_D, stride_i, stride_j)];
                    if f_d > NEG_INF {
                        ln_into = logsumexp2(ln_into, f_d + ln_indel_exit);
                    }
                }
                if ln_into > NEG_INF {
                    let ln_p = ln_into + ln_emit_hom[i * ly + j] + b_after - ln_eof_total;
                    post_fine[i * ly + j] = ln_p.exp().clamp(0.0, 1.0);
                }
            }
        }
    }

    for i in 0..lx {
        let mut ln_tot = NEG_INF;
        for j in 0..=ly {
            let f_bx = fwd[cell(i + 1, j, ST_BX, stride_i, stride_j)];
            let b_bx = bwd[cell(i + 1, j, ST_BX, stride_i, stride_j)];
            if f_bx > NEG_INF && b_bx > NEG_INF {
                ln_tot = logsumexp2(ln_tot, f_bx + b_bx - ln_eof_total);
            }
        }
        if ln_tot > NEG_INF {
            post_bx[i] = ln_tot.exp().clamp(0.0, 1.0);
        }
    }

    for j in 0..ly {
        let mut ln_tot = NEG_INF;
        for i in 1..=lx {
            let f_by = fwd[cell(i, j + 1, ST_BY, stride_i, stride_j)];
            let b_by = bwd[cell(i, j + 1, ST_BY, stride_i, stride_j)];
            if f_by > NEG_INF && b_by > NEG_INF {
                ln_tot = logsumexp2(ln_tot, f_by + b_by - ln_eof_total);
            }
        }
        if ln_tot > NEG_INF {
            post_by[j] = ln_tot.exp().clamp(0.0, 1.0);
        }
    }

    SingleRegimeFB {
        ln_prob_y_given_x: ln_eof_total,
        post_fine,
        post_bx,
        post_by,
    }
}

const BLK_SHIFT: usize = 4;
const BLK_MASK: usize = 15;
const ZERO_EXP: i32 = i32::MIN / 4;

const NB: usize = 5;
const BM: usize = 0;
const BD: usize = 1;
const BI: usize = 2;
const BBX: usize = 3;
const BBY: usize = 4;

#[inline(always)]
fn ldexp_f64(mut val: f64, mut exp2: i32) -> f64 {
    if val == 0.0 || exp2 == 0 {
        return val;
    }
    if exp2 < -2200 {
        return 0.0;
    }
    while exp2 < -1022 {
        val *= f64::from_bits(1u64 << 52);
        exp2 += 1022;
    }
    while exp2 > 1023 {
        val *= f64::from_bits(2046u64 << 52);
        exp2 -= 1023;
    }
    val * f64::from_bits(((1023 + exp2) as u64) << 52)
}

#[inline(always)]
fn canon_scalar(m: &mut f64, e: &mut i32) {
    if *m <= 0.0 || *e == ZERO_EXP {
        *m = 0.0;
        *e = ZERO_EXP;
        return;
    }
    let bits = m.to_bits();
    let mut raw_exp = ((bits >> 52) & 0x7ff) as i32;
    if raw_exp == 0 {
        *m *= f64::from_bits((1023u64 + 54) << 52);
        let b2 = m.to_bits();
        raw_exp = ((b2 >> 52) & 0x7ff) as i32;
        let e2 = raw_exp - 1023;
        *m *= f64::from_bits(((1023 - e2) as u64) << 52);
        *e += e2 - 54;
    } else {
        let e2 = raw_exp - 1023;
        if e2 > 40 || e2 < -40 {
            *m *= f64::from_bits(((1023 - e2) as u64) << 52);
            *e += e2;
        }
    }
}

#[inline(always)]
fn add_scaled(dst_m: &mut f64, dst_e: &mut i32, mut val_m: f64, mut val_e: i32) {
    if val_m <= 0.0 || val_e == ZERO_EXP {
        return;
    }
    canon_scalar(&mut val_m, &mut val_e);
    if val_e == ZERO_EXP {
        return;
    }
    if *dst_e == ZERO_EXP || *dst_m <= 0.0 {
        *dst_m = val_m;
        *dst_e = val_e;
        return;
    }
    let diff = val_e - *dst_e;
    if diff >= 0 {
        if diff > 140 {
            *dst_m = val_m;
            *dst_e = val_e;
        } else {
            *dst_m = val_m + ldexp_f64(*dst_m, -diff);
            *dst_e = val_e;
        }
    } else if diff >= -140 {
        *dst_m += ldexp_f64(val_m, diff);
    }
    canon_scalar(dst_m, dst_e);
}

#[inline(always)]
fn scaled_to_ln(m: f64, e: i32) -> f64 {
    if m <= 0.0 || e == ZERO_EXP {
        NEG_INF
    } else {
        m.ln() + (e as f64) * std::f64::consts::LN_2
    }
}

#[inline]
fn normalize_block_slice(slice: &mut [f64], base_exp: i32) -> i32 {
    if base_exp == ZERO_EXP {
        slice.fill(0.0);
        return ZERO_EXP;
    }
    let mut max_bits = 0u64;
    for &v in slice.iter() {
        let b = if v > 0.0 { v.to_bits() } else { 0 };
        if b > max_bits {
            max_bits = b;
        }
    }
    if max_bits == 0 {
        return ZERO_EXP;
    }
    let mut raw_exp = ((max_bits >> 52) & 0x7ff) as i32;
    let mut exp_shift = 0i32;
    if raw_exp == 0 {
        let up = f64::from_bits((1023u64 + 54) << 52);
        max_bits = 0;
        for v in slice.iter_mut() {
            *v *= up;
            let b = if *v > 0.0 { v.to_bits() } else { 0 };
            if b > max_bits {
                max_bits = b;
            }
        }
        if max_bits == 0 {
            return ZERO_EXP;
        }
        raw_exp = ((max_bits >> 52) & 0x7ff) as i32;
        exp_shift = -54;
    }
    let e_max = raw_exp - 1023;
    if e_max > 64 || e_max < -64 || exp_shift != 0 {
        let scale = f64::from_bits(((1023 - e_max) as u64) << 52);
        for v in slice.iter_mut() {
            *v *= scale;
        }
        base_exp + e_max + exp_shift
    } else {
        base_exp
    }
}

#[derive(Debug, Clone, Copy)]
struct LinearTransducerParams {
    eta_y: f64,
    one_minus_eta_y: f64,
    sx_0: f64,
    sx_rest: f64,
    pre_y_zero: f64,
    pre_y_enter: f64,
    enter_m: f64,
    enter_bx: f64,
    m_to_m: f64,
    m_to_d: f64,
    m_to_i: f64,
    m_to_bx: f64,
    m_to_end: f64,
    indel_ext: f64,
    indel_exit: f64,
    bx_ext: f64,
    bx_to_by: f64,
    by_ext: f64,
    by_exit: f64,
    coterm_eof: f64,
    coterm_post: f64,
    early_eof: f64,
    early_post: f64,
}

impl LinearTransducerParams {
    #[inline]
    fn new(lx: usize, regime: TransducerRegime, prior_mean_len: f64) -> Self {
        let eta_y = prior_mean_len / (prior_mean_len + 1.0);
        let one_minus_eta_y = 1.0 / (prior_mean_len + 1.0);

        let (sx_0, sx_rest) = if lx == 1 {
            (1.0, 0.0)
        } else {
            (0.75, 0.25 / ((lx - 1) as f64))
        };
        let t_eff = regime.t_primary;
        let p_gap_total = (0.0008 + 0.038 * (1.0 - (-t_eff).exp())).clamp(0.0008, 0.045);
        let m_to_d = 0.5 * p_gap_total;
        let m_to_i = 0.5 * p_gap_total;
        let m_to_bx = (0.0018 + 0.014 * (1.0 - (-0.8 * t_eff).exp())).clamp(0.0018, 0.018);
        let m_to_end = (0.25 / (lx as f64 + 10.0)).clamp(0.0002, 0.02);
        let m_to_m = 1.0 - m_to_d - m_to_i - m_to_bx - m_to_end;

        Self {
            eta_y,
            one_minus_eta_y,
            sx_0,
            sx_rest,
            pre_y_zero: 0.75,
            pre_y_enter: 0.25,
            enter_m: 0.95,
            enter_bx: 0.05,
            m_to_m,
            m_to_d,
            m_to_i,
            m_to_bx,
            m_to_end,
            indel_ext: 0.65,
            indel_exit: 0.35,
            bx_ext: 0.90,
            bx_to_by: 0.10,
            by_ext: 0.90,
            by_exit: 0.10,
            coterm_eof: 0.75,
            coterm_post: 0.25,
            early_eof: 0.25,
            early_post: 0.75,
        }
    }
}

fn build_linear_e_pre(
    p: &LinearTransducerParams,
    ly: usize,
    e_pre_m: &mut Vec<f64>,
    e_pre_blk_exp: &mut Vec<i32>,
) {
    let n_blks = (ly >> BLK_SHIFT) + 1;
    e_pre_m.resize(ly + 1, 0.0);
    e_pre_blk_exp.resize(n_blks, 0);

    let mut pre_u_m = p.pre_y_enter;
    let mut pre_u_e = 0i32;

    for b in 0..n_blks {
        let jb_lo = b << BLK_SHIFT;
        let jb_hi = (jb_lo + BLK_MASK).min(ly);
        if b == 0 {
            e_pre_m[0] = p.pre_y_zero;
            for j in 1..=jb_hi {
                e_pre_m[j] = pre_u_m * p.one_minus_eta_y;
                if j < ly {
                    pre_u_m *= p.eta_y;
                }
            }
            e_pre_blk_exp[0] = normalize_block_slice(&mut e_pre_m[0..=jb_hi], 0);
        } else {
            canon_scalar(&mut pre_u_m, &mut pre_u_e);
            let base_e = pre_u_e;
            for j in jb_lo..=jb_hi {
                e_pre_m[j] = pre_u_m * p.one_minus_eta_y;
                if j < ly {
                    pre_u_m *= p.eta_y;
                }
            }
            e_pre_blk_exp[b] = normalize_block_slice(&mut e_pre_m[jb_lo..=jb_hi], base_e);
        }
    }
}

fn build_linear_post_y_bwd(
    p: &LinearTransducerParams,
    ly: usize,
    post_m: &mut Vec<f64>,
    post_blk_exp: &mut Vec<i32>,
) {
    let n_blks = (ly >> BLK_SHIFT) + 1;
    post_m.resize(ly + 1, 0.0);
    post_blk_exp.resize(n_blks, ZERO_EXP);
    if ly == 0 {
        return;
    }

    let mut u_m = p.one_minus_eta_y;
    let mut u_e = 0i32;
    let last_blk = ly >> BLK_SHIFT;

    for b in (0..=last_blk).rev() {
        let jb_lo = (b << BLK_SHIFT).max(1);
        let jb_hi = (jb_lo.max(b << BLK_SHIFT) | BLK_MASK).min(ly);
        canon_scalar(&mut u_m, &mut u_e);
        let base_e = u_e;
        if b == 0 {
            post_m[0] = 0.0;
        }
        for j in (jb_lo..=jb_hi).rev() {
            post_m[j] = u_m;
            if j > 1 {
                u_m *= p.eta_y;
            }
        }
        post_blk_exp[b] = normalize_block_slice(&mut post_m[jb_lo..=jb_hi], base_e);
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn forward_row_step<const HAS_EMIT: bool, const COLLECT_ALL_LY: bool>(
    i: usize,
    lx: usize,
    ly: usize,
    p: &LinearTransducerParams,
    p_lo: usize,
    p_hi: usize,
    prev_slice: &[f64],
    p_base_j: usize,
    prev_blk_exp: &[i32],
    p_b_lo: usize,
    c_lo: usize,
    c_hi: usize,
    cur_slice: &mut [f64],
    c_base_j: usize,
    cur_blk_exp: &mut [i32],
    c_b_lo: usize,
    e_pre_m: &[f64],
    e_pre_blk_exp: &[i32],
    emit_hom: &[f64],
    emit_coarse: &[f64],
    e_prev_base: usize,
    e_prev_lo: usize,
    eof_m: &mut [f64],
    eof_e: &mut [i32],
    post_y_m: &mut [f64],
    post_y_e: &mut [i32],
) {
    let sx = if i == 1 { p.sx_0 } else { p.sx_rest };
    let bx_to_by_w = if i == lx { 1.0 } else { p.bx_to_by };
    let has_prev = i >= 2 && p_lo <= p_hi;
    let pb_lo = p_lo >> BLK_SHIFT;
    let pb_hi = p_hi >> BLK_SHIFT;

    let b_start = c_lo >> BLK_SHIFT;
    let b_end = c_hi >> BLK_SHIFT;

    for b in b_start..=b_end {
        let jb_lo = (b << BLK_SHIFT).max(c_lo);
        let jb_hi = ((b << BLK_SHIFT) | BLK_MASK).min(c_hi);
        let need_left_block = (jb_lo & BLK_MASK) == 0 && jb_lo > 0;

        let mut e_ref = ZERO_EXP;
        if e_pre_blk_exp[b] != ZERO_EXP {
            e_ref = e_pre_blk_exp[b];
        }
        if need_left_block && e_pre_blk_exp[b - 1] != ZERO_EXP && e_pre_blk_exp[b - 1] > e_ref {
            e_ref = e_pre_blk_exp[b - 1];
        }
        if has_prev {
            if b >= pb_lo && b <= pb_hi {
                let pe = prev_blk_exp[b - p_b_lo];
                if pe != ZERO_EXP && pe > e_ref {
                    e_ref = pe;
                }
            }
            if need_left_block && (b - 1) >= pb_lo && (b - 1) <= pb_hi {
                let pe = prev_blk_exp[(b - 1) - p_b_lo];
                if pe != ZERO_EXP && pe > e_ref {
                    e_ref = pe;
                }
            }
        }
        if b > b_start {
            let ce = cur_blk_exp[(b - 1) - c_b_lo];
            if ce != ZERO_EXP && ce > e_ref {
                e_ref = ce;
            }
        }

        if e_ref == ZERO_EXP {
            cur_blk_exp[b - c_b_lo] = ZERO_EXP;
            let clear_lo = (jb_lo - c_base_j) * NB;
            let clear_hi = (jb_hi - c_base_j + 1) * NB;
            cur_slice[clear_lo..clear_hi].fill(0.0);
            continue;
        }

        let s_ep_same = if e_pre_blk_exp[b] != ZERO_EXP {
            ldexp_f64(1.0, e_pre_blk_exp[b] - e_ref)
        } else {
            0.0
        };
        let s_ep_left = if need_left_block && e_pre_blk_exp[b - 1] != ZERO_EXP {
            ldexp_f64(1.0, e_pre_blk_exp[b - 1] - e_ref)
        } else {
            0.0
        };
        let (s_p_same, s_p_left) = if has_prev {
            let ss = if b >= pb_lo && b <= pb_hi && prev_blk_exp[b - p_b_lo] != ZERO_EXP {
                ldexp_f64(1.0, prev_blk_exp[b - p_b_lo] - e_ref)
            } else {
                0.0
            };
            let sl = if need_left_block
                && (b - 1) >= pb_lo
                && (b - 1) <= pb_hi
                && prev_blk_exp[(b - 1) - p_b_lo] != ZERO_EXP
            {
                ldexp_f64(1.0, prev_blk_exp[(b - 1) - p_b_lo] - e_ref)
            } else {
                0.0
            };
            (ss, sl)
        } else {
            (0.0, 0.0)
        };
        let s_c_left = if b > b_start && cur_blk_exp[(b - 1) - c_b_lo] != ZERO_EXP {
            ldexp_f64(1.0, cur_blk_exp[(b - 1) - c_b_lo] - e_ref)
        } else {
            0.0
        };

        let (mut hor_m, mut hor_i, mut hor_bx, mut hor_by) = if b > b_start && s_c_left != 0.0 {
            let l_off = (jb_lo - 1 - c_base_j) * NB;
            (
                cur_slice[l_off + BM] * s_c_left,
                cur_slice[l_off + BI] * s_c_left,
                cur_slice[l_off + BBX] * s_c_left,
                cur_slice[l_off + BBY] * s_c_left,
            )
        } else {
            (0.0, 0.0, 0.0, 0.0)
        };

        for j in jb_lo..=jb_hi {
            let use_left = j == jb_lo && need_left_block;
            let s_ep_diag = if use_left { s_ep_left } else { s_ep_same };
            let s_p_diag = if use_left { s_p_left } else { s_p_same };

            let cur_m = if j >= 1 {
                let mut acc = sx * (e_pre_m[j - 1] * s_ep_diag) * p.enter_m;
                if has_prev && (j - 1) >= p_lo && (j - 1) <= p_hi {
                    let pd = (j - 1 - p_base_j) * NB;
                    let pm = prev_slice[pd + BM];
                    let pdel = prev_slice[pd + BD];
                    let pi = if j > 1 { prev_slice[pd + BI] } else { 0.0 };
                    let pby = prev_slice[pd + BBY];
                    acc += s_p_diag
                        * (pm * p.m_to_m + (pdel + pi) * p.indel_exit + pby * p.by_exit);
                }
                if HAS_EMIT {
                    acc * emit_hom[e_prev_base + (j - 1 - e_prev_lo)]
                } else {
                    acc
                }
            } else {
                0.0
            };

            let mut cur_d = 0.0;
            let mut cur_bx = sx * (e_pre_m[j] * s_ep_same) * p.enter_bx;
            if has_prev && j >= p_lo && j <= p_hi {
                let ps = (j - p_base_j) * NB;
                let pm = prev_slice[ps + BM] * s_p_same;
                let pdel = prev_slice[ps + BD] * s_p_same;
                let pbx = prev_slice[ps + BBX] * s_p_same;
                cur_d = pm * p.m_to_d + pdel * p.indel_ext;
                cur_bx += pm * p.m_to_bx + pbx * p.bx_ext;
            }

            let cur_i = if i < lx && j > c_lo && j < ly {
                hor_m * p.m_to_i + hor_i * p.indel_ext
            } else {
                0.0
            };

            let cur_by = if j > c_lo {
                let raw_by = hor_bx * bx_to_by_w + hor_by * p.by_ext;
                if HAS_EMIT {
                    raw_by * emit_coarse[e_prev_base + (j - 1 - e_prev_lo)]
                } else {
                    raw_by
                }
            } else {
                0.0
            };

            let out_off = (j - c_base_j) * NB;
            cur_slice[out_off + BM] = cur_m;
            cur_slice[out_off + BD] = cur_d;
            cur_slice[out_off + BI] = cur_i;
            cur_slice[out_off + BBX] = cur_bx;
            cur_slice[out_off + BBY] = cur_by;

            hor_m = cur_m;
            hor_i = cur_i;
            hor_bx = cur_bx;
            hor_by = cur_by;
        }

        let blk_lo = (jb_lo - c_base_j) * NB;
        let blk_hi = (jb_hi - c_base_j + 1) * NB;
        let c_exp = normalize_block_slice(&mut cur_slice[blk_lo..blk_hi], e_ref);
        cur_blk_exp[b - c_b_lo] = c_exp;

        if c_exp != ZERO_EXP {
            if i == lx {
                for j in jb_lo..=jb_hi {
                    let off = (j - c_base_j) * NB;
                    let c_end =
                        cur_slice[off + BM] + cur_slice[off + BD] + cur_slice[off + BBY] * p.by_exit;
                    if c_end > 0.0 {
                        if COLLECT_ALL_LY {
                            add_scaled(&mut eof_m[j], &mut eof_e[j], c_end * p.coterm_eof, c_exp);
                        } else if j == ly {
                            add_scaled(&mut eof_m[0], &mut eof_e[0], c_end * p.coterm_eof, c_exp);
                        }
                        if j < ly {
                            add_scaled(
                                &mut post_y_m[j + 1],
                                &mut post_y_e[j + 1],
                                c_end * p.coterm_post,
                                c_exp,
                            );
                        }
                    }
                }
            } else {
                for j in jb_lo..=jb_hi {
                    let off = (j - c_base_j) * NB;
                    let cur_m = cur_slice[off + BM];
                    if cur_m > 0.0 {
                        let ex = cur_m * p.m_to_end;
                        if COLLECT_ALL_LY {
                            add_scaled(&mut eof_m[j], &mut eof_e[j], ex * p.early_eof, c_exp);
                        } else if j == ly {
                            add_scaled(&mut eof_m[0], &mut eof_e[0], ex * p.early_eof, c_exp);
                        }
                        if j < ly {
                            add_scaled(
                                &mut post_y_m[j + 1],
                                &mut post_y_e[j + 1],
                                ex * p.early_post,
                                c_exp,
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Exact 2D O(Lx * max_ly) path partition reference sweep used for small lengths and unit-test verification.
pub fn compute_ln_path_partition_vec_2d_exact(
    lx: usize,
    max_ly: usize,
    regime: TransducerRegime,
    prior_mean_len: f64,
) -> Vec<f64> {
    assert!(lx >= 1, "Transducer requires Lx >= 1");
    let mut eof_by_ly = vec![NEG_INF; max_ly + 1];
    if max_ly == 0 {
        return eof_by_ly;
    }

    let p = LinearTransducerParams::new(lx, regime, prior_mean_len);
    let n_blks = (max_ly >> BLK_SHIFT) + 1;
    let mut e_pre_m = Vec::with_capacity(max_ly + 1);
    let mut e_pre_blk_exp = Vec::with_capacity(n_blks);
    build_linear_e_pre(&p, max_ly, &mut e_pre_m, &mut e_pre_blk_exp);

    let row_len = (max_ly + 1) * NB;
    let mut prev_row = vec![0.0f64; row_len];
    let mut cur_row = vec![0.0f64; row_len];
    let mut prev_blk_exp = vec![ZERO_EXP; n_blks];
    let mut cur_blk_exp = vec![ZERO_EXP; n_blks];
    let mut eof_m = vec![0.0f64; max_ly + 1];
    let mut eof_e = vec![ZERO_EXP; max_ly + 1];
    let mut post_y_m = vec![0.0f64; max_ly + 1];
    let mut post_y_e = vec![ZERO_EXP; max_ly + 1];

    for i in 1..=lx {
        forward_row_step::<false, true>(
            i,
            lx,
            max_ly,
            &p,
            0,
            max_ly,
            &prev_row,
            0,
            &prev_blk_exp,
            0,
            0,
            max_ly,
            &mut cur_row,
            0,
            &mut cur_blk_exp,
            0,
            &e_pre_m,
            &e_pre_blk_exp,
            &[],
            &[],
            0,
            0,
            &mut eof_m,
            &mut eof_e,
            &mut post_y_m,
            &mut post_y_e,
        );
        std::mem::swap(&mut prev_row, &mut cur_row);
        std::mem::swap(&mut prev_blk_exp, &mut cur_blk_exp);
    }

    let mut tail_m = 0.0f64;
    let mut tail_e = ZERO_EXP;
    for j in 1..=max_ly {
        add_scaled(&mut tail_m, &mut tail_e, post_y_m[j], post_y_e[j]);
        if tail_m > 0.0 && tail_e != ZERO_EXP {
            add_scaled(&mut eof_m[j], &mut eof_e[j], tail_m * p.one_minus_eta_y, tail_e);
            if j < max_ly {
                tail_m *= p.eta_y;
                canon_scalar(&mut tail_m, &mut tail_e);
            }
        }
        eof_by_ly[j] = scaled_to_ln(eof_m[j], eof_e[j]);
    }
    eof_by_ly[0] = NEG_INF;
    eof_by_ly
}

#[inline]
fn convolve_e_pre_1pole(
    p: &LinearTransducerParams,
    imp_m: &[f64],
    imp_e: &[i32],
    max_ly: usize,
) -> (Vec<f64>, Vec<i32>) {
    let mut out_m = vec![0.0f64; max_ly + 1];
    let mut out_e = vec![ZERO_EXP; max_ly + 1];
    let mut acc_m = 0.0f64;
    let mut acc_e = ZERO_EXP;
    let enter_scale = p.pre_y_enter * p.one_minus_eta_y;

    for j in 0..=max_ly {
        out_m[j] = acc_m;
        out_e[j] = acc_e;
        if imp_m[j] > 0.0 && imp_e[j] != ZERO_EXP {
            add_scaled(
                &mut out_m[j],
                &mut out_e[j],
                imp_m[j] * p.pre_y_zero,
                imp_e[j],
            );
        }
        if j < max_ly {
            if acc_e != ZERO_EXP && acc_m > 0.0 {
                acc_m *= p.eta_y;
                canon_scalar(&mut acc_m, &mut acc_e);
            }
            if imp_m[j] > 0.0 && imp_e[j] != ZERO_EXP {
                add_scaled(&mut acc_m, &mut acc_e, imp_m[j] * enter_scale, imp_e[j]);
            }
        }
    }
    (out_m, out_e)
}

/// Exact log-space marginal path partition vector `eof_by_ly[0..=max_ly]`.
/// Uses shift-invariant single-impulse propagation over an adaptive convergence band
/// (`2^-96` relative cutoff) + exact O(max_ly) 1-pole geometric pre/post convolution.
pub fn compute_ln_path_partition_vec(
    lx: usize,
    max_ly: usize,
    regime: TransducerRegime,
    prior_mean_len: f64,
) -> Vec<f64> {
    assert!(lx >= 1, "Transducer requires Lx >= 1");
    if lx <= 64 || max_ly <= 64 {
        return compute_ln_path_partition_vec_2d_exact(lx, max_ly, regime, prior_mean_len);
    }

    let mut eof_by_ly = vec![NEG_INF; max_ly + 1];
    let p = LinearTransducerParams::new(lx, regime, prior_mean_len);
    let n_blks = (max_ly >> BLK_SHIFT) + 1;
    let last_blk = max_ly >> BLK_SHIFT;
    let row_len = (max_ly + 1) * NB;

    let mut prev_row = vec![0.0f64; row_len];
    let mut cur_row = vec![0.0f64; row_len];
    let mut prev_blk_exp = vec![ZERO_EXP; n_blks];
    let mut cur_blk_exp = vec![ZERO_EXP; n_blks];
    let mut imp_eof_m = vec![0.0f64; max_ly + 1];
    let mut imp_eof_e = vec![ZERO_EXP; max_ly + 1];
    let mut imp_post_m = vec![0.0f64; max_ly + 1];
    let mut imp_post_e = vec![ZERO_EXP; max_ly + 1];

    let by_coterm_scale = p.by_exit / p.bx_to_by;
    let mut p_lo = 0usize;
    let mut p_hi = 0usize;

    for s in 1..=lx {
        let has_prev = s >= 2;
        let pb_lo = p_lo >> BLK_SHIFT;
        let pb_hi = p_hi >> BLK_SHIFT;

        let (b_start, prev_last_b) = if !has_prev {
            (0usize, 0usize)
        } else {
            let mut peak_prev = ZERO_EXP;
            for b in pb_lo..=pb_hi {
                let pe = prev_blk_exp[b];
                if pe != ZERO_EXP && pe > peak_prev {
                    peak_prev = pe;
                }
            }
            if peak_prev == ZERO_EXP {
                break;
            }
            let cutoff = peak_prev - 96;
            let mut first_b = pb_hi;
            let mut last_b = pb_lo;
            let mut found = false;
            for b in pb_lo..=pb_hi {
                let pe = prev_blk_exp[b];
                if pe != ZERO_EXP && pe >= cutoff {
                    if !found {
                        first_b = b;
                        found = true;
                    }
                    last_b = b;
                }
            }
            if !found {
                break;
            }
            (first_b, last_b)
        };

        let c_lo = b_start << BLK_SHIFT;
        let w_coterm = if s == lx { p.sx_0 } else { p.sx_rest };
        let w_early = if s < lx {
            p.sx_0 + ((lx - 1 - s) as f64) * p.sx_rest
        } else {
            0.0
        };

        let mut row_peak_exp = ZERO_EXP;
        let mut hor_m = 0.0f64;
        let mut hor_i = 0.0f64;
        let mut hor_bx = 0.0f64;
        let mut hor_by = 0.0f64;
        let mut b_end = b_start;

        for b in b_start..=last_blk {
            b_end = b;
            let jb_lo = (b << BLK_SHIFT).max(c_lo);
            let jb_hi = ((b << BLK_SHIFT) | BLK_MASK).min(max_ly);
            let need_left_block = (jb_lo & BLK_MASK) == 0 && jb_lo > 0;

            let mut e_ref = ZERO_EXP;
            if !has_prev && b == 0 {
                e_ref = 0;
            }
            if has_prev {
                if b >= pb_lo && b <= pb_hi {
                    let pe = prev_blk_exp[b];
                    if pe != ZERO_EXP && pe > e_ref {
                        e_ref = pe;
                    }
                }
                if need_left_block && (b - 1) >= pb_lo && (b - 1) <= pb_hi {
                    let pe = prev_blk_exp[b - 1];
                    if pe != ZERO_EXP && pe > e_ref {
                        e_ref = pe;
                    }
                }
            }
            if b > b_start {
                let ce = cur_blk_exp[b - 1];
                if ce != ZERO_EXP && ce > e_ref {
                    e_ref = ce;
                }
            }

            if e_ref == ZERO_EXP {
                cur_blk_exp[b] = ZERO_EXP;
                cur_row[jb_lo * NB..(jb_hi + 1) * NB].fill(0.0);
                hor_m = 0.0;
                hor_i = 0.0;
                hor_bx = 0.0;
                hor_by = 0.0;
                if has_prev && b > prev_last_b {
                    break;
                }
                continue;
            }

            let (s_p_same, s_p_left) = if has_prev {
                let ss = if b >= pb_lo && b <= pb_hi && prev_blk_exp[b] != ZERO_EXP {
                    ldexp_f64(1.0, prev_blk_exp[b] - e_ref)
                } else {
                    0.0
                };
                let sl = if need_left_block
                    && (b - 1) >= pb_lo
                    && (b - 1) <= pb_hi
                    && prev_blk_exp[b - 1] != ZERO_EXP
                {
                    ldexp_f64(1.0, prev_blk_exp[b - 1] - e_ref)
                } else {
                    0.0
                };
                (ss, sl)
            } else {
                (0.0, 0.0)
            };

            if b > b_start {
                let s_c_left = if cur_blk_exp[b - 1] != ZERO_EXP {
                    ldexp_f64(1.0, cur_blk_exp[b - 1] - e_ref)
                } else {
                    0.0
                };
                if s_c_left != 0.0 {
                    hor_m *= s_c_left;
                    hor_i *= s_c_left;
                    hor_bx *= s_c_left;
                    hor_by *= s_c_left;
                } else {
                    hor_m = 0.0;
                    hor_i = 0.0;
                    hor_bx = 0.0;
                    hor_by = 0.0;
                }
            }

            let s_ep_impulse = if !has_prev && b == 0 {
                ldexp_f64(1.0, -e_ref)
            } else {
                0.0
            };

            for j in jb_lo..=jb_hi {
                let s_p_diag = if j == jb_lo && need_left_block {
                    s_p_left
                } else {
                    s_p_same
                };

                let cur_m = if j >= 1 {
                    let mut acc = if !has_prev && j == 1 {
                        s_ep_impulse * p.enter_m
                    } else {
                        0.0
                    };
                    if has_prev && (j - 1) >= p_lo && (j - 1) <= p_hi {
                        let pd = (j - 1) * NB;
                        let pm = prev_row[pd + BM];
                        let pdel = prev_row[pd + BD];
                        let pi = if j > 1 { prev_row[pd + BI] } else { 0.0 };
                        let pby = prev_row[pd + BBY];
                        acc += s_p_diag
                            * (pm * p.m_to_m + (pdel + pi) * p.indel_exit + pby * p.by_exit);
                    }
                    acc
                } else {
                    0.0
                };

                let mut cur_d = 0.0;
                let mut cur_bx = if !has_prev && j == 0 {
                    s_ep_impulse * p.enter_bx
                } else {
                    0.0
                };
                if has_prev && j >= p_lo && j <= p_hi {
                    let ps = j * NB;
                    let pm = prev_row[ps + BM] * s_p_same;
                    let pdel = prev_row[ps + BD] * s_p_same;
                    let pbx = prev_row[ps + BBX] * s_p_same;
                    cur_d = pm * p.m_to_d + pdel * p.indel_ext;
                    cur_bx += pm * p.m_to_bx + pbx * p.bx_ext;
                }

                let cur_i = if s < lx && j > c_lo && j < max_ly {
                    hor_m * p.m_to_i + hor_i * p.indel_ext
                } else {
                    0.0
                };

                let cur_by = if j > c_lo {
                    hor_bx * p.bx_to_by + hor_by * p.by_ext
                } else {
                    0.0
                };

                let out_off = j * NB;
                cur_row[out_off + BM] = cur_m;
                cur_row[out_off + BD] = cur_d;
                cur_row[out_off + BI] = cur_i;
                cur_row[out_off + BBX] = cur_bx;
                cur_row[out_off + BBY] = cur_by;

                hor_m = cur_m;
                hor_i = cur_i;
                hor_bx = cur_bx;
                hor_by = cur_by;
            }

            let c_exp = normalize_block_slice(&mut cur_row[jb_lo * NB..(jb_hi + 1) * NB], e_ref);
            cur_blk_exp[b] = c_exp;

            if c_exp != ZERO_EXP {
                if row_peak_exp == ZERO_EXP || c_exp > row_peak_exp {
                    row_peak_exp = c_exp;
                }
                for j in jb_lo..=jb_hi {
                    let off = j * NB;
                    let cur_m = cur_row[off + BM];
                    let cur_d = cur_row[off + BD];
                    let cur_by = cur_row[off + BBY];
                    let c_end_coterm = w_coterm * (cur_m + cur_d + cur_by * by_coterm_scale);
                    let ex_early = if w_early > 0.0 {
                        w_early * cur_m * p.m_to_end
                    } else {
                        0.0
                    };
                    let eof_add = c_end_coterm * p.coterm_eof + ex_early * p.early_eof;
                    if eof_add > 0.0 {
                        add_scaled(&mut imp_eof_m[j], &mut imp_eof_e[j], eof_add, c_exp);
                    }
                    if j < max_ly {
                        let post_add = c_end_coterm * p.coterm_post + ex_early * p.early_post;
                        if post_add > 0.0 {
                            add_scaled(
                                &mut imp_post_m[j + 1],
                                &mut imp_post_e[j + 1],
                                post_add,
                                c_exp,
                            );
                        }
                    }
                }
            }

            if (has_prev && b > prev_last_b + 2) || (!has_prev && b > 2) {
                if c_exp == ZERO_EXP || (row_peak_exp != ZERO_EXP && c_exp < row_peak_exp - 96) {
                    break;
                }
            }
        }

        let mut last_nz_b = b_start;
        for b in (b_start..=b_end).rev() {
            if cur_blk_exp[b] != ZERO_EXP {
                last_nz_b = b;
                break;
            }
        }
        p_lo = c_lo;
        p_hi = ((last_nz_b << BLK_SHIFT) | BLK_MASK).min(max_ly);
        std::mem::swap(&mut prev_row, &mut cur_row);
        std::mem::swap(&mut prev_blk_exp, &mut cur_blk_exp);
    }

    let (mut eof_m, mut eof_e) = convolve_e_pre_1pole(&p, &imp_eof_m, &imp_eof_e, max_ly);
    let (post_y_m, post_y_e) = convolve_e_pre_1pole(&p, &imp_post_m, &imp_post_e, max_ly);

    let mut tail_m = 0.0f64;
    let mut tail_e = ZERO_EXP;
    for j in 1..=max_ly {
        add_scaled(&mut tail_m, &mut tail_e, post_y_m[j], post_y_e[j]);
        if tail_m > 0.0 && tail_e != ZERO_EXP {
            add_scaled(&mut eof_m[j], &mut eof_e[j], tail_m * p.one_minus_eta_y, tail_e);
            if j < max_ly {
                tail_m *= p.eta_y;
                canon_scalar(&mut tail_m, &mut tail_e);
            }
        }
        eof_by_ly[j] = scaled_to_ln(eof_m[j], eof_e[j]);
    }
    eof_by_ly[0] = NEG_INF;
    eof_by_ly
}

const PATH_PART_SHARDS_COUNT: usize = 64;
static PATH_PART_SHARDS: OnceLock<Vec<RwLock<HashMap<(usize, u64, u64), Vec<f64>>>>> =
    OnceLock::new();
static COHORT_MAX_LY_HINT: AtomicUsize = AtomicUsize::new(0);

#[inline]
fn path_part_shard_idx(lx: usize, t_bits: u64, pml_bits: u64) -> usize {
    let h = (lx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ t_bits.rotate_left(17)
        ^ pml_bits.rotate_left(31);
    ((h ^ (h >> 32)) as usize) & (PATH_PART_SHARDS_COUNT - 1)
}

#[inline]
fn get_path_part_shards() -> &'static [RwLock<HashMap<(usize, u64, u64), Vec<f64>>>] {
    PATH_PART_SHARDS
        .get_or_init(|| {
            (0..PATH_PART_SHARDS_COUNT)
                .map(|_| RwLock::new(HashMap::new()))
                .collect()
        })
        .as_slice()
}

/// Exact log-space path partition function \ln P_len(L_Y | L_X; regime) = \ln \sum_{Y \in \Sigma^{L_Y}} P_T(Y | X; regime).
/// Sharded 64-way RwLock memoization per (Lx, t_primary, prior_mean_len) computed lazily on demand.
pub fn compute_ln_path_partition(
    lx: usize,
    ly: usize,
    regime: TransducerRegime,
    prior_mean_len: f64,
) -> f64 {
    assert!(lx >= 1, "Transducer requires Lx >= 1");
    let t_bits = regime.t_primary.to_bits();
    let pml_bits = prior_mean_len.to_bits();
    let key = (lx, t_bits, pml_bits);
    let cache = &get_path_part_shards()[path_part_shard_idx(lx, t_bits, pml_bits)];
    let prev_len = {
        let guard = cache.read().unwrap();
        if let Some(vec) = guard.get(&key) {
            if ly < vec.len() {
                return vec[ly];
            }
            vec.len()
        } else {
            0
        }
    };
    let hint = COHORT_MAX_LY_HINT.load(Ordering::Relaxed);
    let target_max = (ly + 64).max(hint + 16).max(prev_len * 2);
    let computed = compute_ln_path_partition_vec(lx, target_max, regime, prior_mean_len);
    let ans = computed[ly];
    {
        let mut guard = cache.write().unwrap();
        let replace = guard.get(&key).map_or(true, |v| v.len() <= ly);
        if replace {
            guard.insert(key, computed);
        } else {
            return guard[&key][ly];
        }
    }
    ans
}

/// Record the cohort maximum sequence length hint and prewarm path partitions in parallel for long cohorts.
pub fn prewarm_path_partitions_for_cohort(
    seq_lens: &[usize],
    regimes: &[TransducerRegime],
    prior_mean_len: f64,
) {
    use rayon::prelude::*;

    if let Some(&max_ly) = seq_lens.iter().max() {
        COHORT_MAX_LY_HINT.fetch_max(max_ly, Ordering::Relaxed);
        if max_ly > 2_000 && !regimes.is_empty() {
            let mut uniq_lx: Vec<usize> = seq_lens.iter().copied().filter(|&l| l >= 1).collect();
            uniq_lx.sort_unstable();
            uniq_lx.dedup();

            let mut uniq_regs: Vec<TransducerRegime> = regimes.to_vec();
            uniq_regs.sort_by_key(|r| r.t_primary.to_bits());
            uniq_regs.dedup_by_key(|r| r.t_primary.to_bits());

            let mut jobs = Vec::with_capacity(uniq_lx.len() * uniq_regs.len());
            for &lx in &uniq_lx {
                for &reg in &uniq_regs {
                    jobs.push((lx, reg));
                }
            }
            jobs.into_par_iter().for_each(|(lx, reg)| {
                let _ = compute_ln_path_partition(lx, max_ly, reg, prior_mean_len);
            });
        }
    }
}

pub const Q_STEPS: usize = 64;

/// Cached per-mode Copula tables and lazy quantile exponentiation locks shared safely across all worker threads.
pub struct GlobalRegimeTables {
    pub mode: SeqMode,
    pub k_alpha: usize,
    pub regimes: Vec<TransducerRegime>,
    pub copulas: Vec<Vec<f64>>,
    pub ln_r_bases: Vec<Vec<f64>>,
    pub copulas_coarse: Vec<Vec<f64>>,
    r_pow_locks: Vec<OnceLock<Vec<f64>>>,
}

impl GlobalRegimeTables {
    fn build(mode: SeqMode, gc: &GeneticCode) -> Self {
        let bg_pi = default_background_pi(mode, gc);
        let k_alpha = bg_pi.len();
        let regimes = candidate_regimes(mode);
        let n_reg = regimes.len();
        let mut copulas = Vec::with_capacity(n_reg);
        let mut ln_r_bases = Vec::with_capacity(n_reg);
        let mut copulas_coarse = Vec::with_capacity(n_reg);
        for &reg in &regimes {
            let cop = build_copula_ratio_matrix(mode, reg, &bg_pi, gc);
            let ln_r: Vec<f64> = cop.iter().map(|&v| v.max(1e-300).ln()).collect();
            let cop_c =
                build_copula_ratio_matrix(mode, coarse_regime_for(mode, reg), &bg_pi, gc);
            copulas.push(cop);
            ln_r_bases.push(ln_r);
            copulas_coarse.push(cop_c);
        }
        let total_locks = n_reg * (Q_STEPS + 1);
        let mut r_pow_locks = Vec::with_capacity(total_locks);
        for _ in 0..total_locks {
            r_pow_locks.push(OnceLock::new());
        }
        Self {
            mode,
            k_alpha,
            regimes,
            copulas,
            ln_r_bases,
            copulas_coarse,
            r_pow_locks,
        }
    }

    #[inline]
    pub fn get_r_pow(&self, reg_idx: usize, q: usize) -> &[f64] {
        if q == 0 {
            return &self.copulas[reg_idx];
        }
        let slot = &self.r_pow_locks[reg_idx * (Q_STEPS + 1) + q];
        slot.get_or_init(|| {
            let pwr = 1.0 - 0.78 * (q as f64 / (Q_STEPS as f64));
            self.ln_r_bases[reg_idx]
                .iter()
                .map(|&lr| (pwr * lr).exp())
                .collect()
        })
        .as_slice()
    }
}

static REGIME_TABLES_CACHE: OnceLock<RwLock<HashMap<(u8, bool), Arc<GlobalRegimeTables>>>> =
    OnceLock::new();

pub fn get_global_regime_tables(mode: SeqMode, gc: &GeneticCode) -> Arc<GlobalRegimeTables> {
    let key = (
        match mode {
            SeqMode::Codon => 0u8,
            SeqMode::Protein => 1u8,
            SeqMode::Dna => 2u8,
        },
        gc.mito_tga_trp,
    );
    let cache = REGIME_TABLES_CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    {
        let guard = cache.read().unwrap();
        if let Some(t) = guard.get(&key) {
            return Arc::clone(t);
        }
    }
    let built = Arc::new(GlobalRegimeTables::build(mode, gc));
    let mut guard = cache.write().unwrap();
    if let Some(existing) = guard.get(&key) {
        return Arc::clone(existing);
    }
    guard.insert(key, Arc::clone(&built));
    built
}

/// Reduced 10-class amino-acid equivalence groups for spaced homology seeding:
/// `{LVIM=0, FYW=1, AG=2, ST=3, KR=4, ED=5, NQ=6, P=7, C=8, H=9}`.
const AA_REDUCED_10: [u8; 20] = [
    2, // A
    8, // C
    5, // D
    5, // E
    1, // F
    2, // G
    9, // H
    0, // I
    4, // K
    0, // L
    0, // M
    6, // N
    7, // P
    6, // Q
    4, // R
    3, // S
    3, // T
    0, // V
    1, // W
    1, // Y
];

/// Precomputed per-sequence null profile, quantized causal repeat tiers, AA projection,
/// sorted non-repetitive homology seeds, and lazy per-regime coarse tilts & q==0 normalizers.
pub struct PreparedSeq {
    pub seq: EncodedSeq,
    pub prof: SingleSeqNullProfile,
    pub l0_nats: f64,
    pub qx: Vec<u8>,
    pub qy: Vec<u8>,
    pub aa_proj: Vec<u8>,
    /// Sorted by `(key, pos)`: `(seed_key, start_pos, span, weight)`.
    pub seeds: Vec<(u32, u32, u8, f32)>,
    local_coarse_locks: Vec<OnceLock<Vec<f64>>>,
    inv_z_q0_locks: Vec<OnceLock<Vec<f64>>>,
}

#[inline]
fn quantize_repeats(weights: &[f64]) -> Vec<u8> {
    weights
        .iter()
        .map(|&r| {
            if r <= 0.05 {
                0u8
            } else {
                ((r * (Q_STEPS as f64)).round() as u8).clamp(1, Q_STEPS as u8)
            }
        })
        .collect()
}

fn extract_filtered_seeds(
    seq: &EncodedSeq,
    aa_proj: &[u8],
    prof: &SingleSeqNullProfile,
) -> Vec<(u32, u32, u8, f32)> {
    let l = seq.len();
    let mut raw: Vec<(u32, u32, u8, f32)> = Vec::new();
    let rep = &prof.ref_repeat_weight;

    match seq.mode {
        SeqMode::Protein | SeqMode::Codon => {
            if l >= 4 {
                raw.reserve(2 * l);
                for p in 0..=(l - 4) {
                    if aa_proj[p] >= 20
                        || aa_proj[p + 1] >= 20
                        || aa_proj[p + 2] >= 20
                        || aa_proj[p + 3] >= 20
                        || rep[p] > 0.55
                        || rep[p + 1] > 0.55
                        || rep[p + 2] > 0.55
                        || rep[p + 3] > 0.55
                    {
                        continue;
                    }
                    let key = (0u32 << 28)
                        | ((aa_proj[p] as u32) << 15)
                        | ((aa_proj[p + 1] as u32) << 10)
                        | ((aa_proj[p + 2] as u32) << 5)
                        | (aa_proj[p + 3] as u32);
                    raw.push((key, p as u32, 4, 4.0));
                }
            }
            if l >= 6 {
                for p in 0..=(l - 6) {
                    if aa_proj[p] >= 20
                        || aa_proj[p + 1] >= 20
                        || aa_proj[p + 2] >= 20
                        || aa_proj[p + 4] >= 20
                        || aa_proj[p + 5] >= 20
                        || rep[p] > 0.55
                        || rep[p + 1] > 0.55
                        || rep[p + 2] > 0.55
                        || rep[p + 4] > 0.55
                        || rep[p + 5] > 0.55
                    {
                        continue;
                    }
                    let g0 = AA_REDUCED_10[aa_proj[p] as usize] as u32;
                    let g1 = AA_REDUCED_10[aa_proj[p + 1] as usize] as u32;
                    let g2 = AA_REDUCED_10[aa_proj[p + 2] as usize] as u32;
                    let g3 = AA_REDUCED_10[aa_proj[p + 4] as usize] as u32;
                    let g4 = AA_REDUCED_10[aa_proj[p + 5] as usize] as u32;
                    let key = (1u32 << 28)
                        | (g0 << 16)
                        | (g1 << 12)
                        | (g2 << 8)
                        | (g3 << 4)
                        | g4;
                    raw.push((key, p as u32, 6, 2.5));
                }
            }
        }
        SeqMode::Dna => {
            if l >= 9 {
                raw.reserve(2 * l);
                for p in 0..=(l - 9) {
                    let mut is_rep = false;
                    for off in 0..9 {
                        if aa_proj[p + off] >= 4 || rep[p + off] > 0.55 {
                            is_rep = true;
                            break;
                        }
                    }
                    if is_rep {
                        continue;
                    }
                    let mut k_bits = 0u32;
                    for off in 0..9 {
                        k_bits = (k_bits << 2) | (seq.tokens[p + off] as u32);
                    }
                    raw.push(((0u32 << 28) | k_bits, p as u32, 9, 4.0));
                }
            }
            if l >= 13 {
                // Codon-wobble-invariant 9-of-13 spaced seed: 1101101101101
                const OFFSETS: [usize; 9] = [0, 1, 3, 4, 6, 7, 9, 10, 12];
                for p in 0..=(l - 13) {
                    let mut is_rep = false;
                    for &off in &OFFSETS {
                        if aa_proj[p + off] >= 4 || rep[p + off] > 0.55 {
                            is_rep = true;
                            break;
                        }
                    }
                    if is_rep {
                        continue;
                    }
                    let mut k_bits = 0u32;
                    for &off in &OFFSETS {
                        k_bits = (k_bits << 2) | (seq.tokens[p + off] as u32);
                    }
                    raw.push(((1u32 << 28) | k_bits, p as u32, 13, 2.8));
                }
            }
        }
    }

    raw.sort_unstable_by_key(|&(k, pos, _, _)| (k, pos));

    // Discard any key appearing > 4 times within this single sequence
    let mut filtered = Vec::with_capacity(raw.len());
    let mut idx = 0usize;
    while idx < raw.len() {
        let key = raw[idx].0;
        let mut end = idx + 1;
        while end < raw.len() && raw[end].0 == key {
            end += 1;
        }
        if end - idx <= 4 {
            filtered.extend_from_slice(&raw[idx..end]);
        }
        idx = end;
    }
    filtered
}

impl PreparedSeq {
    pub fn new(seq: EncodedSeq, gc: &GeneticCode) -> Self {
        let bg_pi = default_background_pi(seq.mode, gc);
        let prof = SingleSeqNullProfile::compute_with_masks(
            &seq.tokens,
            &seq.ambig_masks,
            seq.alphabet_size(),
            &bg_pi,
        );
        let l0_nats = geometric_len_nats(seq.len()) + prof.payload_nats;
        let qx = quantize_repeats(&prof.ref_repeat_weight);
        let qy = quantize_repeats(&prof.site_repeat_weight);
        let aa_proj = seq.sym_proj.clone();
        let seeds = extract_filtered_seeds(&seq, &aa_proj, &prof);
        let n_reg = candidate_regimes(seq.mode).len();
        let mut local_coarse_locks = Vec::with_capacity(n_reg);
        let mut inv_z_q0_locks = Vec::with_capacity(n_reg);
        for _ in 0..n_reg {
            local_coarse_locks.push(OnceLock::new());
            inv_z_q0_locks.push(OnceLock::new());
        }
        Self {
            seq,
            prof,
            l0_nats,
            qx,
            qy,
            aa_proj,
            seeds,
            local_coarse_locks,
            inv_z_q0_locks,
        }
    }

    #[inline]
    pub fn get_local_coarse(&self, reg_idx: usize, tables: &GlobalRegimeTables) -> &[f64] {
        self.local_coarse_locks[reg_idx]
            .get_or_init(|| {
                precompute_local_coarse_tilts(&self.seq, &tables.copulas_coarse[reg_idx], 8)
            })
            .as_slice()
    }

    #[inline]
    pub fn get_inv_z_q0(&self, reg_idx: usize, tables: &GlobalRegimeTables) -> &[f64] {
        self.inv_z_q0_locks[reg_idx]
            .get_or_init(|| {
                let ly = self.seq.len();
                let k_alpha = self.seq.alphabet_size();
                let cop = &tables.copulas[reg_idx];
                let m0 = &self.prof.site_m0_dist;
                let mut out = vec![0.0f64; ly * k_alpha];
                for j in 0..ly {
                    let m0_row = &m0[j * k_alpha..(j + 1) * k_alpha];
                    let out_row = &mut out[j * k_alpha..(j + 1) * k_alpha];
                    for xi in 0..k_alpha {
                        let r_row = &cop[xi * k_alpha..(xi + 1) * k_alpha];
                        out_row[xi] = 1.0 / dot_k_unrolled(m0_row, r_row).max(1e-300);
                    }
                }
                out
            })
            .as_slice()
    }

    pub fn clear_dp_caches(&mut self) {
        self.prof.site_m0_dist = Vec::new();
        self.prof.site_ln_m0 = Vec::new();
        self.prof.ref_repeat_weight = Vec::new();
        self.prof.site_repeat_weight = Vec::new();
        self.qx = Vec::new();
        self.qy = Vec::new();
        self.seeds = Vec::new();
        self.local_coarse_locks = Vec::new();
        self.inv_z_q0_locks = Vec::new();
    }

    pub fn reset_regime_locks(&mut self) {
        for lock in &mut self.local_coarse_locks {
            let _ = lock.take();
        }
        for lock in &mut self.inv_z_q0_locks {
            let _ = lock.take();
        }
    }
}

/// Sparse monotonic transposable corridor in DP coordinates `(i, j) in 0..=Lx x 0..=Ly`.
#[derive(Debug, Clone)]
pub struct CorridorBounds {
    pub lx: usize,
    pub ly: usize,
    pub j_lo: Vec<usize>,
    pub j_hi: Vec<usize>,
    pub row_offset: Vec<usize>,
    pub emit_lo: Vec<usize>,
    pub emit_hi: Vec<usize>,
    pub emit_offset: Vec<usize>,
    pub is_full_grid: bool,
    pub is_anchored: bool,
    pub anchor_count: usize,
    pub est_aa_div: f64,
    pub est_syn_div: f64,
    pub est_f_shared: f64,
}

impl CorridorBounds {
    pub fn from_bounds(
        lx: usize,
        ly: usize,
        mut j_lo: Vec<usize>,
        mut j_hi: Vec<usize>,
        is_full_grid: bool,
        is_anchored: bool,
        anchor_count: usize,
        est_aa_div: f64,
        est_syn_div: f64,
        est_f_shared: f64,
    ) -> Self {
        assert_eq!(j_lo.len(), lx + 1);
        assert_eq!(j_hi.len(), lx + 1);

        for i in 0..=lx {
            j_lo[i] = j_lo[i].min(ly);
            j_hi[i] = j_hi[i].min(ly);
            if j_lo[i] > j_hi[i] {
                j_lo[i] = j_hi[i];
            }
        }
        j_lo[0] = 0;
        j_hi[lx] = ly;

        for i in (0..lx).rev() {
            j_lo[i] = j_lo[i].min(j_lo[i + 1]);
        }
        for i in 1..=lx {
            j_hi[i] = j_hi[i].max(j_hi[i - 1]);
        }
        for i in 0..lx {
            if j_lo[i + 1] > j_hi[i] + 1 {
                j_hi[i] = j_lo[i + 1] - 1;
            }
        }
        for i in 1..=lx {
            j_hi[i] = j_hi[i].max(j_hi[i - 1]);
        }

        let mut row_offset = Vec::with_capacity(lx + 2);
        let mut acc = 0usize;
        row_offset.push(0);
        for i in 0..=lx {
            acc += j_hi[i] - j_lo[i] + 1;
            row_offset.push(acc);
        }

        let mut emit_lo = Vec::with_capacity(lx);
        let mut emit_hi = Vec::with_capacity(lx);
        let mut emit_offset = Vec::with_capacity(lx + 1);
        let mut e_acc = 0usize;
        emit_offset.push(0);
        for r in 0..lx {
            let (elo, ehi) = if j_lo[r + 1].max(1) <= j_hi[r + 1] {
                (j_lo[r + 1].max(1) - 1, j_hi[r + 1])
            } else {
                (0, 0)
            };
            emit_lo.push(elo);
            emit_hi.push(ehi);
            e_acc += ehi - elo;
            emit_offset.push(e_acc);
        }

        let full_cells = (lx + 1).saturating_mul(ly + 1);
        let is_full = is_full_grid || acc == full_cells;

        Self {
            lx,
            ly,
            j_lo,
            j_hi,
            row_offset,
            emit_lo,
            emit_hi,
            emit_offset,
            is_full_grid: is_full,
            is_anchored,
            anchor_count,
            est_aa_div,
            est_syn_div,
            est_f_shared,
        }
    }

    pub fn full(lx: usize, ly: usize) -> Self {
        Self::from_bounds(
            lx,
            ly,
            vec![0; lx + 1],
            vec![ly; lx + 1],
            true,
            false,
            0,
            0.30,
            1.20,
            0.50,
        )
    }

    pub fn diagonal_band(lx: usize, ly: usize, half_width: usize) -> Self {
        let mut j_lo = vec![0usize; lx + 1];
        let mut j_hi = vec![ly; lx + 1];
        if lx > 0 {
            for i in 0..=lx {
                let center = (i * ly + lx / 2) / lx;
                j_lo[i] = center.saturating_sub(half_width);
                j_hi[i] = (center + half_width).min(ly);
            }
        }
        Self::from_bounds(lx, ly, j_lo, j_hi, false, false, 0, 0.30, 1.20, 0.50)
    }

    /// Exact cell-for-cell transpose `(j, i) \in C^T <==> (i, j) \in C` in O(Lx + Ly) time.
    pub fn transpose(&self) -> Self {
        let new_lx = self.ly;
        let new_ly = self.lx;
        let mut new_lo = vec![0usize; new_lx + 1];
        let mut new_hi = vec![new_ly; new_lx + 1];

        let mut i_ptr = 0usize;
        for j in 0..=self.ly {
            while i_ptr < self.lx && self.j_hi[i_ptr] < j {
                i_ptr += 1;
            }
            new_lo[j] = i_ptr;
        }

        let mut i_hi_ptr = 0usize;
        for j in 0..=self.ly {
            while i_hi_ptr < self.lx && self.j_lo[i_hi_ptr + 1] <= j {
                i_hi_ptr += 1;
            }
            new_hi[j] = i_hi_ptr;
        }

        Self::from_bounds(
            new_lx,
            new_ly,
            new_lo,
            new_hi,
            self.is_full_grid,
            self.is_anchored,
            self.anchor_count,
            self.est_aa_div,
            self.est_syn_div,
            self.est_f_shared,
        )
    }

    #[inline]
    pub fn in_row(&self, i: usize, j: usize) -> bool {
        j >= self.j_lo[i] && j <= self.j_hi[i]
    }

    #[inline]
    pub fn cell_base(&self, i: usize, j: usize) -> usize {
        debug_assert!(self.in_row(i, j));
        self.row_offset[i] + (j - self.j_lo[i])
    }

    #[inline]
    pub fn emit_idx(&self, r: usize, j: usize) -> usize {
        debug_assert!(j >= self.emit_lo[r] && j < self.emit_hi[r]);
        self.emit_offset[r] + (j - self.emit_lo[r])
    }
}

#[inline]
fn unanchored_corridor(lx: usize, ly: usize) -> CorridorBounds {
    if lx.saturating_mul(ly) > 2_500_000 {
        CorridorBounds::diagonal_band(lx, ly, 384)
    } else {
        CorridorBounds::full(lx, ly)
    }
}

/// Construct an adaptive colinear seed-anchor corridor in O(Lx + Ly + M log M) time.
pub fn build_anchor_corridor(
    px: &PreparedSeq,
    py: &PreparedSeq,
    _gc: &GeneticCode,
) -> CorridorBounds {
    let lx = px.seq.len();
    let ly = py.seq.len();
    if lx == 0 || ly == 0 || lx.max(ly) <= 96 {
        return CorridorBounds::full(lx, ly);
    }

    // Two-pointer merge join over pre-sorted seed lists
    let sx = &px.seeds;
    let sy = &py.seeds;
    let mut hits: Vec<(usize, usize, u8, f32)> = Vec::new();
    let mut ix_p = 0usize;
    let mut iy_p = 0usize;
    while ix_p < sx.len() && iy_p < sy.len() {
        let kx = sx[ix_p].0;
        let ky = sy[iy_p].0;
        if kx < ky {
            ix_p += 1;
        } else if ky < kx {
            iy_p += 1;
        } else {
            let mut ex = ix_p + 1;
            while ex < sx.len() && sx[ex].0 == kx {
                ex += 1;
            }
            let mut ey = iy_p + 1;
            while ey < sy.len() && sy[ey].0 == ky {
                ey += 1;
            }
            for a in ix_p..ex {
                for b in iy_p..ey {
                    hits.push((sx[a].1 as usize, sy[b].1 as usize, sx[a].2, sx[a].3));
                }
            }
            ix_p = ex;
            iy_p = ey;
        }
    }

    if hits.len() < 3 {
        return unanchored_corridor(lx, ly);
    }

    hits.sort_unstable_by_key(|&(ix, iy, _, _)| (ix, iy));
    // Deduplicate identical (ix, iy), keeping highest weight/span
    let mut dedup: Vec<(usize, usize, u8, f32)> = Vec::with_capacity(hits.len());
    for h in hits {
        if let Some(last) = dedup.last_mut() {
            if last.0 == h.0 && last.1 == h.1 {
                if h.3 > last.3 {
                    *last = h;
                }
                continue;
            }
        }
        dedup.push(h);
    }

    let m = dedup.len();
    let mut dp = vec![0.0f64; m];
    let mut prev = vec![usize::MAX; m];
    let mut best_end = 0usize;
    let mut best_score = 0.0f64;
    const LOOKBACK: usize = 256;
    let mut leaders: Vec<(isize, usize)> = Vec::with_capacity(64);

    for b in 0..m {
        let (ix_b, iy_b, sp_b, w_b) = dedup[b];
        let mut best_val = w_b as f64;
        let mut best_prev = usize::MAX;
        let start = b.saturating_sub(LOOKBACK);
        let eval_pred = |a: usize, best_v: &mut f64, best_p: &mut usize| {
            let (ix_a, iy_a, sp_a, _) = dedup[a];
            if ix_a < ix_b && iy_a < iy_b {
                let di = ix_b - ix_a;
                let dj = iy_b - iy_a;
                let cand = if di == dj && di <= sp_a as usize {
                    dp[a] + (di as f64) * ((w_b as f64) / (sp_b as f64))
                } else {
                    let drift = (di as isize - dj as isize).unsigned_abs() as f64;
                    dp[a] + (w_b as f64) - 0.08 * drift - 0.001 * ((di + dj) as f64)
                };
                if cand > *best_v {
                    *best_v = cand;
                    *best_p = a;
                }
            }
        };

        for a in start..b {
            eval_pred(a, &mut best_val, &mut best_prev);
        }
        for &(_, a) in &leaders {
            if a < start {
                eval_pred(a, &mut best_val, &mut best_prev);
            }
        }
        dp[b] = best_val;
        prev[b] = best_prev;
        if best_val > best_score {
            best_score = best_val;
            best_end = b;
        }

        let db = ((iy_b as isize) - (ix_b as isize)) >> 6;
        if let Some(slot) = leaders.iter_mut().find(|x| x.0 == db) {
            if best_val > dp[slot.1] {
                slot.1 = b;
            }
        } else if leaders.len() < 64 {
            leaders.push((db, b));
        } else if let Some((min_pos, &(_, min_idx))) = leaders
            .iter()
            .enumerate()
            .min_by(|a, b| dp[a.1 .1].total_cmp(&dp[b.1 .1]))
        {
            if best_val > dp[min_idx] {
                leaders[min_pos] = (db, b);
            }
        }
    }

    let area = lx.saturating_mul(ly);
    let min_chain_score = if px.seq.mode == SeqMode::Dna && area > 4_000_000 {
        14.0 + 3.5 * ((area as f64) / 1_000_000.0).ln()
    } else {
        14.0
    };
    if best_score < min_chain_score {
        return unanchored_corridor(lx, ly);
    }

    let mut chain = Vec::new();
    let mut cur = best_end;
    while cur != usize::MAX {
        chain.push(dedup[cur]);
        cur = prev[cur];
    }
    chain.reverse();

    // Expand chained seed hits into strictly monotonic residue anchor positions
    let mut anchors: Vec<(usize, usize)> = Vec::new();
    for idx in 0..chain.len() {
        let (ix, iy, span, _) = chain[idx];
        let (lim_x, lim_y) = if idx + 1 < chain.len() {
            (chain[idx + 1].0, chain[idx + 1].1)
        } else {
            (lx, ly)
        };
        for off in 0..(span as usize) {
            let ax = ix + off;
            let ay = iy + off;
            if ax >= lim_x || ay >= lim_y || ax >= lx || ay >= ly {
                break;
            }
            if anchors.last().map_or(true, |&(px_a, py_a)| ax > px_a && ay > py_a) {
                anchors.push((ax, ay));
            }
        }
    }

    if anchors.len() < 8 {
        return unanchored_corridor(lx, ly);
    }

    // Estimate empirical AA and synonymous divergence across anchors and short ungapped bridges
    let mode = px.seq.mode;
    let k_lim: u8 = if mode == SeqMode::Dna { 4 } else { 20 };
    let mut aa_tot = 0usize;
    let mut aa_mm = 0usize;
    let mut aa_id_cod = 0usize;
    let mut cod_mm = 0usize;

    let mut sample_pair = |ax: usize, ay: usize| {
        if px.aa_proj[ax] >= k_lim || py.aa_proj[ay] >= k_lim {
            return;
        }
        aa_tot += 1;
        if px.aa_proj[ax] != py.aa_proj[ay] {
            aa_mm += 1;
        } else if mode == SeqMode::Codon
            && px.seq.ambig_masks[ax].count_ones() == 1
            && py.seq.ambig_masks[ay].count_ones() == 1
        {
            aa_id_cod += 1;
            if px.seq.tokens[ax] != py.seq.tokens[ay] {
                cod_mm += 1;
            }
        }
    };

    for idx in 0..anchors.len() {
        let (ax, ay) = anchors[idx];
        sample_pair(ax, ay);
        if idx + 1 < anchors.len() {
            let (nx, ny) = anchors[idx + 1];
            let di = nx - ax;
            let dj = ny - ay;
            if di == dj && di > 1 && di <= 28 {
                for step in 1..di {
                    sample_pair(ax + step, ay + step);
                }
            }
        }
    }

    let p_aa_mm = aa_mm as f64 / (aa_tot.max(1) as f64);
    let est_aa_div = (-0.95 * (1.0 - (p_aa_mm / 0.92).min(0.88)).ln()).clamp(0.003, 2.0);
    let est_syn_div = if mode == SeqMode::Codon {
        if aa_id_cod >= 6 {
            let p_syn = cod_mm as f64 / (aa_id_cod as f64);
            (-0.85 * (1.0 - (p_syn / 0.72).min(0.92)).ln()).clamp(0.006, 6.0)
        } else {
            (3.0 * est_aa_div).clamp(0.006, 6.0)
        }
    } else {
        est_aa_div
    };

    let mut est_aligned = anchors.len();
    for r in 0..(anchors.len() - 1) {
        let di = anchors[r + 1].0 - anchors[r].0 - 1;
        let dj = anchors[r + 1].1 - anchors[r].1 - 1;
        est_aligned += di.min(dj);
    }
    let &(i0, j0) = &anchors[0];
    let &(ilast, jlast) = &anchors[anchors.len() - 1];
    est_aligned += i0.min(j0) + (lx - 1 - ilast).min(ly - 1 - jlast);
    let est_f_shared = (2.0 * (est_aligned as f64) / ((lx + ly) as f64)).clamp(0.05, 1.0);

    const W: usize = 32;
    let mut raw_lo = vec![ly; lx + 1];
    let mut raw_hi = vec![0usize; lx + 1];
    let mut mark_row = |i: usize, lo: usize, hi: usize| {
        raw_lo[i] = raw_lo[i].min(lo.min(ly));
        raw_hi[i] = raw_hi[i].max(hi.min(ly));
    };

    // N-flank
    if i0 <= 220 || j0 <= 220 {
        let hi0 = (j0 + 1 + W).min(ly);
        for i in 0..=(i0 + 1).min(lx) {
            mark_row(i, 0, hi0);
        }
    } else {
        let diag0 = (j0 as isize) - (i0 as isize);
        let w_flank = (W + 64) as isize;
        mark_row(0, 0, (j0 + 1 + W).min(ly));
        for i in 0..=(i0 + 1).min(lx) {
            let center = (i as isize) + diag0;
            let lo = (center - w_flank).max(0) as usize;
            let hi = (center + w_flank).clamp(0, ly as isize) as usize;
            mark_row(i, lo, hi);
        }
    }

    // Inter-anchor gaps
    for r in 0..(anchors.len() - 1) {
        let (ia, ja) = anchors[r];
        let (ib, jb) = anchors[r + 1];
        let di = ib - ia;
        let dj = jb - ja;
        if ((di <= 220 || dj <= 220) && di.max(dj) <= 1_800)
            || (di.max(dj) <= 1_800 && di.saturating_mul(dj) <= 1_440_000)
        {
            let lo = ja.saturating_sub(W);
            let hi = (jb + 1 + W).min(ly);
            for i in ia..=(ib + 1).min(lx) {
                mark_row(i, lo, hi);
            }
        } else {
            let w_gap = (W + 96) as isize;
            for i in ia..=(ib + 1).min(lx) {
                let num = (i - ia) * dj;
                let center = (ja + num / di.max(1)) as isize;
                let lo = (center - w_gap).max(0) as usize;
                let hi = (center + w_gap).clamp(0, ly as isize) as usize;
                mark_row(i, lo, hi);
            }
        }
    }

    // C-flank
    if (lx - ilast) <= 220 || (ly - jlast) <= 220 {
        let lo_last = jlast.saturating_sub(W);
        for i in ilast..=lx {
            mark_row(i, lo_last, ly);
        }
    } else {
        let diag_last = (jlast as isize) - (ilast as isize);
        let w_flank = (W + 64) as isize;
        for i in ilast..=lx {
            let center = (i as isize) + diag_last;
            let lo = (center - w_flank).max(0) as usize;
            let hi = (center + w_flank).clamp(0, ly as isize) as usize;
            mark_row(i, lo, hi);
        }
        mark_row(lx, jlast.saturating_sub(W), ly);
    }

    let prelim = CorridorBounds::from_bounds(
        lx,
        ly,
        raw_lo,
        raw_hi,
        false,
        true,
        anchors.len(),
        est_aa_div,
        est_syn_div,
        est_f_shared,
    );
    let full_cells = (lx + 1).saturating_mul(ly + 1);
    if prelim.row_offset[lx + 1] * 10 >= 7 * full_cells {
        CorridorBounds::from_bounds(
            lx,
            ly,
            vec![0; lx + 1],
            vec![ly; lx + 1],
            true,
            true,
            anchors.len(),
            est_aa_div,
            est_syn_div,
            est_f_shared,
        )
    } else {
        prelim
    }
}

/// Exact linear-space block-scaled path partition function restricted to a fixed `CorridorBounds`.
pub fn compute_ln_path_partition_corridor(
    lx: usize,
    ly: usize,
    corridor: &CorridorBounds,
    regime: TransducerRegime,
    prior_mean_len: f64,
) -> f64 {
    assert!(lx >= 1, "Transducer requires Lx >= 1");
    assert_eq!(corridor.lx, lx);
    assert_eq!(corridor.ly, ly);
    if ly == 0 {
        return NEG_INF;
    }

    let p = LinearTransducerParams::new(lx, regime, prior_mean_len);
    let n_blks = (ly >> BLK_SHIFT) + 1;
    let mut e_pre_m = Vec::with_capacity(ly + 1);
    let mut e_pre_blk_exp = Vec::with_capacity(n_blks);
    build_linear_e_pre(&p, ly, &mut e_pre_m, &mut e_pre_blk_exp);

    let row_len = (ly + 1) * NB;
    let mut prev_row = vec![0.0f64; row_len];
    let mut cur_row = vec![0.0f64; row_len];
    let mut prev_blk_exp = vec![ZERO_EXP; n_blks];
    let mut cur_blk_exp = vec![ZERO_EXP; n_blks];
    let mut post_y_m = vec![0.0f64; ly + 1];
    let mut post_y_e = vec![ZERO_EXP; ly + 1];
    let mut eof_m = [0.0f64];
    let mut eof_e = [ZERO_EXP];

    for i in 1..=lx {
        let (p_lo, p_hi) = if i >= 2 {
            (corridor.j_lo[i - 1], corridor.j_hi[i - 1])
        } else {
            (1, 0)
        };
        let c_lo = corridor.j_lo[i];
        let c_hi = corridor.j_hi[i];
        forward_row_step::<false, false>(
            i,
            lx,
            ly,
            &p,
            p_lo,
            p_hi,
            &prev_row,
            0,
            &prev_blk_exp,
            0,
            c_lo,
            c_hi,
            &mut cur_row,
            0,
            &mut cur_blk_exp,
            0,
            &e_pre_m,
            &e_pre_blk_exp,
            &[],
            &[],
            0,
            0,
            &mut eof_m,
            &mut eof_e,
            &mut post_y_m,
            &mut post_y_e,
        );
        std::mem::swap(&mut prev_row, &mut cur_row);
        std::mem::swap(&mut prev_blk_exp, &mut cur_blk_exp);
    }

    let mut tail_m = 0.0f64;
    let mut tail_e = ZERO_EXP;
    for j in 1..=ly {
        add_scaled(&mut tail_m, &mut tail_e, post_y_m[j], post_y_e[j]);
        if tail_m > 0.0 && tail_e != ZERO_EXP {
            if j == ly {
                add_scaled(&mut eof_m[0], &mut eof_e[0], tail_m * p.one_minus_eta_y, tail_e);
            } else {
                tail_m *= p.eta_y;
                canon_scalar(&mut tail_m, &mut tail_e);
            }
        }
    }
    scaled_to_ln(eof_m[0], eof_e[0])
}

pub struct BandedRegimeFB {
    pub ln_prob_y_given_x: f64,
    /// Flat array aligned to `corridor.emit_offset[lx]`: index `(i, j)` via `corridor.emit_idx(i, j)`.
    pub post_fine_banded: Vec<f64>,
    pub post_bx: Vec<f64>,
    pub post_by: Vec<f64>,
}

#[derive(Default)]
struct FbWorkspace {
    ln_coarse_x: Vec<f64>,
    r_eff_buf: Vec<f64>,
    emit_hom: Vec<f64>,
    emit_coarse: Vec<f64>,
    e_pre_m: Vec<f64>,
    e_pre_blk_exp: Vec<i32>,
    row0: Vec<f64>,
    row1: Vec<f64>,
    blk0: Vec<i32>,
    blk1: Vec<i32>,
    post_y_m: Vec<f64>,
    post_y_e: Vec<i32>,
    post_y_bwd_m: Vec<f64>,
    post_y_bwd_e: Vec<i32>,
    row_blk_off: Vec<usize>,
    fwd: Vec<f64>,
    fwd_blk_exp: Vec<i32>,
    bwd: Vec<f64>,
    bwd_blk_exp: Vec<i32>,
}

thread_local! {
    static FB_WS: RefCell<FbWorkspace> = RefCell::new(FbWorkspace::default());
}

/// 5-State Interior + 1D Factored Overhang Banded Forward-Backward pass on `corridor`.
/// Runs in block-scaled linear space (`BLK_W = 16` IEEE-754 exponent normalization) with
/// factored null potential `sum_ln_m0_y`, eliminating per-cell `ln`/`exp`/`log1p` overhead.
pub fn run_forward_backward_regime_corridor(
    px: &PreparedSeq,
    py: &PreparedSeq,
    corridor: &CorridorBounds,
    local_coarse_tilts_x: &[f64],
    copula_ratio_mat: &[f64],
    ln_r_base: &[f64],
    r_pow_cache: Option<&[OnceLock<Vec<f64>>]>,
    regime: TransducerRegime,
    prior_mean_len: f64,
    compute_posteriors: bool,
) -> BandedRegimeFB {
    run_forward_backward_regime_corridor_cached(
        px,
        py,
        corridor,
        local_coarse_tilts_x,
        copula_ratio_mat,
        ln_r_base,
        r_pow_cache,
        None,
        regime,
        prior_mean_len,
        compute_posteriors,
    )
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn compute_corridor_row_emissions(
    i: usize,
    elo: usize,
    ehi: usize,
    px: &PreparedSeq,
    py: &PreparedSeq,
    k_alpha: usize,
    full_mask: u64,
    local_coarse_tilts_x: &[f64],
    q_to_slot: &[u8; Q_STEPS + 1],
    r_pow_by_slot: &[&[f64]],
    inv_z_q0_opt: Option<&[f64]>,
    ln_coarse_x: &mut [f64],
    r_eff_buf: &mut [f64],
    out_hom: &mut [f64],
    out_coarse: &mut [f64],
) {
    if elo >= ehi {
        return;
    }
    let xi = px.seq.tokens[i] as usize;
    let qxi = px.qx[i];
    let rho_x = px.prof.ref_repeat_weight[i];
    let c_tilt_row = &local_coarse_tilts_x[i * k_alpha..(i + 1) * k_alpha];
    // rho_bi = sqrt(rho_x * rho_y) <= 0.25 automatically whenever rho_x <= 0.0625 (since rho_y <= 0.99 < 1.0)
    let need_ln_c = rho_x > 0.0625;
    if need_ln_c {
        for z in 0..k_alpha {
            ln_coarse_x[z] = c_tilt_row[z].max(1e-300).ln();
        }
    }

    if !px.seq.has_ambig && !py.seq.has_ambig {
        for j in elo..ehi {
            let yj = py.seq.tokens[j] as usize;
            let qyj = py.qy[j];
            let q = qxi.max(qyj) as usize;
            let q_slot = q_to_slot[q] as usize;
            let m0_row = &py.prof.site_m0_dist[j * k_alpha..(j + 1) * k_alpha];
            let r_row = &r_pow_by_slot[q_slot][xi * k_alpha..(xi + 1) * k_alpha];

            let inv_z = if q == 0 {
                if let Some(iz_q0) = inv_z_q0_opt {
                    iz_q0[j * k_alpha + xi]
                } else {
                    1.0 / dot_k_unrolled(m0_row, r_row).max(1e-300)
                }
            } else {
                1.0 / dot_k_unrolled(m0_row, r_row).max(1e-300)
            };

            let out_idx = j - elo;
            out_hom[out_idx] = r_row[yj].max(1e-300) * inv_z;

            if need_ln_c {
                let rho_y = py.prof.site_repeat_weight[j];
                let rho_bi = (rho_x * rho_y).sqrt();
                if rho_bi > 0.25 {
                    let pwr_c = 1.0 - 0.85 * rho_bi;
                    let mut z_coarse = 0.0;
                    for z in 0..k_alpha {
                        z_coarse += m0_row[z] * (pwr_c * ln_coarse_x[z]).exp();
                    }
                    out_coarse[out_idx] = (pwr_c * ln_coarse_x[yj]).exp() / z_coarse.max(1e-300);
                    continue;
                }
            }
            let z_coarse = dot_k_unrolled(m0_row, c_tilt_row);
            out_coarse[out_idx] = c_tilt_row[yj].max(1e-300) / z_coarse.max(1e-300);
        }
    } else {
        let mx = px.seq.ambig_masks[i];
        for j in elo..ehi {
            let yj = py.seq.tokens[j] as usize;
            let my = py.seq.ambig_masks[j];
            let qyj = py.qy[j];
            let q = qxi.max(qyj) as usize;
            let q_slot = q_to_slot[q] as usize;
            let m0_row = &py.prof.site_m0_dist[j * k_alpha..(j + 1) * k_alpha];

            let inv_z_opt = if q == 0 && mx.count_ones() == 1 && my != full_mask {
                inv_z_q0_opt.map(|iz_q0| iz_q0[j * k_alpha + xi])
            } else {
                None
            };
            let (e_hom, _) = eval_ambig_emit_hom(
                mx,
                xi,
                my,
                yj,
                full_mask,
                k_alpha,
                r_pow_by_slot[q_slot],
                inv_z_opt,
                m0_row,
                r_eff_buf,
            );
            let out_idx = j - elo;
            out_hom[out_idx] = e_hom;

            let rho_bi = if need_ln_c {
                (rho_x * py.prof.site_repeat_weight[j]).sqrt()
            } else {
                0.0
            };
            out_coarse[out_idx] = eval_ambig_emit_coarse(
                my,
                yj,
                full_mask,
                k_alpha,
                c_tilt_row,
                ln_coarse_x,
                rho_bi,
                m0_row,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_forward_backward_regime_corridor_cached(
    px: &PreparedSeq,
    py: &PreparedSeq,
    corridor: &CorridorBounds,
    local_coarse_tilts_x: &[f64],
    copula_ratio_mat: &[f64],
    ln_r_base: &[f64],
    r_pow_cache: Option<&[OnceLock<Vec<f64>>]>,
    inv_z_q0_opt: Option<&[f64]>,
    regime: TransducerRegime,
    prior_mean_len: f64,
    compute_posteriors: bool,
) -> BandedRegimeFB {
    let lx = px.seq.len();
    let ly = py.seq.len();
    let k_alpha = px.seq.alphabet_size();
    assert!(lx >= 1, "Transducer requires Lx >= 1");
    assert_eq!(corridor.lx, lx);
    assert_eq!(corridor.ly, ly);

    if ly == 0 {
        return BandedRegimeFB {
            ln_prob_y_given_x: NEG_INF,
            post_fine_banded: Vec::new(),
            post_bx: Vec::new(),
            post_by: Vec::new(),
        };
    }

    let p = LinearTransducerParams::new(lx, regime, prior_mean_len);
    let ln_m0_y = &py.prof.site_ln_m0;
    let sum_ln_m0_y: f64 = ln_m0_y.iter().sum();

    // Build compact used-q slot map covering 100% of (q, j, xi) queries in this pair
    let mut q_to_slot = [u8::MAX; Q_STEPS + 1];
    q_to_slot[0] = 0;
    let mut used_qs: Vec<usize> = Vec::with_capacity(8);
    used_qs.push(0);
    for &q in px.qx.iter().chain(py.qy.iter()) {
        let qu = q as usize;
        if q_to_slot[qu] == u8::MAX {
            q_to_slot[qu] = used_qs.len() as u8;
            used_qs.push(qu);
        }
    }
    let n_used_q = used_qs.len();

    // Build fallback local r_pow table ONLY when caller didn't provide cached GlobalRegimeTables
    let mut local_r_pow: Vec<Option<Vec<f64>>> = Vec::new();
    if r_pow_cache.is_none() {
        local_r_pow = vec![None; Q_STEPS + 1];
        for &qu in &used_qs {
            if qu >= 1 {
                let pwr = 1.0 - 0.78 * (qu as f64 / (Q_STEPS as f64));
                local_r_pow[qu] = Some(ln_r_base.iter().map(|&lr| (pwr * lr).exp()).collect());
            }
        }
    }

    let mut r_pow_by_slot: Vec<&[f64]> = Vec::with_capacity(n_used_q);
    for &qu in &used_qs {
        let slice = if qu == 0 {
            copula_ratio_mat
        } else if let Some(locks) = r_pow_cache {
            locks[qu]
                .get_or_init(|| {
                    let pwr = 1.0 - 0.78 * (qu as f64 / (Q_STEPS as f64));
                    ln_r_base.iter().map(|&lr| (pwr * lr).exp()).collect()
                })
                .as_slice()
        } else {
            local_r_pow[qu].as_deref().unwrap()
        };
        r_pow_by_slot.push(slice);
    }

    FB_WS.with(|ws_cell| {
        let ws = &mut *ws_cell.borrow_mut();

        ws.ln_coarse_x.resize(k_alpha, 0.0);
        ws.r_eff_buf.resize(k_alpha, 0.0);
        let full_mask: u64 = if k_alpha >= 64 {
            u64::MAX
        } else {
            (1u64 << k_alpha) - 1
        };

        let n_blks = (ly >> BLK_SHIFT) + 1;
        build_linear_e_pre(&p, ly, &mut ws.e_pre_m, &mut ws.e_pre_blk_exp);

        ws.post_y_m.clear();
        ws.post_y_m.resize(ly + 1, 0.0);
        ws.post_y_e.clear();
        ws.post_y_e.resize(ly + 1, ZERO_EXP);
        let mut eof_m = [0.0f64];
        let mut eof_e = [ZERO_EXP];

        if !compute_posteriors {
            ws.emit_hom.resize(ly, 0.0);
            ws.emit_coarse.resize(ly, 0.0);
            let row_len = (ly + 1) * NB;
            ws.row0.resize(row_len, 0.0);
            ws.row1.resize(row_len, 0.0);
            ws.blk0.clear();
            ws.blk0.resize(n_blks, ZERO_EXP);
            ws.blk1.clear();
            ws.blk1.resize(n_blks, ZERO_EXP);

            let prev_row = &mut ws.row0;
            let cur_row = &mut ws.row1;
            let prev_blk_exp = &mut ws.blk0;
            let cur_blk_exp = &mut ws.blk1;

            for i in 1..=lx {
                let (p_lo, p_hi) = if i >= 2 {
                    (corridor.j_lo[i - 1], corridor.j_hi[i - 1])
                } else {
                    (1, 0)
                };
                let c_lo = corridor.j_lo[i];
                let c_hi = corridor.j_hi[i];
                let r = i - 1;
                let elo = corridor.emit_lo[r];
                let ehi = corridor.emit_hi[r];
                let row_span = ehi.saturating_sub(elo);
                if row_span > 0 {
                    compute_corridor_row_emissions(
                        r,
                        elo,
                        ehi,
                        px,
                        py,
                        k_alpha,
                        full_mask,
                        local_coarse_tilts_x,
                        &q_to_slot,
                        &r_pow_by_slot,
                        inv_z_q0_opt,
                        &mut ws.ln_coarse_x,
                        &mut ws.r_eff_buf,
                        &mut ws.emit_hom[..row_span],
                        &mut ws.emit_coarse[..row_span],
                    );
                }

                forward_row_step::<true, false>(
                    i,
                    lx,
                    ly,
                    &p,
                    p_lo,
                    p_hi,
                    prev_row,
                    0,
                    prev_blk_exp,
                    0,
                    c_lo,
                    c_hi,
                    cur_row,
                    0,
                    cur_blk_exp,
                    0,
                    &ws.e_pre_m,
                    &ws.e_pre_blk_exp,
                    &ws.emit_hom,
                    &ws.emit_coarse,
                    0,
                    elo,
                    &mut eof_m,
                    &mut eof_e,
                    &mut ws.post_y_m,
                    &mut ws.post_y_e,
                );
                std::mem::swap(prev_row, cur_row);
                std::mem::swap(prev_blk_exp, cur_blk_exp);
            }

            let mut tail_m = 0.0f64;
            let mut tail_e = ZERO_EXP;
            for j in 1..=ly {
                add_scaled(&mut tail_m, &mut tail_e, ws.post_y_m[j], ws.post_y_e[j]);
                if tail_m > 0.0 && tail_e != ZERO_EXP {
                    if j == ly {
                        add_scaled(
                            &mut eof_m[0],
                            &mut eof_e[0],
                            tail_m * p.one_minus_eta_y,
                            tail_e,
                        );
                    } else {
                        tail_m *= p.eta_y;
                        canon_scalar(&mut tail_m, &mut tail_e);
                    }
                }
            }
            let ln_odds = scaled_to_ln(eof_m[0], eof_e[0]);
            let ln_prob = if ln_odds > NEG_INF {
                ln_odds + sum_ln_m0_y
            } else {
                NEG_INF
            };
            return BandedRegimeFB {
                ln_prob_y_given_x: ln_prob,
                post_fine_banded: Vec::new(),
                post_bx: Vec::new(),
                post_by: Vec::new(),
            };
        }

        let n_emit = corridor.emit_offset[lx];
        ws.emit_hom.resize(n_emit, 0.0);
        ws.emit_coarse.resize(n_emit, 0.0);
        for i in 0..lx {
            let elo = corridor.emit_lo[i];
            let ehi = corridor.emit_hi[i];
            if elo < ehi {
                let ebase = corridor.emit_offset[i];
                let eend = ebase + (ehi - elo);
                compute_corridor_row_emissions(
                    i,
                    elo,
                    ehi,
                    px,
                    py,
                    k_alpha,
                    full_mask,
                    local_coarse_tilts_x,
                    &q_to_slot,
                    &r_pow_by_slot,
                    inv_z_q0_opt,
                    &mut ws.ln_coarse_x,
                    &mut ws.r_eff_buf,
                    &mut ws.emit_hom[ebase..eend],
                    &mut ws.emit_coarse[ebase..eend],
                );
            }
        }

        // Full Linear Forward-Backward on corridor sparse storage
        ws.row_blk_off.clear();
        ws.row_blk_off.reserve(lx + 2);
        let mut blk_acc = 0usize;
        ws.row_blk_off.push(0);
        for i in 0..=lx {
            let b0 = corridor.j_lo[i] >> BLK_SHIFT;
            let b1 = corridor.j_hi[i] >> BLK_SHIFT;
            blk_acc += b1 - b0 + 1;
            ws.row_blk_off.push(blk_acc);
        }

        let total_cells = corridor.row_offset[lx + 1];
        ws.fwd.resize(total_cells * NB, 0.0);
        ws.fwd_blk_exp.clear();
        ws.fwd_blk_exp.resize(blk_acc, ZERO_EXP);

        for i in 1..=lx {
            let p_lo = corridor.j_lo[i - 1];
            let p_hi = corridor.j_hi[i - 1];
            let c_lo = corridor.j_lo[i];
            let c_hi = corridor.j_hi[i];
            let (p_start, p_end) = if i >= 2 {
                (corridor.row_offset[i - 1] * NB, corridor.row_offset[i] * NB)
            } else {
                (0, 0)
            };
            let c_start = corridor.row_offset[i] * NB;
            let c_end = corridor.row_offset[i + 1] * NB;
            let (left_fwd, right_fwd) = ws.fwd.split_at_mut(c_start);
            let prev_slice = if i >= 2 { &left_fwd[p_start..p_end] } else { &[] };
            let cur_slice = &mut right_fwd[..(c_end - c_start)];

            let pb_start = ws.row_blk_off[i - 1];
            let cb_start = ws.row_blk_off[i];
            let cb_end = ws.row_blk_off[i + 1];
            let (left_bexp, right_bexp) = ws.fwd_blk_exp.split_at_mut(cb_start);
            let prev_blk_slice = &left_bexp[pb_start..cb_start];
            let cur_blk_slice = &mut right_bexp[..(cb_end - cb_start)];

            let e_prev_base = corridor.emit_offset[i - 1];
            let e_prev_lo = corridor.emit_lo[i - 1];

            forward_row_step::<true, false>(
                i,
                lx,
                ly,
                &p,
                if i >= 2 { p_lo } else { 1 },
                if i >= 2 { p_hi } else { 0 },
                prev_slice,
                p_lo,
                prev_blk_slice,
                p_lo >> BLK_SHIFT,
                c_lo,
                c_hi,
                cur_slice,
                c_lo,
                cur_blk_slice,
                c_lo >> BLK_SHIFT,
                &ws.e_pre_m,
                &ws.e_pre_blk_exp,
                &ws.emit_hom,
                &ws.emit_coarse,
                e_prev_base,
                e_prev_lo,
                &mut eof_m,
                &mut eof_e,
                &mut ws.post_y_m,
                &mut ws.post_y_e,
            );
        }

        let mut tail_m = 0.0f64;
        let mut tail_e = ZERO_EXP;
        for j in 1..=ly {
            add_scaled(&mut tail_m, &mut tail_e, ws.post_y_m[j], ws.post_y_e[j]);
            if tail_m > 0.0 && tail_e != ZERO_EXP {
                if j == ly {
                    add_scaled(
                        &mut eof_m[0],
                        &mut eof_e[0],
                        tail_m * p.one_minus_eta_y,
                        tail_e,
                    );
                } else {
                    tail_m *= p.eta_y;
                    canon_scalar(&mut tail_m, &mut tail_e);
                }
            }
        }
        canon_scalar(&mut eof_m[0], &mut eof_e[0]);
        if eof_m[0] <= 0.0 || eof_e[0] == ZERO_EXP {
            return BandedRegimeFB {
                ln_prob_y_given_x: NEG_INF,
                post_fine_banded: Vec::new(),
                post_bx: Vec::new(),
                post_by: Vec::new(),
            };
        }
        let ln_eof_total = scaled_to_ln(eof_m[0], eof_e[0]) + sum_ln_m0_y;

        build_linear_post_y_bwd(&p, ly, &mut ws.post_y_bwd_m, &mut ws.post_y_bwd_e);

        ws.bwd.resize(total_cells * NB, 0.0);
        ws.bwd_blk_exp.clear();
        ws.bwd_blk_exp.resize(blk_acc, ZERO_EXP);

        for i in (1..=lx).rev() {
            let c_lo = corridor.j_lo[i];
            let c_hi = corridor.j_hi[i];
            let b_start = c_lo >> BLK_SHIFT;
            let b_end = c_hi >> BLK_SHIFT;

            let c_row_off = corridor.row_offset[i];
            let c_row_end = corridor.row_offset[i + 1];
            let c_blk_off = ws.row_blk_off[i];
            let c_blk_end = ws.row_blk_off[i + 1];

            let (n_lo, n_hi, nb_start, n_blk_end) = if i < lx {
                (
                    corridor.j_lo[i + 1],
                    corridor.j_hi[i + 1],
                    ws.row_blk_off[i + 1],
                    ws.row_blk_off[i + 2],
                )
            } else {
                (1, 0, c_blk_end, c_blk_end)
            };
            let n_b_lo = n_lo >> BLK_SHIFT;
            let n_b_hi = n_hi >> BLK_SHIFT;

            let (cur_slice, next_all) = ws.bwd[c_row_off * NB..].split_at_mut((c_row_end - c_row_off) * NB);
            let (cur_blk_slice, next_blk_all) =
                ws.bwd_blk_exp[c_blk_off..].split_at_mut(c_blk_end - c_blk_off);
            let next_blk_slice = if i < lx {
                &next_blk_all[..(n_blk_end - c_blk_end)]
            } else {
                &[]
            };

            let e_prev_base = corridor.emit_offset[i - 1];
            let e_prev_lo = corridor.emit_lo[i - 1];
            let (e_cur_base, e_cur_lo) = if i < lx {
                (corridor.emit_offset[i], corridor.emit_lo[i])
            } else {
                (0, 0)
            };

            for b in (b_start..=b_end).rev() {
                let jb_lo = (b << BLK_SHIFT).max(c_lo);
                let jb_hi = ((b << BLK_SHIFT) | BLK_MASK).min(c_hi);
                let need_right_block = (jb_hi & BLK_MASK) == BLK_MASK && jb_hi < ly;

                let mut e_ref = ZERO_EXP;
                let mut consider = |e: i32| {
                    if e != ZERO_EXP && (e_ref == ZERO_EXP || e > e_ref) {
                        e_ref = e;
                    }
                };

                if jb_hi == ly {
                    consider(0);
                }
                if (jb_lo & BLK_MASK) < BLK_MASK && jb_lo < ly {
                    consider(ws.post_y_bwd_e[b]);
                }
                if need_right_block {
                    consider(ws.post_y_bwd_e[b + 1]);
                }
                if b < b_end {
                    consider(cur_blk_slice[(b + 1) - b_start]);
                }
                if i < lx {
                    if b >= n_b_lo && b <= n_b_hi {
                        consider(next_blk_slice[b - n_b_lo]);
                    }
                    if need_right_block && (b + 1) >= n_b_lo && (b + 1) <= n_b_hi {
                        consider(next_blk_slice[(b + 1) - n_b_lo]);
                    }
                }

                if e_ref == ZERO_EXP {
                    cur_blk_slice[b - b_start] = ZERO_EXP;
                    let clear_lo = (jb_lo - c_lo) * NB;
                    let clear_hi = (jb_hi - c_lo + 1) * NB;
                    cur_slice[clear_lo..clear_hi].fill(0.0);
                    continue;
                }

                let s_eof = ldexp_f64(1.0, -e_ref);
                let s_py_same = if (jb_lo & BLK_MASK) < BLK_MASK
                    && jb_lo < ly
                    && ws.post_y_bwd_e[b] != ZERO_EXP
                {
                    ldexp_f64(1.0, ws.post_y_bwd_e[b] - e_ref)
                } else {
                    0.0
                };
                let s_py_right =
                    if need_right_block && ws.post_y_bwd_e[b + 1] != ZERO_EXP {
                        ldexp_f64(1.0, ws.post_y_bwd_e[b + 1] - e_ref)
                    } else {
                        0.0
                    };
                let s_c_right = if b < b_end && cur_blk_slice[(b + 1) - b_start] != ZERO_EXP {
                    ldexp_f64(1.0, cur_blk_slice[(b + 1) - b_start] - e_ref)
                } else {
                    0.0
                };
                let (s_n_same, s_n_right) = if i < lx {
                    let ss = if b >= n_b_lo && b <= n_b_hi && next_blk_slice[b - n_b_lo] != ZERO_EXP
                    {
                        ldexp_f64(1.0, next_blk_slice[b - n_b_lo] - e_ref)
                    } else {
                        0.0
                    };
                    let sr = if need_right_block
                        && (b + 1) >= n_b_lo
                        && (b + 1) <= n_b_hi
                        && next_blk_slice[(b + 1) - n_b_lo] != ZERO_EXP
                    {
                        ldexp_f64(1.0, next_blk_slice[(b + 1) - n_b_lo] - e_ref)
                    } else {
                        0.0
                    };
                    (ss, sr)
                } else {
                    (0.0, 0.0)
                };

                let (mut right_bi, mut right_bby) = if b < b_end && s_c_right != 0.0 {
                    let r_off = (jb_hi + 1 - c_lo) * NB;
                    (
                        cur_slice[r_off + BI] * s_c_right,
                        cur_slice[r_off + BBY] * s_c_right,
                    )
                } else {
                    (0.0, 0.0)
                };

                for j in (jb_lo..=jb_hi).rev() {
                    let in_next_same = i < lx && j >= n_lo && j <= n_hi;
                    let in_next_diag = i < lx && j < ly && (j + 1) >= n_lo && (j + 1) <= n_hi;
                    let in_cur_right = j < c_hi && j < ly;
                    let use_right = (j & BLK_MASK) == BLK_MASK;

                    let py_term = if j < ly {
                        let s_py = if use_right { s_py_right } else { s_py_same };
                        ws.post_y_bwd_m[j + 1] * s_py
                    } else {
                        0.0
                    };
                    let ec_prev = if in_cur_right {
                        ws.emit_coarse[e_prev_base + (j - e_prev_lo)]
                    } else {
                        0.0
                    };
                    let (eh_cur, bm_next_diag) = if in_next_diag {
                        let s_nd = if use_right { s_n_right } else { s_n_same };
                        let bm_off = ((corridor.row_offset[i + 1] - c_row_end) + (j + 1 - n_lo)) * NB;
                        (
                            ws.emit_hom[e_cur_base + (j - e_cur_lo)],
                            next_all[bm_off + BM] * s_nd,
                        )
                    } else {
                        (0.0, 0.0)
                    };
                    let eh_bm_next = eh_cur * bm_next_diag;

                    let v_bby = if j >= 1 {
                        let mut v = 0.0;
                        if in_cur_right {
                            v += p.by_ext * ec_prev * right_bby;
                        }
                        if i == lx {
                            if j == ly {
                                v += p.by_exit * p.coterm_eof * s_eof;
                            }
                            if j < ly {
                                v += p.by_exit * p.coterm_post * py_term;
                            }
                        } else if in_next_diag {
                            v += p.by_exit * eh_bm_next;
                        }
                        v
                    } else {
                        0.0
                    };

                    let v_bbx = if i == lx {
                        if in_cur_right {
                            ec_prev * right_bby
                        } else {
                            0.0
                        }
                    } else {
                        let mut v = 0.0;
                        if in_next_same {
                            let n_off = ((corridor.row_offset[i + 1] - c_row_end) + (j - n_lo)) * NB;
                            v += p.bx_ext * next_all[n_off + BBX] * s_n_same;
                        }
                        if in_cur_right {
                            v += p.bx_to_by * ec_prev * right_bby;
                        }
                        v
                    };

                    let v_bi = if i < lx && j >= 1 {
                        let mut v = 0.0;
                        if in_cur_right {
                            v += p.indel_ext * right_bi;
                        }
                        if in_next_diag {
                            v += p.indel_exit * eh_bm_next;
                        }
                        v
                    } else {
                        0.0
                    };

                    let v_bd = if i == lx {
                        let mut v = 0.0;
                        if j == ly {
                            v += p.coterm_eof * s_eof;
                        }
                        if j < ly {
                            v += p.coterm_post * py_term;
                        }
                        v
                    } else {
                        let mut v = 0.0;
                        if in_next_same {
                            let n_off = ((corridor.row_offset[i + 1] - c_row_end) + (j - n_lo)) * NB;
                            v += p.indel_ext * next_all[n_off + BD] * s_n_same;
                        }
                        if in_next_diag {
                            v += p.indel_exit * eh_bm_next;
                        }
                        v
                    };

                    let v_bm = if j >= 1 {
                        if i == lx {
                            let mut v = 0.0;
                            if j == ly {
                                v += p.coterm_eof * s_eof;
                            }
                            if j < ly {
                                v += p.coterm_post * py_term;
                            }
                            v
                        } else {
                            let mut v = 0.0;
                            if j == ly {
                                v += p.m_to_end * p.early_eof * s_eof;
                            }
                            if j < ly {
                                v += p.m_to_end * p.early_post * py_term;
                            }
                            if in_next_same {
                                let n_off =
                                    ((corridor.row_offset[i + 1] - c_row_end) + (j - n_lo)) * NB;
                                v += p.m_to_d * next_all[n_off + BD] * s_n_same;
                                v += p.m_to_bx * next_all[n_off + BBX] * s_n_same;
                            }
                            if in_cur_right {
                                v += p.m_to_i * right_bi;
                            }
                            if in_next_diag {
                                v += p.m_to_m * eh_bm_next;
                            }
                            v
                        }
                    } else {
                        0.0
                    };

                    let cur_off = (j - c_lo) * NB;
                    cur_slice[cur_off + BBY] = v_bby;
                    cur_slice[cur_off + BBX] = v_bbx;
                    cur_slice[cur_off + BI] = v_bi;
                    cur_slice[cur_off + BD] = v_bd;
                    cur_slice[cur_off + BM] = v_bm;

                    right_bi = v_bi;
                    right_bby = v_bby;
                }

                let blk_lo = (jb_lo - c_lo) * NB;
                let blk_hi = (jb_hi - c_lo + 1) * NB;
                cur_blk_slice[b - b_start] =
                    normalize_block_slice(&mut cur_slice[blk_lo..blk_hi], e_ref);
            }
            let _ = nb_start;
        }

        let inv_eof_m = 1.0 / eof_m[0];
        let eof_exp = eof_e[0];
        let mut post_fine_banded = vec![0.0f64; n_emit];
        let mut post_bx = vec![0.0f64; lx];
        let mut post_by = vec![0.0f64; ly];

        for i in 0..lx {
            let r = i + 1;
            let n_lo = corridor.j_lo[r];
            let nb_lo = n_lo >> BLK_SHIFT;
            let n_row_off = corridor.row_offset[r];
            let n_blk_off = ws.row_blk_off[r];
            let elo = corridor.emit_lo[i];
            let ehi = corridor.emit_hi[i];
            let ebase = corridor.emit_offset[i];

            for j in elo..ehi {
                let b1 = (j + 1) >> BLK_SHIFT;
                let fe = ws.fwd_blk_exp[n_blk_off + (b1 - nb_lo)];
                let be = ws.bwd_blk_exp[n_blk_off + (b1 - nb_lo)];
                if fe != ZERO_EXP && be != ZERO_EXP {
                    let s = ldexp_f64(inv_eof_m, fe + be - eof_exp);
                    let c_idx = (n_row_off + (j + 1 - n_lo)) * NB + BM;
                    post_fine_banded[ebase + (j - elo)] =
                        (ws.fwd[c_idx] * ws.bwd[c_idx] * s).clamp(0.0, 1.0);
                }
            }
        }

        for i in 0..lx {
            let r = i + 1;
            let r_lo = corridor.j_lo[r];
            let r_hi = corridor.j_hi[r];
            let rb_start = r_lo >> BLK_SHIFT;
            let rb_end = r_hi >> BLK_SHIFT;
            let r_row_off = corridor.row_offset[r];
            let r_blk_off = ws.row_blk_off[r];
            let mut acc = 0.0;
            for b in rb_start..=rb_end {
                let fe = ws.fwd_blk_exp[r_blk_off + (b - rb_start)];
                let be = ws.bwd_blk_exp[r_blk_off + (b - rb_start)];
                if fe != ZERO_EXP && be != ZERO_EXP {
                    let s = ldexp_f64(inv_eof_m, fe + be - eof_exp);
                    let jb_lo = (b << BLK_SHIFT).max(r_lo);
                    let jb_hi = ((b << BLK_SHIFT) | BLK_MASK).min(r_hi);
                    let mut blk_sum = 0.0;
                    for j in jb_lo..=jb_hi {
                        let c_idx = (r_row_off + (j - r_lo)) * NB + BBX;
                        blk_sum += ws.fwd[c_idx] * ws.bwd[c_idx];
                    }
                    acc += blk_sum * s;
                }
            }
            post_bx[i] = acc.clamp(0.0, 1.0);
        }

        for i in 1..=lx {
            let r_lo = corridor.j_lo[i];
            let rb_start = r_lo >> BLK_SHIFT;
            let r_row_off = corridor.row_offset[i];
            let r_blk_off = ws.row_blk_off[i];
            let elo = corridor.emit_lo[i - 1];
            let ehi = corridor.emit_hi[i - 1];
            for j in elo..ehi {
                let b1 = (j + 1) >> BLK_SHIFT;
                let fe = ws.fwd_blk_exp[r_blk_off + (b1 - rb_start)];
                let be = ws.bwd_blk_exp[r_blk_off + (b1 - rb_start)];
                if fe != ZERO_EXP && be != ZERO_EXP {
                    let s = ldexp_f64(inv_eof_m, fe + be - eof_exp);
                    let c_idx = (r_row_off + (j + 1 - r_lo)) * NB + BBY;
                    post_by[j] += ws.fwd[c_idx] * ws.bwd[c_idx] * s;
                }
            }
        }
        for v in &mut post_by {
            *v = v.clamp(0.0, 1.0);
        }

        BandedRegimeFB {
            ln_prob_y_given_x: ln_eof_total,
            post_fine_banded,
            post_bx,
            post_by,
        }
    })
}

/// Exact analytical check that all outgoing probabilities over Y \in \Sigma^{0..=max_ly} sum to 1.0!
/// Used by unit tests to verify 100% Kraft conservation across arbitrary Lx and regimes.
pub fn exact_marginal_mass_up_to_len(lx: usize, regime: TransducerRegime, prior_mean_len: f64, max_ly: usize) -> f64 {
    // Since every Y-emitting transition sums over y_j \in \Sigma to exact 1.0 by normalization,
    // propagating total probability mass across j = 0..=max_ly (summing out y_j analytically = 1.0 per step!)
    // computes the exact sum \sum_{L_Y=0}^{max_ly} \sum_{Y \in \Sigma^{L_Y}} P_T(Y | X; regime)!
    let eta_y = prior_mean_len / (prior_mean_len + 1.0);
    let one_minus_eta_y = 1.0 / (prior_mean_len + 1.0);

    let (sx_0, sx_rest) = if lx == 1 {
        (1.0, 0.0)
    } else {
        (0.75, 0.25 / ((lx - 1) as f64))
    };

    let t_eff = regime.t_primary;
    let p_gap_total = (0.0008 + 0.038 * (1.0 - (-t_eff).exp())).clamp(0.0008, 0.045);
    let p_m_to_d = 0.5 * p_gap_total;
    let p_m_to_i = 0.5 * p_gap_total;
    let p_m_to_bx = (0.0018 + 0.014 * (1.0 - (-0.8 * t_eff).exp())).clamp(0.0018, 0.018);
    let p_m_to_end = (0.25 / (lx as f64 + 10.0)).clamp(0.0002, 0.02);
    let p_m_to_m = 1.0 - p_m_to_d - p_m_to_i - p_m_to_bx - p_m_to_end;

    let p_indel_ext = 0.65f64;
    let p_indel_exit = 1.0 - p_indel_ext;
    let p_bx_ext = 0.90f64;
    let p_bx_to_by = 1.0 - p_bx_ext;
    let p_by_ext = 0.90f64;
    let p_by_exit = 1.0 - p_by_ext;

    let stride_i = (max_ly + 1) * N_STATES;
    let mut mass = vec![0.0f64; (lx + 1) * stride_i];
    let idx = |i: usize, j: usize, s: usize| i * stride_i + j * N_STATES + s;

    for i0 in 0..lx {
        let sx = if i0 == 0 { sx_0 } else { sx_rest };
        let ec0 = sx * 0.75;
        if max_ly >= 1 {
            mass[idx(i0 + 1, 1, ST_M)] += ec0 * 0.95;
            mass[idx(i0, 1, ST_PRE_Y)] += sx * 0.25;
        }
        mass[idx(i0 + 1, 0, ST_BX)] += ec0 * 0.05;
    }

    let mut total_eof = 0.0f64;
    for i in 0..=lx {
        for j in 0..=max_ly {
            if i < lx && j >= 1 {
                let c = mass[idx(i, j, ST_PRE_Y)];
                if c > 0.0 {
                    if j < max_ly {
                        mass[idx(i, j + 1, ST_PRE_Y)] += c * eta_y;
                        mass[idx(i + 1, j + 1, ST_M)] += c * one_minus_eta_y * 0.95;
                    }
                    mass[idx(i + 1, j, ST_BX)] += c * one_minus_eta_y * 0.05;
                }
            }
            let cm = mass[idx(i, j, ST_M)];
            if cm > 0.0 {
                if i == lx {
                    total_eof += cm * 0.75;
                    if j < max_ly {
                        mass[idx(lx, j + 1, ST_POST_Y)] += cm * 0.25;
                    }
                } else {
                    total_eof += cm * p_m_to_end * 0.25;
                    if j < max_ly {
                        mass[idx(lx, j + 1, ST_POST_Y)] += cm * p_m_to_end * 0.75;
                        mass[idx(i, j + 1, ST_I)] += cm * p_m_to_i;
                        mass[idx(i + 1, j + 1, ST_M)] += cm * p_m_to_m;
                    }
                    mass[idx(i + 1, j, ST_D)] += cm * p_m_to_d;
                    mass[idx(i + 1, j, ST_BX)] += cm * p_m_to_bx;
                }
            }
            let cd = mass[idx(i, j, ST_D)];
            if cd > 0.0 {
                if i == lx {
                    total_eof += cd * 0.75;
                    if j < max_ly {
                        mass[idx(lx, j + 1, ST_POST_Y)] += cd * 0.25;
                    }
                } else {
                    mass[idx(i + 1, j, ST_D)] += cd * p_indel_ext;
                    if j < max_ly {
                        mass[idx(i + 1, j + 1, ST_M)] += cd * p_indel_exit;
                    }
                }
            }
            if i >= 1 && i < lx && j >= 1 {
                let ci = mass[idx(i, j, ST_I)];
                if ci > 0.0 && j < max_ly {
                    mass[idx(i, j + 1, ST_I)] += ci * p_indel_ext;
                    mass[idx(i + 1, j + 1, ST_M)] += ci * p_indel_exit;
                }
            }
            let cbx = mass[idx(i, j, ST_BX)];
            if cbx > 0.0 {
                if i == lx {
                    if j < max_ly {
                        mass[idx(lx, j + 1, ST_BY)] += cbx;
                    }
                } else {
                    mass[idx(i + 1, j, ST_BX)] += cbx * p_bx_ext;
                    if j < max_ly {
                        mass[idx(i, j + 1, ST_BY)] += cbx * p_bx_to_by;
                    }
                }
            }
            let cby = mass[idx(i, j, ST_BY)];
            if cby > 0.0 {
                if j < max_ly {
                    mass[idx(i, j + 1, ST_BY)] += cby * p_by_ext;
                }
                if i == lx {
                    total_eof += cby * p_by_exit * 0.75;
                    if j < max_ly {
                        mass[idx(lx, j + 1, ST_POST_Y)] += cby * p_by_exit * 0.25;
                    }
                } else if j < max_ly {
                    mass[idx(i + 1, j + 1, ST_M)] += cby * p_by_exit;
                }
            }
            if i == lx && j >= 1 {
                let cp = mass[idx(lx, j, ST_POST_Y)];
                if cp > 0.0 {
                    total_eof += cp * one_minus_eta_y;
                    if j < max_ly {
                        mass[idx(lx, j + 1, ST_POST_Y)] += cp * eta_y;
                    }
                }
            }
        }
    }
    total_eof
}

/// Audit and repair coding-sequence frameshifts (including paired compensatory +-1 nt excursions)
/// against a clean reference using glocal terminal borders and adaptive sister divergence.
pub fn attempt_frameshift_repair_with_margin(
    seq_ref: &EncodedSeq,
    seq_target: &EncodedSeq,
    gc: &GeneticCode,
    min_gain_margin: f64,
) -> Option<(EncodedSeq, Vec<FrameshiftEvent>, f64)> {
    if seq_ref.mode != SeqMode::Codon || seq_target.mode != SeqMode::Codon {
        return None;
    }
    let nts = &seq_target.nt_tokens;
    let masks = &seq_target.nt_masks;
    let raw_chars = &seq_target.nt_raw_chars;
    let l_nt = nts.len();
    let lx = seq_ref.len();
    if lx < 10 || l_nt < 30 {
        return None;
    }

    // 1. Estimate sister closeness from unambiguous nucleotide 9-mers for adaptive codon log-odds
    let extract_nt9 = |tok_s: &[u8], msk_s: &[u8], step: usize| -> Vec<u32> {
        if tok_s.len() < 9 || msk_s.len() != tok_s.len() {
            return Vec::new();
        }
        let mut out = Vec::new();
        let mut k = 0usize;
        while k + 9 <= tok_s.len() {
            let mut valid = true;
            let mut h = 0u32;
            for off in 0..9 {
                if msk_s[k + off].count_ones() != 1 {
                    valid = false;
                    break;
                }
                h = (h << 2) | (tok_s[k + off] as u32 & 3);
            }
            if valid {
                out.push(h);
            }
            k += step;
        }
        out.sort_unstable();
        out.dedup();
        out
    };
    let r9 = extract_nt9(&seq_ref.nt_tokens, &seq_ref.nt_masks, 3);
    let t9 = extract_nt9(nts, masks, 1);
    let mut shared9 = 0usize;
    {
        let (mut a, mut b) = (0usize, 0usize);
        while a < r9.len() && b < t9.len() {
            match r9[a].cmp(&t9[b]) {
                std::cmp::Ordering::Less => a += 1,
                std::cmp::Ordering::Greater => b += 1,
                std::cmp::Ordering::Equal => {
                    shared9 += 1;
                    a += 1;
                    b += 1;
                }
            }
        }
    }
    let ov9 = shared9 as f64 / (r9.len().min(t9.len()).max(1) as f64);
    let (t_aa, t_syn) = if ov9 >= 0.40 {
        (0.04, 0.12)
    } else if ov9 >= 0.15 {
        (0.10, 0.32)
    } else {
        (0.25, 0.80)
    };

    let n_sense = gc.num_sense;
    let full_sense_mask: u64 = (1u64 << n_sense) - 1;
    let bg_pi = default_background_pi(SeqMode::Codon, gc);
    let p_mat = build_telescoping_codon_matrix(t_aa, t_syn, &bg_pi, gc);
    let mut log_odds = vec![0.0f64; n_sense * n_sense];
    for c1 in 0..n_sense {
        for c2 in 0..n_sense {
            log_odds[c1 * n_sense + c2] = (p_mat[c1 * n_sense + c2] / bg_pi[c2]).ln();
        }
    }

    let gap_pen = -4.2f64;
    let overhang_pen = -0.25f64;
    let fs_cost_nats = (l_nt as f64).ln() + 4.0f64.ln() + 6.0;
    let fs_single_pen = -fs_cost_nats;
    let allow_paired_excursions = ov9 >= 0.10;
    let exc_open_pen = if allow_paired_excursions {
        -((l_nt as f64).ln() + std::f64::consts::LN_2 + 4.5)
    } else {
        NEG_INF
    };
    let exc_close_pen = -3.0f64;
    let exc_step_pen = -0.30f64;
    let term_clip_pen = -3.0f64.ln();
    let stop_pen = -11.0f64;
    let ln4 = 4.0f64.ln();

    let mut tar_smask = vec![0u64; l_nt + 1];
    for u in 3..=l_nt {
        tar_smask[u] = gc.resolve_masks_to_sense_mask(masks[u - 3], masks[u - 2], masks[u - 1]);
    }
    let ins_stop_pen = |u: usize| -> f64 {
        if tar_smask[u] != 0 || u == l_nt {
            0.0
        } else {
            stop_pen
        }
    };

    let score_smasks = |rmask: u64, tmask: u64| -> f64 {
        if tmask == 0 {
            return stop_pen;
        }
        if rmask == 0 || rmask == full_sense_mask || tmask == full_sense_mask {
            return 0.0;
        }
        let rc = rmask.count_ones();
        let tc = tmask.count_ones();
        if rc == 1 && tc == 1 {
            let cr = rmask.trailing_zeros() as usize;
            let ct = tmask.trailing_zeros() as usize;
            return log_odds[cr * n_sense + ct];
        }
        if rc == 1 {
            let cr = rmask.trailing_zeros() as usize;
            let base = cr * n_sense;
            let mut sum = 0.0;
            let mut mm = tmask;
            while mm != 0 {
                let ct = mm.trailing_zeros() as usize;
                mm &= mm - 1;
                sum += log_odds[base + ct];
            }
            return sum / (tc as f64);
        }
        let mut sum = 0.0;
        let mut mr = rmask;
        while mr != 0 {
            let cr = mr.trailing_zeros() as usize;
            mr &= mr - 1;
            let base = cr * n_sense;
            let mut mt = tmask;
            while mt != 0 {
                let ct = mt.trailing_zeros() as usize;
                mt &= mt - 1;
                sum += log_odds[base + ct];
            }
        }
        sum / ((rc * tc) as f64)
    };

    let m2_score = |rmask: u64, u: usize| -> f64 {
        let ma = masks[u - 2];
        let mb = masks[u - 1];
        if rmask == full_sense_mask || (ma == 0xF && mb == 0xF) {
            return -1.0;
        }
        let mut best = stop_pen;
        for (c0, c1, c2) in [(ma, mb, 0xF), (ma, 0xF, mb), (0xF, ma, mb)] {
            let sm = gc.resolve_masks_to_sense_mask(c0, c1, c2);
            let sc = if sm == 0 {
                stop_pen
            } else if sm == full_sense_mask {
                0.0
            } else {
                let mut m_best = f64::NEG_INFINITY;
                let mut mm = sm;
                while mm != 0 {
                    let ct = mm.trailing_zeros() as usize;
                    mm &= mm - 1;
                    let v = score_smasks(rmask, 1u64 << ct);
                    if v > m_best {
                        m_best = v;
                    }
                }
                m_best - ln4
            };
            if sc > best {
                best = sc;
            }
        }
        best - 0.8
    };

    let m4_score = |rmask: u64, u: usize| -> (f64, u8) {
        if rmask == full_sense_mask {
            return (0.0, 0);
        }
        let (m0, m1, m2, m3) = (masks[u - 4], masks[u - 3], masks[u - 2], masks[u - 1]);
        let mut best = f64::NEG_INFINITY;
        let mut best_k = 0u8;
        let triples = [(0u8, m1, m2, m3), (1u8, m0, m2, m3), (2u8, m0, m1, m3)];
        for (k_skip, a, b, c) in triples {
            let sm = gc.resolve_masks_to_sense_mask(a, b, c);
            let sc = score_smasks(rmask, sm);
            if sc > best {
                best = sc;
                best_k = k_skip;
            }
        }
        (best, best_k)
    };

    let w = l_nt + 1;
    let row_stride = w * 3;
    let mut dp_prev = vec![NEG_INF; row_stride];
    let mut dp_cur = vec![NEG_INF; row_stride];
    // bt packing: (prev_state << 5) | action (actions: 1=Match3, 2=DelRef, 3=InsTar3, 4=Skip1, 5=Skip2, 6=DelCod2, 7..=9=InsCod4(k_skip=action-7))
    let mut bt = vec![0u8; (lx + 1) * row_stride];

    dp_prev[0] = 0.0;
    for u in 1..=l_nt {
        if u >= 3 {
            let cand = dp_prev[(u - 3) * 3] + overhang_pen + ins_stop_pen(u);
            if cand > dp_prev[u * 3] {
                dp_prev[u * 3] = cand;
                bt[u * 3] = 3;
            }
        }
        let p_step = if u <= 2 { term_clip_pen } else { fs_single_pen };
        if dp_prev[(u - 1) * 3] + p_step > dp_prev[u * 3] {
            dp_prev[u * 3] = dp_prev[(u - 1) * 3] + p_step;
            bt[u * 3] = 4;
        }
    }

    let u_inframe = 3 * (l_nt / 3);
    for i in 1..=lx {
        dp_cur.fill(NEG_INF);
        dp_cur[0] = dp_prev[0] + overhang_pen;
        bt[i * row_stride] = 2;

        let rmask = seq_ref.ambig_masks[i - 1];
        let row_bt_off = i * row_stride;

        for u in 1..=l_nt {
            let del_pen = if u == l_nt || u == u_inframe {
                overhang_pen
            } else {
                gap_pen
            };
            let is_term_1 = (u <= 2) || (i == lx && u >= l_nt - 1);
            let is_term_2 = (u == 2) || (i == lx && u == l_nt);
            let m3_sc = if u >= 3 {
                score_smasks(rmask, tar_smask[u])
            } else {
                NEG_INF
            };
            let sc2 = if u >= 2 {
                m2_score(rmask, u)
            } else {
                NEG_INF
            };
            let (sc4, k_skip4) = if u >= 4 {
                m4_score(rmask, u)
            } else {
                (NEG_INF, 0)
            };

            // State 0: S_BASE
            let mut best0 = dp_prev[u * 3] + del_pen;
            let mut code0 = 2u8;

            let cand_s1 = dp_cur[(u - 1) * 3] + if is_term_1 { term_clip_pen } else { fs_single_pen };
            if cand_s1 > best0 {
                best0 = cand_s1;
                code0 = 4;
            }
            if !is_term_1 {
                let cand_close_m1 = dp_cur[(u - 1) * 3 + 2] + exc_close_pen;
                if cand_close_m1 > best0 {
                    best0 = cand_close_m1;
                    code0 = (2 << 5) | 4;
                }
            }
            if u >= 2 {
                let cand_s2 =
                    dp_cur[(u - 2) * 3] + if is_term_2 { term_clip_pen } else { fs_single_pen };
                if cand_s2 > best0 {
                    best0 = cand_s2;
                    code0 = 5;
                }
                if !is_term_2 {
                    let cand_close_p1_s2 = dp_cur[(u - 2) * 3 + 1] + exc_close_pen;
                    if cand_close_p1_s2 > best0 {
                        best0 = cand_close_p1_s2;
                        code0 = (1 << 5) | 5;
                    }
                }
                let cand_dc2_single = dp_prev[(u - 2) * 3] + sc2 + fs_single_pen;
                if cand_dc2_single > best0 {
                    best0 = cand_dc2_single;
                    code0 = 6;
                }
                let cand_close_p1_dc2 = dp_prev[(u - 2) * 3 + 1] + sc2 + exc_close_pen;
                if cand_close_p1_dc2 > best0 {
                    best0 = cand_close_p1_dc2;
                    code0 = (1 << 5) | 6;
                }
            }
            if u >= 3 {
                let cand_m = dp_prev[(u - 3) * 3] + m3_sc;
                if cand_m > best0 {
                    best0 = cand_m;
                    code0 = 1;
                }
                let ins_pen = (if i == lx { overhang_pen } else { gap_pen }) + ins_stop_pen(u);
                let cand_ins = dp_cur[(u - 3) * 3] + ins_pen;
                if cand_ins > best0 {
                    best0 = cand_ins;
                    code0 = 3;
                }
            }
            if u >= 4 {
                let cand_ic4_single = dp_prev[(u - 4) * 3] + sc4 + fs_single_pen;
                if cand_ic4_single > best0 {
                    best0 = cand_ic4_single;
                    code0 = 7 + k_skip4;
                }
                let cand_close_m1_ic4 = dp_prev[(u - 4) * 3 + 2] + sc4 + exc_close_pen;
                if cand_close_m1_ic4 > best0 {
                    best0 = cand_close_m1_ic4;
                    code0 = (2 << 5) | (7 + k_skip4);
                }
            }

            // State 1: S_EXC_P1 (+1 nt excursion)
            let mut best1 = NEG_INF;
            let mut code1 = 0u8;
            if u >= 3 && dp_prev[(u - 3) * 3 + 1] > NEG_INF * 0.5 {
                let cand_ext = dp_prev[(u - 3) * 3 + 1] + m3_sc + exc_step_pen;
                if cand_ext > best1 {
                    best1 = cand_ext;
                    code1 = (1 << 5) | 1;
                }
            }
            if !is_term_1 {
                let cand_open_s1 = dp_cur[(u - 1) * 3] + exc_open_pen;
                if cand_open_s1 > best1 {
                    best1 = cand_open_s1;
                    code1 = 4;
                }
            }
            if u >= 4 {
                let cand_open_ic4 = dp_prev[(u - 4) * 3] + sc4 + exc_open_pen;
                if cand_open_ic4 > best1 {
                    best1 = cand_open_ic4;
                    code1 = 7 + k_skip4;
                }
            }

            // State 2: S_EXC_M1 (-1 nt excursion)
            let mut best2 = NEG_INF;
            let mut code2 = 0u8;
            if u >= 3 && dp_prev[(u - 3) * 3 + 2] > NEG_INF * 0.5 {
                let cand_ext = dp_prev[(u - 3) * 3 + 2] + m3_sc + exc_step_pen;
                if cand_ext > best2 {
                    best2 = cand_ext;
                    code2 = (2 << 5) | 1;
                }
            }
            if u >= 2 {
                if !is_term_2 {
                    let cand_open_s2 = dp_cur[(u - 2) * 3] + exc_open_pen;
                    if cand_open_s2 > best2 {
                        best2 = cand_open_s2;
                        code2 = 5;
                    }
                }
                let cand_open_dc2 = dp_prev[(u - 2) * 3] + sc2 + exc_open_pen;
                if cand_open_dc2 > best2 {
                    best2 = cand_open_dc2;
                    code2 = 6;
                }
            }

            dp_cur[u * 3] = best0;
            dp_cur[u * 3 + 1] = best1;
            dp_cur[u * 3 + 2] = best2;
            bt[row_bt_off + u * 3] = code0;
            bt[row_bt_off + u * 3 + 1] = code1;
            bt[row_bt_off + u * 3 + 2] = code2;
        }
        std::mem::swap(&mut dp_prev, &mut dp_cur);
    }

    // In-frame baseline (rolling 1-row DP using identical ambiguity-aware `score_smasks`)
    let w0 = l_nt / 3 + 1;
    let mut dp0_prev = vec![NEG_INF; w0];
    let mut dp0_cur = vec![NEG_INF; w0];
    dp0_prev[0] = 0.0;
    for j in 1..w0 {
        let u = 3 * j;
        dp0_prev[j] = dp0_prev[j - 1] + overhang_pen + ins_stop_pen(u);
    }
    for i in 1..=lx {
        dp0_cur[0] = dp0_prev[0] + overhang_pen;
        let rmask = seq_ref.ambig_masks[i - 1];
        for j in 1..w0 {
            let u = 3 * j;
            let m_sc = score_smasks(rmask, tar_smask[u]);
            let del_pen = if j == w0 - 1 { overhang_pen } else { gap_pen };
            let ins_pen = (if i == lx { overhang_pen } else { gap_pen }) + ins_stop_pen(u);
            dp0_cur[j] = (dp0_prev[j] + del_pen)
                .max(dp0_cur[j - 1] + ins_pen)
                .max(dp0_prev[j - 1] + m_sc);
        }
        std::mem::swap(&mut dp0_prev, &mut dp0_cur);
    }

    let score_fs = dp_prev[l_nt * 3];
    let score_inframe = if u_inframe == l_nt {
        dp0_prev[w0 - 1]
    } else {
        dp0_prev[w0 - 1] - 2.0
    };

    if score_fs <= score_inframe + min_gain_margin {
        return None;
    }

    let push_codon_at = |p0: usize,
                         p1: usize,
                         p2: usize,
                         out: &mut Vec<(u8, u64, [u8; 3], [u8; 3], String)>|
     -> bool {
        let smask = gc.resolve_masks_to_sense_mask(masks[p0], masks[p1], masks[p2]);
        if smask == 0 {
            return false;
        }
        let s = smask.trailing_zeros() as u8;
        let cmasks = [masks[p0], masks[p1], masks[p2]];
        let cchars = if raw_chars.len() > p2 {
            [raw_chars[p0], raw_chars[p1], raw_chars[p2]]
        } else {
            let fallback = gc.codon_to_string(s);
            let fb = fallback.as_bytes();
            [fb[0], fb[1], fb[2]]
        };
        let s_str = String::from_utf8(vec![cchars[0], cchars[1], cchars[2]]).unwrap();
        out.push((s, smask, cmasks, cchars, s_str));
        true
    };

    let mut cur_i = lx;
    let mut cur_u = l_nt;
    let mut cur_s = 0usize;
    let mut rev_codons: Vec<(u8, u64, [u8; 3], [u8; 3], String)> = Vec::new();
    let mut rev_fs: Vec<(usize, u8, bool)> = Vec::new(); // (nt_pos, shift, is_paired_exc_event)

    while cur_i > 0 || cur_u > 0 {
        let packed = bt[cur_i * row_stride + cur_u * 3 + cur_s];
        let prev_s = (packed >> 5) as usize;
        let action = packed & 31;
        let is_paired = prev_s != cur_s;
        match action {
            1 => {
                if cur_u < 3 {
                    return None;
                }
                if !push_codon_at(cur_u - 3, cur_u - 2, cur_u - 1, &mut rev_codons)
                    && cur_u < l_nt
                {
                    return None;
                }
                cur_i -= 1;
                cur_u -= 3;
            }
            2 => {
                cur_i -= 1;
            }
            3 => {
                if cur_u < 3 {
                    return None;
                }
                if !push_codon_at(cur_u - 3, cur_u - 2, cur_u - 1, &mut rev_codons)
                    && cur_u < l_nt
                {
                    return None;
                }
                cur_u -= 3;
            }
            4 => {
                rev_fs.push((cur_u - 1, 1, is_paired));
                cur_u -= 1;
            }
            5 => {
                rev_fs.push((cur_u - 2, 2, is_paired));
                cur_u -= 2;
            }
            6 => {
                rev_fs.push((cur_u - 2, 2, is_paired));
                cur_i -= 1;
                cur_u -= 2;
            }
            7..=9 => {
                let k_skip = (action - 7) as usize;
                let base_u = cur_u - 4;
                rev_fs.push((base_u + k_skip, 1, is_paired));
                let mut surv = [0usize; 3];
                let mut si = 0usize;
                for k in 0..4 {
                    if k != k_skip {
                        surv[si] = base_u + k;
                        si += 1;
                    }
                }
                if !push_codon_at(surv[0], surv[1], surv[2], &mut rev_codons) {
                    return None;
                }
                cur_i -= 1;
                cur_u -= 4;
            }
            _ => break,
        }
        cur_s = prev_s;
    }

    if rev_fs.is_empty() {
        return None;
    }

    rev_codons.reverse();
    rev_fs.reverse();

    // Separate terminal 5'/3' partial-codon amplicon phase offsets from genuine interior frameshifts
    let interior_fs: Vec<(usize, u8, bool)> = rev_fs
        .iter()
        .copied()
        .filter(|&(nt_pos, shift, _)| nt_pos >= 2 && nt_pos + (shift as usize) + 2 < l_nt)
        .collect();

    let n_paired_events = interior_fs.iter().filter(|x| x.2).count();
    let n_comp_pairs = n_paired_events / 2;
    let mut exact_fs_penalty = if interior_fs.is_empty() {
        3.0f64.ln()
    } else {
        let mut p = elias_rissanen_nats(interior_fs.len());
        p += (n_comp_pairs as f64) * ((l_nt as f64).ln() + std::f64::consts::LN_2 + 7.5);
        let mut paired_rem = n_comp_pairs * 2;
        for &(_, shift, is_p) in &interior_fs {
            if is_p && paired_rem > 0 {
                paired_rem -= 1;
            } else {
                p += (l_nt as f64).ln() + std::f64::consts::LN_2 + (shift as f64) * 4.0f64.ln();
            }
        }
        p
    };
    if interior_fs.len() < rev_fs.len() && !interior_fs.is_empty() {
        exact_fs_penalty += 3.0f64.ln();
    }

    let denom_ev = interior_fs.len().max(1) as f64;
    let net_gain_each = (score_fs - score_inframe) / denom_ev;
    let events: Vec<FrameshiftEvent> = interior_fs
        .into_iter()
        .map(|(nt_pos, shift, _)| FrameshiftEvent {
            seq_id: seq_target.id.clone(),
            approx_codon_pos: nt_pos / 3,
            nt_pos,
            phase_shift: shift,
            mdl_gain_nats: net_gain_each,
        })
        .collect();

    let mut rep_tokens = Vec::with_capacity(rev_codons.len());
    let mut rep_ambig_masks = Vec::with_capacity(rev_codons.len());
    let mut rep_sym_proj = Vec::with_capacity(rev_codons.len());
    let mut rep_token_strings = Vec::with_capacity(rev_codons.len());
    let mut rep_has_ambig = false;
    let mut rep_non_ambig_len = 0usize;
    let mut rep_nt_tokens = Vec::with_capacity(3 * rev_codons.len());
    let mut rep_nt_masks = Vec::with_capacity(3 * rev_codons.len());
    let mut rep_nt_chars = Vec::with_capacity(3 * rev_codons.len());
    for (s, smask, cmasks, cchars, s_str) in rev_codons {
        rep_tokens.push(s);
        rep_ambig_masks.push(smask);
        if smask.count_ones() > 1 {
            rep_has_ambig = true;
        }
        let aa_m = gc.sense_mask_to_aa_mask(smask);
        if aa_m.count_ones() == 1 {
            rep_sym_proj.push(aa_m.trailing_zeros() as u8);
            rep_non_ambig_len += 1;
        } else {
            rep_sym_proj.push(255);
        }
        rep_token_strings.push(s_str);
        for k in 0..3 {
            rep_nt_masks.push(cmasks[k]);
            rep_nt_tokens.push(cmasks[k].trailing_zeros() as u8);
            rep_nt_chars.push(cchars[k]);
        }
    }

    let repaired = EncodedSeq {
        id: seq_target.id.clone(),
        mode: SeqMode::Codon,
        alpha_size: n_sense,
        tokens: rep_tokens,
        ambig_masks: rep_ambig_masks,
        sym_proj: rep_sym_proj,
        has_ambig: rep_has_ambig,
        non_ambig_len: rep_non_ambig_len,
        nt_tokens: rep_nt_tokens,
        nt_masks: rep_nt_masks,
        nt_raw_chars: rep_nt_chars,
        token_strings: rep_token_strings,
        has_unambiguous_internal_stop: false,
    };

    Some((repaired, events, exact_fs_penalty))
}

pub fn attempt_frameshift_repair(
    seq_ref: &EncodedSeq,
    seq_target: &EncodedSeq,
    gc: &GeneticCode,
) -> Option<(EncodedSeq, Vec<FrameshiftEvent>, f64)> {
    attempt_frameshift_repair_with_margin(seq_ref, seq_target, gc, 2.0)
}

pub(crate) fn summarize_dna_bubble_composition(
    counts_x: &[usize; 4],
    counts_y: &[usize; 4],
) -> (Option<String>, Option<f64>, Option<f64>, Option<f64>) {
    let nx: usize = counts_x.iter().sum();
    let ny: usize = counts_y.iter().sum();
    if nx == 0 || ny == 0 {
        return (None, None, None, None);
    }
    let mut joint = [0usize; 4];
    for k in 0..4 {
        joint[k] = counts_x[k] + counts_y[k];
    }
    let alpha4 = [1.0f64; 4];
    let gain = exchangeable_dirichlet_seq_ln_prob(&joint, &alpha4)
        - exchangeable_dirichlet_seq_ln_prob(counts_x, &alpha4)
        - exchangeable_dirichlet_seq_ln_prob(counts_y, &alpha4);
    let tot = nx + ny;
    let at_tot = joint[0] + joint[3];
    let gc_tot = joint[1] + joint[2];
    let pur_tot = joint[0] + joint[2];
    let label = if at_tot * 10 >= tot * 6 {
        "AT-rich(AT)"
    } else if gc_tot * 10 >= tot * 6 {
        "GC-rich(GC)"
    } else if pur_tot * 10 >= tot * 6 {
        "Purine-rich(AG)"
    } else if (tot - pur_tot) * 10 >= tot * 6 {
        "Pyrimidine-rich(CT)"
    } else {
        "Balanced-NT(ACGT)"
    };
    let gc_x = ((counts_x[1] + counts_x[2]) as f64) / (nx as f64);
    let gc_y = ((counts_y[1] + counts_y[2]) as f64) / (ny as f64);
    let pur_x = ((counts_x[0] + counts_x[2]) as f64) / (nx as f64);
    let pur_y = ((counts_y[0] + counts_y[2]) as f64) / (ny as f64);
    (
        Some(label.to_string()),
        Some(gain),
        Some(gc_y - gc_x),
        Some(pur_y - pur_x),
    )
}

fn analyze_coarse_bubble_biophysics(
    seq_x: &EncodedSeq,
    seq_y: &EncodedSeq,
    range_x: (usize, usize),
    range_y: (usize, usize),
    _gc: &GeneticCode,
) -> (Option<String>, Option<f64>, Option<f64>, Option<f64>) {
    if range_x.1 <= range_x.0 || range_y.1 <= range_y.0 {
        return (None, None, None, None);
    }

    if seq_x.mode == SeqMode::Dna {
        let mut counts_x = [0usize; 4];
        let mut counts_y = [0usize; 4];
        for i in range_x.0..range_x.1 {
            let nt = seq_x.sym_proj[i] as usize;
            if nt < 4 {
                counts_x[nt] += 1;
            }
        }
        for j in range_y.0..range_y.1 {
            let nt = seq_y.sym_proj[j] as usize;
            if nt < 4 {
                counts_y[nt] += 1;
            }
        }
        return summarize_dna_bubble_composition(&counts_x, &counts_y);
    }

    let mut counts_x = [0usize; NUM_BIOPHYS_CLASSES];
    let mut counts_y = [0usize; NUM_BIOPHYS_CLASSES];
    let mut prop_x = [0.0f64; 3];
    let mut prop_y = [0.0f64; 3];
    let mut nx_eff = 0usize;
    let mut ny_eff = 0usize;

    for i in range_x.0..range_x.1 {
        let aa = seq_x.sym_proj[i] as usize;
        if aa >= 20 {
            continue;
        }
        counts_x[AA_TO_BIOPHYS[aa]] += 1;
        for p in 0..3 {
            prop_x[p] += AA_PROPERTIES[aa][p];
        }
        nx_eff += 1;
    }
    for j in range_y.0..range_y.1 {
        let aa = seq_y.sym_proj[j] as usize;
        if aa >= 20 {
            continue;
        }
        counts_y[AA_TO_BIOPHYS[aa]] += 1;
        for p in 0..3 {
            prop_y[p] += AA_PROPERTIES[aa][p];
        }
        ny_eff += 1;
    }

    if nx_eff == 0 || ny_eff == 0 {
        return (None, None, None, None);
    }

    let mut joint_counts = [0usize; NUM_BIOPHYS_CLASSES];
    let mut dom_idx = 0usize;
    for c in 0..NUM_BIOPHYS_CLASSES {
        joint_counts[c] = counts_x[c] + counts_y[c];
        if joint_counts[c] > joint_counts[dom_idx] {
            dom_idx = c;
        }
    }

    let alpha = [1.0f64; NUM_BIOPHYS_CLASSES];
    let ln_indep = exchangeable_dirichlet_seq_ln_prob(&counts_x, &alpha)
        + exchangeable_dirichlet_seq_ln_prob(&counts_y, &alpha);
    let ln_shared = exchangeable_dirichlet_seq_ln_prob(&joint_counts, &alpha);
    let biophys_gain = ln_shared - ln_indep;

    let d_hydro = (prop_y[0] / ny_eff as f64) - (prop_x[0] / nx_eff as f64);
    let d_charge = (prop_y[2] / ny_eff as f64) - (prop_x[2] / nx_eff as f64);

    (
        Some(BIOPHYS_CLASS_NAMES[dom_idx].to_string()),
        Some(biophys_gain),
        Some(d_hydro),
        Some(d_charge),
    )
}

fn extract_multiscale_segments(
    seq_x: &EncodedSeq,
    seq_y: &EncodedSeq,
    fine_edges: &[SparseHomologyEdge],
    bubble_post_x: &[f64],
    bubble_post_y: &[f64],
    gc: &GeneticCode,
) -> Vec<MultiScaleSegment> {
    let lx = seq_x.len();
    let ly = seq_y.len();

    let mut strong: Vec<&SparseHomologyEdge> =
        fine_edges.iter().filter(|e| e.posterior >= 0.50).collect();
    strong.sort_by_key(|e| (e.idx_x, e.idx_y));

    let m = strong.len();
    let mut dp = vec![0.0f64; m];
    let mut prev = vec![None::<usize>; m];
    let mut best_end = None::<usize>;
    let mut best_val = 0.0;

    for a in 0..m {
        let mut val = strong[a].posterior;
        let mut p_idx = None;
        for b in 0..a {
            if strong[b].idx_x < strong[a].idx_x && strong[b].idx_y < strong[a].idx_y {
                if dp[b] + strong[a].posterior > val {
                    val = dp[b] + strong[a].posterior;
                    p_idx = Some(b);
                }
            }
        }
        dp[a] = val;
        prev[a] = p_idx;
        if val > best_val {
            best_val = val;
            best_end = Some(a);
        }
    }

    let mut anchors: Vec<(usize, usize, f64)> = Vec::new();
    let mut cur = best_end;
    while let Some(idx) = cur {
        anchors.push((strong[idx].idx_x, strong[idx].idx_y, strong[idx].posterior));
        cur = prev[idx];
    }
    anchors.reverse();

    if anchors.is_empty() {
        let mean_bx = if lx > 0 {
            bubble_post_x.iter().sum::<f64>() / lx as f64
        } else {
            0.0
        };
        let mean_by = if ly > 0 {
            bubble_post_y.iter().sum::<f64>() / ly as f64
        } else {
            0.0
        };
        let kind = if 0.5 * (mean_bx + mean_by) >= 0.50 {
            SegmentKind::CoarseBubble
        } else {
            SegmentKind::NonHomologous
        };
        let (dom, bgain, dh, dc) = if kind == SegmentKind::CoarseBubble {
            analyze_coarse_bubble_biophysics(seq_x, seq_y, (0, lx), (0, ly), gc)
        } else {
            (None, None, None, None)
        };
        return vec![MultiScaleSegment {
            kind,
            range_x: (0, lx),
            range_y: (0, ly),
            mean_posterior: 0.5 * (mean_bx + mean_by),
            dominant_biophys_class: dom,
            macro_biophys_gain_nats: bgain,
            delta_hydropathy: dh,
            delta_charge: dc,
        }];
    }

    let mut fine_blocks: Vec<(usize, usize, usize, usize, f64, usize)> = Vec::new();
    for &(ix, iy, p) in &anchors {
        if let Some(last) = fine_blocks.last_mut() {
            let gap_x = ix - last.1;
            let gap_y = iy - last.3;
            let bub_x: f64 = if gap_x > 0 {
                bubble_post_x[last.1..ix].iter().sum::<f64>() / (gap_x as f64)
            } else {
                0.0
            };
            let bub_y: f64 = if gap_y > 0 {
                bubble_post_y[last.3..iy].iter().sum::<f64>() / (gap_y as f64)
            } else {
                0.0
            };
            if (gap_x == 0 && gap_y == 0)
                || (gap_x.max(gap_y) <= 5 && gap_x.min(gap_y) <= 1 && bub_x.max(bub_y) < 0.45)
            {
                last.1 = ix + 1;
                last.3 = iy + 1;
                last.4 += p;
                last.5 += 1;
                continue;
            }
        }
        fine_blocks.push((ix, ix + 1, iy, iy + 1, p, 1));
    }

    let mut segments = Vec::new();
    let mut cx = 0usize;
    let mut cy = 0usize;

    for &(sx, ex, sy, ey, p_sum, cnt) in &fine_blocks {
        if sx > cx || sy > cy {
            let is_flank = cx == 0 && cy == 0;
            let mean_bx = if sx > cx {
                bubble_post_x[cx..sx].iter().sum::<f64>() / ((sx - cx) as f64)
            } else {
                0.0
            };
            let mean_by = if sy > cy {
                bubble_post_y[cy..sy].iter().sum::<f64>() / ((sy - cy) as f64)
            } else {
                0.0
            };
            let bub_score = 0.5 * (mean_bx + mean_by);
            let kind = if !is_flank && sx > cx && sy > cy && bub_score >= 0.25 {
                SegmentKind::CoarseBubble
            } else if bub_score >= 0.50 && sx > cx && sy > cy {
                SegmentKind::CoarseBubble
            } else {
                SegmentKind::NonHomologous
            };
            let (dom, bgain, dh, dc) = if kind == SegmentKind::CoarseBubble {
                analyze_coarse_bubble_biophysics(seq_x, seq_y, (cx, sx), (cy, sy), gc)
            } else {
                (None, None, None, None)
            };
            segments.push(MultiScaleSegment {
                kind,
                range_x: (cx, sx),
                range_y: (cy, sy),
                mean_posterior: bub_score,
                dominant_biophys_class: dom,
                macro_biophys_gain_nats: bgain,
                delta_hydropathy: dh,
                delta_charge: dc,
            });
        }

        segments.push(MultiScaleSegment {
            kind: SegmentKind::FineHomology,
            range_x: (sx, ex),
            range_y: (sy, ey),
            mean_posterior: p_sum / (cnt as f64),
            dominant_biophys_class: None,
            macro_biophys_gain_nats: None,
            delta_hydropathy: None,
            delta_charge: None,
        });
        cx = ex;
        cy = ey;
    }

    if cx < lx || cy < ly {
        let mean_bx = if lx > cx {
            bubble_post_x[cx..lx].iter().sum::<f64>() / ((lx - cx) as f64)
        } else {
            0.0
        };
        let mean_by = if ly > cy {
            bubble_post_y[cy..ly].iter().sum::<f64>() / ((ly - cy) as f64)
        } else {
            0.0
        };
        let bub_score = 0.5 * (mean_bx + mean_by);
        let kind = if lx > cx && ly > cy && bub_score >= 0.50 {
            SegmentKind::CoarseBubble
        } else {
            SegmentKind::NonHomologous
        };
        let (dom, bgain, dh, dc) = if kind == SegmentKind::CoarseBubble {
            analyze_coarse_bubble_biophysics(seq_x, seq_y, (cx, lx), (cy, ly), gc)
        } else {
            (None, None, None, None)
        };
        segments.push(MultiScaleSegment {
            kind,
            range_x: (cx, lx),
            range_y: (cy, ly),
            mean_posterior: bub_score,
            dominant_biophys_class: dom,
            macro_biophys_gain_nats: bgain,
            delta_hydropathy: dh,
            delta_charge: dc,
        });
    }

    segments
}

#[derive(Debug, Clone)]
pub struct PairForwardState {
    pub lx: usize,
    pub ly: usize,
    pub j_lo: Vec<u32>,
    pub j_hi: Vec<u32>,
    pub is_full_grid: bool,
    pub is_anchored: bool,
    pub anchor_count: usize,
    pub est_aa_div: f64,
    pub est_syn_div: f64,
    pub est_f_shared: f64,
    pub active_regimes: Vec<(usize, f64)>,
}

pub fn populate_pair_posteriors_prepared(
    res: &mut PairHomologyResult,
    state: &PairForwardState,
    px: &PreparedSeq,
    py: &PreparedSeq,
    gc: &GeneticCode,
    tables: &GlobalRegimeTables,
) {
    let lx = px.seq.len();
    let ly = py.seq.len();
    if lx == 0 || ly == 0 || state.active_regimes.is_empty() {
        res.fine_edges = Vec::new();
        res.bubble_post_x = vec![0.0; lx];
        res.bubble_post_y = vec![0.0; ly];
        res.fine_cov_x = Vec::new();
        res.fine_cov_y = Vec::new();
        res.segments = Vec::new();
        return;
    }

    let corridor_xy = CorridorBounds::from_bounds(
        state.lx,
        state.ly,
        state.j_lo.iter().map(|&v| v as usize).collect(),
        state.j_hi.iter().map(|&v| v as usize).collect(),
        state.is_full_grid,
        state.is_anchored,
        state.anchor_count,
        state.est_aa_div,
        state.est_syn_div,
        state.est_f_shared,
    );
    let corridor_yx = corridor_xy.transpose();
    let n_emit = corridor_xy.emit_offset[lx];
    let mut post_fine_banded = vec![0.0f64; n_emit];
    let mut bubble_post_x = vec![0.0f64; lx];
    let mut bubble_post_y = vec![0.0f64; ly];

    let r_pow_stride = Q_STEPS + 1;
    for &(g, wn) in &state.active_regimes {
        let r_locks = &tables.r_pow_locks[g * r_pow_stride..(g + 1) * r_pow_stride];
        let fb_xy = run_forward_backward_regime_corridor_cached(
            px,
            py,
            &corridor_xy,
            px.get_local_coarse(g, tables),
            &tables.copulas[g],
            &tables.ln_r_bases[g],
            Some(r_locks),
            Some(py.get_inv_z_q0(g, tables)),
            tables.regimes[g],
            PRIOR_MEAN_LEN,
            true,
        );
        let fb_yx = run_forward_backward_regime_corridor_cached(
            py,
            px,
            &corridor_yx,
            py.get_local_coarse(g, tables),
            &tables.copulas[g],
            &tables.ln_r_bases[g],
            Some(r_locks),
            Some(px.get_inv_z_q0(g, tables)),
            tables.regimes[g],
            PRIOR_MEAN_LEN,
            true,
        );

        if fb_xy.post_fine_banded.len() != n_emit
            || fb_yx.post_fine_banded.len() != corridor_yx.emit_offset[ly]
        {
            continue;
        }

        for i in 0..lx {
            bubble_post_x[i] += wn * 0.5 * (fb_xy.post_bx[i] + fb_yx.post_by[i]);
            let elo = corridor_xy.emit_lo[i];
            let ehi = corridor_xy.emit_hi[i];
            let eb = corridor_xy.emit_offset[i];
            for j in elo..ehi {
                let p_xy = fb_xy.post_fine_banded[eb + (j - elo)];
                let p_yx = fb_yx.post_fine_banded[corridor_yx.emit_idx(j, i)];
                post_fine_banded[eb + (j - elo)] += wn * 0.5 * (p_xy + p_yx);
            }
        }
        for j in 0..ly {
            bubble_post_y[j] += wn * 0.5 * (fb_xy.post_by[j] + fb_yx.post_bx[j]);
        }
    }

    let hpost = res.homology_posterior;
    for v in &mut bubble_post_x {
        *v *= hpost;
    }
    for v in &mut bubble_post_y {
        *v *= hpost;
    }

    let mut fine_edges = Vec::new();
    let mut sum_fine = 0.0f64;
    for i in 0..lx {
        let elo = corridor_xy.emit_lo[i];
        let ehi = corridor_xy.emit_hi[i];
        let eb = corridor_xy.emit_offset[i];
        for j in elo..ehi {
            let raw_p = post_fine_banded[eb + (j - elo)];
            let p = raw_p * hpost;
            if p > 0.0 {
                sum_fine += p;
            }
            if p >= 0.01 || raw_p >= 0.015 {
                fine_edges.push(SparseHomologyEdge {
                    idx_x: i,
                    idx_y: j,
                    posterior: p,
                    cond_posterior: raw_p,
                });
            }
        }
    }

    res.est_f_shared = (2.0 * sum_fine / ((lx + ly) as f64)).clamp(0.02, 1.0);
    res.segments = extract_multiscale_segments(
        &px.seq,
        &py.seq,
        &fine_edges,
        &bubble_post_x,
        &bubble_post_y,
        gc,
    );
    fine_edges.shrink_to_fit();
    res.fine_edges = fine_edges;
    res.bubble_post_x = bubble_post_x;
    res.bubble_post_y = bubble_post_y;
    res.fine_cov_x = Vec::new();
    res.fine_cov_y = Vec::new();
}

pub fn evaluate_pair_homology_prepared(
    px: &PreparedSeq,
    py: &PreparedSeq,
    gc: &GeneticCode,
    tables: &GlobalRegimeTables,
    compute_posteriors: bool,
    frameshifts: Vec<FrameshiftEvent>,
    fs_extra_nats: f64,
) -> (PairHomologyResult, Option<PairForwardState>) {
    assert_eq!(px.seq.mode, py.seq.mode);
    let mode = px.seq.mode;
    let lx = px.seq.len();
    let ly = py.seq.len();

    let l0_x = px.l0_nats;
    let l0_y = py.l0_nats;
    let null_joint_nats = l0_x + l0_y;

    if lx == 0 || ly == 0 {
        let res = PairHomologyResult {
            id_x: px.seq.id.clone(),
            id_y: py.seq.id.clone(),
            len_x: lx,
            len_y: ly,
            null_joint_nats,
            hom_joint_nats: null_joint_nats + 2.0,
            mdl_gain_nats: -2.0,
            homology_posterior: mdl_gain_to_prob(-2.0),
            est_t_primary: 5.0,
            est_t_syn: None,
            est_f_shared: 0.02,
            fine_edges: Vec::new(),
            bubble_post_x: vec![0.0; lx],
            bubble_post_y: vec![0.0; ly],
            fine_cov_x: Vec::new(),
            fine_cov_y: Vec::new(),
            segments: Vec::new(),
            frameshifts,
        };
        return (res, None);
    }

    let n_reg = tables.regimes.len();
    let ln_prior_reg = -(n_reg as f64).ln();
    let len_nats_y = geometric_len_nats(ly);
    let len_nats_x = geometric_len_nats(lx);

    let corridor_xy = build_anchor_corridor(px, py, gc);
    let corridor_yx = corridor_xy.transpose();
    let r_pow_stride = Q_STEPS + 1;
    let use_xy_primary = lx <= ly;

    let mut ln_p_xy_all = vec![None::<f64>; n_reg];
    let mut ln_p_yx_all = vec![None::<f64>; n_reg];

    let eval_xy = |g: usize, ln_p_xy_slot: &mut [Option<f64>]| {
        if ln_p_xy_slot[g].is_some() {
            return;
        }
        let reg = tables.regimes[g];
        // CRITICAL KRAFT INVARIANT: always divide by FULL (unbanded) path partition function!
        let ln_path_xy = compute_ln_path_partition(lx, ly, reg, PRIOR_MEAN_LEN);
        let r_locks = &tables.r_pow_locks[g * r_pow_stride..(g + 1) * r_pow_stride];
        let fb_xy = run_forward_backward_regime_corridor_cached(
            px,
            py,
            &corridor_xy,
            px.get_local_coarse(g, tables),
            &tables.copulas[g],
            &tables.ln_r_bases[g],
            Some(r_locks),
            Some(py.get_inv_z_q0(g, tables)),
            reg,
            PRIOR_MEAN_LEN,
            false,
        );
        ln_p_xy_slot[g] = Some(ln_prior_reg + fb_xy.ln_prob_y_given_x - ln_path_xy);
    };

    let eval_yx = |g: usize, ln_p_yx_slot: &mut [Option<f64>]| {
        if ln_p_yx_slot[g].is_some() {
            return;
        }
        let reg = tables.regimes[g];
        // CRITICAL KRAFT INVARIANT: always divide by FULL (unbanded) path partition function!
        let ln_path_yx = compute_ln_path_partition(ly, lx, reg, PRIOR_MEAN_LEN);
        let r_locks = &tables.r_pow_locks[g * r_pow_stride..(g + 1) * r_pow_stride];
        let fb_yx = run_forward_backward_regime_corridor_cached(
            py,
            px,
            &corridor_yx,
            py.get_local_coarse(g, tables),
            &tables.copulas[g],
            &tables.ln_r_bases[g],
            Some(r_locks),
            Some(px.get_inv_z_q0(g, tables)),
            reg,
            PRIOR_MEAN_LEN,
            false,
        );
        ln_p_yx_slot[g] = Some(ln_prior_reg + fb_yx.ln_prob_y_given_x - ln_path_yx);
    };

    // Fast non-homolog rejection screen when no anchor chain exists on long sequences
    if !corridor_xy.is_anchored && lx.min(ly) > 120 {
        let g_mid = n_reg / 2;
        let g_deep = n_reg.saturating_sub(2).max(g_mid);
        let dir_gain = |vals: &[f64]| -> f64 {
            let ln_marg = logsumexp_slice(vals);
            if use_xy_primary {
                l0_y + ln_marg - len_nats_y - fs_extra_nats
            } else {
                l0_x + ln_marg - len_nats_x - fs_extra_nats
            }
        };

        let p_deep = if use_xy_primary {
            eval_xy(g_deep, &mut ln_p_xy_all);
            ln_p_xy_all[g_deep].unwrap()
        } else {
            eval_yx(g_deep, &mut ln_p_yx_all);
            ln_p_yx_all[g_deep].unwrap()
        };
        let mut scr_vals = vec![p_deep];
        let mut scr_gain = dir_gain(&scr_vals);
        let reject = if scr_gain < -20.0 {
            true
        } else {
            if g_mid != g_deep {
                let p_mid = if use_xy_primary {
                    eval_xy(g_mid, &mut ln_p_xy_all);
                    ln_p_xy_all[g_mid].unwrap()
                } else {
                    eval_yx(g_mid, &mut ln_p_yx_all);
                    ln_p_yx_all[g_mid].unwrap()
                };
                scr_vals.push(p_mid);
                scr_gain = dir_gain(&scr_vals);
            }
            scr_gain < -12.0
        };
        if reject {
            let scr_hom_joint = null_joint_nats - scr_gain;
            let res = PairHomologyResult {
                id_x: px.seq.id.clone(),
                id_y: py.seq.id.clone(),
                len_x: lx,
                len_y: ly,
                null_joint_nats,
                hom_joint_nats: scr_hom_joint,
                mdl_gain_nats: scr_gain,
                homology_posterior: mdl_gain_to_prob(scr_gain),
                est_t_primary: tables.regimes[g_deep].t_primary,
                est_t_syn: if mode == SeqMode::Codon {
                    Some(tables.regimes[g_deep].t_syn)
                } else {
                    None
                },
                est_f_shared: 0.02,
                fine_edges: Vec::new(),
                bubble_post_x: if compute_posteriors {
                    vec![0.0; lx]
                } else {
                    Vec::new()
                },
                bubble_post_y: if compute_posteriors {
                    vec![0.0; ly]
                } else {
                    Vec::new()
                },
                fine_cov_x: Vec::new(),
                fine_cov_y: Vec::new(),
                segments: Vec::new(),
                frameshifts,
            };
            return (res, None);
        }
    }

    let nearly_equal_len = lx * 10 >= ly * 9 && ly * 10 >= lx * 9;
    if n_reg >= 6
        && corridor_xy.is_anchored
        && corridor_xy.anchor_count >= 30
        && corridor_xy.est_f_shared >= 0.88
        && corridor_xy.est_aa_div <= 0.12
        && nearly_equal_len
    {
        // Fast anchored peak-bracketing reserved strictly for full-length high-identity colinear sisters.
        // Preserves strict Kraft sub-probability via -ln(G).
        let ln_aa = corridor_xy.est_aa_div.ln();
        let ln_syn = corridor_xy.est_syn_div.ln();
        let mut ranked: Vec<(f64, usize)> = (0..n_reg)
            .map(|g| {
                let reg = tables.regimes[g];
                let d_aa = reg.t_primary.ln() - ln_aa;
                let d_syn = if mode == SeqMode::Codon {
                    reg.t_syn.ln() - ln_syn
                } else {
                    0.0
                };
                (d_aa * d_aa + 0.25 * d_syn * d_syn, g)
            })
            .collect();
        ranked.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        let seed_g: Vec<usize> = ranked.iter().take(3).map(|&(_, g)| g).collect();
        for &g in &seed_g {
            if use_xy_primary {
                eval_xy(g, &mut ln_p_xy_all);
            } else {
                eval_yx(g, &mut ln_p_yx_all);
            }
        }

        let primary_val = |g: usize, xy_s: &[Option<f64>], yx_s: &[Option<f64>]| -> Option<f64> {
            if use_xy_primary { xy_s[g] } else { yx_s[g] }
        };

        let mut g_best = seed_g[0];
        let mut v_best = f64::NEG_INFINITY;
        for &g in &seed_g {
            if let Some(v) = primary_val(g, &ln_p_xy_all, &ln_p_yx_all) {
                if v > v_best {
                    v_best = v;
                    g_best = g;
                }
            }
        }
        let neighs = [g_best.saturating_sub(1), (g_best + 1).min(n_reg - 1)];
        for &gn in &neighs {
            if use_xy_primary {
                eval_xy(gn, &mut ln_p_xy_all);
            } else {
                eval_yx(gn, &mut ln_p_yx_all);
            }
        }

        let mut prim_scored: Vec<(f64, usize)> = (0..n_reg)
            .filter_map(|g| primary_val(g, &ln_p_xy_all, &ln_p_yx_all).map(|v| (v, g)))
            .collect();
        prim_scored.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        let max_p = prim_scored.first().map(|x| x.0).unwrap_or(f64::NEG_INFINITY);
        let mut peak_cnt = 0usize;
        for &(vp, gp) in &prim_scored {
            if peak_cnt == 0 || (vp >= max_p - 10.0 && peak_cnt < 3) {
                if use_xy_primary {
                    eval_yx(gp, &mut ln_p_yx_all);
                } else {
                    eval_xy(gp, &mut ln_p_xy_all);
                }
                peak_cnt += 1;
            } else {
                break;
            }
        }
    } else if n_reg >= 6 {
        // Unified empirical likelihood-peak refinement in short->long direction, followed by
        // symmetric evaluation over the posterior peak (strict Kraft sub-probability via -ln(G))
        let skeleton: &[usize] = if mode == SeqMode::Codon {
            &[0, 2, 5, 7, 9, 11, 13]
        } else {
            &[1, 3, 5, 7]
        };
        let mut initial_seed = Vec::with_capacity(10);
        for &g_raw in skeleton {
            if g_raw < n_reg {
                initial_seed.push(g_raw);
            }
        }
        if corridor_xy.is_anchored {
            let ln_aa = corridor_xy.est_aa_div.ln();
            let ln_syn = corridor_xy.est_syn_div.ln();
            let mut ranked: Vec<(f64, usize)> = (0..n_reg)
                .map(|g| {
                    let reg = tables.regimes[g];
                    let d_aa = reg.t_primary.ln() - ln_aa;
                    let d_syn = if mode == SeqMode::Codon {
                        reg.t_syn.ln() - ln_syn
                    } else {
                        0.0
                    };
                    (d_aa * d_aa + 0.25 * d_syn * d_syn, g)
                })
                .collect();
            ranked.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            for &(_, g) in ranked.iter().take(2) {
                initial_seed.push(g);
            }
        }
        initial_seed.sort_unstable();
        initial_seed.dedup();
        for &g in &initial_seed {
            eval_xy(g, &mut ln_p_xy_all);
            eval_yx(g, &mut ln_p_yx_all);
        }

        let mut ranked_coarse: Vec<(f64, usize)> = (0..n_reg)
            .filter_map(|g| match (ln_p_xy_all[g], ln_p_yx_all[g]) {
                (Some(pxy), Some(pyx)) => Some((0.5 * (pxy + pyx), g)),
                _ => None,
            })
            .collect();
        ranked_coarse.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        let mut neighbors = Vec::with_capacity(4);
        for &(_, g_peak) in ranked_coarse.iter().take(2) {
            neighbors.push(g_peak.saturating_sub(1));
            neighbors.push((g_peak + 1).min(n_reg - 1));
        }
        neighbors.sort_unstable();
        neighbors.dedup();
        for g in neighbors {
            eval_xy(g, &mut ln_p_xy_all);
            eval_yx(g, &mut ln_p_yx_all);
        }
    } else {
        for g in 0..n_reg {
            eval_xy(g, &mut ln_p_xy_all);
            eval_yx(g, &mut ln_p_yx_all);
        }
    }

    let mut eval_xy_vals = Vec::with_capacity(n_reg);
    let mut eval_yx_vals = Vec::with_capacity(n_reg);
    let mut eval_g = Vec::with_capacity(n_reg);
    for g in 0..n_reg {
        if let (Some(pxy), Some(pyx)) = (ln_p_xy_all[g], ln_p_yx_all[g]) {
            eval_g.push(g);
            eval_xy_vals.push(pxy);
            eval_yx_vals.push(pyx);
        }
    }

    let ln_marg_y_given_x = logsumexp_slice(&eval_xy_vals);
    let ln_marg_x_given_y = logsumexp_slice(&eval_yx_vals);

    let hom_xy = l0_x + len_nats_y - ln_marg_y_given_x;
    let hom_yx = l0_y + len_nats_x - ln_marg_x_given_y;
    let hom_joint_nats = 0.5 * (hom_xy + hom_yx) + fs_extra_nats;
    let mdl_gain_nats = null_joint_nats - hom_joint_nats;
    let homology_posterior = mdl_gain_to_prob(mdl_gain_nats);

    let sym_reg_ln: Vec<f64> = (0..eval_g.len())
        .map(|k| 0.5 * (eval_xy_vals[k] + eval_yx_vals[k]))
        .collect();
    let sym_norm = logsumexp_slice(&sym_reg_ln);
    let mut cand_active: Vec<(usize, f64)> = Vec::new();
    let mut best_g = eval_g.first().copied().unwrap_or(0);
    let mut best_w = -1.0f64;
    let mut est_t_primary = 0.0;
    let mut est_t_syn_acc = 0.0;
    for k in 0..eval_g.len() {
        let g = eval_g[k];
        let w = (sym_reg_ln[k] - sym_norm).exp();
        est_t_primary += w * tables.regimes[g].t_primary;
        est_t_syn_acc += w * tables.regimes[g].t_syn;
        if w > best_w {
            best_w = w;
            best_g = g;
        }
        if w >= 0.03 {
            cand_active.push((g, w));
        }
    }
    cand_active.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let mut active_regimes: Vec<(usize, f64)> = Vec::with_capacity(3);
    let mut cum_w = 0.0f64;
    for &(g, w) in &cand_active {
        active_regimes.push((g, w));
        cum_w += w;
        if cum_w >= 0.985 || active_regimes.len() >= 3 {
            break;
        }
    }
    if active_regimes.is_empty() && !eval_g.is_empty() {
        active_regimes.push((best_g, best_w.max(1.0)));
    }
    let active_w_sum: f64 = active_regimes.iter().map(|x| x.1).sum();
    if active_w_sum > 0.0 {
        for (_, w) in &mut active_regimes {
            *w /= active_w_sum;
        }
    }

    let approx_f_shared = if corridor_xy.is_anchored {
        corridor_xy.est_f_shared
    } else {
        (mdl_gain_nats / l0_x.min(l0_y).max(1.0)).clamp(0.02, 1.0)
    };

    let mut res = PairHomologyResult {
        id_x: px.seq.id.clone(),
        id_y: py.seq.id.clone(),
        len_x: lx,
        len_y: ly,
        null_joint_nats,
        hom_joint_nats,
        mdl_gain_nats,
        homology_posterior,
        est_t_primary,
        est_t_syn: if mode == SeqMode::Codon {
            Some(est_t_syn_acc)
        } else {
            None
        },
        est_f_shared: approx_f_shared,
        fine_edges: Vec::new(),
        bubble_post_x: Vec::new(),
        bubble_post_y: Vec::new(),
        fine_cov_x: Vec::new(),
        fine_cov_y: Vec::new(),
        segments: Vec::new(),
        frameshifts,
    };

    let state = PairForwardState {
        lx: corridor_xy.lx,
        ly: corridor_xy.ly,
        j_lo: corridor_xy.j_lo.iter().map(|&v| v as u32).collect(),
        j_hi: corridor_xy.j_hi.iter().map(|&v| v as u32).collect(),
        is_full_grid: corridor_xy.is_full_grid,
        is_anchored: corridor_xy.is_anchored,
        anchor_count: corridor_xy.anchor_count,
        est_aa_div: corridor_xy.est_aa_div,
        est_syn_div: corridor_xy.est_syn_div,
        est_f_shared: corridor_xy.est_f_shared,
        active_regimes,
    };

    if compute_posteriors {
        populate_pair_posteriors_prepared(&mut res, &state, px, py, gc, tables);
    }

    (res, Some(state))
}

pub fn evaluate_pair_homology(
    seq_x_in: &EncodedSeq,
    seq_y_in: &EncodedSeq,
    gc: &GeneticCode,
    compute_posteriors: bool,
) -> PairHomologyResult {
    assert_eq!(seq_x_in.mode, seq_y_in.mode);
    let mode = seq_x_in.mode;

    let mut seq_x = seq_x_in.clone();
    let mut seq_y = seq_y_in.clone();
    let mut frameshifts = Vec::new();
    let mut fs_extra_nats = 0.0;

    if mode == SeqMode::Codon {
        let y_needs_audit =
            seq_y.nt_tokens.len() % 3 != 0 || seq_y.has_unambiguous_internal_stop;
        let x_needs_audit =
            seq_x.nt_tokens.len() % 3 != 0 || seq_x.has_unambiguous_internal_stop;

        if y_needs_audit {
            if let Some((rep_y, evs, pen)) = attempt_frameshift_repair(&seq_x, &seq_y, gc) {
                seq_y = rep_y;
                frameshifts.extend(evs);
                fs_extra_nats += pen;
            }
        }
        if x_needs_audit {
            if let Some((rep_x, evs, pen)) = attempt_frameshift_repair(&seq_y, &seq_x, gc) {
                seq_x = rep_x;
                frameshifts.extend(evs);
                fs_extra_nats += pen;
            }
        }
    }

    let tables = get_global_regime_tables(mode, gc);
    let px = PreparedSeq::new(seq_x, gc);
    let py = PreparedSeq::new(seq_y, gc);
    let (res, _) = evaluate_pair_homology_prepared(
        &px,
        &py,
        gc,
        &tables,
        compute_posteriors,
        frameshifts,
        fs_extra_nats,
    );
    res
}
