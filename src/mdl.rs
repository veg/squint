/// Core Minimum Description Length (MDL) & Kraft-tight probability primitives in natural units (nats).

pub const NEG_INF: f64 = f64::NEG_INFINITY;
pub const MAX_REPEAT_PERIOD: usize = 12;

/// Exact Elias-Rissanen universal prefix code over non-negative integers n >= 0,
/// satisfying \sum_{n=0}^{\infty} exp(-L_N(n)) == 1 identically (Bushwhacker Eq. 1).
#[inline]
pub fn elias_rissanen_nats(n: usize) -> f64 {
    if n == 0 {
        std::f64::consts::LN_2
    } else {
        let nf = n as f64;
        let p = 0.5 * std::f64::consts::LN_2 * (1.0 / (nf + 1.0).ln() - 1.0 / (nf + 2.0).ln());
        -p.ln()
    }
}

/// Numerically stable log(exp(a) + exp(b)) with exact IEEE-754 underflow cutoff.
#[inline(always)]
pub fn logsumexp2(a: f64, b: f64) -> f64 {
    let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
    if hi == NEG_INF {
        return NEG_INF;
    }
    let d = lo - hi;
    if d <= -45.0 {
        return hi;
    }
    hi + d.exp().ln_1p()
}

/// Numerically stable logsumexp across a slice.
#[inline]
pub fn logsumexp_slice(xs: &[f64]) -> f64 {
    let mut max_v = NEG_INF;
    for &x in xs {
        if x > max_v {
            max_v = x;
        }
    }
    if max_v == NEG_INF {
        return NEG_INF;
    }
    let mut sum = 0.0;
    for &x in xs {
        if x > NEG_INF {
            sum += (x - max_v).exp();
        }
    }
    max_v + sum.ln()
}

/// Logistic conversion from MDL gain (nats) to local posterior probability P = 1 / (1 + exp(-gain)).
#[inline]
pub fn mdl_gain_to_prob(gain_nats: f64) -> f64 {
    if gain_nats >= 40.0 {
        1.0
    } else if gain_nats <= -40.0 {
        0.0
    } else {
        1.0 / (1.0 + (-gain_nats).exp())
    }
}

/// Pure-Rust IEEE-754 high-precision Lanczos log-gamma ln(Gamma(z)) for z > 0.
#[inline]
pub fn lgamma(z: f64) -> f64 {
    const COEFFS: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if z < 0.5 {
        let pi = std::f64::consts::PI;
        return pi.ln() - (pi * z).sin().ln() - lgamma(1.0 - z);
    }
    let z = z - 1.0;
    let mut x = COEFFS[0];
    for (i, &c) in COEFFS.iter().enumerate().skip(1) {
        x += c / (z + i as f64);
    }
    let t = z + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (z + 0.5) * t.ln() - t + x.ln()
}

#[inline]
pub fn ln_binom(n: usize, k: usize) -> f64 {
    if k > n {
        return NEG_INF;
    }
    if k == 0 || k == n {
        return 0.0;
    }
    lgamma((n + 1) as f64) - lgamma((k + 1) as f64) - lgamma((n - k + 1) as f64)
}

/// Exact log-probability of an ordered string of length N = sum(counts) under a Dirichlet-Multinomial
/// Polya-urn distribution with concentration vector `alpha` (Kraft-tight over all K^N strings!).
/// Note: Because we encode the ordered sequence of tokens (not just unordered histogram),
/// we do NOT multiply by the multinomial coefficient N! / prod(c_k!)—making it an exact
/// exchangeable sequence distribution \sum_{y \in \Sigma^N} P_DM(y | alpha) == 1.
pub fn exchangeable_dirichlet_seq_ln_prob(counts: &[usize], alpha: &[f64]) -> f64 {
    debug_assert_eq!(counts.len(), alpha.len());
    let mut n_tot = 0usize;
    let mut alpha_0 = 0.0;
    for (&c, &a) in counts.iter().zip(alpha.iter()) {
        n_tot += c;
        alpha_0 += a;
    }
    if n_tot == 0 {
        return 0.0;
    }
    let mut ln_p = lgamma(alpha_0) - lgamma(alpha_0 + n_tot as f64);
    for (&c, &a) in counts.iter().zip(alpha.iter()) {
        if c > 0 {
            ln_p += lgamma(a + c as f64) - lgamma(a);
        }
    }
    ln_p
}

/// Fractional exchangeable Dirichlet-Multinomial log marginal likelihood for phylogenetically
/// de-replicated continuous residue counts.
pub fn exchangeable_dirichlet_frac_ln_prob(counts: &[f64], alpha: &[f64]) -> f64 {
    debug_assert_eq!(counts.len(), alpha.len());
    let mut n_tot = 0.0;
    let mut alpha_0 = 0.0;
    for (&c, &a) in counts.iter().zip(alpha.iter()) {
        n_tot += c;
        alpha_0 += a;
    }
    if n_tot <= 1e-12 {
        return 0.0;
    }
    let mut ln_p = lgamma(alpha_0) - lgamma(alpha_0 + n_tot);
    for (&c, &a) in counts.iter().zip(alpha.iter()) {
        if c > 1e-12 {
            ln_p += lgamma(a + c) - lgamma(a);
        }
    }
    ln_p
}

/// Causal single-sequence baseline compressor M_0(X):
/// Combines prequential Dirichlet-Multinomial composition learning with Fixed-Share
/// Bayesian switching across tandem slippage repeat periods p = 1..=MAX_REPEAT_PERIOD.
#[derive(Debug, Clone)]
pub struct SingleSeqNullProfile {
    pub alphabet_size: usize,
    pub len: usize,
    /// Total lossless self-information L_0(X) = L_N(|X|) + \sum_i -ln M_0(x_i | X_{<i}) in nats.
    pub total_nats: f64,
    /// Payload nats excluding length header: \sum_i -ln M_0(x_i | X_{<i}).
    pub payload_nats: f64,
    /// Per-position causal log-probability ln M_0(x_i | X_{<i}).
    pub site_ln_m0: Vec<f64>,
    /// Per-position causal repeat weight \rho_i \in [0, 1) strictly given prefix X_{<i}.
    pub site_repeat_weight: Vec<f64>,
    /// Bidirectional repeat occupancy \max(\rho_fwd(i), \rho_rev(i)) on a fully-known conditioning sequence X.
    /// Never used for causal Y-decoding; strictly used for conditioning on an already-transmitted parent X!
    pub ref_repeat_weight: Vec<f64>,
    /// Per-position full causal distribution M_0(. | X_{<i}) (length L * K).
    pub site_m0_dist: Vec<f64>,
    /// Per-position causal repeat-only distribution P_rep(. | X_{<i}) (length L * K).
    pub site_rep_dist: Vec<f64>,
    /// Smoothed empirical stationary distribution \tilde{\pi}^{(X)} over alphabet K.
    pub stationary_pi: Vec<f64>,
}

impl SingleSeqNullProfile {
    pub fn compute(tokens: &[u8], alphabet_size: usize, bg_pi: &[f64]) -> Self {
        let masks: Vec<u64> = tokens.iter().map(|&t| 1u64 << t).collect();
        Self::compute_with_masks(tokens, &masks, alphabet_size, bg_pi)
    }

    pub fn compute_with_masks(
        tokens: &[u8],
        ambig_masks: &[u64],
        alphabet_size: usize,
        bg_pi: &[f64],
    ) -> Self {
        assert_eq!(bg_pi.len(), alphabet_size);
        assert_eq!(tokens.len(), ambig_masks.len());
        let l = tokens.len();
        let k_f = alphabet_size as f64;
        let alpha_0 = (0.5 * k_f).max(2.0);
        let full_mask: u64 = if alphabet_size >= 64 {
            u64::MAX
        } else {
            (1u64 << alphabet_size) - 1
        };

        // Fixed-Share switching hyperparameters (purely causal, sum to 1 at every step):
        // Initial prior: 98% compositional, 2% split across repeat periods 1..=P.
        let p_max = MAX_REPEAT_PERIOD;
        let init_rep_total = 0.02;
        let switch_to_rep = 0.008; // cost ~4.8 nats to enter a repeat tract
        let stay_in_rep = 0.95; // expected repeat run ~20 units once active
        let rep_eps = 0.02; // smoothing inside repeat expert

        #[inline]
        fn fixed_share_step(
            weights: &mut [f64],
            post: &mut [f64],
            switch_to_rep: f64,
            stay_in_rep: f64,
            p_max: usize,
        ) {
            let post_sum: f64 = post.iter().sum();
            if post_sum > 0.0 {
                for w in post.iter_mut() {
                    *w /= post_sum;
                }
            } else {
                post.copy_from_slice(weights);
            }
            let post_comp = post[0];
            let post_rep_tot: f64 = post[1..].iter().sum();
            weights[0] = post_comp * (1.0 - switch_to_rep) + post_rep_tot * (1.0 - stay_in_rep);
            let enter_each = (post_comp * switch_to_rep) / (p_max as f64);
            for p in 1..=p_max {
                weights[p] = post[p] * stay_in_rep + enter_each;
            }
            let w_sum: f64 = weights.iter().sum();
            for w in weights.iter_mut() {
                *w /= w_sum;
            }
        }

        #[inline]
        fn period_mask_lik(
            obs_mask: u64,
            prev_mask: u64,
            full_mask: u64,
            lik_comp: f64,
            rep_eps: f64,
            k_f: f64,
        ) -> f64 {
            if prev_mask == full_mask {
                lik_comp
            } else {
                let n_obs = obs_mask.count_ones() as f64;
                let n_prev = prev_mask.count_ones() as f64;
                let overlap = (obs_mask & prev_mask).count_ones() as f64;
                n_obs * (rep_eps / k_f) + (overlap / n_prev) * (1.0 - rep_eps)
            }
        }

        let mut counts = vec![0.0f64; alphabet_size];
        let mut n_obs = 0.0f64;
        // Expert weights: index 0 = compositional expert, indices 1..=p_max = period-p repeat experts.
        let mut weights = vec![0.0f64; p_max + 1];
        weights[0] = 1.0 - init_rep_total;
        for p in 1..=p_max {
            weights[p] = init_rep_total / (p_max as f64);
        }

        let mut site_ln_m0 = Vec::with_capacity(l);
        let mut site_repeat_weight = Vec::with_capacity(l);
        let mut site_m0_dist = vec![0.0f64; l * alphabet_size];
        let mut site_rep_dist = vec![0.0f64; l * alphabet_size];
        let mut payload_nats = 0.0;

        let mut comp_dist = vec![0.0f64; alphabet_size];
        let mut post = vec![0.0f64; p_max + 1];

        for i in 0..l {
            // 1. Prequential compositional distribution P_comp(. | X_{<i})
            let denom = n_obs + alpha_0;
            for a in 0..alphabet_size {
                comp_dist[a] = (counts[a] + alpha_0 * bg_pi[a]) / denom;
            }

            // 2. Active repeat experts (for p > i or wildcard lag, period expert falls back to comp_dist so normalization holds!)
            let w_comp = weights[0];
            let w_rep_sum: f64 = weights[1..].iter().sum();

            let rep_slice = &mut site_rep_dist[i * alphabet_size..(i + 1) * alphabet_size];
            if w_rep_sum > 1e-15 && i > 0 {
                rep_slice.fill(0.0);
                for p in 1..=p_max {
                    let wp = weights[p] / w_rep_sum;
                    if p <= i && ambig_masks[i - p] != full_mask {
                        let pmask = ambig_masks[i - p];
                        let base = rep_eps / k_f;
                        for a in 0..alphabet_size {
                            rep_slice[a] += wp * base;
                        }
                        let pc = pmask.count_ones();
                        if pc == 1 {
                            rep_slice[tokens[i - p] as usize] += wp * (1.0 - rep_eps);
                        } else {
                            let share = wp * (1.0 - rep_eps) / (pc as f64);
                            let mut mm = pmask;
                            while mm != 0 {
                                let a = mm.trailing_zeros() as usize;
                                mm &= mm - 1;
                                rep_slice[a] += share;
                            }
                        }
                    } else {
                        for a in 0..alphabet_size {
                            rep_slice[a] += wp * comp_dist[a];
                        }
                    }
                }
            } else {
                rep_slice.copy_from_slice(&comp_dist);
            }

            let mask = ambig_masks[i];
            // Effective repeat predictability weight (0 at wildcard sites or i=0)
            let active_rep_mass: f64 = if mask == full_mask {
                0.0
            } else {
                (1..=p_max.min(i))
                    .filter(|&p| ambig_masks[i - p] != full_mask)
                    .map(|p| weights[p])
                    .sum()
            };
            site_repeat_weight.push(active_rep_mass.clamp(0.0, 0.99));

            // 3. Exact causal mixture M_0(. | X_{<i})
            let m0_slice = &mut site_m0_dist[i * alphabet_size..(i + 1) * alphabet_size];
            for a in 0..alphabet_size {
                m0_slice[a] = w_comp * comp_dist[a] + w_rep_sum * rep_slice[a];
            }

            // 4. Causal observation probability, Bayes update, and Fixed-Share step
            if mask.count_ones() == 1 {
                let obs = tokens[i] as usize;
                let p_obs = m0_slice[obs].max(1e-300);
                let ln_p = p_obs.ln();
                site_ln_m0.push(ln_p);
                payload_nats -= ln_p;

                post[0] = weights[0] * comp_dist[obs];
                for p in 1..=p_max {
                    let lik_p = if p <= i {
                        period_mask_lik(
                            mask,
                            ambig_masks[i - p],
                            full_mask,
                            comp_dist[obs],
                            rep_eps,
                            k_f,
                        )
                    } else {
                        comp_dist[obs]
                    };
                    post[p] = weights[p] * lik_p;
                }
                fixed_share_step(&mut weights, &mut post, switch_to_rep, stay_in_rep, p_max);
                counts[obs] += 1.0;
                n_obs += 1.0;
            } else if mask == full_mask {
                // Full wildcard: P_0(Sigma | X_{<i}) == 1.0 identically => 0.0 nats, zero composition/repeat drift
                site_ln_m0.push(0.0);
                post.copy_from_slice(&weights);
                fixed_share_step(&mut weights, &mut post, switch_to_rep, stay_in_rep, p_max);
            } else {
                // Partial IUPAC ambiguity: sum M_0 over compatible states
                let mut p_obs = 0.0;
                let mut lik_0 = 0.0;
                let mut mm = mask;
                while mm != 0 {
                    let a = mm.trailing_zeros() as usize;
                    mm &= mm - 1;
                    p_obs += m0_slice[a];
                    lik_0 += comp_dist[a];
                }
                let ln_p = p_obs.max(1e-300).ln();
                site_ln_m0.push(ln_p);
                payload_nats -= ln_p;

                post[0] = weights[0] * lik_0;
                for p in 1..=p_max {
                    let lik_p = if p <= i {
                        period_mask_lik(mask, ambig_masks[i - p], full_mask, lik_0, rep_eps, k_f)
                    } else {
                        lik_0
                    };
                    post[p] = weights[p] * lik_p;
                }
                fixed_share_step(&mut weights, &mut post, switch_to_rep, stay_in_rep, p_max);
                let inv_lik0 = 1.0 / lik_0.max(1e-300);
                let mut mm = mask;
                while mm != 0 {
                    let a = mm.trailing_zeros() as usize;
                    mm &= mm - 1;
                    counts[a] += comp_dist[a] * inv_lik0;
                }
                n_obs += 1.0;
            }
        }

        let mut stationary_pi = vec![0.0f64; alphabet_size];
        let tot_denom = n_obs + alpha_0;
        for a in 0..alphabet_size {
            stationary_pi[a] = (counts[a] + alpha_0 * bg_pi[a]) / tot_denom;
        }

        // Compute reverse-pass repeat weights on tokens.rev() so `ref_repeat_weight` captures
        // the start boundary of tandem repeats on an already-transmitted conditioning sequence X!
        let mut ref_repeat_weight = site_repeat_weight.clone();
        if l > 1 {
            let mut r_counts = vec![0.0f64; alphabet_size];
            let mut r_n_obs = 0.0f64;
            let mut r_weights = vec![0.0f64; p_max + 1];
            r_weights[0] = 1.0 - init_rep_total;
            for p in 1..=p_max {
                r_weights[p] = init_rep_total / (p_max as f64);
            }
            for step in 0..l {
                let orig_i = l - 1 - step;
                let mask = ambig_masks[orig_i];
                let r_denom = r_n_obs + alpha_0;

                if mask == full_mask {
                    post.copy_from_slice(&r_weights);
                    fixed_share_step(&mut r_weights, &mut post, switch_to_rep, stay_in_rep, p_max);
                    continue;
                }

                let active_r: f64 = (1..=p_max.min(step))
                    .filter(|&p| ambig_masks[orig_i + p] != full_mask)
                    .map(|p| r_weights[p])
                    .sum();
                if active_r > ref_repeat_weight[orig_i] {
                    ref_repeat_weight[orig_i] = active_r.clamp(0.0, 0.99);
                }

                if mask.count_ones() == 1 {
                    let obs = tokens[orig_i] as usize;
                    let r_comp_obs = (r_counts[obs] + alpha_0 * bg_pi[obs]) / r_denom;
                    post[0] = r_weights[0] * r_comp_obs;
                    for p in 1..=p_max {
                        let lik_p = if p <= step {
                            period_mask_lik(
                                mask,
                                ambig_masks[orig_i + p],
                                full_mask,
                                r_comp_obs,
                                rep_eps,
                                k_f,
                            )
                        } else {
                            r_comp_obs
                        };
                        post[p] = r_weights[p] * lik_p;
                    }
                    fixed_share_step(&mut r_weights, &mut post, switch_to_rep, stay_in_rep, p_max);
                    r_counts[obs] += 1.0;
                    r_n_obs += 1.0;
                } else {
                    let mut lik_0 = 0.0;
                    let mut mm = mask;
                    while mm != 0 {
                        let a = mm.trailing_zeros() as usize;
                        mm &= mm - 1;
                        lik_0 += (r_counts[a] + alpha_0 * bg_pi[a]) / r_denom;
                    }
                    post[0] = r_weights[0] * lik_0;
                    for p in 1..=p_max {
                        let lik_p = if p <= step {
                            period_mask_lik(
                                mask,
                                ambig_masks[orig_i + p],
                                full_mask,
                                lik_0,
                                rep_eps,
                                k_f,
                            )
                        } else {
                            lik_0
                        };
                        post[p] = r_weights[p] * lik_p;
                    }
                    fixed_share_step(&mut r_weights, &mut post, switch_to_rep, stay_in_rep, p_max);
                    let inv_lik0 = 1.0 / lik_0.max(1e-300);
                    let mut mm = mask;
                    while mm != 0 {
                        let a = mm.trailing_zeros() as usize;
                        mm &= mm - 1;
                        let cp = (r_counts[a] + alpha_0 * bg_pi[a]) / r_denom;
                        r_counts[a] += cp * inv_lik0;
                    }
                    r_n_obs += 1.0;
                }
            }
        }

        let total_nats = elias_rissanen_nats(l) + payload_nats;
        Self {
            alphabet_size,
            len: l,
            total_nats,
            payload_nats,
            site_ln_m0,
            site_repeat_weight,
            ref_repeat_weight,
            site_m0_dist,
            site_rep_dist,
            stationary_pi,
        }
    }

    #[inline]
    pub fn m0_prob(&self, pos: usize, state: usize) -> f64 {
        self.site_m0_dist[pos * self.alphabet_size + state]
    }

    #[inline]
    pub fn rep_prob(&self, pos: usize, state: usize) -> f64 {
        self.site_rep_dist[pos * self.alphabet_size + state]
    }
}
