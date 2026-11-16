use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};

#[cfg(not(target_arch = "wasm32"))]
use rayon::join;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
#[cfg(target_arch = "wasm32")]
use crate::wasm_rayon_shim::*;
use serde::{Deserialize, Serialize};

use crate::alphabet::{
    detect_seq_mode, EncodedSeq, GeneticCode, RawSeq, SeqMode, AA_PROPERTIES, AA_TO_BIOPHYS,
    BIOPHYS_CLASS_NAMES, NUM_BIOPHYS_CLASSES,
};
use crate::mdl::{exchangeable_dirichlet_seq_ln_prob, mdl_gain_to_prob};
use crate::models::column_niche_mixture_ln_prob;
use crate::transducer::{
    attempt_frameshift_repair_with_margin, evaluate_pair_homology,
    evaluate_pair_homology_prepared, get_global_regime_tables, populate_pair_posteriors_prepared,
    prewarm_path_partitions_for_cohort, summarize_dna_bubble_composition, FrameshiftEvent,
    GlobalRegimeTables, PairForwardState, PairHomologyResult, PreparedSeq, PRIOR_MEAN_LEN,
};

/// Minimum posterior fine-homology probability for merging two subtree columns across a branch
/// (exact Bayes / MDL decision boundary P_fine > 1 - P_fine <=> P_fine > 0.50).
/// Below this horizon, fine 1-to-1 homology is epistemically unidentifiable and contracts into
/// clade-local columns linked by a Coarse Homology Bubble!
pub const FINE_MERGE_HORIZON: f64 = 0.50;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColumnScale {
    /// Fine 1-to-1 column spanning multiple clades / global alignment core.
    GlobalFine,
    /// Clade-local fine column inside a region where deeper cross-clade homology hit a Coarse Horizon.
    CladeLocalFine,
    /// Single-lineage unaligned insertion / overhang token.
    LineagePrivate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeNodeRecord {
    pub node_id: usize,
    pub is_leaf: bool,
    pub label: String,
    pub left_node_id: Option<usize>,
    pub right_node_id: Option<usize>,
    pub len_left: f64,
    pub len_right: f64,
    pub local_seq_idx: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotatedColumn {
    pub col_index: usize,
    pub scale: ColumnScale,
    pub owner_node_id: usize,
    pub num_present: usize,
    /// Mean pairwise posterior fine-homology confidence among present residues in this column.
    pub mean_fine_posterior: f64,
    /// Mean posterior probability that present residues belong to a Coarse Homology Bubble across deeper splits.
    pub coarse_bubble_posterior: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HorizonBubbleReport {
    pub tree_merge_step: usize,
    pub split_node_id: usize,
    pub left_node_id: usize,
    pub right_node_id: usize,
    pub left_clade_size: usize,
    pub right_clade_size: usize,
    pub left_col_span: usize,
    pub right_col_span: usize,
    /// Token coordinate spans `(local_seq_idx, start_pos, len)` on the left side of the bubble.
    pub left_intervals: Vec<(usize, usize, usize)>,
    /// Token coordinate spans `(local_seq_idx, start_pos, len)` on the right side of the bubble.
    pub right_intervals: Vec<(usize, usize, usize)>,
    pub dominant_biophys_class: Option<String>,
    pub macro_biophys_gain_nats: f64,
    pub delta_hydropathy: f64,
    pub delta_charge: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomologyComponentReport {
    pub component_id: usize,
    pub sequence_ids: Vec<String>,
    pub newick_tree: String,
    pub root_node_id: usize,
    pub tree_nodes: Vec<TreeNodeRecord>,
    pub unaligned_seqs: Vec<(String, String)>,
    pub num_columns: usize,
    pub num_global_fine_cols: usize,
    pub num_clade_local_cols: usize,
    pub num_lineage_private_cols: usize,
    pub mean_core_posterior: f64,
    pub horizon_bubbles: Vec<HorizonBubbleReport>,
    pub frameshifts_repaired: Vec<FrameshiftEvent>,
    pub aligned_fasta: Vec<(String, String)>,
    pub columns: Vec<AnnotatedColumn>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohortHomologyLedger {
    pub mode: SeqMode,
    pub mito_tga_trp: bool,
    pub num_sequences: usize,
    /// Sum of independent single-sequence lossless code lengths \sum_k L_0(X^(k)) in nats.
    pub raw_null_cohort_nats: f64,
    /// Estimated joint provenance-compressed cohort description length in nats.
    pub provenance_cohort_nats: f64,
    /// Net cohort homology information savings in nats.
    pub net_cohort_gain_nats: f64,
    /// Number of distinct homologous sequence families (connected components with \Delta L_hom > 0).
    pub num_homology_components: usize,
    pub components: Vec<HomologyComponentReport>,
    pub pairwise_summary: Vec<PairSummaryRow>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairSummaryRow {
    pub id_x: String,
    pub id_y: String,
    pub mdl_gain_nats: f64,
    pub homology_posterior: f64,
    pub est_t_primary: f64,
    pub est_t_syn: Option<f64>,
    pub fine_homologous_pairs: usize,
    pub coarse_bubble_segments: usize,
}

#[derive(Debug, Clone)]
enum PhyloNode {
    Leaf(usize),
    Internal {
        left: Box<PhyloNode>,
        right: Box<PhyloNode>,
        len_left: f64,
        len_right: f64,
    },
}

impl PhyloNode {
    fn flatten_and_label(&self, names: &[String]) -> (Vec<TreeNodeRecord>, usize, String) {
        let m = names.len();
        assert!(m >= 1);
        let mut records: Vec<Option<TreeNodeRecord>> = vec![None; 2 * m - 1];
        let mut next_internal = m;

        fn walk(
            node: &PhyloNode,
            names: &[String],
            m: usize,
            records: &mut [Option<TreeNodeRecord>],
            next_internal: &mut usize,
        ) -> (usize, String) {
            match node {
                PhyloNode::Leaf(idx) => {
                    let lbl = sanitize_newick_label(&names[*idx]);
                    records[*idx] = Some(TreeNodeRecord {
                        node_id: *idx,
                        is_leaf: true,
                        label: lbl.clone(),
                        left_node_id: None,
                        right_node_id: None,
                        len_left: 0.0,
                        len_right: 0.0,
                        local_seq_idx: Some(*idx),
                    });
                    (*idx, lbl)
                }
                PhyloNode::Internal {
                    left,
                    right,
                    len_left,
                    len_right,
                } => {
                    let (lid, l_str) = walk(left, names, m, records, next_internal);
                    let (rid, r_str) = walk(right, names, m, records, next_internal);
                    let id = *next_internal;
                    *next_internal += 1;
                    let lbl = format!("N{}", id - m);
                    let l_len = len_left.max(0.0);
                    let r_len = len_right.max(0.0);
                    records[id] = Some(TreeNodeRecord {
                        node_id: id,
                        is_leaf: false,
                        label: lbl.clone(),
                        left_node_id: Some(lid),
                        right_node_id: Some(rid),
                        len_left: l_len,
                        len_right: r_len,
                        local_seq_idx: None,
                    });
                    (
                        id,
                        format!("({}:{:.5},{}:{:.5}){}", l_str, l_len, r_str, r_len, lbl),
                    )
                }
            }
        }

        let (root_id, root_str) = walk(self, names, m, &mut records, &mut next_internal);
        let tree_nodes: Vec<TreeNodeRecord> = records
            .into_iter()
            .map(|r| r.expect("unpopulated PhyloNode record"))
            .collect();
        (tree_nodes, root_id, format!("{};", root_str))
    }
}

fn sanitize_newick_label(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '(' | ')' | ':' | ';' | ',' | ' ' | '\t' => '_',
            _ => c,
        })
        .collect()
}

fn farthest_leaf_in_unrooted_tree(
    start: usize,
    adj: &[Vec<(usize, f64)>],
    m: usize,
) -> (usize, Vec<usize>, Vec<f64>) {
    let n_v = adj.len();
    let mut parent = vec![usize::MAX; n_v];
    let mut edge_len = vec![0.0f64; n_v];
    let mut w_dist = vec![0.0f64; n_v];
    let mut visited = vec![false; n_v];
    let mut stack = Vec::with_capacity(n_v);
    visited[start] = true;
    stack.push(start);

    while let Some(u) = stack.pop() {
        for &(v, w) in &adj[u] {
            if !visited[v] {
                visited[v] = true;
                parent[v] = u;
                edge_len[v] = w;
                w_dist[v] = w_dist[u] + w + 1e-6;
                stack.push(v);
            }
        }
    }

    let mut best_leaf = start;
    let mut best_d = -1.0f64;
    for u in 0..m {
        if visited[u]
            && u != start
            && (w_dist[u] > best_d + 1e-12
                || ((w_dist[u] - best_d).abs() <= 1e-12 && u < best_leaf))
        {
            best_d = w_dist[u];
            best_leaf = u;
        }
    }
    (best_leaf, parent, edge_len)
}

fn orient_unrooted_tree(
    u: usize,
    blocked_parent: usize,
    adj: &[Vec<(usize, f64)>],
    indices: &[usize],
    m: usize,
) -> PhyloNode {
    if u < m {
        return PhyloNode::Leaf(indices[u]);
    }
    let mut children: Vec<(usize, f64)> = Vec::with_capacity(2);
    for &(v, w) in &adj[u] {
        if v != blocked_parent {
            children.push((v, w));
        }
    }
    debug_assert_eq!(children.len(), 2, "internal unrooted vertex must have degree 3");
    let (c1, w1) = children[0];
    let (c2, w2) = children[1];
    PhyloNode::Internal {
        left: Box::new(orient_unrooted_tree(c1, u, adj, indices, m)),
        right: Box::new(orient_unrooted_tree(c2, u, adj, indices, m)),
        len_left: w1.max(0.0),
        len_right: w2.max(0.0),
    }
}

/// BioNJ (Gascuel 1997) variance-weighted Neighbor-Joining tree construction with
/// exact diameter midpoint rerooting on decoupled evolutionary divergences `(dist_full, var_full)`.
fn build_bionj_tree(
    indices: &[usize],
    dist_full: &[f64],
    var_full: &[f64],
    n_full: usize,
) -> PhyloNode {
    let m = indices.len();
    assert!(m >= 1);
    if m == 1 {
        return PhyloNode::Leaf(indices[0]);
    }
    if m == 2 {
        let d = dist_full[indices[0] * n_full + indices[1]].max(0.0);
        return PhyloNode::Internal {
            left: Box::new(PhyloNode::Leaf(indices[0])),
            right: Box::new(PhyloNode::Leaf(indices[1])),
            len_left: 0.5 * d,
            len_right: 0.5 * d,
        };
    }

    let n_unrooted = 2 * m - 2;
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::with_capacity(3); n_unrooted];
    let mut dmat = vec![0.0f64; m * m];
    let mut vmat = vec![0.0f64; m * m];
    for a in 0..m {
        for b in 0..m {
            dmat[a * m + b] = dist_full[indices[a] * n_full + indices[b]].max(0.0);
            vmat[a * m + b] = var_full[indices[a] * n_full + indices[b]].max(1e-8);
        }
    }
    let mut alive: Vec<(usize, usize)> = (0..m).map(|i| (i, i)).collect();
    let mut cur_dim = m;

    let mut r_sum_by_slot = vec![0.0f64; m];
    for &(su, _) in &alive {
        let mut s = 0.0;
        let row = &dmat[su * m..(su + 1) * m];
        for &(sv, _) in &alive {
            if su != sv {
                s += row[sv];
            }
        }
        r_sum_by_slot[su] = s;
    }

    #[inline]
    fn better_bionj(
        q1: f64,
        d1: f64,
        p1: (usize, usize),
        q2: f64,
        d2: f64,
        p2: (usize, usize),
    ) -> bool {
        if q1 < q2 - 1e-11 {
            true
        } else if (q1 - q2).abs() <= 1e-11 {
            if (d1 - d2).abs() > 1e-12 {
                d1 < d2
            } else {
                p1 < p2
            }
        } else {
            false
        }
    }

    while alive.len() > 3 {
        let k = alive.len();
        let km2 = (k - 2) as f64;
        let (best_pair, best_len_u, best_len_v) = if k < 384 {
            let mut best_q = f64::INFINITY;
            let mut best_d = f64::INFINITY;
            let mut bp = (0usize, 1usize);
            for a in 0..k {
                let su = alive[a].0;
                let ru = r_sum_by_slot[su];
                let d_row = &dmat[su * m..(su + 1) * m];
                for b in (a + 1)..k {
                    let sv = alive[b].0;
                    let d_uv = d_row[sv];
                    let q = km2 * d_uv - ru - r_sum_by_slot[sv];
                    if better_bionj(q, d_uv, (a, b), best_q, best_d, bp) {
                        best_q = q;
                        best_d = d_uv;
                        bp = (a, b);
                    }
                }
            }
            let su = alive[bp.0].0;
            let sv = alive[bp.1].0;
            let d_uv = dmat[su * m + sv];
            let delta = (r_sum_by_slot[su] - r_sum_by_slot[sv]) / km2;
            let len_u = (0.5 * (d_uv + delta)).max(0.0);
            let len_v = (d_uv - len_u).max(0.0);
            (bp, len_u, len_v)
        } else {
            #[cfg(not(target_arch = "wasm32"))]
            let (_, _, bp) = (0..k)
                .into_par_iter()
                .map(|a| {
                    let su = alive[a].0;
                    let ru = r_sum_by_slot[su];
                    let d_row = &dmat[su * m..(su + 1) * m];
                    let mut l_q = f64::INFINITY;
                    let mut l_d = f64::INFINITY;
                    let mut l_bp = (a, a + 1);
                    for b in (a + 1)..k {
                        let sv = alive[b].0;
                        let d_uv = d_row[sv];
                        let q = km2 * d_uv - ru - r_sum_by_slot[sv];
                        if better_bionj(q, d_uv, (a, b), l_q, l_d, l_bp) {
                            l_q = q;
                            l_d = d_uv;
                            l_bp = (a, b);
                        }
                    }
                    (l_q, l_d, l_bp)
                })
                .reduce(
                    || (f64::INFINITY, f64::INFINITY, (usize::MAX, usize::MAX)),
                    |x, y| {
                        if better_bionj(x.0, x.1, x.2, y.0, y.1, y.2) {
                            x
                        } else {
                            y
                        }
                    },
                );
            #[cfg(target_arch = "wasm32")]
            let (_, _, bp) = (0..k)
                .into_par_iter()
                .map(|a| {
                    let su = alive[a].0;
                    let ru = r_sum_by_slot[su];
                    let d_row = &dmat[su * m..(su + 1) * m];
                    let mut l_q = f64::INFINITY;
                    let mut l_d = f64::INFINITY;
                    let mut l_bp = (a, a + 1);
                    for b in (a + 1)..k {
                        let sv = alive[b].0;
                        let d_uv = d_row[sv];
                        let q = km2 * d_uv - ru - r_sum_by_slot[sv];
                        if better_bionj(q, d_uv, (a, b), l_q, l_d, l_bp) {
                            l_q = q;
                            l_d = d_uv;
                            l_bp = (a, b);
                        }
                    }
                    (l_q, l_d, l_bp)
                })
                .fold(
                    (f64::INFINITY, f64::INFINITY, (usize::MAX, usize::MAX)),
                    |x, y| {
                        if better_bionj(x.0, x.1, x.2, y.0, y.1, y.2) {
                            x
                        } else {
                            y
                        }
                    },
                );
            let su = alive[bp.0].0;
            let sv = alive[bp.1].0;
            let d_uv = dmat[su * m + sv];
            let delta = (r_sum_by_slot[su] - r_sum_by_slot[sv]) / km2;
            let len_u = (0.5 * (d_uv + delta)).max(0.0);
            let len_v = (d_uv - len_u).max(0.0);
            (bp, len_u, len_v)
        };

        let (pa, pb) = best_pair;
        let (su, u) = alive[pa];
        let (sv, v) = alive[pb];
        let len_u = best_len_u;
        let len_v = best_len_v;
        let v_uv = vmat[su * m + sv];

        let lambda = if v_uv <= 1e-12 {
            0.5
        } else {
            let mut diff_sum = 0.0;
            for &(sw, _) in &alive {
                if sw != su && sw != sv {
                    diff_sum += vmat[sv * m + sw] - vmat[su * m + sw];
                }
            }
            (0.5 + diff_sum / (2.0 * km2 * v_uv)).clamp(0.0, 1.0)
        };

        let new_id = cur_dim;
        cur_dim += 1;
        let mut new_r = 0.0f64;
        for &(sw, _) in &alive {
            if sw != su && sw != sv {
                let d_uw = dmat[su * m + sw];
                let d_vw = dmat[sv * m + sw];
                let v_uw = vmat[su * m + sw];
                let v_vw = vmat[sv * m + sw];
                let d_nw = (lambda * (d_uw - len_u) + (1.0 - lambda) * (d_vw - len_v)).max(0.0);
                let v_nw = (lambda * lambda * v_uw
                    + (1.0 - lambda) * (1.0 - lambda) * v_vw
                    - lambda * (1.0 - lambda) * v_uv)
                    .max(1e-8);
                dmat[su * m + sw] = d_nw;
                dmat[sw * m + su] = d_nw;
                vmat[su * m + sw] = v_nw;
                vmat[sw * m + su] = v_nw;
                r_sum_by_slot[sw] += d_nw - d_uw - d_vw;
                new_r += d_nw;
            }
        }
        r_sum_by_slot[su] = new_r;

        adj[new_id].push((u, len_u));
        adj[u].push((new_id, len_u));
        adj[new_id].push((v, len_v));
        adj[v].push((new_id, len_v));

        alive.remove(pb);
        alive.remove(pa);
        alive.push((su, new_id));
    }

    let (s0, u0) = alive[0];
    let (s1, u1) = alive[1];
    let (s2, u2) = alive[2];
    let c = cur_dim;
    let d01 = dmat[s0 * m + s1];
    let d02 = dmat[s0 * m + s2];
    let d12 = dmat[s1 * m + s2];
    let l0 = (0.5 * (d01 + d02 - d12)).max(0.0);
    let l1 = (0.5 * (d01 + d12 - d02)).max(0.0);
    let l2 = (0.5 * (d02 + d12 - d01)).max(0.0);
    adj[c].push((u0, l0));
    adj[u0].push((c, l0));
    adj[c].push((u1, l1));
    adj[u1].push((c, l1));
    adj[c].push((u2, l2));
    adj[u2].push((c, l2));

    // Exact diameter midpoint rerooting so deep clades merge before outgroups
    let (leaf_a, _, _) = farthest_leaf_in_unrooted_tree(0, &adj, m);
    let (leaf_b, parent_a, edge_len_a) = farthest_leaf_in_unrooted_tree(leaf_a, &adj, m);
    let mut rev_path = Vec::new();
    let mut cur = leaf_b;
    while cur != usize::MAX {
        rev_path.push(cur);
        cur = parent_a[cur];
    }
    rev_path.reverse();

    let n_edges = rev_path.len() - 1;
    let edge_weights: Vec<f64> = (0..n_edges).map(|k| edge_len_a[rev_path[k + 1]]).collect();
    let total_len: f64 = edge_weights.iter().sum();

    let (x, y, len_x, len_y) = if total_len > 0.0 {
        let half = 0.5 * total_len;
        let mut cum = 0.0f64;
        let mut split_k = n_edges - 1;
        for k in 0..n_edges {
            if cum + edge_weights[k] >= half - 1e-12 {
                split_k = k;
                break;
            }
            cum += edge_weights[k];
        }
        let w_xy = edge_weights[split_k];
        let lx = (half - cum).clamp(0.0, w_xy);
        let ly = (w_xy - lx).max(0.0);
        (rev_path[split_k], rev_path[split_k + 1], lx, ly)
    } else {
        let mid_k = (n_edges - 1) / 2;
        (rev_path[mid_k], rev_path[mid_k + 1], 0.0, 0.0)
    };

    PhyloNode::Internal {
        left: Box::new(orient_unrooted_tree(x, y, &adj, indices, m)),
        right: Box::new(orient_unrooted_tree(y, x, &adj, indices, m)),
        len_left: len_x,
        len_right: len_y,
    }
}

/// Representation of a Subtree Multi-Scale Profile during profile alignment and refinement.
struct SubtreeProfile {
    /// Taxon indices (into component-local `comp_seqs`) present in this subtree.
    taxa: Vec<usize>,
    /// Ordered columns: each column holds sorted `(local_taxon_idx, token_pos)` entries.
    /// Guaranteed monotonic in `token_pos` for every taxon!
    cols: Vec<Vec<(usize, usize)>>,
}

#[inline]
fn pair_key(a: usize, b: usize) -> u64 {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    ((lo as u64) << 32) | (hi as u64)
}

#[inline]
fn is_post_eval(res: &PairHomologyResult) -> bool {
    res.len_x > 0 && res.bubble_post_x.len() == res.len_x && res.bubble_post_y.len() == res.len_y
}

/// Sparse lookup table for consistent fine-homology posteriors P_cons(a:i ~ b:j)
/// within a connected homology component of size M.
struct ConsistentEdgeLookup {
    /// Deterministically sorted `(a, b)` with `a < b` that have consistent edges.
    active_keys: Vec<(usize, usize)>,
    /// Keyed by `pair_key(a, b)` for `a < b`: sorted `(idx_a, idx_b, p_cons)` with `p_cons >= 0.02`.
    edges_map: HashMap<u64, Vec<(u32, u32, f32)>>,
    /// Keyed by `pair_key(a, b)`: `(pair_rel, pair_mass)`.
    meta_map: HashMap<u64, (f64, f64)>,
    /// Sorted adjacent component-local partners `b` for each taxon `a`.
    nbrs: Vec<Vec<usize>>,
    bubble_x_by_seq: Vec<Vec<f64>>,
}

impl ConsistentEdgeLookup {
    #[inline]
    fn pair_rel(&self, a: usize, b: usize) -> f64 {
        self.meta_map.get(&pair_key(a, b)).map_or(0.0, |x| x.0)
    }

    #[inline]
    fn pair_mass(&self, a: usize, b: usize) -> f64 {
        self.meta_map.get(&pair_key(a, b)).map_or(0.0, |x| x.1)
    }

    #[inline]
    fn pair_edges_lt(&self, lo: usize, hi: usize) -> &[(u32, u32, f32)] {
        debug_assert!(lo < hi);
        self.edges_map
            .get(&(((lo as u64) << 32) | (hi as u64)))
            .map_or(&[], Vec::as_slice)
    }
}

struct DirectAdjData {
    edges_ab: Vec<(u32, f32)>,
    off_ab: Vec<u32>,
    edges_ba: Vec<(u32, f32)>,
    off_ba: Vec<u32>,
    reach_ab: Vec<f32>,
    reach_ba: Vec<f32>,
    cov: f64,
}

#[inline]
fn csr_row<'a>(edges: &'a [(u32, f32)], off: &'a [u32], idx: usize) -> &'a [(u32, f32)] {
    let s = off[idx] as usize;
    let e = off[idx + 1] as usize;
    &edges[s..e]
}

/// Compute Tree-Distance-Weighted Probabilistic Consistency across component sequences in parallel!
/// Exact all-pairs for M <= 96; sparse active neighborhood + relay graph for M > 96.
fn build_consistent_lookup(
    comp_indices: &[usize],
    pair_map: &HashMap<u64, PairHomologyResult>,
    dist_mat: &[f64],
    seq_lens: &[usize],
) -> ConsistentEdgeLookup {
    let m = comp_indices.len();
    let mut direct_data: HashMap<u64, DirectAdjData> = HashMap::new();
    let mut direct_nbrs: Vec<Vec<usize>> = vec![Vec::new(); m];

    let mut bubble_x_by_seq: Vec<Vec<f64>> = (0..m)
        .map(|a| vec![0.0f64; seq_lens[comp_indices[a]]])
        .collect();
    let mut bubble_norm = vec![0.0f64; m];

    let mut raw_comp_pairs: Vec<(usize, usize, &PairHomologyResult)> = Vec::new();
    if m <= 96 {
        for a in 0..m {
            for b in (a + 1)..m {
                let ga = comp_indices[a];
                let gb = comp_indices[b];
                if let Some(res) = pair_map.get(&pair_key(ga, gb)) {
                    if is_post_eval(res) {
                        raw_comp_pairs.push((a, b, res));
                    }
                }
            }
        }
    } else {
        let mut g2l: HashMap<usize, usize> = HashMap::with_capacity(m);
        for (loc, &g) in comp_indices.iter().enumerate() {
            g2l.insert(g, loc);
        }
        for (&key, res) in pair_map {
            if !is_post_eval(res) {
                continue;
            }
            let ga = (key >> 32) as usize;
            let gb = (key & 0xFFFF_FFFF) as usize;
            if let (Some(&la), Some(&lb)) = (g2l.get(&ga), g2l.get(&gb)) {
                let (a, b) = if la < lb { (la, lb) } else { (lb, la) };
                raw_comp_pairs.push((a, b, res));
            }
        }
        raw_comp_pairs.sort_unstable_by_key(|x| (x.0, x.1));
    }

    for (a, b, res) in raw_comp_pairs {
        let ga = comp_indices[a];
        let gb = comp_indices[b];
        let flip = ga > gb;

        let pi_comp = if m >= 2 && res.mdl_gain_nats > -3.0 {
            mdl_gain_to_prob(res.mdl_gain_nats + 2.5).max(res.homology_posterior)
        } else {
            res.homology_posterior
        };
        let h_raw = res.homology_posterior;
        let recover_bub = |bp: f64| -> f64 {
            if h_raw > 1e-6 {
                (bp / h_raw).clamp(0.0, 1.0) * pi_comp
            } else {
                bp
            }
        };

        let d = res.est_t_primary;
        let w_pair = (-0.5 * d).exp();
        if !flip {
            for (i, &bp) in res.bubble_post_x.iter().enumerate() {
                if i < bubble_x_by_seq[a].len() {
                    bubble_x_by_seq[a][i] += w_pair * recover_bub(bp);
                }
            }
            for (j, &bp) in res.bubble_post_y.iter().enumerate() {
                if j < bubble_x_by_seq[b].len() {
                    bubble_x_by_seq[b][j] += w_pair * recover_bub(bp);
                }
            }
        } else {
            for (i, &bp) in res.bubble_post_y.iter().enumerate() {
                if i < bubble_x_by_seq[a].len() {
                    bubble_x_by_seq[a][i] += w_pair * recover_bub(bp);
                }
            }
            for (j, &bp) in res.bubble_post_x.iter().enumerate() {
                if j < bubble_x_by_seq[b].len() {
                    bubble_x_by_seq[b][j] += w_pair * recover_bub(bp);
                }
            }
        }
        bubble_norm[a] += w_pair;
        bubble_norm[b] += w_pair;

        let la = seq_lens[ga];
        let lb = seq_lens[gb];
        let mut off_ab = vec![0u32; la + 1];
        let mut off_ba = vec![0u32; lb + 1];
        let mut reach_ab = vec![0.0f32; la];
        let mut reach_ba = vec![0.0f32; lb];
        let mut edge_mass = 0.0f64;
        for e in &res.fine_edges {
            let p_edge = if e.cond_posterior > 0.0 {
                (e.cond_posterior * pi_comp).clamp(0.0, 1.0)
            } else {
                e.posterior
            };
            if p_edge >= 0.02 {
                let (ia, ib) = if !flip {
                    (e.idx_x, e.idx_y)
                } else {
                    (e.idx_y, e.idx_x)
                };
                if ia < la && ib < lb {
                    off_ab[ia + 1] += 1;
                    off_ba[ib + 1] += 1;
                    let pf = p_edge as f32;
                    reach_ab[ia] += pf;
                    reach_ba[ib] += pf;
                    edge_mass += p_edge;
                }
            }
        }
        let cov_ab = (edge_mass / (la.min(lb).max(1) as f64)).clamp(0.0, 1.0);
        if cov_ab >= 0.01 {
            for ia in 0..la {
                off_ab[ia + 1] += off_ab[ia];
                reach_ab[ia] = reach_ab[ia].clamp(0.0, 1.0);
            }
            for ib in 0..lb {
                off_ba[ib + 1] += off_ba[ib];
                reach_ba[ib] = reach_ba[ib].clamp(0.0, 1.0);
            }
            let mut edges_ab = vec![(0u32, 0.0f32); off_ab[la] as usize];
            let mut edges_ba = vec![(0u32, 0.0f32); off_ba[lb] as usize];
            let mut cur_ab = off_ab[..la].to_vec();
            let mut cur_ba = off_ba[..lb].to_vec();
            for e in &res.fine_edges {
                let p_edge = if e.cond_posterior > 0.0 {
                    (e.cond_posterior * pi_comp).clamp(0.0, 1.0)
                } else {
                    e.posterior
                };
                if p_edge >= 0.02 {
                    let (ia, ib) = if !flip {
                        (e.idx_x, e.idx_y)
                    } else {
                        (e.idx_y, e.idx_x)
                    };
                    if ia < la && ib < lb {
                        let pf = p_edge as f32;
                        let pa = cur_ab[ia] as usize;
                        cur_ab[ia] += 1;
                        edges_ab[pa] = (ib as u32, pf);
                        let pb = cur_ba[ib] as usize;
                        cur_ba[ib] += 1;
                        edges_ba[pb] = (ia as u32, pf);
                    }
                }
            }
            if flip {
                for ia in 0..la {
                    let s = off_ab[ia] as usize;
                    let t = off_ab[ia + 1] as usize;
                    if t - s > 1 {
                        edges_ab[s..t].sort_unstable_by_key(|x| x.0);
                    }
                }
            } else {
                for ib in 0..lb {
                    let s = off_ba[ib] as usize;
                    let t = off_ba[ib + 1] as usize;
                    if t - s > 1 {
                        edges_ba[s..t].sort_unstable_by_key(|x| x.0);
                    }
                }
            }
            direct_nbrs[a].push(b);
            direct_nbrs[b].push(a);
            direct_data.insert(
                pair_key(a, b),
                DirectAdjData {
                    edges_ab,
                    off_ab,
                    edges_ba,
                    off_ba,
                    reach_ab,
                    reach_ba,
                    cov: cov_ab,
                },
            );
        }
    }

    for nbr_list in &mut direct_nbrs {
        nbr_list.sort_unstable();
    }

    for a in 0..m {
        if bubble_norm[a] > 0.0 {
            for v in &mut bubble_x_by_seq[a] {
                *v /= bubble_norm[a];
            }
        }
    }

    let pairs_ab: Vec<(usize, usize)> = if m <= 96 {
        let mut p = Vec::with_capacity(m * m.saturating_sub(1) / 2);
        for a in 0..m {
            for b in (a + 1)..m {
                p.push((a, b));
            }
        }
        p
    } else {
        let top_dir_cap = if m > 512 { 5usize } else { 8usize };
        let d_nbr_cap = if m > 512 { 12usize } else { 20usize };
        let far_cap = if m > 512 { 2usize } else { 4usize };
        let far_top_limit = if m > 512 { 4usize } else { top_dir_cap };
        let mut pair_set: HashSet<u64> = HashSet::with_capacity(m * 24);
        for &key in direct_data.keys() {
            pair_set.insert(key);
        }
        let mut top_dir: Vec<Vec<usize>> = vec![Vec::new(); m];
        let mut far_bridges: Vec<Vec<usize>> = vec![Vec::new(); m];
        for u in 0..m {
            let mut scored: Vec<(f64, usize)> = direct_nbrs[u]
                .iter()
                .map(|&v| {
                    let cov = direct_data.get(&pair_key(u, v)).map_or(0.0, |d| d.cov);
                    let sc = cov * (-0.35 * dist_mat[u * m + v]).exp();
                    (sc, v)
                })
                .collect();
            scored.sort_unstable_by(|x, y| y.0.total_cmp(&x.0).then_with(|| x.1.cmp(&y.1)));
            top_dir[u] = scored.into_iter().take(top_dir_cap).map(|x| x.1).collect();

            let mut far_cands: Vec<(f64, usize)> = direct_nbrs[u]
                .iter()
                .copied()
                .filter(|&v| {
                    !top_dir[u].contains(&v)
                        && direct_data.get(&pair_key(u, v)).map_or(0.0, |d| d.cov) >= 0.12
                })
                .map(|v| (dist_mat[u * m + v], v))
                .collect();
            far_cands.sort_unstable_by(|x, y| y.0.total_cmp(&x.0).then_with(|| x.1.cmp(&y.1)));
            far_bridges[u] = far_cands.into_iter().take(far_cap).map(|x| x.1).collect();
        }
        let mut d_nbr: Vec<(f64, usize)> = Vec::with_capacity(m);
        for a in 0..m {
            d_nbr.clear();
            for b in 0..m {
                if a != b {
                    d_nbr.push((dist_mat[a * m + b], b));
                }
            }
            if d_nbr.len() > d_nbr_cap {
                d_nbr.select_nth_unstable_by(d_nbr_cap, |x, y| {
                    x.0.total_cmp(&y.0).then_with(|| x.1.cmp(&y.1))
                });
            }
            for &(_, b) in d_nbr.iter().take(d_nbr_cap) {
                pair_set.insert(pair_key(a, b));
            }
            for &c in &top_dir[a] {
                for &b in top_dir[c].iter().chain(far_bridges[c].iter()) {
                    if a != b {
                        pair_set.insert(pair_key(a, b));
                    }
                }
            }
            for &c in &far_bridges[a] {
                for &b in top_dir[c].iter().take(far_top_limit) {
                    if a != b {
                        pair_set.insert(pair_key(a, b));
                    }
                }
            }
        }
        let mut p: Vec<(usize, usize)> = pair_set
            .into_iter()
            .map(|k| ((k >> 32) as usize, (k & 0xFFFF_FFFF) as usize))
            .collect();
        p.sort_unstable();
        p
    };

    #[inline]
    fn direct_adj_view<'a>(
        data: &'a HashMap<u64, DirectAdjData>,
        src: usize,
        dst: usize,
    ) -> (&'a [(u32, f32)], &'a [u32], &'a [f32], f64) {
        let d = &data[&pair_key(src, dst)];
        if src < dst {
            (&d.edges_ab, &d.off_ab, &d.reach_ab, d.cov)
        } else {
            (&d.edges_ba, &d.off_ba, &d.reach_ba, d.cov)
        }
    }

    let mut edges_map: HashMap<u64, Vec<(u32, u32, f32)>> =
        HashMap::with_capacity(pairs_ab.len().min(32_768));
    let mut meta_map: HashMap<u64, (f64, f64)> =
        HashMap::with_capacity(pairs_ab.len().min(32_768));
    let mut nbrs: Vec<Vec<usize>> = vec![Vec::new(); m];

    for chunk in pairs_ab.chunks(20_000) {
        let computed: Vec<((usize, usize), f64, Vec<(u32, u32, f32)>)> = chunk
            .par_iter()
            .copied()
            .filter_map(|(a, b)| {
                let la = seq_lens[comp_indices[a]];
                let lb = seq_lens[comp_indices[b]];
                let dir_opt = direct_data.get(&pair_key(a, b));
                let is_dir = dir_opt.is_some();
                let d_ab = dist_mat[a * m + b];
                let w_dir = if let Some(d_ab_rec) = dir_opt {
                    2.0 * (-0.35 * d_ab).exp() * d_ab_rec.cov.sqrt().max(0.08)
                } else {
                    0.0
                };
                let mut w_tot = w_dir;
                let mut relays: Vec<(usize, f64)> = Vec::new();
                let na = &direct_nbrs[a];
                let nb = &direct_nbrs[b];
                let (mut ia_n, mut ib_n) = (0usize, 0usize);
                while ia_n < na.len() && ib_n < nb.len() {
                    match na[ia_n].cmp(&nb[ib_n]) {
                        std::cmp::Ordering::Less => ia_n += 1,
                        std::cmp::Ordering::Greater => ib_n += 1,
                        std::cmp::Ordering::Equal => {
                            let c = na[ia_n];
                            ia_n += 1;
                            ib_n += 1;
                            let d_path = dist_mat[a * m + c] + dist_mat[c * m + b];
                            let excess_d = (d_path - d_ab).max(0.0);
                            let cov_ac = direct_data[&pair_key(a, c)].cov;
                            let cov_cb = direct_data[&pair_key(c, b)].cov;
                            let cov_acb = (cov_ac * cov_cb).sqrt().max(0.15);
                            let w_c = if is_dir {
                                1.2 * (-0.45 * d_path - 1.5 * excess_d).exp() * cov_acb
                            } else {
                                1.6 * (-0.35 * d_path).exp() * cov_acb
                            };
                            if w_c >= 0.015 {
                                relays.push((c, w_c));
                                w_tot += w_c;
                            }
                        }
                    }
                }

                if w_tot <= 0.0 {
                    return None;
                }

                let mut out_edges: Vec<(u32, u32, f32)> = Vec::new();
                let mut out_mass = 0.0f64;
                if relays.is_empty() {
                    if let Some(d_ab_rec) = dir_opt {
                        for ia in 0..la {
                            for &(ib, p_ab) in csr_row(&d_ab_rec.edges_ab, &d_ab_rec.off_ab, ia) {
                                let p_cons = (p_ab as f64).clamp(0.0, 1.0);
                                if p_cons >= 0.02 {
                                    out_edges.push((ia as u32, ib, p_cons as f32));
                                    out_mass += p_cons;
                                }
                            }
                        }
                    }
                    let out_cov = (out_mass / (la.min(lb).max(1) as f64)).clamp(0.0, 1.0);
                    if out_edges.is_empty() || out_cov < 0.005 {
                        return None;
                    }
                    let rel = if is_dir {
                        out_cov.sqrt().clamp(0.15, 1.0)
                    } else {
                        (w_tot / (w_tot + 1.2)).clamp(0.05, 0.75) * out_cov.sqrt().clamp(0.15, 1.0)
                    };
                    out_edges.shrink_to_fit();
                    return Some(((a, b), rel, out_edges));
                }

                let ab_csr_opt = dir_opt.map(|d| (&d.edges_ab, &d.off_ab));
                let relay_views: Vec<_> = relays
                    .iter()
                    .map(|&(c, w_c)| {
                        let (edges_ac, off_ac, reach_ac, _) = direct_adj_view(&direct_data, a, c);
                        let (edges_cb, off_cb, _, _) = direct_adj_view(&direct_data, c, b);
                        let (_, _, reach_bc, _) = direct_adj_view(&direct_data, b, c);
                        (w_c, edges_ac, off_ac, edges_cb, off_cb, reach_ac, reach_bc)
                    })
                    .collect();

                let mut acc_row = vec![0.0f64; lb];
                let mut touched: Vec<usize> = Vec::with_capacity(32);

                for ia in 0..la {
                    for &idx in &touched {
                        acc_row[idx] = 0.0;
                    }
                    touched.clear();

                    if let Some((edges_ab, off_ab)) = ab_csr_opt {
                        for &(ib_u, p_ab) in csr_row(edges_ab, off_ab, ia) {
                            let ib = ib_u as usize;
                            if acc_row[ib] == 0.0 {
                                touched.push(ib);
                            }
                            acc_row[ib] += w_dir * (p_ab as f64);
                        }
                    }

                    for &(w_c, edges_ac, off_ac, edges_cb, off_cb, _, _) in &relay_views {
                        for &(ic_u, p_ac_f) in csr_row(edges_ac, off_ac, ia) {
                            let p_ac = p_ac_f as f64;
                            if p_ac < 0.03 {
                                continue;
                            }
                            for &(ib_u, p_cb_f) in csr_row(edges_cb, off_cb, ic_u as usize) {
                                let p_cb = p_cb_f as f64;
                                if p_cb >= 0.03 {
                                    let ib = ib_u as usize;
                                    if acc_row[ib] == 0.0 {
                                        touched.push(ib);
                                    }
                                    acc_row[ib] += w_c * p_ac * p_cb;
                                }
                            }
                        }
                    }

                    touched.sort_unstable();
                    for &ib in &touched {
                        let mut denom_cell = w_dir;
                        for &(w_c, _, _, _, _, reach_ac, reach_bc) in &relay_views {
                            let rac = reach_ac[ia] as f64;
                            let rbc = reach_bc[ib] as f64;
                            denom_cell += w_c * rac.max(rbc);
                        }
                        let p_cons = if denom_cell > 0.0 {
                            (acc_row[ib] / denom_cell).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        if p_cons >= 0.02 {
                            out_edges.push((ia as u32, ib as u32, p_cons as f32));
                            out_mass += p_cons;
                        }
                    }
                }

                let out_cov = (out_mass / (la.min(lb).max(1) as f64)).clamp(0.0, 1.0);
                if out_edges.is_empty() || out_cov < 0.005 {
                    return None;
                }
                let rel = if is_dir {
                    out_cov.sqrt().clamp(0.15, 1.0)
                } else {
                    (w_tot / (w_tot + 1.2)).clamp(0.05, 0.75) * out_cov.sqrt().clamp(0.15, 1.0)
                };
                out_edges.shrink_to_fit();
                Some(((a, b), rel, out_edges))
            })
            .collect();

        for ((a, b), rel, edges) in computed {
            let s: f64 = edges.iter().map(|e| e.2 as f64).sum();
            let k = pair_key(a, b);
            meta_map.insert(k, (rel, s));
            edges_map.insert(k, edges);
            nbrs[a].push(b);
            nbrs[b].push(a);
        }
    }

    drop(direct_data);
    drop(direct_nbrs);
    drop(pairs_ab);

    // Sparse Round-2 relay pass for mid-sized components (30 < M <= 512)
    if m > 30 && m <= 512 {
        const R2_TOP_K: usize = 10;
        let mut r1_adj: Vec<Vec<(usize, Vec<Vec<(u32, f64)>>, Vec<f64>)>> =
            vec![Vec::new(); m];
        for u in 0..m {
            let lu = seq_lens[comp_indices[u]];
            let mut n_cand: Vec<(f64, usize)> = Vec::new();
            for &v in &nbrs[u] {
                let rel_uv = meta_map[&pair_key(u, v)].0;
                if rel_uv >= 0.20 {
                    let sc = rel_uv * (-0.35 * dist_mat[u * m + v]).exp();
                    n_cand.push((sc, v));
                }
            }
            n_cand.sort_unstable_by(|x, y| y.0.total_cmp(&x.0).then_with(|| x.1.cmp(&y.1)));
            for &(_, v) in n_cand.iter().take(R2_TOP_K) {
                let mut adj_uv = vec![Vec::new(); lu];
                let e_slice = edges_map[&pair_key(u, v)].as_slice();
                if u < v {
                    for &(iu, iv, p) in e_slice {
                        adj_uv[iu as usize].push((iv, p as f64));
                    }
                } else {
                    for &(iv, iu, p) in e_slice {
                        adj_uv[iu as usize].push((iv, p as f64));
                    }
                }
                let reach_u: Vec<f64> = adj_uv
                    .iter()
                    .map(|row| row.iter().map(|x| x.1).sum::<f64>().clamp(0.0, 1.0))
                    .collect();
                r1_adj[u].push((v, adj_uv, reach_u));
            }
        }

        let missing_pairs: Vec<(usize, usize)> = if m <= 96 {
            let mut mp = Vec::new();
            for a in 0..m {
                for b in (a + 1)..m {
                    if !meta_map.contains_key(&pair_key(a, b)) {
                        mp.push((a, b));
                    }
                }
            }
            mp
        } else {
            let mut by_c: Vec<Vec<usize>> = vec![Vec::new(); m];
            for u in 0..m {
                for &(c, _, _) in &r1_adj[u] {
                    by_c[c].push(u);
                }
            }
            let mut cand_set: HashSet<u64> = HashSet::new();
            for list in &by_c {
                if list.len() <= 64 {
                    for i in 0..list.len() {
                        let a = list[i];
                        for j in (i + 1)..list.len() {
                            let b = list[j];
                            let k = pair_key(a, b);
                            if !meta_map.contains_key(&k) {
                                cand_set.insert(k);
                            }
                        }
                    }
                } else {
                    for i in 0..list.len() {
                        let a = list[i];
                        for d in 1..=32 {
                            let b = list[(i + d) % list.len()];
                            if a != b {
                                let k = pair_key(a, b);
                                if !meta_map.contains_key(&k) {
                                    cand_set.insert(k);
                                }
                            }
                        }
                    }
                }
            }
            let mut mp: Vec<(usize, usize)> = cand_set
                .into_iter()
                .map(|k| ((k >> 32) as usize, (k & 0xFFFF_FFFF) as usize))
                .collect();
            mp.sort_unstable();
            mp
        };

        if !missing_pairs.is_empty() {
            let r2_res: Vec<((usize, usize), f64, Vec<(u32, u32, f32)>)> = missing_pairs
                .into_par_iter()
                .filter_map(|(a, b)| {
                    let la = seq_lens[comp_indices[a]];
                    let lb = seq_lens[comp_indices[b]];
                    let mut r2_relays: Vec<(usize, f64, usize, usize)> = Vec::new();
                    let mut w_tot = 0.0f64;
                    for (idx_ac, &(c, _, _)) in r1_adj[a].iter().enumerate() {
                        if let Some(idx_bc) = r1_adj[b].iter().position(|x| x.0 == c) {
                            let rel_ac = meta_map[&pair_key(a, c)].0;
                            let rel_cb = meta_map[&pair_key(c, b)].0;
                            let w_c = 1.2
                                * rel_ac
                                * rel_cb
                                * (-0.30 * (dist_mat[a * m + c] + dist_mat[c * m + b])).exp();
                            if w_c >= 0.015 {
                                r2_relays.push((c, w_c, idx_ac, idx_bc));
                                w_tot += w_c;
                            }
                        }
                    }
                    if r2_relays.is_empty() || w_tot <= 0.0 {
                        return None;
                    }

                    // Invert `r1_adj[b][idx_bc]` (`b -> c`) into `c -> b` for O(deg) lookup
                    let mut adj_cb_list: Vec<Vec<Vec<(u32, f64)>>> =
                        Vec::with_capacity(r2_relays.len());
                    for &(c, _, _, idx_bc) in &r2_relays {
                        let lc = seq_lens[comp_indices[c]];
                        let mut cb = vec![Vec::new(); lc];
                        for (ib, row_bc) in r1_adj[b][idx_bc].1.iter().enumerate() {
                            for &(ic, p) in row_bc {
                                if (ic as usize) < lc {
                                    cb[ic as usize].push((ib as u32, p));
                                }
                            }
                        }
                        adj_cb_list.push(cb);
                    }

                    let mut out_edges: Vec<(u32, u32, f32)> = Vec::new();
                    let mut out_mass = 0.0f64;
                    let mut acc_row = vec![0.0f64; lb];
                    let mut touched: Vec<usize> = Vec::with_capacity(32);

                    for ia in 0..la {
                        for &idx in &touched {
                            acc_row[idx] = 0.0;
                        }
                        touched.clear();
                        for (r_i, &(_, w_c, idx_ac, _)) in r2_relays.iter().enumerate() {
                            let adj_ac = &r1_adj[a][idx_ac].1;
                            let adj_cb = &adj_cb_list[r_i];
                            for &(ic_u, p_ac) in &adj_ac[ia] {
                                if p_ac < 0.03 {
                                    continue;
                                }
                                for &(ib_u, p_cb) in &adj_cb[ic_u as usize] {
                                    if p_cb >= 0.03 {
                                        let ib = ib_u as usize;
                                        if acc_row[ib] == 0.0 {
                                            touched.push(ib);
                                        }
                                        acc_row[ib] += w_c * p_ac * p_cb;
                                    }
                                }
                            }
                        }
                        touched.sort_unstable();
                        for &ib in &touched {
                            let mut denom_cell = 0.0f64;
                            for &(_, w_c, idx_ac, idx_bc) in &r2_relays {
                                let rac = r1_adj[a][idx_ac].2[ia];
                                let rbc = r1_adj[b][idx_bc].2[ib];
                                denom_cell += w_c * rac.max(rbc);
                            }
                            let p_cons = if denom_cell > 0.0 {
                                (acc_row[ib] / denom_cell).clamp(0.0, 1.0)
                            } else {
                                0.0
                            };
                            if p_cons >= 0.02 {
                                out_edges.push((ia as u32, ib as u32, p_cons as f32));
                                out_mass += p_cons;
                            }
                        }
                    }

                    let out_cov = (out_mass / (la.min(lb).max(1) as f64)).clamp(0.0, 1.0);
                    if out_edges.is_empty() || out_cov < 0.005 {
                        return None;
                    }
                    let rel = ((w_tot / (w_tot + 1.2)).clamp(0.05, 0.55)
                        * out_cov.sqrt().clamp(0.15, 0.75))
                    .min(0.55);
                    out_edges.shrink_to_fit();
                    Some(((a, b), rel, out_edges))
                })
                .collect();

            for ((a, b), rel, edges) in r2_res {
                let s: f64 = edges.iter().map(|e| e.2 as f64).sum();
                let k = pair_key(a, b);
                meta_map.insert(k, (rel, s));
                edges_map.insert(k, edges);
                nbrs[a].push(b);
                nbrs[b].push(a);
            }
        }
    }

    for list in &mut nbrs {
        list.sort_unstable();
    }
    let mut active_keys: Vec<(usize, usize)> = edges_map
        .keys()
        .map(|&k| ((k >> 32) as usize, (k & 0xFFFF_FFFF) as usize))
        .collect();
    active_keys.sort_unstable();

    ConsistentEdgeLookup {
        active_keys,
        edges_map,
        meta_map,
        nbrs,
        bubble_x_by_seq,
    }
}

/// Component-level phylogenetic redundancy de-replication weights `\nu_t \in [0.08, 1.0]`,
/// independent of tree rooting and split order.
fn compute_taxon_nu(dist_mat: &[f64], m: usize) -> Vec<f64> {
    (0..m)
        .map(|t| {
            let mut sum_sim = 0.0f64;
            for u in 0..m {
                if u != t {
                    sum_sim += (-3.0 * dist_mat[t * m + u].max(0.0)).exp();
                }
            }
            (1.0 + sum_sim).powf(-0.70).clamp(0.08, 1.0)
        })
        .collect()
}

/// Build a phylogenetically de-replicated fractional residue count profile and its
/// biophysical Dirichlet-mixture column log marginal likelihood for a profile column.
fn build_col_profile(
    col: &[(usize, usize)],
    mode: SeqMode,
    comp_seqs: &[EncodedSeq],
    _gc: &GeneticCode,
    taxon_nu: &[f64],
    k_dim: usize,
) -> (Vec<f64>, f64, f64) {
    let mut counts = vec![0.0f64; k_dim];
    let mut n_eff = 0.0f64;
    for &(tr, posr) in col {
        let w = taxon_nu[tr];
        let sym = comp_seqs[tr].sym_proj[posr] as usize;
        if sym < k_dim {
            counts[sym] += w;
            n_eff += w;
        }
    }
    let ln_p = column_niche_mixture_ln_prob(mode, &counts);
    (counts, n_eff, ln_p)
}

/// Align two `SubtreeProfile`s via the 5-State Affine Homology-Horizon Automaton using
/// fixed taxon weights `taxon_nu` and symmetric relative-affinity partner selection.
fn align_subprofiles_5state(
    left: SubtreeProfile,
    right: SubtreeProfile,
    lookup: &ConsistentEdgeLookup,
    comp_seqs: &[EncodedSeq],
    gc: &GeneticCode,
    dist_mat: &[f64],
    taxon_nu: &[f64],
    m_comp: usize,
) -> SubtreeProfile {
    let ml = left.cols.len();
    let mr = right.cols.len();
    let mode = comp_seqs.first().map(|s| s.mode).unwrap_or(SeqMode::Protein);
    let k_dim = if mode == SeqMode::Dna { 4 } else { 20 };

    let l_prof: Vec<(Vec<f64>, f64, f64)> = left
        .cols
        .iter()
        .map(|c| build_col_profile(c, mode, comp_seqs, gc, taxon_nu, k_dim))
        .collect();
    let r_prof: Vec<(Vec<f64>, f64, f64)> = right
        .cols
        .iter()
        .map(|c| build_col_profile(c, mode, comp_seqs, gc, taxon_nu, k_dim))
        .collect();

    // Compute sparse phylogenetic pair weights (redundancy-normalized across left x right)
    let nl = left.taxa.len();
    let nr = right.taxa.len();
    let mut right_slot = vec![usize::MAX; m_comp];
    for (ri, &tb) in right.taxa.iter().enumerate() {
        right_slot[tb] = ri;
    }

    let mut raw_l_nbrs: Vec<Vec<(usize, f64)>> = vec![Vec::new(); nl];
    let mut best_l = vec![0.0f64; nl];
    let mut best_r = vec![0.0f64; nr];
    let mut raw_r_nbrs: Vec<Vec<(usize, f64)>> = if nl + nr > 8 && nl * nr > 32 {
        vec![Vec::new(); nr]
    } else {
        Vec::new()
    };

    for (li, &ta) in left.taxa.iter().enumerate() {
        for &tb in &lookup.nbrs[ta] {
            let ri = right_slot[tb];
            if ri != usize::MAX {
                let d = dist_mat[ta * m_comp + tb];
                let w = taxon_nu[ta]
                    * taxon_nu[tb]
                    * lookup.pair_rel(ta, tb)
                    * (-0.45 * d).exp();
                if w >= 1e-4 {
                    raw_l_nbrs[li].push((ri, w));
                    if w > best_l[li] {
                        best_l[li] = w;
                    }
                    if w > best_r[ri] {
                        best_r[ri] = w;
                    }
                    if !raw_r_nbrs.is_empty() {
                        raw_r_nbrs[ri].push((li, w));
                    }
                }
            }
        }
    }

    // Balanced-union vs unbalanced-cap partner filtering for large splits:
    // preserves full cross-subfamily bridges on balanced BB30-scale splits while
    // preventing 20 redundant sisters from diluting 1-taxon / small-clade LOO merges.
    let kept_l_pairs: Vec<Vec<(usize, f64)>> = if nl + nr > 8 && nl * nr > 32 {
        const TOP_K_PARTNERS: usize = 10;
        let mut keep_l_set: HashSet<u64> = HashSet::new();
        let mut keep_r_set: HashSet<u64> = HashSet::new();
        let mut cands: Vec<(f64, usize)> = Vec::new();
        for li in 0..nl {
            cands.clear();
            let thresh = 0.12 * best_l[li];
            for &(ri, w) in &raw_l_nbrs[li] {
                if w >= thresh {
                    cands.push((w, right.taxa[ri]));
                }
            }
            cands.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            for &(_, tb) in cands.iter().take(TOP_K_PARTNERS) {
                keep_l_set.insert(((li as u64) << 32) | (right_slot[tb] as u64));
            }
        }
        for ri in 0..nr {
            cands.clear();
            let thresh = 0.12 * best_r[ri];
            for &(li, w) in &raw_r_nbrs[ri] {
                if w >= thresh {
                    cands.push((w, left.taxa[li]));
                }
            }
            cands.sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            for &(_, ta) in cands.iter().take(TOP_K_PARTNERS) {
                if let Ok(li) = left.taxa.binary_search(&ta) {
                    keep_r_set.insert(((li as u64) << 32) | (ri as u64));
                }
            }
        }

        let mut kept: Vec<Vec<(usize, f64)>> = vec![Vec::new(); nl];
        if nl * 3 >= nr && nr * 3 >= nl {
            for li in 0..nl {
                for &(ri, w) in &raw_l_nbrs[li] {
                    let k = ((li as u64) << 32) | (ri as u64);
                    if keep_l_set.contains(&k) || keep_r_set.contains(&k) {
                        kept[li].push((right.taxa[ri], w));
                    }
                }
            }
        } else if nl < nr {
            let mut extra: Vec<(f64, usize, usize)> = Vec::new();
            for li in 0..nl {
                for &(ri, w) in &raw_l_nbrs[li] {
                    let k = ((li as u64) << 32) | (ri as u64);
                    if keep_l_set.contains(&k) {
                        kept[li].push((right.taxa[ri], w));
                    } else if keep_r_set.contains(&k) {
                        extra.push((w, left.taxa[li], right.taxa[ri]));
                    }
                }
            }
            extra.sort_unstable_by(|a, b| {
                b.0.total_cmp(&a.0)
                    .then_with(|| a.1.cmp(&b.1))
                    .then_with(|| a.2.cmp(&b.2))
            });
            for &(w, ta, tb) in extra.iter().take(8 * nl) {
                if let Ok(li) = left.taxa.binary_search(&ta) {
                    kept[li].push((tb, w));
                }
            }
        } else {
            let mut extra: Vec<(f64, usize, usize)> = Vec::new();
            for li in 0..nl {
                for &(ri, w) in &raw_l_nbrs[li] {
                    let k = ((li as u64) << 32) | (ri as u64);
                    if keep_r_set.contains(&k) {
                        kept[li].push((right.taxa[ri], w));
                    } else if keep_l_set.contains(&k) {
                        extra.push((w, left.taxa[li], right.taxa[ri]));
                    }
                }
            }
            extra.sort_unstable_by(|a, b| {
                b.0.total_cmp(&a.0)
                    .then_with(|| a.1.cmp(&b.1))
                    .then_with(|| a.2.cmp(&b.2))
            });
            for &(w, ta, tb) in extra.iter().take(8 * nr) {
                if let Ok(li) = left.taxa.binary_search(&ta) {
                    kept[li].push((tb, w));
                }
            }
        }
        kept
    } else {
        raw_l_nbrs
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|(ri, w)| (right.taxa[ri], w))
                    .collect()
            })
            .collect()
    };

    let mut left_slot = vec![usize::MAX; m_comp];
    for (li, &ta) in left.taxa.iter().enumerate() {
        left_slot[ta] = li;
    }
    let mut right_stamp = vec![0usize; m_comp];
    let mut cur_stamp = 0usize;

    let l_bub: Vec<f64> = left
        .cols
        .iter()
        .map(|col| {
            if col.is_empty() {
                0.0
            } else {
                col.iter()
                    .map(|&(ta, ia)| lookup.bubble_x_by_seq[ta][ia])
                    .sum::<f64>()
                    / (col.len() as f64)
            }
        })
        .collect();
    let r_bub: Vec<f64> = right
        .cols
        .iter()
        .map(|col| {
            if col.is_empty() {
                0.0
            } else {
                col.iter()
                    .map(|&(tb, ib)| lookup.bubble_x_by_seq[tb][ib])
                    .sum::<f64>()
                    / (col.len() as f64)
            }
        })
        .collect();
    let l_is_anc: Vec<bool> = left
        .cols
        .iter()
        .map(|col| 2 * col.len() >= left.taxa.len())
        .collect();
    let r_is_anc: Vec<bool> = right
        .cols
        .iter()
        .map(|col| 2 * col.len() >= right.taxa.len())
        .collect();

    // Sparse cross-column homology score S_fine(p, q) for p in 0..ml, q in 0..mr
    let mut right_pos_to_col = vec![Vec::new(); nr];
    for (ri, &t) in right.taxa.iter().enumerate() {
        right_pos_to_col[ri] = vec![usize::MAX; comp_seqs[t].len()];
    }
    for (q, col_r) in right.cols.iter().enumerate() {
        for &(tb, ib) in col_r {
            right_pos_to_col[right_slot[tb]][ib] = q;
        }
    }

    let mut left_pos_to_col = vec![Vec::new(); nl];
    for (li, &ta) in left.taxa.iter().enumerate() {
        left_pos_to_col[li] = vec![usize::MAX; comp_seqs[ta].len()];
    }
    for (p, col_l) in left.cols.iter().enumerate() {
        for &(ta, ia) in col_l {
            left_pos_to_col[left_slot[ta]][ia] = p;
        }
    }

    let mut row_cands: Vec<Vec<(usize, f64, f64)>> = vec![Vec::new(); ml];
    for (li, &ta) in left.taxa.iter().enumerate() {
        let lp_col = &left_pos_to_col[li];
        for &(tb, w) in &kept_l_pairs[li] {
            if w == 0.0 {
                continue;
            }
            let rp_col = &right_pos_to_col[right_slot[tb]];
            if ta < tb {
                for &(ia, ib, prob) in lookup.pair_edges_lt(ta, tb) {
                    let p = lp_col[ia as usize];
                    let q = rp_col[ib as usize];
                    if p != usize::MAX && q != usize::MAX {
                        let pf = prob as f64;
                        row_cands[p].push((q, w * pf, pf));
                    }
                }
            } else {
                for &(ib, ia, prob) in lookup.pair_edges_lt(tb, ta) {
                    let p = lp_col[ia as usize];
                    let q = rp_col[ib as usize];
                    if p != usize::MAX && q != usize::MAX {
                        let pf = prob as f64;
                        row_cands[p].push((q, w * pf, pf));
                    }
                }
            }
        }
    }

    for vec_p in &mut row_cands {
        if vec_p.len() > 1 {
            vec_p.sort_unstable_by_key(|x| x.0);
            let mut w_idx = 0usize;
            for r_idx in 1..vec_p.len() {
                if vec_p[r_idx].0 == vec_p[w_idx].0 {
                    vec_p[w_idx].1 += vec_p[r_idx].1;
                    vec_p[w_idx].2 += vec_p[r_idx].2;
                } else {
                    w_idx += 1;
                    vec_p[w_idx] = vec_p[r_idx];
                }
            }
            vec_p.truncate(w_idx + 1);
        }
    }

    // 5-State Affine Homology-Horizon Automaton:
    //   S_PRE (0): Before first merged column (zero overhang cost)
    //   S_M   (1): Active fine-homology merged corridor
    //   S_L   (2): One-sided Left indel after S_M (clean affine deletion/insertion)
    //   S_R   (3): One-sided Right indel after S_M (clean affine deletion/insertion)
    //   S_BUB (4): Bilateral unmerged Horizon Bubble after S_M (requires multi-column anchor to re-enter S_M,
    //              preventing spurious 1-residue "confetti" pinches inside disordered/turned-over loops!)
    const N_MS: usize = 5;
    const S_PRE: usize = 0;
    const S_M: usize = 1;
    const S_L: usize = 2;
    const S_R: usize = 3;
    const S_BUB: usize = 4;
    const INDEL_OPEN_COST: f64 = 0.10;
    const HORIZON_REENTRY_COST: f64 = 0.90;

    let mut joint_counts = vec![0.0f64; k_dim];
    let mut eval_merge_gain = |p_idx: usize, q_idx: usize, w_prob_sum: f64| -> Option<f64> {
        cur_stamp += 1;
        for &(tb, _) in &right.cols[q_idx] {
            right_stamp[tb] = cur_stamp;
        }
        let mut denom = 0.0;
        for &(ta, _) in &left.cols[p_idx] {
            for &(tb, w) in &kept_l_pairs[left_slot[ta]] {
                if right_stamp[tb] == cur_stamp {
                    denom += w;
                }
            }
        }
        let s_fine = if denom > 0.0 { w_prob_sum / denom } else { 0.0 };
        let max_bub = l_bub[p_idx].max(r_bub[q_idx]);
        let p_eff = if left.cols[p_idx].len() + right.cols[q_idx].len() >= 3
            && s_fine >= 0.15
            && max_bub < 0.40
        {
            let bub_gate = ((0.40 - max_bub) / 0.40).clamp(0.0, 1.0);
            let (ref lc, l_n, l_lp) = l_prof[p_idx];
            let (ref rc, r_n, r_lp) = r_prof[q_idx];
            for k in 0..k_dim {
                joint_counts[k] = lc[k] + rc[k];
            }
            let delta_col = column_niche_mixture_ln_prob(mode, &joint_counts) - l_lp - r_lp;
            let excess_n = (l_n - 1.0).max(0.0) + (r_n - 1.0).max(0.0);
            let eta = excess_n / (excess_n + 2.0);
            let s_clamped = s_fine.clamp(1e-4, 1.0 - 1e-4);
            let logit = (s_clamped / (1.0 - s_clamped)).ln() + 0.65 * eta * bub_gate * delta_col;
            mdl_gain_to_prob(logit)
        } else {
            s_fine
        };
        if p_eff > FINE_MERGE_HORIZON {
            Some(p_eff - FINE_MERGE_HORIZON)
        } else {
            None
        }
    };

    #[inline(always)]
    fn relax_dp(
        row: &mut [f64],
        bt: &mut [u8],
        dst_r: usize,
        dst_b: usize,
        cand_val: f64,
        op: u8,
        prev_s: usize,
    ) {
        if cand_val > row[dst_r] {
            row[dst_r] = cand_val;
            bt[dst_b] = (op << 3) | (prev_s as u8);
        }
    }

    let ops: Vec<u8> = if ml.max(mr) <= 1_200 {
        let w_dp = mr + 1;
        let row_dp_len = w_dp * N_MS;
        let mut dp_prev = vec![f64::NEG_INFINITY; row_dp_len];
        let mut dp_cur = vec![f64::NEG_INFINITY; row_dp_len];
        let mut bt = vec![0u8; (ml + 1) * w_dp * N_MS]; // packed: (op << 3) | prev_state

        for p in 0..=ml {
            dp_cur.fill(f64::NEG_INFINITY);
            if p == 0 {
                dp_cur[S_PRE] = 0.0;
            }
            let mut cand_cursor = 0usize;
            for q in 0..=mr {
                if p == 0 && q == 0 {
                    continue;
                }
                if p >= 1 {
                    if l_is_anc[p - 1] {
                        let v_pre = dp_prev[q * N_MS + S_PRE];
                        relax_dp(&mut dp_cur, &mut bt, q * N_MS + S_PRE, (p * w_dp + q) * N_MS + S_PRE, v_pre, 2, S_PRE);

                        let v_m = dp_prev[q * N_MS + S_M] - INDEL_OPEN_COST;
                        relax_dp(&mut dp_cur, &mut bt, q * N_MS + S_L, (p * w_dp + q) * N_MS + S_L, v_m, 2, S_M);
                        let v_l = dp_prev[q * N_MS + S_L];
                        relax_dp(&mut dp_cur, &mut bt, q * N_MS + S_L, (p * w_dp + q) * N_MS + S_L, v_l, 2, S_L);

                        let v_r = dp_prev[q * N_MS + S_R];
                        relax_dp(&mut dp_cur, &mut bt, q * N_MS + S_BUB, (p * w_dp + q) * N_MS + S_BUB, v_r, 2, S_R);
                        let v_b = dp_prev[q * N_MS + S_BUB];
                        relax_dp(&mut dp_cur, &mut bt, q * N_MS + S_BUB, (p * w_dp + q) * N_MS + S_BUB, v_b, 2, S_BUB);
                    } else {
                        for s in 0..N_MS {
                            let v_same = dp_prev[q * N_MS + s];
                            relax_dp(&mut dp_cur, &mut bt, q * N_MS + s, (p * w_dp + q) * N_MS + s, v_same, 2, s);
                        }
                    }
                }
                if q >= 1 {
                    if r_is_anc[q - 1] {
                        let v_pre = dp_cur[(q - 1) * N_MS + S_PRE];
                        relax_dp(&mut dp_cur, &mut bt, q * N_MS + S_PRE, (p * w_dp + q) * N_MS + S_PRE, v_pre, 3, S_PRE);

                        let v_m = dp_cur[(q - 1) * N_MS + S_M] - INDEL_OPEN_COST;
                        relax_dp(&mut dp_cur, &mut bt, q * N_MS + S_R, (p * w_dp + q) * N_MS + S_R, v_m, 3, S_M);
                        let v_r = dp_cur[(q - 1) * N_MS + S_R];
                        relax_dp(&mut dp_cur, &mut bt, q * N_MS + S_R, (p * w_dp + q) * N_MS + S_R, v_r, 3, S_R);

                        let v_l = dp_cur[(q - 1) * N_MS + S_L];
                        relax_dp(&mut dp_cur, &mut bt, q * N_MS + S_BUB, (p * w_dp + q) * N_MS + S_BUB, v_l, 3, S_L);
                        let v_b = dp_cur[(q - 1) * N_MS + S_BUB];
                        relax_dp(&mut dp_cur, &mut bt, q * N_MS + S_BUB, (p * w_dp + q) * N_MS + S_BUB, v_b, 3, S_BUB);
                    } else {
                        for s in 0..N_MS {
                            let v_same = dp_cur[(q - 1) * N_MS + s];
                            relax_dp(&mut dp_cur, &mut bt, q * N_MS + s, (p * w_dp + q) * N_MS + s, v_same, 3, s);
                        }
                    }
                }
                if p >= 1 && q >= 1 {
                    let row = &row_cands[p - 1];
                    while cand_cursor < row.len() && row[cand_cursor].0 < q - 1 {
                        cand_cursor += 1;
                    }
                    if cand_cursor < row.len() && row[cand_cursor].0 == q - 1 {
                        if let Some(gain) = eval_merge_gain(p - 1, q - 1, row[cand_cursor].1) {
                            let dst_r = q * N_MS + S_M;
                            let dst_b = (p * w_dp + q) * N_MS + S_M;
                            for &ps in &[S_PRE, S_M, S_L, S_R] {
                                let cand = dp_prev[(q - 1) * N_MS + ps] + gain;
                                relax_dp(&mut dp_cur, &mut bt, dst_r, dst_b, cand, 1, ps);
                            }
                            let cand_bub =
                                dp_prev[(q - 1) * N_MS + S_BUB] + gain - HORIZON_REENTRY_COST;
                            relax_dp(&mut dp_cur, &mut bt, dst_r, dst_b, cand_bub, 1, S_BUB);
                        }
                    }
                }
            }
            std::mem::swap(&mut dp_prev, &mut dp_cur);
        }

        let mut best_end_s = S_PRE;
        let mut best_end_val = dp_prev[mr * N_MS + S_PRE];
        for s in 1..N_MS {
            let refund = if s >= S_L { INDEL_OPEN_COST } else { 0.0 };
            let val = dp_prev[mr * N_MS + s] + refund;
            if val > best_end_val {
                best_end_val = val;
                best_end_s = s;
            }
        }

        let mut ops_unbanded: Vec<u8> = Vec::with_capacity(ml + mr);
        let mut cp = ml;
        let mut cq = mr;
        let mut cs = best_end_s;
        while cp > 0 || cq > 0 {
            let packed = bt[(cp * w_dp + cq) * N_MS + cs];
            let op = packed >> 3;
            let prev_s = (packed & 7) as usize;
            ops_unbanded.push(op);
            match op {
                1 => {
                    cp -= 1;
                    cq -= 1;
                }
                2 => {
                    cp -= 1;
                }
                3 => {
                    cq -= 1;
                }
                _ => unreachable!(),
            }
            cs = prev_s;
        }
        ops_unbanded.reverse();
        ops_unbanded
    } else {
        // Long-genome banded 5-state DP via exact sparse Fenwick backbone chain extraction
        let mut pos_rows: Vec<Vec<(usize, f64)>> = vec![Vec::new(); ml];
        let mut k_pos = 0usize;
        for p_idx in 0..ml {
            for &(q_idx, w_prob_sum, _) in &row_cands[p_idx] {
                if let Some(gain) = eval_merge_gain(p_idx, q_idx, w_prob_sum) {
                    pos_rows[p_idx].push((q_idx + 1, gain));
                    k_pos += 1;
                }
            }
        }

        if k_pos == 0 {
            let mut ops_empty = Vec::with_capacity(ml + mr);
            ops_empty.resize(ml, 2u8);
            ops_empty.resize(ml + mr, 3u8);
            ops_empty
        } else {
            // Exact O(K log mr) monotonic Fenwick max-tree over positive-gain merges
            let mut bit: Vec<(f64, usize)> = vec![(0.0, usize::MAX); mr + 1];
            let mut chain_nodes: Vec<(usize, usize, usize)> = Vec::with_capacity(k_pos);
            let mut row_tmp: Vec<(usize, usize, f64, usize)> = Vec::new();
            let mut best_chain_sc = -1.0f64;
            let mut best_chain_end = usize::MAX;

            for p in 1..=ml {
                if pos_rows[p - 1].is_empty() {
                    continue;
                }
                row_tmp.clear();
                for &(q, gain) in &pos_rows[p - 1] {
                    let mut idx = q - 1;
                    let mut b_sc = 0.0f64;
                    let mut b_idx = usize::MAX;
                    while idx > 0 {
                        let (v, vi) = bit[idx];
                        if v > b_sc {
                            b_sc = v;
                            b_idx = vi;
                        }
                        idx -= idx & idx.wrapping_neg();
                    }
                    row_tmp.push((p, q, b_sc + gain, b_idx));
                }
                for &(cp, cq, sc, prev_idx) in &row_tmp {
                    let cid = chain_nodes.len();
                    chain_nodes.push((cp, cq, prev_idx));
                    if sc > best_chain_sc {
                        best_chain_sc = sc;
                        best_chain_end = cid;
                    }
                    let mut idx = cq;
                    while idx <= mr {
                        if sc > bit[idx].0 {
                            bit[idx] = (sc, cid);
                        }
                        idx += idx & idx.wrapping_neg();
                    }
                }
            }

            let mut chain: Vec<(usize, usize)> = Vec::new();
            let mut cur = best_chain_end;
            while cur != usize::MAX {
                let (cp, cq, prev) = chain_nodes[cur];
                chain.push((cp, cq));
                cur = prev;
            }
            chain.reverse();

            const W_PROF: usize = 192;
            let mut q_lo = vec![mr; ml + 1];
            let mut q_hi = vec![0usize; ml + 1];
            let mut mark = |p: usize, lo: usize, hi: usize| {
                q_lo[p] = q_lo[p].min(lo.min(mr));
                q_hi[p] = q_hi[p].max(hi.min(mr));
            };
            let mark_span = |mark_fn: &mut dyn FnMut(usize, usize, usize),
                             pa: usize,
                             qa: usize,
                             pb: usize,
                             qb: usize| {
                let dp = pb - pa;
                let dq = qb - qa;
                if dp.max(dq) <= 1_400 && dp.saturating_mul(dq) <= 1_000_000 {
                    let lo = qa.saturating_sub(W_PROF);
                    let hi = (qb + W_PROF).min(mr);
                    for p in pa..=pb {
                        mark_fn(p, lo, hi);
                    }
                } else {
                    mark_fn(pa, qa.saturating_sub(W_PROF), (qa + W_PROF).min(mr));
                    mark_fn(pb, qb.saturating_sub(W_PROF), (qb + W_PROF).min(mr));
                    for p in pa..=pb {
                        let center = qa + (p - pa) * dq / dp.max(1);
                        mark_fn(p, center.saturating_sub(W_PROF), (center + W_PROF).min(mr));
                    }
                }
            };

            let (p0, q0) = chain[0];
            mark_span(&mut mark, 0, 0, p0, q0);
            for r in 0..chain.len().saturating_sub(1) {
                let (pa, qa) = chain[r];
                let (pb, qb) = chain[r + 1];
                mark_span(&mut mark, pa, qa, pb, qb);
            }
            let &(pl, ql) = chain.last().unwrap();
            mark_span(&mut mark, pl, ql, ml, mr);

            // Include off-chain positive candidates within 384 cols of the preliminary band
            for p in 1..=ml {
                let lo_lim = q_lo[p].saturating_sub(384);
                let hi_lim = (q_hi[p] + 384).min(mr);
                for &(q, _) in &pos_rows[p - 1] {
                    if q >= lo_lim && q <= hi_lim {
                        q_lo[p] = q_lo[p].min(q.saturating_sub(8));
                        q_hi[p] = q_hi[p].max((q + 8).min(mr));
                    }
                }
            }

            q_lo[0] = 0;
            q_hi[ml] = mr;
            for p in (0..ml).rev() {
                q_lo[p] = q_lo[p].min(q_lo[p + 1]);
            }
            for p in 1..=ml {
                q_hi[p] = q_hi[p].max(q_hi[p - 1]);
            }
            for p in 0..ml {
                if q_lo[p + 1] > q_hi[p] {
                    q_hi[p] = q_lo[p + 1];
                }
            }
            for p in 1..=ml {
                q_hi[p] = q_hi[p].max(q_hi[p - 1]);
            }

            let mut row_off = Vec::with_capacity(ml + 2);
            row_off.push(0usize);
            let mut max_w = 0usize;
            for p in 0..=ml {
                let w = q_hi[p] - q_lo[p] + 1;
                if w > max_w {
                    max_w = w;
                }
                row_off.push(row_off[p] + w);
            }

            let mut dp_prev = vec![f64::NEG_INFINITY; max_w * N_MS];
            let mut dp_cur = vec![f64::NEG_INFINITY; max_w * N_MS];
            let mut bt = vec![0u8; row_off[ml + 1] * N_MS];

            for p in 0..=ml {
                let c_lo = q_lo[p];
                let c_hi = q_hi[p];
                let c_w = c_hi - c_lo + 1;
                dp_cur[..c_w * N_MS].fill(f64::NEG_INFINITY);
                if p == 0 {
                    dp_cur[S_PRE] = 0.0;
                }
                let (p_lo, p_hi) = if p >= 1 {
                    (q_lo[p - 1], q_hi[p - 1])
                } else {
                    (1, 0)
                };
                let c_base = row_off[p];
                let mut cand_cursor = 0usize;
                let row = if p >= 1 { &pos_rows[p - 1][..] } else { &[] };

                for q in c_lo..=c_hi {
                    if p == 0 && q == 0 {
                        continue;
                    }
                    let qc = q - c_lo;
                    let dst_b_base = (c_base + qc) * N_MS;
                    if p >= 1 && q >= p_lo && q <= p_hi {
                        let qp = q - p_lo;
                        if l_is_anc[p - 1] {
                            let v_pre = dp_prev[qp * N_MS + S_PRE];
                            relax_dp(&mut dp_cur, &mut bt, qc * N_MS + S_PRE, dst_b_base + S_PRE, v_pre, 2, S_PRE);

                            let v_m = dp_prev[qp * N_MS + S_M] - INDEL_OPEN_COST;
                            relax_dp(&mut dp_cur, &mut bt, qc * N_MS + S_L, dst_b_base + S_L, v_m, 2, S_M);
                            let v_l = dp_prev[qp * N_MS + S_L];
                            relax_dp(&mut dp_cur, &mut bt, qc * N_MS + S_L, dst_b_base + S_L, v_l, 2, S_L);

                            let v_r = dp_prev[qp * N_MS + S_R];
                            relax_dp(&mut dp_cur, &mut bt, qc * N_MS + S_BUB, dst_b_base + S_BUB, v_r, 2, S_R);
                            let v_b = dp_prev[qp * N_MS + S_BUB];
                            relax_dp(&mut dp_cur, &mut bt, qc * N_MS + S_BUB, dst_b_base + S_BUB, v_b, 2, S_BUB);
                        } else {
                            for s in 0..N_MS {
                                let v_same = dp_prev[qp * N_MS + s];
                                relax_dp(&mut dp_cur, &mut bt, qc * N_MS + s, dst_b_base + s, v_same, 2, s);
                            }
                        }
                    }
                    if q > c_lo {
                        if r_is_anc[q - 1] {
                            let v_pre = dp_cur[(qc - 1) * N_MS + S_PRE];
                            relax_dp(&mut dp_cur, &mut bt, qc * N_MS + S_PRE, dst_b_base + S_PRE, v_pre, 3, S_PRE);

                            let v_m = dp_cur[(qc - 1) * N_MS + S_M] - INDEL_OPEN_COST;
                            relax_dp(&mut dp_cur, &mut bt, qc * N_MS + S_R, dst_b_base + S_R, v_m, 3, S_M);
                            let v_r = dp_cur[(qc - 1) * N_MS + S_R];
                            relax_dp(&mut dp_cur, &mut bt, qc * N_MS + S_R, dst_b_base + S_R, v_r, 3, S_R);

                            let v_l = dp_cur[(qc - 1) * N_MS + S_L];
                            relax_dp(&mut dp_cur, &mut bt, qc * N_MS + S_BUB, dst_b_base + S_BUB, v_l, 3, S_L);
                            let v_b = dp_cur[(qc - 1) * N_MS + S_BUB];
                            relax_dp(&mut dp_cur, &mut bt, qc * N_MS + S_BUB, dst_b_base + S_BUB, v_b, 3, S_BUB);
                        } else {
                            for s in 0..N_MS {
                                let v_same = dp_cur[(qc - 1) * N_MS + s];
                                relax_dp(&mut dp_cur, &mut bt, qc * N_MS + s, dst_b_base + s, v_same, 3, s);
                            }
                        }
                    }
                    if p >= 1 && q >= 1 && (q - 1) >= p_lo && (q - 1) <= p_hi {
                        while cand_cursor < row.len() && row[cand_cursor].0 < q {
                            cand_cursor += 1;
                        }
                        if cand_cursor < row.len() && row[cand_cursor].0 == q {
                            let gain = row[cand_cursor].1;
                            let qp_prev = q - 1 - p_lo;
                            let dst_r = qc * N_MS + S_M;
                            let dst_b = dst_b_base + S_M;
                            for &ps in &[S_PRE, S_M, S_L, S_R] {
                                let cand = dp_prev[qp_prev * N_MS + ps] + gain;
                                relax_dp(&mut dp_cur, &mut bt, dst_r, dst_b, cand, 1, ps);
                            }
                            let cand_bub =
                                dp_prev[qp_prev * N_MS + S_BUB] + gain - HORIZON_REENTRY_COST;
                            relax_dp(&mut dp_cur, &mut bt, dst_r, dst_b, cand_bub, 1, S_BUB);
                        }
                    }
                }
                std::mem::swap(&mut dp_prev, &mut dp_cur);
            }

            let end_q = mr - q_lo[ml];
            let mut best_end_s = S_PRE;
            let mut best_end_val = dp_prev[end_q * N_MS + S_PRE];
            for s in 1..N_MS {
                let refund = if s >= S_L { INDEL_OPEN_COST } else { 0.0 };
                let val = dp_prev[end_q * N_MS + s] + refund;
                if val > best_end_val {
                    best_end_val = val;
                    best_end_s = s;
                }
            }

            let mut ops_banded: Vec<u8> = Vec::with_capacity(ml + mr);
            let mut cp = ml;
            let mut cq = mr;
            let mut cs = best_end_s;
            while cp > 0 || cq > 0 {
                let packed = bt[(row_off[cp] + (cq - q_lo[cp])) * N_MS + cs];
                let op = packed >> 3;
                let prev_s = (packed & 7) as usize;
                ops_banded.push(op);
                match op {
                    1 => {
                        cp -= 1;
                        cq -= 1;
                    }
                    2 => {
                        cp -= 1;
                    }
                    3 => {
                        cq -= 1;
                    }
                    _ => unreachable!(),
                }
                cs = prev_s;
            }
            ops_banded.reverse();
            ops_banded
        }
    };

    let mut merged_cols = Vec::with_capacity(ops.len());
    let mut lp = 0usize;
    let mut rq = 0usize;

    for &op in &ops {
        match op {
            1 => {
                let mut col = Vec::with_capacity(left.cols[lp].len() + right.cols[rq].len());
                col.extend_from_slice(&left.cols[lp]);
                col.extend_from_slice(&right.cols[rq]);
                col.sort_unstable_by_key(|&(t, _)| t);
                merged_cols.push(col);
                lp += 1;
                rq += 1;
            }
            2 => {
                merged_cols.push(left.cols[lp].clone());
                lp += 1;
            }
            3 => {
                merged_cols.push(right.cols[rq].clone());
                rq += 1;
            }
            _ => unreachable!(),
        }
    }

    let mut taxa = left.taxa;
    taxa.extend(right.taxa);
    taxa.sort_unstable();

    SubtreeProfile {
        taxa,
        cols: merged_cols,
    }
}

fn assemble_profile_on_tree(
    node_id: usize,
    tree_nodes: &[TreeNodeRecord],
    lookup: &ConsistentEdgeLookup,
    comp_seqs: &[EncodedSeq],
    gc: &GeneticCode,
    dist_mat: &[f64],
    taxon_nu: &[f64],
    m_comp: usize,
) -> SubtreeProfile {
    let rec = &tree_nodes[node_id];
    if rec.is_leaf {
        let local_idx = rec.local_seq_idx.expect("leaf missing local_seq_idx");
        let l = comp_seqs[local_idx].len();
        let cols = (0..l).map(|pos| vec![(local_idx, pos)]).collect();
        return SubtreeProfile {
            taxa: vec![local_idx],
            cols,
        };
    }
    let left_id = rec.left_node_id.expect("internal node missing left_node_id");
    let right_id = rec.right_node_id.expect("internal node missing right_node_id");
    let (pl, pr) = if m_comp >= 16 && !tree_nodes[left_id].is_leaf && !tree_nodes[right_id].is_leaf
    {
        join(
            || {
                assemble_profile_on_tree(
                    left_id,
                    tree_nodes,
                    lookup,
                    comp_seqs,
                    gc,
                    dist_mat,
                    taxon_nu,
                    m_comp,
                )
            },
            || {
                assemble_profile_on_tree(
                    right_id,
                    tree_nodes,
                    lookup,
                    comp_seqs,
                    gc,
                    dist_mat,
                    taxon_nu,
                    m_comp,
                )
            },
        )
    } else {
        (
            assemble_profile_on_tree(
                left_id,
                tree_nodes,
                lookup,
                comp_seqs,
                gc,
                dist_mat,
                taxon_nu,
                m_comp,
            ),
            assemble_profile_on_tree(
                right_id,
                tree_nodes,
                lookup,
                comp_seqs,
                gc,
                dist_mat,
                taxon_nu,
                m_comp,
            ),
        )
    };
    align_subprofiles_5state(pl, pr, lookup, comp_seqs, gc, dist_mat, taxon_nu, m_comp)
}

#[inline]
fn uf_find(p: &mut [usize], x: usize) -> usize {
    let mut r = x;
    while p[r] != r {
        r = p[r];
    }
    let mut c = x;
    while p[c] != r {
        let nxt = p[c];
        p[c] = r;
        c = nxt;
    }
    r
}

#[inline]
fn is_clean_seq(p: &PreparedSeq) -> bool {
    p.seq.mode != SeqMode::Codon
        || (p.seq.nt_tokens.len() % 3 == 0 && !p.seq.has_unambiguous_internal_stop)
}

#[inline]
fn needs_frameshift_fallback(px: &PreparedSeq, py: &PreparedSeq) -> bool {
    if px.seq.mode != SeqMode::Codon {
        return false;
    }
    !is_clean_seq(px) || !is_clean_seq(py)
}

fn eval_pair_with_fallback(
    px: &PreparedSeq,
    py: &PreparedSeq,
    gc: &GeneticCode,
    tables: &GlobalRegimeTables,
    compute_posteriors: bool,
) -> (PairHomologyResult, Option<PairForwardState>) {
    if needs_frameshift_fallback(px, py) {
        (
            evaluate_pair_homology(&px.seq, &py.seq, gc, compute_posteriors),
            None,
        )
    } else {
        evaluate_pair_homology_prepared(px, py, gc, tables, compute_posteriors, Vec::new(), 0.0)
    }
}

fn build_sparse_candidate_pairs_subset(prepared: &[PreparedSeq], subset: &[usize]) -> Vec<(usize, usize)> {
    let n = subset.len();
    let mut uniq_seed_cnt = vec![0usize; n];
    let mut postings: Vec<(u32, u32)> = Vec::new();
    for (a, &g_idx) in subset.iter().enumerate() {
        let p = &prepared[g_idx];
        let mut prev_k: Option<u32> = None;
        let mut cnt = 0usize;
        for &(k, _, _, _) in &p.seeds {
            if prev_k != Some(k) {
                postings.push((k, a as u32));
                prev_k = Some(k);
                cnt += 1;
            }
        }
        uniq_seed_cnt[a] = cnt;
    }

    postings.sort_unstable();
    let mut shared_map: HashMap<u64, u16> = HashMap::with_capacity(n * 24);
    let mut idx = 0usize;
    while idx < postings.len() {
        let k = postings[idx].0;
        let start = idx;
        while idx < postings.len() && postings[idx].0 == k {
            idx += 1;
        }
        let g = &postings[start..idx];
        let glen = g.len();
        if glen < 2 {
            continue;
        }
        if n > 400 && glen > 128 {
            // Bounded clique sampling on massive cohorts only (> 400 seqs & > 128 group hits)
            for i in 0..glen {
                let a = g[i].1 as usize;
                for d in 1..=32 {
                    let j = (i + d) % glen;
                    if j == i {
                        continue;
                    }
                    let b = g[j].1 as usize;
                    let slot = shared_map.entry(pair_key(a, b)).or_insert(0);
                    *slot = slot.saturating_add(1);
                }
            }
        } else {
            for i in 0..glen {
                let a = g[i].1 as usize;
                for j in (i + 1)..glen {
                    let b = g[j].1 as usize;
                    let slot = shared_map.entry(pair_key(a, b)).or_insert(0);
                    *slot = slot.saturating_add(1);
                }
            }
        }
    }

    let mut adj_shared: Vec<Vec<(usize, u16)>> = vec![Vec::new(); n];
    let mut all_pos_edges: Vec<(f64, u16, usize, usize)> = Vec::with_capacity(shared_map.len());
    let mut parent_cl: Vec<usize> = (0..n).collect();
    for (&key, &s) in &shared_map {
        if s == 0 {
            continue;
        }
        let a = (key >> 32) as usize;
        let b = (key & 0xFFFF_FFFF) as usize;
        adj_shared[a].push((b, s));
        adj_shared[b].push((a, s));
        let denom = (uniq_seed_cnt[a] + uniq_seed_cnt[b])
            .saturating_sub(s as usize)
            .max(1) as f64;
        let jacc = (s as f64) / denom;
        all_pos_edges.push((jacc, s, a, b));
        if s >= 2 {
            let ra = uf_find(&mut parent_cl, a);
            let rb = uf_find(&mut parent_cl, b);
            if ra != rb {
                parent_cl[rb] = ra;
            }
        }
    }
    drop(shared_map);

    const K_CAND: usize = 20;
    let mut cand_edges: Vec<(usize, usize)> = Vec::with_capacity(n * K_CAND);
    let mut nbrs: Vec<(f64, u16, usize)> = Vec::new();
    for a in 0..n {
        nbrs.clear();
        for &(b, s) in &adj_shared[a] {
            let denom = (uniq_seed_cnt[a] + uniq_seed_cnt[b])
                .saturating_sub(s as usize)
                .max(1) as f64;
            let jacc = (s as f64) / denom;
            nbrs.push((jacc, s, b));
        }
        nbrs.sort_unstable_by(|x, y| {
            y.0.total_cmp(&x.0)
                .then_with(|| y.1.cmp(&x.1))
                .then_with(|| x.2.cmp(&y.2))
        });
        for &(_, _, b) in nbrs.iter().take(K_CAND) {
            cand_edges.push((a.min(b), a.max(b)));
        }
        if nbrs.len() < 4 {
            let strides = [
                1usize,
                (n / 4).max(1),
                (n / 2).max(2),
                ((3 * n) / 4).max(3),
            ];
            for &d in &strides {
                let b = (a + d) % n;
                if b != a {
                    cand_edges.push((a.min(b), a.max(b)));
                }
            }
        }
    }

    // Exact Kruskal Maximum Spanning Forest over shared > 0
    all_pos_edges.sort_unstable_by(|x, y| {
        y.0.total_cmp(&x.0)
            .then_with(|| y.1.cmp(&x.1))
            .then_with(|| x.2.cmp(&y.2))
            .then_with(|| x.3.cmp(&y.3))
    });
    let mut parent_msf: Vec<usize> = (0..n).collect();
    for &(_, _, a, b) in &all_pos_edges {
        let ra = uf_find(&mut parent_msf, a);
        let rb = uf_find(&mut parent_msf, b);
        if ra != rb {
            parent_msf[rb] = ra;
            cand_edges.push((a, b));
        }
    }

    // Cross-seed-cluster representatives over shared >= 2 connected components
    let mut clusters: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for a in 0..n {
        let r = uf_find(&mut parent_cl, a);
        clusters.entry(r).or_default().push(a);
    }

    let mut cluster_reps: Vec<(bool, usize, usize, Vec<usize>)> = Vec::with_capacity(clusters.len());
    for (_, mut members) in clusters {
        let size = members.len();
        members.sort_unstable_by(|&a, &b| {
            let pa = &prepared[subset[a]];
            let pb = &prepared[subset[b]];
            is_clean_seq(pb)
                .cmp(&is_clean_seq(pa))
                .then_with(|| pb.seq.non_ambig_len.cmp(&pa.seq.non_ambig_len))
                .then_with(|| pb.seq.len().cmp(&pa.seq.len()))
                .then_with(|| a.cmp(&b))
        });
        let rep0 = members[0];
        let p0 = &prepared[subset[rep0]];
        let clean0 = is_clean_seq(p0);
        let max_len = p0.seq.non_ambig_len;
        members.truncate(3);
        cluster_reps.push((clean0, size, max_len, members));
    }

    let num_clusters = cluster_reps.len();
    if num_clusters >= 2 {
        cluster_reps.sort_unstable_by(|a, b| {
            b.0.cmp(&a.0)
                .then_with(|| b.1.cmp(&a.1))
                .then_with(|| b.2.cmp(&a.2))
        });
        if num_clusters > 16 {
            for (c, entry) in cluster_reps.iter_mut().enumerate() {
                entry.3.truncate(if c < 12 { 2 } else { 1 });
            }
        }
        if num_clusters <= 40 {
            for c1 in 0..num_clusters {
                for c2 in (c1 + 1)..num_clusters {
                    for &r1 in &cluster_reps[c1].3 {
                        for &r2 in &cluster_reps[c2].3 {
                            cand_edges.push((r1.min(r2), r1.max(r2)));
                        }
                    }
                }
            }
        } else {
            let hub_count = num_clusters.min(24);
            for c1 in 0..hub_count {
                for c2 in (c1 + 1)..hub_count {
                    for &r1 in &cluster_reps[c1].3 {
                        for &r2 in &cluster_reps[c2].3 {
                            cand_edges.push((r1.min(r2), r1.max(r2)));
                        }
                    }
                }
            }
            for c in hub_count..num_clusters {
                for h in 0..hub_count.min(2) {
                    for &r1 in &cluster_reps[c].3 {
                        for &r2 in &cluster_reps[h].3 {
                            cand_edges.push((r1.min(r2), r1.max(r2)));
                        }
                    }
                }
            }
            for c in 0..num_clusters {
                let nxt = (c + 1) % num_clusters;
                if nxt != c {
                    for &r1 in &cluster_reps[c].3 {
                        for &r2 in &cluster_reps[nxt].3 {
                            cand_edges.push((r1.min(r2), r1.max(r2)));
                        }
                    }
                }
            }
        }
    }

    cand_edges.sort_unstable();
    cand_edges.dedup();
    cand_edges
}

fn ensure_pairs_evaluated_with_posteriors(
    target_pairs: &[(usize, usize)],
    prepared: &mut [PreparedSeq],
    gc: &GeneticCode,
    tables: &GlobalRegimeTables,
    pair_map: &mut HashMap<u64, PairHomologyResult>,
    state_map: &mut HashMap<u64, PairForwardState>,
) {
    if target_pairs.is_empty() {
        return;
    }
    let mut uniq: Vec<(usize, usize)> = target_pairs
        .iter()
        .filter_map(|&(a, b)| if a == b { None } else { Some((a.min(b), a.max(b))) })
        .collect();
    uniq.sort_unstable();
    uniq.dedup();

    enum PostJob {
        Populate(usize, usize, PairHomologyResult, PairForwardState),
        EvalFull(usize, usize),
    }

    let mut jobs = Vec::new();
    for (a, b) in uniq {
        let key = pair_key(a, b);
        if let Some(res_ref) = pair_map.get(&key) {
            if is_post_eval(res_ref) {
                continue;
            }
            if res_ref.mdl_gain_nats <= -8.0 {
                continue;
            }
            if let Some(st) = state_map.remove(&key) {
                let res = pair_map.remove(&key).unwrap();
                jobs.push(PostJob::Populate(a, b, res, st));
                continue;
            }
        }
        jobs.push(PostJob::EvalFull(a, b));
    }

    if jobs.is_empty() {
        return;
    }

    while !jobs.is_empty() {
        let take = jobs.len().min(2_500);
        let chunk: Vec<PostJob> = jobs.drain(..take).collect();
        let results: Vec<(usize, usize, PairHomologyResult)> = chunk
            .into_par_iter()
            .map(|job| match job {
                PostJob::Populate(a, b, mut res, st) => {
                    populate_pair_posteriors_prepared(
                        &mut res,
                        &st,
                        &prepared[a],
                        &prepared[b],
                        gc,
                        tables,
                    );
                    (a, b, res)
                }
                PostJob::EvalFull(a, b) => {
                    let (res, _) =
                        eval_pair_with_fallback(&prepared[a], &prepared[b], gc, tables, true);
                    (a, b, res)
                }
            })
            .collect();
        for (a, b, res) in results {
            pair_map.insert(pair_key(a, b), res);
        }
        if prepared.len() > 300 {
            for p in prepared.iter_mut() {
                p.reset_regime_locks();
            }
        }
    }
}

fn relax_shortest_paths_with_var(d_mat: &mut [f64], v_mat: &mut [f64], m: usize) {
    for k in 0..m {
        for i in 0..m {
            let dik = d_mat[i * m + k];
            if !dik.is_finite() {
                continue;
            }
            let vik = v_mat[i * m + k];
            for j in 0..m {
                let dkj = d_mat[k * m + j];
                if dkj.is_finite() {
                    let cand = dik + dkj;
                    if cand < d_mat[i * m + j] - 1e-12 {
                        d_mat[i * m + j] = cand;
                        v_mat[i * m + j] = vik + v_mat[k * m + j];
                    }
                }
            }
        }
    }
}

fn pair_f_min(r: &PairHomologyResult, la: usize, lb: usize) -> f64 {
    let l_min = la.min(lb).max(1) as f64;
    if is_post_eval(r) {
        let pi = mdl_gain_to_prob(r.mdl_gain_nats + 2.5).max(r.homology_posterior);
        let mut edge_mass = 0.0f64;
        for e in &r.fine_edges {
            let p_edge = if e.cond_posterior > 0.0 {
                (e.cond_posterior * pi).clamp(0.0, 1.0)
            } else {
                e.posterior
            };
            if p_edge >= 0.02 {
                edge_mass += p_edge;
            }
        }
        (edge_mass / l_min).clamp(0.04, 1.0)
    } else {
        ((r.est_f_shared * 0.5 * ((la + lb) as f64)) / l_min).clamp(0.04, 1.0)
    }
}

#[derive(Copy, Clone, PartialEq)]
struct DjNode {
    dist: f64,
    node: usize,
}
impl Eq for DjNode {}
impl Ord for DjNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .dist
            .total_cmp(&self.dist)
            .then_with(|| other.node.cmp(&self.node))
    }
}
impl PartialOrd for DjNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Decoupled, survivorship-corrected evolutionary divergence `d_evol` and estimation
/// variance `v_evol` matrices (normalized by `min(la, lb)` so true length fragments
/// incur no distortion while small partial motif hits avoid artificial twin proximity),
/// completed via Floyd-Warshall (M <= 128) or parallel multi-source sparse Dijkstra (M > 128).
fn compute_evol_dist_and_var(
    comp_indices: &[usize],
    pair_map: &HashMap<u64, PairHomologyResult>,
    seq_lens: &[usize],
    min_gain_nats: f64,
) -> (Vec<f64>, Vec<f64>) {
    let m = comp_indices.len();
    if m <= 128 {
        let mut sp_d = vec![f64::INFINITY; m * m];
        let mut sp_v = vec![f64::INFINITY; m * m];
        for a in 0..m {
            sp_d[a * m + a] = 0.0;
            sp_v[a * m + a] = 0.0;
            for b in (a + 1)..m {
                let ga = comp_indices[a];
                let gb = comp_indices[b];
                if let Some(r) = pair_map.get(&pair_key(ga, gb)) {
                    if r.mdl_gain_nats > min_gain_nats || r.homology_posterior >= 0.25 {
                        let f_min = pair_f_min(r, seq_lens[ga], seq_lens[gb]);
                        let t_eff = r.est_t_primary.max(0.0) - 0.75 * f_min.ln();
                        let n_eff = (f_min * (seq_lens[ga].min(seq_lens[gb]) as f64)).max(1.0);
                        let v = (t_eff + 0.05)
                            / ((n_eff + 5.0) * r.homology_posterior.clamp(0.10, 1.0));
                        sp_d[a * m + b] = t_eff;
                        sp_d[b * m + a] = t_eff;
                        sp_v[a * m + b] = v;
                        sp_v[b * m + a] = v;
                    }
                }
            }
        }
        relax_shortest_paths_with_var(&mut sp_d, &mut sp_v, m);
        for a in 0..m {
            for b in (a + 1)..m {
                if !sp_d[a * m + b].is_finite() {
                    let ga = comp_indices[a];
                    let gb = comp_indices[b];
                    let (d, v) = if let Some(r) = pair_map.get(&pair_key(ga, gb)) {
                        (r.est_t_primary.max(0.0), 1.5)
                    } else {
                        (3.5, 2.0)
                    };
                    sp_d[a * m + b] = d;
                    sp_d[b * m + a] = d;
                    sp_v[a * m + b] = v;
                    sp_v[b * m + a] = v;
                } else {
                    let v = sp_v[a * m + b].max(1e-6);
                    sp_v[a * m + b] = v;
                    sp_v[b * m + a] = v;
                }
            }
        }
        return (sp_d, sp_v);
    }

    let mut g2l: HashMap<usize, usize> = HashMap::with_capacity(m);
    for (loc, &g) in comp_indices.iter().enumerate() {
        g2l.insert(g, loc);
    }
    let mut adj: Vec<Vec<(usize, f64, f64)>> = vec![Vec::new(); m];
    for (&key, r) in pair_map {
        if !(r.mdl_gain_nats > min_gain_nats || r.homology_posterior >= 0.25) {
            continue;
        }
        let ga = (key >> 32) as usize;
        let gb = (key & 0xFFFF_FFFF) as usize;
        if let (Some(&a), Some(&b)) = (g2l.get(&ga), g2l.get(&gb)) {
            let f_min = pair_f_min(r, seq_lens[ga], seq_lens[gb]);
            let t_eff = r.est_t_primary.max(0.0) - 0.75 * f_min.ln();
            let n_eff = (f_min * (seq_lens[ga].min(seq_lens[gb]) as f64)).max(1.0);
            let v = (t_eff + 0.05)
                / ((n_eff + 5.0) * r.homology_posterior.clamp(0.10, 1.0));
            adj[a].push((b, t_eff, v));
            adj[b].push((a, t_eff, v));
        }
    }
    for row in &mut adj {
        row.sort_unstable_by_key(|x| x.0);
    }

    let mut d_mat = vec![f64::INFINITY; m * m];
    let mut v_mat = vec![f64::INFINITY; m * m];
    d_mat
        .par_chunks_mut(m)
        .zip(v_mat.par_chunks_mut(m))
        .enumerate()
        .for_each(|(src, (dist, var))| {
            let mut heap = BinaryHeap::new();
            dist[src] = 0.0;
            var[src] = 0.0;
            heap.push(DjNode {
                dist: 0.0,
                node: src,
            });
            while let Some(DjNode { dist: du, node: u }) = heap.pop() {
                if du > dist[u] + 1e-12 {
                    continue;
                }
                let vu = var[u];
                for &(v, w, w_v) in &adj[u] {
                    let cand = du + w;
                    if cand < dist[v] - 1e-12 {
                        dist[v] = cand;
                        var[v] = vu + w_v;
                        heap.push(DjNode { dist: cand, node: v });
                    }
                }
            }
        });
    drop(adj);

    for (&key, r) in pair_map {
        let ga = (key >> 32) as usize;
        let gb = (key & 0xFFFF_FFFF) as usize;
        if let (Some(&a), Some(&b)) = (g2l.get(&ga), g2l.get(&gb)) {
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            if !(r.mdl_gain_nats > min_gain_nats || r.homology_posterior >= 0.25)
                && !d_mat[lo * m + hi].is_finite()
            {
                let d = r.est_t_primary.max(0.0);
                d_mat[lo * m + hi] = d;
                d_mat[hi * m + lo] = d;
                v_mat[lo * m + hi] = 1.5;
                v_mat[hi * m + lo] = 1.5;
            }
        }
    }

    for a in 0..m {
        d_mat[a * m + a] = 0.0;
        v_mat[a * m + a] = 0.0;
        for b in (a + 1)..m {
            let d_raw = d_mat[a * m + b];
            let (d, v) = if d_raw.is_finite() {
                (d_raw, v_mat[a * m + b].max(1e-6))
            } else {
                (3.5, 2.0)
            };
            d_mat[a * m + b] = d;
            d_mat[b * m + a] = d;
            v_mat[a * m + b] = v;
            v_mat[b * m + a] = v;
        }
    }
    (d_mat, v_mat)
}

fn collect_split_bridges(
    node: &PhyloNode,
    comp_indices: &[usize],
    d_mat: &[f64],
    m: usize,
    seq_lens: &[usize],
    max_closest: usize,
    max_longest_side: usize,
    out_pairs: &mut Vec<(usize, usize)>,
) -> Vec<usize> {
    match node {
        PhyloNode::Leaf(idx) => vec![*idx],
        PhyloNode::Internal { left, right, .. } => {
            let l_leaves = collect_split_bridges(
                left,
                comp_indices,
                d_mat,
                m,
                seq_lens,
                max_closest,
                max_longest_side,
                out_pairs,
            );
            let r_leaves = collect_split_bridges(
                right,
                comp_indices,
                d_mat,
                m,
                seq_lens,
                max_closest,
                max_longest_side,
                out_pairs,
            );

            let total_cross = l_leaves.len() * r_leaves.len();
            let eff_closest = if l_leaves.len().min(r_leaves.len()) >= 64 {
                max_closest.max(28)
            } else {
                max_closest
            };
            let take_k = total_cross.min(eff_closest);
            if take_k > 0 {
                let (l_sample, r_sample): (Vec<usize>, Vec<usize>) = if total_cross <= 65_536 {
                    (l_leaves.clone(), r_leaves.clone())
                } else {
                    let cap = 256usize;
                    let ls: Vec<usize> = if l_leaves.len() <= cap {
                        l_leaves.clone()
                    } else {
                        let step = l_leaves.len() as f64 / cap as f64;
                        (0..cap)
                            .map(|k| l_leaves[((k as f64 * step) as usize).min(l_leaves.len() - 1)])
                            .collect()
                    };
                    let rs: Vec<usize> = if r_leaves.len() <= cap {
                        r_leaves.clone()
                    } else {
                        let step = r_leaves.len() as f64 / cap as f64;
                        (0..cap)
                            .map(|k| r_leaves[((k as f64 * step) as usize).min(r_leaves.len() - 1)])
                            .collect()
                    };
                    (ls, rs)
                };
                let mut cross: Vec<(f64, usize, usize, usize, usize)> =
                    Vec::with_capacity(l_sample.len() * r_sample.len());
                for &ta in &l_sample {
                    for &tb in &r_sample {
                        let ga = comp_indices[ta];
                        let gb = comp_indices[tb];
                        cross.push((d_mat[ta * m + tb], ga.min(gb), ga.max(gb), ta, tb));
                    }
                }
                if l_leaves.len() >= 6 && r_leaves.len() >= 6 && cross.len() > take_k {
                    let pool = (take_k * 8).min(cross.len());
                    if cross.len() > pool {
                        cross.select_nth_unstable_by(pool, |x, y| {
                            x.0.total_cmp(&y.0)
                                .then_with(|| x.1.cmp(&y.1))
                                .then_with(|| x.2.cmp(&y.2))
                        });
                    }
                    cross[..pool].sort_unstable_by(|x, y| {
                        x.0.total_cmp(&y.0)
                            .then_with(|| x.1.cmp(&y.1))
                            .then_with(|| x.2.cmp(&y.2))
                    });
                    let mut deg_l: HashMap<usize, u8> = HashMap::with_capacity(pool);
                    let mut deg_r: HashMap<usize, u8> = HashMap::with_capacity(pool);
                    let mut picked = vec![false; pool];
                    let mut picked_cnt = 0usize;
                    for idx in 0..pool {
                        let (_, lo, hi, ta, tb) = cross[idx];
                        let dl = deg_l.get(&ta).copied().unwrap_or(0);
                        let dr = deg_r.get(&tb).copied().unwrap_or(0);
                        if dl < 2 && dr < 2 {
                            deg_l.insert(ta, dl + 1);
                            deg_r.insert(tb, dr + 1);
                            out_pairs.push((lo, hi));
                            picked[idx] = true;
                            picked_cnt += 1;
                            if picked_cnt == take_k {
                                break;
                            }
                        }
                    }
                    if picked_cnt < take_k {
                        for idx in 0..pool {
                            if !picked[idx] {
                                out_pairs.push((cross[idx].1, cross[idx].2));
                                picked_cnt += 1;
                                if picked_cnt == take_k {
                                    break;
                                }
                            }
                        }
                    }
                } else {
                    if cross.len() > take_k {
                        cross.select_nth_unstable_by(take_k, |x, y| {
                            x.0.total_cmp(&y.0)
                                .then_with(|| x.1.cmp(&y.1))
                                .then_with(|| x.2.cmp(&y.2))
                        });
                    }
                    for &(_, lo, hi, _, _) in cross.iter().take(take_k) {
                        out_pairs.push((lo, hi));
                    }
                }
            }

            if max_longest_side > 0 {
                let mut l_long = l_leaves.clone();
                l_long.sort_unstable_by(|&a, &b| {
                    seq_lens[comp_indices[b]]
                        .cmp(&seq_lens[comp_indices[a]])
                        .then_with(|| a.cmp(&b))
                });
                l_long.truncate(max_longest_side);

                let mut r_long = r_leaves.clone();
                r_long.sort_unstable_by(|&a, &b| {
                    seq_lens[comp_indices[b]]
                        .cmp(&seq_lens[comp_indices[a]])
                        .then_with(|| a.cmp(&b))
                });
                r_long.truncate(max_longest_side);

                for &ta in &l_long {
                    for &tb in &r_long {
                        let ga = comp_indices[ta];
                        let gb = comp_indices[tb];
                        out_pairs.push((ga.min(gb), ga.max(gb)));
                    }
                }
            }

            let mut merged = l_leaves;
            merged.extend_from_slice(&r_leaves);
            merged
        }
    }
}

fn extract_nt9mers(seq: &EncodedSeq) -> Vec<u32> {
    let n = seq.nt_tokens.len();
    if n < 9 || seq.nt_masks.len() != n {
        return Vec::new();
    }
    let mut kmers = Vec::with_capacity(n - 8);
    let mut bad = 0usize;
    let mut hash = 0u32;
    const MASK: u32 = (1 << 18) - 1;
    for k in 0..n {
        if seq.nt_masks[k].count_ones() != 1 {
            bad += 1;
        }
        hash = ((hash << 2) | (seq.nt_tokens[k] as u32 & 3)) & MASK;
        if k >= 9 && seq.nt_masks[k - 9].count_ones() != 1 {
            bad -= 1;
        }
        if k >= 8 && bad == 0 {
            kmers.push(hash);
        }
    }
    kmers.sort_unstable();
    kmers.dedup();
    kmers
}

fn extract_aa3_codes(seq: &EncodedSeq) -> Vec<u16> {
    let sp = &seq.sym_proj;
    if sp.len() < 3 {
        return Vec::new();
    }
    let mut codes = Vec::with_capacity(sp.len() - 2);
    for k in 0..=(sp.len() - 3) {
        let a0 = sp[k] as usize;
        let a1 = sp[k + 1] as usize;
        let a2 = sp[k + 2] as usize;
        if a0 < 20 && a1 < 20 && a2 < 20 {
            codes.push(((a0 * 20 + a1) * 20 + a2) as u16);
        }
    }
    codes.sort_unstable();
    codes.dedup();
    codes
}

fn count_sorted_shared(a: &[u32], b: &[u32]) -> usize {
    let (mut i, mut j, mut cnt) = (0usize, 0usize, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                cnt += 1;
                i += 1;
                j += 1;
            }
        }
    }
    cnt
}

fn select_closest_clean_refs(
    target_idx: usize,
    clean_candidates: &[usize],
    nt9mers: &[Vec<u32>],
    seqs: &[EncodedSeq],
) -> Vec<usize> {
    let tar_na_len = seqs[target_idx].non_ambig_len;
    let mut scored: Vec<(bool, bool, usize, isize, usize)> = clean_candidates
        .iter()
        .copied()
        .filter(|&r| r != target_idx)
        .map(|r| {
            let shared = count_sorted_shared(&nt9mers[r], &nt9mers[target_idx]);
            let has_shared = shared > 0;
            let good_cov = shared >= 5 && seqs[r].non_ambig_len * 10 >= tar_na_len * 9;
            let len_diff = (seqs[r].len() as isize - seqs[target_idx].len() as isize).abs();
            (has_shared, good_cov, shared, len_diff, r)
        })
        .collect();
    scored.sort_unstable_by(|a, b| {
        b.0.cmp(&a.0)
            .then_with(|| b.1.cmp(&a.1))
            .then_with(|| b.2.cmp(&a.2))
            .then_with(|| a.3.cmp(&b.3))
            .then_with(|| a.4.cmp(&b.4))
    });
    let mut refs: Vec<usize> = scored.into_iter().take(4).map(|x| x.4).collect();
    if !clean_candidates.is_empty() {
        let mut by_len = clean_candidates.to_vec();
        by_len.sort_by_key(|&i| seqs[i].non_ambig_len);
        let med = by_len[by_len.len() / 2];
        if med != target_idx && !refs.contains(&med) {
            refs.push(med);
        }
    }
    refs
}

fn audit_cohort_frameshifts(
    seqs: &mut [EncodedSeq],
    gc: &GeneticCode,
) -> (Vec<FrameshiftEvent>, f64) {
    let n = seqs.len();
    if n < 2 || seqs[0].mode != SeqMode::Codon {
        return (Vec::new(), 0.0);
    }

    let clean_candidates: Vec<usize> = (0..n)
        .filter(|&i| seqs[i].nt_tokens.len() % 3 == 0 && !seqs[i].has_unambiguous_internal_stop)
        .collect();
    if clean_candidates.is_empty() {
        return (Vec::new(), 0.0);
    }

    let nt9mers: Vec<Vec<u32>> = seqs.par_iter().map(extract_nt9mers).collect();
    let mut cohort_frameshifts: Vec<FrameshiftEvent> = Vec::new();
    let mut cohort_fs_pen = 0.0f64;

    // Pass A: explicit length-mod-3 != 0 or internal-stop frameshift candidates
    let repairs_a: Vec<Option<(EncodedSeq, Vec<FrameshiftEvent>, f64)>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let needs =
                seqs[i].nt_tokens.len() % 3 != 0 || seqs[i].has_unambiguous_internal_stop;
            if !needs {
                return None;
            }
            let refs = select_closest_clean_refs(i, &clean_candidates, &nt9mers, seqs);
            let mut best: Option<(f64, EncodedSeq, Vec<FrameshiftEvent>, f64)> = None;
            for r in refs {
                if let Some((rep, evs, pen)) =
                    attempt_frameshift_repair_with_margin(&seqs[r], &seqs[i], gc, 2.0)
                {
                    let gain_sum: f64 = evs.iter().map(|e| e.mdl_gain_nats).sum();
                    let replace = match &best {
                        None => true,
                        Some((bg, _, _, bpen)) => {
                            gain_sum > *bg
                                || ((gain_sum - *bg).abs() <= 1e-9 && pen < *bpen)
                        }
                    };
                    if replace {
                        best = Some((gain_sum, rep, evs, pen));
                    }
                }
            }
            best.map(|(_, rep, evs, pen)| (rep, evs, pen))
        })
        .collect();

    let mut pass_a_repaired = vec![false; n];
    for (i, res_opt) in repairs_a.into_iter().enumerate() {
        if let Some((repaired, evs, pen)) = res_opt {
            seqs[i] = repaired;
            cohort_frameshifts.extend(evs);
            cohort_fs_pen += pen;
            pass_a_repaired[i] = true;
        }
    }

    // Pass B: O(N * L) cohort AA 3-mer stealth audit for compensated / +-1 nt excursions at any N >= 3
    if n >= 3 {
        let clean_unrepaired: Vec<usize> = (0..n)
            .filter(|&i| {
                !pass_a_repaired[i]
                    && seqs[i].nt_tokens.len() % 3 == 0
                    && !seqs[i].has_unambiguous_internal_stop
            })
            .collect();
        if clean_unrepaired.len() >= 2 {
            let min_sup_df: u32 = ((n / 100) as u32).clamp(2, 5);
            let aa3_codes: Vec<Vec<u16>> = seqs.par_iter().map(extract_aa3_codes).collect();
            let mut df_aa3 = vec![0u32; 8000];
            for codes in &aa3_codes {
                for &c in codes {
                    df_aa3[c as usize] += 1;
                }
            }

            let mut aa3_shared_frac = vec![0.0f64; n];
            let mut max_unsup_w = vec![0usize; n];
            let mut term_unsup2 = vec![false; n];
            for s in 0..n {
                let sp = &seqs[s].sym_proj;
                let l_s = sp.len();
                if l_s < 3 {
                    continue;
                }
                let mut w_tot = 0usize;
                let mut w_shared = 0usize;
                let mut cur_run = 0usize;
                let mut max_run = 0usize;
                let mut first0 = true;
                let mut first1 = true;
                let mut last0 = true;
                let mut last1 = true;
                for k in 0..=(l_s - 3) {
                    let a0 = sp[k] as usize;
                    let a1 = sp[k + 1] as usize;
                    let a2 = sp[k + 2] as usize;
                    if a0 < 20 && a1 < 20 && a2 < 20 {
                        let code = (a0 * 20 + a1) * 20 + a2;
                        let supported = df_aa3[code] >= min_sup_df;
                        if w_tot == 0 {
                            first0 = supported;
                        } else if w_tot == 1 {
                            first1 = supported;
                        }
                        last0 = last1;
                        last1 = supported;
                        w_tot += 1;
                        if supported {
                            w_shared += 1;
                            cur_run = 0;
                        } else {
                            cur_run += 1;
                            if cur_run > max_run {
                                max_run = cur_run;
                            }
                        }
                    } else {
                        cur_run = 0;
                    }
                }
                if w_tot > 0 {
                    aa3_shared_frac[s] = w_shared as f64 / w_tot as f64;
                }
                max_unsup_w[s] = max_run;
                if w_tot >= 6 {
                    term_unsup2[s] = (!first0 && !first1) || (!last0 && !last1);
                }
            }
            let is_flagged: Vec<bool> = (0..n)
                .map(|s| aa3_shared_frac[s] >= 0.35 && (max_unsup_w[s] >= 4 || term_unsup2[s]))
                .collect();

            let repairs_b: Vec<Option<(EncodedSeq, Vec<FrameshiftEvent>, f64)>> = (0..n)
                .into_par_iter()
                .map(|i| {
                    if pass_a_repaired[i]
                        || seqs[i].nt_tokens.len() % 3 != 0
                        || seqs[i].has_unambiguous_internal_stop
                        || seqs[i].nt_tokens.len() > 6000
                        || !is_flagged[i]
                    {
                        return None;
                    }
                    let sup_cands: Vec<usize> = clean_unrepaired
                        .iter()
                        .copied()
                        .filter(|&r| {
                            r != i
                                && (!is_flagged[r]
                                    || aa3_shared_frac[r] > aa3_shared_frac[i] + 1e-12
                                    || max_unsup_w[r] < max_unsup_w[i])
                        })
                        .collect();
                    if sup_cands.is_empty() {
                        return None;
                    }
                    let before_sup = aa3_codes[i]
                        .iter()
                        .filter(|&&c| df_aa3[c as usize] >= min_sup_df)
                        .count();
                    let after_req_df = min_sup_df.saturating_sub(1).max(1);
                    let refs = select_closest_clean_refs(i, &sup_cands, &nt9mers, seqs);
                    let mut cands: Vec<(f64, EncodedSeq, Vec<FrameshiftEvent>, f64)> = Vec::new();
                    for r in refs {
                        if let Some((rep, evs, pen)) =
                            attempt_frameshift_repair_with_margin(&seqs[r], &seqs[i], gc, 2.5)
                        {
                            if evs.is_empty() {
                                continue;
                            }
                            let rep_codes = extract_aa3_codes(&rep);
                            let after_sup = rep_codes
                                .iter()
                                .filter(|&&c| {
                                    let sub = if aa3_codes[i].binary_search(&c).is_ok() {
                                        1
                                    } else {
                                        0
                                    };
                                    df_aa3[c as usize].saturating_sub(sub) >= after_req_df
                                })
                                .count();
                            let req_delta = if sup_cands.len() >= 2 { 2 } else { 3 };
                            if after_sup < before_sup + req_delta {
                                continue;
                            }
                            let gain_sum: f64 = evs.iter().map(|e| e.mdl_gain_nats).sum();
                            cands.push((gain_sum, rep, evs, pen));
                        }
                    }
                    if cands.is_empty() {
                        return None;
                    }
                    cands.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
                    if sup_cands.len() >= 2 {
                        for idx_a in 0..cands.len() {
                            let p_a = cands[idx_a].2[0].nt_pos;
                            let corroborated = (0..cands.len()).any(|idx_b| {
                                idx_a != idx_b
                                    && cands[idx_b].2[0].nt_pos.abs_diff(p_a) <= 12
                            });
                            if corroborated {
                                let (_, rep, evs, pen) = cands.swap_remove(idx_a);
                                return Some((rep, evs, pen));
                            }
                        }
                        None
                    } else if n == 3 && sup_cands.len() == 1 && cands[0].0 >= 8.0 {
                        let (_, rep, evs, pen) = cands.swap_remove(0);
                        Some((rep, evs, pen))
                    } else {
                        None
                    }
                })
                .collect();

            for (i, res_opt) in repairs_b.into_iter().enumerate() {
                if let Some((repaired, evs, pen)) = res_opt {
                    seqs[i] = repaired;
                    cohort_frameshifts.extend(evs);
                    cohort_fs_pen += pen;
                }
            }
        }
    }

    (cohort_frameshifts, cohort_fs_pen)
}

struct QuickCompProfile {
    indices: Vec<usize>,
    pos_to_col: Vec<Vec<usize>>,
    col_counts: Vec<Vec<f64>>,
    col_ln_prob: Vec<f64>,
    col_bub: Vec<f64>,
    cols: Vec<Vec<(usize, usize)>>,
}

fn build_quick_component_profile(
    comp_indices: &[usize],
    _n_total: usize,
    prepared: &mut [PreparedSeq],
    gc: &GeneticCode,
    tables: &GlobalRegimeTables,
    seq_lens: &[usize],
    pair_map: &mut HashMap<u64, PairHomologyResult>,
    state_map: &mut HashMap<u64, PairForwardState>,
) -> Option<QuickCompProfile> {
    let m = comp_indices.len();
    if m == 0 || m > 40 {
        return None;
    }
    let mode = prepared.first().map(|p| p.seq.mode).unwrap_or(SeqMode::Protein);
    let k_dim = if mode == SeqMode::Dna { 4 } else { 20 };

    if m == 1 {
        let g0 = comp_indices[0];
        let l0 = seq_lens[g0];
        let comp_seqs = [prepared[g0].seq.clone()];
        let taxon_nu = [1.0f64];
        let cols: Vec<Vec<(usize, usize)>> = (0..l0).map(|pos| vec![(0, pos)]).collect();
        let mut col_counts = Vec::with_capacity(l0);
        let mut col_ln_prob = Vec::with_capacity(l0);
        for c in &cols {
            let (cnt, _, lp) = build_col_profile(c, mode, &comp_seqs, gc, &taxon_nu, k_dim);
            col_counts.push(cnt);
            col_ln_prob.push(lp);
        }
        let pos_to_col = vec![(0..l0).collect()];
        return Some(QuickCompProfile {
            indices: comp_indices.to_vec(),
            pos_to_col,
            col_counts,
            col_ln_prob,
            col_bub: vec![0.0; l0],
            cols,
        });
    }

    if _n_total > 30 {
        let mut internal_pairs = Vec::new();
        for a in 0..m {
            for b in (a + 1)..m {
                let ga = comp_indices[a];
                let gb = comp_indices[b];
                let (lo, hi) = (ga.min(gb), ga.max(gb));
                if m <= 24 || pair_map.get(&pair_key(lo, hi)).map_or(false, |r| r.mdl_gain_nats > -2.5) {
                    internal_pairs.push((lo, hi));
                }
            }
        }
        ensure_pairs_evaluated_with_posteriors(
            &internal_pairs,
            prepared,
            gc,
            tables,
            pair_map,
            state_map,
        );
    }

    let (d_mat, v_mat) = compute_evol_dist_and_var(comp_indices, pair_map, seq_lens, 0.0);
    let local_ids: Vec<usize> = (0..m).collect();
    let tree = build_bionj_tree(&local_ids, &d_mat, &v_mat, m);
    let comp_seqs: Vec<EncodedSeq> = comp_indices.iter().map(|&idx| prepared[idx].seq.clone()).collect();
    let comp_names: Vec<String> = comp_seqs.iter().map(|s| s.id.clone()).collect();
    let (tree_nodes, root_id, _) = tree.flatten_and_label(&comp_names);
    let lookup = build_consistent_lookup(comp_indices, pair_map, &d_mat, seq_lens);
    let taxon_nu = compute_taxon_nu(&d_mat, m);
    let prof = assemble_profile_on_tree(
        root_id,
        &tree_nodes,
        &lookup,
        &comp_seqs,
        gc,
        &d_mat,
        &taxon_nu,
        m,
    );

    let mut pos_to_col: Vec<Vec<usize>> = comp_indices
        .iter()
        .map(|&g| vec![usize::MAX; seq_lens[g]])
        .collect();
    for (col_idx, entries) in prof.cols.iter().enumerate() {
        for &(t_loc, pos) in entries {
            pos_to_col[t_loc][pos] = col_idx;
        }
    }

    let mut col_counts = Vec::with_capacity(prof.cols.len());
    let mut col_ln_prob = Vec::with_capacity(prof.cols.len());
    let mut col_bub = Vec::with_capacity(prof.cols.len());
    for col in &prof.cols {
        let (cnt, _, lp) = build_col_profile(col, mode, &comp_seqs, gc, &taxon_nu, k_dim);
        let bub = if col.is_empty() {
            0.0
        } else {
            col.iter()
                .map(|&(t_loc, pos)| lookup.bubble_x_by_seq[t_loc][pos])
                .sum::<f64>()
                / (col.len() as f64)
        };
        col_counts.push(cnt);
        col_ln_prob.push(lp);
        col_bub.push(bub);
    }

    Some(QuickCompProfile {
        indices: comp_indices.to_vec(),
        pos_to_col,
        col_counts,
        col_ln_prob,
        col_bub,
        cols: prof.cols,
    })
}

fn colinear_chain_niche_gain(
    pu: &QuickCompProfile,
    pv: &QuickCompProfile,
    pair_map: &HashMap<u64, PairHomologyResult>,
    mode: SeqMode,
) -> f64 {
    let lu = pu.indices.len();
    let lv = pv.indices.len();
    let mu = pu.cols.len();
    let mv = pv.cols.len();
    if mu < 14 || mv < 14 || lu + lv < 3 {
        return 0.0;
    }
    let k_dim = if mode == SeqMode::Dna { 4 } else { 20 };
    let min_occ_u = if lu >= 2 { 2 } else { 1 };
    let min_occ_v = if lv >= 2 { 2 } else { 1 };
    let min_denom = if lu >= 2 && lv >= 2 { 4 } else { 2 };
    let mut eval_uv = vec![false; lu * lv];
    let mut raw_hits: Vec<((usize, usize), f64)> = Vec::new();

    for (loc_u, &ga) in pu.indices.iter().enumerate() {
        for (loc_v, &gb) in pv.indices.iter().enumerate() {
            let flip = ga > gb;
            if let Some(res) = pair_map.get(&pair_key(ga, gb)) {
                if is_post_eval(res) || res.mdl_gain_nats <= -2.5 {
                    eval_uv[loc_u * lv + loc_v] = true;
                }
                if !is_post_eval(res) || res.mdl_gain_nats <= -2.5 {
                    continue;
                }
                let pi = mdl_gain_to_prob(res.mdl_gain_nats + 1.5).max(res.homology_posterior);
                for e in &res.fine_edges {
                    let p_edge = if e.cond_posterior > 0.0 {
                        (e.cond_posterior * pi).clamp(0.0, 1.0)
                    } else {
                        e.posterior
                    };
                    if p_edge < 0.02 {
                        continue;
                    }
                    let (ia, ib) = if !flip { (e.idx_x, e.idx_y) } else { (e.idx_y, e.idx_x) };
                    if ia < pu.pos_to_col[loc_u].len() && ib < pv.pos_to_col[loc_v].len() {
                        let p = pu.pos_to_col[loc_u][ia];
                        let q = pv.pos_to_col[loc_v][ib];
                        if p != usize::MAX
                            && q != usize::MAX
                            && pu.cols[p].len() >= min_occ_u
                            && pv.cols[q].len() >= min_occ_v
                        {
                            raw_hits.push(((p, q), p_edge));
                        }
                    }
                }
            }
        }
    }
    if raw_hits.is_empty() {
        return 0.0;
    }
    raw_hits.sort_unstable_by(|a, b| a.0.cmp(&b.0));

    let mut cands: Vec<(usize, usize, f64)> = Vec::new();
    let mut joint = vec![0.0f64; k_dim];
    let mut idx = 0usize;
    while idx < raw_hits.len() {
        let (p, q) = raw_hits[idx].0;
        let mut p_sum = 0.0f64;
        while idx < raw_hits.len() && raw_hits[idx].0 == (p, q) {
            p_sum += raw_hits[idx].1;
            idx += 1;
        }
        if pu.col_bub[p].max(pv.col_bub[q]) >= 0.35 {
            continue;
        }
        let mut denom_pq = 0usize;
        for &(tu, _) in &pu.cols[p] {
            for &(tv, _) in &pv.cols[q] {
                if eval_uv[tu * lv + tv] {
                    denom_pq += 1;
                }
            }
        }
        if denom_pq < min_denom {
            continue;
        }
        let s_fine = p_sum / (denom_pq as f64);
        if s_fine >= 0.30 {
            for k in 0..k_dim {
                joint[k] = pu.col_counts[p][k] + pv.col_counts[q][k];
            }
            let delta_col = (column_niche_mixture_ln_prob(mode, &joint)
                - pu.col_ln_prob[p]
                - pv.col_ln_prob[q])
                .clamp(0.0, 1.5);
            cands.push((p, q, delta_col));
        }
    }

    if cands.len() < 14 {
        return 0.0;
    }

    #[inline]
    fn better_chain(a: (usize, f64), b: (usize, f64)) -> (usize, f64) {
        if a.0 > b.0 || (a.0 == b.0 && a.1 > b.1) {
            a
        } else {
            b
        }
    }

    let mut bit = vec![(0usize, 0.0f64); mv + 1];
    let mut best_overall = (0usize, 0.0f64);
    let mut start = 0usize;
    while start < cands.len() {
        let cur_p = cands[start].0;
        let mut end = start;
        while end < cands.len() && cands[end].0 == cur_p {
            end += 1;
        }
        let mut updates: Vec<(usize, (usize, f64))> = Vec::with_capacity(end - start);
        for &(_, q, delta) in &cands[start..end] {
            let mut prev = (0usize, 0.0f64);
            let mut k = q;
            while k > 0 {
                prev = better_chain(prev, bit[k]);
                k -= k & k.wrapping_neg();
            }
            let val = (prev.0 + 1, prev.1 + delta);
            best_overall = better_chain(best_overall, val);
            updates.push((q + 1, val));
        }
        for (mut k, val) in updates {
            while k <= mv {
                bit[k] = better_chain(bit[k], val);
                k += k & k.wrapping_neg();
            }
        }
        start = end;
    }

    if best_overall.0 >= 14 {
        best_overall.1
    } else {
        0.0
    }
}

fn try_rescue_components(
    n_total: usize,
    prepared: &mut [PreparedSeq],
    gc: &GeneticCode,
    tables: &GlobalRegimeTables,
    seq_lens: &[usize],
    pair_map: &mut HashMap<u64, PairHomologyResult>,
    state_map: &mut HashMap<u64, PairForwardState>,
    parent: &mut [usize],
    comp_groups: &BTreeMap<usize, Vec<usize>>,
) -> bool {
    let init_comps: Vec<Vec<usize>> = comp_groups.values().cloned().collect();
    let k_c = init_comps.len();
    if k_c < 2 || n_total < 3 {
        return false;
    }

    let mut seq_to_comp = vec![0usize; n_total];
    for (ci, members) in init_comps.iter().enumerate() {
        for &g in members {
            seq_to_comp[g] = ci;
        }
    }

    // Group evaluated cross-component pairs by `(cu, cv)` with `cu < cv`
    let mut pair_lists: BTreeMap<(usize, usize), Vec<(f64, usize, usize)>> = BTreeMap::new();
    for (&k, res) in pair_map.iter() {
        let a = (k >> 32) as usize;
        let b = (k & 0xFFFF_FFFF) as usize;
        let ca = seq_to_comp[a];
        let cb = seq_to_comp[b];
        if ca == cb {
            continue;
        }
        let key = (ca.min(cb), ca.max(cb));
        pair_lists.entry(key).or_default().push((res.mdl_gain_nats, a, b));
    }

    let mut cand_pairs: Vec<(f64, usize, usize, Vec<(f64, usize, usize)>)> = Vec::new();
    for ((cu, cv), mut plist) in pair_lists {
        plist.sort_unstable_by(|x, y| {
            y.0.total_cmp(&x.0)
                .then_with(|| x.1.cmp(&y.1))
                .then_with(|| x.2.cmp(&y.2))
        });
        if plist[0].0 > -0.25 {
            cand_pairs.push((plist[0].0, cu, cv, plist));
        }
    }
    if cand_pairs.is_empty() {
        return false;
    }
    cand_pairs.sort_unstable_by(|x, y| {
        y.0.total_cmp(&x.0)
            .then_with(|| x.1.cmp(&y.1))
            .then_with(|| x.2.cmp(&y.2))
    });
    cand_pairs.truncate(64);

    let mut cross_to_eval: Vec<(usize, usize)> = Vec::new();
    let mut involved_comps: BTreeMap<usize, ()> = BTreeMap::new();
    for &(_, cu, cv, ref plist) in &cand_pairs {
        involved_comps.insert(cu, ());
        involved_comps.insert(cv, ());
        for &(g, a, b) in plist.iter().filter(|x| x.0 > -2.5).take(6) {
            let _ = g;
            cross_to_eval.push((a.min(b), a.max(b)));
        }
        if init_comps[cu].len() * init_comps[cv].len() <= 36 {
            for &ga in &init_comps[cu] {
                for &gb in &init_comps[cv] {
                    cross_to_eval.push((ga.min(gb), ga.max(gb)));
                }
            }
        }
    }

    ensure_pairs_evaluated_with_posteriors(
        &cross_to_eval,
        prepared,
        gc,
        tables,
        pair_map,
        state_map,
    );

    let mut profiles: BTreeMap<usize, Option<QuickCompProfile>> = BTreeMap::new();
    for &ci in involved_comps.keys() {
        let prof = build_quick_component_profile(
            &init_comps[ci],
            n_total,
            prepared,
            gc,
            tables,
            seq_lens,
            pair_map,
            state_map,
        );
        profiles.insert(ci, prof);
    }

    let mode = prepared.first().map(|p| p.seq.mode).unwrap_or(SeqMode::Protein);
    let rescue_thresh = if n_total > 2 {
        ((n_total - 1) as f64).ln()
    } else {
        0.0
    };
    let mut merged_any = false;
    let mut cur_members: Vec<Vec<usize>> = (0..n_total).map(|_| Vec::new()).collect();
    for g in 0..n_total {
        let r = uf_find(parent, g);
        cur_members[r].push(g);
    }

    for (_, cu, cv, _) in cand_pairs {
        let ra = uf_find(parent, init_comps[cu][0]);
        let rb = uf_find(parent, init_comps[cv][0]);
        if ra == rb {
            continue;
        }
        let mut cross_eval: Vec<(f64, usize, usize)> = Vec::new();
        for &ga in &init_comps[cu] {
            for &gb in &init_comps[cv] {
                let (lo, hi) = (ga.min(gb), ga.max(gb));
                if let Some(r) = pair_map.get(&pair_key(lo, hi)) {
                    cross_eval.push((r.mdl_gain_nats, lo, hi));
                }
            }
        }
        if cross_eval.is_empty() {
            continue;
        }
        cross_eval.sort_unstable_by(|x, y| {
            y.0.total_cmp(&x.0)
                .then_with(|| x.1.cmp(&y.1))
                .then_with(|| x.2.cmp(&y.2))
        });

        if cross_eval.len() >= 2 && cross_eval[1].0 <= -1.20 {
            continue;
        }

        let (best_gain, best_a, best_b) = cross_eval[0];
        let mut used = vec![false; n_total];
        used[best_a] = true;
        used[best_b] = true;
        let mut disjoint_pos_sum = 0.0f64;
        for &(g, a, b) in &cross_eval[1..] {
            if g <= 0.25 {
                break;
            }
            if !used[a] && !used[b] {
                used[a] = true;
                used[b] = true;
                disjoint_pos_sum += g - 0.25;
            }
        }

        let pos_col_niche_gain = match (
            profiles.get(&cu).and_then(|o| o.as_ref()),
            profiles.get(&cv).and_then(|o| o.as_ref()),
        ) {
            (Some(pu), Some(pv)) => colinear_chain_niche_gain(pu, pv, pair_map, mode),
            _ => 0.0,
        };

        let multi_clade = init_comps[cu].len() >= 2 && init_comps[cv].len() >= 2;
        if multi_clade && cross_eval.len() >= 9 {
            let neg_cnt = cross_eval.iter().filter(|x| x.0 <= -2.5).count();
            let med_gain = cross_eval[cross_eval.len() / 2].0;
            if neg_cnt * 5 >= cross_eval.len()
                && med_gain < rescue_thresh
                && best_gain < 8.0
                && pos_col_niche_gain < 10.0
            {
                continue;
            }
        }
        if multi_clade && disjoint_pos_sum <= 0.0 && pos_col_niche_gain < 4.0 {
            continue;
        }

        let clade_corrob = if init_comps[cu].len() == 1 || init_comps[cv].len() == 1 {
            cross_eval.get(1).map(|x| 0.40 * x.0.max(0.0)).unwrap_or(0.0)
        } else {
            0.60 * disjoint_pos_sum
        };
        let eff_thresh = if multi_clade {
            rescue_thresh + 0.5 * (init_comps[cu].len().min(init_comps[cv].len()) as f64).ln()
        } else {
            rescue_thresh
        };
        let rescue_score = best_gain + clade_corrob + 0.25 * pos_col_niche_gain;
        if rescue_score > eff_thresh {
            if cur_members[ra].len() >= 2
                && cur_members[rb].len() >= 2
                && (cur_members[ra].len() > init_comps[cu].len()
                    || cur_members[rb].len() > init_comps[cv].len())
            {
                let mut all_ab: Vec<f64> = Vec::new();
                for &ga in &cur_members[ra] {
                    for &gb in &cur_members[rb] {
                        let (lo, hi) = (ga.min(gb), ga.max(gb));
                        if let Some(r) = pair_map.get(&pair_key(lo, hi)) {
                            all_ab.push(r.mdl_gain_nats);
                        }
                    }
                }
                all_ab.sort_unstable_by(|x, y| y.total_cmp(x));
                let pos_cnt = all_ab.iter().filter(|&&g| g > 0.25).count();
                if (all_ab.len() >= 2 && all_ab[1] <= -1.20)
                    || (best_gain <= rescue_thresh && pos_cnt < 2 && pos_col_niche_gain < 4.0)
                {
                    continue;
                }
            }
            parent[rb] = ra;
            let moved = std::mem::take(&mut cur_members[rb]);
            cur_members[ra].extend(moved);
            merged_any = true;
        }
    }

    merged_any
}

/// Fixed, tree-independent cohort scoring context for telescoping MDL refinement J(A).
struct CohortObjectiveCtx {
    m: usize,
    mode: SeqMode,
    k_dim: usize,
    taxon_nu: Vec<f64>,
    omega_adj: Vec<Vec<(usize, f64)>>,
    active_pairs: Vec<(usize, usize, f64)>,
    single_dir_ln_p: Vec<f64>,
}

fn build_cohort_objective_ctx(
    m: usize,
    lookup: &ConsistentEdgeLookup,
    dist_mat: &[f64],
    taxon_nu: &[f64],
    comp_seqs: &[EncodedSeq],
    gc: &GeneticCode,
) -> CohortObjectiveCtx {
    let mode = comp_seqs.first().map(|s| s.mode).unwrap_or(SeqMode::Protein);
    let k_dim = if mode == SeqMode::Dna { 4 } else { 20 };

    let mut omega_adj = vec![Vec::new(); m];
    let mut active_pairs = Vec::new();
    if m <= 256 {
        for &(a, b) in &lookup.active_keys {
            let w = taxon_nu[a]
                * taxon_nu[b]
                * lookup.pair_rel(a, b)
                * (-0.35 * dist_mat[a * m + b].max(0.0)).exp();
            if w > 0.0 {
                omega_adj[a].push((b, w));
                omega_adj[b].push((a, w));
                active_pairs.push((a, b, w));
            }
        }
    } else {
        let mut raw_nbrs: Vec<Vec<(f64, usize)>> = vec![Vec::new(); m];
        for &(a, b) in &lookup.active_keys {
            let w = taxon_nu[a]
                * taxon_nu[b]
                * lookup.pair_rel(a, b)
                * (-0.35 * dist_mat[a * m + b].max(0.0)).exp();
            if w > 0.0 {
                raw_nbrs[a].push((w, b));
                raw_nbrs[b].push((w, a));
            }
        }
        let mut kept: HashMap<u64, f64> = HashMap::with_capacity(m * 16);
        for a in 0..m {
            let row = &mut raw_nbrs[a];
            row.sort_unstable_by(|x, y| y.0.total_cmp(&x.0).then_with(|| x.1.cmp(&y.1)));
            for &(w, b) in row.iter().take(12) {
                kept.insert(pair_key(a, b), w);
            }
        }
        let mut keys_sorted: Vec<(usize, usize, f64)> = kept
            .into_iter()
            .map(|(k, w)| ((k >> 32) as usize, (k & 0xFFFF_FFFF) as usize, w))
            .collect();
        keys_sorted.sort_unstable_by_key(|x| (x.0, x.1));
        for (a, b, w) in keys_sorted {
            omega_adj[a].push((b, w));
            omega_adj[b].push((a, w));
            active_pairs.push((a, b, w));
        }
    }

    let mut single_dir_ln_p = vec![0.0f64; m * k_dim];
    let mut tmp = vec![0.0f64; k_dim];
    for t in 0..m {
        let wt = taxon_nu[t];
        for sym in 0..k_dim {
            tmp.fill(0.0);
            tmp[sym] = wt;
            single_dir_ln_p[t * k_dim + sym] = column_niche_mixture_ln_prob(mode, &tmp);
        }
    }
    let _ = gc;

    CohortObjectiveCtx {
        m,
        mode,
        k_dim,
        taxon_nu: taxon_nu.to_vec(),
        omega_adj,
        active_pairs,
        single_dir_ln_p,
    }
}

/// Tree-free telescoping 5-state-consistent cohort alignment objective J(A):
///   \sum_{|C_c|>=2} w_col (p_eff(C_c) - FINE_MERGE_HORIZON)
///   - exact O(L_a + L_b) two-pointer 5-state continuity penalty across active pairs (a, b).
fn score_alignment_cohort_mdl(
    cols: &[Vec<(usize, usize)>],
    lookup: &ConsistentEdgeLookup,
    comp_seqs: &[EncodedSeq],
    _gc: &GeneticCode,
    ctx: &CohortObjectiveCtx,
) -> f64 {
    let m = ctx.m;
    let n_cols = cols.len();
    if n_cols == 0 || m < 2 {
        return 0.0;
    }

    let mut pos_to_col: Vec<Vec<usize>> = (0..m)
        .map(|t| vec![usize::MAX; comp_seqs[t].len()])
        .collect();
    for (c_idx, col) in cols.iter().enumerate() {
        for &(t, pos) in col {
            pos_to_col[t][pos] = c_idx;
        }
    }

    let mut col_pair_surplus = vec![0.0f64; n_cols];
    let mut col_sup_edges: Vec<Vec<(u16, u16)>> = vec![Vec::new(); n_cols];
    for &(a, b, w_ab) in &ctx.active_pairs {
        let pa = &pos_to_col[a];
        let pb = &pos_to_col[b];
        for &(ia, ib, prob) in lookup.pair_edges_lt(a, b) {
            let pf = prob as f64;
            let ca = pa[ia as usize];
            if ca != usize::MAX && ca == pb[ib as usize] {
                col_pair_surplus[ca] += w_ab * pf;
                if pf >= 0.10 {
                    col_sup_edges[ca].push((a as u16, b as u16));
                }
            }
        }
    }

    let mut score = 0.0f64;
    let mut tax_stamp = vec![0usize; m];
    let mut tax_slot = vec![0usize; m];
    let mut uf_p: Vec<usize> = Vec::with_capacity(m);
    let mut sub_counts = vec![0.0f64; ctx.k_dim];
    let mut grouped: Vec<(usize, usize)> = Vec::with_capacity(m);

    for (c_idx, col) in cols.iter().enumerate() {
        let occ = col.len();
        if occ < 2 {
            continue;
        }
        let stamp = c_idx + 1;
        for (slot_i, &(t, _)) in col.iter().enumerate() {
            tax_stamp[t] = stamp;
            tax_slot[t] = slot_i;
        }

        let mut w_col = 0.0f64;
        for &(a, _) in col {
            for &(b, w_ab) in &ctx.omega_adj[a] {
                if b > a && tax_stamp[b] == stamp {
                    w_col += w_ab;
                }
            }
        }
        if w_col <= 0.0 {
            continue;
        }
        let s_bar = (col_pair_surplus[c_idx] / w_col).clamp(0.0, 1.0);

        let mut delta_dir_conn = 0.0f64;
        if occ >= 3 && s_bar >= 0.15 && !col_sup_edges[c_idx].is_empty() {
            uf_p.clear();
            uf_p.extend(0..occ);
            for &(au, bu) in &col_sup_edges[c_idx] {
                let a = au as usize;
                let b = bu as usize;
                if tax_stamp[a] == stamp && tax_stamp[b] == stamp {
                    let ra = uf_find(&mut uf_p, tax_slot[a]);
                    let rb = uf_find(&mut uf_p, tax_slot[b]);
                    if ra != rb {
                        uf_p[rb] = ra;
                    }
                }
            }
            grouped.clear();
            for slot_i in 0..occ {
                grouped.push((uf_find(&mut uf_p, slot_i), slot_i));
            }
            grouped.sort_unstable_by_key(|x| x.0);

            let mut g_start = 0usize;
            while g_start < grouped.len() {
                let root = grouped[g_start].0;
                let mut g_end = g_start + 1;
                while g_end < grouped.len() && grouped[g_end].0 == root {
                    g_end += 1;
                }
                let g_sz = g_end - g_start;
                if g_sz >= 2 {
                    let mut single_lp_acc = 0.0f64;
                    sub_counts.fill(0.0);
                    for &(_, slot_i) in &grouped[g_start..g_end] {
                        let (t, pos) = col[slot_i];
                        let sym = comp_seqs[t].sym_proj[pos] as usize;
                        if sym < ctx.k_dim {
                            sub_counts[sym] += ctx.taxon_nu[t];
                            single_lp_acc += ctx.single_dir_ln_p[t * ctx.k_dim + sym];
                        }
                    }
                    let joint_lp = column_niche_mixture_ln_prob(ctx.mode, &sub_counts);
                    delta_dir_conn += (joint_lp - single_lp_acc).max(0.0);
                }
                g_start = g_end;
            }
        }

        let mean_bub: f64 = col
            .iter()
            .map(|&(t, pos)| lookup.bubble_x_by_seq[t][pos])
            .sum::<f64>()
            / (occ as f64);

        let p_eff = if occ >= 3 && s_bar >= 0.15 && mean_bub < 0.40 && delta_dir_conn > 0.0 {
            let n_tot: f64 = col.iter().map(|&(t, _)| ctx.taxon_nu[t]).sum();
            let excess_n = (n_tot - 2.0).max(0.0);
            let eta = excess_n / (excess_n + 2.0);
            let bub_gate = ((0.40 - mean_bub) / 0.40).clamp(0.0, 1.0);
            let delta_step = delta_dir_conn / ((occ - 1) as f64);
            let s_cl = s_bar.clamp(1e-4, 1.0 - 1e-4);
            let logit = (s_cl / (1.0 - s_cl)).ln() + 0.65 * eta * bub_gate * delta_step;
            mdl_gain_to_prob(logit)
        } else {
            s_bar
        };

        score += w_col * (p_eff - FINE_MERGE_HORIZON);
    }

    // Exact O(L_a + L_b) two-pointer 5-state continuity penalty across active pairs
    for &(a, b, w_ab) in &ctx.active_pairs {
        let pa = &pos_to_col[a];
        let pb = &pos_to_col[b];
        let (mut ia, mut ib) = (0usize, 0usize);
        let mut prev_ia = usize::MAX;
        let mut prev_ib = usize::MAX;
        while ia < pa.len() && ib < pb.len() {
            match pa[ia].cmp(&pb[ib]) {
                std::cmp::Ordering::Equal => {
                    if prev_ia != usize::MAX {
                        let gap_a = ia > prev_ia + 1;
                        let gap_b = ib > prev_ib + 1;
                        if gap_a && gap_b {
                            score -= 0.90 * w_ab;
                        } else if gap_a || gap_b {
                            score -= 0.10 * w_ab;
                        }
                    }
                    prev_ia = ia;
                    prev_ib = ib;
                    ia += 1;
                    ib += 1;
                }
                std::cmp::Ordering::Less => ia += 1,
                std::cmp::Ordering::Greater => ib += 1,
            }
        }
    }

    score
}

/// Partition an existing MSA column list cleanly into two disjoint `SubtreeProfile`s,
/// preserving exact residue order inside each subset.
fn split_cols_into_subprofiles(
    cols: &[Vec<(usize, usize)>],
    in_left: &[bool],
) -> (SubtreeProfile, SubtreeProfile) {
    let m = in_left.len();
    let mut l_taxa = Vec::with_capacity(m);
    let mut r_taxa = Vec::with_capacity(m);
    for (t, &isl) in in_left.iter().enumerate() {
        if isl {
            l_taxa.push(t);
        } else {
            r_taxa.push(t);
        }
    }
    let mut l_cols = Vec::with_capacity(cols.len());
    let mut r_cols = Vec::with_capacity(cols.len());
    for col in cols {
        let mut lc = Vec::new();
        let mut rc = Vec::new();
        for &(t, pos) in col {
            if in_left[t] {
                lc.push((t, pos));
            } else {
                rc.push((t, pos));
            }
        }
        if !lc.is_empty() {
            l_cols.push(lc);
        }
        if !rc.is_empty() {
            r_cols.push(rc);
        }
    }
    (
        SubtreeProfile {
            taxa: l_taxa,
            cols: l_cols,
        },
        SubtreeProfile {
            taxa: r_taxa,
            cols: r_cols,
        },
    )
}

/// Tree-free Maximum-Posterior Overlap (MPO) greedy agglomerative profile assembly.
fn assemble_profile_mpo(
    m_comp: usize,
    lookup: &ConsistentEdgeLookup,
    comp_seqs: &[EncodedSeq],
    gc: &GeneticCode,
    dist_mat: &[f64],
    taxon_nu: &[f64],
) -> Vec<Vec<(usize, usize)>> {
    let mut clusters: Vec<Option<SubtreeProfile>> = (0..m_comp)
        .map(|t| {
            Some(SubtreeProfile {
                taxa: vec![t],
                cols: (0..comp_seqs[t].len()).map(|p| vec![(t, p)]).collect(),
            })
        })
        .collect();
    let mut cl_mass = vec![0.0f64; m_comp * m_comp];
    for &(a, b) in &lookup.active_keys {
        let pm = lookup.pair_mass(a, b);
        cl_mass[a * m_comp + b] = pm;
        cl_mass[b * m_comp + a] = pm;
    }
    let mut cl_eff_len: Vec<f64> = (0..m_comp)
        .map(|t| (comp_seqs[t].len().max(1) as f64) * taxon_nu[t])
        .collect();
    let mut alive: Vec<usize> = (0..m_comp).collect();

    while alive.len() > 1 {
        let k = alive.len();
        let mut best_sc = -1.0f64;
        let mut best_pair = (0usize, 1usize);
        for a_idx in 0..k {
            let u = alive[a_idx];
            for b_idx in (a_idx + 1)..k {
                let v = alive[b_idx];
                let denom = (cl_eff_len[u] * cl_eff_len[v]).sqrt().max(1.0);
                let sc = cl_mass[u * m_comp + v] / denom;
                if sc > best_sc + 1e-12 {
                    best_sc = sc;
                    best_pair = (a_idx, b_idx);
                }
            }
        }
        let (pa, pb) = best_pair;
        let u = alive[pa];
        let v = alive[pb];
        let pl = clusters[u].take().unwrap();
        let pr = clusters[v].take().unwrap();
        clusters[u] = Some(align_subprofiles_5state(
            pl,
            pr,
            lookup,
            comp_seqs,
            gc,
            dist_mat,
            taxon_nu,
            m_comp,
        ));
        cl_eff_len[u] += cl_eff_len[v];
        for &w in &alive {
            if w != u && w != v {
                let m_uw = cl_mass[u * m_comp + w] + cl_mass[v * m_comp + w];
                cl_mass[u * m_comp + w] = m_uw;
                cl_mass[w * m_comp + u] = m_uw;
            }
        }
        alive.remove(pb);
    }

    clusters[alive[0]].take().unwrap().cols
}

fn extract_tree_bipartitions(node: &PhyloNode, m: usize) -> (Vec<usize>, Vec<(f64, Vec<usize>)>) {
    match node {
        PhyloNode::Leaf(idx) => (vec![*idx], Vec::new()),
        PhyloNode::Internal {
            left,
            right,
            len_left,
            len_right,
        } => {
            let (mut l_leaves, mut l_splits) = extract_tree_bipartitions(left, m);
            let (mut r_leaves, r_splits) = extract_tree_bipartitions(right, m);
            l_splits.extend(r_splits);
            if l_leaves.len() >= 2 && m - l_leaves.len() >= 2 {
                let weight = (len_left + len_right)
                    * (l_leaves.len().min(m - l_leaves.len()) as f64);
                l_splits.push((weight, l_leaves.clone()));
            }
            if r_leaves.len() >= 2 && m - r_leaves.len() >= 2 && l_leaves.len() != r_leaves.len() {
                let weight = (len_left + len_right)
                    * (r_leaves.len().min(m - r_leaves.len()) as f64);
                l_splits.push((weight, r_leaves.clone()));
            }
            l_leaves.append(&mut r_leaves);
            (l_leaves, l_splits)
        }
    }
}

/// Deterministic Fiedler (spectral conductance) bipartitions of the posterior similarity graph.
fn compute_fiedler_bipartitions(
    m: usize,
    lookup: &ConsistentEdgeLookup,
    comp_seqs: &[EncodedSeq],
) -> Vec<Vec<bool>> {
    if m < 4 {
        return Vec::new();
    }
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); m];
    let mut deg = vec![0.0f64; m];
    for &(a, b) in &lookup.active_keys {
        let min_l = comp_seqs[a].len().min(comp_seqs[b].len()).max(1) as f64;
        let w = (lookup.pair_mass(a, b) / min_l).clamp(0.0, 2.0) * lookup.pair_rel(a, b);
        if w > 0.0 {
            adj[a].push((b, w));
            adj[b].push((a, w));
            deg[a] += w;
            deg[b] += w;
        }
    }
    let total_deg: f64 = deg.iter().sum();
    if total_deg <= 1e-9 {
        return Vec::new();
    }

    let mut inv_sqrt_d = vec![0.0f64; m];
    let mut v0 = vec![0.0f64; m];
    let mut v0_norm_sq = 0.0f64;
    for a in 0..m {
        if deg[a] > 1e-12 {
            inv_sqrt_d[a] = deg[a].sqrt().recip();
            v0[a] = deg[a].sqrt();
            v0_norm_sq += deg[a];
        }
    }
    let v0_inv = v0_norm_sq.sqrt().recip();
    for x in &mut v0 {
        *x *= v0_inv;
    }

    let mut x: Vec<f64> = (0..m).map(|a| ((a as f64) * 1.17 + 0.31).cos()).collect();
    let project_ortho = |vec: &mut [f64]| {
        let dot: f64 = vec.iter().zip(v0.iter()).map(|(a, b)| a * b).sum();
        for (a, &b) in vec.iter_mut().zip(v0.iter()) {
            *a -= dot * b;
        }
    };
    project_ortho(&mut x);
    let mut norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm <= 1e-12 {
        return Vec::new();
    }
    for v in &mut x {
        *v /= norm;
    }

    let mut next_x = vec![0.0f64; m];
    for _ in 0..28 {
        for a in 0..m {
            let mut s = 0.0f64;
            if inv_sqrt_d[a] > 0.0 {
                let ia = inv_sqrt_d[a];
                for &(b, w) in &adj[a] {
                    s += w * ia * inv_sqrt_d[b] * x[b];
                }
            }
            next_x[a] = 0.5 * (x[a] + s);
        }
        project_ortho(&mut next_x);
        norm = next_x.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm <= 1e-12 {
            break;
        }
        for (dst, &val) in x.iter_mut().zip(next_x.iter()) {
            *dst = val / norm;
        }
    }

    let mut order: Vec<(f64, usize)> = (0..m)
        .map(|a| (x[a] * inv_sqrt_d[a].max(1e-6), a))
        .collect();
    order.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    let mut masks = Vec::new();
    // 1. Sign cut
    let mut sign_mask = vec![false; m];
    let mut cnt_neg = 0usize;
    for &(coord, a) in &order {
        if coord < 0.0 {
            sign_mask[a] = true;
            cnt_neg += 1;
        }
    }
    if (2..=(m - 2)).contains(&cnt_neg) {
        masks.push(sign_mask);
    }

    // 2. Min-conductance sweep cut along Fiedler ordering
    let mut in_s = vec![false; m];
    let mut vol_s = 0.0f64;
    let mut cut_w = 0.0f64;
    let mut best_cond = f64::INFINITY;
    let mut best_k = None::<usize>;
    for step in 0..(m - 1) {
        let u = order[step].1;
        for &(v, w) in &adj[u] {
            if in_s[v] {
                cut_w -= w;
            } else {
                cut_w += w;
            }
        }
        in_s[u] = true;
        vol_s += deg[u];
        let k_sz = step + 1;
        if (2..=(m - 2)).contains(&k_sz) {
            let vol_min = vol_s.min(total_deg - vol_s).max(1e-9);
            let cond = cut_w / vol_min;
            if cond < best_cond - 1e-12 {
                best_cond = cond;
                best_k = Some(k_sz);
            }
        }
    }
    if let Some(bk) = best_k {
        let mut sweep_mask = vec![false; m];
        for &(_, u) in order.iter().take(bk) {
            sweep_mask[u] = true;
        }
        masks.push(sweep_mask);
    }
    masks
}

fn compute_affinity_bipartitions(m: usize, dist_mat: &[f64]) -> Vec<Vec<bool>> {
    if m < 4 {
        return Vec::new();
    }
    let mut masks = Vec::new();
    let mut best_d = -1.0f64;
    let mut p1 = 0usize;
    let mut p2 = 1usize;
    for a in 0..m {
        for b in (a + 1)..m {
            let d = dist_mat[a * m + b];
            if d > best_d {
                best_d = d;
                p1 = a;
                p2 = b;
            }
        }
    }
    let mut m1 = vec![false; m];
    let mut cnt1 = 0usize;
    for t in 0..m {
        if dist_mat[t * m + p1] <= dist_mat[t * m + p2] {
            m1[t] = true;
            cnt1 += 1;
        }
    }
    if (2..=(m - 2)).contains(&cnt1) {
        masks.push(m1);
    }

    // Second orthogonal extreme pair: farthest from {p1, p2}, then farthest from q1
    let mut q1 = 0usize;
    let mut max_min_d = -1.0f64;
    for t in 0..m {
        if t != p1 && t != p2 {
            let md = dist_mat[t * m + p1].min(dist_mat[t * m + p2]);
            if md > max_min_d {
                max_min_d = md;
                q1 = t;
            }
        }
    }
    let mut q2 = if q1 == p1 { p2 } else { p1 };
    let mut max_q = -1.0f64;
    for t in 0..m {
        if t != q1 && dist_mat[q1 * m + t] > max_q {
            max_q = dist_mat[q1 * m + t];
            q2 = t;
        }
    }
    let mut m2 = vec![false; m];
    let mut cnt2 = 0usize;
    for t in 0..m {
        if dist_mat[t * m + q1] <= dist_mat[t * m + q2] {
            m2[t] = true;
            cnt2 += 1;
        }
    }
    if (2..=(m - 2)).contains(&cnt2) {
        masks.push(m2);
    }
    masks
}

/// Topology-free iterative refinement maximizing telescoping cohort objective J(A) via
/// (1) 1-vs-(M-1) Leave-One-Out realignment and (2) Fiedler/Affinity/Tree-Split bipartition cuts.
fn refine_alignment_topology_free(
    cur_cols: Vec<Vec<(usize, usize)>>,
    tree: &PhyloNode,
    lookup: &ConsistentEdgeLookup,
    comp_seqs: &[EncodedSeq],
    gc: &GeneticCode,
    dist_mat: &[f64],
    ctx: &CohortObjectiveCtx,
) -> Vec<Vec<(usize, usize)>> {
    let default_passes = if ctx.m <= 40 { 2 } else { 1 };
    refine_alignment_topology_free_passes(
        cur_cols,
        tree,
        lookup,
        comp_seqs,
        gc,
        dist_mat,
        ctx,
        default_passes,
    )
}

fn refine_alignment_topology_free_passes(
    mut cur_cols: Vec<Vec<(usize, usize)>>,
    tree: &PhyloNode,
    lookup: &ConsistentEdgeLookup,
    comp_seqs: &[EncodedSeq],
    gc: &GeneticCode,
    dist_mat: &[f64],
    ctx: &CohortObjectiveCtx,
    max_passes: usize,
) -> Vec<Vec<(usize, usize)>> {
    let m = ctx.m;
    if m < 3 {
        return cur_cols;
    }

    let mut cur_score = score_alignment_cohort_mdl(&cur_cols, lookup, comp_seqs, gc, ctx);

    // Precompute canonicalized, deduplicated Phase-B bipartition masks (`0` always in `mask`)
    let mut raw_masks: Vec<Vec<bool>> = Vec::new();
    raw_masks.extend(compute_fiedler_bipartitions(m, lookup, comp_seqs));
    raw_masks.extend(compute_affinity_bipartitions(m, dist_mat));
    let (_, mut tree_splits) = extract_tree_bipartitions(tree, m);
    tree_splits.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    let max_tree_splits = if m <= 30 {
        usize::MAX
    } else if m <= 512 {
        10
    } else {
        4
    };
    for (_, side) in tree_splits.into_iter().take(max_tree_splits) {
        let mut mask = vec![false; m];
        for t in side {
            mask[t] = true;
        }
        raw_masks.push(mask);
    }

    let mut cut_masks: Vec<Vec<bool>> = Vec::new();
    for mut mask in raw_masks {
        let cnt_true = mask.iter().filter(|&&b| b).count();
        if !(2..=(m - 2)).contains(&cnt_true) {
            continue;
        }
        if !mask[0] {
            for b in &mut mask {
                *b = !*b;
            }
        }
        if !cut_masks.contains(&mask) {
            cut_masks.push(mask);
        }
    }

    let mut k_loo = if m <= 24 {
        m
    } else if m <= 60 {
        14
    } else if m <= 512 {
        8
    } else {
        3
    };
    if cur_cols.len() > 4_000 {
        k_loo = k_loo.min(6);
        cut_masks.truncate(6);
    }

    for _pass in 0..max_passes {
        let mut improved = false;

        // Phase A: 1 vs (M - 1) Leave-One-Out realignment (orphans & poorly integrated taxa first)
        let mut multi_cnt = vec![0usize; m];
        for col in &cur_cols {
            if col.len() >= 2 {
                for &(t, _) in col {
                    multi_cnt[t] += 1;
                }
            }
        }
        let mut loo_order: Vec<(f64, usize)> = (0..m)
            .map(|t| {
                let len = comp_seqs[t].len().max(1) as f64;
                ((multi_cnt[t] as f64) / len, t)
            })
            .collect();
        loo_order.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        let mut single_mask = vec![false; m];
        for &(_, u) in loo_order.iter().take(k_loo) {
            single_mask.fill(false);
            single_mask[u] = true;
            let (sub_u, sub_rest) = split_cols_into_subprofiles(&cur_cols, &single_mask);
            if sub_u.cols.len().max(sub_rest.cols.len()) > 60_000 {
                continue;
            }
            let aligned = align_subprofiles_5state(
                sub_u,
                sub_rest,
                lookup,
                comp_seqs,
                gc,
                dist_mat,
                &ctx.taxon_nu,
                m,
            );
            let cand_score =
                score_alignment_cohort_mdl(&aligned.cols, lookup, comp_seqs, gc, ctx);
            if cand_score > cur_score + 1e-6 {
                cur_cols = aligned.cols;
                cur_score = cand_score;
                improved = true;
            }
        }

        // Phase B: Non-tree Fiedler/Affinity + balanced internal tree split cuts
        for mask in &cut_masks {
            let (sub_a, sub_b) = split_cols_into_subprofiles(&cur_cols, mask);
            if sub_a.cols.len().max(sub_b.cols.len()) > 60_000 {
                continue;
            }
            let aligned = align_subprofiles_5state(
                sub_a,
                sub_b,
                lookup,
                comp_seqs,
                gc,
                dist_mat,
                &ctx.taxon_nu,
                m,
            );
            let cand_score =
                score_alignment_cohort_mdl(&aligned.cols, lookup, comp_seqs, gc, ctx);
            if cand_score > cur_score + 1e-6 {
                cur_cols = aligned.cols;
                cur_score = cand_score;
                improved = true;
            }
        }

        if !improved {
            break;
        }
    }

    cur_cols
}

/// Re-estimate the component BioNJ tree jointly from verified co-aligned core columns
/// (using purely empirical cohort state frequencies, zero external matrices) and transducer (d_evol, v_evol).
fn reestimate_core_bionj_tree(
    cols: &[Vec<(usize, usize)>],
    lookup: &ConsistentEdgeLookup,
    comp_seqs: &[EncodedSeq],
    _gc: &GeneticCode,
    init_dist: &[f64],
    init_var: &[f64],
    taxon_nu: &[f64],
    m: usize,
) -> (PhyloNode, Vec<f64>) {
    let mode = comp_seqs.first().map(|s| s.mode).unwrap_or(SeqMode::Protein);
    let k_dim = if mode == SeqMode::Dna { 4 } else { 20 };

    let mut pi_counts = vec![0.5f64; k_dim];
    let mut tot_w = 0.5 * (k_dim as f64);
    let mut d_comb = init_dist.to_vec();
    let mut v_comb = init_var.to_vec();

    if m <= 128 {
        let mut n_co = vec![0usize; m * m];
        let mut m_co = vec![0usize; m * m];

        for col in cols {
            if col.len() < 2 {
                continue;
            }
            let bub_sum: f64 = col
                .iter()
                .map(|&(t, pos)| lookup.bubble_x_by_seq[t][pos])
                .sum();
            if bub_sum / (col.len() as f64) >= 0.35 {
                continue;
            }
            let syms: Vec<(usize, usize)> = col
                .iter()
                .filter_map(|&(t, pos)| {
                    let sym = comp_seqs[t].sym_proj[pos] as usize;
                    if sym < k_dim {
                        pi_counts[sym] += taxon_nu[t];
                        tot_w += taxon_nu[t];
                        Some((t, sym))
                    } else {
                        None
                    }
                })
                .collect();
            for i in 0..syms.len() {
                let (ta, sa) = syms[i];
                for j in (i + 1)..syms.len() {
                    let (tb, sb) = syms[j];
                    n_co[ta * m + tb] += 1;
                    if sa == sb {
                        m_co[ta * m + tb] += 1;
                    }
                }
            }
        }

        let b0: f64 = pi_counts
            .iter()
            .map(|&c| {
                let p = c / tot_w;
                p * p
            })
            .sum::<f64>()
            .clamp(1.0 / (k_dim as f64), 0.85);
        let one_minus_b0 = (1.0 - b0).max(0.15);

        for a in 0..m {
            for b in (a + 1)..m {
                let n_ab = n_co[a * m + b] + n_co[b * m + a];
                if n_ab >= 12 {
                    let matches = m_co[a * m + b] + m_co[b * m + a];
                    let f_id = (matches as f64) / (n_ab as f64);
                    let norm_s = ((f_id - b0) / one_minus_b0).clamp(0.03, 1.0);
                    let t_core = -one_minus_b0 * norm_s.ln();
                    let v_core = (t_core + 0.04) / ((n_ab as f64) + 5.0);
                    let v_init = init_var[a * m + b].max(1e-6);
                    let w_core = (v_init / (v_init + v_core)).clamp(0.15, 0.85);
                    let d = w_core * t_core + (1.0 - w_core) * init_dist[a * m + b];
                    let v = 1.0 / (1.0 / v_init + 1.0 / v_core.max(1e-6));
                    d_comb[a * m + b] = d;
                    d_comb[b * m + a] = d;
                    v_comb[a * m + b] = v;
                    v_comb[b * m + a] = v;
                }
            }
        }
        relax_shortest_paths_with_var(&mut d_comb, &mut v_comb, m);
    } else {
        let mut tax_core_cols: Vec<Vec<(usize, usize)>> = vec![Vec::new(); m];
        for (c_idx, col) in cols.iter().enumerate() {
            if col.len() < 2 {
                continue;
            }
            let bub_sum: f64 = col
                .iter()
                .map(|&(t, pos)| lookup.bubble_x_by_seq[t][pos])
                .sum();
            if bub_sum / (col.len() as f64) >= 0.35 {
                continue;
            }
            for &(t, pos) in col {
                let sym = comp_seqs[t].sym_proj[pos] as usize;
                if sym < k_dim {
                    pi_counts[sym] += taxon_nu[t];
                    tot_w += taxon_nu[t];
                    tax_core_cols[t].push((c_idx, sym));
                }
            }
        }

        let b0: f64 = pi_counts
            .iter()
            .map(|&c| {
                let p = c / tot_w;
                p * p
            })
            .sum::<f64>()
            .clamp(1.0 / (k_dim as f64), 0.85);
        let one_minus_b0 = (1.0 - b0).max(0.15);

        let updates: Vec<(usize, usize, f64, f64)> = lookup
            .active_keys
            .par_iter()
            .filter_map(|&(a, b)| {
                let ca = &tax_core_cols[a];
                let cb = &tax_core_cols[b];
                let (mut ia, mut ib) = (0usize, 0usize);
                let mut n_ab = 0usize;
                let mut matches = 0usize;
                while ia < ca.len() && ib < cb.len() {
                    match ca[ia].0.cmp(&cb[ib].0) {
                        std::cmp::Ordering::Less => ia += 1,
                        std::cmp::Ordering::Greater => ib += 1,
                        std::cmp::Ordering::Equal => {
                            n_ab += 1;
                            if ca[ia].1 == cb[ib].1 {
                                matches += 1;
                            }
                            ia += 1;
                            ib += 1;
                        }
                    }
                }
                if n_ab < 12 {
                    return None;
                }
                let f_id = (matches as f64) / (n_ab as f64);
                let norm_s = ((f_id - b0) / one_minus_b0).clamp(0.03, 1.0);
                let t_core = -one_minus_b0 * norm_s.ln();
                let v_core = (t_core + 0.04) / ((n_ab as f64) + 5.0);
                let v_init = init_var[a * m + b].max(1e-6);
                let w_core = (v_init / (v_init + v_core)).clamp(0.15, 0.85);
                let d = w_core * t_core + (1.0 - w_core) * init_dist[a * m + b];
                let v = 1.0 / (1.0 / v_init + 1.0 / v_core.max(1e-6));
                Some((a, b, d, v))
            })
            .collect();

        for (a, b, d, v) in updates {
            d_comb[a * m + b] = d;
            d_comb[b * m + a] = d;
            v_comb[a * m + b] = v;
            v_comb[b * m + a] = v;
        }
        if m <= 384 {
            relax_shortest_paths_with_var(&mut d_comb, &mut v_comb, m);
        }
    }

    let local_leaf_ids: Vec<usize> = (0..m).collect();
    let tree = build_bionj_tree(&local_leaf_ids, &d_comb, &v_comb, m);
    (tree, d_comb)
}

struct ProjectedComponentAlignment {
    annotated_cols: Vec<AnnotatedColumn>,
    num_global_fine: usize,
    num_clade_local: usize,
    num_private: usize,
    core_post_acc: f64,
    core_post_n: usize,
    horizon_bubbles: Vec<HorizonBubbleReport>,
}

/// Losslessly project refined MSA columns onto the updated phylogenetic tree via exact LCA,
/// using Euler-tour intervals (O(m) RAM), sparse lookup pair counts, and parallel split scans.
#[allow(clippy::too_many_arguments)]
fn project_alignment_onto_tree_with_extras(
    cols: &[Vec<(usize, usize)>],
    tree_nodes: &[TreeNodeRecord],
    root_node_id: usize,
    lookup_opt: Option<&ConsistentEdgeLookup>,
    bubble_x_override: Option<&[Vec<f64>]>,
    extra_col_post: Option<(&[f64], &[usize])>,
    comp_seqs: &[EncodedSeq],
    _gc: &GeneticCode,
    m: usize,
) -> ProjectedComponentAlignment {
    let num_nodes = tree_nodes.len();
    let mut leaf_to_node = vec![0usize; m];
    let mut parent_of = vec![usize::MAX; num_nodes];

    for rec in tree_nodes {
        if rec.is_leaf {
            if let Some(t) = rec.local_seq_idx {
                leaf_to_node[t] = rec.node_id;
            }
        } else {
            if let Some(lid) = rec.left_node_id {
                parent_of[lid] = rec.node_id;
            }
            if let Some(rid) = rec.right_node_id {
                parent_of[rid] = rec.node_id;
            }
        }
    }

    let mut depth = vec![0usize; num_nodes];
    let mut stack = vec![root_node_id];
    while let Some(u) = stack.pop() {
        let rec = &tree_nodes[u];
        if !rec.is_leaf {
            if let Some(lid) = rec.left_node_id {
                depth[lid] = depth[u] + 1;
                stack.push(lid);
            }
            if let Some(rid) = rec.right_node_id {
                depth[rid] = depth[u] + 1;
                stack.push(rid);
            }
        }
    }

    // Iterative Euler tour for O(1) subtree containment without O(m^2) desc_mask
    let mut leaf_dfs_pos = vec![0usize; m];
    let mut tin = vec![0usize; num_nodes];
    let mut tout = vec![0usize; num_nodes];
    {
        let mut dfs_st: Vec<(usize, bool)> = vec![(root_node_id, false)];
        let mut cursor = 0usize;
        while let Some((u, exiting)) = dfs_st.pop() {
            if exiting {
                tout[u] = cursor;
            } else {
                tin[u] = cursor;
                let rec = &tree_nodes[u];
                if rec.is_leaf {
                    if let Some(t) = rec.local_seq_idx {
                        leaf_dfs_pos[t] = cursor;
                        cursor += 1;
                    }
                    tout[u] = cursor;
                } else {
                    dfs_st.push((u, true));
                    if let Some(rid) = rec.right_node_id {
                        dfs_st.push((rid, false));
                    }
                    if let Some(lid) = rec.left_node_id {
                        dfs_st.push((lid, false));
                    }
                }
            }
        }
    }

    let num_cols = cols.len();
    let mut col_post_sum = vec![0.0f64; num_cols];
    if extra_col_post.is_none() {
        let lookup = lookup_opt.expect("lookup required when extra_col_post is None");
        let mut pos_to_col: Vec<Vec<usize>> = (0..m)
            .map(|t| vec![usize::MAX; comp_seqs[t].len()])
            .collect();
        for (c_idx, col) in cols.iter().enumerate() {
            for &(t, pos) in col {
                pos_to_col[t][pos] = c_idx;
            }
        }
        for &(a, b) in &lookup.active_keys {
            let pa = &pos_to_col[a];
            let pb = &pos_to_col[b];
            for &(ia, ib, prob) in lookup.pair_edges_lt(a, b) {
                let ca = pa[ia as usize];
                if ca != usize::MAX && ca == pb[ib as usize] {
                    col_post_sum[ca] += prob as f64;
                }
            }
        }
    }

    let bubble_src = bubble_x_override.unwrap_or_else(|| &lookup_opt.unwrap().bubble_x_by_seq);
    let mut annotated_cols = Vec::with_capacity(num_cols);
    let mut num_global_fine = 0usize;
    let mut num_clade_local = 0usize;
    let mut num_private = 0usize;
    let mut core_post_acc = 0.0f64;
    let mut core_post_n = 0usize;

    let mut col_stamp = vec![0usize; m];
    for (c_idx, col) in cols.iter().enumerate() {
        let occ = col.len();
        let bub_sum: f64 = col.iter().map(|&(t, pos)| bubble_src[t][pos]).sum();
        let mean_bub = if occ > 0 { bub_sum / (occ as f64) } else { 0.0 };

        let (post_sum_c, pair_cnt) = if let Some((ex_mass, ex_cnt)) = extra_col_post {
            (ex_mass[c_idx], ex_cnt[c_idx])
        } else {
            let lookup = lookup_opt.unwrap();
            let stamp = c_idx + 1;
            for &(t, _) in col {
                col_stamp[t] = stamp;
            }
            let mut pc = 0usize;
            for &(ta, _) in col {
                for &tb in &lookup.nbrs[ta] {
                    if tb > ta && col_stamp[tb] == stamp {
                        pc += 1;
                    }
                }
            }
            (col_post_sum[c_idx], pc)
        };
        let mean_post = if pair_cnt > 0 {
            (post_sum_c / (pair_cnt as f64)).clamp(0.0, 1.0)
        } else {
            1.0
        };

        let mut owner_node_id = leaf_to_node[col[0].0];
        for &(t, _) in &col[1..] {
            let mut a = owner_node_id;
            let mut b = leaf_to_node[t];
            while depth[a] > depth[b] {
                a = parent_of[a];
            }
            while depth[b] > depth[a] {
                b = parent_of[b];
            }
            while a != b {
                a = parent_of[a];
                b = parent_of[b];
            }
            owner_node_id = a;
        }

        let spans_root = m >= 2 && owner_node_id == root_node_id;
        let scale = if occ == 1 {
            num_private += 1;
            ColumnScale::LineagePrivate
        } else if spans_root || (m >= 4 && occ * 3 > 2 * m && mean_bub < 0.35) {
            num_global_fine += 1;
            core_post_acc += mean_post;
            core_post_n += 1;
            ColumnScale::GlobalFine
        } else {
            num_clade_local += 1;
            ColumnScale::CladeLocalFine
        };

        annotated_cols.push(AnnotatedColumn {
            col_index: c_idx,
            scale,
            owner_node_id,
            num_present: occ,
            mean_fine_posterior: mean_post,
            coarse_bubble_posterior: mean_bub,
        });
    }

    let mut horizon_bubbles = Vec::new();
    let mode = comp_seqs.first().map(|s| s.mode).unwrap_or(SeqMode::Protein);
    if m >= 2 {
        let stride = m + 1;
        let col_prefs: Vec<Vec<u32>> = cols
            .par_iter()
            .map(|col| {
                let mut pref = vec![0u32; stride];
                for &(t, _) in col {
                    pref[leaf_dfs_pos[t] + 1] = 1;
                }
                for p in 1..stride {
                    pref[p] += pref[p - 1];
                }
                pref
            })
            .collect();

        horizon_bubbles = (m..num_nodes)
            .into_par_iter()
            .flat_map(|split_node_id| {
                let cur_step = split_node_id - m + 1;
                let rec = &tree_nodes[split_node_id];
                let (Some(left_id), Some(right_id)) = (rec.left_node_id, rec.right_node_id) else {
                    return Vec::new();
                };
                let l_tin = tin[left_id];
                let l_tout = tout[left_id];
                let r_tin = tin[right_id];
                let r_tout = tout[right_id];
                let n_l = l_tout - l_tin;
                let n_r = r_tout - r_tin;

                let mut ul: Vec<(usize, bool)> = Vec::new();
                let mut ur: Vec<(usize, bool)> = Vec::new();
                let mut seen_anchor = false;
                let mut local_bubbles = Vec::new();

                for c_idx in 0..num_cols {
                    let pref = &col_prefs[c_idx];
                    let l_cnt = (pref[l_tout] - pref[l_tin]) as usize;
                    let r_cnt = (pref[r_tout] - pref[r_tin]) as usize;
                    if l_cnt == 0 && r_cnt == 0 {
                        continue;
                    }
                    if l_cnt > 0 && r_cnt > 0 {
                        let anc_ul = ul.iter().filter(|x| x.1).count();
                        let anc_ur = ur.iter().filter(|x| x.1).count();
                        let min_anc_cols = if mode == SeqMode::Dna { 9 } else { 3 };
                        if seen_anchor && anc_ul >= min_anc_cols && anc_ur >= min_anc_cols {
                            let mut counts_l = [0usize; NUM_BIOPHYS_CLASSES];
                            let mut counts_r = [0usize; NUM_BIOPHYS_CLASSES];
                            let mut nt_l = [0usize; 4];
                            let mut nt_r = [0usize; 4];
                            let mut prop_l = [0.0f64; 3];
                            let mut prop_r = [0.0f64; 3];
                            let mut nl = 0usize;
                            let mut nr = 0usize;
                            let mut l_bounds: Vec<Option<(usize, usize)>> = vec![None; m];
                            let mut r_bounds: Vec<Option<(usize, usize)>> = vec![None; m];

                            for &(col_i, _) in &ul {
                                for &(ta, ia) in &cols[col_i] {
                                    let dp = leaf_dfs_pos[ta];
                                    if dp < l_tin || dp >= l_tout {
                                        continue;
                                    }
                                    l_bounds[ta] = match l_bounds[ta] {
                                        None => Some((ia, ia)),
                                        Some((lo, hi)) => Some((lo.min(ia), hi.max(ia))),
                                    };
                                    let sym = comp_seqs[ta].sym_proj[ia] as usize;
                                    if mode == SeqMode::Dna {
                                        if sym < 4 {
                                            nt_l[sym] += 1;
                                            nl += 1;
                                        }
                                    } else if sym < 20 {
                                        counts_l[AA_TO_BIOPHYS[sym]] += 1;
                                        for k in 0..3 {
                                            prop_l[k] += AA_PROPERTIES[sym][k];
                                        }
                                        nl += 1;
                                    }
                                }
                            }
                            for &(col_i, _) in &ur {
                                for &(tb, ib) in &cols[col_i] {
                                    let dp = leaf_dfs_pos[tb];
                                    if dp < r_tin || dp >= r_tout {
                                        continue;
                                    }
                                    r_bounds[tb] = match r_bounds[tb] {
                                        None => Some((ib, ib)),
                                        Some((lo, hi)) => Some((lo.min(ib), hi.max(ib))),
                                    };
                                    let sym = comp_seqs[tb].sym_proj[ib] as usize;
                                    if mode == SeqMode::Dna {
                                        if sym < 4 {
                                            nt_r[sym] += 1;
                                            nr += 1;
                                        }
                                    } else if sym < 20 {
                                        counts_r[AA_TO_BIOPHYS[sym]] += 1;
                                        for k in 0..3 {
                                            prop_r[k] += AA_PROPERTIES[sym][k];
                                        }
                                        nr += 1;
                                    }
                                }
                            }
                            let min_sym = if mode == SeqMode::Dna { 9 } else { 3 };
                            if nl >= min_sym && nr >= min_sym {
                                let left_intervals: Vec<(usize, usize, usize)> = l_bounds
                                    .into_iter()
                                    .enumerate()
                                    .filter_map(|(t, opt)| opt.map(|(lo, hi)| (t, lo, hi - lo + 1)))
                                    .collect();
                                let right_intervals: Vec<(usize, usize, usize)> = r_bounds
                                    .into_iter()
                                    .enumerate()
                                    .filter_map(|(t, opt)| opt.map(|(lo, hi)| (t, lo, hi - lo + 1)))
                                    .collect();
                                let (dom_label, gain, d_hydro, d_chg) = if mode == SeqMode::Dna {
                                    let (lbl, gn, d_gc, d_pur) =
                                        summarize_dna_bubble_composition(&nt_l, &nt_r);
                                    (
                                        lbl,
                                        gn.unwrap_or(0.0),
                                        d_gc.unwrap_or(0.0),
                                        d_pur.unwrap_or(0.0),
                                    )
                                } else {
                                    let mut joint = [0usize; NUM_BIOPHYS_CLASSES];
                                    let mut dom = 0usize;
                                    for k in 0..NUM_BIOPHYS_CLASSES {
                                        joint[k] = counts_l[k] + counts_r[k];
                                        if joint[k] > joint[dom] {
                                            dom = k;
                                        }
                                    }
                                    let alpha = [1.0f64; NUM_BIOPHYS_CLASSES];
                                    let gn = exchangeable_dirichlet_seq_ln_prob(&joint, &alpha)
                                        - exchangeable_dirichlet_seq_ln_prob(&counts_l, &alpha)
                                        - exchangeable_dirichlet_seq_ln_prob(&counts_r, &alpha);
                                    (
                                        Some(BIOPHYS_CLASS_NAMES[dom].to_string()),
                                        gn,
                                        (prop_r[0] / nr as f64) - (prop_l[0] / nl as f64),
                                        (prop_r[2] / nr as f64) - (prop_l[2] / nl as f64),
                                    )
                                };
                                local_bubbles.push(HorizonBubbleReport {
                                    tree_merge_step: cur_step,
                                    split_node_id,
                                    left_node_id: left_id,
                                    right_node_id: right_id,
                                    left_clade_size: n_l,
                                    right_clade_size: n_r,
                                    left_col_span: anc_ul,
                                    right_col_span: anc_ur,
                                    left_intervals,
                                    right_intervals,
                                    dominant_biophys_class: dom_label,
                                    macro_biophys_gain_nats: gain,
                                    delta_hydropathy: d_hydro,
                                    delta_charge: d_chg,
                                });
                            }
                        }
                        ul.clear();
                        ur.clear();
                        seen_anchor = true;
                    } else if l_cnt > 0 {
                        ul.push((c_idx, 2 * l_cnt >= n_l));
                    } else {
                        ur.push((c_idx, 2 * r_cnt >= n_r));
                    }
                }
                local_bubbles
            })
            .collect();
        horizon_bubbles.sort_unstable_by_key(|hb| hb.tree_merge_step);
    }

    ProjectedComponentAlignment {
        annotated_cols,
        num_global_fine,
        num_clade_local,
        num_private,
        core_post_acc,
        core_post_n,
        horizon_bubbles,
    }
}

fn project_alignment_onto_tree(
    cols: &[Vec<(usize, usize)>],
    tree_nodes: &[TreeNodeRecord],
    root_node_id: usize,
    lookup: &ConsistentEdgeLookup,
    comp_seqs: &[EncodedSeq],
    gc: &GeneticCode,
    m: usize,
) -> ProjectedComponentAlignment {
    project_alignment_onto_tree_with_extras(
        cols,
        tree_nodes,
        root_node_id,
        Some(lookup),
        None,
        None,
        comp_seqs,
        gc,
        m,
    )
}

#[derive(Clone)]
struct FollowerAttachment {
    leaf_global: usize,
    delta: usize,
    d_leaf: f64,
    cond_nats: f64,
}

struct ColinearSkeleton {
    med_global: Vec<usize>,
    followers_by_med: Vec<Vec<FollowerAttachment>>,
}

#[inline]
fn hash_sym_proj(sym_proj: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ (sym_proj.len() as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    for &b in sym_proj {
        h ^= b as u64;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    h
}

/// Strictly lossless, zero-indel verification of attaching `u` onto medoid `m` at ungapped offset `delta`.
fn verify_colinear_attachment(
    u: &PreparedSeq,
    m: &PreparedSeq,
    delta: usize,
    mode: SeqMode,
    n_total: usize,
) -> Option<(f64, f64)> {
    let lu = u.seq.len();
    let lm = m.seq.len();
    if lu < 24 || lm < lu || delta + lu > lm {
        return None;
    }
    if u.seq.non_ambig_len * 10 < lu * 9 {
        return None;
    }
    let k_dim = if mode == SeqMode::Dna { 4u8 } else { 20u8 };
    let (num_req, den_req) = if mode == SeqMode::Dna {
        (24usize, 25usize) // 96%
    } else {
        (19usize, 20usize) // 95%
    };

    let u_sym = &u.aa_proj;
    let m_sym = &m.aa_proj[delta..delta + lu];
    let u_tok = &u.seq.tokens;
    let m_tok = &m.seq.tokens[delta..delta + lu];

    let mut info_tot = 0usize;
    let mut info_match = 0usize;
    let mut tok_diff = 0usize;

    let mut ring = [0u8; 8];
    let mut win_info = 0usize;
    let mut win_match = 0usize;

    let mut head_info = 0usize;
    let mut head_match = 0usize;
    let mut tail_info = 0usize;
    let mut tail_match = 0usize;

    for p in 0..lu {
        let us = u_sym[p];
        let ms = m_sym[p];
        if u_tok[p] != m_tok[p] {
            tok_diff += 1;
        }
        let code = if us < k_dim {
            // Never collapse a concrete position onto an unresolved/ambiguous medoid position
            if ms >= k_dim {
                return None;
            }
            info_tot += 1;
            let eq = us == ms;
            if eq {
                info_match += 1;
                2u8
            } else {
                1u8
            }
        } else {
            0u8
        };

        if p < 6 && code > 0 {
            head_info += 1;
            if code == 2 {
                head_match += 1;
            }
        }
        if p + 6 >= lu && code > 0 {
            tail_info += 1;
            if code == 2 {
                tail_match += 1;
            }
        }

        if p >= 8 {
            let old = ring[p & 7];
            if old > 0 {
                win_info -= 1;
                if old == 2 {
                    win_match -= 1;
                }
            }
        }
        ring[p & 7] = code;
        if code > 0 {
            win_info += 1;
            if code == 2 {
                win_match += 1;
            }
        }
        if p >= 7 && win_info >= 6 && win_match * 4 < win_info * 3 {
            return None;
        }
    }

    if info_tot < 24 || info_match * den_req < info_tot * num_req {
        return None;
    }
    if (head_info >= 5 && head_info - head_match > 1) || (tail_info >= 5 && tail_info - tail_match > 1) {
        return None;
    }

    let one_minus_bg = if mode == SeqMode::Dna { 0.75 } else { 0.95 };
    let p_mm = 1.0 - (info_match as f64) / (info_tot as f64);
    let d_leaf = if p_mm <= 0.0 {
        0.0
    } else {
        (-one_minus_bg * (1.0 - p_mm / one_minus_bg).clamp(0.02, 1.0).ln()).max(1e-4)
    };

    let k_alpha = u.seq.alphabet_size().max(2);
    let q = ((tok_diff as f64) + 0.5) / ((lu as f64) + 1.0);
    let h_bin = -q * q.ln() - (1.0 - q) * (1.0 - q).ln();
    let ptr_cost = if n_total > 2 {
        ((n_total - 1) as f64).ln()
    } else {
        0.0
    };
    let cond_nats = ((lu as f64) * h_bin
        + (tok_diff as f64) * ((k_alpha - 1) as f64).ln()
        + ptr_cost
        + 6.0)
        .min(u.l0_nats);

    Some((d_leaf, cond_nats))
}

fn build_colinear_skeleton(
    prepared: &[PreparedSeq],
    repaired_ids: &HashSet<&str>,
    mode: SeqMode,
) -> ColinearSkeleton {
    let n = prepared.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| {
        let rep_a = repaired_ids.contains(prepared[a].seq.id.as_str());
        let rep_b = repaired_ids.contains(prepared[b].seq.id.as_str());
        let clean_a = !prepared[a].seq.has_ambig && !rep_a;
        let clean_b = !prepared[b].seq.has_ambig && !rep_b;
        clean_b
            .cmp(&clean_a)
            .then_with(|| (!rep_b).cmp(&(!rep_a)))
            .then_with(|| {
                prepared[b]
                    .seq
                    .non_ambig_len
                    .cmp(&prepared[a].seq.non_ambig_len)
            })
            .then_with(|| prepared[b].seq.len().cmp(&prepared[a].seq.len()))
            .then_with(|| a.cmp(&b))
    });

    let mut med_global: Vec<usize> = Vec::new();
    let mut followers_by_med: Vec<Vec<FollowerAttachment>> = Vec::new();
    let mut exact_map: HashMap<u64, Vec<usize>> = HashMap::new();
    let mut same_len_seeds: HashMap<(u32, u16, u16), Vec<u32>> = HashMap::with_capacity(n * 8);
    let mut seed_index: HashMap<u32, Vec<(u32, u32)>> = HashMap::with_capacity(n * 4);
    let mut same_raw: Vec<u32> = Vec::with_capacity(256);
    let mut same_scored: Vec<(usize, usize)> = Vec::with_capacity(64);
    let mut raw_hits: Vec<u64> = Vec::with_capacity(512);
    let mut scored_hits: Vec<(usize, usize, usize)> = Vec::with_capacity(64);

    for &u_idx in &order {
        let u = &prepared[u_idx];
        let lu = u.seq.len();
        let lu_key = lu.min(u16::MAX as usize) as u16;
        let mut attached = false;

        if lu >= 24 && u.seq.non_ambig_len * 10 >= lu * 9 {
            let sig = hash_sym_proj(&u.aa_proj);
            if let Some(ex_meds) = exact_map.get(&sig) {
                for &sk_idx in ex_meds {
                    let mg = med_global[sk_idx];
                    if prepared[mg].seq.len() == lu {
                        if let Some((d_leaf, cond_nats)) =
                            verify_colinear_attachment(u, &prepared[mg], 0, mode, n)
                        {
                            followers_by_med[sk_idx].push(FollowerAttachment {
                                leaf_global: u_idx,
                                delta: 0,
                                d_leaf,
                                cond_nats,
                            });
                            attached = true;
                            break;
                        }
                    }
                }
            }

            // Fast zero-offset same-length seed lookup (prevents insertion outliers from crowding out point mutants)
            if !attached && !u.seeds.is_empty() {
                same_raw.clear();
                let n_seeds = u.seeds.len();
                let step = if n_seeds <= 24 {
                    1
                } else {
                    (n_seeds / 24).max(1)
                };
                let mut s_i = 0usize;
                let mut probed = 0usize;
                while s_i < n_seeds && probed < 24 {
                    let (key, pos_u, _, _) = u.seeds[s_i];
                    if let Some(list) =
                        same_len_seeds.get(&(key, lu_key, pos_u.min(u16::MAX as u32) as u16))
                    {
                        same_raw.extend_from_slice(list);
                    }
                    s_i += step;
                    probed += 1;
                }
                if !same_raw.is_empty() {
                    same_raw.sort_unstable();
                    same_scored.clear();
                    let mut hi = 0usize;
                    while hi < same_raw.len() {
                        let sk = same_raw[hi];
                        let mut end = hi + 1;
                        while end < same_raw.len() && same_raw[end] == sk {
                            end += 1;
                        }
                        same_scored.push((end - hi, sk as usize));
                        hi = end;
                    }
                    same_scored.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
                    for &(_, sk_idx) in same_scored.iter().take(4) {
                        let mg = med_global[sk_idx];
                        if let Some((d_leaf, cond_nats)) =
                            verify_colinear_attachment(u, &prepared[mg], 0, mode, n)
                        {
                            followers_by_med[sk_idx].push(FollowerAttachment {
                                leaf_global: u_idx,
                                delta: 0,
                                d_leaf,
                                cond_nats,
                            });
                            attached = true;
                            break;
                        }
                    }
                }
            }

            if !attached && !u.seeds.is_empty() {
                raw_hits.clear();
                let n_seeds = u.seeds.len();
                let step = if n_seeds <= 36 {
                    1
                } else {
                    (n_seeds / 36).max(1)
                };
                let mut s_i = 0usize;
                let mut probed = 0usize;
                while s_i < n_seeds && probed < 36 {
                    let (key, pos_u, _, _) = u.seeds[s_i];
                    if let Some(list) = seed_index.get(&key) {
                        for &(sk_u32, pos_m) in list {
                            if pos_m >= pos_u {
                                let delta = (pos_m - pos_u) as usize;
                                let mg = med_global[sk_u32 as usize];
                                let lm = prepared[mg].seq.len();
                                if lm > lu && delta + lu <= lm {
                                    raw_hits.push(((sk_u32 as u64) << 32) | (delta as u64));
                                }
                            }
                        }
                    }
                    s_i += step;
                    probed += 1;
                }

                if !raw_hits.is_empty() {
                    raw_hits.sort_unstable();
                    scored_hits.clear();
                    let mut hi = 0usize;
                    while hi < raw_hits.len() {
                        let key_sd = raw_hits[hi];
                        let mut end = hi + 1;
                        while end < raw_hits.len() && raw_hits[end] == key_sd {
                            end += 1;
                        }
                        scored_hits.push((
                            end - hi,
                            (key_sd >> 32) as usize,
                            (key_sd & 0xFFFF_FFFF) as usize,
                        ));
                        hi = end;
                    }
                    scored_hits.sort_unstable_by(|a, b| {
                        b.0.cmp(&a.0)
                            .then_with(|| a.1.cmp(&b.1))
                            .then_with(|| a.2.cmp(&b.2))
                    });
                    for &(_, sk_idx, delta) in scored_hits.iter().take(6) {
                        let mg = med_global[sk_idx];
                        if let Some((d_leaf, cond_nats)) =
                            verify_colinear_attachment(u, &prepared[mg], delta, mode, n)
                        {
                            followers_by_med[sk_idx].push(FollowerAttachment {
                                leaf_global: u_idx,
                                delta,
                                d_leaf,
                                cond_nats,
                            });
                            attached = true;
                            break;
                        }
                    }
                }
            }
        }

        if !attached {
            let sk_idx = med_global.len();
            med_global.push(u_idx);
            followers_by_med.push(Vec::new());
            let sig = hash_sym_proj(&u.aa_proj);
            exact_map.entry(sig).or_default().push(sk_idx);

            let n_seeds = u.seeds.len();
            if n_seeds > 0 {
                let step = if n_seeds <= 48 {
                    1
                } else {
                    (n_seeds / 48).max(1)
                };
                let mut s_i = 0usize;
                let mut added = 0usize;
                while s_i < n_seeds && added < 48 {
                    let (key, pos, _, _) = u.seeds[s_i];
                    let sl_vec = same_len_seeds
                        .entry((key, lu_key, (pos as usize).min(u16::MAX as usize) as u16))
                        .or_default();
                    if sl_vec.len() < 16 {
                        sl_vec.push(sk_idx as u32);
                    }
                    let vec_ref = seed_index.entry(key).or_default();
                    if vec_ref.len() < 48 {
                        let same_len_cnt = vec_ref
                            .iter()
                            .filter(|&&(ms, _)| prepared[med_global[ms as usize]].seq.len() == lu)
                            .count();
                        if same_len_cnt < 4 {
                            vec_ref.push((sk_idx as u32, pos));
                        }
                    }
                    s_i += step;
                    added += 1;
                }
            }
        }
    }

    let mut pairs: Vec<(usize, Vec<FollowerAttachment>)> = med_global
        .into_iter()
        .zip(followers_by_med.into_iter())
        .collect();
    pairs.sort_unstable_by_key(|x| x.0);
    let mut sorted_meds = Vec::with_capacity(pairs.len());
    let mut sorted_followers = Vec::with_capacity(pairs.len());
    for (g, fs) in pairs {
        sorted_meds.push(g);
        sorted_followers.push(fs);
    }

    ColinearSkeleton {
        med_global: sorted_meds,
        followers_by_med: sorted_followers,
    }
}

fn graft_followers_into_tree(
    node: &PhyloNode,
    skel_comp_indices: &[usize],
    skel_to_full_local: &[usize],
    followers_by_med: &[Vec<FollowerAttachment>],
    global_to_full_local: &HashMap<usize, usize>,
) -> PhyloNode {
    match node {
        PhyloNode::Leaf(s_loc) => {
            let sk_idx = skel_comp_indices[*s_loc];
            let med_full = skel_to_full_local[*s_loc];
            let fs = &followers_by_med[sk_idx];
            if fs.is_empty() {
                return PhyloNode::Leaf(med_full);
            }
            let mut sorted_fs: Vec<(f64, usize)> = fs
                .iter()
                .map(|f| (f.d_leaf, global_to_full_local[&f.leaf_global]))
                .collect();
            sorted_fs.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            let mut cur = PhyloNode::Leaf(med_full);
            for (d, f_full) in sorted_fs {
                cur = PhyloNode::Internal {
                    left: Box::new(cur),
                    right: Box::new(PhyloNode::Leaf(f_full)),
                    len_left: 0.0,
                    len_right: d,
                };
            }
            cur
        }
        PhyloNode::Internal {
            left,
            right,
            len_left,
            len_right,
        } => PhyloNode::Internal {
            left: Box::new(graft_followers_into_tree(
                left,
                skel_comp_indices,
                skel_to_full_local,
                followers_by_med,
                global_to_full_local,
            )),
            right: Box::new(graft_followers_into_tree(
                right,
                skel_comp_indices,
                skel_to_full_local,
                followers_by_med,
                global_to_full_local,
            )),
            len_left: *len_left,
            len_right: *len_right,
        },
    }
}

/// Run the complete Multi-Sequence MDL Provenance & Joint Phylogeny Engine on raw unaligned sequences.
pub fn analyze_unaligned_cohort(
    raw_seqs: &[RawSeq],
    explicit_mode: Option<SeqMode>,
) -> CohortHomologyLedger {
    let gc = GeneticCode::for_cohort(raw_seqs);
    let mode = explicit_mode.unwrap_or_else(|| detect_seq_mode(raw_seqs, &gc));
    let mut seqs: Vec<EncodedSeq> = raw_seqs
        .iter()
        .map(|r| EncodedSeq::from_raw(r, mode, &gc))
        .collect();
    let n = seqs.len();

    // 1. Pre-audit reading frames on coding cohorts against closest clean sisters + stealth check!
    let (cohort_frameshifts, cohort_fs_pen) = audit_cohort_frameshifts(&mut seqs, &gc);

    // 2. Single-sequence lossless null profiles & Global Regime Tables
    let tables = get_global_regime_tables(mode, &gc);
    let mut prepared: Vec<PreparedSeq> = seqs
        .into_par_iter()
        .map(|s| PreparedSeq::new(s, &gc))
        .collect();
    let seq_lens: Vec<usize> = prepared.iter().map(|p| p.seq.len()).collect();
    let non_ambig_lens: Vec<usize> = prepared.iter().map(|p| p.seq.non_ambig_len).collect();
    if n >= 2 {
        prewarm_path_partitions_for_cohort(&seq_lens, &tables.regimes, PRIOR_MEAN_LEN);
    }
    let single_null_nats: Vec<f64> = prepared.iter().map(|p| p.l0_nats).collect();
    let raw_null_cohort_nats: f64 = single_null_nats.iter().sum();

    if n == 0 {
        return CohortHomologyLedger {
            mode,
            mito_tga_trp: gc.mito_tga_trp,
            num_sequences: 0,
            raw_null_cohort_nats: 0.0,
            provenance_cohort_nats: 0.0,
            net_cohort_gain_nats: 0.0,
            num_homology_components: 0,
            components: Vec::new(),
            pairwise_summary: Vec::new(),
        };
    }

    if n == 1 {
        let aln_str: String = prepared[0].seq.token_strings.concat();
        let l1 = prepared[0].seq.len();
        let cols: Vec<AnnotatedColumn> = (0..l1)
            .map(|c| AnnotatedColumn {
                col_index: c,
                scale: ColumnScale::LineagePrivate,
                owner_node_id: 0,
                num_present: 1,
                mean_fine_posterior: 1.0,
                coarse_bubble_posterior: 0.0,
            })
            .collect();
        let (tree_nodes, root_node_id, newick_tree) =
            PhyloNode::Leaf(0).flatten_and_label(&[prepared[0].seq.id.clone()]);
        return CohortHomologyLedger {
            mode,
            mito_tga_trp: gc.mito_tga_trp,
            num_sequences: 1,
            raw_null_cohort_nats,
            provenance_cohort_nats: raw_null_cohort_nats,
            net_cohort_gain_nats: 0.0,
            num_homology_components: 1,
            components: vec![HomologyComponentReport {
                component_id: 0,
                sequence_ids: vec![prepared[0].seq.id.clone()],
                newick_tree,
                root_node_id,
                tree_nodes,
                unaligned_seqs: vec![(prepared[0].seq.id.clone(), aln_str.clone())],
                num_columns: l1,
                num_global_fine_cols: 0,
                num_clade_local_cols: 0,
                num_lineage_private_cols: l1,
                mean_core_posterior: 1.0,
                horizon_bubbles: Vec::new(),
                frameshifts_repaired: cohort_frameshifts,
                aligned_fasta: vec![(prepared[0].seq.id.clone(), aln_str)],
                columns: cols,
            }],
            pairwise_summary: Vec::new(),
        };
    }

    // 2.5 Lossless Zero-Indel Colinear Skeleton Collapsing for large cohorts (N > 80)
    let skeleton = if n > 80 {
        let repaired_ids: HashSet<&str> = cohort_frameshifts
            .iter()
            .map(|ev| ev.seq_id.as_str())
            .collect();
        build_colinear_skeleton(&prepared, &repaired_ids, mode)
    } else {
        ColinearSkeleton {
            med_global: (0..n).collect(),
            followers_by_med: vec![Vec::new(); n],
        }
    };
    let n_skel = skeleton.med_global.len();
    for fs in &skeleton.followers_by_med {
        for f in fs {
            prepared[f.leaf_global].clear_dp_caches();
        }
    }
    let mut seq_to_fs_idx: HashMap<&str, Vec<usize>> = HashMap::new();
    for (idx, ev) in cohort_frameshifts.iter().enumerate() {
        seq_to_fs_idx.entry(ev.seq_id.as_str()).or_default().push(idx);
    }

    // 3. Parallel Pairwise Multi-Scale MDL Homology Evaluation on skeleton medoids (`0..n_skel`)
    let mut pair_map: HashMap<u64, PairHomologyResult> = HashMap::new();
    let mut state_map: HashMap<u64, PairForwardState> = HashMap::new();
    let mut parent: Vec<usize> = (0..n_skel).collect();

    // Exact Kraft parent-pointer code length calibrated to FULL cohort size N
    let edge_ptr_cost = if n > 2 { ((n - 1) as f64).ln() } else { 0.0 };
    let mst_ptr_cost = edge_ptr_cost;
    const DENSE_PAIR_MAX_N: usize = 20;

    let mut cand_edges: Vec<(f64, usize, usize)> = Vec::new();
    if n_skel <= 1 {
        // Single skeleton medoid covers the entire cohort via colinear followers!
    } else if n_skel <= DENSE_PAIR_MAX_N {
        let mut pair_coords = Vec::with_capacity(n_skel * (n_skel - 1) / 2);
        for a in 0..n_skel {
            for b in (a + 1)..n_skel {
                pair_coords.push((a, b));
            }
        }
        let evaluated: Vec<((usize, usize), PairHomologyResult)> = pair_coords
            .into_par_iter()
            .map(|(a, b)| {
                let ga = skeleton.med_global[a];
                let gb = skeleton.med_global[b];
                let (lo, hi) = if ga < gb { (ga, gb) } else { (gb, ga) };
                let (res, _) =
                    eval_pair_with_fallback(&prepared[lo], &prepared[hi], &gc, &tables, true);
                ((lo, hi), res)
            })
            .collect();
        for ((lo, hi), res) in evaluated {
            pair_map.insert(pair_key(lo, hi), res);
        }
        for a in 0..n_skel {
            for b in (a + 1)..n_skel {
                let gain = pair_map[&pair_key(skeleton.med_global[a], skeleton.med_global[b])]
                    .mdl_gain_nats;
                if gain > edge_ptr_cost {
                    cand_edges.push((gain, a, b));
                }
            }
        }
    } else {
        let e_cand_local = build_sparse_candidate_pairs_subset(&prepared, &skeleton.med_global);
        const PASS1_CHUNK: usize = 3_000;
        for chunk in e_cand_local.chunks(PASS1_CHUNK) {
            let evaluated: Vec<((usize, usize), (usize, usize), PairHomologyResult, Option<PairForwardState>)> =
                chunk
                    .par_iter()
                    .copied()
                    .map(|(a, b)| {
                        let ga = skeleton.med_global[a];
                        let gb = skeleton.med_global[b];
                        let (lo, hi) = if ga < gb { (ga, gb) } else { (gb, ga) };
                        let (res, st) =
                            eval_pair_with_fallback(&prepared[lo], &prepared[hi], &gc, &tables, false);
                        ((a, b), (lo, hi), res, st)
                    })
                    .collect();
            for ((a, b), (lo, hi), res, st) in evaluated {
                if res.mdl_gain_nats > edge_ptr_cost {
                    cand_edges.push((res.mdl_gain_nats, a, b));
                }
                let pk = pair_key(lo, hi);
                pair_map.insert(pk, res);
                if let Some(s) = st {
                    state_map.insert(pk, s);
                }
            }
            if n_skel > 300 {
                for &g in &skeleton.med_global {
                    prepared[g].reset_regime_locks();
                }
            }
        }
    }

    cand_edges.sort_unstable_by(|x, y| {
        y.0.total_cmp(&x.0)
            .then_with(|| x.1.cmp(&y.1))
            .then_with(|| x.2.cmp(&y.2))
    });
    let mut comp_sz: Vec<usize> = (0..n_skel)
        .map(|s| 1 + skeleton.followers_by_med[s].len())
        .collect();
    for (gain, a, b) in cand_edges {
        let ra = uf_find(&mut parent, a);
        let rb = uf_find(&mut parent, b);
        if ra != rb {
            let sa = comp_sz[ra];
            let sb = comp_sz[rb];
            let req = if sa >= 2 && sb >= 2 {
                edge_ptr_cost + ((sa * sb) as f64).ln()
            } else {
                edge_ptr_cost
            };
            if gain > req {
                parent[rb] = ra;
                comp_sz[ra] += sb;
            }
        }
    }

    // Group skeleton medoids into connected Homology Components (Pass 1)
    let mut comp_groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for a in 0..n_skel {
        let r = uf_find(&mut parent, a);
        comp_groups.entry(r).or_default().push(a);
    }

    // Pass 2: Profile-guided cross-component rescue over global medoids
    if comp_groups.len() >= 2 && n_skel >= 3 {
        let mut g_parent: Vec<usize> = (0..n).collect();
        let mut g_groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for members in comp_groups.values() {
            let g_rep = skeleton.med_global[members[0]];
            let mut g_members = Vec::with_capacity(members.len());
            for &sk in members {
                let g = skeleton.med_global[sk];
                g_parent[g] = g_rep;
                g_members.push(g);
            }
            g_groups.insert(g_rep, g_members);
        }
        if try_rescue_components(
            n,
            &mut prepared,
            &gc,
            &tables,
            &seq_lens,
            &mut pair_map,
            &mut state_map,
            &mut g_parent,
            &g_groups,
        ) {
            let mut g2sk: HashMap<usize, usize> = HashMap::with_capacity(n_skel);
            for (sk, &g) in skeleton.med_global.iter().enumerate() {
                g2sk.insert(g, sk);
            }
            comp_groups.clear();
            for sk in 0..n_skel {
                let rg = uf_find(&mut g_parent, skeleton.med_global[sk]);
                let r_sk = g2sk[&rg];
                comp_groups.entry(r_sk).or_default().push(sk);
            }
        }
    }

    // Exact Stirling 3-way partition header calibration for genuinely isolated `m_full == 2` components:
    let s_iso: usize = comp_groups
        .values()
        .map(|v| {
            let tot: usize = v
                .iter()
                .map(|&sk| 1 + skeleton.followers_by_med[sk].len())
                .sum();
            if tot <= 2 { tot } else { 0 }
        })
        .sum();
    if s_iso >= 4 {
        let iso_pair_thresh = (((s_iso - 2) * s_iso * (s_iso - 1) / 2) as f64).ln();
        let mut to_split: Vec<(usize, usize, usize)> = Vec::new();
        for (&root_key, members) in &comp_groups {
            if members.len() == 2
                && skeleton.followers_by_med[members[0]].is_empty()
                && skeleton.followers_by_med[members[1]].is_empty()
            {
                let g0 = skeleton.med_global[members[0]];
                let g1 = skeleton.med_global[members[1]];
                let gain = pair_map
                    .get(&pair_key(g0, g1))
                    .map(|r| r.mdl_gain_nats)
                    .unwrap_or(f64::NEG_INFINITY);
                if gain <= iso_pair_thresh {
                    to_split.push((root_key, members[0], members[1]));
                }
            }
        }
        for (root_key, s0, s1) in to_split {
            comp_groups.remove(&root_key);
            comp_groups.insert(s0, vec![s0]);
            comp_groups.insert(s1, vec![s1]);
        }
    }

    let gap_token = if mode == SeqMode::Codon { "---" } else { "-" };

    let mut components = Vec::new();
    let mut provenance_cohort_nats = cohort_fs_pen;
    let mut pair_summary_counts: HashMap<u64, (usize, usize)> = HashMap::new();

    for (comp_id, skel_comp_indices) in comp_groups.values().enumerate() {
        let m_skel = skel_comp_indices.len();
        let n_followers: usize = skel_comp_indices
            .iter()
            .map(|&sk| skeleton.followers_by_med[sk].len())
            .sum();

        if m_skel == 1 && n_followers == 0 {
            let g_idx = skeleton.med_global[skel_comp_indices[0]];
            prepared[g_idx].clear_dp_caches();
            let seq0 = &prepared[g_idx].seq;
            provenance_cohort_nats += single_null_nats[g_idx];
            let aln_str = seq0.token_strings.concat();
            let comp_names = vec![seq0.id.clone()];
            let cols: Vec<AnnotatedColumn> = (0..seq0.len())
                .map(|c| AnnotatedColumn {
                    col_index: c,
                    scale: ColumnScale::LineagePrivate,
                    owner_node_id: 0,
                    num_present: 1,
                    mean_fine_posterior: 1.0,
                    coarse_bubble_posterior: 0.0,
                })
                .collect();
            let (tree_nodes, root_node_id, newick_tree) =
                PhyloNode::Leaf(0).flatten_and_label(&comp_names);
            let comp_fs: Vec<FrameshiftEvent> = seq_to_fs_idx
                .get(comp_names[0].as_str())
                .map(|idxs| {
                    idxs.iter()
                        .map(|&i| cohort_frameshifts[i].clone())
                        .collect()
                })
                .unwrap_or_default();
            components.push(HomologyComponentReport {
                component_id: comp_id,
                sequence_ids: comp_names.clone(),
                newick_tree,
                root_node_id,
                tree_nodes,
                unaligned_seqs: vec![(comp_names[0].clone(), aln_str.clone())],
                num_columns: seq0.len(),
                num_global_fine_cols: 0,
                num_clade_local_cols: 0,
                num_lineage_private_cols: seq0.len(),
                mean_core_posterior: 1.0,
                horizon_bubbles: Vec::new(),
                frameshifts_repaired: comp_fs,
                aligned_fasta: vec![(comp_names[0].clone(), aln_str)],
                columns: cols,
            });
            continue;
        }

        let skel_global_indices: Vec<usize> = skel_comp_indices
            .iter()
            .map(|&sk| skeleton.med_global[sk])
            .collect();
        let skel_comp_seqs: Vec<EncodedSeq> = skel_global_indices
            .iter()
            .map(|&g| prepared[g].seq.clone())
            .collect();
        let skel_comp_names: Vec<String> = skel_comp_seqs.iter().map(|s| s.id.clone()).collect();

        let (skel_cols, skel_tree, skel_lookup) = if m_skel == 1 {
            prepared[skel_global_indices[0]].clear_dp_caches();
            let lm = skel_comp_seqs[0].len();
            provenance_cohort_nats += single_null_nats[skel_global_indices[0]];
            let cols: Vec<Vec<(usize, usize)>> = (0..lm).map(|p| vec![(0, p)]).collect();
            let empty_lookup = ConsistentEdgeLookup {
                active_keys: Vec::new(),
                edges_map: HashMap::new(),
                meta_map: HashMap::new(),
                nbrs: vec![Vec::new()],
                bubble_x_by_seq: vec![vec![0.0; lm]],
            };
            (cols, PhyloNode::Leaf(0), empty_lookup)
        } else {
            // If n_skel > DENSE_PAIR_MAX_N but m_skel <= DENSE_PAIR_MAX_N, ensure full pairwise posteriors
            if n_skel > DENSE_PAIR_MAX_N && m_skel <= DENSE_PAIR_MAX_N {
                let mut comp_pairs = Vec::with_capacity(m_skel * (m_skel - 1) / 2);
                for a in 0..m_skel {
                    for b in (a + 1)..m_skel {
                        let ga = skel_global_indices[a];
                        let gb = skel_global_indices[b];
                        comp_pairs.push((ga.min(gb), ga.max(gb)));
                    }
                }
                ensure_pairs_evaluated_with_posteriors(
                    &comp_pairs,
                    &mut prepared,
                    &gc,
                    &tables,
                    &mut pair_map,
                    &mut state_map,
                );
            }

            // Deterministic Kruskal MSF over evaluated intra-component edges (O(|E|) when large)
            let mut cand_mst: Vec<(f64, usize, usize)> = Vec::new();
            if m_skel <= 96 {
                for u in 0..m_skel {
                    let ga = skel_global_indices[u];
                    for v in (u + 1)..m_skel {
                        let gb = skel_global_indices[v];
                        if let Some(r) = pair_map.get(&pair_key(ga, gb)) {
                            cand_mst.push((r.mdl_gain_nats, u, v));
                        }
                    }
                }
            } else {
                let mut g2l: HashMap<usize, usize> = HashMap::with_capacity(m_skel);
                for (loc, &g) in skel_global_indices.iter().enumerate() {
                    g2l.insert(g, loc);
                }
                for (&key, r) in &pair_map {
                    let ga = (key >> 32) as usize;
                    let gb = (key & 0xFFFF_FFFF) as usize;
                    if let (Some(&u), Some(&v)) = (g2l.get(&ga), g2l.get(&gb)) {
                        cand_mst.push((r.mdl_gain_nats, u.min(v), u.max(v)));
                    }
                }
            }
            cand_mst.sort_unstable_by(|x, y| {
                y.0.total_cmp(&x.0)
                    .then_with(|| x.1.cmp(&y.1))
                    .then_with(|| x.2.cmp(&y.2))
            });
            let mut mst_uf: Vec<usize> = (0..m_skel).collect();
            let mut mst_gain_sum = 0.0f64;
            let mut mst_edges: Vec<(usize, usize)> = Vec::with_capacity(m_skel - 1);
            for (gain, u, v) in cand_mst {
                let ru = uf_find(&mut mst_uf, u);
                let rv = uf_find(&mut mst_uf, v);
                if ru != rv {
                    mst_uf[rv] = ru;
                    let ga = skel_global_indices[u];
                    let gb = skel_global_indices[v];
                    mst_edges.push((ga.min(gb), ga.max(gb)));
                    if gain > mst_ptr_cost {
                        mst_gain_sum += gain - mst_ptr_cost;
                    }
                    if mst_edges.len() + 1 == m_skel {
                        break;
                    }
                }
            }
            let comp_null: f64 = skel_global_indices.iter().map(|&idx| single_null_nats[idx]).sum();
            provenance_cohort_nats += (comp_null - mst_gain_sum).max(0.0);

            let local_leaf_ids: Vec<usize> = (0..m_skel).collect();
            let (local_dist, local_var, init_tree) = if n_skel > DENSE_PAIR_MAX_N
                && m_skel > DENSE_PAIR_MAX_N
            {
                let (d_init, v_init) =
                    compute_evol_dist_and_var(&skel_global_indices, &pair_map, &seq_lens, 0.0);
                let guide_tree = build_bionj_tree(&local_leaf_ids, &d_init, &v_init, m_skel);
                drop(v_init);

                let mut e_post: Vec<(usize, usize)> = mst_edges;
                let k_post = 15usize;
                let (sb_close, sb_long) = (20usize, 3usize);
                let mut nbr_d: Vec<(f64, usize)> = Vec::with_capacity(m_skel);
                for a in 0..m_skel {
                    nbr_d.clear();
                    for b in 0..m_skel {
                        if a != b {
                            nbr_d.push((d_init[a * m_skel + b], b));
                        }
                    }
                    if nbr_d.len() > k_post {
                        nbr_d.select_nth_unstable_by(k_post, |x, y| {
                            x.0.total_cmp(&y.0).then_with(|| x.1.cmp(&y.1))
                        });
                    }
                    let ga = skel_global_indices[a];
                    for &(_, b) in nbr_d.iter().take(k_post) {
                        let gb = skel_global_indices[b];
                        e_post.push((ga.min(gb), ga.max(gb)));
                    }
                }
                collect_split_bridges(
                    &guide_tree,
                    &skel_global_indices,
                    &d_init,
                    m_skel,
                    &non_ambig_lens,
                    sb_close,
                    sb_long,
                    &mut e_post,
                );
                drop(d_init);
                ensure_pairs_evaluated_with_posteriors(
                    &e_post,
                    &mut prepared,
                    &gc,
                    &tables,
                    &mut pair_map,
                    &mut state_map,
                );

                let (mut final_dist, mut final_var) =
                    compute_evol_dist_and_var(&skel_global_indices, &pair_map, &seq_lens, 0.0);
                let mut final_tree =
                    build_bionj_tree(&local_leaf_ids, &final_dist, &final_var, m_skel);

                let mut verify_pairs = Vec::new();
                collect_split_bridges(
                    &final_tree,
                    &skel_global_indices,
                    &final_dist,
                    m_skel,
                    &non_ambig_lens,
                    14,
                    2,
                    &mut verify_pairs,
                );
                let missing_before = verify_pairs
                    .iter()
                    .filter(|&&(lo, hi)| {
                        !pair_map.get(&pair_key(lo, hi)).map_or(false, is_post_eval)
                    })
                    .count();
                if missing_before > 0 {
                    ensure_pairs_evaluated_with_posteriors(
                        &verify_pairs,
                        &mut prepared,
                        &gc,
                        &tables,
                        &mut pair_map,
                        &mut state_map,
                    );
                    let (d2, v2) = compute_evol_dist_and_var(
                        &skel_global_indices,
                        &pair_map,
                        &seq_lens,
                        0.0,
                    );
                    final_dist = d2;
                    final_var = v2;
                    final_tree = build_bionj_tree(&local_leaf_ids, &final_dist, &final_var, m_skel);

                    let mut verify_pairs_2 = Vec::new();
                    collect_split_bridges(
                        &final_tree,
                        &skel_global_indices,
                        &final_dist,
                        m_skel,
                        &non_ambig_lens,
                        12,
                        1,
                        &mut verify_pairs_2,
                    );
                    ensure_pairs_evaluated_with_posteriors(
                        &verify_pairs_2,
                        &mut prepared,
                        &gc,
                        &tables,
                        &mut pair_map,
                        &mut state_map,
                    );
                }
                (final_dist, final_var, final_tree)
            } else {
                let (d_mat, v_mat) =
                    compute_evol_dist_and_var(&skel_global_indices, &pair_map, &seq_lens, 0.0);
                let t = build_bionj_tree(&local_leaf_ids, &d_mat, &v_mat, m_skel);
                (d_mat, v_mat, t)
            };

            if comp_groups.len() == 1 {
                state_map.clear();
                state_map.shrink_to_fit();
            }
            let (init_tree_nodes, init_root_id, _) =
                init_tree.flatten_and_label(&skel_comp_names);
            let lookup =
                build_consistent_lookup(&skel_global_indices, &pair_map, &local_dist, &seq_lens);
            for &g in &skel_global_indices {
                prepared[g].clear_dp_caches();
            }
            let comp_set: HashSet<usize> = skel_global_indices.iter().copied().collect();
            for (&pk, res) in pair_map.iter_mut() {
                let ga = (pk >> 32) as usize;
                let gb = (pk & 0xFFFF_FFFF) as usize;
                if comp_set.contains(&ga) && comp_set.contains(&gb) {
                    let fine_cnt = res.fine_edges.iter().filter(|e| e.posterior >= 0.5).count();
                    let bub_cnt = res
                        .segments
                        .iter()
                        .filter(|s| s.kind == crate::transducer::SegmentKind::CoarseBubble)
                        .count();
                    pair_summary_counts.insert(pk, (fine_cnt, bub_cnt));
                    res.fine_edges = Vec::new();
                    res.bubble_post_x = Vec::new();
                    res.bubble_post_y = Vec::new();
                    res.fine_cov_x = Vec::new();
                    res.fine_cov_y = Vec::new();
                    res.segments = Vec::new();
                    state_map.remove(&pk);
                }
            }
            let taxon_nu = compute_taxon_nu(&local_dist, m_skel);

            let bionj_prof = assemble_profile_on_tree(
                init_root_id,
                &init_tree_nodes,
                &lookup,
                &skel_comp_seqs,
                &gc,
                &local_dist,
                &taxon_nu,
                m_skel,
            );

            let (final_cols, tree) = if m_skel >= 3 {
                let obj_ctx = build_cohort_objective_ctx(
                    m_skel,
                    &lookup,
                    &local_dist,
                    &taxon_nu,
                    &skel_comp_seqs,
                    &gc,
                );
                let mut best_cols = bionj_prof.cols;
                if m_skel <= 36 {
                    let bionj_score = score_alignment_cohort_mdl(
                        &best_cols,
                        &lookup,
                        &skel_comp_seqs,
                        &gc,
                        &obj_ctx,
                    );
                    let mpo_cols = assemble_profile_mpo(
                        m_skel,
                        &lookup,
                        &skel_comp_seqs,
                        &gc,
                        &local_dist,
                        &taxon_nu,
                    );
                    let mpo_score = score_alignment_cohort_mdl(
                        &mpo_cols,
                        &lookup,
                        &skel_comp_seqs,
                        &gc,
                        &obj_ctx,
                    );
                    if mpo_score > bionj_score + 1e-6 {
                        best_cols = mpo_cols;
                    }
                }
                let mut refined_cols = refine_alignment_topology_free(
                    best_cols,
                    &init_tree,
                    &lookup,
                    &skel_comp_seqs,
                    &gc,
                    &local_dist,
                    &obj_ctx,
                );
                let (core_tree, core_dist) = reestimate_core_bionj_tree(
                    &refined_cols,
                    &lookup,
                    &skel_comp_seqs,
                    &gc,
                    &local_dist,
                    &local_var,
                    &taxon_nu,
                    m_skel,
                );
                drop(local_var);
                drop(local_dist);
                if m_skel <= 48 {
                    let (core_tree_nodes, core_root_id, _) =
                        core_tree.flatten_and_label(&skel_comp_names);
                    let core_prof = assemble_profile_on_tree(
                        core_root_id,
                        &core_tree_nodes,
                        &lookup,
                        &skel_comp_seqs,
                        &gc,
                        &core_dist,
                        &taxon_nu,
                        m_skel,
                    );
                    let ref_score = score_alignment_cohort_mdl(
                        &refined_cols,
                        &lookup,
                        &skel_comp_seqs,
                        &gc,
                        &obj_ctx,
                    );
                    let core_score = score_alignment_cohort_mdl(
                        &core_prof.cols,
                        &lookup,
                        &skel_comp_seqs,
                        &gc,
                        &obj_ctx,
                    );
                    if core_score > ref_score + 1e-6 {
                        refined_cols = refine_alignment_topology_free_passes(
                            core_prof.cols,
                            &core_tree,
                            &lookup,
                            &skel_comp_seqs,
                            &gc,
                            &core_dist,
                            &obj_ctx,
                            1,
                        );
                    }
                }
                (refined_cols, core_tree)
            } else {
                drop(local_var);
                drop(local_dist);
                (bionj_prof.cols, init_tree)
            };

            (final_cols, tree, lookup)
        };

        // Expand / graft collapsed followers (if any) into full component alignment, tree & stats!
        let (final_cols, final_tree, comp_seqs, comp_names, projected) = if n_followers == 0 {
            let (tree_nodes, root_node_id, newick_tree) =
                skel_tree.flatten_and_label(&skel_comp_names);
            let projected = project_alignment_onto_tree(
                &skel_cols,
                &tree_nodes,
                root_node_id,
                &skel_lookup,
                &skel_comp_seqs,
                &gc,
                m_skel,
            );
            drop(skel_lookup);
            (
                skel_cols,
                (tree_nodes, root_node_id, newick_tree),
                skel_comp_seqs,
                skel_comp_names,
                projected,
            )
        } else {
            let num_cols = skel_cols.len();
            let mut med_pos_to_col: Vec<Vec<usize>> = (0..m_skel)
                .map(|s| vec![usize::MAX; skel_comp_seqs[s].len()])
                .collect();
            for (c_idx, col) in skel_cols.iter().enumerate() {
                for &(s, pos) in col {
                    med_pos_to_col[s][pos] = c_idx;
                }
            }

            let mut full_global_indices =
                Vec::with_capacity(m_skel + n_followers);
            let mut global_role: HashMap<usize, (usize, Option<(usize, f64)>)> =
                HashMap::with_capacity(m_skel + n_followers);
            for (s_loc, &sk_idx) in skel_comp_indices.iter().enumerate() {
                let mg = skeleton.med_global[sk_idx];
                full_global_indices.push(mg);
                global_role.insert(mg, (s_loc, None));
                for f in &skeleton.followers_by_med[sk_idx] {
                    full_global_indices.push(f.leaf_global);
                    global_role.insert(f.leaf_global, (s_loc, Some((f.delta, f.cond_nats))));
                }
            }
            full_global_indices.sort_unstable();
            let m_full = full_global_indices.len();

            let mut global_to_full_local = HashMap::with_capacity(m_full);
            for (t_full, &g_idx) in full_global_indices.iter().enumerate() {
                global_to_full_local.insert(g_idx, t_full);
            }
            let skel_to_full_local: Vec<usize> = skel_global_indices
                .iter()
                .map(|g| global_to_full_local[g])
                .collect();

            let full_comp_seqs: Vec<EncodedSeq> = full_global_indices
                .iter()
                .map(|&g| prepared[g].seq.clone())
                .collect();
            if comp_groups.len() == 1 {
                prepared.clear();
                prepared.shrink_to_fit();
            }
            let full_comp_names: Vec<String> =
                full_comp_seqs.iter().map(|s| s.id.clone()).collect();

            let mut full_cols: Vec<Vec<(usize, usize)>> = (0..num_cols).map(|_| Vec::new()).collect();
            let mut full_bubble_x: Vec<Vec<f64>> = Vec::with_capacity(m_full);

            for (t_full, &g_idx) in full_global_indices.iter().enumerate() {
                let (s_loc, fl_opt) = global_role[&g_idx];
                let seq_len = full_comp_seqs[t_full].len();
                match fl_opt {
                    None => {
                        for pos in 0..seq_len {
                            let c_idx = med_pos_to_col[s_loc][pos];
                            full_cols[c_idx].push((t_full, pos));
                        }
                        full_bubble_x.push(skel_lookup.bubble_x_by_seq[s_loc].clone());
                    }
                    Some((delta, cond_nats)) => {
                        provenance_cohort_nats += cond_nats;
                        for pos in 0..seq_len {
                            let c_idx = med_pos_to_col[s_loc][delta + pos];
                            full_cols[c_idx].push((t_full, pos));
                        }
                        full_bubble_x.push(
                            skel_lookup.bubble_x_by_seq[s_loc][delta..delta + seq_len].to_vec(),
                        );
                    }
                }
            }

            // Compute exact column posterior mass + pair count via 1D difference array per medoid cluster
            let mut col_post_sum = vec![0.0f64; num_cols];
            let mut col_pair_cnt = vec![0usize; num_cols];
            let max_med_len = skel_global_indices.iter().map(|&g| seq_lens[g]).max().unwrap_or(0);
            let mut diff = vec![0i32; max_med_len + 1];
            for (s_loc, &sk_idx) in skel_comp_indices.iter().enumerate() {
                let lm = seq_lens[skel_global_indices[s_loc]];
                diff[..=lm].fill(0);
                for f in &skeleton.followers_by_med[sk_idx] {
                    let flen = seq_lens[f.leaf_global];
                    diff[f.delta] += 1;
                    diff[f.delta + flen] -= 1;
                }
                let mut occ_f = 0i32;
                for pos in 0..lm {
                    occ_f += diff[pos];
                    let k = (1 + occ_f) as usize;
                    if k >= 2 {
                        let c_idx = med_pos_to_col[s_loc][pos];
                        let intra = k * (k - 1) / 2;
                        col_post_sum[c_idx] += intra as f64;
                        col_pair_cnt[c_idx] += intra;
                    }
                }
            }
            for &(sa, sb) in &skel_lookup.active_keys {
                let pa = &med_pos_to_col[sa];
                let pb = &med_pos_to_col[sb];
                for &(ia, ib, prob) in skel_lookup.pair_edges_lt(sa, sb) {
                    let ca = pa[ia as usize];
                    if ca != usize::MAX && ca == pb[ib as usize] {
                        col_post_sum[ca] += prob as f64;
                    }
                }
            }
            let mut skel_stamp = vec![0usize; m_skel];
            for (c_idx, col_s) in skel_cols.iter().enumerate() {
                let st = c_idx + 1;
                for &(s, _) in col_s {
                    skel_stamp[s] = st;
                }
                for &(sa, _) in col_s {
                    for &sb in &skel_lookup.nbrs[sa] {
                        if sb > sa && skel_stamp[sb] == st {
                            col_pair_cnt[c_idx] += 1;
                        }
                    }
                }
            }

            drop(skel_lookup);
            drop(med_pos_to_col);
            drop(skel_cols);
            drop(skel_comp_seqs);

            let grafted_tree = graft_followers_into_tree(
                &skel_tree,
                skel_comp_indices,
                &skel_to_full_local,
                &skeleton.followers_by_med,
                &global_to_full_local,
            );
            let (tree_nodes, root_node_id, newick_tree) =
                grafted_tree.flatten_and_label(&full_comp_names);
            let projected = project_alignment_onto_tree_with_extras(
                &full_cols,
                &tree_nodes,
                root_node_id,
                None,
                Some(&full_bubble_x),
                Some((&col_post_sum, &col_pair_cnt)),
                &full_comp_seqs,
                &gc,
                m_full,
            );
            (
                full_cols,
                (tree_nodes, root_node_id, newick_tree),
                full_comp_seqs,
                full_comp_names,
                projected,
            )
        };

        let m = comp_seqs.len();
        let (tree_nodes, root_node_id, newick_tree) = final_tree;
        let num_cols = final_cols.len();
        let tok_bytes = if mode == SeqMode::Codon { 3 } else { 1 };
        let mut aligned_rows: Vec<String> = (0..m)
            .map(|_| String::with_capacity(num_cols * tok_bytes))
            .collect();
        let mut present_pos: Vec<Option<usize>> = vec![None; m];
        for col_entries in &final_cols {
            present_pos.fill(None);
            for &(t_loc, pos) in col_entries {
                present_pos[t_loc] = Some(pos);
            }
            for t_loc in 0..m {
                if let Some(pos) = present_pos[t_loc] {
                    aligned_rows[t_loc].push_str(&comp_seqs[t_loc].token_strings[pos]);
                } else {
                    aligned_rows[t_loc].push_str(gap_token);
                }
            }
        }

        let aligned_fasta: Vec<(String, String)> = comp_names
            .iter()
            .cloned()
            .zip(aligned_rows.into_iter())
            .collect();
        let unaligned_seqs: Vec<(String, String)> = comp_seqs
            .iter()
            .map(|s| (s.id.clone(), s.token_strings.concat()))
            .collect();

        let mut comp_fs: Vec<FrameshiftEvent> = Vec::new();
        for name in &comp_names {
            if let Some(idxs) = seq_to_fs_idx.get(name.as_str()) {
                for &idx in idxs {
                    comp_fs.push(cohort_frameshifts[idx].clone());
                }
            }
        }

        components.push(HomologyComponentReport {
            component_id: comp_id,
            sequence_ids: comp_names,
            newick_tree,
            root_node_id,
            tree_nodes,
            unaligned_seqs,
            num_columns: num_cols,
            num_global_fine_cols: projected.num_global_fine,
            num_clade_local_cols: projected.num_clade_local,
            num_lineage_private_cols: projected.num_private,
            mean_core_posterior: if projected.core_post_n > 0 {
                projected.core_post_acc / (projected.core_post_n as f64)
            } else {
                1.0
            },
            horizon_bubbles: projected.horizon_bubbles,
            frameshifts_repaired: comp_fs,
            aligned_fasta,
            columns: projected.annotated_cols,
        });
    }

    let mut pair_keys_sorted: Vec<u64> = pair_map.keys().copied().collect();
    pair_keys_sorted.sort_unstable();
    let mut pairwise_summary = Vec::with_capacity(pair_keys_sorted.len());
    for pk in pair_keys_sorted {
        let res = &pair_map[&pk];
        let (fine_cnt, bub_cnt) = pair_summary_counts.get(&pk).copied().unwrap_or_else(|| {
            (
                res.fine_edges.iter().filter(|e| e.posterior >= 0.5).count(),
                res.segments
                    .iter()
                    .filter(|s| s.kind == crate::transducer::SegmentKind::CoarseBubble)
                    .count(),
            )
        });
        pairwise_summary.push(PairSummaryRow {
            id_x: res.id_x.clone(),
            id_y: res.id_y.clone(),
            mdl_gain_nats: res.mdl_gain_nats,
            homology_posterior: res.homology_posterior,
            est_t_primary: res.est_t_primary,
            est_t_syn: res.est_t_syn,
            fine_homologous_pairs: fine_cnt,
            coarse_bubble_segments: bub_cnt,
        });
    }

    let net_cohort_gain_nats = raw_null_cohort_nats - provenance_cohort_nats;

    CohortHomologyLedger {
        mode,
        mito_tga_trp: gc.mito_tga_trp,
        num_sequences: n,
        raw_null_cohort_nats,
        provenance_cohort_nats,
        net_cohort_gain_nats,
        num_homology_components: components.len(),
        components,
        pairwise_summary,
    }
}
