use std::collections::{BTreeMap, HashMap, HashSet};
#[cfg(not(target_arch = "wasm32"))]
use std::ffi::{CStr, CString};
#[cfg(not(target_arch = "wasm32"))]
use std::fs;
#[cfg(not(target_arch = "wasm32"))]
use std::os::raw::{c_char, c_int, c_void};
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::alphabet::SeqMode;
use crate::multiseq::{
    AnnotatedColumn, CohortHomologyLedger, ColumnScale, HomologyComponentReport,
    HorizonBubbleReport, TreeNodeRecord,
};
use crate::transducer::FrameshiftEvent;

// -----------------------------------------------------------------------------
// Native macOS libsqlite3 FFI (zero external Cargo dependencies)
// -----------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
mod sqlite_ffi {
    use super::*;

#[repr(C)]
pub(super) struct sqlite3 {
    _private: [u8; 0],
}

#[repr(C)]
pub(super) struct sqlite3_stmt {
    _private: [u8; 0],
}

pub(super) const SQLITE_OK: c_int = 0;
pub(super) const SQLITE_ROW: c_int = 100;
pub(super) const SQLITE_DONE: c_int = 101;
pub(super) const SQLITE_NULL: c_int = 5;
pub(super) const SQLITE_OPEN_READONLY: c_int = 0x0000_0001;
pub(super) const SQLITE_OPEN_READWRITE: c_int = 0x0000_0002;
pub(super) const SQLITE_OPEN_CREATE: c_int = 0x0000_0004;

#[link(name = "sqlite3")]
extern "C" {
    fn sqlite3_open_v2(
        filename: *const c_char,
        pp_db: *mut *mut sqlite3,
        flags: c_int,
        z_vfs: *const c_char,
    ) -> c_int;
    fn sqlite3_close_v2(db: *mut sqlite3) -> c_int;
    fn sqlite3_exec(
        db: *mut sqlite3,
        sql: *const c_char,
        callback: Option<unsafe extern "C" fn(*mut c_void, c_int, *mut *mut c_char, *mut *mut c_char) -> c_int>,
        arg: *mut c_void,
        errmsg: *mut *mut c_char,
    ) -> c_int;
    fn sqlite3_free(ptr: *mut c_void);
    fn sqlite3_errmsg(db: *mut sqlite3) -> *const c_char;
    fn sqlite3_prepare_v2(
        db: *mut sqlite3,
        z_sql: *const c_char,
        n_byte: c_int,
        pp_stmt: *mut *mut sqlite3_stmt,
        pz_tail: *mut *const c_char,
    ) -> c_int;
    fn sqlite3_step(stmt: *mut sqlite3_stmt) -> c_int;
    fn sqlite3_reset(stmt: *mut sqlite3_stmt) -> c_int;
    fn sqlite3_clear_bindings(stmt: *mut sqlite3_stmt) -> c_int;
    fn sqlite3_finalize(stmt: *mut sqlite3_stmt) -> c_int;
    fn sqlite3_bind_int64(stmt: *mut sqlite3_stmt, idx: c_int, val: i64) -> c_int;
    fn sqlite3_bind_double(stmt: *mut sqlite3_stmt, idx: c_int, val: f64) -> c_int;
    fn sqlite3_bind_text(
        stmt: *mut sqlite3_stmt,
        idx: c_int,
        val: *const c_char,
        n_bytes: c_int,
        destructor: isize,
    ) -> c_int;
    fn sqlite3_bind_null(stmt: *mut sqlite3_stmt, idx: c_int) -> c_int;
    fn sqlite3_column_int64(stmt: *mut sqlite3_stmt, col: c_int) -> i64;
    fn sqlite3_column_double(stmt: *mut sqlite3_stmt, col: c_int) -> f64;
    fn sqlite3_column_text(stmt: *mut sqlite3_stmt, col: c_int) -> *const u8;
    fn sqlite3_column_bytes(stmt: *mut sqlite3_stmt, col: c_int) -> c_int;
    fn sqlite3_column_type(stmt: *mut sqlite3_stmt, col: c_int) -> c_int;
}

const SQLITE_TRANSIENT: isize = -1;

pub(super) struct SqliteDb {
    raw: *mut sqlite3,
}

impl Drop for SqliteDb {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                sqlite3_close_v2(self.raw);
            }
            self.raw = std::ptr::null_mut();
        }
    }
}

impl SqliteDb {
    pub(super) fn open(path: &Path, flags: c_int) -> Result<Self, String> {
        let p_str = path
            .to_str()
            .ok_or_else(|| format!("Non-UTF8 path: {}", path.display()))?;
        let c_path = CString::new(p_str).map_err(|e| format!("Invalid path: {e}"))?;
        let mut raw: *mut sqlite3 = std::ptr::null_mut();
        let rc = unsafe { sqlite3_open_v2(c_path.as_ptr(), &mut raw, flags, std::ptr::null()) };
        if rc != SQLITE_OK {
            let msg = if !raw.is_null() {
                let s = unsafe { CStr::from_ptr(sqlite3_errmsg(raw)) }
                    .to_string_lossy()
                    .into_owned();
                unsafe { sqlite3_close_v2(raw) };
                s
            } else {
                format!("sqlite3_open_v2 failed with rc={rc}")
            };
            return Err(msg);
        }
        Ok(Self { raw })
    }

    fn err_msg(&self) -> String {
        unsafe { CStr::from_ptr(sqlite3_errmsg(self.raw)) }
            .to_string_lossy()
            .into_owned()
    }

    pub(super) fn exec(&self, sql: &str) -> Result<(), String> {
        let c_sql = CString::new(sql).map_err(|e| format!("Invalid SQL: {e}"))?;
        let mut errmsg: *mut c_char = std::ptr::null_mut();
        let rc = unsafe {
            sqlite3_exec(
                self.raw,
                c_sql.as_ptr(),
                None,
                std::ptr::null_mut(),
                &mut errmsg,
            )
        };
        if rc != SQLITE_OK {
            let msg = if !errmsg.is_null() {
                let s = unsafe { CStr::from_ptr(errmsg) }
                    .to_string_lossy()
                    .into_owned();
                unsafe { sqlite3_free(errmsg as *mut c_void) };
                s
            } else {
                self.err_msg()
            };
            return Err(format!("SQLite exec error ({rc}): {msg}"));
        }
        Ok(())
    }

    pub(super) fn prepare<'a>(&'a self, sql: &str) -> Result<SqliteStmt<'a>, String> {
        let mut raw_stmt: *mut sqlite3_stmt = std::ptr::null_mut();
        let rc = unsafe {
            sqlite3_prepare_v2(
                self.raw,
                sql.as_ptr() as *const c_char,
                sql.len() as c_int,
                &mut raw_stmt,
                std::ptr::null_mut(),
            )
        };
        if rc != SQLITE_OK || raw_stmt.is_null() {
            return Err(format!("SQLite prepare error ({rc}): {}", self.err_msg()));
        }
        Ok(SqliteStmt {
            raw: raw_stmt,
            db: self,
        })
    }
}

pub(super) struct SqliteStmt<'a> {
    raw: *mut sqlite3_stmt,
    db: &'a SqliteDb,
}

impl<'a> Drop for SqliteStmt<'a> {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                sqlite3_finalize(self.raw);
            }
            self.raw = std::ptr::null_mut();
        }
    }
}

impl<'a> SqliteStmt<'a> {
    pub(super) fn reset(&self) -> Result<(), String> {
        let rc1 = unsafe { sqlite3_reset(self.raw) };
        let rc2 = unsafe { sqlite3_clear_bindings(self.raw) };
        if rc1 != SQLITE_OK || rc2 != SQLITE_OK {
            return Err(format!("SQLite reset error: {}", self.db.err_msg()));
        }
        Ok(())
    }

    pub(super) fn bind_i64(&self, idx: c_int, val: i64) -> Result<(), String> {
        let rc = unsafe { sqlite3_bind_int64(self.raw, idx, val) };
        if rc != SQLITE_OK {
            return Err(format!("SQLite bind_int64({idx}): {}", self.db.err_msg()));
        }
        Ok(())
    }

    pub(super) fn bind_f64(&self, idx: c_int, val: f64) -> Result<(), String> {
        let rc = unsafe { sqlite3_bind_double(self.raw, idx, val) };
        if rc != SQLITE_OK {
            return Err(format!("SQLite bind_double({idx}): {}", self.db.err_msg()));
        }
        Ok(())
    }

    pub(super) fn bind_text(&self, idx: c_int, val: &str) -> Result<(), String> {
        let ptr = if val.is_empty() {
            b"\0".as_ptr() as *const c_char
        } else {
            val.as_ptr() as *const c_char
        };
        let rc = unsafe {
            sqlite3_bind_text(self.raw, idx, ptr, val.len() as c_int, SQLITE_TRANSIENT)
        };
        if rc != SQLITE_OK {
            return Err(format!("SQLite bind_text({idx}): {}", self.db.err_msg()));
        }
        Ok(())
    }

    pub(super) fn bind_opt_i64(&self, idx: c_int, val: Option<usize>) -> Result<(), String> {
        match val {
            Some(v) => self.bind_i64(idx, v as i64),
            None => {
                let rc = unsafe { sqlite3_bind_null(self.raw, idx) };
                if rc != SQLITE_OK {
                    return Err(format!("SQLite bind_null({idx}): {}", self.db.err_msg()));
                }
                Ok(())
            }
        }
    }

    pub(super) fn bind_opt_text(&self, idx: c_int, val: Option<&str>) -> Result<(), String> {
        match val {
            Some(v) => self.bind_text(idx, v),
            None => {
                let rc = unsafe { sqlite3_bind_null(self.raw, idx) };
                if rc != SQLITE_OK {
                    return Err(format!("SQLite bind_null({idx}): {}", self.db.err_msg()));
                }
                Ok(())
            }
        }
    }

    pub(super) fn step_done(&self) -> Result<(), String> {
        let rc = unsafe { sqlite3_step(self.raw) };
        if rc != SQLITE_DONE {
            return Err(format!("SQLite step_done error ({rc}): {}", self.db.err_msg()));
        }
        Ok(())
    }

    pub(super) fn step_row(&self) -> Result<bool, String> {
        let rc = unsafe { sqlite3_step(self.raw) };
        match rc {
            SQLITE_ROW => Ok(true),
            SQLITE_DONE => Ok(false),
            _ => Err(format!("SQLite step_row error ({rc}): {}", self.db.err_msg())),
        }
    }

    pub(super) fn col_i64(&self, col: c_int) -> i64 {
        unsafe { sqlite3_column_int64(self.raw, col) }
    }

    pub(super) fn col_usize(&self, col: c_int) -> usize {
        unsafe { sqlite3_column_int64(self.raw, col) }.max(0) as usize
    }

    pub(super) fn col_opt_usize(&self, col: c_int) -> Option<usize> {
        if unsafe { sqlite3_column_type(self.raw, col) } == SQLITE_NULL {
            None
        } else {
            Some(self.col_usize(col))
        }
    }

    pub(super) fn col_f64(&self, col: c_int) -> f64 {
        unsafe { sqlite3_column_double(self.raw, col) }
    }

    pub(super) fn col_text(&self, col: c_int) -> String {
        if unsafe { sqlite3_column_type(self.raw, col) } == SQLITE_NULL {
            return String::new();
        }
        let ptr = unsafe { sqlite3_column_text(self.raw, col) };
        let len = unsafe { sqlite3_column_bytes(self.raw, col) }.max(0) as usize;
        if ptr.is_null() || len == 0 {
            return String::new();
        }
        let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
        String::from_utf8_lossy(slice).into_owned()
    }

    pub(super) fn col_opt_text(&self, col: c_int) -> Option<String> {
        if unsafe { sqlite3_column_type(self.raw, col) } == SQLITE_NULL {
            None
        } else {
            Some(self.col_text(col))
        }
    }
}
} // mod sqlite_ffi

#[cfg(not(target_arch = "wasm32"))]
use sqlite_ffi::*;

// -----------------------------------------------------------------------------
// Hierarchical Alignment Archive Model
// -----------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchSeq {
    pub seq_id: usize,
    pub name: String,
    pub comp_id: usize,
    pub local_seq_idx: usize,
    pub orig_offset: usize,
    pub seq_len: usize,
    pub raw_seq: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockSegment {
    pub local_seq_idx: usize,
    pub seq_id: usize,
    pub start_pos: usize,
    pub end_pos: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollinearBlock {
    pub block_id: usize,
    pub col_start: usize,
    pub col_len: usize,
    pub scale: ColumnScale,
    pub owner_node_id: usize,
    pub segments: Vec<BlockSegment>,
    pub fine_posteriors: Vec<f64>,
    pub bubble_posteriors: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchComponent {
    pub comp_id: usize,
    pub root_node_id: usize,
    pub num_columns: usize,
    pub num_global_fine_cols: usize,
    pub num_clade_local_cols: usize,
    pub num_lineage_private_cols: usize,
    pub mean_core_posterior: f64,
    pub seq_ids: Vec<usize>,
    pub newick_tree: String,
    pub tree_nodes: Vec<TreeNodeRecord>,
    pub blocks: Vec<CollinearBlock>,
    pub horizon_bubbles: Vec<HorizonBubbleReport>,
    pub frameshifts_repaired: Vec<FrameshiftEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalArchive {
    pub format_version: u32,
    pub mode: SeqMode,
    pub mito_tga_trp: bool,
    pub raw_null_cohort_nats: f64,
    pub provenance_cohort_nats: f64,
    pub net_cohort_gain_nats: f64,
    pub seqs: Vec<ArchSeq>,
    pub components: Vec<ArchComponent>,
}

#[derive(Debug, Clone, Default)]
pub struct SubsetSpec {
    pub component_id: Option<usize>,
    pub clade: Option<String>,
    pub taxa: Option<Vec<String>>,
    /// `(seq_name, start1, end1)` in 1-based inclusive coordinates.
    pub region: Option<(String, usize, usize)>,
    pub global_core_only: bool,
}

#[inline]
fn token_step(mode: SeqMode) -> usize {
    if mode == SeqMode::Codon {
        3
    } else {
        1
    }
}

#[inline]
fn mode_to_str(mode: SeqMode) -> &'static str {
    match mode {
        SeqMode::Protein => "protein",
        SeqMode::Dna => "dna",
        SeqMode::Codon => "codon",
    }
}

fn str_to_mode(s: &str) -> Result<SeqMode, String> {
    match s.to_ascii_lowercase().as_str() {
        "protein" | "aa" => Ok(SeqMode::Protein),
        "dna" | "nt" => Ok(SeqMode::Dna),
        "codon" => Ok(SeqMode::Codon),
        other => Err(format!("Unknown sequence mode in archive: '{other}'")),
    }
}

#[inline]
fn scale_to_str(scale: ColumnScale) -> &'static str {
    match scale {
        ColumnScale::GlobalFine => "GlobalFine",
        ColumnScale::CladeLocalFine => "CladeLocalFine",
        ColumnScale::LineagePrivate => "LineagePrivate",
    }
}

fn str_to_scale(s: &str) -> Result<ColumnScale, String> {
    match s {
        "GlobalFine" => Ok(ColumnScale::GlobalFine),
        "CladeLocalFine" => Ok(ColumnScale::CladeLocalFine),
        "LineagePrivate" => Ok(ColumnScale::LineagePrivate),
        other => Err(format!("Unknown ColumnScale: '{other}'")),
    }
}

impl HierarchicalArchive {
    pub fn from_ledger(ledger: &CohortHomologyLedger) -> Self {
        let step = token_step(ledger.mode);
        let mut seqs = Vec::new();
        let mut components = Vec::with_capacity(ledger.components.len());

        for comp in &ledger.components {
            let m = comp.sequence_ids.len();
            let mut comp_seq_ids = Vec::with_capacity(m);
            for local_idx in 0..m {
                let global_id = seqs.len();
                let (ref name, ref raw_seq) = comp.unaligned_seqs[local_idx];
                comp_seq_ids.push(global_id);
                seqs.push(ArchSeq {
                    seq_id: global_id,
                    name: name.clone(),
                    comp_id: comp.component_id,
                    local_seq_idx: local_idx,
                    orig_offset: 0,
                    seq_len: raw_seq.len() / step,
                    raw_seq: raw_seq.clone(),
                });
            }

            let aln_bytes: Vec<&[u8]> = comp
                .aligned_fasta
                .iter()
                .map(|(_, s)| s.as_bytes())
                .collect();
            let mut cursors = vec![0usize; m];
            let mut blocks: Vec<CollinearBlock> = Vec::new();

            for c_idx in 0..comp.num_columns {
                let meta_c = &comp.columns[c_idx];
                let mut present_pos: Vec<(usize, usize)> = Vec::new();
                for t in 0..m {
                    let sl = &aln_bytes[t][c_idx * step..(c_idx + 1) * step];
                    let is_gap = sl.iter().all(|&b| b == b'-' || b == b'.' || b == b'?');
                    if !is_gap {
                        present_pos.push((t, cursors[t]));
                        cursors[t] += 1;
                    }
                }

                let can_coalesce = if let Some(last) = blocks.last() {
                    last.scale == meta_c.scale
                        && last.owner_node_id == meta_c.owner_node_id
                        && last.segments.len() == present_pos.len()
                        && last
                            .segments
                            .iter()
                            .zip(present_pos.iter())
                            .all(|(seg, &(t, pos))| {
                                seg.local_seq_idx == t && seg.end_pos == pos
                            })
                } else {
                    false
                };

                if can_coalesce {
                    let last = blocks.last_mut().unwrap();
                    last.col_len += 1;
                    for seg in &mut last.segments {
                        seg.end_pos += 1;
                    }
                    last.fine_posteriors.push(meta_c.mean_fine_posterior);
                    last.bubble_posteriors.push(meta_c.coarse_bubble_posterior);
                } else {
                    let block_id = blocks.len();
                    let segments = present_pos
                        .into_iter()
                        .map(|(t, pos)| BlockSegment {
                            local_seq_idx: t,
                            seq_id: comp_seq_ids[t],
                            start_pos: pos,
                            end_pos: pos + 1,
                        })
                        .collect();
                    blocks.push(CollinearBlock {
                        block_id,
                        col_start: c_idx,
                        col_len: 1,
                        scale: meta_c.scale,
                        owner_node_id: meta_c.owner_node_id,
                        segments,
                        fine_posteriors: vec![meta_c.mean_fine_posterior],
                        bubble_posteriors: vec![meta_c.coarse_bubble_posterior],
                    });
                }
            }

            components.push(ArchComponent {
                comp_id: comp.component_id,
                root_node_id: comp.root_node_id,
                num_columns: comp.num_columns,
                num_global_fine_cols: comp.num_global_fine_cols,
                num_clade_local_cols: comp.num_clade_local_cols,
                num_lineage_private_cols: comp.num_lineage_private_cols,
                mean_core_posterior: comp.mean_core_posterior,
                seq_ids: comp_seq_ids,
                newick_tree: comp.newick_tree.clone(),
                tree_nodes: comp.tree_nodes.clone(),
                blocks,
                horizon_bubbles: comp.horizon_bubbles.clone(),
                frameshifts_repaired: comp.frameshifts_repaired.clone(),
            });
        }

        HierarchicalArchive {
            format_version: 1,
            mode: ledger.mode,
            mito_tga_trp: ledger.mito_tga_trp,
            raw_null_cohort_nats: ledger.raw_null_cohort_nats,
            provenance_cohort_nats: ledger.provenance_cohort_nats,
            net_cohort_gain_nats: ledger.net_cohort_gain_nats,
            seqs,
            components,
        }
    }

    pub fn to_ledger(&self) -> CohortHomologyLedger {
        let seqs_by_id: HashMap<usize, &ArchSeq> =
            self.seqs.iter().map(|s| (s.seq_id, s)).collect();
        let mut components = Vec::with_capacity(self.components.len());
        for comp in &self.components {
            let mut sequence_ids = Vec::with_capacity(comp.seq_ids.len());
            let mut unaligned_seqs = Vec::with_capacity(comp.seq_ids.len());
            for &sid in &comp.seq_ids {
                if let Some(s) = seqs_by_id.get(&sid) {
                    sequence_ids.push(s.name.clone());
                    unaligned_seqs.push((s.name.clone(), s.raw_seq.clone()));
                }
            }
            let aligned_fasta = render_component_matrix(comp, &self.seqs, self.mode, false);
            let mut columns = Vec::with_capacity(comp.num_columns);
            let mut sorted_blocks: Vec<&CollinearBlock> = comp.blocks.iter().collect();
            sorted_blocks.sort_by_key(|b| b.col_start);
            for blk in sorted_blocks {
                let np = blk.segments.len();
                for offset in 0..blk.col_len {
                    let c_idx = blk.col_start + offset;
                    let fine_p = blk.fine_posteriors.get(offset).copied().unwrap_or(1.0);
                    let bub_p = blk.bubble_posteriors.get(offset).copied().unwrap_or(0.0);
                    columns.push(AnnotatedColumn {
                        col_index: c_idx,
                        scale: blk.scale,
                        owner_node_id: blk.owner_node_id,
                        num_present: np,
                        mean_fine_posterior: fine_p,
                        coarse_bubble_posterior: bub_p,
                    });
                }
            }
            components.push(HomologyComponentReport {
                component_id: comp.comp_id,
                sequence_ids,
                newick_tree: comp.newick_tree.clone(),
                root_node_id: comp.root_node_id,
                tree_nodes: comp.tree_nodes.clone(),
                unaligned_seqs,
                num_columns: comp.num_columns,
                num_global_fine_cols: comp.num_global_fine_cols,
                num_clade_local_cols: comp.num_clade_local_cols,
                num_lineage_private_cols: comp.num_lineage_private_cols,
                mean_core_posterior: comp.mean_core_posterior,
                horizon_bubbles: comp.horizon_bubbles.clone(),
                frameshifts_repaired: comp.frameshifts_repaired.clone(),
                aligned_fasta,
                columns,
            });
        }
        CohortHomologyLedger {
            mode: self.mode,
            mito_tga_trp: self.mito_tga_trp,
            num_sequences: self.seqs.len(),
            raw_null_cohort_nats: self.raw_null_cohort_nats,
            provenance_cohort_nats: self.provenance_cohort_nats,
            net_cohort_gain_nats: self.net_cohort_gain_nats,
            num_homology_components: components.len(),
            components,
            pairwise_summary: Vec::new(),
        }
    }
}

/// Compute leaf descendant masks `desc[node_id][local_seq_idx]` for a component's tree.
fn compute_leaf_descendants(tree_nodes: &[TreeNodeRecord], m: usize) -> Vec<Vec<bool>> {
    let mut node_by_id: HashMap<usize, usize> = HashMap::with_capacity(tree_nodes.len());
    let mut max_id = 0usize;
    for (idx, rec) in tree_nodes.iter().enumerate() {
        node_by_id.insert(rec.node_id, idx);
        max_id = max_id.max(rec.node_id);
    }
    let n_slots = max_id + 1;
    let mut desc = vec![vec![false; m]; n_slots];
    let mut done = vec![false; n_slots];

    fn fill(
        node_id: usize,
        node_by_id: &HashMap<usize, usize>,
        tree_nodes: &[TreeNodeRecord],
        desc: &mut [Vec<bool>],
        done: &mut [bool],
    ) {
        if done[node_id] {
            return;
        }
        let rec = &tree_nodes[node_by_id[&node_id]];
        if rec.is_leaf {
            if let Some(loc) = rec.local_seq_idx {
                if loc < desc[node_id].len() {
                    desc[node_id][loc] = true;
                }
            }
        } else {
            if let Some(lid) = rec.left_node_id {
                fill(lid, node_by_id, tree_nodes, desc, done);
                let l_row = desc[lid].clone();
                for (dst, src) in desc[node_id].iter_mut().zip(l_row.iter()) {
                    *dst |= *src;
                }
            }
            if let Some(rid) = rec.right_node_id {
                fill(rid, node_by_id, tree_nodes, desc, done);
                let r_row = desc[rid].clone();
                for (dst, src) in desc[node_id].iter_mut().zip(r_row.iter()) {
                    *dst |= *src;
                }
            }
        }
        done[node_id] = true;
    }

    for rec in tree_nodes {
        fill(rec.node_id, &node_by_id, tree_nodes, &mut desc, &mut done);
    }
    desc
}

/// Expand RLE collinear blocks back into per-sequence aligned strings.
/// When `distinguish_horizon` is true, absent descendants of `block.owner_node_id` emit `-`/`---`
/// (genuine clade-internal indels), while absent non-descendants emit `.`/`...` (across a Coarse Horizon).
pub fn render_component_matrix(
    comp: &ArchComponent,
    seqs: &[ArchSeq],
    mode: SeqMode,
    distinguish_horizon: bool,
) -> Vec<(String, String)> {
    let m = comp.seq_ids.len();
    let step = token_step(mode);
    let gap_tok = if mode == SeqMode::Codon { "---" } else { "-" };
    let miss_tok = if mode == SeqMode::Codon { "..." } else { "." };

    let desc = if distinguish_horizon {
        compute_leaf_descendants(&comp.tree_nodes, m)
    } else {
        Vec::new()
    };

    let mut rows: Vec<String> = (0..m)
        .map(|_| String::with_capacity(comp.num_columns * step))
        .collect();
    let comp_set: HashSet<usize> = comp.seq_ids.iter().copied().collect();
    let mut seq_map: HashMap<usize, &ArchSeq> = HashMap::with_capacity(m);
    for s in seqs {
        if comp_set.contains(&s.seq_id) {
            seq_map.insert(s.seq_id, s);
        }
    }

    let mut present_start: Vec<Option<(usize, usize)>> = vec![None; m];
    for block in &comp.blocks {
        present_start.fill(None);
        for seg in &block.segments {
            present_start[seg.local_seq_idx] = Some((seg.seq_id, seg.start_pos));
        }
        for off in 0..block.col_len {
            for t in 0..m {
                if let Some((sid, st)) = present_start[t] {
                    let s = &seq_map[&sid].raw_seq;
                    let a = (st + off) * step;
                    rows[t].push_str(&s[a..a + step]);
                } else if distinguish_horizon
                    && block.owner_node_id < desc.len()
                    && !desc[block.owner_node_id][t]
                {
                    rows[t].push_str(miss_tok);
                } else {
                    rows[t].push_str(gap_tok);
                }
            }
        }
    }

    comp.seq_ids
        .iter()
        .enumerate()
        .map(|(t, &sid)| (seq_map[&sid].name.clone(), std::mem::take(&mut rows[t])))
        .collect()
}

pub fn write_flat_fasta_str(arch: &HierarchicalArchive, distinguish_horizon: bool) -> String {
    let mut buf = String::new();
    let multi = arch.components.len() > 1;
    for comp in &arch.components {
        for (name, aln) in render_component_matrix(comp, &arch.seqs, arch.mode, distinguish_horizon) {
            if multi {
                buf.push_str(&format!(">{} component={}\n{}\n", name, comp.comp_id, aln));
            } else {
                buf.push_str(&format!(">{}\n{}\n", name, aln));
            }
        }
    }
    buf
}

#[cfg(not(target_arch = "wasm32"))]
pub fn write_flat_fasta_file(path: &Path, arch: &HierarchicalArchive) -> Result<(), String> {
    let buf = write_flat_fasta_str(arch, false);
    fs::write(path, buf).map_err(|e| format!("Failed to write FASTA '{}': {e}", path.display()))
}

// -----------------------------------------------------------------------------
// Tier 1: Extended NEXUS (.hnex)
// -----------------------------------------------------------------------------

fn nexus_safe_label(s: &str) -> String {
    let needs_quotes = s.is_empty()
        || s.bytes().any(|b| {
            b.is_ascii_whitespace()
                || matches!(
                    b,
                    b'(' | b')' | b'[' | b']' | b'{' | b'}' | b'/' | b'\\' | b',' | b';' | b':' | b'=' | b'*' | b'\'' | b'"' | b'<' | b'>' | b'^' | b'+' | b'-'
                )
        });
    if needs_quotes {
        format!("'{}'", s.replace('\'', "''"))
    } else {
        s.to_string()
    }
}

/// Escape attribute string so NO `;`, `[`, `]`, `"`, `%`, CR, or LF appears unencoded inside `BEGIN HOMOLOGY;`.
fn hnex_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'%' => out.push_str("%25"),
            b'"' => out.push_str("%22"),
            b';' => out.push_str("%3B"),
            b'[' => out.push_str("%5B"),
            b']' => out.push_str("%5D"),
            b'\n' => out.push_str("%0A"),
            b'\r' => out.push_str("%0D"),
            _ => out.push(b as char),
        }
    }
    out
}

fn hnex_unescape(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(h), Some(l)) = (hex_nibble(bytes[i + 1]), hex_nibble(bytes[i + 2])) {
                out.push((h << 4) | l);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[inline]
fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

fn fmt_f64_csv(vals: &[f64]) -> String {
    let mut out = String::new();
    for (i, v) in vals.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&v.to_string());
    }
    out
}

fn parse_f64_csv(s: &str) -> Result<Vec<f64>, String> {
    if s.is_empty() {
        return Ok(Vec::new());
    }
    s.split(',')
        .map(|tok| tok.parse::<f64>().map_err(|e| format!("Invalid f64 '{tok}': {e}")))
        .collect()
}

fn fmt_intervals(ivs: &[(usize, usize, usize)]) -> String {
    let mut out = String::new();
    for (i, &(t, st, len)) in ivs.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!("{t}:{st}:{len}"));
    }
    out
}

fn parse_intervals(s: &str) -> Result<Vec<(usize, usize, usize)>, String> {
    if s.is_empty() {
        return Ok(Vec::new());
    }
    s.split(',')
        .map(|tok| {
            let mut it = tok.split(':');
            let a = it
                .next()
                .ok_or_else(|| format!("Malformed interval '{tok}'"))?
                .parse::<usize>()
                .map_err(|e| format!("Malformed interval '{tok}': {e}"))?;
            let b = it
                .next()
                .ok_or_else(|| format!("Malformed interval '{tok}'"))?
                .parse::<usize>()
                .map_err(|e| format!("Malformed interval '{tok}': {e}"))?;
            let c = it
                .next()
                .ok_or_else(|| format!("Malformed interval '{tok}'"))?
                .parse::<usize>()
                .map_err(|e| format!("Malformed interval '{tok}': {e}"))?;
            Ok((a, b, c))
        })
        .collect()
}

fn fmt_segments(segs: &[BlockSegment]) -> String {
    let mut out = String::new();
    for (i, seg) in segs.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{}:{}:{}:{}",
            seg.local_seq_idx, seg.seq_id, seg.start_pos, seg.end_pos
        ));
    }
    out
}

fn parse_segments(s: &str) -> Result<Vec<BlockSegment>, String> {
    if s.is_empty() {
        return Ok(Vec::new());
    }
    s.split(',')
        .map(|tok| {
            let parts: Vec<&str> = tok.split(':').collect();
            if parts.len() != 4 {
                return Err(format!("Malformed segment '{tok}'"));
            }
            Ok(BlockSegment {
                local_seq_idx: parts[0].parse().map_err(|e| format!("{e}"))?,
                seq_id: parts[1].parse().map_err(|e| format!("{e}"))?,
                start_pos: parts[2].parse().map_err(|e| format!("{e}"))?,
                end_pos: parts[3].parse().map_err(|e| format!("{e}"))?,
            })
        })
        .collect()
}

pub fn write_hnex_str(arch: &HierarchicalArchive) -> String {
    let mut out = String::new();
    out.push_str("#NEXUS\n");
    out.push_str("[Generated by SQUINT (Scale-Quotiented Unaligned-Interval Network Transducer) - Two-Tier Multi-Scale Homology Archive]\n\n");

    let datatype = match arch.mode {
        SeqMode::Protein => "PROTEIN",
        SeqMode::Dna | SeqMode::Codon => "DNA",
    };

    if let Some(comp0) = arch.components.first() {
        let step = token_step(arch.mode);
        let ntax = comp0.seq_ids.len();
        let nchar = comp0.num_columns * step;
        if ntax > 0 && nchar > 0 {
            out.push_str("BEGIN DATA;\n");
            out.push_str(&format!("    DIMENSIONS NTAX={} NCHAR={};\n", ntax, nchar));
            out.push_str(&format!(
                "    FORMAT DATATYPE={} GAP=- MISSING=.;\n",
                datatype
            ));
            out.push_str("    MATRIX\n");
            for (name, row) in render_component_matrix(comp0, &arch.seqs, arch.mode, true) {
                out.push_str(&format!("    {} {}\n", nexus_safe_label(&name), row));
            }
            out.push_str("    ;\n");
            out.push_str("END;\n\n");
        }
    }

    if !arch.components.is_empty() {
        out.push_str("BEGIN TREES;\n");
        for comp in &arch.components {
            let tree_body = comp.newick_tree.trim_end_matches(';');
            out.push_str(&format!(
                "    TREE comp_{} = {};\n",
                comp.comp_id, tree_body
            ));
        }
        out.push_str("END;\n\n");
    }

    out.push_str("BEGIN HOMOLOGY;\n");
    out.push_str(&format!(
        "    META version={} mode={} mito_tga_trp={} raw_null_nats={} prov_nats={} net_gain_nats={} nseqs={} ncomps={};\n",
        arch.format_version,
        mode_to_str(arch.mode),
        arch.mito_tga_trp,
        arch.raw_null_cohort_nats,
        arch.provenance_cohort_nats,
        arch.net_cohort_gain_nats,
        arch.seqs.len(),
        arch.components.len()
    ));

    for seq in &arch.seqs {
        out.push_str(&format!(
            "    SEQ id={} name=\"{}\" comp={} local={} orig_offset={} len={} raw=\"{}\";\n",
            seq.seq_id,
            hnex_escape(&seq.name),
            seq.comp_id,
            seq.local_seq_idx,
            seq.orig_offset,
            seq.seq_len,
            hnex_escape(&seq.raw_seq),
        ));
    }

    for comp in &arch.components {
        out.push_str(&format!(
            "    COMP id={} root={} n_cols={} n_global={} n_clade={} n_priv={} mean_p_core={} seq_ids=\"{}\" newick=\"{}\";\n",
            comp.comp_id,
            comp.root_node_id,
            comp.num_columns,
            comp.num_global_fine_cols,
            comp.num_clade_local_cols,
            comp.num_lineage_private_cols,
            comp.mean_core_posterior,
            comp.seq_ids
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(","),
            hnex_escape(&comp.newick_tree),
        ));

        for node in &comp.tree_nodes {
            let l_str = node
                .left_node_id
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".to_string());
            let r_str = node
                .right_node_id
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".to_string());
            let s_str = node
                .local_seq_idx
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".to_string());
            out.push_str(&format!(
                "    NODE comp={} id={} leaf={} label=\"{}\" left={} right={} len_l={} len_r={} seq={};\n",
                comp.comp_id,
                node.node_id,
                node.is_leaf,
                hnex_escape(&node.label),
                l_str,
                r_str,
                node.len_left,
                node.len_right,
                s_str,
            ));
        }

        for blk in &comp.blocks {
            out.push_str(&format!(
                "    BLOCK comp={} id={} col_start={} col_len={} scale={} owner={} segs=\"{}\" p_fine=\"{}\" p_bub=\"{}\";\n",
                comp.comp_id,
                blk.block_id,
                blk.col_start,
                blk.col_len,
                scale_to_str(blk.scale),
                blk.owner_node_id,
                fmt_segments(&blk.segments),
                fmt_f64_csv(&blk.fine_posteriors),
                fmt_f64_csv(&blk.bubble_posteriors),
            ));
        }

        for hb in &comp.horizon_bubbles {
            let cls_str = hb.dominant_biophys_class.as_deref().unwrap_or("-");
            out.push_str(&format!(
                "    BUBBLE comp={} step={} split={} left_node={} right_node={} l_size={} r_size={} l_span={} r_span={} l_iv=\"{}\" r_iv=\"{}\" class=\"{}\" gain={} d_hydro={} d_chg={};\n",
                comp.comp_id,
                hb.tree_merge_step,
                hb.split_node_id,
                hb.left_node_id,
                hb.right_node_id,
                hb.left_clade_size,
                hb.right_clade_size,
                hb.left_col_span,
                hb.right_col_span,
                fmt_intervals(&hb.left_intervals),
                fmt_intervals(&hb.right_intervals),
                hnex_escape(cls_str),
                hb.macro_biophys_gain_nats,
                hb.delta_hydropathy,
                hb.delta_charge,
            ));
        }

        for fs_ev in &comp.frameshifts_repaired {
            out.push_str(&format!(
                "    FS comp={} seq=\"{}\" codon_pos={} nt_pos={} shift={} gain={};\n",
                comp.comp_id,
                hnex_escape(&fs_ev.seq_id),
                fs_ev.approx_codon_pos,
                fs_ev.nt_pos,
                fs_ev.phase_shift,
                fs_ev.mdl_gain_nats,
            ));
        }
    }

    out.push_str("END;\n");
    out
}

#[cfg(not(target_arch = "wasm32"))]
pub fn write_hnex_file(path: &Path, arch: &HierarchicalArchive) -> Result<(), String> {
    let text = write_hnex_str(arch);
    fs::write(path, text).map_err(|e| format!("Failed to write .hnex '{}': {e}", path.display()))
}

fn parse_kv_tokens(line: &str) -> Result<(&str, HashMap<String, String>), String> {
    let trimmed = line.trim().trim_end_matches(';').trim();
    let bytes = trimmed.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    let tag = &trimmed[..i];
    let mut map = HashMap::new();

    while i < bytes.len() {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        let k_start = i;
        while i < bytes.len() && bytes[i] != b'=' {
            i += 1;
        }
        if i >= bytes.len() {
            return Err(format!("Expected '=' in HOMOLOGY record: {line}"));
        }
        let key = trimmed[k_start..i].to_string();
        i += 1; // skip '='
        if i < bytes.len() && bytes[i] == b'"' {
            i += 1;
            let v_start = i;
            while i < bytes.len() && bytes[i] != b'"' {
                i += 1;
            }
            if i >= bytes.len() {
                return Err(format!("Unterminated quote in HOMOLOGY record: {line}"));
            }
            let val = hnex_unescape(&trimmed[v_start..i]);
            map.insert(key, val);
            i += 1;
        } else {
            let v_start = i;
            while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            map.insert(key, trimmed[v_start..i].to_string());
        }
    }
    Ok((tag, map))
}

fn req_str<'a>(m: &'a HashMap<String, String>, k: &str) -> Result<&'a str, String> {
    m.get(k)
        .map(|s| s.as_str())
        .ok_or_else(|| format!("Missing key '{k}' in HOMOLOGY record"))
}

fn req_parse<T: std::str::FromStr>(m: &HashMap<String, String>, k: &str) -> Result<T, String>
where
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    let s = req_str(m, k)?;
    s.parse::<T>()
        .map_err(|e| format!("Failed parsing '{k}'='{s}': {e}"))
}

fn req_opt_usize(m: &HashMap<String, String>, k: &str) -> Result<Option<usize>, String> {
    let s = req_str(m, k)?;
    if s == "-" {
        Ok(None)
    } else {
        s.parse::<usize>()
            .map(Some)
            .map_err(|e| format!("Failed parsing '{k}'='{s}': {e}"))
    }
}

pub fn read_hnex_str(text: &str) -> Result<HierarchicalArchive, String> {
    let mut in_homology = false;
    let mut saw_meta = false;
    let mut format_version = 1u32;
    let mut mode = SeqMode::Protein;
    let mut mito_tga_trp = false;
    let mut raw_null_cohort_nats = 0.0;
    let mut provenance_cohort_nats = 0.0;
    let mut net_cohort_gain_nats = 0.0;

    let mut seqs: Vec<ArchSeq> = Vec::new();
    let mut comps: BTreeMap<usize, ArchComponent> = BTreeMap::new();

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let upper = line.to_ascii_uppercase();
        if !in_homology {
            if upper.starts_with("BEGIN HOMOLOGY") {
                in_homology = true;
            }
            continue;
        }
        if upper == "END;" || upper == "END" {
            break;
        }

        let (tag, kv) = parse_kv_tokens(line)?;
        match tag {
            "META" => {
                saw_meta = true;
                format_version = req_parse(&kv, "version")?;
                mode = str_to_mode(req_str(&kv, "mode")?)?;
                mito_tga_trp = req_parse(&kv, "mito_tga_trp")?;
                raw_null_cohort_nats = req_parse(&kv, "raw_null_nats")?;
                provenance_cohort_nats = req_parse(&kv, "prov_nats")?;
                net_cohort_gain_nats = req_parse(&kv, "net_gain_nats")?;
            }
            "SEQ" => {
                seqs.push(ArchSeq {
                    seq_id: req_parse(&kv, "id")?,
                    name: req_str(&kv, "name")?.to_string(),
                    comp_id: req_parse(&kv, "comp")?,
                    local_seq_idx: req_parse(&kv, "local")?,
                    orig_offset: req_parse(&kv, "orig_offset")?,
                    seq_len: req_parse(&kv, "len")?,
                    raw_seq: req_str(&kv, "raw")?.to_string(),
                });
            }
            "COMP" => {
                let comp_id: usize = req_parse(&kv, "id")?;
                let seq_ids_s = req_str(&kv, "seq_ids")?;
                let seq_ids: Vec<usize> = if seq_ids_s.is_empty() {
                    Vec::new()
                } else {
                    seq_ids_s
                        .split(',')
                        .map(|t| t.parse::<usize>().map_err(|e| format!("{e}")))
                        .collect::<Result<Vec<_>, _>>()?
                };
                comps.insert(
                    comp_id,
                    ArchComponent {
                        comp_id,
                        root_node_id: req_parse(&kv, "root")?,
                        num_columns: req_parse(&kv, "n_cols")?,
                        num_global_fine_cols: req_parse(&kv, "n_global")?,
                        num_clade_local_cols: req_parse(&kv, "n_clade")?,
                        num_lineage_private_cols: req_parse(&kv, "n_priv")?,
                        mean_core_posterior: req_parse(&kv, "mean_p_core")?,
                        seq_ids,
                        newick_tree: req_str(&kv, "newick")?.to_string(),
                        tree_nodes: Vec::new(),
                        blocks: Vec::new(),
                        horizon_bubbles: Vec::new(),
                        frameshifts_repaired: Vec::new(),
                    },
                );
            }
            "NODE" => {
                let comp_id: usize = req_parse(&kv, "comp")?;
                let comp = comps
                    .get_mut(&comp_id)
                    .ok_or_else(|| format!("NODE before COMP {comp_id}"))?;
                comp.tree_nodes.push(TreeNodeRecord {
                    node_id: req_parse(&kv, "id")?,
                    is_leaf: req_parse(&kv, "leaf")?,
                    label: req_str(&kv, "label")?.to_string(),
                    left_node_id: req_opt_usize(&kv, "left")?,
                    right_node_id: req_opt_usize(&kv, "right")?,
                    len_left: req_parse(&kv, "len_l")?,
                    len_right: req_parse(&kv, "len_r")?,
                    local_seq_idx: req_opt_usize(&kv, "seq")?,
                });
            }
            "BLOCK" => {
                let comp_id: usize = req_parse(&kv, "comp")?;
                let comp = comps
                    .get_mut(&comp_id)
                    .ok_or_else(|| format!("BLOCK before COMP {comp_id}"))?;
                comp.blocks.push(CollinearBlock {
                    block_id: req_parse(&kv, "id")?,
                    col_start: req_parse(&kv, "col_start")?,
                    col_len: req_parse(&kv, "col_len")?,
                    scale: str_to_scale(req_str(&kv, "scale")?)?,
                    owner_node_id: req_parse(&kv, "owner")?,
                    segments: parse_segments(req_str(&kv, "segs")?)?,
                    fine_posteriors: parse_f64_csv(req_str(&kv, "p_fine")?)?,
                    bubble_posteriors: parse_f64_csv(req_str(&kv, "p_bub")?)?,
                });
            }
            "BUBBLE" => {
                let comp_id: usize = req_parse(&kv, "comp")?;
                let comp = comps
                    .get_mut(&comp_id)
                    .ok_or_else(|| format!("BUBBLE before COMP {comp_id}"))?;
                let cls_raw = req_str(&kv, "class")?;
                let cls = if cls_raw == "-" {
                    None
                } else {
                    Some(cls_raw.to_string())
                };
                comp.horizon_bubbles.push(HorizonBubbleReport {
                    tree_merge_step: req_parse(&kv, "step")?,
                    split_node_id: req_parse(&kv, "split")?,
                    left_node_id: req_parse(&kv, "left_node")?,
                    right_node_id: req_parse(&kv, "right_node")?,
                    left_clade_size: req_parse(&kv, "l_size")?,
                    right_clade_size: req_parse(&kv, "r_size")?,
                    left_col_span: req_parse(&kv, "l_span")?,
                    right_col_span: req_parse(&kv, "r_span")?,
                    left_intervals: parse_intervals(req_str(&kv, "l_iv")?)?,
                    right_intervals: parse_intervals(req_str(&kv, "r_iv")?)?,
                    dominant_biophys_class: cls,
                    macro_biophys_gain_nats: req_parse(&kv, "gain")?,
                    delta_hydropathy: req_parse(&kv, "d_hydro")?,
                    delta_charge: req_parse(&kv, "d_chg")?,
                });
            }
            "FS" => {
                let comp_id: usize = req_parse(&kv, "comp")?;
                let comp = comps
                    .get_mut(&comp_id)
                    .ok_or_else(|| format!("FS before COMP {comp_id}"))?;
                comp.frameshifts_repaired.push(FrameshiftEvent {
                    seq_id: req_str(&kv, "seq")?.to_string(),
                    approx_codon_pos: req_parse(&kv, "codon_pos")?,
                    nt_pos: req_parse(&kv, "nt_pos")?,
                    phase_shift: req_parse(&kv, "shift")?,
                    mdl_gain_nats: req_parse(&kv, "gain")?,
                });
            }
            other => return Err(format!("Unknown HOMOLOGY statement '{other}'")),
        }
    }

    if !saw_meta {
        return Err("Missing BEGIN HOMOLOGY / META block in .hnex input".to_string());
    }

    seqs.sort_by_key(|s| s.seq_id);
    for comp in comps.values_mut() {
        comp.tree_nodes.sort_by_key(|n| n.node_id);
        comp.blocks.sort_by_key(|b| b.block_id);
    }

    Ok(HierarchicalArchive {
        format_version,
        mode,
        mito_tga_trp,
        raw_null_cohort_nats,
        provenance_cohort_nats,
        net_cohort_gain_nats,
        seqs,
        components: comps.into_values().collect(),
    })
}

#[cfg(not(target_arch = "wasm32"))]
pub fn read_hnex_file(path: &Path) -> Result<HierarchicalArchive, String> {
    let text = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read .hnex '{}': {e}", path.display()))?;
    read_hnex_str(&text)
}

// -----------------------------------------------------------------------------
// Tier 2: Native SQLite (.hdb)
// -----------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
const HDB_SCHEMA: &str = r#"
PRAGMA synchronous = OFF;
PRAGMA journal_mode = MEMORY;
CREATE TABLE meta (
    key TEXT PRIMARY KEY,
    val TEXT NOT NULL
);
CREATE TABLE sequences (
    seq_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    comp_id INTEGER NOT NULL,
    local_seq_idx INTEGER NOT NULL,
    orig_offset INTEGER NOT NULL,
    seq_len INTEGER NOT NULL,
    raw_seq TEXT NOT NULL
);
CREATE INDEX idx_sequences_name ON sequences(name);
CREATE TABLE components (
    comp_id INTEGER PRIMARY KEY,
    root_node_id INTEGER NOT NULL,
    num_columns INTEGER NOT NULL,
    num_global_fine_cols INTEGER NOT NULL,
    num_clade_local_cols INTEGER NOT NULL,
    num_lineage_private_cols INTEGER NOT NULL,
    mean_core_posterior REAL NOT NULL,
    seq_ids_csv TEXT NOT NULL,
    newick_tree TEXT NOT NULL
);
CREATE TABLE tree_nodes (
    comp_id INTEGER NOT NULL,
    node_id INTEGER NOT NULL,
    is_leaf INTEGER NOT NULL,
    label TEXT NOT NULL,
    left_node_id INTEGER,
    right_node_id INTEGER,
    len_left REAL NOT NULL,
    len_right REAL NOT NULL,
    local_seq_idx INTEGER,
    PRIMARY KEY (comp_id, node_id)
);
CREATE TABLE blocks (
    comp_id INTEGER NOT NULL,
    block_id INTEGER NOT NULL,
    col_start INTEGER NOT NULL,
    col_len INTEGER NOT NULL,
    scale TEXT NOT NULL,
    owner_node_id INTEGER NOT NULL,
    fine_posteriors_csv TEXT NOT NULL,
    bubble_posteriors_csv TEXT NOT NULL,
    PRIMARY KEY (comp_id, block_id)
);
CREATE INDEX idx_blocks_cols ON blocks(comp_id, col_start);
CREATE TABLE block_segments (
    comp_id INTEGER NOT NULL,
    block_id INTEGER NOT NULL,
    local_seq_idx INTEGER NOT NULL,
    seq_id INTEGER NOT NULL,
    start_pos INTEGER NOT NULL,
    end_pos INTEGER NOT NULL,
    PRIMARY KEY (comp_id, block_id, local_seq_idx)
);
CREATE INDEX idx_segments_spatial ON block_segments(seq_id, start_pos, end_pos);
CREATE TABLE horizon_bubbles (
    comp_id INTEGER NOT NULL,
    bubble_idx INTEGER NOT NULL,
    tree_merge_step INTEGER NOT NULL,
    split_node_id INTEGER NOT NULL,
    left_node_id INTEGER NOT NULL,
    right_node_id INTEGER NOT NULL,
    left_clade_size INTEGER NOT NULL,
    right_clade_size INTEGER NOT NULL,
    left_col_span INTEGER NOT NULL,
    right_col_span INTEGER NOT NULL,
    left_intervals_str TEXT NOT NULL,
    right_intervals_str TEXT NOT NULL,
    dominant_biophys_class TEXT,
    macro_biophys_gain_nats REAL NOT NULL,
    delta_hydropathy REAL NOT NULL,
    delta_charge REAL NOT NULL,
    PRIMARY KEY (comp_id, bubble_idx)
);
CREATE TABLE frameshifts (
    comp_id INTEGER NOT NULL,
    fs_idx INTEGER NOT NULL,
    seq_id TEXT NOT NULL,
    approx_codon_pos INTEGER NOT NULL,
    nt_pos INTEGER NOT NULL,
    phase_shift INTEGER NOT NULL,
    mdl_gain_nats REAL NOT NULL,
    PRIMARY KEY (comp_id, fs_idx)
);
"#;

#[cfg(not(target_arch = "wasm32"))]
pub fn write_hdb_file(path: &Path, arch: &HierarchicalArchive) -> Result<(), String> {
    if path.exists() {
        fs::remove_file(path)
            .map_err(|e| format!("Failed replacing existing .hdb '{}': {e}", path.display()))?;
    }
    let db = SqliteDb::open(path, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE)?;
    db.exec(HDB_SCHEMA)?;
    db.exec("BEGIN TRANSACTION;")?;

    {
        let st_meta = db.prepare("INSERT INTO meta(key, val) VALUES(?1, ?2);")?;
        let metas = [
            ("format_version", arch.format_version.to_string()),
            ("mode", mode_to_str(arch.mode).to_string()),
            ("mito_tga_trp", arch.mito_tga_trp.to_string()),
            ("raw_null_cohort_nats", arch.raw_null_cohort_nats.to_string()),
            ("provenance_cohort_nats", arch.provenance_cohort_nats.to_string()),
            ("net_cohort_gain_nats", arch.net_cohort_gain_nats.to_string()),
        ];
        for (k, v) in &metas {
            st_meta.reset()?;
            st_meta.bind_text(1, k)?;
            st_meta.bind_text(2, v)?;
            st_meta.step_done()?;
        }

        let st_seq = db.prepare(
            "INSERT INTO sequences(seq_id, name, comp_id, local_seq_idx, orig_offset, seq_len, raw_seq) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7);",
        )?;
        for s in &arch.seqs {
            st_seq.reset()?;
            st_seq.bind_i64(1, s.seq_id as i64)?;
            st_seq.bind_text(2, &s.name)?;
            st_seq.bind_i64(3, s.comp_id as i64)?;
            st_seq.bind_i64(4, s.local_seq_idx as i64)?;
            st_seq.bind_i64(5, s.orig_offset as i64)?;
            st_seq.bind_i64(6, s.seq_len as i64)?;
            st_seq.bind_text(7, &s.raw_seq)?;
            st_seq.step_done()?;
        }

        let st_comp = db.prepare(
            "INSERT INTO components(comp_id, root_node_id, num_columns, num_global_fine_cols, num_clade_local_cols, num_lineage_private_cols, mean_core_posterior, seq_ids_csv, newick_tree) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9);",
        )?;
        let st_node = db.prepare(
            "INSERT INTO tree_nodes(comp_id, node_id, is_leaf, label, left_node_id, right_node_id, len_left, len_right, local_seq_idx) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9);",
        )?;
        let st_blk = db.prepare(
            "INSERT INTO blocks(comp_id, block_id, col_start, col_len, scale, owner_node_id, fine_posteriors_csv, bubble_posteriors_csv) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8);",
        )?;
        let st_seg = db.prepare(
            "INSERT INTO block_segments(comp_id, block_id, local_seq_idx, seq_id, start_pos, end_pos) VALUES(?1, ?2, ?3, ?4, ?5, ?6);",
        )?;
        let st_hb = db.prepare(
            "INSERT INTO horizon_bubbles(comp_id, bubble_idx, tree_merge_step, split_node_id, left_node_id, right_node_id, left_clade_size, right_clade_size, left_col_span, right_col_span, left_intervals_str, right_intervals_str, dominant_biophys_class, macro_biophys_gain_nats, delta_hydropathy, delta_charge) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16);",
        )?;
        let st_fs = db.prepare(
            "INSERT INTO frameshifts(comp_id, fs_idx, seq_id, approx_codon_pos, nt_pos, phase_shift, mdl_gain_nats) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7);",
        )?;

        for comp in &arch.components {
            st_comp.reset()?;
            st_comp.bind_i64(1, comp.comp_id as i64)?;
            st_comp.bind_i64(2, comp.root_node_id as i64)?;
            st_comp.bind_i64(3, comp.num_columns as i64)?;
            st_comp.bind_i64(4, comp.num_global_fine_cols as i64)?;
            st_comp.bind_i64(5, comp.num_clade_local_cols as i64)?;
            st_comp.bind_i64(6, comp.num_lineage_private_cols as i64)?;
            st_comp.bind_f64(7, comp.mean_core_posterior)?;
            let sids_csv: String = comp
                .seq_ids
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",");
            st_comp.bind_text(8, &sids_csv)?;
            st_comp.bind_text(9, &comp.newick_tree)?;
            st_comp.step_done()?;

            for node in &comp.tree_nodes {
                st_node.reset()?;
                st_node.bind_i64(1, comp.comp_id as i64)?;
                st_node.bind_i64(2, node.node_id as i64)?;
                st_node.bind_i64(3, if node.is_leaf { 1 } else { 0 })?;
                st_node.bind_text(4, &node.label)?;
                st_node.bind_opt_i64(5, node.left_node_id)?;
                st_node.bind_opt_i64(6, node.right_node_id)?;
                st_node.bind_f64(7, node.len_left)?;
                st_node.bind_f64(8, node.len_right)?;
                st_node.bind_opt_i64(9, node.local_seq_idx)?;
                st_node.step_done()?;
            }

            for blk in &comp.blocks {
                st_blk.reset()?;
                st_blk.bind_i64(1, comp.comp_id as i64)?;
                st_blk.bind_i64(2, blk.block_id as i64)?;
                st_blk.bind_i64(3, blk.col_start as i64)?;
                st_blk.bind_i64(4, blk.col_len as i64)?;
                st_blk.bind_text(5, scale_to_str(blk.scale))?;
                st_blk.bind_i64(6, blk.owner_node_id as i64)?;
                st_blk.bind_text(7, &fmt_f64_csv(&blk.fine_posteriors))?;
                st_blk.bind_text(8, &fmt_f64_csv(&blk.bubble_posteriors))?;
                st_blk.step_done()?;

                for seg in &blk.segments {
                    st_seg.reset()?;
                    st_seg.bind_i64(1, comp.comp_id as i64)?;
                    st_seg.bind_i64(2, blk.block_id as i64)?;
                    st_seg.bind_i64(3, seg.local_seq_idx as i64)?;
                    st_seg.bind_i64(4, seg.seq_id as i64)?;
                    st_seg.bind_i64(5, seg.start_pos as i64)?;
                    st_seg.bind_i64(6, seg.end_pos as i64)?;
                    st_seg.step_done()?;
                }
            }

            for (b_idx, hb) in comp.horizon_bubbles.iter().enumerate() {
                st_hb.reset()?;
                st_hb.bind_i64(1, comp.comp_id as i64)?;
                st_hb.bind_i64(2, b_idx as i64)?;
                st_hb.bind_i64(3, hb.tree_merge_step as i64)?;
                st_hb.bind_i64(4, hb.split_node_id as i64)?;
                st_hb.bind_i64(5, hb.left_node_id as i64)?;
                st_hb.bind_i64(6, hb.right_node_id as i64)?;
                st_hb.bind_i64(7, hb.left_clade_size as i64)?;
                st_hb.bind_i64(8, hb.right_clade_size as i64)?;
                st_hb.bind_i64(9, hb.left_col_span as i64)?;
                st_hb.bind_i64(10, hb.right_col_span as i64)?;
                st_hb.bind_text(11, &fmt_intervals(&hb.left_intervals))?;
                st_hb.bind_text(12, &fmt_intervals(&hb.right_intervals))?;
                st_hb.bind_opt_text(13, hb.dominant_biophys_class.as_deref())?;
                st_hb.bind_f64(14, hb.macro_biophys_gain_nats)?;
                st_hb.bind_f64(15, hb.delta_hydropathy)?;
                st_hb.bind_f64(16, hb.delta_charge)?;
                st_hb.step_done()?;
            }

            for (f_idx, fs_ev) in comp.frameshifts_repaired.iter().enumerate() {
                st_fs.reset()?;
                st_fs.bind_i64(1, comp.comp_id as i64)?;
                st_fs.bind_i64(2, f_idx as i64)?;
                st_fs.bind_text(3, &fs_ev.seq_id)?;
                st_fs.bind_i64(4, fs_ev.approx_codon_pos as i64)?;
                st_fs.bind_i64(5, fs_ev.nt_pos as i64)?;
                st_fs.bind_i64(6, fs_ev.phase_shift as i64)?;
                st_fs.bind_f64(7, fs_ev.mdl_gain_nats)?;
                st_fs.step_done()?;
            }
        }
    }

    db.exec("COMMIT;")?;
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn read_hdb_meta(db: &SqliteDb) -> Result<HashMap<String, String>, String> {
    let st = db.prepare("SELECT key, val FROM meta;")?;
    let mut map = HashMap::new();
    while st.step_row()? {
        map.insert(st.col_text(0), st.col_text(1));
    }
    Ok(map)
}

#[cfg(not(target_arch = "wasm32"))]
fn load_hdb_component(
    db: &SqliteDb,
    comp_id: usize,
    col_window: Option<(usize, usize)>,
) -> Result<ArchComponent, String> {
    let st_c = db.prepare(
        "SELECT root_node_id, num_columns, num_global_fine_cols, num_clade_local_cols, num_lineage_private_cols, mean_core_posterior, seq_ids_csv, newick_tree FROM components WHERE comp_id = ?1;",
    )?;
    st_c.bind_i64(1, comp_id as i64)?;
    if !st_c.step_row()? {
        return Err(format!("Component {comp_id} not found in .hdb"));
    }
    let root_node_id = st_c.col_usize(0);
    let num_columns = st_c.col_usize(1);
    let num_global_fine_cols = st_c.col_usize(2);
    let num_clade_local_cols = st_c.col_usize(3);
    let num_lineage_private_cols = st_c.col_usize(4);
    let mean_core_posterior = st_c.col_f64(5);
    let seq_ids_csv = st_c.col_text(6);
    let newick_tree = st_c.col_text(7);
    let seq_ids: Vec<usize> = if seq_ids_csv.is_empty() {
        Vec::new()
    } else {
        seq_ids_csv
            .split(',')
            .map(|t| t.parse::<usize>().map_err(|e| format!("{e}")))
            .collect::<Result<Vec<_>, _>>()?
    };

    let mut tree_nodes = Vec::new();
    let st_n = db.prepare(
        "SELECT node_id, is_leaf, label, left_node_id, right_node_id, len_left, len_right, local_seq_idx FROM tree_nodes WHERE comp_id = ?1 ORDER BY node_id;",
    )?;
    st_n.bind_i64(1, comp_id as i64)?;
    while st_n.step_row()? {
        tree_nodes.push(TreeNodeRecord {
            node_id: st_n.col_usize(0),
            is_leaf: st_n.col_i64(1) != 0,
            label: st_n.col_text(2),
            left_node_id: st_n.col_opt_usize(3),
            right_node_id: st_n.col_opt_usize(4),
            len_left: st_n.col_f64(5),
            len_right: st_n.col_f64(6),
            local_seq_idx: st_n.col_opt_usize(7),
        });
    }

    let mut blocks = Vec::new();
    let mut blk_idx_by_id: HashMap<usize, usize> = HashMap::new();
    if let Some((w_lo, w_hi)) = col_window {
        let st_b = db.prepare(
            "SELECT block_id, col_start, col_len, scale, owner_node_id, fine_posteriors_csv, bubble_posteriors_csv FROM blocks WHERE comp_id = ?1 AND (col_start + col_len) > ?2 AND col_start < ?3 ORDER BY col_start;",
        )?;
        st_b.bind_i64(1, comp_id as i64)?;
        st_b.bind_i64(2, w_lo as i64)?;
        st_b.bind_i64(3, w_hi as i64)?;
        while st_b.step_row()? {
            let bid = st_b.col_usize(0);
            blk_idx_by_id.insert(bid, blocks.len());
            blocks.push(CollinearBlock {
                block_id: bid,
                col_start: st_b.col_usize(1),
                col_len: st_b.col_usize(2),
                scale: str_to_scale(&st_b.col_text(3))?,
                owner_node_id: st_b.col_usize(4),
                segments: Vec::new(),
                fine_posteriors: parse_f64_csv(&st_b.col_text(5))?,
                bubble_posteriors: parse_f64_csv(&st_b.col_text(6))?,
            });
        }
        let st_s = db.prepare(
            "SELECT s.block_id, s.local_seq_idx, s.seq_id, s.start_pos, s.end_pos FROM block_segments s JOIN blocks b ON b.comp_id = s.comp_id AND b.block_id = s.block_id WHERE b.comp_id = ?1 AND (b.col_start + b.col_len) > ?2 AND b.col_start < ?3 ORDER BY s.block_id, s.local_seq_idx;",
        )?;
        st_s.bind_i64(1, comp_id as i64)?;
        st_s.bind_i64(2, w_lo as i64)?;
        st_s.bind_i64(3, w_hi as i64)?;
        while st_s.step_row()? {
            let bid = st_s.col_usize(0);
            if let Some(&idx) = blk_idx_by_id.get(&bid) {
                blocks[idx].segments.push(BlockSegment {
                    local_seq_idx: st_s.col_usize(1),
                    seq_id: st_s.col_usize(2),
                    start_pos: st_s.col_usize(3),
                    end_pos: st_s.col_usize(4),
                });
            }
        }
    } else {
        let st_b = db.prepare(
            "SELECT block_id, col_start, col_len, scale, owner_node_id, fine_posteriors_csv, bubble_posteriors_csv FROM blocks WHERE comp_id = ?1 ORDER BY block_id;",
        )?;
        st_b.bind_i64(1, comp_id as i64)?;
        while st_b.step_row()? {
            let bid = st_b.col_usize(0);
            blk_idx_by_id.insert(bid, blocks.len());
            blocks.push(CollinearBlock {
                block_id: bid,
                col_start: st_b.col_usize(1),
                col_len: st_b.col_usize(2),
                scale: str_to_scale(&st_b.col_text(3))?,
                owner_node_id: st_b.col_usize(4),
                segments: Vec::new(),
                fine_posteriors: parse_f64_csv(&st_b.col_text(5))?,
                bubble_posteriors: parse_f64_csv(&st_b.col_text(6))?,
            });
        }
        let st_s = db.prepare(
            "SELECT block_id, local_seq_idx, seq_id, start_pos, end_pos FROM block_segments WHERE comp_id = ?1 ORDER BY block_id, local_seq_idx;",
        )?;
        st_s.bind_i64(1, comp_id as i64)?;
        while st_s.step_row()? {
            let bid = st_s.col_usize(0);
            if let Some(&idx) = blk_idx_by_id.get(&bid) {
                blocks[idx].segments.push(BlockSegment {
                    local_seq_idx: st_s.col_usize(1),
                    seq_id: st_s.col_usize(2),
                    start_pos: st_s.col_usize(3),
                    end_pos: st_s.col_usize(4),
                });
            }
        }
    }

    let mut horizon_bubbles = Vec::new();
    let st_hb = db.prepare(
        "SELECT tree_merge_step, split_node_id, left_node_id, right_node_id, left_clade_size, right_clade_size, left_col_span, right_col_span, left_intervals_str, right_intervals_str, dominant_biophys_class, macro_biophys_gain_nats, delta_hydropathy, delta_charge FROM horizon_bubbles WHERE comp_id = ?1 ORDER BY bubble_idx;",
    )?;
    st_hb.bind_i64(1, comp_id as i64)?;
    while st_hb.step_row()? {
        horizon_bubbles.push(HorizonBubbleReport {
            tree_merge_step: st_hb.col_usize(0),
            split_node_id: st_hb.col_usize(1),
            left_node_id: st_hb.col_usize(2),
            right_node_id: st_hb.col_usize(3),
            left_clade_size: st_hb.col_usize(4),
            right_clade_size: st_hb.col_usize(5),
            left_col_span: st_hb.col_usize(6),
            right_col_span: st_hb.col_usize(7),
            left_intervals: parse_intervals(&st_hb.col_text(8))?,
            right_intervals: parse_intervals(&st_hb.col_text(9))?,
            dominant_biophys_class: st_hb.col_opt_text(10),
            macro_biophys_gain_nats: st_hb.col_f64(11),
            delta_hydropathy: st_hb.col_f64(12),
            delta_charge: st_hb.col_f64(13),
        });
    }

    let mut frameshifts_repaired = Vec::new();
    let st_fs = db.prepare(
        "SELECT seq_id, approx_codon_pos, nt_pos, phase_shift, mdl_gain_nats FROM frameshifts WHERE comp_id = ?1 ORDER BY fs_idx;",
    )?;
    st_fs.bind_i64(1, comp_id as i64)?;
    while st_fs.step_row()? {
        frameshifts_repaired.push(FrameshiftEvent {
            seq_id: st_fs.col_text(0),
            approx_codon_pos: st_fs.col_usize(1),
            nt_pos: st_fs.col_usize(2),
            phase_shift: st_fs.col_usize(3) as u8,
            mdl_gain_nats: st_fs.col_f64(4),
        });
    }

    Ok(ArchComponent {
        comp_id,
        root_node_id,
        num_columns,
        num_global_fine_cols,
        num_clade_local_cols,
        num_lineage_private_cols,
        mean_core_posterior,
        seq_ids,
        newick_tree,
        tree_nodes,
        blocks,
        horizon_bubbles,
        frameshifts_repaired,
    })
}

#[cfg(not(target_arch = "wasm32"))]
pub fn read_hdb_file(path: &Path) -> Result<HierarchicalArchive, String> {
    let db = SqliteDb::open(path, SQLITE_OPEN_READONLY)?;
    let meta = read_hdb_meta(&db)?;

    let format_version: u32 = meta
        .get("format_version")
        .ok_or_else(|| "Missing format_version in .hdb meta".to_string())?
        .parse()
        .map_err(|e| format!("{e}"))?;
    let mode = str_to_mode(
        meta.get("mode")
            .ok_or_else(|| "Missing mode in .hdb meta".to_string())?,
    )?;
    let mito_tga_trp: bool = meta
        .get("mito_tga_trp")
        .map(|s| s.parse())
        .transpose()
        .map_err(|e| format!("{e}"))?
        .unwrap_or(false);
    let raw_null_cohort_nats: f64 = meta
        .get("raw_null_cohort_nats")
        .map(|s| s.parse())
        .transpose()
        .map_err(|e| format!("{e}"))?
        .unwrap_or(0.0);
    let provenance_cohort_nats: f64 = meta
        .get("provenance_cohort_nats")
        .map(|s| s.parse())
        .transpose()
        .map_err(|e| format!("{e}"))?
        .unwrap_or(0.0);
    let net_cohort_gain_nats: f64 = meta
        .get("net_cohort_gain_nats")
        .map(|s| s.parse())
        .transpose()
        .map_err(|e| format!("{e}"))?
        .unwrap_or(0.0);

    let mut seqs = Vec::new();
    let st_s = db.prepare(
        "SELECT seq_id, name, comp_id, local_seq_idx, orig_offset, seq_len, raw_seq FROM sequences ORDER BY seq_id;",
    )?;
    while st_s.step_row()? {
        seqs.push(ArchSeq {
            seq_id: st_s.col_usize(0),
            name: st_s.col_text(1),
            comp_id: st_s.col_usize(2),
            local_seq_idx: st_s.col_usize(3),
            orig_offset: st_s.col_usize(4),
            seq_len: st_s.col_usize(5),
            raw_seq: st_s.col_text(6),
        });
    }

    let mut comp_ids = Vec::new();
    let st_cid = db.prepare("SELECT comp_id FROM components ORDER BY comp_id;")?;
    while st_cid.step_row()? {
        comp_ids.push(st_cid.col_usize(0));
    }
    let mut components = Vec::with_capacity(comp_ids.len());
    for cid in comp_ids {
        components.push(load_hdb_component(&db, cid, None)?);
    }

    Ok(HierarchicalArchive {
        format_version,
        mode,
        mito_tga_trp,
        raw_null_cohort_nats,
        provenance_cohort_nats,
        net_cohort_gain_nats,
        seqs,
        components,
    })
}

#[cfg(not(target_arch = "wasm32"))]
pub fn is_hdb_file(path: &Path) -> bool {
    if path
        .extension()
        .and_then(|e| e.to_str())
        .map_or(false, |e| e.eq_ignore_ascii_case("hdb") || e.eq_ignore_ascii_case("sqlite") || e.eq_ignore_ascii_case("db"))
    {
        return true;
    }
    if let Ok(bytes) = fs::read(path) {
        return bytes.starts_with(b"SQLite format 3\0");
    }
    false
}

#[cfg(not(target_arch = "wasm32"))]
pub fn read_archive_auto(path: &Path) -> Result<HierarchicalArchive, String> {
    if is_hdb_file(path) {
        read_hdb_file(path)
    } else {
        read_hnex_file(path)
    }
}

// -----------------------------------------------------------------------------
// Hierarchical Subsetting (2 -> 1 / Component / Clade / Taxa / Spatial Region)
// -----------------------------------------------------------------------------

/// Prune `tree_nodes` down to leaves where `keep_leaf[old_local_idx]` is true,
/// contracting degree-2 internal nodes while adding branch lengths (preserving patristic distances).
/// Also returns `old_to_new: Vec<Option<usize>>` mapping every old `node_id` to its surviving
/// descendant representative in the pruned tree.
fn prune_and_contract_tree(
    root_node_id: usize,
    tree_nodes: &[TreeNodeRecord],
    keep_leaf: &[bool],
    old_to_new_leaf: &[Option<usize>],
) -> Result<(Vec<TreeNodeRecord>, usize, String, Vec<Option<usize>>), String> {
    let mut node_by_id: HashMap<usize, usize> = HashMap::with_capacity(tree_nodes.len());
    let mut max_old_id = 0usize;
    for (i, rec) in tree_nodes.iter().enumerate() {
        node_by_id.insert(rec.node_id, i);
        max_old_id = max_old_id.max(rec.node_id);
    }

    fn flatten_exact(
        old_id: usize,
        node_by_id: &HashMap<usize, usize>,
        tree_nodes: &[TreeNodeRecord],
        keep_leaf: &[bool],
        old_to_new_leaf: &[Option<usize>],
        records: &mut [Option<TreeNodeRecord>],
        next_internal: &mut usize,
        old_to_new: &mut [Option<usize>],
    ) -> Option<(usize, f64, String)> {
        let rec = &tree_nodes[node_by_id[&old_id]];
        if rec.is_leaf {
            let loc = rec.local_seq_idx?;
            if loc < keep_leaf.len() && keep_leaf[loc] {
                let new_loc = old_to_new_leaf[loc]?;
                records[new_loc] = Some(TreeNodeRecord {
                    node_id: new_loc,
                    is_leaf: true,
                    label: rec.label.clone(),
                    left_node_id: None,
                    right_node_id: None,
                    len_left: 0.0,
                    len_right: 0.0,
                    local_seq_idx: Some(new_loc),
                });
                old_to_new[old_id] = Some(new_loc);
                return Some((new_loc, 0.0, rec.label.clone()));
            }
            return None;
        }
        let res_l = rec
            .left_node_id
            .and_then(|lid| {
                flatten_exact(
                    lid,
                    node_by_id,
                    tree_nodes,
                    keep_leaf,
                    old_to_new_leaf,
                    records,
                    next_internal,
                    old_to_new,
                )
            })
            .map(|(nid, d, s)| (nid, d + rec.len_left, s));
        let res_r = rec
            .right_node_id
            .and_then(|rid| {
                flatten_exact(
                    rid,
                    node_by_id,
                    tree_nodes,
                    keep_leaf,
                    old_to_new_leaf,
                    records,
                    next_internal,
                    old_to_new,
                )
            })
            .map(|(nid, d, s)| (nid, d + rec.len_right, s));

        match (res_l, res_r) {
            (None, None) => None,
            (Some((nl, dl, sl)), None) => {
                old_to_new[old_id] = Some(nl);
                Some((nl, dl, sl))
            }
            (None, Some((nr, dr, sr))) => {
                old_to_new[old_id] = Some(nr);
                Some((nr, dr, sr))
            }
            (Some((nl, dl, sl)), Some((nr, dr, sr))) => {
                let id = *next_internal;
                *next_internal += 1;
                let dl = dl.max(0.0);
                let dr = dr.max(0.0);
                records[id] = Some(TreeNodeRecord {
                    node_id: id,
                    is_leaf: false,
                    label: rec.label.clone(),
                    left_node_id: Some(nl),
                    right_node_id: Some(nr),
                    len_left: dl,
                    len_right: dr,
                    local_seq_idx: None,
                });
                old_to_new[old_id] = Some(id);
                let newick_sub = format!("({}:{:.5},{}:{:.5}){}", sl, dl, sr, dr, rec.label);
                Some((id, 0.0, newick_sub))
            }
        }
    }

    let new_m = keep_leaf.iter().filter(|&&b| b).count();
    let mut records: Vec<Option<TreeNodeRecord>> = vec![None; 2 * new_m - 1];
    let mut next_internal = new_m;
    let mut old_to_new = vec![None; max_old_id + 1];
    let (new_root, _, newick_body) = flatten_exact(
        root_node_id,
        &node_by_id,
        tree_nodes,
        keep_leaf,
        old_to_new_leaf,
        &mut records,
        &mut next_internal,
        &mut old_to_new,
    )
    .ok_or_else(|| "No surviving taxa remain in tree after subsetting".to_string())?;

    let out_nodes: Vec<TreeNodeRecord> = records
        .into_iter()
        .map(|r| r.expect("unpopulated pruned tree node"))
        .collect();
    Ok((out_nodes, new_root, format!("{};", newick_body), old_to_new))
}

fn find_clade_node_id(comp: &ArchComponent, clade_str: &str) -> Option<usize> {
    if let Some(rec) = comp.tree_nodes.iter().find(|n| n.label == clade_str) {
        return Some(rec.node_id);
    }
    if let Ok(nid) = clade_str.parse::<usize>() {
        if comp.tree_nodes.iter().any(|n| n.node_id == nid) {
            return Some(nid);
        }
    }
    None
}

fn pack_subset_component(
    comp: &ArchComponent,
    seqs_by_id: &HashMap<usize, &ArchSeq>,
    mode: SeqMode,
    spec: &SubsetSpec,
    col_window: Option<(usize, usize)>,
    new_comp_id: usize,
    next_seq_id: &mut usize,
) -> Result<Option<(ArchComponent, Vec<ArchSeq>)>, String> {
    let old_m = comp.seq_ids.len();
    if old_m == 0 {
        return Ok(None);
    }

    // 1. Resolve initial surviving leaf mask from --clade and --taxa
    let mut keep_leaf = vec![true; old_m];
    if let Some(ref clade_lbl) = spec.clade {
        let Some(clade_id) = find_clade_node_id(comp, clade_lbl) else {
            return Ok(None);
        };
        let desc = compute_leaf_descendants(&comp.tree_nodes, old_m);
        for t in 0..old_m {
            if !desc[clade_id][t] {
                keep_leaf[t] = false;
            }
        }
    }
    if let Some(ref taxa_names) = spec.taxa {
        let allowed: HashSet<&str> = taxa_names.iter().map(|s| s.as_str()).collect();
        for t in 0..old_m {
            let sid = comp.seq_ids[t];
            if !allowed.contains(seqs_by_id[&sid].name.as_str()) {
                keep_leaf[t] = false;
            }
        }
    }
    if !keep_leaf.iter().any(|&b| b) {
        return Ok(None);
    }

    // 2. If a column window is active, find token span [p_lo, p_hi) per surviving taxon within [w_lo, w_hi)
    // and ensure only taxa with residues in the sliced window (if region is set) stay active.
    let (w_lo, w_hi) = col_window.unwrap_or((0, comp.num_columns));
    let has_region = spec.region.is_some();
    let mut tok_bounds: Vec<Option<(usize, usize)>> = vec![None; old_m];

    // Check Identity fast path when no filters shrink anything
    if spec.clade.is_none()
        && spec.taxa.is_none()
        && col_window.is_none()
        && !spec.global_core_only
    {
        let mut out_seqs = Vec::with_capacity(old_m);
        let mut new_seq_ids = Vec::with_capacity(old_m);
        let mut seg_map_seq: HashMap<usize, usize> = HashMap::with_capacity(old_m);
        for (t, &old_sid) in comp.seq_ids.iter().enumerate() {
            let new_sid = *next_seq_id;
            *next_seq_id += 1;
            let orig = seqs_by_id[&old_sid];
            seg_map_seq.insert(old_sid, new_sid);
            new_seq_ids.push(new_sid);
            out_seqs.push(ArchSeq {
                seq_id: new_sid,
                name: orig.name.clone(),
                comp_id: new_comp_id,
                local_seq_idx: t,
                orig_offset: orig.orig_offset,
                seq_len: orig.seq_len,
                raw_seq: orig.raw_seq.clone(),
            });
        }
        let mut new_blocks = comp.blocks.clone();
        for blk in &mut new_blocks {
            for seg in &mut blk.segments {
                seg.seq_id = seg_map_seq[&seg.seq_id];
            }
        }
        let mut out_comp = comp.clone();
        out_comp.comp_id = new_comp_id;
        out_comp.seq_ids = new_seq_ids;
        out_comp.blocks = new_blocks;
        return Ok(Some((out_comp, out_seqs)));
    }

    // Collect filtered column slices before coalescing into RLE blocks
    struct TmpCol {
        scale: ColumnScale,
        old_owner: usize,
        present: Vec<(usize, usize)>, // (old_local_idx, old_pos)
        p_fine: f64,
        p_bub: f64,
    }
    let mut raw_cols: Vec<TmpCol> = Vec::new();
    for blk in &comp.blocks {
        let b_lo = blk.col_start;
        let b_hi = blk.col_start + blk.col_len;
        let ov_lo = b_lo.max(w_lo);
        let ov_hi = b_hi.min(w_hi);
        if ov_lo >= ov_hi {
            continue;
        }
        if spec.global_core_only && blk.scale != ColumnScale::GlobalFine {
            continue;
        }
        for col_abs in ov_lo..ov_hi {
            let off = col_abs - b_lo;
            let mut present = Vec::new();
            for seg in &blk.segments {
                if keep_leaf[seg.local_seq_idx] {
                    let pos = seg.start_pos + off;
                    present.push((seg.local_seq_idx, pos));
                }
            }
            if present.is_empty() {
                continue;
            }
            if has_region {
                for &(t, pos) in &present {
                    tok_bounds[t] = match tok_bounds[t] {
                        None => Some((pos, pos + 1)),
                        Some((lo, hi)) => Some((lo.min(pos), hi.max(pos + 1))),
                    };
                }
            }
            raw_cols.push(TmpCol {
                scale: blk.scale,
                old_owner: blk.owner_node_id,
                present,
                p_fine: blk.fine_posteriors.get(off).copied().unwrap_or(1.0),
                p_bub: blk.bubble_posteriors.get(off).copied().unwrap_or(0.0),
            });
        }
    }

    if raw_cols.is_empty() {
        return Ok(None);
    }

    if has_region {
        for t in 0..old_m {
            if keep_leaf[t] && tok_bounds[t].is_none() {
                keep_leaf[t] = false;
            }
        }
        if !keep_leaf.iter().any(|&b| b) {
            return Ok(None);
        }
    }

    // 3. Re-index surviving leaves tightly 0..new_m
    let mut old_to_new_leaf = vec![None; old_m];
    let mut surviving_old_indices = Vec::new();
    for t in 0..old_m {
        if keep_leaf[t] {
            old_to_new_leaf[t] = Some(surviving_old_indices.len());
            surviving_old_indices.push(t);
        }
    }
    let new_m = surviving_old_indices.len();

    // 4. Prune and contract phylogeny
    let (new_tree_nodes, new_root_id, new_newick, old_to_new_node) =
        prune_and_contract_tree(comp.root_node_id, &comp.tree_nodes, &keep_leaf, &old_to_new_leaf)?;

    // 5. Build sliced/unaligned ArchSeq records
    let step = token_step(mode);
    let mut out_seqs = Vec::with_capacity(new_m);
    let mut new_seq_ids = Vec::with_capacity(new_m);
    let mut seq_offsets = vec![0usize; old_m];
    let mut seq_slice_lens = vec![0usize; old_m];

    for (new_loc, &old_loc) in surviving_old_indices.iter().enumerate() {
        let old_sid = comp.seq_ids[old_loc];
        let orig = seqs_by_id[&old_sid];
        let new_sid = *next_seq_id;
        *next_seq_id += 1;

        let (p_lo, p_hi) = if let Some(b) = tok_bounds[old_loc] {
            b
        } else {
            (0, orig.seq_len)
        };
        seq_offsets[old_loc] = p_lo;
        seq_slice_lens[old_loc] = p_hi - p_lo;
        let sub_raw = orig.raw_seq[p_lo * step..p_hi * step].to_string();

        new_seq_ids.push(new_sid);
        out_seqs.push(ArchSeq {
            seq_id: new_sid,
            name: orig.name.clone(),
            comp_id: new_comp_id,
            local_seq_idx: new_loc,
            orig_offset: orig.orig_offset + p_lo,
            seq_len: p_hi - p_lo,
            raw_seq: sub_raw,
        });
    }

    // 6. Re-compute column scales & owners and pack into RLE CollinearBlocks
    let mut blocks: Vec<CollinearBlock> = Vec::new();
    let mut num_global_fine_cols = 0usize;
    let mut num_clade_local_cols = 0usize;
    let mut num_lineage_private_cols = 0usize;
    let mut core_post_acc = 0.0f64;
    let mut core_post_cnt = 0usize;

    for (new_c_idx, col) in raw_cols.into_iter().enumerate() {
        let new_owner = col
            .old_owner
            .lt(&old_to_new_node.len())
            .then(|| old_to_new_node[col.old_owner])
            .flatten()
            .unwrap_or(new_root_id);
        let occ = col.present.len();
        let new_scale = if occ == 1 {
            ColumnScale::LineagePrivate
        } else if new_owner == new_root_id
            || (col.scale == ColumnScale::GlobalFine && new_m >= 4 && occ * 3 > 2 * new_m)
        {
            ColumnScale::GlobalFine
        } else {
            ColumnScale::CladeLocalFine
        };

        match new_scale {
            ColumnScale::GlobalFine => {
                num_global_fine_cols += 1;
                core_post_acc += col.p_fine;
                core_post_cnt += 1;
            }
            ColumnScale::CladeLocalFine => num_clade_local_cols += 1,
            ColumnScale::LineagePrivate => num_lineage_private_cols += 1,
        }

        let new_present: Vec<(usize, usize)> = col
            .present
            .into_iter()
            .map(|(old_loc, old_pos)| {
                let new_loc = old_to_new_leaf[old_loc].unwrap();
                let new_pos = old_pos - seq_offsets[old_loc];
                (new_loc, new_pos)
            })
            .collect();

        let can_coalesce = if let Some(last) = blocks.last() {
            last.scale == new_scale
                && last.owner_node_id == new_owner
                && last.segments.len() == new_present.len()
                && last
                    .segments
                    .iter()
                    .zip(new_present.iter())
                    .all(|(seg, &(nl, np))| seg.local_seq_idx == nl && seg.end_pos == np)
        } else {
            false
        };

        if can_coalesce {
            let last = blocks.last_mut().unwrap();
            last.col_len += 1;
            for seg in &mut last.segments {
                seg.end_pos += 1;
            }
            last.fine_posteriors.push(col.p_fine);
            last.bubble_posteriors.push(col.p_bub);
        } else {
            let bid = blocks.len();
            let segments = new_present
                .into_iter()
                .map(|(nl, np)| BlockSegment {
                    local_seq_idx: nl,
                    seq_id: new_seq_ids[nl],
                    start_pos: np,
                    end_pos: np + 1,
                })
                .collect();
            blocks.push(CollinearBlock {
                block_id: bid,
                col_start: new_c_idx,
                col_len: 1,
                scale: new_scale,
                owner_node_id: new_owner,
                segments,
                fine_posteriors: vec![col.p_fine],
                bubble_posteriors: vec![col.p_bub],
            });
        }
    }

    let num_columns = blocks.iter().map(|b| b.col_len).sum();
    let new_desc = compute_leaf_descendants(&new_tree_nodes, new_m);

    // 7. Rebase surviving HorizonBubbleReports
    let mut horizon_bubbles = Vec::new();
    for hb in &comp.horizon_bubbles {
        let Some(new_split) = old_to_new_node.get(hb.split_node_id).copied().flatten() else {
            continue;
        };
        let Some(new_l) = old_to_new_node.get(hb.left_node_id).copied().flatten() else {
            continue;
        };
        let Some(new_r) = old_to_new_node.get(hb.right_node_id).copied().flatten() else {
            continue;
        };
        if new_l == new_r {
            continue;
        }

        let rebase_ivs = |ivs: &[(usize, usize, usize)]| -> Vec<(usize, usize, usize)> {
            let mut out = Vec::new();
            for &(old_loc, st, len) in ivs {
                let Some(new_loc) = old_to_new_leaf.get(old_loc).copied().flatten() else {
                    continue;
                };
                let off = seq_offsets[old_loc];
                let slen = seq_slice_lens[old_loc];
                let hi_bound = off + slen;
                let ov_lo = st.max(off);
                let ov_hi = (st + len).min(hi_bound);
                if ov_lo < ov_hi {
                    out.push((new_loc, ov_lo - off, ov_hi - ov_lo));
                }
            }
            out
        };

        let left_intervals = rebase_ivs(&hb.left_intervals);
        let right_intervals = rebase_ivs(&hb.right_intervals);
        if left_intervals.is_empty() || right_intervals.is_empty() {
            continue;
        }

        let l_sz = new_desc[new_l].iter().filter(|&&b| b).count();
        let r_sz = new_desc[new_r].iter().filter(|&&b| b).count();
        let mut h2 = hb.clone();
        h2.split_node_id = new_split;
        h2.left_node_id = new_l;
        h2.right_node_id = new_r;
        h2.left_clade_size = l_sz;
        h2.right_clade_size = r_sz;
        h2.left_intervals = left_intervals;
        h2.right_intervals = right_intervals;
        horizon_bubbles.push(h2);
    }

    // 8. Filter surviving FrameshiftEvents
    let surviving_name_idx: HashMap<&str, usize> = surviving_old_indices
        .iter()
        .map(|&old_loc| (seqs_by_id[&comp.seq_ids[old_loc]].name.as_str(), old_loc))
        .collect();
    let mut frameshifts_repaired = Vec::new();
    for ev in &comp.frameshifts_repaired {
        if let Some(&old_loc) = surviving_name_idx.get(ev.seq_id.as_str()) {
            let off = seq_offsets[old_loc];
            let slen = seq_slice_lens[old_loc];
            if !has_region || (ev.approx_codon_pos >= off && ev.approx_codon_pos < off + slen) {
                let mut e2 = ev.clone();
                e2.approx_codon_pos -= off;
                e2.nt_pos = e2.nt_pos.saturating_sub(off * 3);
                frameshifts_repaired.push(e2);
            }
        }
    }

    Ok(Some((
        ArchComponent {
            comp_id: new_comp_id,
            root_node_id: new_root_id,
            num_columns,
            num_global_fine_cols,
            num_clade_local_cols,
            num_lineage_private_cols,
            mean_core_posterior: if core_post_cnt > 0 {
                core_post_acc / (core_post_cnt as f64)
            } else {
                1.0
            },
            seq_ids: new_seq_ids,
            newick_tree: new_newick,
            tree_nodes: new_tree_nodes,
            blocks,
            horizon_bubbles,
            frameshifts_repaired,
        },
        out_seqs,
    )))
}

fn find_col_window_in_component(
    comp: &ArchComponent,
    seqs_by_id: &HashMap<usize, &ArchSeq>,
    seq_name: &str,
    start1: usize,
    end1: usize,
) -> Result<(usize, usize), String> {
    if start1 == 0 || end1 < start1 {
        return Err(format!("Invalid 1-based region {start1}-{end1}"));
    }
    let Some(old_loc) = comp
        .seq_ids
        .iter()
        .position(|sid| seqs_by_id[sid].name == seq_name)
    else {
        return Err(format!("Sequence '{seq_name}' not found in component {}", comp.comp_id));
    };
    let anchor = seqs_by_id[&comp.seq_ids[old_loc]];
    let q_lo = (start1 - 1).saturating_sub(anchor.orig_offset);
    let q_hi = end1.saturating_sub(anchor.orig_offset).min(anchor.seq_len);
    if q_lo >= q_hi {
        return Err(format!(
            "Region {seq_name}:{start1}-{end1} does not overlap stored span {}..={}",
            anchor.orig_offset + 1,
            anchor.orig_offset + anchor.seq_len
        ));
    }

    let mut col_min = usize::MAX;
    let mut col_max = 0usize;
    for blk in &comp.blocks {
        if let Some(seg) = blk.segments.iter().find(|s| s.local_seq_idx == old_loc) {
            if seg.end_pos > q_lo && seg.start_pos < q_hi {
                let off_lo = q_lo.saturating_sub(seg.start_pos);
                let off_hi = q_hi.min(seg.end_pos) - seg.start_pos;
                col_min = col_min.min(blk.col_start + off_lo);
                col_max = col_max.max(blk.col_start + off_hi);
            }
        }
    }
    if col_min >= col_max {
        return Err(format!("No blocks overlap region {seq_name}:{start1}-{end1}"));
    }

    // Expand window if anchor's intervals overlap a sister-clade Horizon Bubble so both sides are included
    for hb in &comp.horizon_bubbles {
        let mut hits_hb = false;
        for &(loc, st, len) in hb.left_intervals.iter().chain(hb.right_intervals.iter()) {
            if loc == old_loc && st + len > q_lo && st < q_hi {
                hits_hb = true;
                break;
            }
        }
        if hits_hb {
            for &(loc, st, len) in hb.left_intervals.iter().chain(hb.right_intervals.iter()) {
                for blk in &comp.blocks {
                    if let Some(seg) = blk.segments.iter().find(|s| s.local_seq_idx == loc) {
                        if seg.end_pos > st && seg.start_pos < st + len {
                            let off_lo = st.saturating_sub(seg.start_pos);
                            let off_hi = (st + len).min(seg.end_pos) - seg.start_pos;
                            col_min = col_min.min(blk.col_start + off_lo);
                            col_max = col_max.max(blk.col_start + off_hi);
                        }
                    }
                }
            }
        }
    }

    Ok((col_min, col_max))
}

pub fn subset_archive(
    arch: &HierarchicalArchive,
    spec: &SubsetSpec,
) -> Result<HierarchicalArchive, String> {
    let seqs_by_id: HashMap<usize, &ArchSeq> =
        arch.seqs.iter().map(|s| (s.seq_id, s)).collect();

    let region_comp_id = if let Some((ref r_seq, _, _)) = spec.region {
        let found = arch
            .seqs
            .iter()
            .find(|s| s.name == *r_seq)
            .ok_or_else(|| format!("Region sequence '{r_seq}' not found in archive"))?;
        if let Some(cid) = spec.component_id {
            if cid != found.comp_id {
                return Err(format!(
                    "Sequence '{r_seq}' belongs to component {}, not {cid}",
                    found.comp_id
                ));
            }
        }
        Some(found.comp_id)
    } else {
        spec.component_id
    };

    let mut out_seqs = Vec::new();
    let mut out_comps = Vec::new();
    let mut next_sid = 0usize;

    for comp in &arch.components {
        if let Some(cid) = region_comp_id {
            if comp.comp_id != cid {
                continue;
            }
        }
        let col_window = if let Some((ref r_seq, s1, e1)) = spec.region {
            Some(find_col_window_in_component(comp, &seqs_by_id, r_seq, s1, e1)?)
        } else {
            None
        };
        let new_cid = out_comps.len();
        if let Some((c2, mut s2)) = pack_subset_component(
            comp,
            &seqs_by_id,
            arch.mode,
            spec,
            col_window,
            new_cid,
            &mut next_sid,
        )? {
            out_comps.push(c2);
            out_seqs.append(&mut s2);
        }
    }

    if out_comps.is_empty() {
        return Err("Subset matched zero components/columns".to_string());
    }

    let (raw_null, prov_nats, net_gain) = if out_comps.len() == arch.components.len()
        && out_seqs.len() == arch.seqs.len()
        && spec.region.is_none()
        && !spec.global_core_only
    {
        (
            arch.raw_null_cohort_nats,
            arch.provenance_cohort_nats,
            arch.net_cohort_gain_nats,
        )
    } else {
        (0.0, 0.0, 0.0)
    };

    Ok(HierarchicalArchive {
        format_version: arch.format_version,
        mode: arch.mode,
        mito_tga_trp: arch.mito_tga_trp,
        raw_null_cohort_nats: raw_null,
        provenance_cohort_nats: prov_nats,
        net_cohort_gain_nats: net_gain,
        seqs: out_seqs,
        components: out_comps,
    })
}

/// Indexed SQLite subsetting (`2 -> 1`): queries `sequences`, `idx_segments_spatial`, and
/// `idx_blocks_cols` directly so whole-genome/large `.hdb` files load only matching components and loci.
#[cfg(not(target_arch = "wasm32"))]
pub fn subset_hdb_file(path: &Path, spec: &SubsetSpec) -> Result<HierarchicalArchive, String> {
    let db = SqliteDb::open(path, SQLITE_OPEN_READONLY)?;
    let meta = read_hdb_meta(&db)?;

    let format_version: u32 = meta
        .get("format_version")
        .ok_or_else(|| "Missing format_version in .hdb meta".to_string())?
        .parse()
        .map_err(|e| format!("{e}"))?;
    let mode = str_to_mode(
        meta.get("mode")
            .ok_or_else(|| "Missing mode in .hdb meta".to_string())?,
    )?;
    let mito_tga_trp: bool = meta
        .get("mito_tga_trp")
        .map(|s| s.parse())
        .transpose()
        .map_err(|e| format!("{e}"))?
        .unwrap_or(false);
    let raw_null: f64 = meta
        .get("raw_null_cohort_nats")
        .map(|s| s.parse())
        .transpose()
        .map_err(|e| format!("{e}"))?
        .unwrap_or(0.0);
    let prov_nats: f64 = meta
        .get("provenance_cohort_nats")
        .map(|s| s.parse())
        .transpose()
        .map_err(|e| format!("{e}"))?
        .unwrap_or(0.0);
    let net_gain: f64 = meta
        .get("net_cohort_gain_nats")
        .map(|s| s.parse())
        .transpose()
        .map_err(|e| format!("{e}"))?
        .unwrap_or(0.0);

    // 1. Resolve target components & optional spatial column window via SQLite indexes
    let mut region_window_by_comp: HashMap<usize, (usize, usize)> = HashMap::new();
    let target_cids: Vec<usize> = if let Some((ref r_seq, start1, end1)) = spec.region {
        if start1 == 0 || end1 < start1 {
            return Err(format!("Invalid 1-based region {start1}-{end1}"));
        }
        let st_r = db.prepare(
            "SELECT seq_id, comp_id, local_seq_idx, orig_offset, seq_len FROM sequences WHERE name = ?1 LIMIT 1;",
        )?;
        st_r.bind_text(1, r_seq)?;
        if !st_r.step_row()? {
            return Err(format!("Sequence '{r_seq}' not found in .hdb"));
        }
        let r_sid = st_r.col_usize(0);
        let r_cid = st_r.col_usize(1);
        let r_loc = st_r.col_usize(2);
        let r_off = st_r.col_usize(3);
        let r_len = st_r.col_usize(4);
        if let Some(cid) = spec.component_id {
            if cid != r_cid {
                return Err(format!("Sequence '{r_seq}' belongs to component {r_cid}, not {cid}"));
            }
        }
        let q_lo = (start1 - 1).saturating_sub(r_off);
        let q_hi = end1.saturating_sub(r_off).min(r_len);
        if q_lo >= q_hi {
            return Err(format!(
                "Region {r_seq}:{start1}-{end1} outside stored span {}..={}",
                r_off + 1,
                r_off + r_len
            ));
        }

        // Spatial index lookup!
        let st_sp = db.prepare(
            "SELECT b.col_start, b.col_len, s.start_pos, s.end_pos FROM block_segments s JOIN blocks b ON b.comp_id = s.comp_id AND b.block_id = s.block_id WHERE s.seq_id = ?1 AND s.end_pos > ?2 AND s.start_pos < ?3 ORDER BY b.col_start;",
        )?;
        st_sp.bind_i64(1, r_sid as i64)?;
        st_sp.bind_i64(2, q_lo as i64)?;
        st_sp.bind_i64(3, q_hi as i64)?;
        let mut col_min = usize::MAX;
        let mut col_max = 0usize;
        while st_sp.step_row()? {
            let cs = st_sp.col_usize(0);
            let _cl = st_sp.col_usize(1);
            let ss = st_sp.col_usize(2);
            let se = st_sp.col_usize(3);
            let off_lo = q_lo.saturating_sub(ss);
            let off_hi = q_hi.min(se) - ss;
            col_min = col_min.min(cs + off_lo);
            col_max = col_max.max(cs + off_hi);
        }

        // Expand window if anchor overlaps any Horizon Bubble in this component
        let st_hb = db.prepare(
            "SELECT left_intervals_str, right_intervals_str FROM horizon_bubbles WHERE comp_id = ?1;",
        )?;
        st_hb.bind_i64(1, r_cid as i64)?;
        let mut extra_ivs = Vec::new();
        while st_hb.step_row()? {
            let liv = parse_intervals(&st_hb.col_text(0))?;
            let riv = parse_intervals(&st_hb.col_text(1))?;
            let hits = liv
                .iter()
                .chain(riv.iter())
                .any(|&(loc, st, len)| loc == r_loc && st + len > q_lo && st < q_hi);
            if hits {
                extra_ivs.extend(liv);
                extra_ivs.extend(riv);
            }
        }
        if !extra_ivs.is_empty() {
            let st_loc_sid = db.prepare(
                "SELECT seq_id FROM sequences WHERE comp_id = ?1 AND local_seq_idx = ?2 LIMIT 1;",
            )?;
            let st_sp2 = db.prepare(
                "SELECT b.col_start, s.start_pos, s.end_pos FROM block_segments s JOIN blocks b ON b.comp_id = s.comp_id AND b.block_id = s.block_id WHERE s.seq_id = ?1 AND s.end_pos > ?2 AND s.start_pos < ?3;",
            )?;
            for (loc, st, len) in extra_ivs {
                st_loc_sid.reset()?;
                st_loc_sid.bind_i64(1, r_cid as i64)?;
                st_loc_sid.bind_i64(2, loc as i64)?;
                if st_loc_sid.step_row()? {
                    let partner_sid = st_loc_sid.col_usize(0);
                    st_sp2.reset()?;
                    st_sp2.bind_i64(1, partner_sid as i64)?;
                    st_sp2.bind_i64(2, st as i64)?;
                    st_sp2.bind_i64(3, (st + len) as i64)?;
                    while st_sp2.step_row()? {
                        let cs = st_sp2.col_usize(0);
                        let ss = st_sp2.col_usize(1);
                        let se = st_sp2.col_usize(2);
                        let off_lo = st.saturating_sub(ss);
                        let off_hi = (st + len).min(se) - ss;
                        col_min = col_min.min(cs + off_lo);
                        col_max = col_max.max(cs + off_hi);
                    }
                }
            }
        }

        if col_min >= col_max {
            return Err(format!("No blocks overlap region {r_seq}:{start1}-{end1}"));
        }
        region_window_by_comp.insert(r_cid, (col_min, col_max));
        vec![r_cid]
    } else if let Some(cid) = spec.component_id {
        vec![cid]
    } else {
        let mut cids = Vec::new();
        let st_cid = db.prepare("SELECT comp_id FROM components ORDER BY comp_id;")?;
        while st_cid.step_row()? {
            cids.push(st_cid.col_usize(0));
        }
        cids
    };

    let mut out_seqs = Vec::new();
    let mut out_comps = Vec::new();
    let mut next_sid = 0usize;
    let total_comps = {
        let st = db.prepare("SELECT COUNT(*) FROM components;")?;
        st.step_row()?;
        st.col_usize(0)
    };
    let total_seqs = {
        let st = db.prepare("SELECT COUNT(*) FROM sequences;")?;
        st.step_row()?;
        st.col_usize(0)
    };

    let st_cseqs = db.prepare(
        "SELECT seq_id, name, comp_id, local_seq_idx, orig_offset, seq_len, raw_seq FROM sequences WHERE comp_id = ?1 ORDER BY local_seq_idx;",
    )?;

    for cid in target_cids {
        let win = region_window_by_comp.get(&cid).copied();
        let comp = load_hdb_component(&db, cid, win)?;

        let mut comp_seqs = Vec::new();
        st_cseqs.reset()?;
        st_cseqs.bind_i64(1, cid as i64)?;
        while st_cseqs.step_row()? {
            comp_seqs.push(ArchSeq {
                seq_id: st_cseqs.col_usize(0),
                name: st_cseqs.col_text(1),
                comp_id: st_cseqs.col_usize(2),
                local_seq_idx: st_cseqs.col_usize(3),
                orig_offset: st_cseqs.col_usize(4),
                seq_len: st_cseqs.col_usize(5),
                raw_seq: st_cseqs.col_text(6),
            });
        }
        let seqs_by_id: HashMap<usize, &ArchSeq> =
            comp_seqs.iter().map(|s| (s.seq_id, s)).collect();

        let new_cid = out_comps.len();
        if let Some((c2, mut s2)) =
            pack_subset_component(&comp, &seqs_by_id, mode, spec, win, new_cid, &mut next_sid)?
        {
            out_comps.push(c2);
            out_seqs.append(&mut s2);
        }
    }

    if out_comps.is_empty() {
        return Err("Subset matched zero components/columns".to_string());
    }

    let keep_totals = out_comps.len() == total_comps
        && out_seqs.len() == total_seqs
        && spec.region.is_none()
        && !spec.global_core_only;

    Ok(HierarchicalArchive {
        format_version,
        mode,
        mito_tga_trp,
        raw_null_cohort_nats: if keep_totals { raw_null } else { 0.0 },
        provenance_cohort_nats: if keep_totals { prov_nats } else { 0.0 },
        net_cohort_gain_nats: if keep_totals { net_gain } else { 0.0 },
        seqs: out_seqs,
        components: out_comps,
    })
}

pub fn inspect_archive_summary(arch: &HierarchicalArchive) -> String {
    let total_cols: usize = arch.components.iter().map(|c| c.num_columns).sum();
    let total_blocks: usize = arch.components.iter().map(|c| c.blocks.len()).sum();
    let ratio = if total_blocks > 0 {
        (total_cols as f64) / (total_blocks as f64)
    } else {
        1.0
    };
    let mut s = format!(
        "HierarchicalArchive v{} | mode={} (mito_tga_trp={}) | seqs={} | components={} | cols={} -> RLE blocks={} ({:.2}x) | L0={:.1} -> L_prov={:.1} (net {:+.1} nats)\n",
        arch.format_version,
        mode_to_str(arch.mode),
        arch.mito_tga_trp,
        arch.seqs.len(),
        arch.components.len(),
        total_cols,
        total_blocks,
        ratio,
        arch.raw_null_cohort_nats,
        arch.provenance_cohort_nats,
        arch.net_cohort_gain_nats,
    );
    for comp in &arch.components {
        let clades: Vec<&str> = comp
            .tree_nodes
            .iter()
            .filter(|n| !n.is_leaf)
            .map(|n| n.label.as_str())
            .collect();
        s.push_str(&format!(
            "  Component #{} (root_node={}): ntax={} | cols={} [GlobalFine={}, CladeLocal={}, Private={}] | blocks={} | bubbles={} | frameshifts={} | clades=[{}]\n",
            comp.comp_id,
            comp.root_node_id,
            comp.seq_ids.len(),
            comp.num_columns,
            comp.num_global_fine_cols,
            comp.num_clade_local_cols,
            comp.num_lineage_private_cols,
            comp.blocks.len(),
            comp.horizon_bubbles.len(),
            comp.frameshifts_repaired.len(),
            clades.join(", "),
        ));
    }
    s
}
