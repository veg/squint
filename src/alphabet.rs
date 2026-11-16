use serde::{Deserialize, Serialize};

/// Alphabetical 20 amino acid order (`ACDEFGHIKLMNPQRSTVWY`).
pub const AA_CHARS: [u8; 20] = [
    b'A', b'C', b'D', b'E', b'F', b'G', b'H', b'I', b'K', b'L',
    b'M', b'N', b'P', b'Q', b'R', b'S', b'T', b'V', b'W', b'Y',
];

pub const NT_CHARS: [u8; 4] = [b'A', b'C', b'G', b'T'];

/// Six biophysical classes (Dayhoff / PRIME-inspired):
/// 0: Aliphatic/Hydrophobic (A, V, I, L, M)
/// 1: Aromatic (F, Y, W)
/// 2: Polar neutral (S, T, N, Q)
/// 3: Basic / Positively charged (K, R, H)
/// 4: Acidic / Negatively charged (D, E)
/// 5: Structural / Special (G, P, C)
pub const NUM_BIOPHYS_CLASSES: usize = 6;

pub const AA_TO_BIOPHYS: [usize; 20] = [
    0, // 0  A: Hydrophobic
    5, // 1  C: Special
    4, // 2  D: Acidic
    4, // 3  E: Acidic
    1, // 4  F: Aromatic
    5, // 5  G: Special
    3, // 6  H: Basic
    0, // 7  I: Hydrophobic
    3, // 8  K: Basic
    0, // 9  L: Hydrophobic
    0, // 10 M: Hydrophobic
    2, // 11 N: Polar
    5, // 12 P: Special
    2, // 13 Q: Polar
    3, // 14 R: Basic
    2, // 15 S: Polar
    2, // 16 T: Polar
    0, // 17 V: Hydrophobic
    1, // 18 W: Aromatic
    1, // 19 Y: Aromatic
];

pub const BIOPHYS_CLASS_NAMES: [&str; 6] = [
    "Hydrophobic(AVILM)",
    "Aromatic(FYW)",
    "Polar(STNQ)",
    "Basic(KRH)",
    "Acidic(DE)",
    "Special(GPC)",
];

/// Standardized biophysical property scales per amino acid (`ACDEFGHIKLMNPQRSTVWY`):
/// [Kyte-Doolittle hydropathy, Grantham volume/100, net charge at pH 7.4]
pub const AA_PROPERTIES: [[f64; 3]; 20] = [
    [ 1.8, 0.31,  0.0], // A
    [ 2.5, 0.55,  0.0], // C
    [-3.5, 0.54, -1.0], // D
    [-3.5, 0.83, -1.0], // E
    [ 2.8, 1.32,  0.0], // F
    [-0.4, 0.03,  0.0], // G
    [-3.2, 0.96,  0.2], // H
    [ 4.5, 1.11,  0.0], // I
    [-3.9, 1.19,  1.0], // K
    [ 3.8, 1.11,  0.0], // L
    [ 1.9, 1.05,  0.0], // M
    [-3.5, 0.56,  0.0], // N
    [-1.6, 0.32,  0.0], // P
    [-3.5, 0.85,  0.0], // Q
    [-4.5, 1.24,  1.0], // R
    [-0.8, 0.32,  0.0], // S
    [-0.7, 0.61,  0.0], // T
    [ 4.2, 0.84,  0.0], // V
    [-0.9, 1.70,  0.0], // W
    [-1.3, 1.36,  0.0], // Y
];

/// Alignment-free experimental properties of isolated amino acids (`ACDEFGHIKLMNPQRSTVWY`):
/// [Kyte-Doolittle hydropathy, Grantham polarity, Grantham volume / 100, net formal charge at pH 7.4, aromatic pi-ring electron count]
pub const AA_INTRINSIC_PROPS: [[f64; 5]; 20] = [
    [ 1.8,  8.1, 0.31,  0.0, 0.0], // A
    [ 2.5,  5.5, 0.55,  0.0, 0.0], // C
    [-3.5, 13.0, 0.54, -1.0, 0.0], // D
    [-3.5, 12.3, 0.83, -1.0, 0.0], // E
    [ 2.8,  5.2, 1.32,  0.0, 6.0], // F
    [-0.4,  9.0, 0.03,  0.0, 0.0], // G
    [-3.2, 10.4, 0.96,  0.2, 5.0], // H
    [ 4.5,  5.2, 1.11,  0.0, 0.0], // I
    [-3.9, 11.3, 1.19,  1.0, 0.0], // K
    [ 3.8,  4.9, 1.11,  0.0, 0.0], // L
    [ 1.9,  5.7, 1.05,  0.0, 0.0], // M
    [-3.5, 11.6, 0.56,  0.0, 0.0], // N
    [-1.6,  8.0, 0.32,  0.0, 0.0], // P
    [-3.5, 10.5, 0.85,  0.0, 0.0], // Q
    [-4.5, 10.5, 1.24,  1.0, 0.0], // R
    [-0.8,  9.2, 0.32,  0.0, 0.0], // S
    [-0.7,  8.6, 0.61,  0.0, 0.0], // T
    [ 4.2,  5.9, 0.84,  0.0, 0.0], // V
    [-0.9,  5.4, 1.70,  0.0, 9.0], // W
    [-1.3,  6.2, 1.36,  0.0, 6.0], // Y
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SeqMode {
    Codon,
    Protein,
    Dna,
}

#[inline]
pub fn byte_to_nt_mask(b: u8) -> Option<u8> {
    match b.to_ascii_uppercase() {
        b'A' => Some(0b0001),
        b'C' => Some(0b0010),
        b'G' => Some(0b0100),
        b'T' | b'U' => Some(0b1000),
        b'R' => Some(0b0101), // A | G
        b'Y' => Some(0b1010), // C | T
        b'S' => Some(0b0110), // C | G
        b'W' => Some(0b1001), // A | T
        b'K' => Some(0b1100), // G | T
        b'M' => Some(0b0011), // A | C
        b'B' => Some(0b1110), // C | G | T
        b'D' => Some(0b1101), // A | G | T
        b'H' => Some(0b1011), // A | C | T
        b'V' => Some(0b0111), // A | C | G
        b'N' => Some(0b1111), // A | C | G | T
        _ => None,
    }
}

#[inline]
pub fn byte_to_nt(b: u8) -> Option<u8> {
    let m = byte_to_nt_mask(b)?;
    Some(m.trailing_zeros() as u8)
}

#[inline]
pub fn byte_to_aa_mask(b: u8) -> Option<u32> {
    match b.to_ascii_uppercase() {
        b'A' => Some(1u32 << 0),
        b'C' | b'U' => Some(1u32 << 1),
        b'D' => Some(1u32 << 2),
        b'E' => Some(1u32 << 3),
        b'F' => Some(1u32 << 4),
        b'G' => Some(1u32 << 5),
        b'H' => Some(1u32 << 6),
        b'I' => Some(1u32 << 7),
        b'K' | b'O' => Some(1u32 << 8),
        b'L' => Some(1u32 << 9),
        b'M' => Some(1u32 << 10),
        b'N' => Some(1u32 << 11),
        b'P' => Some(1u32 << 12),
        b'Q' => Some(1u32 << 13),
        b'R' => Some(1u32 << 14),
        b'S' => Some(1u32 << 15),
        b'T' => Some(1u32 << 16),
        b'V' => Some(1u32 << 17),
        b'W' => Some(1u32 << 18),
        b'Y' => Some(1u32 << 19),
        b'B' => Some((1u32 << 2) | (1u32 << 11)), // D | N
        b'Z' => Some((1u32 << 3) | (1u32 << 13)), // E | Q
        b'J' => Some((1u32 << 7) | (1u32 << 9)),  // I | L
        b'X' => Some((1u32 << 20) - 1),           // Any AA
        _ => None,
    }
}

#[inline]
pub fn byte_to_aa(b: u8) -> Option<u8> {
    byte_to_aa_mask(b).map(|m| m.trailing_zeros() as u8)
}

/// Universal genetic code table over 64 triplets (16*n0 + 4*n1 + n2, A=0, C=1, G=2, T=3)
/// mapped into `ACDEFGHIKLMNPQRSTVWY` indices (None for TAA=48, TAG=50, TGA=56).
pub const TRIPLET_TO_AA: [Option<u8>; 64] = [
    // AAA(K=8), AAC(N=11), AAG(K=8), AAT(N=11)
    Some(8), Some(11), Some(8), Some(11),
    // ACA(T=16), ACC(T=16), ACG(T=16), ACT(T=16)
    Some(16), Some(16), Some(16), Some(16),
    // AGA(R=14), AGC(S=15), AGG(R=14), AGT(S=15)
    Some(14), Some(15), Some(14), Some(15),
    // ATA(I=7), ATC(I=7), ATG(M=10), ATT(I=7)
    Some(7), Some(7), Some(10), Some(7),
    // CAA(Q=13), CAC(H=6), CAG(Q=13), CAT(H=6)
    Some(13), Some(6), Some(13), Some(6),
    // CCA..CCT (P=12)
    Some(12), Some(12), Some(12), Some(12),
    // CGA..CGT (R=14)
    Some(14), Some(14), Some(14), Some(14),
    // CTA..CTT (L=9)
    Some(9), Some(9), Some(9), Some(9),
    // GAA(E=3), GAC(D=2), GAG(E=3), GAT(D=2)
    Some(3), Some(2), Some(3), Some(2),
    // GCA..GCT (A=0)
    Some(0), Some(0), Some(0), Some(0),
    // GGA..GGT (G=5)
    Some(5), Some(5), Some(5), Some(5),
    // GTA..GTT (V=17)
    Some(17), Some(17), Some(17), Some(17),
    // TAA(*), TAC(Y=19), TAG(*), TAT(Y=19)
    None, Some(19), None, Some(19),
    // TCA..TCT (S=15)
    Some(15), Some(15), Some(15), Some(15),
    // TGA(*), TGC(C=1), TGG(W=18), TGT(C=1)
    None, Some(1), Some(18), Some(1),
    // TTA(L=9), TTC(F=4), TTG(L=9), TTT(F=4)
    Some(9), Some(4), Some(9), Some(4),
];

/// Precomputed genetic code lookup tables for sense codons (61 universal or 62 mitochondrial TGA=Trp).
#[derive(Debug, Clone)]
pub struct GeneticCode {
    pub num_sense: usize,
    pub mito_tga_trp: bool,
    pub triplet_to_sense: [Option<u8>; 64],
    pub sense_to_triplet: Vec<u8>,
    pub sense_to_aa: Vec<u8>,
    pub aa_to_senses: Vec<Vec<u8>>,
    pub nt_diffs: Vec<Vec<u8>>,
    pub is_single_transition: Vec<Vec<bool>>,
}

impl GeneticCode {
    fn from_table(table: &[Option<u8>; 64], mito_tga_trp: bool) -> Self {
        let mut triplet_to_sense = [None; 64];
        let mut sense_to_triplet = Vec::with_capacity(62);
        let mut sense_to_aa = Vec::with_capacity(62);
        let mut aa_to_senses = vec![Vec::new(); 20];

        let mut s_idx = 0u8;
        for t in 0..64u8 {
            if let Some(aa) = table[t as usize] {
                triplet_to_sense[t as usize] = Some(s_idx);
                sense_to_triplet.push(t);
                sense_to_aa.push(aa);
                aa_to_senses[aa as usize].push(s_idx);
                s_idx += 1;
            }
        }
        let num_sense = s_idx as usize;

        let mut nt_diffs = vec![vec![0u8; num_sense]; num_sense];
        let mut is_single_transition = vec![vec![false; num_sense]; num_sense];
        for c1 in 0..num_sense {
            let t1 = sense_to_triplet[c1];
            let b1 = [t1 >> 4, (t1 >> 2) & 3, t1 & 3];
            for c2 in 0..num_sense {
                let t2 = sense_to_triplet[c2];
                let b2 = [t2 >> 4, (t2 >> 2) & 3, t2 & 3];
                let mut diffs = 0u8;
                let mut ti = false;
                for pos in 0..3 {
                    if b1[pos] != b2[pos] {
                        diffs += 1;
                        if (b1[pos] ^ b2[pos]) == 2 {
                            ti = true;
                        }
                    }
                }
                nt_diffs[c1][c2] = diffs;
                is_single_transition[c1][c2] = diffs == 1 && ti;
            }
        }

        Self {
            num_sense,
            mito_tga_trp,
            triplet_to_sense,
            sense_to_triplet,
            sense_to_aa,
            aa_to_senses,
            nt_diffs,
            is_single_transition,
        }
    }

    pub fn standard() -> Self {
        let gc = Self::from_table(&TRIPLET_TO_AA, false);
        assert_eq!(gc.num_sense, 61);
        gc
    }

    pub fn mitochondrial_tga_trp() -> Self {
        let mut tbl = TRIPLET_TO_AA;
        tbl[56] = Some(18); // TGA -> Trp (W)
        let gc = Self::from_table(&tbl, true);
        assert_eq!(gc.num_sense, 62);
        gc
    }

    /// Unsupervised cohort-level detection of mitochondrial `TGA = Trp` coding table:
    /// Activates iff sequences are nucleotide-alphabet, internal `TAA/TAG` stops are near-zero (<0.4%),
    /// and internal in-frame `TGA` triplets are pervasive (>=0.8%).
    pub fn for_cohort(seqs: &[RawSeq]) -> Self {
        let mut total_alpha = 0usize;
        let mut non_dna_aa = 0usize;
        for s in seqs {
            for &b in &s.raw_bytes {
                let u = b.to_ascii_uppercase();
                if u.is_ascii_alphabetic() {
                    total_alpha += 1;
                    if !matches!(
                        u,
                        b'A' | b'C' | b'G' | b'T' | b'U' | b'N' | b'R' | b'Y' | b'S' | b'W' | b'K' | b'M'
                    ) {
                        non_dna_aa += 1;
                    }
                }
            }
        }
        if total_alpha > 0 && (non_dna_aa as f64 / total_alpha as f64) > 0.03 {
            return Self::standard();
        }

        let mut total_codons = 0usize;
        let mut taa_tag_stops = 0usize;
        let mut tga_codons = 0usize;
        for s in seqs {
            let masks: Vec<u8> = s.raw_bytes.iter().filter_map(|&b| byte_to_nt_mask(b)).collect();
            let n_cod = masks.len() / 3;
            if n_cod > 1 {
                for c in 0..(n_cod - 1) {
                    let m0 = masks[3 * c];
                    let m1 = masks[3 * c + 1];
                    let m2 = masks[3 * c + 2];
                    if m0.count_ones() == 1 && m1.count_ones() == 1 && m2.count_ones() == 1 {
                        let trip = ((m0.trailing_zeros() as u8) << 4)
                            | ((m1.trailing_zeros() as u8) << 2)
                            | (m2.trailing_zeros() as u8);
                        total_codons += 1;
                        if trip == 48 || trip == 50 {
                            taa_tag_stops += 1;
                        } else if trip == 56 {
                            tga_codons += 1;
                        }
                    }
                }
            }
        }
        if total_codons >= 30 {
            let taa_tag_frac = taa_tag_stops as f64 / total_codons as f64;
            let tga_frac = tga_codons as f64 / total_codons as f64;
            if taa_tag_frac < 0.004 && tga_frac >= 0.008 {
                return Self::mitochondrial_tga_trp();
            }
        }
        Self::standard()
    }

    /// Resolve a 3-position IUPAC bitmask `(m0, m1, m2)` into a `u64` bitmask of compatible sense codons.
    #[inline]
    pub fn resolve_masks_to_sense_mask(&self, m0: u8, m1: u8, m2: u8) -> u64 {
        if m0 == 0 || m1 == 0 || m2 == 0 {
            return 0;
        }
        if m0 == 0xF && m1 == 0xF && m2 == 0xF {
            return (1u64 << self.num_sense) - 1;
        }
        if m0.count_ones() == 1 && m1.count_ones() == 1 && m2.count_ones() == 1 {
            let trip = ((m0.trailing_zeros() as u8) << 4)
                | ((m1.trailing_zeros() as u8) << 2)
                | (m2.trailing_zeros() as u8);
            return match self.triplet_to_sense[trip as usize] {
                Some(s) => 1u64 << s,
                None => 0,
            };
        }
        let mut smask = 0u64;
        for x0 in 0..4u8 {
            if (m0 & (1 << x0)) == 0 {
                continue;
            }
            for x1 in 0..4u8 {
                if (m1 & (1 << x1)) == 0 {
                    continue;
                }
                for x2 in 0..4u8 {
                    if (m2 & (1 << x2)) == 0 {
                        continue;
                    }
                    let trip = (x0 << 4) | (x1 << 2) | x2;
                    if let Some(s) = self.triplet_to_sense[trip as usize] {
                        smask |= 1u64 << s;
                    }
                }
            }
        }
        smask
    }

    /// Map a sense-codon `u64` bitmask to a `u32` bitmask over the 20 amino acids.
    #[inline]
    pub fn sense_mask_to_aa_mask(&self, mut sense_mask: u64) -> u32 {
        let mut aa_m = 0u32;
        while sense_mask != 0 {
            let s = sense_mask.trailing_zeros() as usize;
            sense_mask &= sense_mask - 1;
            aa_m |= 1u32 << self.sense_to_aa[s];
        }
        aa_m
    }

    /// Resolve a 3-position IUPAC bitmask `(m0, m1, m2)` to a compatible sense codon if one exists.
    #[inline]
    pub fn resolve_masks_to_sense(&self, m0: u8, m1: u8, m2: u8) -> Option<u8> {
        let smask = self.resolve_masks_to_sense_mask(m0, m1, m2);
        if smask != 0 {
            Some(smask.trailing_zeros() as u8)
        } else {
            None
        }
    }

    pub fn codon_to_string(&self, sense: u8) -> String {
        let t = self.sense_to_triplet[sense as usize];
        let b0 = NT_CHARS[(t >> 4) as usize] as char;
        let b1 = NT_CHARS[((t >> 2) & 3) as usize] as char;
        let b2 = NT_CHARS[(t & 3) as usize] as char;
        format!("{}{}{}", b0, b1, b2)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawSeq {
    pub id: String,
    pub raw_bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct EncodedSeq {
    pub id: String,
    pub mode: SeqMode,
    pub alpha_size: usize,
    pub tokens: Vec<u8>,
    pub ambig_masks: Vec<u64>,
    pub sym_proj: Vec<u8>,
    pub has_ambig: bool,
    pub non_ambig_len: usize,
    pub nt_tokens: Vec<u8>,
    pub nt_masks: Vec<u8>,
    pub nt_raw_chars: Vec<u8>,
    pub token_strings: Vec<String>,
    pub has_unambiguous_internal_stop: bool,
}

impl EncodedSeq {
    pub fn alphabet_size(&self) -> usize {
        self.alpha_size
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn from_raw(raw: &RawSeq, mode: SeqMode, gc: &GeneticCode) -> Self {
        match mode {
            SeqMode::Protein => {
                let mut tokens = Vec::with_capacity(raw.raw_bytes.len());
                let mut ambig_masks = Vec::with_capacity(raw.raw_bytes.len());
                let mut sym_proj = Vec::with_capacity(raw.raw_bytes.len());
                let mut token_strings = Vec::with_capacity(raw.raw_bytes.len());
                let mut has_ambig = false;
                let mut non_ambig_len = 0usize;
                for &b in &raw.raw_bytes {
                    if b == b'*' {
                        continue;
                    }
                    if let Some(aa_m) = byte_to_aa_mask(b) {
                        let tok = aa_m.trailing_zeros() as u8;
                        tokens.push(tok);
                        ambig_masks.push(aa_m as u64);
                        if aa_m.count_ones() == 1 {
                            sym_proj.push(tok);
                            non_ambig_len += 1;
                        } else {
                            sym_proj.push(255);
                            has_ambig = true;
                        }
                        token_strings.push((b.to_ascii_uppercase() as char).to_string());
                    }
                }
                Self {
                    id: raw.id.clone(),
                    mode,
                    alpha_size: 20,
                    tokens,
                    ambig_masks,
                    sym_proj,
                    has_ambig,
                    non_ambig_len,
                    nt_tokens: Vec::new(),
                    nt_masks: Vec::new(),
                    nt_raw_chars: Vec::new(),
                    token_strings,
                    has_unambiguous_internal_stop: false,
                }
            }
            SeqMode::Dna => {
                let mut tokens = Vec::with_capacity(raw.raw_bytes.len());
                let mut ambig_masks = Vec::with_capacity(raw.raw_bytes.len());
                let mut sym_proj = Vec::with_capacity(raw.raw_bytes.len());
                let mut nt_masks = Vec::with_capacity(raw.raw_bytes.len());
                let mut nt_raw_chars = Vec::with_capacity(raw.raw_bytes.len());
                let mut token_strings = Vec::with_capacity(raw.raw_bytes.len());
                let mut has_ambig = false;
                let mut non_ambig_len = 0usize;
                for &b in &raw.raw_bytes {
                    if let Some(m) = byte_to_nt_mask(b) {
                        let nt = m.trailing_zeros() as u8;
                        let ch = b.to_ascii_uppercase();
                        tokens.push(nt);
                        ambig_masks.push(m as u64);
                        if m.count_ones() == 1 {
                            sym_proj.push(nt);
                            non_ambig_len += 1;
                        } else {
                            sym_proj.push(255);
                            has_ambig = true;
                        }
                        nt_masks.push(m);
                        nt_raw_chars.push(ch);
                        token_strings.push((ch as char).to_string());
                    }
                }
                let nt_tokens = tokens.clone();
                Self {
                    id: raw.id.clone(),
                    mode,
                    alpha_size: 4,
                    tokens,
                    ambig_masks,
                    sym_proj,
                    has_ambig,
                    non_ambig_len,
                    nt_tokens,
                    nt_masks,
                    nt_raw_chars,
                    token_strings,
                    has_unambiguous_internal_stop: false,
                }
            }
            SeqMode::Codon => {
                let mut nt_tokens = Vec::with_capacity(raw.raw_bytes.len());
                let mut nt_masks = Vec::with_capacity(raw.raw_bytes.len());
                let mut nt_raw_chars = Vec::with_capacity(raw.raw_bytes.len());
                for &b in &raw.raw_bytes {
                    if let Some(m) = byte_to_nt_mask(b) {
                        nt_tokens.push(m.trailing_zeros() as u8);
                        nt_masks.push(m);
                        nt_raw_chars.push(b.to_ascii_uppercase());
                    }
                }
                if nt_tokens.len() >= 3 && nt_tokens.len() % 3 == 0 {
                    let n = nt_tokens.len();
                    if gc
                        .resolve_masks_to_sense_mask(nt_masks[n - 3], nt_masks[n - 2], nt_masks[n - 1])
                        == 0
                    {
                        nt_tokens.truncate(n - 3);
                        nt_masks.truncate(n - 3);
                        nt_raw_chars.truncate(n - 3);
                    }
                }
                let n_codons = nt_tokens.len() / 3;
                let mut tokens = Vec::with_capacity(n_codons);
                let mut ambig_masks = Vec::with_capacity(n_codons);
                let mut sym_proj = Vec::with_capacity(n_codons);
                let mut token_strings = Vec::with_capacity(n_codons);
                let mut has_unambiguous_internal_stop = false;
                let mut has_ambig = false;
                let mut non_ambig_len = 0usize;
                for c in 0..n_codons {
                    let m0 = nt_masks[3 * c];
                    let m1 = nt_masks[3 * c + 1];
                    let m2 = nt_masks[3 * c + 2];
                    let s_raw = String::from_utf8(vec![
                        nt_raw_chars[3 * c],
                        nt_raw_chars[3 * c + 1],
                        nt_raw_chars[3 * c + 2],
                    ])
                    .unwrap();
                    let smask = gc.resolve_masks_to_sense_mask(m0, m1, m2);
                    if smask != 0 {
                        let s = smask.trailing_zeros() as u8;
                        tokens.push(s);
                        ambig_masks.push(smask);
                        if smask.count_ones() > 1 {
                            has_ambig = true;
                        }
                        let aa_m = gc.sense_mask_to_aa_mask(smask);
                        if aa_m.count_ones() == 1 {
                            sym_proj.push(aa_m.trailing_zeros() as u8);
                            non_ambig_len += 1;
                        } else {
                            sym_proj.push(255);
                        }
                        token_strings.push(s_raw);
                    } else {
                        has_unambiguous_internal_stop = true;
                        let trip = (nt_tokens[3 * c] << 4)
                            | (nt_tokens[3 * c + 1] << 2)
                            | nt_tokens[3 * c + 2];
                        let fallback_sense = if trip == 56 {
                            gc.triplet_to_sense[58].unwrap()
                        } else {
                            gc.triplet_to_sense[49].unwrap()
                        };
                        tokens.push(fallback_sense);
                        ambig_masks.push(1u64 << fallback_sense);
                        sym_proj.push(gc.sense_to_aa[fallback_sense as usize]);
                        non_ambig_len += 1;
                        token_strings.push(s_raw);
                    }
                }
                Self {
                    id: raw.id.clone(),
                    mode,
                    alpha_size: gc.num_sense,
                    tokens,
                    ambig_masks,
                    sym_proj,
                    has_ambig,
                    non_ambig_len,
                    nt_tokens,
                    nt_masks,
                    nt_raw_chars,
                    token_strings,
                    has_unambiguous_internal_stop,
                }
            }
        }
    }
}

pub fn detect_seq_mode(seqs: &[RawSeq], gc: &GeneticCode) -> SeqMode {
    let mut total_alpha = 0usize;
    let mut non_dna_aa = 0usize;
    for s in seqs {
        for &b in &s.raw_bytes {
            let u = b.to_ascii_uppercase();
            if u.is_ascii_alphabetic() {
                total_alpha += 1;
                if !matches!(
                    u,
                    b'A' | b'C' | b'G' | b'T' | b'U' | b'N' | b'R' | b'Y' | b'S' | b'W' | b'K' | b'M'
                ) {
                    non_dna_aa += 1;
                } else if matches!(u, b'E' | b'F' | b'I' | b'L' | b'P' | b'Q') {
                    non_dna_aa += 1;
                }
            }
        }
    }
    if total_alpha > 0 && (non_dna_aa as f64 / total_alpha as f64) > 0.03 {
        return SeqMode::Protein;
    }

    let mut total_codons = 0usize;
    let mut internal_stops = 0usize;
    let mut div3_count = 0usize;
    for s in seqs {
        let masks: Vec<u8> = s.raw_bytes.iter().filter_map(|&b| byte_to_nt_mask(b)).collect();
        if masks.len() >= 30 && masks.len() % 3 == 0 {
            div3_count += 1;
        }
        let n_cod = masks.len() / 3;
        if n_cod > 1 {
            // Check minimum-stop reading frame for partial/5'-clipped amplicons as well as frame 0
            let mut best_stops = usize::MAX;
            let mut best_cod = 0usize;
            for phase in 0..3usize {
                if masks.len() < phase + 6 {
                    continue;
                }
                let nc = (masks.len() - phase) / 3;
                let mut st = 0usize;
                let mut informative_cod = 0usize;
                for c in 0..(nc - 1) {
                    let idx = phase + 3 * c;
                    let n_exact = (masks[idx].count_ones() == 1) as usize
                        + (masks[idx + 1].count_ones() == 1) as usize
                        + (masks[idx + 2].count_ones() == 1) as usize;
                    if n_exact >= 2 {
                        informative_cod += 1;
                        if gc
                            .resolve_masks_to_sense_mask(masks[idx], masks[idx + 1], masks[idx + 2])
                            == 0
                        {
                            st += 1;
                        }
                    }
                }
                let eff_st = if phase == 0 { st } else { st + 1 };
                if informative_cod > 0 && eff_st < best_stops {
                    best_stops = st;
                    best_cod = informative_cod;
                }
            }
            if best_cod > 0 {
                total_codons += best_cod;
                internal_stops += best_stops;
            }
        }
    }
    if total_codons >= 20 {
        let stop_frac = internal_stops as f64 / total_codons as f64;
        let div3_frac = div3_count as f64 / seqs.len().max(1) as f64;
        if stop_frac < 0.018 && (div3_frac >= 0.5 || stop_frac < 0.005) {
            return SeqMode::Codon;
        }
    }
    SeqMode::Dna
}

pub fn parse_sequences_str(input: &str) -> (Vec<RawSeq>, Vec<(String, Vec<u8>)>) {
    let trimmed = input.trim_start();
    if trimmed.len() >= 6 && trimmed[..6].eq_ignore_ascii_case("#NEXUS") {
        parse_nexus_str(trimmed)
    } else {
        parse_fasta_str(trimmed)
    }
}

fn parse_fasta_str(input: &str) -> (Vec<RawSeq>, Vec<(String, Vec<u8>)>) {
    let mut raw_seqs = Vec::new();
    let mut aligned_seqs = Vec::new();
    let mut cur_id: Option<String> = None;
    let mut cur_aligned: Vec<u8> = Vec::new();

    for line in input.split(|c| c == '\n' || c == '\r') {
        let line = line.trim();
        if line.is_empty() || line.starts_with(';') {
            continue;
        }
        if let Some(rest) = line.strip_prefix('>') {
            if let Some(id) = cur_id.take() {
                let raw_bytes = cur_aligned
                    .iter()
                    .copied()
                    .filter(|&b| b != b'-' && b != b'.' && !b.is_ascii_whitespace())
                    .collect();
                raw_seqs.push(RawSeq { id: id.clone(), raw_bytes });
                aligned_seqs.push((id, std::mem::take(&mut cur_aligned)));
            }
            let id = rest.split_whitespace().next().unwrap_or("seq").to_string();
            cur_id = Some(id);
        } else if cur_id.is_some() {
            for b in line.bytes() {
                if !b.is_ascii_whitespace() {
                    cur_aligned.push(b);
                }
            }
        }
    }
    if let Some(id) = cur_id.take() {
        let raw_bytes = cur_aligned
            .iter()
            .copied()
            .filter(|&b| b != b'-' && b != b'.' && !b.is_ascii_whitespace())
            .collect();
        raw_seqs.push(RawSeq { id: id.clone(), raw_bytes });
        aligned_seqs.push((id, cur_aligned));
    }
    (raw_seqs, aligned_seqs)
}

fn parse_nexus_str(input: &str) -> (Vec<RawSeq>, Vec<(String, Vec<u8>)>) {
    let mut in_matrix = false;
    let mut order: Vec<String> = Vec::new();
    let mut map: std::collections::HashMap<String, Vec<u8>> = std::collections::HashMap::new();

    for raw_line in input.split(|c| c == '\n' || c == '\r') {
        let mut line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let upper = line.to_ascii_uppercase();
        if !in_matrix {
            if upper.starts_with("MATRIX") {
                in_matrix = true;
            }
            continue;
        }
        if line.starts_with(';') || upper.starts_with("END;") || upper == "END" {
            break;
        }
        let semi = line.ends_with(';');
        if semi {
            line = line[..line.len() - 1].trim();
        }
        if !line.is_empty() && !line.starts_with('[') {
            let (id, seq_part) = if let Some(rest) = line.strip_prefix('\'') {
                if let Some(end_q) = rest.find('\'') {
                    (&rest[..end_q], rest[end_q + 1..].trim())
                } else {
                    continue;
                }
            } else {
                let mut parts = line.splitn(2, char::is_whitespace);
                let id = parts.next().unwrap_or("");
                let seq_part = parts.next().unwrap_or("").trim();
                (id, seq_part)
            };
            if !id.is_empty() && !seq_part.is_empty() {
                let entry = map.entry(id.to_string()).or_insert_with(|| {
                    order.push(id.to_string());
                    Vec::new()
                });
                for b in seq_part.bytes() {
                    if !b.is_ascii_whitespace() {
                        entry.push(b);
                    }
                }
            }
        }
        if semi {
            break;
        }
    }

    let mut raw_seqs = Vec::with_capacity(order.len());
    let mut aligned_seqs = Vec::with_capacity(order.len());
    for id in order {
        if let Some(aln) = map.remove(&id) {
            let raw_bytes = aln
                .iter()
                .copied()
                .filter(|&b| b != b'-' && b != b'.' && b != b'?' && !b.is_ascii_whitespace())
                .collect();
            raw_seqs.push(RawSeq { id: id.clone(), raw_bytes });
            aligned_seqs.push((id, aln));
        }
    }
    (raw_seqs, aligned_seqs)
}

pub fn extract_reference_homologous_pairs(
    aln_a: &[u8],
    aln_b: &[u8],
    step_bytes: usize,
) -> Vec<(usize, usize)> {
    assert_eq!(aln_a.len(), aln_b.len());
    let n_cols = aln_a.len() / step_bytes;
    let mut pairs = Vec::new();
    let mut idx_a = 0usize;
    let mut idx_b = 0usize;
    for col in 0..n_cols {
        let slice_a = &aln_a[col * step_bytes..(col + 1) * step_bytes];
        let slice_b = &aln_b[col * step_bytes..(col + 1) * step_bytes];
        let has_a = slice_a.iter().all(|&b| b != b'-' && b != b'.' && b != b'?');
        let has_b = slice_b.iter().all(|&b| b != b'-' && b != b'.' && b != b'?');
        if has_a && has_b {
            pairs.push((idx_a, idx_b));
        }
        if has_a {
            idx_a += 1;
        }
        if has_b {
            idx_b += 1;
        }
    }
    pairs
}
