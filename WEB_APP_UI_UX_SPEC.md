# Web Application Design System & UI/UX Specification for SQUINT

**Status:** Production Blueprint (`v1.0`)
**Target Repository:** `https://github.com/veg/squint` (`docs/index.html`)
**Adapted From:** `/Users/sergei/Projects/trees/WEB_APP_UI_UX_SPEC.md`

---

## 1. Core Design Philosophy & Aesthetic Principles

The SQUINT WebAssembly workbench follows a **Dieter Rams / Braun Industrial Functionalist** and **Warm-Ivory Editorial Workbench** aesthetic. Multiple sequence homology inference under Minimum Description Length (MDL) produces rich hierarchical objects — disconnected homology families ($K \ge 1$), guide trees with patristic branch lengths, `+1/+2 nt` codon frameshift repairs, multi-scale column provenance (`GlobalFine`, `CladeLocalFine`, `LineagePrivate`), and Coarse Horizon Bubbles preserving macro-biophysical signal where residue-level homology dissolves. The UI makes these objects **legible to a first-time biologist in under 10 seconds** while preserving **exact probabilistic and mathematical auditability** for power users.

### Ten Governing UX Rules
1. **Verdict First, Evidence on Demand (Progressive Disclosure):** Every stage begins with a single plain-language **Hero Verdict Card** summarizing *what was found* and *why it matters*, accompanied by 5 tabular KPI blocks. Sequence repeat sparklines, node scale ownership, column posterior vectors, and horizon bubble biophysical audits live in slide-over **Forensic Inspector Drawers**.
2. **Explicit Multi-Stage Pipeline Mental Model:** Decompose inference into 4 inspectable stages (`⌂ Start` $\to$ `0 QA` $\to$ `1 Components` $\to$ `2 Alignment` $\to$ `3 Dossier`), paired with a **Step-by-Step vs. ⚡ Full Auto** launch mode.
3. **Zero-Friction Data Ingestion:** One-click launch via curated biological showcase cards, drag-and-drop (`.fa`, `.fasta`, `.faa`, `.fna`, `.phy`, `.phylip`, `.nex`, `.nexus`, `.hnex`, `.json`), clipboard paste, or **Live Browser Streaming (`🌐 Fetch`)** from NCBI Entrez, EMBL-EBI ENA, or CORS-enabled GitHub/HTTPS URLs.
4. **Built-In Interactive Guided Walkthrough (`🎓 Tour`):** Stage-aware Coach Banner with step progress dots and one-click *"Try It Now"* chips pointing to real sequence rows, frameshift repairs, homology component splits, and horizon bubbles.
5. **Deterministic State Flushing (`flushAnalysisResults(fromStage)`):** Changing upstream mode (`auto|codon|protein|dna`), duplicate deduplication, or taxon inclusion checkboxes immediately flushes downstream stage results, badges, and drawers so stale state never leaks across runs.
6. **Pre-Flight Compute & Memory Gate:** Inspects $(N, \sum L)$ before launching Stage 1+ WASM alignment and warns only when cohort dimensions predict heavy browser latency or memory footprint, offering Stage 0 QA-only inspection or full alignment.
7. **Linked Multi-Scale Views:** Clicking a homology component pill updates both Stage 1 (guide tree) and Stage 2 (multi-scale alignment); clicking a horizon bubble card scrolls and highlights its genomic/sequence window in the matrix canvas; clicking a column in the top provenance barcode scrolls directly to that locus.
8. **Overflow-Proof Responsive Chrome:** Strict `max-width: 100vw; overflow-x: hidden;` across `390px` mobile, `1024px` tablet/laptop, and `1440px–2560px+` desktop monitors via the two-row `.topbar` contract at `<= 1260px`.
9. **100% Client-Side Privacy & Reproducibility:** All parsing, Fixed-Share $M_0$ profiling, 5-state Kraft-tight Forward-Backward scoring, progressive profile alignment, and `.hnex` subsetting execute locally inside a dedicated Web Worker (`worker.js` + `pkg/squint_bg.wasm`). Zero empirical substitution matrices are used anywhere.
10. **Strict Scientific Integrity:** Every metric, code length ($L_0$, $L_{\text{prov}}$, $\Delta L$ in nats), posterior $P_{\text{fine}}$, component $K$, repaired frameshift, and horizon bubble is computed directly by Rust WebAssembly on authentic sequence data — never hardcoded or synthesized in JavaScript.

---

## 2. Visual Design Tokens & Typography

```css
:root {
  --bg:            #F6F5F1; /* Warm ivory workbench canvas */
  --bg-card:       #FFFFFF; /* Elevated card surface */
  --bg-subtle:     #EFECE6; /* Recessed toolbar / pill container */
  --bg-hover:      #E6E2DA; /* Interactive row/button hover */

  --border:        #DFDCD4; /* Primary card & control border */
  --border-subtle: #EAE7DF; /* Internal table/row hairline */
  --border-strong: #9C9A94; /* Active/focused control border */

  --ink:           #141413; /* Primary headings, labels, values */
  --ink-secondary: #4A4945; /* Body copy, secondary copy */
  --ink-muted:     #787670; /* Captions, axis labels */

  --orange:        #E04F16; /* Braun Signal Orange accent */
  --orange-hover:  #C8430F;
  --orange-soft:   rgba(224, 79, 22, 0.08);

  --green:         #18794E; /* GlobalFine core / Admitted / Compression gain */
  --green-bg:      #E9F6EF;
  --blue:          #1D4ED8; /* CladeLocalFine / Structural conservation */
  --blue-bg:       #EFF6FF;
  --amber:         #B45309; /* Frameshift repair / LineagePrivate / Caution */
  --amber-bg:      #FEF6E7;
  --purple:        #6D28D9; /* Coarse Horizon Bubble / Macro-PRIME gain */
  --purple-bg:     #F5F3FF;
  --red:           #C81E1E; /* Quarantined / Internal stop / Disconnected */
  --red-bg:        #FDECEB;

  --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  --font-mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
}
```

All numeric KPIs, bit/nat ledgers, posteriors, coordinates, and sequence accessions use `var(--font-mono)` with `font-variant-numeric: tabular-nums`.

---

## 3. Pipeline Architecture (Stages 0 → 3)

| Stage | View ID | Engine Call (`WasmSquint`) | Primary Visualizations & Forensics |
| :--- | :--- | :--- | :--- |
| **`⌂ Start`** | `#landingView` | — | Hero launchpad + drag/drop + Step-by-Step vs Full Auto selector + Guided Walkthrough cards + 6 Showcase Cohort cards |
| **`0 QA`** | `#stage0View` | `inspect_stage0` | Length/frame/repeat SVG horizon, filterable Sequence QA table, **Drawer 0: Fixed-Share $M_0$ & Repeat Sparkline Inspector** |
| **`1 Components`** | `#stage1View` | `align_cohort` / `load_hnex` | Homology Component switcher ($K \ge 1$), High-DPI `<canvas>` Guide Tree + Minimap + Horizon Node markers (`◆`), Pairwise MDL / Frameshift / Component tables, **Drawer 1: Clade Split & Scale Ownership Inspector** |
| **`2 Alignment`** | `#stage2View` | Cached ledger + horizon FASTA | Synced Multi-Scale Barcode Overview + Virtualized Matrix `<canvas>` (Biophysical/Scale/Posterior colors, Codon vs AA toggle, `.` horizon vs `-` indel toggle), Horizon Bubble cards, **Drawers 2A & 2B** |
| **`3 Dossier`** | `#stage3View` | `subset_cached`, `export_cached_hnex`, `export_cached_fasta` | Interactive `squint subset` UI (`--component`, `--clade`, `--taxa`, `--region`, `--scale`), live standalone HTML Publication Dossier `<iframe>`, `.hnex` / `.fa` / `.nwk` / `.json` exports |

---

## 4. Worker Concurrency & Security Hygiene

- **Dedicated Web Worker (`docs/worker.js`):** Executes `squint.js` / `squint_bg.wasm` off the main thread so 60fps tree panning, alignment matrix scrolling, and drawer animations remain smooth. If a new cohort is loaded while the worker is busy, the active worker is terminated (`worker.terminate()`) and seamlessly respawned.
- **State Flushing (`flushAnalysisResults(fromStage)`):** Clearing Stage 0 flushes Stages 1–3; editing Stage 0 inclusion/mode preserves Stage 0 while flushing Stages 1–3.
- **Sanitization:** Every sequence name, organism string, Newick label, and user string interpolated into DOM HTML or SVG/HTML dossiers passes through `escapeHtml(s)`.
