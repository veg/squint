// SQUINT // Dedicated Rust WebAssembly Worker (Off-Main-Thread MDL Sequence Homology Engine)
import init, { WasmSquint } from './pkg/squint.js';

let engine = null;
let ready = false;

async function ensureInitialized() {
  if (ready && engine) return;
  await init();
  engine = new WasmSquint();
  ready = true;
}

self.onmessage = async (ev) => {
  const msg = ev.data || {};
  const { id, type } = msg;
  try {
    if (type === 'init') {
      await ensureInitialized();
      self.postMessage({ id, type: 'ready' });
      return;
    }

    await ensureInitialized();

    if (type === 'inspect_stage0') {
      const t0 = performance.now();
      const jsonStr = engine.inspect_stage0(
        String(msg.inputText || ''),
        String(msg.mode || 'auto'),
        Boolean(msg.deduplicate)
      );
      const wallMs = performance.now() - t0;
      self.postMessage({ id, type: 'stage0_result', jsonStr, wallMs });
      return;
    }

    if (type === 'align_cohort') {
      const t0 = performance.now();
      const jsonStr = engine.align_cohort(
        String(msg.inputText || ''),
        String(msg.mode || 'auto'),
        Boolean(msg.evalRef)
      );
      const wallMs = performance.now() - t0;
      self.postMessage({ id, type: 'align_result', jsonStr, wallMs });
      return;
    }

    if (type === 'load_hnex') {
      const t0 = performance.now();
      const jsonStr = engine.load_hnex(String(msg.hnexText || ''));
      const wallMs = performance.now() - t0;
      self.postMessage({ id, type: 'align_result', jsonStr, wallMs });
      return;
    }

    if (type === 'export_hnex') {
      const text = engine.export_cached_hnex();
      self.postMessage({ id, type: 'export_hnex_result', text });
      return;
    }

    if (type === 'export_fasta') {
      const text = engine.export_cached_fasta(Boolean(msg.distinguishHorizon));
      self.postMessage({ id, type: 'export_fasta_result', text });
      return;
    }

    if (type === 'subset_cached') {
      const t0 = performance.now();
      const jsonStr = engine.subset_cached(
        Number(msg.compId ?? -1),
        String(msg.clade || ''),
        String(msg.taxaCsv || ''),
        String(msg.regionStr || ''),
        Boolean(msg.globalCoreOnly)
      );
      const wallMs = performance.now() - t0;
      self.postMessage({ id, type: 'subset_result', jsonStr, wallMs });
      return;
    }

    throw new Error(`Unknown worker command: ${type}`);
  } catch (err) {
    const errMsg = (err && err.message) ? err.message : String(err);
    self.postMessage({ id, type: 'error', error: errMsg });
  }
};
