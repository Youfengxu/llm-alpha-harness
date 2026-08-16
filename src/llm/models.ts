/**
 * Model registry — every model the harness may use, with its knowledge cutoff.
 *
 * ── Why a registry rather than a config string ────────────────────────
 * The single largest threat to LLM forecasting research is that the model has
 * already read the outcome. Gao, Jiang & Yan show that "Lookahead Propensity"
 * — the probability a model has internalised a firm-date's realised outcome —
 * is materially positive across the training period and collapses to
 * essentially zero right after the cutoff, and that apparent forecasting skill
 * is amplified exactly where propensity is high.
 *
 * The defence is to score only samples dated after the cutoff. That defence is
 * worth nothing if the cutoff is a comment in a README, so it is data here, and
 * `assertScorable` turns it into a runtime check.
 *
 * ── Unknown means unusable, not assumed ───────────────────────────────
 * A model whose cutoff we cannot establish gets `cutoff: null`, and the harness
 * REFUSES to evaluate with it rather than guessing. Guessing a cutoff too early
 * silently converts a contaminated sample into an apparently clean one, which
 * is the exact failure the control exists to prevent. A wrong cutoff is worse
 * than no cutoff, because it looks like rigour.
 *
 * ── Local models are preferred, and not mainly for cost ───────────────
 * A pinned local GGUF/AWQ file has a fixed cutoff that cannot change underneath
 * a study. Hosted APIs are updated silently; if the model behind a post-cutoff
 * evaluation shifts mid-study, the control evaporates and nothing warns you.
 * Reproducibility is the argument, cheap inference is a bonus.
 */

export interface ModelSpec {
  /** Identifier as the serving layer knows it. */
  id: string;
  /** OpenAI-compatible base URL. */
  baseUrl: string;
  /**
   * End of training data, as an ISO date. Samples at or before this are
   * CONTAMINATED and may only be used for the LAP calibration itself.
   *
   * null means "not established" — the harness will refuse to evaluate.
   */
  cutoff: string | null;
  /** How the cutoff was established. Required when cutoff is non-null. */
  cutoffSource?: string;
  /** Maximum context window in tokens, so callers can chunk deliberately. */
  contextTokens: number;
  notes?: string;
}

/**
 * Cutoffs are deliberately left null until each is verified against a primary
 * source (model card or paper). Publishing a plausible date from memory would
 * be exactly the "looks like rigour" failure this module warns about.
 *
 * To establish one, run `scripts/probeCutoff.ts`, which asks the model about
 * events at known dates and finds where recall collapses — then record BOTH the
 * date and how it was determined.
 */
export const MODELS: Record<string, ModelSpec> = {
  qwen32: {
    id: "qwen32",
    baseUrl: process.env.GX10_VLLM_URL ?? "http://100.119.29.75:8000/v1",
    cutoff: null,
    contextTokens: 8192,
    notes:
      "Qwen2.5-32B-Instruct-AWQ on GX10 vLLM, ~500 tok/s at 64 concurrency. " +
      "Batch workhorse. 8k context does NOT fit a 10-K — chunk deliberately. " +
      "Serving it stops llama-swap on that host (systemd Conflicts=).",
  },
  /** GX10's llama-swap, which is what runs there when vLLM is stopped. */
  "gx10/Qwen3-Coder-Next-UD-Q4_K_M": {
    id: "gx10/Qwen3-Coder-Next-UD-Q4_K_M",
    baseUrl: process.env.MAC_LLAMASWAP_URL ?? "http://127.0.0.1:8085/v1",
    cutoff: null,
    contextTokens: 32768,
    notes: "Federated from GX10 via the Mac's llama-swap peers list.",
  },
  "qwen3.6-35b-a3b": {
    id: "qwen3.6-35b-a3b",
    baseUrl: process.env.MAC_LLAMASWAP_URL ?? "http://127.0.0.1:8085/v1",
    cutoff: null,
    contextTokens: 32768,
    notes: "Mac Studio llama-swap. Longer context, lower throughput than vLLM.",
  },
  "gemma-4-31b-it-qat-ud-q4-k-xl": {
    id: "gemma-4-31b-it-qat-ud-q4-k-xl",
    baseUrl: process.env.MAC_LLAMASWAP_URL ?? "http://127.0.0.1:8085/v1",
    cutoff: null,
    contextTokens: 32768,
    notes: "Mac Studio llama-swap. Different family — useful as a second opinion.",
  },
};

export function getModel(key: string): ModelSpec {
  const m = MODELS[key];
  if (!m) {
    throw new Error(
      `Unknown model "${key}". Known: ${Object.keys(MODELS).join(", ")}. ` +
      `Add it to src/llm/models.ts WITH its cutoff before using it.`
    );
  }
  return m;
}

export class ContaminationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ContaminationError";
  }
}

/**
 * Throws unless `sampleDate` is safely after the model's knowledge cutoff.
 *
 * Call this before scoring anything whose outcome will be used to measure
 * predictive skill. It is deliberately a hard error rather than a warning: a
 * warning in a long batch log is indistinguishable from no warning at all, and
 * this is the control the entire harness rests on.
 *
 * `allowContaminated` exists for one legitimate case — the LAP calibration
 * itself, which MUST query pre-cutoff dates to measure recall. It is verbose by
 * design so it cannot be passed casually.
 */
export function assertScorable(
  model: ModelSpec,
  sampleDate: Date | string,
  opts: { allowContaminated?: "yes-this-is-the-LAP-calibration" } = {}
): void {
  if (opts.allowContaminated === "yes-this-is-the-LAP-calibration") return;

  if (model.cutoff === null) {
    throw new ContaminationError(
      `Model "${model.id}" has no established knowledge cutoff, so post-cutoff ` +
      `evaluation is impossible and any result would be uninterpretable. ` +
      `Establish it with scripts/probeCutoff.ts and record it in src/llm/models.ts ` +
      `with its source. Do NOT guess: a wrong cutoff looks like rigour.`
    );
  }

  const sample = typeof sampleDate === "string" ? new Date(sampleDate) : sampleDate;
  if (Number.isNaN(sample.getTime())) {
    throw new ContaminationError(`Invalid sample date: ${String(sampleDate)}`);
  }

  const cutoff = new Date(model.cutoff);
  if (sample <= cutoff) {
    throw new ContaminationError(
      `Sample dated ${sample.toISOString().slice(0, 10)} is at or before ` +
      `"${model.id}"'s knowledge cutoff (${model.cutoff}). The model may simply ` +
      `recall the outcome, so any measured "skill" is memory. Use a later sample, ` +
      `or a model with an earlier cutoff.`
    );
  }
}

/** True when the sample is safe to score. Non-throwing form, for filtering. */
export function isScorable(model: ModelSpec, sampleDate: Date | string): boolean {
  try {
    assertScorable(model, sampleDate);
    return true;
  } catch {
    return false;
  }
}
