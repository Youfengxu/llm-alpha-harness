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
 *
 * ── Endpoints are host NAMES, overridable by environment ──────────────
 * Defaults use `gx10` and `127.0.0.1`, resolved locally (Tailscale MagicDNS in
 * this case). Set GX10_VLLM_URL / MAC_LLAMASWAP_URL for any other topology.
 * Baking a specific private IP into a published repo is both fragile and
 * needless disclosure of a network layout.
 *
 * ── The serving layer is SHARED, and eviction is the default ──────────
 * llama-swap holds one primary model per host and swaps on demand, so calling
 * any non-resident model EVICTS whatever is loaded. Two models are pinned
 * because other services depend on them being warm:
 *
 *   Mac Studio  muse-glimmer-30b
 *   GX10        Qwen3-Coder-Next-UD-Q4_K_M  (ttl: -1)
 *
 * A batch scoring run over hundreds of names would otherwise thrash those
 * continuously, and a ~15s reload each way is both slow and disruptive to
 * whatever else is using them. This was not hypothetical: the harness's own
 * smoke test evicted muse-glimmer-30b on its first run.
 *
 * `evicts` records the cost, and `assertSafeToCall` makes it an explicit
 * decision rather than a side effect. The sanctioned way to do heavy batch work
 * is the GX10's vLLM mode, which stops llama-swap on that host deliberately and
 * restores it on exit — a decision taken once, not per request.
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
  /** Which box serves it. Eviction is per host. */
  host: "mac-studio" | "gx10";
  /** True when other services depend on this staying warm. Never evict casually. */
  pinned?: boolean;
  /**
   * The pinned model this call would evict, if any. Non-null means calling this
   * model has a cost beyond its own latency.
   */
  evicts?: string;
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
    baseUrl: process.env.GX10_VLLM_URL ?? "http://gx10:8000/v1",
    cutoff: null,
    contextTokens: 8192,
    host: "gx10",
    // vLLM mode stops llama-swap on the whole host, so Qwen3-Coder-Next goes
    // down with it. That is the SANCTIONED way to batch: one deliberate switch
    // and one restore, rather than an eviction per request.
    evicts: "gx10/Qwen3-Coder-Next-UD-Q4_K_M (whole llama-swap stops)",
    notes:
      "Qwen2.5-32B-Instruct-AWQ on GX10 vLLM, ~500 tok/s at 64 concurrency. " +
      "Batch workhorse. 8k context does NOT fit a 10-K — chunk deliberately. " +
      "Serving it stops llama-swap on that host (systemd Conflicts=).",
  },
  /**
   * gpt-oss-120b on GX10 vLLM. 117B total / 5.1B active MoE, native MXFP4.
   *
   * The batch workhorse: published GX10 figures are ~6,700 tok/s PREFILL at
   * concurrency 10 against ~160 tok/s generation, and this harness's workload is
   * prefill-dominated (long disclosure in, one score out). The 273 GB/s memory
   * bandwidth punishes decode, which is exactly why an MoE activating 4 of 128
   * experts suits this box where a dense model of the same size would crawl.
   *
   * Serving it stops llama-swap on GX10 entirely, taking Coder-Next — the
   * orchestrator's worker and opencode's default — offline. That is a deliberate
   * once-per-session switch, not a per-request cost.
   */
  "gpt-oss-120b": {
    id: "gpt-oss-120b",
    baseUrl: process.env.GX10_VLLM_URL ?? "http://gx10:8000/v1",
    // End of JUNE 2024. 2024-06 measured 0.17 — partial recall — so the true
    // boundary falls inside that month; month-end excludes all of it. Erring
    // late costs samples, erring early admits contaminated ones.
    cutoff: "2024-06-30",
    cutoffSource:
      "probeCutoff.ts 2026-08-16, 3 entities (AAPL/NVDA/TSLA). Control 2021-06/" +
      "2022-03 both 1.00. Recall 0.50-0.83 through 2024-05, 0.17 at 2024-06, " +
      "0.00 from 2024-07 and flat to 2026-07. 72 probes in 91s on vLLM.",
    contextTokens: 16384,
    host: "gx10",
    evicts: "gx10/Qwen3-Coder-Next-UD-Q4_K_M (whole llama-swap stops)",
    notes:
      "vLLM 0.24, mxfp4 auto-detected from config.json, fp8 KV, async scheduling. " +
      "Model supports 131072 ctx but is served at 16384 — KV at full length x 64 " +
      "sequences would dominate memory for no benefit on scoring prompts.",
  },
  /** PINNED on GX10 (ttl: -1). Free to call — it is already resident. */
  "gx10/Qwen3-Coder-Next-UD-Q4_K_M": {
    id: "gx10/Qwen3-Coder-Next-UD-Q4_K_M",
    baseUrl: process.env.MAC_LLAMASWAP_URL ?? "http://127.0.0.1:8085/v1",
    cutoff: null,
    contextTokens: 32768,
    host: "gx10",
    pinned: true,
    notes: "Federated from GX10 via the Mac's llama-swap peers list. Already warm.",
  },
  /** PINNED on the Mac Studio. Free to call — it is already resident. */
  "muse-glimmer-30b": {
    id: "muse-glimmer-30b",
    baseUrl: process.env.MAC_LLAMASWAP_URL ?? "http://127.0.0.1:8085/v1",
    // End of MAY 2024, not the collapse month, because erring LATE only costs
    // samples while erring early admits contaminated ones. The probe measures
    // day 15 of each month, so 2024-05-31 excludes the whole ambiguous
    // transition month rather than assuming its second half was unknown.
    cutoff: "2024-05-31",
    cutoffSource:
      "probeCutoff.ts 2026-08-16, 3 entities (AAPL/NVDA/TSLA), quarterly then " +
      "monthly LAP sweep. Recall 0.50 through 2024-04, 0.17 at 2024-05, 0.00 from " +
      "2024-06 onward and flat to 2026-07. Control dates 2021-06/2022-03 both 0.50.",
    contextTokens: 32768,
    host: "mac-studio",
    pinned: true,
    notes:
      "Resident on the Mac — costs nothing to call. REASONING model: emits " +
      "reasoning_content before content, ~570 tokens and 10-50s for a one-line " +
      "score, and /no_think does not disable it. Excellent at scoring (spearman " +
      "0.939 vs human on subtle text, against a lexicon at -0.476) but suited to " +
      "DEPTH not breadth. Reports LAP binarily — 0.5 when it knows a period, 0.0 " +
      "when it does not — so the collapse is sharp but ungraded.",
  },
  "qwen3.6-35b-a3b": {
    id: "qwen3.6-35b-a3b",
    baseUrl: process.env.MAC_LLAMASWAP_URL ?? "http://127.0.0.1:8085/v1",
    cutoff: null,
    contextTokens: 32768,
    host: "mac-studio",
    evicts: "muse-glimmer-30b",
    notes: "Mac Studio llama-swap. Longer context, lower throughput than vLLM.",
  },
  "gemma-4-31b-it-qat-ud-q4-k-xl": {
    id: "gemma-4-31b-it-qat-ud-q4-k-xl",
    baseUrl: process.env.MAC_LLAMASWAP_URL ?? "http://127.0.0.1:8085/v1",
    cutoff: null,
    contextTokens: 32768,
    host: "mac-studio",
    evicts: "muse-glimmer-30b",
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

export class EvictionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "EvictionError";
  }
}

/**
 * Throws when calling `model` would evict a pinned model.
 *
 * A warning would not do. Eviction is invisible at the call site — the request
 * succeeds, just 15 seconds slower, and the damage lands on some other service
 * that expected its model warm. The harness's own smoke test did exactly this.
 *
 * Pass `acknowledgeEviction` to proceed deliberately, and restore afterwards
 * with `restorePinned`.
 */
export function assertSafeToCall(
  model: ModelSpec,
  opts: { acknowledgeEviction?: boolean } = {}
): void {
  // LLM_ACK_EVICTION=1 means the operator has ALREADY made the switch
  // deliberately at the service level — e.g. `systemctl start vllm`, which stops
  // llama-swap on that host once rather than per request. The guard exists to
  // catch an ACCIDENTAL eviction; once the host has been switched on purpose
  // there is nothing further to evict and blocking every call is just noise.
  if (process.env.LLM_ACK_EVICTION === "1") return;
  if (!model.evicts || opts.acknowledgeEviction) return;
  throw new EvictionError(
    `Calling "${model.id}" on ${model.host} would evict ${model.evicts}, which is ` +
    `pinned because other services rely on it being warm (~15s to reload each way).\n` +
    `  Prefer a pinned model: ${pinnedFor(model.host) ?? "none on this host"}\n` +
    `  For heavy batch work use the GX10 vLLM mode, which switches the host once ` +
    `deliberately instead of evicting per request.\n` +
    `  To proceed anyway pass { acknowledgeEviction: true } and call restorePinned() after.`
  );
}

/** The pinned model on a host, if any. */
export function pinnedFor(host: ModelSpec["host"]): string | null {
  return Object.values(MODELS).find((m) => m.host === host && m.pinned)?.id ?? null;
}

/**
 * Reloads a host's pinned model by issuing a minimal request against it.
 *
 * Call after any acknowledged eviction. llama-swap loads on demand, so touching
 * the model is enough to make it resident again.
 */
export async function restorePinned(host: ModelSpec["host"]): Promise<boolean> {
  const id = pinnedFor(host);
  if (!id) return false;
  const spec = MODELS[id];
  try {
    const res = await fetch(`${spec.baseUrl}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: id, messages: [{ role: "user", content: "ok" }], max_tokens: 1 }),
      signal: AbortSignal.timeout(180_000),
    });
    return res.ok;
  } catch {
    return false;
  }
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
