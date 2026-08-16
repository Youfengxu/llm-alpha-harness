/**
 * OpenAI-compatible chat client with a deterministic on-disk cache.
 *
 * ── Why caching is a correctness feature, not a speed one ─────────────
 * Sampling makes an LLM non-deterministic, so re-running an evaluation gives
 * different numbers and no result can be reproduced or bisected. The predecessor
 * project produced three separate bugs where consecutive runs of the same script
 * disagreed — once flipping the SIGN of the decisive statistic — and each cost
 * more to diagnose than the analysis itself.
 *
 * Every response is therefore keyed by a hash of (model, prompt, and every
 * sampling parameter). A rerun with identical inputs replays from disk and is
 * byte-identical. Changing a prompt changes the key, so stale answers can never
 * silently survive an edit.
 *
 * temperature defaults to 0 for the same reason.
 *
 * ── Concurrency and pacing ────────────────────────────────────────────
 * vLLM on the GX10 serves ~500 tok/s at 64 concurrent requests, so batch work
 * should be genuinely concurrent — but unbounded concurrency causes timeouts
 * that look like refusals. A small semaphore keeps it in the serving layer's
 * sweet spot.
 */

import * as fs from "fs";
import * as path from "path";
import * as crypto from "crypto";
import { getModel, assertSafeToCall, type ModelSpec } from "./models.js";

const CACHE_DIR = process.env.LLM_CACHE_DIR ?? path.resolve(process.cwd(), ".cache/llm");

export interface CompletionOptions {
  model: string;
  prompt: string;
  system?: string;
  /** Defaults to 0. Non-zero makes results irreproducible even with the cache. */
  temperature?: number;
  maxTokens?: number;
  /** Bypass the cache read (still writes). For deliberately re-measuring. */
  refresh?: boolean;
  /**
   * Permit a call that evicts a pinned model. Deliberate, and the caller is
   * responsible for restorePinned() afterwards.
   */
  acknowledgeEviction?: boolean;
  timeoutMs?: number;
}

export interface CompletionResult {
  text: string;
  model: string;
  cached: boolean;
  /** Milliseconds spent on the wire. 0 when served from cache. */
  latencyMs: number;
}

function cacheKey(spec: ModelSpec, o: CompletionOptions): string {
  // Every input that can change the output must be in the key, or an edited
  // prompt would silently replay a stale answer.
  const material = JSON.stringify({
    id: spec.id,
    baseUrl: spec.baseUrl,
    system: o.system ?? "",
    prompt: o.prompt,
    temperature: o.temperature ?? 0,
    maxTokens: o.maxTokens ?? 512,
  });
  return crypto.createHash("sha256").update(material).digest("hex");
}

class Semaphore {
  private active = 0;
  private queue: Array<() => void> = [];
  constructor(private readonly limit: number) {}
  async acquire(): Promise<void> {
    if (this.active < this.limit) { this.active++; return; }
    await new Promise<void>((r) => this.queue.push(r));
    this.active++;
  }
  release(): void {
    this.active--;
    this.queue.shift()?.();
  }
}

const gate = new Semaphore(parseInt(process.env.LLM_CONCURRENCY ?? "16", 10));

/** Chat completion, cached. Throws on transport or API errors — never returns a placeholder. */
export async function complete(o: CompletionOptions): Promise<CompletionResult> {
  const spec = getModel(o.model);
  const key = cacheKey(spec, o);
  const file = path.join(CACHE_DIR, `${key}.json`);

  // Cache hits are checked BEFORE the eviction guard: replaying a stored
  // response touches no serving layer and so cannot evict anything.
  if (!o.refresh && fs.existsSync(file)) {
    const hit = JSON.parse(fs.readFileSync(file, "utf8")) as { text: string };
    return { text: hit.text, model: spec.id, cached: true, latencyMs: 0 };
  }

  assertSafeToCall(spec, { acknowledgeEviction: o.acknowledgeEviction });

  await gate.acquire();
  const started = Date.now();
  try {
    // Reachability is checked here so a stopped serving layer produces an
    // actionable message. vLLM on the GX10 is ON-DEMAND and conflicts with
    // llama-swap on the same host, so mid-batch it can simply vanish; a raw
    // ECONNREFUSED buried in a long run is a poor way to learn that.
    const messages = [
      ...(o.system ? [{ role: "system", content: o.system }] : []),
      { role: "user", content: o.prompt },
    ];
    const res = await fetch(`${spec.baseUrl}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: spec.id,
        messages,
        temperature: o.temperature ?? 0,
        max_tokens: o.maxTokens ?? 512,
      }),
      signal: AbortSignal.timeout(o.timeoutMs ?? 120_000),
    }).catch((e: unknown) => {
      const cause = (e as { cause?: { code?: string } })?.cause?.code ?? "";
      if (cause === "ECONNREFUSED" || cause === "ENOTFOUND" || cause === "EHOSTUNREACH") {
        throw new Error(
          `Cannot reach "${spec.id}" at ${spec.baseUrl} (${cause}).\n` +
          `  If this is the GX10 vLLM batch endpoint it is ON-DEMAND and conflicts ` +
          `with llama-swap on that host:\n` +
          `    ssh gx10 'sudo systemctl start vllm'   # serves :8000, stops llama-swap\n` +
          `    ssh gx10 'sudo systemctl stop vllm'    # restores llama-swap on :8085\n` +
          `  Otherwise pick a model whose serving layer is running.`
        );
      }
      throw e;
    });
    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      throw new Error(`${spec.id} returned ${res.status}: ${detail.slice(0, 300)}`);
    }
    const body = (await res.json()) as { choices?: Array<{ message?: { content?: string } }> };
    const text = body.choices?.[0]?.message?.content;
    // An empty completion is a failure, not an answer. Caching it would poison
    // every later run with a permanent blank.
    if (typeof text !== "string" || text.trim() === "") {
      throw new Error(`${spec.id} returned an empty completion`);
    }

    fs.mkdirSync(CACHE_DIR, { recursive: true });
    fs.writeFileSync(file, JSON.stringify({ text, model: spec.id, prompt: o.prompt }));
    return { text, model: spec.id, cached: false, latencyMs: Date.now() - started };
  } finally {
    gate.release();
  }
}

/** Number of cached responses on disk — a cheap check that a batch actually ran. */
export function cacheSize(): number {
  if (!fs.existsSync(CACHE_DIR)) return 0;
  return fs.readdirSync(CACHE_DIR).filter((f) => f.endsWith(".json")).length;
}

/**
 * Extracts the first number in [-1, 1] from a model response.
 *
 * Returns null rather than 0 when nothing parses. 0 is a legitimate score
 * meaning "neutral", so coercing a parse failure to 0 would silently convert
 * every malformed response into a confident neutral opinion and bias the sample
 * toward the middle.
 */
export function parseScore(text: string): number | null {
  const m = text.match(/-?\d+(?:\.\d+)?/);
  if (!m) return null;
  const v = parseFloat(m[0]);
  if (!Number.isFinite(v) || v < -1 || v > 1) return null;
  return v;
}
