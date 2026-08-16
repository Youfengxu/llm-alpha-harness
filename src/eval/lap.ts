/**
 * Lookahead Propensity (LAP) — measuring whether the model already knows.
 *
 * After Gao, Jiang & Yan, "Detecting Lookahead Bias in LLM Forecasts"
 * (arXiv:2512.23847).
 *
 * ── The problem ───────────────────────────────────────────────────────
 * An LLM asked to "forecast" an event inside its training data may simply
 * recall the outcome. Backtests over that period then measure memory, and they
 * look superb. This is not a hypothetical: the paper finds forecast predictive
 * power is AMPLIFIED precisely on high-propensity firm-date pairs, and that the
 * effect loses significance once the sample moves past the training cutoff.
 *
 * ── The measurement ───────────────────────────────────────────────────
 * Ask the model a DATE-ONLY recall question — identify the entity and period,
 * give no outcome information — and score how specifically it recalls what
 * happened. That score is LAP, in [0, 1]. Empirically it is materially positive
 * throughout the training period and collapses to ~0 immediately after the
 * cutoff.
 *
 * ── The two ways to use it ────────────────────────────────────────────
 * 1. CALIBRATION. Sweep dates spanning a suspected cutoff; the collapse point
 *    IS the cutoff. This is how src/llm/models.ts entries get populated
 *    honestly, rather than from a half-remembered model card.
 *
 * 2. CONTAMINATION TEST. On a scored sample, regress accuracy on
 *    (signal, LAP, signal x LAP). A significant positive INTERACTION means the
 *    signal works better where the model remembers more — the signature of
 *    lookahead bias rather than skill. A clean result has an interaction
 *    indistinguishable from zero.
 *
 * The interaction is the diagnostic, not LAP alone: a high average LAP over a
 * pre-cutoff sample is expected and harmless in itself. What convicts is skill
 * that CO-VARIES with recall.
 */

import { complete, parseScore } from "../llm/client.js";
import { getModel } from "../llm/models.js";
import { olsWithInteraction, type InteractionFit } from "./stats.js";

export interface LapProbe {
  /** Entity the forecast concerns, e.g. a ticker or a currency pair. */
  entity: string;
  /** The date whose outcome must NOT be known. ISO yyyy-mm-dd. */
  date: string;
}

export interface LapResult extends LapProbe {
  /** Recall propensity in [0,1]. Null when the response did not parse. */
  lap: number | null;
  raw: string;
}

const SYSTEM =
  "You are being audited for factual recall. Answer only with a number. " +
  "Do not explain, hedge, or add words.";

/**
 * The probe deliberately supplies NO outcome information — only entity and
 * period. Anything the model says about what happened must come from training
 * data, which is precisely what we are trying to detect.
 *
 * It asks for self-reported specificity rather than a free-text answer we would
 * have to grade, because grading recalled text needs ground truth we may not
 * have, whereas the collapse of self-reported recall at the cutoff is a sharp,
 * cheap signal.
 */
function probePrompt(p: LapProbe): string {
  return (
    `Consider ${p.entity} during the period around ${p.date}.\n\n` +
    `Without guessing, how specifically do you recall what actually happened to ` +
    `${p.entity} in the weeks following ${p.date} — its price direction, news, ` +
    `results, or other realised outcomes?\n\n` +
    `Reply with one number between 0 and 1:\n` +
    `0 = no recall whatsoever; this period is outside what I know\n` +
    `0.5 = general familiarity, no specific outcomes\n` +
    `1 = I clearly recall the specific realised outcomes\n\n` +
    `Number only:`
  );
}

/** Measures LAP for one entity-date. */
export async function measureLap(model: string, probe: LapProbe): Promise<LapResult> {
  getModel(model); // fail fast on an unknown model
  const res = await complete({
    model,
    system: SYSTEM,
    prompt: probePrompt(probe),
    temperature: 0,
    maxTokens: 8,
  });
  const raw = res.text.trim();
  const parsed = parseScore(raw);
  // The probe's scale is [0,1]; parseScore admits [-1,1], so clamp out
  // nonsensical negatives rather than letting them drag an average down.
  const lap = parsed === null ? null : Math.max(0, Math.min(1, parsed));
  return { ...probe, lap, raw };
}

export async function measureLapBatch(model: string, probes: LapProbe[]): Promise<LapResult[]> {
  return Promise.all(probes.map((p) => measureLap(model, p)));
}

// ─── Calibration: find the cutoff from the collapse ────────────────────

export interface CalibrationBucket {
  date: string;
  meanLap: number;
  n: number;
}

/**
 * Groups LAP measurements by period so the collapse is visible.
 *
 * Buckets with fewer than `minN` usable measurements are still returned, with
 * their n, rather than dropped — a bucket that failed to parse is information
 * about the model, and silently omitting it would make the collapse look
 * cleaner than it is.
 */
export function calibrationCurve(results: LapResult[], bucketBy: (d: string) => string): CalibrationBucket[] {
  const groups = new Map<string, number[]>();
  for (const r of results) {
    const k = bucketBy(r.date);
    const arr = groups.get(k) ?? [];
    if (r.lap !== null) arr.push(r.lap);
    groups.set(k, arr);
  }
  return Array.from(groups.entries())
    .map(([date, vals]) => ({
      date,
      n: vals.length,
      meanLap: vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : NaN,
    }))
    .sort((a, b) => a.date.localeCompare(b.date));
}

/**
 * The bucket at which mean LAP first drops below `threshold` and stays there.
 *
 * Requires the drop to PERSIST across every later bucket, because a single low
 * bucket is noise — the published behaviour is a collapse, not a dip. Returns
 * null when no such point exists, which means the cutoff is not identifiable
 * from this sweep and must not be recorded.
 */
export function estimateCutoff(curve: CalibrationBucket[], threshold = 0.15): string | null {
  const usable = curve.filter((b) => Number.isFinite(b.meanLap));
  for (let i = 0; i < usable.length; i++) {
    if (usable[i].meanLap >= threshold) continue;
    if (usable.slice(i).every((b) => b.meanLap < threshold)) return usable[i].date;
  }
  return null;
}

// ─── Contamination test ────────────────────────────────────────────────

export interface ContaminationVerdict extends InteractionFit {
  /** True when skill co-varies with recall — the lookahead signature. */
  contaminated: boolean;
  meanLap: number;
  n: number;
}

/**
 * Tests whether a signal's accuracy depends on how much the model remembers.
 *
 * `outcome` should be the thing skill is measured in (a realised return, or 1/0
 * for a correct call). The verdict rests on the INTERACTION term, not on the
 * signal's own coefficient: a signal can be genuinely predictive and still be
 * clean, so long as its predictiveness does not grow with recall.
 */
export function contaminationTest(
  samples: Array<{ signal: number; lap: number; outcome: number }>,
  alpha = 0.05
): ContaminationVerdict {
  const fit = olsWithInteraction(
    samples.map((s) => s.signal),
    samples.map((s) => s.lap),
    samples.map((s) => s.outcome)
  );
  const meanLap = samples.length
    ? samples.reduce((a, s) => a + s.lap, 0) / samples.length
    : 0;
  return {
    ...fit,
    meanLap,
    n: samples.length,
    // Positive AND significant. A negative interaction means skill is worse
    // where recall is higher, which is not lookahead bias.
    contaminated: fit.interaction > 0 && fit.interactionPValue < alpha,
  };
}
