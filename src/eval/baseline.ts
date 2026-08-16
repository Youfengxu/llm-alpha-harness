/**
 * Cheap baselines that any LLM signal must beat before it is worth anything.
 *
 * ── Why this file matters more than it looks ──────────────────────────
 * In the predecessor project the dumb control won every single time:
 *
 *   - A static 40% allocation matched a 3,698-trade signal engine on return,
 *     beat it on drawdown, and needed 10 trades.
 *   - A 20-line trailing average beat a fitted Gaussian mixture at forecasting
 *     volatility (r = 0.619 vs 0.488) AND economically.
 *   - Eight threshold rules had accuracy uncorrelated with return, the
 *     signature of noise dressed as signal.
 *
 * Assume the same here. An LLM that cannot beat a word-count is not reading;
 * it is producing an expensive sentiment proxy. That is a cheap, decisive test
 * and it should be run FIRST, before any expensive evaluation.
 *
 * The lexicon is a compact Loughran-McDonald-style financial word list. General
 * sentiment lexicons are actively wrong on financial text — "liability",
 * "cost", "tax" and "capital" are neutral accounting terms that a consumer
 * lexicon scores as negative — which is the reason a finance-specific list is
 * used at all.
 */

const NEGATIVE = new Set([
  "adverse", "adversely", "against", "bankruptcy", "breach", "cancel", "cancelled",
  "cease", "challenging", "claims", "closure", "concern", "concerns", "contraction",
  "decline", "declined", "declines", "decrease", "decreased", "default", "deficiency",
  "deficit", "delay", "delayed", "deteriorate", "deterioration", "difficult",
  "difficulties", "diminished", "disappointing", "discontinued", "dispute", "downgrade",
  "downturn", "drop", "dropped", "failure", "fell", "impair", "impairment", "inability",
  "insufficient", "investigation", "lawsuit", "layoff", "layoffs", "litigation", "loss",
  "losses", "misconduct", "negative", "penalty", "postponed", "pressure", "recall",
  "recession", "restructuring", "risk", "risks", "shortfall", "shrink", "slowdown",
  "slowing", "sluggish", "subpoena", "suspended", "termination", "unable", "uncertain",
  "uncertainty", "underperform", "unfavorable", "violation", "weak", "weaken", "weakness",
  "worse", "writedown", "writeoff",
]);

const POSITIVE = new Set([
  "accelerate", "accelerated", "achieve", "achieved", "advantage", "beat", "benefit",
  "best", "better", "boost", "breakthrough", "confident", "constructive", "delivered",
  "efficiency", "encouraged", "exceed", "exceeded", "exceeding", "excellent", "expansion",
  "favorable", "gain", "gained", "gains", "good", "grew", "growing", "growth", "high",
  "higher", "improve", "improved", "improvement", "increase", "increased", "leading",
  "momentum", "opportunity", "outperform", "outstanding", "positive", "profit",
  "profitability", "profitable", "progress", "raised", "record", "recovery", "resilient",
  "robust", "rose", "solid", "strength", "strong", "stronger", "succeed", "success",
  "successful", "surpassed", "upgrade", "upside", "win", "won",
]);

/**
 * Negators that flip the polarity of the next few words.
 *
 * Without this, "no impairment" and "not weak" — both reassuring — score as
 * negative, which is common enough in filings to matter. The window is short
 * because negation scope in English rarely reaches far, and a long window
 * flips words it should not.
 */
const NEGATORS = new Set(["no", "not", "never", "without", "nor", "neither", "cannot", "none"]);
const NEGATION_WINDOW = 3;

export interface LexiconScore {
  /** Net polarity in [-1, 1]: (pos - neg) / (pos + neg). 0 when no words matched. */
  score: number;
  positive: number;
  negative: number;
  /** Total sentiment-bearing words found. 0 means the score is meaningless. */
  matched: number;
  words: number;
}

export function lexiconSentiment(text: string): LexiconScore {
  const tokens = text.toLowerCase().match(/[a-z']+/g) ?? [];
  let pos = 0, neg = 0;
  let negateUntil = -1;

  for (let i = 0; i < tokens.length; i++) {
    const w = tokens[i];
    if (NEGATORS.has(w)) { negateUntil = i + NEGATION_WINDOW; continue; }
    const negated = i <= negateUntil;
    if (POSITIVE.has(w)) { negated ? neg++ : pos++; }
    else if (NEGATIVE.has(w)) { negated ? pos++ : neg++; }
  }

  const matched = pos + neg;
  return {
    score: matched > 0 ? (pos - neg) / matched : 0,
    positive: pos,
    negative: neg,
    matched,
    words: tokens.length,
  };
}

/**
 * Standardised surprise: (actual - expected) / |expected|.
 *
 * The relevant baseline for earnings work, since post-earnings-announcement
 * drift is already predictable from the numeric surprise alone. The question an
 * LLM transcript signal must answer is not "does it predict drift" but "does it
 * predict drift BEYOND this" — a far better-posed question, and the reason
 * Option B was recommended as the first build.
 *
 * Returns null when expected is ~0, where the ratio explodes and the number
 * would be dominated by division noise.
 */
export function numericSurprise(actual: number, expected: number): number | null {
  if (!Number.isFinite(actual) || !Number.isFinite(expected)) return null;
  if (Math.abs(expected) < 1e-9) return null;
  return (actual - expected) / Math.abs(expected);
}

/**
 * Does the candidate add anything beyond the baseline?
 *
 * Returns the incremental R^2 from adding `candidate` to a regression already
 * containing `baseline`. Near zero means the LLM is re-deriving the cheap
 * signal at great expense — the outcome to expect, and the one worth detecting
 * early and cheaply.
 */
export function incrementalR2(baseline: number[], candidate: number[], y: number[]): number {
  const n = Math.min(baseline.length, candidate.length, y.length);
  if (n < 5) return 0;

  const r2 = (preds: number[][]): number => {
    const k = preds.length + 1;
    const X = Array.from({ length: n }, (_, i) => [1, ...preds.map((p) => p[i])]);
    const xtx = Array.from({ length: k }, () => new Array(k).fill(0));
    const xty = new Array(k).fill(0);
    for (let i = 0; i < n; i++) {
      for (let a = 0; a < k; a++) {
        xty[a] += X[i][a] * y[i];
        for (let b = 0; b < k; b++) xtx[a][b] += X[i][a] * X[i][b];
      }
    }
    const beta = solve(xtx, xty);
    if (!beta) return 0;
    const ybar = y.slice(0, n).reduce((a, b) => a + b, 0) / n;
    let sse = 0, sst = 0;
    for (let i = 0; i < n; i++) {
      let yhat = 0;
      for (let a = 0; a < k; a++) yhat += beta[a] * X[i][a];
      sse += (y[i] - yhat) ** 2;
      sst += (y[i] - ybar) ** 2;
    }
    return sst > 0 ? 1 - sse / sst : 0;
  };

  return Math.max(0, r2([baseline, candidate]) - r2([baseline]));
}

function solve(A: number[][], b: number[]): number[] | null {
  const k = A.length;
  const m = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < k; col++) {
    let piv = col;
    for (let r = col + 1; r < k; r++) if (Math.abs(m[r][col]) > Math.abs(m[piv][col])) piv = r;
    if (Math.abs(m[piv][col]) < 1e-12) return null;
    [m[col], m[piv]] = [m[piv], m[col]];
    const p = m[col][col];
    for (let j = col; j <= k; j++) m[col][j] /= p;
    for (let r = 0; r < k; r++) {
      if (r === col) continue;
      const f = m[r][col];
      if (f === 0) continue;
      for (let j = col; j <= k; j++) m[r][j] -= f * m[col][j];
    }
  }
  return m.map((row) => row[k]);
}
