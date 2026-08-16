import { describe, it, expect } from "vitest";
import { assertScorable, isScorable, ContaminationError, type ModelSpec } from "../src/llm/models.js";
import { parseScore } from "../src/llm/client.js";
import {
  effectiveSampleSize, effectiveSampleCeiling, meanPairwiseCorrelation,
  spearman, pearson, tTest, tTestEffective, olsWithInteraction,
} from "../src/eval/stats.js";
import { lexiconSentiment, numericSurprise, incrementalR2 } from "../src/eval/baseline.js";
import { evaluate, registrationHash, type PreRegistration } from "../src/eval/criteria.js";
import { calibrationCurve, estimateCutoff, contaminationTest, type LapResult } from "../src/eval/lap.js";

const model = (cutoff: string | null): ModelSpec => ({
  id: "test", baseUrl: "http://localhost/v1", cutoff, contextTokens: 8192,
});

// ─── Contamination guard ──────────────────────────────────────────────

describe("assertScorable", () => {
  it("refuses a model with no established cutoff rather than assuming one", () => {
    // A guessed cutoff silently converts contaminated samples into apparently
    // clean ones — worse than no cutoff, because it looks like rigour.
    expect(() => assertScorable(model(null), "2026-01-01")).toThrow(ContaminationError);
  });

  it("rejects samples at or before the cutoff", () => {
    expect(() => assertScorable(model("2024-06-30"), "2024-01-15")).toThrow(ContaminationError);
    expect(() => assertScorable(model("2024-06-30"), "2024-06-30")).toThrow(ContaminationError);
  });

  it("accepts samples strictly after the cutoff", () => {
    expect(() => assertScorable(model("2024-06-30"), "2024-07-01")).not.toThrow();
  });

  it("rejects an unparseable date instead of silently passing it", () => {
    expect(() => assertScorable(model("2024-06-30"), "not-a-date")).toThrow(ContaminationError);
  });

  it("only bypasses the guard with the explicit LAP-calibration opt-in", () => {
    expect(() =>
      assertScorable(model(null), "2020-01-01", { allowContaminated: "yes-this-is-the-LAP-calibration" })
    ).not.toThrow();
  });

  it("isScorable is the non-throwing form", () => {
    expect(isScorable(model("2024-06-30"), "2024-07-01")).toBe(true);
    expect(isScorable(model("2024-06-30"), "2024-01-01")).toBe(false);
    expect(isScorable(model(null), "2026-01-01")).toBe(false);
  });
});

describe("parseScore", () => {
  it("returns null rather than 0 when nothing parses", () => {
    // 0 is a legitimate 'neutral' score; coercing failures to 0 would bias the
    // whole sample toward the middle and hide broken prompts.
    expect(parseScore("I cannot answer that")).toBeNull();
    expect(parseScore("")).toBeNull();
  });

  it("rejects values outside [-1, 1]", () => {
    expect(parseScore("42")).toBeNull();
    expect(parseScore("-3")).toBeNull();
  });

  it("extracts a valid score from noisy output", () => {
    expect(parseScore("0.7")).toBe(0.7);
    expect(parseScore("Score: -0.25 (bearish)")).toBe(-0.25);
  });
});

// ─── Effective sample size ────────────────────────────────────────────

describe("effectiveSampleSize", () => {
  it("reproduces the crypto finding that killed the predecessor project", () => {
    // At rho = 0.449, breadth buys almost nothing: 12 majors are worth ~2.02
    // independent observations and 20 are worth ~2.10 — an extra 8 assets bought
    // 0.08 of an observation. That is why every "improvement" there was
    // unfalsifiable in practice.
    expect(effectiveSampleSize(12, 0.449)).toBeCloseTo(2.02, 2);
    expect(effectiveSampleSize(20, 0.449)).toBeCloseTo(2.10, 2);
  });

  it("asymptotes at 1/rho — adding correlated assets cannot fix it", () => {
    const ceiling = effectiveSampleCeiling(0.449);
    expect(effectiveSampleSize(1000, 0.449)).toBeLessThan(ceiling);
    expect(effectiveSampleSize(1000, 0.449)).toBeCloseTo(ceiling, 1);
    expect(ceiling).toBeCloseTo(2.23, 1);
  });

  it("returns n when series are independent", () => {
    expect(effectiveSampleSize(50, 0)).toBe(50);
  });

  it("measures mean pairwise correlation across series", () => {
    const a = [1, 2, 3, 4, 5];
    expect(meanPairwiseCorrelation([a, a, a])).toBeCloseTo(1, 6);
    expect(meanPairwiseCorrelation([a, [...a].reverse()])).toBeCloseTo(-1, 6);
  });
});

describe("tTestEffective", () => {
  it("is far less significant than the naive test under correlation", () => {
    // The central discipline: correlated observations are not independent ones.
    const xs = Array.from({ length: 40 }, (_, i) => 0.4 + Math.sin(i) * 0.5);
    const naive = tTest(xs);
    const adj = tTestEffective(xs, 0.45);
    expect(adj.nEff).toBeLessThan(xs.length);
    expect(adj.pValue).toBeGreaterThan(naive.pValue);
  });

  it("refuses to report significance when n_eff collapses below 2", () => {
    expect(tTestEffective([1, 2, 3, 4], 0.99).pValue).toBe(1);
  });
});

// ─── Correlation ──────────────────────────────────────────────────────

describe("correlation", () => {
  it("spearman handles ties via average ranks", () => {
    expect(spearman([1, 1, 2, 3], [1, 1, 2, 3])).toBeCloseTo(1, 6);
  });

  it("detects the anti-correlation that sank exit-policy tuning", () => {
    // rho = -0.339 there: historically-best parameters did WORSE next fold.
    expect(spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])).toBeCloseTo(-1, 6);
  });

  it("pearson is 0 for unrelated constant input", () => {
    expect(pearson([1, 1, 1, 1], [1, 2, 3, 4])).toBe(0);
  });
});

// ─── LAP ──────────────────────────────────────────────────────────────

describe("LAP calibration", () => {
  const r = (date: string, lap: number | null): LapResult => ({ entity: "X", date, lap, raw: String(lap) });

  it("finds the collapse point as the cutoff", () => {
    const curve = calibrationCurve(
      [
        r("2024-01", 0.8), r("2024-01", 0.9),
        r("2024-02", 0.7), r("2024-02", 0.8),
        r("2024-03", 0.05), r("2024-03", 0.0),
        r("2024-04", 0.0), r("2024-04", 0.1),
      ],
      (d) => d
    );
    expect(estimateCutoff(curve)).toBe("2024-03");
  });

  it("requires the collapse to persist — a single dip is not a cutoff", () => {
    const curve = calibrationCurve(
      [r("2024-01", 0.8), r("2024-02", 0.02), r("2024-03", 0.9), r("2024-04", 0.85)],
      (d) => d
    );
    expect(estimateCutoff(curve)).toBeNull();
  });

  it("returns null when recall never collapses, so no cutoff is recorded", () => {
    const curve = calibrationCurve([r("2024-01", 0.8), r("2024-02", 0.9)], (d) => d);
    expect(estimateCutoff(curve)).toBeNull();
  });

  it("keeps unparseable buckets visible rather than dropping them", () => {
    const curve = calibrationCurve([r("2024-01", null), r("2024-02", 0.5)], (d) => d);
    expect(curve).toHaveLength(2);
    expect(curve[0].n).toBe(0);
  });
});

describe("contaminationTest", () => {
  it("flags a signal whose accuracy grows with recall", () => {
    // outcome = signal * lap: the signal only 'works' where the model remembers.
    const samples = [];
    for (let i = 0; i < 60; i++) {
      const signal = (i % 7) / 6 - 0.5;
      const lap = (i % 5) / 4;
      samples.push({ signal, lap, outcome: signal * lap * 4 });
    }
    const v = contaminationTest(samples);
    expect(v.interaction).toBeGreaterThan(0);
    expect(v.contaminated).toBe(true);
  });

  it("clears a signal that predicts independently of recall", () => {
    const samples = [];
    for (let i = 0; i < 60; i++) {
      const signal = (i % 7) / 6 - 0.5;
      const lap = (i % 5) / 4;
      samples.push({ signal, lap, outcome: signal * 2 });
    }
    expect(contaminationTest(samples).contaminated).toBe(false);
  });

  it("does not flag a NEGATIVE interaction — that is not lookahead bias", () => {
    const samples = [];
    for (let i = 0; i < 60; i++) {
      const signal = (i % 7) / 6 - 0.5;
      const lap = (i % 5) / 4;
      samples.push({ signal, lap, outcome: signal * (2 - lap * 3) });
    }
    const v = contaminationTest(samples);
    expect(v.interaction).toBeLessThan(0);
    expect(v.contaminated).toBe(false);
  });
});

describe("olsWithInteraction", () => {
  it("returns an empty fit rather than garbage on too few points", () => {
    expect(olsWithInteraction([1, 2], [1, 2], [1, 2]).n).toBe(2);
    expect(olsWithInteraction([1, 2], [1, 2], [1, 2]).interaction).toBe(0);
  });

  it("recovers a known interaction coefficient", () => {
    const s: number[] = [], l: number[] = [], y: number[] = [];
    for (let i = 0; i < 50; i++) {
      const si = (i % 10) / 5 - 1, li = (i % 7) / 6;
      s.push(si); l.push(li);
      y.push(1 + 2 * si + 0.5 * li + 3 * si * li);
    }
    const fit = olsWithInteraction(s, l, y);
    expect(fit.interaction).toBeCloseTo(3, 4);
    expect(fit.signal).toBeCloseTo(2, 4);
  });
});

// ─── Baselines ────────────────────────────────────────────────────────

describe("lexiconSentiment", () => {
  it("scores financial polarity", () => {
    expect(lexiconSentiment("Revenue growth was strong and margins improved").score).toBeGreaterThan(0);
    expect(lexiconSentiment("Impairment and litigation drove a significant loss").score).toBeLessThan(0);
  });

  it("handles negation, which filings use constantly", () => {
    // "no impairment" is reassuring; a naive count reads it as negative.
    expect(lexiconSentiment("There was no impairment and no litigation").score).toBeGreaterThan(0);
  });

  it("reports 0 with matched=0 when nothing is sentiment-bearing", () => {
    const r = lexiconSentiment("The meeting is scheduled for Tuesday");
    expect(r.matched).toBe(0);
    expect(r.score).toBe(0);
  });
});

describe("numericSurprise", () => {
  it("standardises by the magnitude of the expectation", () => {
    expect(numericSurprise(1.2, 1.0)).toBeCloseTo(0.2, 9);
    expect(numericSurprise(0.8, 1.0)).toBeCloseTo(-0.2, 9);
  });

  it("returns null near zero expectations instead of exploding", () => {
    expect(numericSurprise(1, 0)).toBeNull();
    expect(numericSurprise(1, 1e-12)).toBeNull();
  });
});

describe("incrementalR2", () => {
  it("is ~0 when the candidate merely re-derives the baseline", () => {
    // The expected outcome: an expensive LLM reproducing a word count.
    const base = Array.from({ length: 40 }, (_, i) => (i % 9) / 4 - 1);
    const cand = base.map((b) => b * 1.01);
    const y = base.map((b) => b * 3);
    expect(incrementalR2(base, cand, y)).toBeLessThan(0.01);
  });

  it("is positive when the candidate carries genuinely new information", () => {
    const base = Array.from({ length: 40 }, (_, i) => (i % 9) / 4 - 1);
    const extra = Array.from({ length: 40 }, (_, i) => (i % 5) / 2 - 1);
    const y = base.map((b, i) => b + 2 * extra[i]);
    expect(incrementalR2(base, extra, y)).toBeGreaterThan(0.2);
  });
});

// ─── Pre-registration ─────────────────────────────────────────────────

describe("pre-registration", () => {
  const reg: PreRegistration = {
    study: "test-study",
    registeredAt: "2026-08-16",
    hypothesis: "H",
    prior: 0.3,
    abandonIf: "no incremental R2 over the lexicon",
    criteria: [
      { name: "a", statement: "A", threshold: 0.1, direction: "gte", rationale: "r" },
      { name: "b", statement: "B", threshold: 0.05, direction: "lte", rationale: "r" },
    ],
  };

  it("requires ALL criteria — a partial pass is a fail", () => {
    expect(evaluate(reg, { a: 0.2, b: 0.01 }).passed).toBe(true);
    const partial = evaluate(reg, { a: 0.2, b: 0.9 });
    expect(partial.passed).toBe(false);
    expect(partial.passedCount).toBe(1);
  });

  it("FAILS an unmeasured criterion rather than skipping it", () => {
    // Otherwise a study passes by quietly omitting its hardest test.
    const res = evaluate(reg, { a: 0.2 });
    expect(res.passed).toBe(false);
    expect(res.results[1].passed).toBe(false);
  });

  it("changes the hash when a threshold moves, so drift shows in a diff", () => {
    const moved = { ...reg, criteria: [{ ...reg.criteria[0], threshold: 0.05 }, reg.criteria[1]] };
    expect(registrationHash(moved)).not.toBe(registrationHash(reg));
  });

  it("keeps the hash stable when only the read date changes", () => {
    expect(registrationHash({ ...reg, registeredAt: "2027-01-01" })).toBe(registrationHash(reg));
  });
});
