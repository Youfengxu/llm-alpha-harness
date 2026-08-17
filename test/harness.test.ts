import { describe, it, expect } from "vitest";
import {
  assertScorable, isScorable, ContaminationError,
  assertSafeToCall, EvictionError, pinnedFor, MODELS, type ModelSpec,
} from "../src/llm/models.js";
import { parseScore } from "../src/llm/client.js";
import {
  effectiveSampleSize, effectiveSampleCeiling, meanPairwiseCorrelation,
  spearman, pearson, tTest, tTestEffective, olsWithInteraction,
} from "../src/eval/stats.js";
import { lexiconSentiment, numericSurprise, incrementalR2 } from "../src/eval/baseline.js";
import { evaluate, registrationHash, type PreRegistration } from "../src/eval/criteria.js";
import { calibrationCurve, estimateCutoff, contaminationTest, type LapResult } from "../src/eval/lap.js";
import { nextSessionAfter, filterFilings, htmlToText, findEarningsExhibit, type Filing } from "../src/data/edgar.js";

const model = (cutoff: string | null): ModelSpec => ({
  id: "test", baseUrl: "http://localhost/v1", cutoff, contextTokens: 8192, host: "mac-studio",
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

// ─── Pinned-model eviction guard ──────────────────────────────────────

describe("assertSafeToCall", () => {
  const spec = (over: Partial<ModelSpec>): ModelSpec => ({
    id: "m", baseUrl: "http://x/v1", cutoff: null, contextTokens: 8192,
    host: "mac-studio", ...over,
  });

  it("allows a model that evicts nothing", () => {
    expect(() => assertSafeToCall(spec({}))).not.toThrow();
  });

  it("allows a pinned model — it is already resident", () => {
    expect(() => assertSafeToCall(spec({ pinned: true }))).not.toThrow();
  });

  it("refuses a call that would evict a pinned model", () => {
    // A warning would be useless: eviction is invisible at the call site. The
    // request still succeeds, just slower, and the damage lands elsewhere.
    expect(() => assertSafeToCall(spec({ evicts: "muse-glimmer-30b" }))).toThrow(EvictionError);
  });

  it("proceeds when eviction is acknowledged explicitly", () => {
    expect(() =>
      assertSafeToCall(spec({ evicts: "muse-glimmer-30b" }), { acknowledgeEviction: true })
    ).not.toThrow();
  });

  it("names the pinned model per host so the error can suggest an alternative", () => {
    expect(pinnedFor("mac-studio")).toBe("muse-glimmer-30b");
    expect(pinnedFor("gx10")).toBe("gx10/Qwen3-Coder-Next-UD-Q4_K_M");
  });

  it("registry: exactly one pinned model per host", () => {
    for (const host of ["mac-studio", "gx10"] as const) {
      const pinned = Object.values(MODELS).filter((m) => m.host === host && m.pinned);
      expect(pinned).toHaveLength(1);
    }
  });

  it("registry: every non-pinned Mac model declares what it evicts", () => {
    // Forgetting this is how the guard silently stops protecting anything.
    for (const m of Object.values(MODELS)) {
      if (m.host === "mac-studio" && !m.pinned) expect(m.evicts).toBeTruthy();
    }
  });
});

// ─── EDGAR point-in-time discipline ───────────────────────────────────

describe("nextSessionAfter", () => {
  it("keeps a pre-market filing on the same day", () => {
    // 10:01Z ≈ 06:01 ET — well before the open.
    expect(nextSessionAfter("2026-07-31T10:01:02.000Z")).toBe("2026-07-31");
  });

  it("rolls a late-day filing to the next session", () => {
    // 22:30Z is after the close on any reading of the timestamp.
    expect(nextSessionAfter("2026-08-13T22:30:20.000Z")).toBe("2026-08-14");
  });

  it("rolls conservatively near the boundary, because the timezone is ambiguous", () => {
    // SEC stamps carry Z but the agency documents Eastern. 18:30Z is either
    // 14:30 ET (before close) or 18:30 ET (after). We assume the worse case.
    expect(nextSessionAfter("2026-08-12T18:30:00.000Z")).toBe("2026-08-13");
  });

  it("skips weekends", () => {
    // 2026-08-14 is a Friday; a late filing lands on Monday the 17th.
    expect(nextSessionAfter("2026-08-14T23:00:00.000Z")).toBe("2026-08-17");
    // Saturday filing → Monday.
    expect(nextSessionAfter("2026-08-15T10:00:00.000Z")).toBe("2026-08-17");
  });

  it("throws on an unparseable timestamp rather than silently returning a date", () => {
    expect(() => nextSessionAfter("not-a-time")).toThrow();
  });
});

describe("filterFilings", () => {
  const f = (over: Partial<Filing>): Filing => ({
    cik: 1, accessionNumber: "a", form: "8-K", filingDate: "2026-01-02",
    reportDate: "", acceptanceDateTime: "2026-01-02T12:00:00.000Z", items: "",
    primaryDocument: "d.htm", isXBRL: false, size: 0, ...over,
  });

  it("matches item codes exactly, not by substring", () => {
    // "2.02" must not match inside "12.02" — a substring test would.
    expect(filterFilings([f({ items: "12.02" })], { item: "2.02" })).toHaveLength(0);
    expect(filterFilings([f({ items: "2.02,9.01" })], { item: "2.02" })).toHaveLength(1);
    expect(filterFilings([f({ items: "9.01, 2.02" })], { item: "2.02" })).toHaveLength(1);
  });

  it("filters on acceptance time, not filing date", () => {
    const late = f({ filingDate: "2026-01-02", acceptanceDateTime: "2026-01-02T23:00:00.000Z" });
    expect(filterFilings([late], { from: "2026-01-02T00:00:00.000Z" })).toHaveLength(1);
    expect(filterFilings([late], { to: "2026-01-02T12:00:00.000Z" })).toHaveLength(0);
  });

  it("filters by form", () => {
    const rows = [f({ form: "8-K" }), f({ form: "10-Q" })];
    expect(filterFilings(rows, { forms: ["10-Q"] })).toHaveLength(1);
  });
});

describe("htmlToText", () => {
  it("drops script and style content so inline JS never reaches the model", () => {
    const out = htmlToText("<p>Revenue grew</p><script>var x=1;</script><style>.a{}</style>");
    expect(out).toContain("Revenue grew");
    expect(out).not.toContain("var x");
    expect(out).not.toContain(".a{");
  });

  it("decodes the entities filings actually use", () => {
    expect(htmlToText("<p>A&nbsp;&amp;&nbsp;B &lt;tag&gt;</p>")).toBe("A & B <tag>");
  });
});

describe("findEarningsExhibit", () => {
  const idx = (names: Array<[string, number]>) => names.map(([name, size]) => ({ name, size }));

  it("finds an exhibit whose name embeds ex99 mid-string", () => {
    // The real Apple case an anchored /^ex99/ regex silently missed.
    const r = findEarningsExhibit(
      idx([["aapl-20260730.htm", 38350], ["a8-kex991q3202606272026.htm", 173484], ["R1.htm", 55284]]),
      "aapl-20260730.htm"
    );
    expect(r).toEqual({ name: "a8-kex991q3202606272026.htm", isExhibit: true });
  });

  it("handles the common naming variants", () => {
    for (const n of ["ex-99.1.htm", "ex991.htm", "exhibit99_1.htm", "EX-99.1.HTM"]) {
      expect(findEarningsExhibit(idx([[n, 1000]]), "body.htm").name).toBe(n);
    }
  });

  it("prefers 99.1 over other 99.x exhibits even when smaller", () => {
    const r = findEarningsExhibit(
      idx([["ex-99.2.htm", 99999], ["ex-99.1.htm", 1000]]), "body.htm"
    );
    expect(r.name).toBe("ex-99.1.htm");
  });

  it("ignores XBRL rendering artifacts that are .htm but never prose", () => {
    const r = findEarningsExhibit(idx([["R1.htm", 55284], ["MetaLinks.json", 1]]), "body.htm");
    expect(r.isExhibit).toBe(false);
  });

  it("signals a fallback rather than silently returning the 8-K body", () => {
    // Callers must be able to DROP the event: the body is XBRL cover-page
    // tagging, which looks like a successful fetch and poisons a study.
    const r = findEarningsExhibit(idx([["aapl-20260730.htm", 38350]]), "aapl-20260730.htm");
    expect(r).toEqual({ name: "aapl-20260730.htm", isExhibit: false });
  });
});
