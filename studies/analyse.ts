/**
 * Phase 3 — evaluate against the criteria pre-registered in earningsDrift.ts.
 * Nothing here was chosen after seeing a result.
 */
import * as fs from "fs";
import * as path from "path";
import {
  spearman, pearson, mean, stdDev, tTest, tTestEffective,
  effectiveSampleSize, meanPairwiseCorrelation,
} from "../src/eval/stats.js";
import { incrementalR2 } from "../src/eval/baseline.js";
import { evaluate, formatResult, type PreRegistration } from "../src/eval/criteria.js";
import { DRIFT_SESSIONS, type Event } from "./earningsDrift.js";

const OUT = path.resolve(process.cwd(), ".cache/study1");

const REG: PreRegistration = {
  study: "Study 1 — earnings-release text vs post-earnings drift",
  registeredAt: "2026-08-18",
  hypothesis:
    "An LLM score of the Exhibit 99.1 earnings release carries information about " +
    `market-adjusted drift over the following ${DRIFT_SESSIONS} sessions that a finance lexicon does not.`,
  prior: 0.30,
  abandonIf: "incremental R-squared over the lexicon is not positive — comprehension adds nothing over cheap text features",
  criteria: [
    { name: "direction", statement: "Rank correlation between LLM score and market-adjusted drift is positive.",
      threshold: 0, direction: "gte", rationale: "The sign must be right before magnitude matters." },
    { name: "significance_eff", statement: "p < 0.05 after discounting for cross-sectional correlation.",
      threshold: 0.05, direction: "lte", rationale: "Naive p-values treat correlated events as independent — the predecessor project's central error." },
    { name: "incremental_r2", statement: "Incremental R-squared of the LLM score over the lexicon exceeds 0.005.",
      threshold: 0.005, direction: "gte", rationale: "If the LLM re-derives a word count it is an expensive sentiment proxy and the thesis is wrong." },
    { name: "quintile_spread", statement: "Top-quintile minus bottom-quintile mean drift is positive.",
      threshold: 0, direction: "gte", rationale: "A tradeable form of the same claim, robust to outliers." },
  ],
};

function main() {
  const events: Event[] = JSON.parse(fs.readFileSync(path.join(OUT, "events.json"), "utf8"));
  const raw: Array<{ i: number; llm: number | null; lex: number }> =
    JSON.parse(fs.readFileSync(path.join(OUT, "scores.json"), "utf8"));

  const rows = raw
    .filter((s) => s.llm !== null)
    .map((s) => ({ ...events[s.i], llm: s.llm as number, lex: s.lex }))
    .filter((r) => r.driftAdj !== null);

  const y = rows.map((r) => r.driftAdj as number);
  const llm = rows.map((r) => r.llm);
  const lex = rows.map((r) => r.lex);

  console.log(`\n${"═".repeat(92)}`);
  console.log(`${rows.length} scored events · ${new Set(rows.map((r) => r.ticker)).size} tickers`);
  console.log(`LLM score: mean ${mean(llm).toFixed(3)} sd ${stdDev(llm).toFixed(3)} · ` +
    `lexicon: mean ${mean(lex).toFixed(3)} sd ${stdDev(lex).toFixed(3)}`);
  console.log(`drift: mean ${(mean(y) * 100).toFixed(2)}% sd ${(stdDev(y) * 100).toFixed(2)}%`);
  console.log("═".repeat(92));

  const rhoLlm = spearman(llm, y), rhoLex = spearman(lex, y);
  console.log(`\nrank correlation with market-adjusted drift`);
  console.log(`  LLM      ${rhoLlm >= 0 ? "+" : ""}${rhoLlm.toFixed(4)}`);
  console.log(`  lexicon  ${rhoLex >= 0 ? "+" : ""}${rhoLex.toFixed(4)}`);
  console.log(`  LLM vs lexicon agreement: r = ${pearson(llm, lex).toFixed(3)}`);

  // Quintiles.
  const sorted = [...rows].sort((a, b) => a.llm - b.llm);
  const q = Math.max(1, Math.floor(sorted.length / 5));
  const bot = sorted.slice(0, q).map((r) => r.driftAdj as number);
  const top = sorted.slice(-q).map((r) => r.driftAdj as number);
  const spread = mean(top) - mean(bot);
  console.log(`\nquintiles by LLM score (n=${q} each)`);
  console.log(`  bottom  ${(mean(bot) * 100).toFixed(2)}%`);
  console.log(`  top     ${(mean(top) * 100).toFixed(2)}%`);
  console.log(`  spread  ${(spread * 100).toFixed(2)}%`);

  // Effective sample size: events cluster in earnings season and share a macro
  // backdrop, so they are not independent. Estimate rho from per-ticker mean
  // drift co-movement across quarters.
  const byQuarter = new Map<string, number[]>();
  for (const r of rows) {
    const k = `${r.session.slice(0, 4)}Q${Math.ceil(+r.session.slice(5, 7) / 3)}`;
    byQuarter.set(k, [...(byQuarter.get(k) ?? []), r.driftAdj as number]);
  }
  const qs = Array.from(byQuarter.values()).filter((v) => v.length >= 5);
  const minLen = Math.min(...qs.map((v) => v.length));
  const rho = qs.length >= 2 ? Math.max(0, meanPairwiseCorrelation(qs.map((v) => v.slice(0, minLen)))) : 0.3;

  const signed = rows.map((r) => (r.llm >= mean(llm) ? 1 : -1) * (r.driftAdj as number));
  const naive = tTest(signed);
  const adj = tTestEffective(signed, rho);
  console.log(`\nsignificance`);
  console.log(`  naive           t=${naive.t.toFixed(2)}  p=${naive.pValue.toFixed(3)}   (n=${signed.length})`);
  console.log(`  rho estimate    ${rho.toFixed(3)} (from ${qs.length} quarterly cohorts)`);
  console.log(`  effective n     ${effectiveSampleSize(signed.length, rho).toFixed(1)}`);
  console.log(`  DISCOUNTED      t=${adj.t.toFixed(2)}  p=${adj.pValue.toFixed(3)}`);

  const incR2 = incrementalR2(lex, llm, y);
  const incR2rev = incrementalR2(llm, lex, y);
  console.log(`\nincremental information`);
  console.log(`  LLM over lexicon   R2 += ${incR2.toFixed(4)}`);
  console.log(`  lexicon over LLM   R2 += ${incR2rev.toFixed(4)}`);

  const res = evaluate(REG, {
    direction: rhoLlm,
    significance_eff: adj.pValue,
    incremental_r2: incR2,
    quintile_spread: spread,
  });
  console.log("\n" + formatResult(REG, res));
}
main();
