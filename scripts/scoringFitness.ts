/**
 * Is a given model fit to score financial text at all?
 *
 * A reusable pre-flight, run BEFORE any expensive study. It asks the cheap
 * questions that decide whether a model can do the job, separately from whether
 * the job has any alpha in it:
 *
 *   1. COMPLIANCE   Does it return a bare number, or prose we must parse?
 *   2. POLARITY     Does it get the sign right on unambiguous text?
 *   3. ORDERING     Do its scores rank text the way a reader would?
 *   4. DISCRIMINATION  Does it use the scale, or cluster at 0 / ±1?
 *   5. BEATS LEXICON   Does it add anything over a word count?
 *
 * (5) is the one that matters. A model that only matches the lexicon on obvious
 * cases is not reading — it is an expensive sentiment proxy. So the cases are
 * split into OBVIOUS text, where a word count should also succeed, and SUBTLE
 * text (negation, hedged guidance, mixed news) where a word count should fail.
 * Winning on the subtle subset is the evidence that comprehension is happening.
 *
 * This test is NOT contaminated: it measures reading of text with known
 * polarity, not forecasting of realised outcomes, so no knowledge-cutoff
 * constraint applies.
 *
 * Usage:
 *   npx tsx scripts/scoringFitness.ts                     # defaults to the pinned Mac model
 *   npx tsx scripts/scoringFitness.ts --model=qwen32
 */

import { complete, parseScore } from "../src/llm/client.js";
import { lexiconSentiment } from "../src/eval/baseline.js";
import { spearman, stdDev, mean } from "../src/eval/stats.js";
import { MODELS, pinnedFor } from "../src/llm/models.js";

const arg = (n: string) => process.argv.find((a) => a.startsWith(`--${n}=`))?.split("=")[1];
const MODEL = arg("model") ?? pinnedFor("mac-studio") ?? "muse-glimmer-30b";

interface Case {
  text: string;
  /** Expected polarity on a -2..+2 ordinal scale, by human reading. */
  expected: number;
  /** SUBTLE cases are where a word count should fail and comprehension should win. */
  subtle: boolean;
  label: string;
}

const CASES: Case[] = [
  // ── Obvious: a lexicon should handle these too ──────────────────────
  { label: "strong beat", expected: 2, subtle: false,
    text: "Revenue grew 34% year over year, margins improved across every segment, and we are raising full-year guidance. Free cash flow reached a record high." },
  { label: "clear miss", expected: -2, subtle: false,
    text: "Revenue declined 18%, we recorded a goodwill impairment, and we are withdrawing full-year guidance amid deteriorating demand and ongoing litigation." },
  { label: "modest growth", expected: 1, subtle: false,
    text: "Revenue increased modestly and margins were slightly better than the prior quarter. Management expressed confidence in the outlook." },
  { label: "modest decline", expected: -1, subtle: false,
    text: "Revenue decreased slightly and margins were a little weaker. Management noted some pressure on pricing." },
  { label: "procedural", expected: 0, subtle: false,
    text: "The annual meeting of shareholders will be held on 14 March. Holders of record as of 1 February are entitled to vote. Proxy materials will be mailed." },

  // ── Subtle: a word count should get these WRONG ─────────────────────
  { label: "negated bad news", expected: 1, subtle: true,
    text: "The audit identified no material weakness, no impairment was required, and there is no outstanding litigation against the company." },
  { label: "negated good news", expected: -1, subtle: true,
    text: "We did not achieve the growth we targeted, margins did not improve, and we cannot confirm that the recovery we described last quarter is underway." },
  { label: "good words, bad substance", expected: -2, subtle: true,
    text: "Despite strong momentum and record engagement, the board has initiated a strategic review, suspended the dividend, and the chief executive will depart immediately." },
  { label: "bad words, good substance", expected: 2, subtle: true,
    text: "Having absorbed the restructuring costs, impairment charges and litigation expense entirely within this quarter, the company now enters the year with no remaining overhang and reinstates its dividend." },
  { label: "hedged guidance cut", expected: -1, subtle: true,
    text: "While we remain confident in the long-term opportunity, we now expect results toward the lower end of the range we previously communicated." },
  { label: "hedged guidance raise", expected: 1, subtle: true,
    text: "Although conditions remain uncertain and visibility is limited, results are trending toward the upper end of our previously communicated range." },
  { label: "mixed, net negative", expected: -1, subtle: true,
    text: "Subscriber growth exceeded expectations, but average revenue per user fell sharply and churn rose for the third consecutive quarter, which management expects to continue." },
  { label: "mixed, net positive", expected: 1, subtle: true,
    text: "Headline revenue fell owing to the divestment completed last year; on a like-for-like basis the continuing business grew 12% and order backlog reached a record." },
  { label: "boilerplate risk factors", expected: 0, subtle: true,
    text: "Our business faces risks including competition, regulatory change, supply disruption, cyber incidents and loss of key personnel, any of which could adversely affect results." },
];

const SYSTEM =
  "You are a financial analyst scoring text. Reply with ONLY a number between " +
  "-1 and 1. No words, no explanation, no units. -1 is maximally bearish for " +
  "the company's equity, 0 is neutral, +1 is maximally bullish.";

const pad = (s: string | number, w: number) => String(s).padStart(w);

async function main() {
  const spec = MODELS[MODEL];
  console.log(`\n${"═".repeat(96)}`);
  console.log(`Scoring fitness — ${MODEL}${spec?.pinned ? "  (pinned, free to call)" : ""}`);
  console.log(`${CASES.length} cases · ${CASES.filter((c) => c.subtle).length} subtle`);
  console.log("═".repeat(96));

  const rows: Array<Case & { score: number | null; raw: string; lex: number }> = [];
  for (const c of CASES) {
    const res = await complete({
      model: MODEL,
      system: SYSTEM,
      prompt: `Score this disclosure:\n\n"""${c.text}"""\n\nNumber only:`,
      temperature: 0,
      // Generous: reasoning models spend hundreds of tokens before answering.
      maxTokens: 2000,
    });
    rows.push({
      ...c,
      score: parseScore(res.text),
      raw: res.text.trim().replace(/\s+/g, " ").slice(0, 24),
      lex: lexiconSentiment(c.text).score,
    });
  }

  console.log("case".padEnd(26) + pad("want", 6) + pad("llm", 8) + pad("lexicon", 9) + "  raw");
  console.log("─".repeat(96));
  for (const r of rows) {
    const flag = r.subtle ? "*" : " ";
    console.log(
      `${flag}${r.label}`.padEnd(26) +
      pad(r.expected, 6) +
      pad(r.score === null ? "PARSE-FAIL" : r.score.toFixed(2), 8) +
      pad(r.lex.toFixed(2), 9) +
      `  ${r.raw}`
    );
  }
  console.log("─".repeat(96));

  const parsed = rows.filter((r) => r.score !== null);
  const parseRate = parsed.length / rows.length;

  const signOk = (a: number, b: number) => Math.sign(a) === Math.sign(b) || (a === 0 && Math.abs(b) < 0.15);
  const llmSign = parsed.filter((r) => signOk(r.expected, r.score!)).length / Math.max(1, parsed.length);
  const lexSign = rows.filter((r) => signOk(r.expected, r.lex)).length / rows.length;

  const rhoAll = spearman(parsed.map((r) => r.expected), parsed.map((r) => r.score!));
  const rhoLex = spearman(rows.map((r) => r.expected), rows.map((r) => r.lex));

  const sub = parsed.filter((r) => r.subtle);
  const subRows = rows.filter((r) => r.subtle);
  const rhoSubLlm = spearman(sub.map((r) => r.expected), sub.map((r) => r.score!));
  const rhoSubLex = spearman(subRows.map((r) => r.expected), subRows.map((r) => r.lex));
  const subSignLlm = sub.filter((r) => signOk(r.expected, r.score!)).length / Math.max(1, sub.length);
  const subSignLex = subRows.filter((r) => signOk(r.expected, r.lex)).length / subRows.length;

  const spread = stdDev(parsed.map((r) => r.score!));
  const atRails = parsed.filter((r) => Math.abs(r.score!) >= 0.99).length / Math.max(1, parsed.length);

  console.log(`\n1. COMPLIANCE     parse rate            ${(parseRate * 100).toFixed(0)}%`);
  console.log(`2. POLARITY       sign correct          LLM ${(llmSign * 100).toFixed(0)}%   lexicon ${(lexSign * 100).toFixed(0)}%`);
  console.log(`3. ORDERING       spearman vs human     LLM ${rhoAll.toFixed(3)}   lexicon ${rhoLex.toFixed(3)}`);
  console.log(`4. DISCRIMINATION score sd ${spread.toFixed(3)}   at rails ${(atRails * 100).toFixed(0)}%   mean ${mean(parsed.map((r) => r.score!)).toFixed(3)}`);
  console.log(`5. SUBTLE SUBSET  sign correct          LLM ${(subSignLlm * 100).toFixed(0)}%   lexicon ${(subSignLex * 100).toFixed(0)}%`);
  console.log(`                  spearman              LLM ${rhoSubLlm.toFixed(3)}   lexicon ${rhoSubLex.toFixed(3)}`);

  // The verdict deliberately hinges on the SUBTLE subset. Matching a word count
  // on obvious text proves nothing; the whole thesis is that comprehension adds
  // something a word count cannot.
  const verdict =
    parseRate < 0.9 ? "UNFIT — cannot reliably return a number"
    : spread < 0.05 ? "UNFIT — no discrimination, scores are effectively constant"
    : rhoSubLlm <= rhoSubLex ? "NOT WORTH IT — no better than a lexicon where it matters"
    : rhoSubLlm < 0.5 ? "WEAK — reads subtle text poorly"
    : "FIT — beats the lexicon on subtle text";

  console.log(`\n${"═".repeat(96)}`);
  console.log(`VERDICT: ${verdict}`);
  console.log("═".repeat(96) + "\n");
}

main().catch((e) => { console.error(e); process.exit(1); });
