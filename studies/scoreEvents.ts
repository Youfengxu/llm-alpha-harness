/**
 * Phase 2 — score each earnings release with the LLM and with the lexicon.
 *
 * ── The prompt asks about SURPRISE, not sentiment ─────────────────────
 * Post-earnings drift is under-reaction to news relative to expectations. A
 * release can be full of good numbers and still be a negative surprise, and the
 * drift follows the surprise, not the tone. Asking "is this positive?" would
 * measure something already in the price by the time anyone could trade it.
 *
 * The model is given no price data, no date, and no outcome — only the text. It
 * cannot infer the market reaction from anything supplied.
 */

import * as fs from "fs";
import * as path from "path";
import { complete, parseScore } from "../src/llm/client.js";
import { lexiconSentiment } from "../src/eval/baseline.js";
import { MODEL, type Event } from "./earningsDrift.js";

const OUT = path.resolve(process.cwd(), ".cache/study1");
/** ~7.5k tokens, comfortably inside the 16k served context with room for output. */
const MAX_CHARS = 30_000;

const SYSTEM =
  "You are an equity analyst. You will be shown a company's earnings press " +
  "release. Judge ONLY how the information compares with what the market was " +
  "likely expecting beforehand. Reply with a single number between -1 and 1: " +
  "-1 = a large negative surprise, 0 = broadly in line, +1 = a large positive " +
  "surprise. No words, no explanation.";

async function main() {
  const events: Event[] = JSON.parse(fs.readFileSync(path.join(OUT, "events.json"), "utf8"));
  console.log(`scoring ${events.length} events with ${MODEL}\n`);

  const scores: Array<{ i: number; llm: number | null; lex: number; raw: string }> = [];
  let done = 0, failed = 0;
  const t0 = Date.now();

  const CONC = parseInt(process.env.LLM_CONCURRENCY ?? "24", 10);
  for (let start = 0; start < events.length; start += CONC) {
    const batch = events.slice(start, start + CONC);
    const out = await Promise.all(batch.map(async (e, k) => {
      const i = start + k;
      try {
        const res = await complete({
          model: MODEL,
          system: SYSTEM,
          prompt: `Earnings release:\n\n"""${e.text.slice(0, MAX_CHARS)}"""\n\nSurprise score (number only):`,
          temperature: 0, maxTokens: 2000,
        });
        return { i, llm: parseScore(res.text), lex: lexiconSentiment(e.text).score, raw: res.text.trim().slice(0, 20) };
      } catch {
        return { i, llm: null, lex: lexiconSentiment(e.text).score, raw: "ERROR" };
      }
    }));
    for (const o of out) { scores.push(o); if (o.llm === null) failed++; }
    done += batch.length;
    const rate = done / ((Date.now() - t0) / 1000);
    process.stdout.write(`  ${done}/${events.length}  ${rate.toFixed(1)}/s  ${failed} unparsed\r`);
  }
  fs.writeFileSync(path.join(OUT, "scores.json"), JSON.stringify(scores));
  console.log(`\n\n${scores.length} scored, ${failed} unparsed, ${((Date.now() - t0) / 60000).toFixed(1)} min`);
}
main().catch((e) => { console.error(e); process.exit(1); });
