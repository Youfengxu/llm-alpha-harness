/**
 * Establish a model's knowledge cutoff empirically, by finding where recall
 * collapses.
 *
 * ── Why probe rather than read the model card ─────────────────────────
 * Every control in this harness rests on the cutoff being right, and
 * `assertScorable` refuses to evaluate without one. A date copied from a model
 * card is a claim about the base model, not about whatever fine-tuning,
 * merging or distillation produced the weights actually being served — and for
 * a locally merged model there may be no card at all. Since a WRONG cutoff is
 * worse than none (it silently launders contaminated samples into apparently
 * clean ones), it is measured here.
 *
 * ── Method ────────────────────────────────────────────────────────────
 * Lookahead Propensity is materially positive across the training period and
 * collapses to ~0 immediately after it. So: sweep dates, probe several
 * well-known entities at each, and find where the collapse begins and persists.
 *
 * Two stages, because a reasoning model costs ~15s per probe and a monthly
 * sweep over three years would take an hour:
 *   1. COARSE, quarterly, wide range — locate the collapse.
 *   2. FINE, monthly, around it — pin it down.
 *
 * ── The control that makes a null result meaningful ───────────────────
 * A model with weak recall of everything would show low LAP at every date and
 * produce a spuriously early "cutoff". So the sweep includes CONTROL dates far
 * inside any plausible training window. If LAP is not high there, the probe is
 * not working for this model and the run reports that instead of a cutoff.
 *
 * Entities are deliberately high-profile: recall is likeliest where coverage is
 * densest, which sharpens the collapse. Obscure names would blur it.
 *
 * Usage:
 *   npx tsx scripts/probeCutoff.ts --model=muse-glimmer-30b
 *   npx tsx scripts/probeCutoff.ts --model=X --from=2024-01 --to=2026-09
 */

import { measureLapBatch, calibrationCurve, estimateCutoff, type LapProbe } from "../src/eval/lap.js";
import { MODELS, pinnedFor } from "../src/llm/models.js";
import { mean } from "../src/eval/stats.js";

const arg = (n: string) => process.argv.find((a) => a.startsWith(`--${n}=`))?.split("=")[1];
const MODEL = arg("model") ?? pinnedFor("mac-studio") ?? "muse-glimmer-30b";
const FROM = arg("from") ?? "2023-01";
const TO = arg("to") ?? "2026-09";
const THRESHOLD = parseFloat(arg("threshold") ?? "0.15");

/** Dense-coverage entities, so weak recall means "after the cutoff", not "obscure". */
const ENTITIES = (arg("entities") ??
  "Apple Inc (AAPL),Microsoft Corporation (MSFT),Tesla Inc (TSLA),Nvidia Corporation (NVDA)").split(",");

/** Deep inside any plausible training window. LAP must be high here or the probe is broken. */
const CONTROL_DATES = ["2021-06", "2022-03"];

function monthsBetween(from: string, to: string, stepMonths: number): string[] {
  const out: string[] = [];
  const [fy, fm] = from.split("-").map(Number);
  const [ty, tm] = to.split("-").map(Number);
  let y = fy, m = fm;
  while (y < ty || (y === ty && m <= tm)) {
    out.push(`${y}-${String(m).padStart(2, "0")}`);
    m += stepMonths;
    while (m > 12) { m -= 12; y++; }
  }
  return out;
}

const probesFor = (months: string[]): LapProbe[] =>
  months.flatMap((mo) => ENTITIES.map((entity) => ({ entity, date: `${mo}-15` })));

const bar = (v: number) => "█".repeat(Math.round(Math.max(0, v) * 30));

async function sweep(label: string, months: string[]) {
  const probes = probesFor(months);
  process.stdout.write(`${label}: ${probes.length} probes (${months.length} periods x ${ENTITIES.length} entities)... `);
  const t0 = Date.now();
  const results = await measureLapBatch(MODEL, probes);
  console.log(`${((Date.now() - t0) / 1000).toFixed(0)}s\n`);
  const curve = calibrationCurve(results, (d) => d.slice(0, 7));
  for (const b of curve) {
    const v = Number.isFinite(b.meanLap) ? b.meanLap : 0;
    console.log(`  ${b.date}  ${v.toFixed(2)}  ${bar(v)}${b.n < ENTITIES.length ? `  (${b.n}/${ENTITIES.length} parsed)` : ""}`);
  }
  return { curve, results };
}

async function main() {
  const spec = MODELS[MODEL];
  console.log(`\n${"═".repeat(78)}`);
  console.log(`Knowledge-cutoff probe — ${MODEL}`);
  console.log(`entities: ${ENTITIES.join(", ")}`);
  console.log(`recorded cutoff: ${spec?.cutoff ?? "null (not established)"}`);
  console.log("═".repeat(78) + "\n");

  // ── Control ────────────────────────────────────────────────────────
  const control = await sweep("CONTROL (must show high recall)", CONTROL_DATES);
  const controlMean = mean(control.curve.map((b) => b.meanLap).filter(Number.isFinite));
  console.log(`\n  control mean LAP = ${controlMean.toFixed(2)}`);
  if (controlMean < 0.4) {
    console.log(
      `\n${"═".repeat(78)}\n` +
      `PROBE NOT WORKING for this model.\n\n` +
      `Recall is weak (${controlMean.toFixed(2)}) even at control dates deep inside any\n` +
      `plausible training window. A sweep would show low LAP everywhere and yield a\n` +
      `spuriously early "cutoff", so NO cutoff is reported.\n\n` +
      `Likely causes: the model does not self-report recall reliably, or it is not\n` +
      `knowledgeable about equities. Try different entities, or a model whose recall\n` +
      `the probe can measure.\n` +
      `${"═".repeat(78)}\n`
    );
    return;
  }

  // ── Coarse ─────────────────────────────────────────────────────────
  console.log("");
  const coarse = await sweep("COARSE (quarterly)", monthsBetween(FROM, TO, 3));
  const coarseCut = estimateCutoff(coarse.curve, THRESHOLD);
  console.log(`\n  coarse collapse: ${coarseCut ?? "none found"}`);

  if (!coarseCut) {
    console.log(
      `\n${"═".repeat(78)}\n` +
      `NO COLLAPSE FOUND in ${FROM}..${TO}.\n\n` +
      `Recall never persistently drops below ${THRESHOLD}, so the cutoff is not\n` +
      `identifiable from this sweep and MUST NOT be recorded. Either it lies after\n` +
      `${TO} — widen --to — or this model's self-reported recall does not collapse\n` +
      `sharply enough to locate.\n` +
      `${"═".repeat(78)}\n`
    );
    return;
  }

  // ── Fine ───────────────────────────────────────────────────────────
  const [cy, cm] = coarseCut.split("-").map(Number);
  const start = new Date(Date.UTC(cy, cm - 1 - 4, 1));
  const end = new Date(Date.UTC(cy, cm - 1 + 2, 1));
  const fineFrom = `${start.getUTCFullYear()}-${String(start.getUTCMonth() + 1).padStart(2, "0")}`;
  const fineTo = `${end.getUTCFullYear()}-${String(end.getUTCMonth() + 1).padStart(2, "0")}`;
  console.log("");
  const fine = await sweep(`FINE (monthly, ${fineFrom}..${fineTo})`, monthsBetween(fineFrom, fineTo, 1));
  const fineCut = estimateCutoff(fine.curve, THRESHOLD);

  console.log(`\n${"═".repeat(78)}`);
  if (!fineCut) {
    console.log(
      `Coarse sweep collapsed at ${coarseCut} but the monthly sweep does not confirm it.\n` +
      `That inconsistency means the boundary is not sharp; NO cutoff recorded.`
    );
  } else {
    // The last period still showing recall is the last SAFE-to-assume-known one;
    // the cutoff is the boundary before the collapse begins.
    const idx = fine.curve.findIndex((b) => b.date === fineCut);
    const lastKnown = idx > 0 ? fine.curve[idx - 1].date : fineFrom;
    console.log(`ESTIMATED CUTOFF: recall collapses at ${fineCut}; last period with recall is ${lastKnown}.`);
    console.log(`\nRecord in src/llm/models.ts:`);
    console.log(`    cutoff: "${lastKnown}-28",`);
    console.log(`    cutoffSource: "probeCutoff.ts, ${ENTITIES.length} entities, LAP collapse at ${fineCut}",`);
    console.log(`\nUse the END of the last recalled period, not the collapse month — it is the`);
    console.log(`conservative choice, and erring late costs samples while erring early admits`);
    console.log(`contaminated ones.`);
  }
  console.log("═".repeat(78) + "\n");
}

main().catch((e) => { console.error(e); process.exit(1); });
