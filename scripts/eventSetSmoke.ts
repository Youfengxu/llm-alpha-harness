/**
 * The full Option B pipeline, end to end, with no LLM in it yet.
 *
 * EDGAR earnings 8-K → tradeable session → announcement and drift returns,
 * market-adjusted. This is the event set a text signal would have to beat, and
 * building it BEFORE any scoring is deliberate: if the event set is thin, or the
 * drift is not there, no amount of model quality rescues the study.
 *
 * It reports the two numbers that decide whether Option B is worth running:
 *
 *   USABLE EVENTS  after the model's knowledge cutoff, with a full drift window
 *   DRIFT SPREAD   whether events sort at all on the crudest possible signal
 *
 * The second uses the SIGN OF THE ANNOUNCEMENT RETURN as a stand-in signal —
 * the cheapest imaginable predictor, and free of any text. If drift does not
 * separate on that, post-earnings-announcement drift is absent in this sample
 * and the LLM has nothing to add to.
 *
 * Usage:
 *   SEC_USER_AGENT="project you@example.com" npx tsx scripts/eventSetSmoke.ts
 *   ... --tickers=AAPL,NVDA,MSFT,AMZN --drift=20
 */

import {
  fetchCompanyTickers, fetchSubmissions, filterFilings, nextSessionAfter,
} from "../src/data/edgar.js";
import { fetchDailyBars, eventReturns, coverage } from "../src/data/prices.js";
import { getModel } from "../src/llm/models.js";
import { mean, stdDev, tTest, tTestEffective, effectiveSampleSize } from "../src/eval/stats.js";

const arg = (n: string) => process.argv.find((a) => a.startsWith(`--${n}=`))?.split("=")[1];
const TICKERS = (arg("tickers") ?? "AAPL,NVDA,MSFT,AMZN,GOOGL,META").split(",");
const DRIFT = parseInt(arg("drift") ?? "20", 10);
const MODEL = arg("model") ?? "gpt-oss-120b";
const BENCH = arg("bench") ?? "SPY";
const FROM = arg("from") ?? "2023-01-01";

const pct = (n: number | null) => (n === null ? "    -  " : `${n >= 0 ? "+" : ""}${(n * 100).toFixed(2)}%`);
const pad = (s: string | number, w: number) => String(s).padEnd(w);

interface Event {
  ticker: string;
  accepted: string;
  session: string;
  announcement: number | null;
  drift: number | null;
  driftAdj: number | null;
  usable: boolean;
}

async function main() {
  const model = getModel(MODEL);
  console.log(`\n${"═".repeat(96)}`);
  console.log(`Option B event set — earnings 8-K → ${DRIFT}-session drift, market-adjusted vs ${BENCH}`);
  console.log(`model ${MODEL} · cutoff ${model.cutoff ?? "NOT ESTABLISHED"}`);
  console.log("═".repeat(96));

  const today = new Date().toISOString().slice(0, 10);
  const bench = await fetchDailyBars(BENCH, FROM, today);
  const bc = coverage(bench);
  console.log(`\n${BENCH}: ${bc.sessions} sessions ${bc.first} → ${bc.last}` +
    (bc.gapsOver3d ? `  ⚠ ${bc.gapsOver3d} gaps > 4d` : "  contiguous"));

  const all = await fetchCompanyTickers();
  const events: Event[] = [];

  console.log("\nticker  events  usable  coverage");
  console.log("─".repeat(96));
  for (const t of TICKERS) {
    const c = all.find((x) => x.ticker === t.toUpperCase());
    if (!c) { console.log(`${pad(t, 8)}not found on EDGAR`); continue; }

    const subs = await fetchSubmissions(c.cik);
    const filings = filterFilings(subs.filings, {
      forms: ["8-K"], item: "2.02", from: `${FROM}T00:00:00.000Z`,
    });
    const bars = await fetchDailyBars(t, FROM, today);
    const cov = coverage(bars);

    let usable = 0;
    for (const f of filings) {
      const session = nextSessionAfter(f.acceptanceDateTime);
      const r = eventReturns(bars, session, DRIFT, bench);
      // Usable requires BOTH a full drift window and a post-cutoff date. A
      // partial window is dropped, not shortened.
      const isUsable = r.drift !== null && !!model.cutoff && session > model.cutoff;
      if (isUsable) usable++;
      events.push({
        ticker: t, accepted: f.acceptanceDateTime, session,
        announcement: r.announcement, drift: r.drift, driftAdj: r.driftMarketAdj,
        usable: isUsable,
      });
    }
    console.log(`${pad(t, 8)}${pad(filings.length, 8)}${pad(usable, 8)}${cov.sessions} sessions ${cov.first} → ${cov.last}`);
  }

  const usable = events.filter((e) => e.usable && e.announcement !== null && e.driftAdj !== null);
  console.log(`\n${"─".repeat(96)}`);
  console.log(`${events.length} events total · ${usable.length} usable (post-cutoff, full ${DRIFT}-session window)`);
  console.log("─".repeat(96));

  if (usable.length < 8) {
    console.log("\nToo few usable events to say anything. Widen the ticker list or the date range.\n");
    return;
  }

  console.log("\nrecent usable events:");
  console.log("  ticker  session      announcement    drift   drift-adj");
  for (const e of usable.slice(-8)) {
    console.log(`  ${pad(e.ticker, 8)}${pad(e.session, 13)}${pad(pct(e.announcement), 16)}${pad(pct(e.drift), 9)}${pct(e.driftAdj)}`);
  }

  // ── Does drift sort on the crudest possible signal? ────────────────
  const up = usable.filter((e) => e.announcement! > 0).map((e) => e.driftAdj!);
  const down = usable.filter((e) => e.announcement! <= 0).map((e) => e.driftAdj!);
  const spread = mean(up) - mean(down);
  const t = tTest([...up.map((x) => x), ...down.map((x) => -x)]);

  console.log(`\n${"─".repeat(96)}`);
  console.log("PEAD check — does drift separate on the sign of the announcement return?");
  console.log("─".repeat(96));
  console.log(`  positive reaction  n=${pad(up.length, 5)} mean drift-adj ${pct(mean(up))}   sd ${(stdDev(up) * 100).toFixed(2)}%`);
  console.log(`  negative reaction  n=${pad(down.length, 5)} mean drift-adj ${pct(mean(down))}   sd ${(stdDev(down) * 100).toFixed(2)}%`);
  console.log(`  spread ${pct(spread)}   t=${t.t.toFixed(2)}  p=${t.pValue.toFixed(3)}  (naive, assumes independence)`);

  // The naive p-value assumes 65 independent observations. These are 8
  // mega-caps whose residuals still co-move after market adjustment, and
  // same-quarter events share a macro backdrop. The predecessor project's
  // central error was exactly this: treating correlated observations as
  // independent, which turned noise into significance. Rather than guess one
  // correlation, show how fast the result decays.
  const signed = [...up, ...down.map((x) => -x)];
  console.log("\n  sensitivity to residual cross-sectional correlation:");
  console.log("    rho     n_eff     p");
  for (const rho of [0, 0.05, 0.1, 0.2, 0.3]) {
    const adj = tTestEffective(signed, rho);
    const flag = adj.pValue < 0.05 ? "" : "   <- no longer significant";
    console.log(`    ${rho.toFixed(2)}   ${effectiveSampleSize(signed.length, rho).toFixed(1).padStart(6)}   ${adj.pValue.toFixed(3)}${flag}`);
  }
  console.log(
    `\n  ${spread > 0 && t.pValue < 0.10
      ? "Drift sorts in the documented DIRECTION, so there is plausibly something for a\n  text signal to add to — but see the sensitivity table above before believing it."
      : "Drift does NOT sort here. Before blaming a model, note that the anomaly itself\n  is absent in this sample — a text signal would have nothing to improve on."}`
  );
  console.log("\n  This is 8 mega-caps over 2 years. Treat it as a PIPELINE CHECK, not evidence\n" +
    "  about PEAD: the sample is small, narrow, and the horizon was chosen by hand.");
  console.log("═".repeat(96) + "\n");
}

main().catch((e) => { console.error(String(e).slice(0, 600)); process.exit(1); });
