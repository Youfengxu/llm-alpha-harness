/**
 * STUDY 1 — Does an LLM's reading of an earnings release predict post-earnings
 * drift, INCREMENTALLY over cheap baselines?
 *
 * ── Pre-registered before any data was scored ─────────────────────────
 * Registered 2026-08-18. Hypothesis: an LLM score of the Exhibit 99.1 earnings
 * release carries information about market-adjusted drift over the following 20
 * sessions that a finance lexicon does not.
 *
 * Prior: 25-35%. All four criteria required:
 *
 *   1. DIRECTION      rank correlation between LLM score and market-adjusted
 *                     drift > 0.
 *   2. SIGNIFICANCE   p < 0.05 AFTER discounting for cross-sectional correlation.
 *                     The naive p-value is not admissible: the predecessor
 *                     project's central error was treating correlated
 *                     observations as independent.
 *   3. INCREMENTAL    incremental R-squared over the lexicon > 0.005. If the LLM
 *                     merely re-derives a word count, it is an expensive
 *                     sentiment proxy and the thesis is wrong.
 *   4. SPREAD         top-quintile minus bottom-quintile mean drift > 0.
 *
 * Abandon condition: criterion 3 fails. That would say comprehension adds
 * nothing over cheap text features, which is the entire premise.
 *
 * ── Controls ──────────────────────────────────────────────────────────
 * - Every event is strictly AFTER the model's measured knowledge cutoff;
 *   assertScorable throws otherwise.
 * - Returns start at nextSessionAfter(acceptanceDateTime): all 26 earnings 8-Ks
 *   in the pilot landed after the close, so a filingDate join would trade a full
 *   session early.
 * - Drift is market-adjusted; over 20 sessions the index move otherwise swamps
 *   anything firm-specific.
 * - Events whose EX-99.1 cannot be located are DROPPED, not scored on the 8-K
 *   body, which is XBRL cover-page tagging that looks like a successful fetch.
 * - Truncated drift windows are dropped, not shortened.
 */

import * as fs from "fs";
import * as path from "path";
import {
  fetchCompanyTickers, fetchSubmissions, filterFilings, fetchFilingIndex,
  fetchDocumentText, findEarningsExhibit, nextSessionAfter,
} from "../src/data/edgar.js";
import { fetchDailyBars, eventReturns } from "../src/data/prices.js";
import { getModel, assertScorable } from "../src/llm/models.js";

const arg = (n: string) => process.argv.find((a) => a.startsWith(`--${n}=`))?.split("=")[1];
export const MODEL = arg("model") ?? "gpt-oss-120b";
export const DRIFT_SESSIONS = parseInt(arg("drift") ?? "20", 10);
const BENCH = "SPY";
const OUT = path.resolve(process.cwd(), ".cache/study1");

/** ~120 liquid US names across sectors and sizes, fixed in advance.
 *  The literature reports the effect concentrating in SMALLER caps, so a
 *  mega-cap-only universe would test where it is weakest. */
export const UNIVERSE = `AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO ORCL CRM ADBE AMD INTC CSCO QCOM TXN
IBM NOW UBER ABNB SHOP SQ PYPL SNOW DDOG NET CRWD ZS PANW FTNT MDB TEAM WDAY
JPM BAC WFC GS MS C SCHW AXP BLK SPGI CME ICE COF USB PNC TFC
JNJ UNH PFE MRK ABBV LLY TMO ABT DHR BMY AMGN GILD CVS CI HUM ISRG
WMT COST TGT HD LOW NKE SBUX MCD YUM CMG DG DLTR ROST TJX
XOM CVX COP SLB EOG PSX VLO MPC OXY
CAT DE HON GE MMM LMT RTX BA UPS FDX UNP CSX NSC
PG KO PEP PM MO CL KMB GIS K HSY
DIS NFLX CMCSA T VZ TMUS CHTR WBD
LIN APD SHW ECL NEM FCX NUE
AMT PLD CCI EQIX SPG O PSA`.split(/\s+/).filter(Boolean);

export interface Event {
  ticker: string; cik: number; accession: string;
  accepted: string; session: string;
  text: string; textChars: number;
  drift: number | null; driftAdj: number | null; announcement: number | null;
}

async function main() {
  const model = getModel(MODEL);
  if (!model.cutoff) throw new Error(`${MODEL} has no established cutoff — run probeCutoff first`);
  fs.mkdirSync(OUT, { recursive: true });

  console.log(`\n${"═".repeat(96)}`);
  console.log(`STUDY 1 — earnings-release text vs ${DRIFT_SESSIONS}-session market-adjusted drift`);
  console.log(`model ${MODEL} · cutoff ${model.cutoff} · universe ${UNIVERSE.length} tickers`);
  console.log("═".repeat(96));

  const today = new Date().toISOString().slice(0, 10);
  const bench = await fetchDailyBars(BENCH, "2024-01-01", today);
  const all = await fetchCompanyTickers();

  const events: Event[] = [];
  let noCik = 0, noExhibit = 0, noWindow = 0, preCutoff = 0, scanned = 0;

  for (const t of UNIVERSE) {
    const c = all.find((x) => x.ticker === t);
    if (!c) { noCik++; continue; }
    let bars;
    try { bars = await fetchDailyBars(t, "2024-01-01", today); }
    catch { noCik++; continue; }

    const subs = await fetchSubmissions(c.cik);
    const filings = filterFilings(subs.filings, {
      forms: ["8-K"], item: "2.02", from: `${model.cutoff}T23:59:59.000Z`,
    });

    for (const f of filings) {
      scanned++;
      const session = nextSessionAfter(f.acceptanceDateTime);
      try { assertScorable(model, session); } catch { preCutoff++; continue; }

      const r = eventReturns(bars, session, DRIFT_SESSIONS, bench);
      if (r.drift === null || r.driftMarketAdj === null) { noWindow++; continue; }

      let text = "";
      try {
        const index = await fetchFilingIndex(f);
        const pick = findEarningsExhibit(index, f.primaryDocument);
        // Dropping rather than falling back to the 8-K body: the body is XBRL
        // tagging that would look like a successful fetch and poison the sample.
        if (!pick.isExhibit) { noExhibit++; continue; }
        text = await fetchDocumentText(f, pick.name);
      } catch { noExhibit++; continue; }
      if (text.length < 1200) { noExhibit++; continue; }

      events.push({
        ticker: t, cik: c.cik, accession: f.accessionNumber,
        accepted: f.acceptanceDateTime, session,
        text, textChars: text.length,
        drift: r.drift, driftAdj: r.driftMarketAdj, announcement: r.announcement,
      });
    }
    if (events.length && events.length % 50 === 0) process.stdout.write(`  ${events.length} events\r`);
  }

  fs.writeFileSync(path.join(OUT, "events.json"), JSON.stringify(events));
  console.log(`\nscanned ${scanned} earnings 8-Ks post-cutoff`);
  console.log(`  dropped: ${noCik} no CIK/prices · ${preCutoff} pre-cutoff · ${noWindow} no full drift window · ${noExhibit} no EX-99.1`);
  console.log(`  USABLE: ${events.length} events across ${new Set(events.map((e) => e.ticker)).size} tickers`);
  const chars = events.map((e) => e.textChars).sort((a, b) => a - b);
  if (chars.length) {
    console.log(`  text length: median ${chars[Math.floor(chars.length / 2)].toLocaleString()} chars, ` +
      `max ${chars[chars.length - 1].toLocaleString()}`);
  }
  console.log(`\nwritten to ${path.join(OUT, "events.json")}`);
}

// Only gather when run directly. scoreEvents.ts and analyse.ts import MODEL,
// DRIFT_SESSIONS and the Event type from here; without this guard, importing
// them would silently re-run the entire EDGAR crawl.
const invokedDirectly = process.argv[1]?.endsWith("earningsDrift.ts");
if (invokedDirectly) main().catch((e) => { console.error(String(e).slice(0, 500)); process.exit(1); });
