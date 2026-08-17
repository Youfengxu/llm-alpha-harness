/**
 * End-to-end check of the EDGAR connector against live filings.
 *
 * Proves the three properties the harness depends on, rather than asserting
 * them: history is COMPLETE (older pages followed), timestamps are POINT-IN-TIME
 * (acceptance, not filing date), and documents are RETRIEVABLE as text.
 *
 * Also reports how many events fall after each registered model's knowledge
 * cutoff, which is the number that decides whether a study is possible at all.
 *
 * Usage:
 *   SEC_USER_AGENT="project you@example.com" npx tsx scripts/edgarSmoke.ts
 *   ... --tickers=AAPL,NVDA --from=2024-07-01
 */

import {
  fetchCompanyTickers, fetchSubmissions, filterFilings, fetchFilingIndex,
  fetchDocumentText, nextSessionAfter, cacheSize, findEarningsExhibit,
} from "../src/data/edgar.js";
import { MODELS } from "../src/llm/models.js";

const arg = (n: string) => process.argv.find((a) => a.startsWith(`--${n}=`))?.split("=")[1];
const TICKERS = (arg("tickers") ?? "AAPL,NVDA,MSFT").split(",");
const FROM = arg("from") ?? "2024-07-01T00:00:00.000Z";

const pad = (s: string | number, w: number) => String(s).padEnd(w);

async function main() {
  console.log(`\n${"═".repeat(92)}`);
  console.log("EDGAR connector — end-to-end check");
  console.log("═".repeat(92));

  const t0 = Date.now();
  const all = await fetchCompanyTickers();
  console.log(`\ncompany_tickers.json: ${all.length} tickers\n`);

  const picked = TICKERS.map((t) => {
    const c = all.find((x) => x.ticker === t.toUpperCase());
    if (!c) throw new Error(`ticker not found on EDGAR: ${t}`);
    return c;
  });

  console.log("ticker  CIK        filings   pages   earliest      latest        8-K 2.02");
  console.log("─".repeat(92));

  const earnings8k: Array<{ ticker: string; accepted: string; session: string; acc: string; cik: number }> = [];

  for (const c of picked) {
    const subs = await fetchSubmissions(c.cik);
    const dates = subs.filings.map((f) => f.filingDate).filter(Boolean).sort();
    // 8-K Item 2.02 is "Results of Operations and Financial Condition" — the
    // earnings release. This is the event set Option B is built on.
    const results = filterFilings(subs.filings, { forms: ["8-K"], item: "2.02" });
    console.log(
      pad(c.ticker, 8) + pad(c.cik, 11) + pad(subs.filings.length, 10) +
      pad(subs.pagesRead, 8) + pad(dates[0] ?? "-", 14) +
      pad(dates[dates.length - 1] ?? "-", 14) + results.length
    );
    for (const f of filterFilings(results, { from: FROM })) {
      earnings8k.push({
        ticker: c.ticker, accepted: f.acceptanceDateTime,
        session: nextSessionAfter(f.acceptanceDateTime),
        acc: f.accessionNumber, cik: f.cik,
      });
    }
  }

  // ── Point-in-time behaviour on real events ────────────────────────
  console.log(`\n${"─".repeat(92)}`);
  console.log(`Earnings 8-Ks accepted since ${FROM.slice(0, 10)} — acceptance vs tradeable session`);
  console.log("─".repeat(92));
  earnings8k.sort((a, b) => a.accepted.localeCompare(b.accepted));
  for (const e of earnings8k.slice(-8)) {
    const rolled = e.accepted.slice(0, 10) !== e.session;
    console.log(
      `  ${pad(e.ticker, 7)}accepted ${e.accepted.slice(0, 16).replace("T", " ")}Z  →  ` +
      `session ${e.session}${rolled ? "   (rolled forward)" : ""}`
    );
  }
  console.log(`\n  ${earnings8k.length} events total, ` +
    `${earnings8k.filter((e) => e.accepted.slice(0, 10) !== e.session).length} rolled to a later session`);

  // ── Document retrieval ────────────────────────────────────────────
  const sample = earnings8k[earnings8k.length - 1];
  if (sample) {
    const subs = await fetchSubmissions(sample.cik);
    const f = subs.filings.find((x) => x.accessionNumber === sample.acc)!;
    const index = await fetchFilingIndex(f);
    const pick = findEarningsExhibit(index, f.primaryDocument);
    const text = await fetchDocumentText(f, pick.name);
    console.log(`\n${"─".repeat(92)}`);
    console.log(`Document retrieval — ${sample.ticker} ${f.form} ${f.filingDate}`);
    console.log("─".repeat(92));
    console.log(`  files in filing: ${index.length}   using: ${pick.name}` +
      (pick.isExhibit ? "  (exhibit)" : "  ⚠ FALLBACK to 8-K body — drop this event"));
    console.log(`  extracted ${text.length.toLocaleString()} chars`);
    console.log(`  first 240: ${JSON.stringify(text.slice(0, 240))}`);
  }

  // ── Usable window per model ───────────────────────────────────────
  console.log(`\n${"─".repeat(92)}`);
  console.log("Events usable per model (strictly after its knowledge cutoff)");
  console.log("─".repeat(92));
  for (const m of Object.values(MODELS)) {
    if (!m.cutoff) { console.log(`  ${pad(m.id, 34)} cutoff not established — cannot evaluate`); continue; }
    const usable = earnings8k.filter((e) => e.accepted.slice(0, 10) > m.cutoff!).length;
    console.log(`  ${pad(m.id, 34)} cutoff ${m.cutoff}   ${usable}/${earnings8k.length} events usable`);
  }

  console.log(`\n${"═".repeat(92)}`);
  console.log(`${cacheSize()} cached responses · ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  console.log("═".repeat(92) + "\n");
}

main().catch((e) => { console.error(String(e).slice(0, 600)); process.exit(1); });
