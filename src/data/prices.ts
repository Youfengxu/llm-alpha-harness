/**
 * Daily price bars and event-study returns.
 *
 * ── Three decisions that decide whether a study measures anything ─────
 *
 * 1. ADJUSTED CLOSE, always. A 2-for-1 split shows as a −50% return on raw
 *    closes. Any study using raw prices would find "events" on every split date
 *    and, worse, they would look like large negative surprises.
 *
 * 2. MARKET-ADJUSTED returns. Raw forward returns are dominated by the market:
 *    over a 60-session drift window the index move typically swamps anything
 *    firm-specific, so an unadjusted study largely measures whether the market
 *    went up during the sample. Subtracting the benchmark return (beta = 1) is
 *    the standard fix and is deliberately NOT upgraded to an estimated beta —
 *    beta fitted on a short pre-event window adds more estimation noise than the
 *    bias it removes.
 *
 * 3. SESSIONS, not calendar days. "Five days forward" must mean five trading
 *    sessions. Counting calendar days silently shortens every window that spans
 *    a weekend and produces a different horizon for Monday events than Thursday
 *    ones.
 *
 * ── The event decomposition ───────────────────────────────────────────
 * For an earnings release accepted after the close of session D−1:
 *
 *   ANNOUNCEMENT  close(D−1) → close(D)    the immediate reaction
 *   DRIFT         close(D)   → close(D+N)  what a signal must predict
 *
 * The split matters because post-earnings-announcement drift is the documented
 * anomaly, and it lives entirely in the second leg. A signal that only predicts
 * the announcement return is predicting something already impounded by the time
 * anyone could trade on the text.
 */

import * as fs from "fs";
import * as path from "path";
import * as crypto from "crypto";

const CACHE_DIR = process.env.PRICE_CACHE_DIR ?? path.resolve(process.cwd(), ".cache/prices");
const UA = "Mozilla/5.0 (compatible; llm-alpha-harness/1.0)";
const MIN_INTERVAL_MS = 250;
let lastRequest = 0;

export interface Bar {
  /** Exchange-local trading date, ISO yyyy-mm-dd. */
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  /** Split- and dividend-adjusted. Use THIS for returns. */
  adjClose: number;
  volume: number;
}

async function paced<T>(fn: () => Promise<T>): Promise<T> {
  const wait = Math.max(0, lastRequest + MIN_INTERVAL_MS - Date.now());
  if (wait > 0) await new Promise((r) => setTimeout(r, wait));
  lastRequest = Date.now();
  return fn();
}

/**
 * Daily bars for `ticker` covering [from, to] inclusive.
 *
 * Cached by (ticker, from, to) so reruns replay identical data. The window is
 * part of the key deliberately: a study must not silently gain bars because it
 * was re-run on a later date.
 */
export async function fetchDailyBars(ticker: string, from: string, to: string): Promise<Bar[]> {
  const key = crypto.createHash("sha256").update(`${ticker}|${from}|${to}`).digest("hex");
  const file = path.join(CACHE_DIR, `${key}.json`);
  if (fs.existsSync(file)) return JSON.parse(fs.readFileSync(file, "utf8")) as Bar[];

  const p1 = Math.floor(Date.parse(`${from}T00:00:00Z`) / 1000);
  const p2 = Math.floor(Date.parse(`${to}T23:59:59Z`) / 1000);
  const url =
    `https://query2.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(ticker)}` +
    `?interval=1d&period1=${p1}&period2=${p2}`;

  let lastErr: unknown;
  for (let attempt = 0; attempt < 4; attempt++) {
    try {
      const res = await paced(() =>
        fetch(url, { headers: { "User-Agent": UA }, signal: AbortSignal.timeout(45_000) })
      );
      if (!res.ok) throw new Error(`Yahoo ${res.status} for ${ticker}`);
      const body = (await res.json()) as {
        chart?: {
          error?: unknown;
          result?: Array<{
            timestamp?: number[];
            indicators?: {
              quote?: Array<Record<string, (number | null)[]>>;
              adjclose?: Array<{ adjclose?: (number | null)[] }>;
            };
          }>;
        };
      };
      if (body.chart?.error) throw new Error(`Yahoo error for ${ticker}: ${JSON.stringify(body.chart.error).slice(0, 150)}`);
      const r = body.chart?.result?.[0];
      const ts = r?.timestamp;
      const q = r?.indicators?.quote?.[0];
      if (!ts || !q) throw new Error(`Yahoo returned no bars for ${ticker} ${from}..${to}`);
      const adj = r?.indicators?.adjclose?.[0]?.adjclose;

      const bars: Bar[] = [];
      for (let i = 0; i < ts.length; i++) {
        const close = q.close?.[i];
        if (close == null) continue; // holidays and halts arrive as nulls
        bars.push({
          // US equity sessions run 13:30–21:00 UTC and never cross midnight UTC,
          // so the UTC date equals the exchange trading date. This would NOT
          // hold for an Asian exchange and must be revisited if one is added.
          date: new Date(ts[i] * 1000).toISOString().slice(0, 10),
          open: q.open?.[i] ?? close,
          high: q.high?.[i] ?? close,
          low: q.low?.[i] ?? close,
          close,
          adjClose: adj?.[i] ?? close,
          volume: q.volume?.[i] ?? 0,
        });
      }
      bars.sort((a, b) => a.date.localeCompare(b.date));
      fs.mkdirSync(CACHE_DIR, { recursive: true });
      fs.writeFileSync(file, JSON.stringify(bars));
      return bars;
    } catch (e) {
      lastErr = e;
      await new Promise((r) => setTimeout(r, 1000 * (attempt + 1)));
    }
  }
  throw lastErr;
}

// ─── Session arithmetic ───────────────────────────────────────────────

/**
 * Index of the first session at or after `date`.
 *
 * Returns -1 when the date is past the end of the data — which callers must
 * treat as "not measurable yet" rather than clamping to the last bar. Clamping
 * would silently shorten the horizon and, for recent events, compare a 5-session
 * window against a 1-session one.
 */
export function sessionIndexAtOrAfter(bars: Bar[], date: string): number {
  for (let i = 0; i < bars.length; i++) if (bars[i].date >= date) return i;
  return -1;
}

/** Simple return between two indices, on adjusted closes. */
function retBetween(bars: Bar[], i0: number, i1: number): number | null {
  if (i0 < 0 || i1 < 0 || i0 >= bars.length || i1 >= bars.length) return null;
  const a = bars[i0].adjClose, b = bars[i1].adjClose;
  if (!(a > 0) || !(b > 0)) return null;
  return b / a - 1;
}

export interface EventReturns {
  /** The first session that could act on the event. */
  session: string;
  /** close(D−1) → close(D). The immediate reaction. */
  announcement: number | null;
  /** close(D) → close(D+N). What a text signal must predict. */
  drift: number | null;
  /** Drift minus the benchmark's drift over the identical sessions. */
  driftMarketAdj: number | null;
  /** Sessions actually available; less than requested means truncated. */
  driftSessions: number;
}

/**
 * Announcement and drift returns around an event.
 *
 * `session` should come from `nextSessionAfter` on the filing's acceptance time,
 * so the window starts where trading was actually possible.
 *
 * Every leg returns null rather than a number when the data does not support it.
 * A null is a dropped observation; a fabricated 0 is a fabricated finding.
 */
export function eventReturns(
  bars: Bar[],
  session: string,
  driftSessions: number,
  benchmark?: Bar[]
): EventReturns {
  const d = sessionIndexAtOrAfter(bars, session);
  const out: EventReturns = {
    session, announcement: null, drift: null, driftMarketAdj: null, driftSessions: 0,
  };
  if (d < 0) return out;

  out.announcement = d > 0 ? retBetween(bars, d - 1, d) : null;

  const end = Math.min(d + driftSessions, bars.length - 1);
  out.driftSessions = end - d;
  // A truncated window is not a short window: comparing a 12-session drift
  // against a 60-session one across events would make recency look like signal.
  if (out.driftSessions < driftSessions) return out;

  out.drift = retBetween(bars, d, end);

  if (benchmark && out.drift !== null) {
    // Align on DATES, not indices: a halted stock has fewer bars than the index,
    // and index-aligning would silently compare different calendar windows.
    const b0 = sessionIndexAtOrAfter(benchmark, bars[d].date);
    const b1 = sessionIndexAtOrAfter(benchmark, bars[end].date);
    const benchRet = retBetween(benchmark, b0, b1);
    if (benchRet !== null) out.driftMarketAdj = out.drift - benchRet;
  }
  return out;
}

/** Missing-session report, so thin coverage is visible rather than assumed. */
export function coverage(bars: Bar[]): { first: string; last: string; sessions: number; gapsOver3d: number } {
  if (!bars.length) return { first: "-", last: "-", sessions: 0, gapsOver3d: 0 };
  let gaps = 0;
  for (let i = 1; i < bars.length; i++) {
    const days = (Date.parse(bars[i].date) - Date.parse(bars[i - 1].date)) / 86400_000;
    if (days > 4) gaps++; // >4 calendar days is more than a weekend plus a holiday
  }
  return { first: bars[0].date, last: bars[bars.length - 1].date, sessions: bars.length, gapsOver3d: gaps };
}

export function cacheSize(): number {
  if (!fs.existsSync(CACHE_DIR)) return 0;
  return fs.readdirSync(CACHE_DIR).filter((f) => f.endsWith(".json")).length;
}
