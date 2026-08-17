/**
 * SEC EDGAR connector — point-in-time filings, free and complete.
 *
 * ── Why EDGAR is the right first data source ──────────────────────────
 * It is point-in-time BY CONSTRUCTION. Every filing carries the moment the SEC
 * accepted it, so there is no question of what was knowable when — the usual
 * failure of vendor fundamentals, which silently restate history. It is also
 * free, which matters because a strategy needing data you will not buy is not a
 * strategy.
 *
 * ── acceptanceDateTime, not filingDate ────────────────────────────────
 * `filingDate` is a calendar day; `acceptanceDateTime` is the timestamp the
 * document actually became public. A filing accepted after the close cannot
 * affect that day's return, and treating it as if it could is lookahead bias
 * dressed as a date join.
 *
 * A caveat this module does NOT paper over: SEC publishes these stamps with a
 * `Z` suffix, but the agency documents its timestamps as Eastern. That ambiguity
 * is worth roughly a trading session, so `nextSessionAfter` deliberately does
 * NOT attempt intraday precision — it advances to the next trading day whenever
 * a filing lands anywhere near a boundary. Giving up a few hours of edge is the
 * cheap side of this trade; the expensive side is a backtest that quietly trades
 * on information it did not have.
 *
 * ── Completeness ──────────────────────────────────────────────────────
 * A submissions response holds only the most recent ~1000 filings; everything
 * older sits in separate pages listed under `filings.files`. Reading only
 * `filings.recent` silently truncates history — the same class of bug that has
 * already appeared three times in this project's predecessor, once dropping an
 * entire asset from a study. `fetchSubmissions` follows every page and reports
 * how many it read.
 */

import * as fs from "fs";
import * as path from "path";
import * as crypto from "crypto";

const CACHE_DIR = process.env.EDGAR_CACHE_DIR ?? path.resolve(process.cwd(), ".cache/edgar");

/**
 * SEC requires a User-Agent identifying the requester, and blocks traffic that
 * omits it. This is deliberately NOT defaulted to an invented value: sending a
 * fabricated contact address to a regulator's system is not a reasonable thing
 * to do silently, and a fake one may get the whole IP blocked.
 */
function userAgent(): string {
  const ua = process.env.SEC_USER_AGENT;
  if (!ua || !ua.includes("@")) {
    throw new Error(
      "SEC_USER_AGENT must be set to a real contact, e.g.\n" +
      '  export SEC_USER_AGENT="your-project your.email@example.com"\n' +
      "The SEC requires requests to identify the requester and rate-limits or " +
      "blocks those that do not. This is not defaulted because inventing a " +
      "contact address for a regulator's system is not ours to do."
    );
  }
  return ua;
}

/** SEC permits 10 requests/second. 8 leaves headroom for clock skew. */
const MIN_INTERVAL_MS = 125;
let lastRequest = 0;

async function paced<T>(fn: () => Promise<T>): Promise<T> {
  const wait = Math.max(0, lastRequest + MIN_INTERVAL_MS - Date.now());
  if (wait > 0) await new Promise((r) => setTimeout(r, wait));
  lastRequest = Date.now();
  return fn();
}

/**
 * Cached GET. Responses are keyed by URL, so a rerun of any study replays
 * byte-identical bytes and never re-hits the SEC.
 *
 * `binary: false` only — every EDGAR endpoint used here is text.
 */
async function get(url: string, opts: { retries?: number } = {}): Promise<string> {
  const key = crypto.createHash("sha256").update(url).digest("hex");
  const file = path.join(CACHE_DIR, `${key}.txt`);
  if (fs.existsSync(file)) return fs.readFileSync(file, "utf8");

  const retries = opts.retries ?? 3;
  let lastErr: unknown;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const res = await paced(() =>
        fetch(url, {
          headers: { "User-Agent": userAgent(), "Accept-Encoding": "gzip, deflate" },
          signal: AbortSignal.timeout(60_000),
        })
      );
      // 429 and 5xx are transient; 404 is not, and retrying it just wastes the
      // rate limit we are trying to respect.
      if (res.status === 404) throw new Error(`EDGAR 404: ${url}`);
      if (!res.ok) throw new Error(`EDGAR ${res.status} for ${url}`);
      const text = await res.text();
      fs.mkdirSync(CACHE_DIR, { recursive: true });
      fs.writeFileSync(file, text);
      return text;
    } catch (e) {
      lastErr = e;
      if (String(e).includes("404")) throw e;
      await new Promise((r) => setTimeout(r, 800 * (attempt + 1)));
    }
  }
  throw lastErr;
}

// ─── Ticker → CIK ─────────────────────────────────────────────────────

export interface Company {
  cik: number;
  ticker: string;
  title: string;
}

/** All SEC-registered tickers. ~10k entries, one request, cached. */
export async function fetchCompanyTickers(): Promise<Company[]> {
  const raw = await get("https://www.sec.gov/files/company_tickers.json");
  const obj = JSON.parse(raw) as Record<string, { cik_str: number; ticker: string; title: string }>;
  return Object.values(obj).map((c) => ({ cik: c.cik_str, ticker: c.ticker, title: c.title }));
}

/** Zero-padded CIK as EDGAR paths require. */
export const padCik = (cik: number): string => String(cik).padStart(10, "0");

// ─── Filings ──────────────────────────────────────────────────────────

export interface Filing {
  cik: number;
  accessionNumber: string;
  form: string;
  /** Calendar day filed. Use `acceptanceDateTime` for anything time-sensitive. */
  filingDate: string;
  /** Period the filing covers, when applicable. */
  reportDate: string;
  /** The moment it became public. THIS is the point-in-time anchor. */
  acceptanceDateTime: string;
  /** 8-K item codes, e.g. "2.02,9.01". Empty for most other forms. */
  items: string;
  primaryDocument: string;
  isXBRL: boolean;
  size: number;
}

interface SubmissionsPage {
  cik?: string;
  name?: string;
  filings?: {
    recent?: Record<string, unknown[]>;
    files?: Array<{ name: string }>;
  };
}

function rowsToFilings(cik: number, r: Record<string, unknown[]>): Filing[] {
  const n = (r.accessionNumber as string[] | undefined)?.length ?? 0;
  const out: Filing[] = [];
  for (let i = 0; i < n; i++) {
    out.push({
      cik,
      accessionNumber: String((r.accessionNumber as string[])[i]),
      form: String((r.form as string[])[i] ?? ""),
      filingDate: String((r.filingDate as string[])[i] ?? ""),
      reportDate: String((r.reportDate as string[])[i] ?? ""),
      acceptanceDateTime: String((r.acceptanceDateTime as string[])[i] ?? ""),
      items: String((r.items as string[])?.[i] ?? ""),
      primaryDocument: String((r.primaryDocument as string[])?.[i] ?? ""),
      isXBRL: Boolean((r.isXBRL as unknown[])?.[i]),
      size: Number((r.size as number[])?.[i] ?? 0),
    });
  }
  return out;
}

export interface SubmissionsResult {
  cik: number;
  name: string;
  filings: Filing[];
  /** How many pages were read. >1 means older history was followed. */
  pagesRead: number;
}

/**
 * Every filing for a company, following the older pages.
 *
 * `filings.recent` caps at ~1000 entries and the rest live in `filings.files`.
 * Reading only `recent` looks complete and silently truncates anything older,
 * which for an active filer is barely two years.
 */
export async function fetchSubmissions(cik: number): Promise<SubmissionsResult> {
  const first = JSON.parse(
    await get(`https://data.sec.gov/submissions/CIK${padCik(cik)}.json`)
  ) as SubmissionsPage;

  const filings: Filing[] = first.filings?.recent ? rowsToFilings(cik, first.filings.recent) : [];
  let pagesRead = 1;

  for (const f of first.filings?.files ?? []) {
    const page = JSON.parse(
      await get(`https://data.sec.gov/submissions/${f.name}`)
    ) as Record<string, unknown[]>;
    filings.push(...rowsToFilings(cik, page));
    pagesRead++;
  }

  filings.sort((a, b) => b.acceptanceDateTime.localeCompare(a.acceptanceDateTime));
  return { cik, name: first.name ?? "", filings, pagesRead };
}

export interface FilingQuery {
  /** e.g. ["8-K"], ["10-Q","10-K"]. Exact match on the form field. */
  forms?: string[];
  /** 8-K item code that must be present, e.g. "2.02" for results of operations. */
  item?: string;
  /** Inclusive ISO bounds on acceptanceDateTime. */
  from?: string;
  to?: string;
}

export function filterFilings(filings: Filing[], q: FilingQuery): Filing[] {
  return filings.filter((f) => {
    if (q.forms && !q.forms.includes(f.form)) return false;
    // Item codes are a comma-joined list; a substring test would match 2.02
    // inside 12.02, so split and compare exactly.
    if (q.item && !f.items.split(",").map((s) => s.trim()).includes(q.item)) return false;
    if (q.from && f.acceptanceDateTime < q.from) return false;
    if (q.to && f.acceptanceDateTime > q.to) return false;
    return true;
  });
}

// ─── Documents ────────────────────────────────────────────────────────

const accessionPath = (a: string) => a.replace(/-/g, "");

/** URL of a filing's primary document. */
export function documentUrl(f: Filing): string {
  return `https://www.sec.gov/Archives/edgar/data/${f.cik}/${accessionPath(f.accessionNumber)}/${f.primaryDocument}`;
}

/** All files in a filing, so exhibits (e.g. the earnings release) are reachable. */
export async function fetchFilingIndex(f: Filing): Promise<Array<{ name: string; type: string; size: number }>> {
  const url = `https://www.sec.gov/Archives/edgar/data/${f.cik}/${accessionPath(f.accessionNumber)}/index.json`;
  const j = JSON.parse(await get(url)) as {
    directory?: { item?: Array<{ name: string; type: string; size: string }> };
  };
  return (j.directory?.item ?? []).map((i) => ({
    name: i.name,
    type: i.type,
    size: Number(i.size ?? 0),
  }));
}

/**
 * Strips HTML/XBRL to readable text.
 *
 * Deliberately simple. Filings are heavily tagged and a full parse would pull in
 * a dependency for little gain when the consumer is an LLM that tolerates
 * imperfect whitespace. Script and style contents are dropped because inline JS
 * would otherwise reach the model as prose.
 */
export function htmlToText(html: string): string {
  return html
    .replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, " ")
    .replace(/<style\b[^>]*>[\s\S]*?<\/style>/gi, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&#(\d+);/g, (_, d) => String.fromCharCode(Number(d)))
    .replace(/[ \t ]+/g, " ")
    .replace(/\n\s*\n\s*\n+/g, "\n\n")
    .trim();
}

/**
 * Matches every exhibit-99 spelling filers actually use: `ex-99.1.htm`,
 * `ex991.htm`, `exhibit99_1.htm`, `EX-99.1.HTM`, and mid-name forms such as
 * `a8-kex991q3202606272026.htm`. Unanchored on purpose — anchoring to the start
 * silently fell back to the 8-K body on Apple's real filing.
 */
const EX99 = /ex(?:hibit)?[-_ .]?99/i;
/** The same, narrowed to 99.1 specifically, so 99.2 does not win by size. */
const EX99_1 = /ex(?:hibit)?[-_ .]?99[.\-_]?1(?!\d)/i;

export interface ExhibitPick {
  name: string;
  /** False when this is the 8-K body rather than the press release. */
  isExhibit: boolean;
}

/**
 * Locates the earnings press release within an 8-K.
 *
 * The 8-K body is typically three sentences of cross-reference wrapped in XBRL
 * cover-page tagging; the substance is in Exhibit 99.1. Scoring the body instead
 * feeds the model text like "aapl-20260730 false 0000320193 us-gaap:
 * CommonStockMember" — which is not merely useless but actively poisonous to a
 * study, since it looks like a successful fetch.
 *
 * Naming is genuinely inconsistent across filers — `ex-99.1.htm`, `ex991.htm`,
 * `a8-kex991q3202606272026.htm` — so "ex99" is matched ANYWHERE in the name, not
 * anchored. An earlier anchored regex silently fell back to the body.
 *
 * Returns `isExhibit: false` when only the body was found, so callers can DROP
 * the event rather than score noise. Returning a bare string is how that
 * distinction gets lost.
 */
export function findEarningsExhibit(
  index: Array<{ name: string; size: number }>,
  primaryDocument: string
): ExhibitPick {
  // Rendering and XBRL artifacts are .htm but never prose.
  const noise = /^(R\d+\.htm|FilingSummary|MetaLinks|report\.css|Show\.js|.*-index)/i;
  const candidates = index
    .filter((i) => /\.html?$/i.test(i.name) && !noise.test(i.name))
    .filter((i) => EX99.test(i.name));

  if (candidates.length) {
    // Prefer 99.1 where numbered; otherwise the largest, since the release is
    // substantive while other exhibits tend to be short.
    const preferred =
      candidates.find((c) => EX99_1.test(c.name)) ??
      candidates.sort((a, b) => b.size - a.size)[0];
    return { name: preferred.name, isExhibit: true };
  }
  return { name: primaryDocument, isExhibit: false };
}

export async function fetchDocumentText(f: Filing, fileName?: string): Promise<string> {
  const url = fileName
    ? `https://www.sec.gov/Archives/edgar/data/${f.cik}/${accessionPath(f.accessionNumber)}/${fileName}`
    : documentUrl(f);
  return htmlToText(await get(url));
}

// ─── Point-in-time discipline ─────────────────────────────────────────

/** US market holidays are not modelled; weekends are. See `nextSessionAfter`. */
const CLOSE_HOUR_UTC = 20; // 16:00 ET during DST

/**
 * The first trading session that could act on a filing.
 *
 * Conservative on purpose. SEC stamps carry `Z` but the agency documents its
 * times as Eastern, an ambiguity worth about a session — so anything accepted
 * from mid-afternoon onward rolls to the next day rather than being credited to
 * the same close. Weekends roll forward; holidays are NOT modelled, which is a
 * known gap that costs a handful of events a year and never creates lookahead
 * (a holiday simply means the "next session" is one day later than returned).
 *
 * Returns an ISO date. Join returns on this, never on `filingDate`.
 */
export function nextSessionAfter(acceptanceDateTime: string): string {
  const d = new Date(acceptanceDateTime);
  if (Number.isNaN(d.getTime())) throw new Error(`Unparseable acceptance time: ${acceptanceDateTime}`);
  const out = new Date(d);
  if (d.getUTCHours() >= CLOSE_HOUR_UTC - 2) out.setUTCDate(out.getUTCDate() + 1);
  // 0 = Sunday, 6 = Saturday
  while (out.getUTCDay() === 0 || out.getUTCDay() === 6) out.setUTCDate(out.getUTCDate() + 1);
  return out.toISOString().slice(0, 10);
}

/** Number of cached EDGAR responses — a cheap check that a fetch ran. */
export function cacheSize(): number {
  if (!fs.existsSync(CACHE_DIR)) return 0;
  return fs.readdirSync(CACHE_DIR).filter((f) => f.endsWith(".txt")).length;
}
