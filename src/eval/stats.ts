/**
 * Statistics for the harness. Small, dependency-free, and deliberately
 * conservative about what it claims.
 *
 * The predecessor project's failures were almost never modelling failures —
 * they were counting failures. The three that mattered:
 *
 *   - Reporting an in-sample fit as held-out (93% against a true -1.6%).
 *   - Treating 12 correlated assets as 12 observations when correlation of
 *     0.449 made the effective sample ~2.2.
 *   - Ranking parameters by historical performance when historical and future
 *     rank were ANTI-correlated (Spearman rho = -0.339).
 *
 * `effectiveSampleSize` exists because of the second, and is the function most
 * likely to stop a bad decision here.
 */

// ─── Basics ───────────────────────────────────────────────────────────

export const mean = (xs: number[]): number =>
  xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;

export function variance(xs: number[], sample = true): number {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  const ss = xs.reduce((s, x) => s + (x - m) ** 2, 0);
  return ss / (sample ? xs.length - 1 : xs.length);
}

export const stdDev = (xs: number[], sample = true): number => Math.sqrt(variance(xs, sample));

export function pearson(a: number[], b: number[]): number {
  const n = Math.min(a.length, b.length);
  if (n < 2) return 0;
  const ma = mean(a.slice(0, n)), mb = mean(b.slice(0, n));
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) {
    const u = a[i] - ma, v = b[i] - mb;
    num += u * v; da += u * u; db += v * v;
  }
  return da > 0 && db > 0 ? num / Math.sqrt(da * db) : 0;
}

/** Rank correlation. Ties get average ranks, so repeated scores do not distort it. */
export function spearman(a: number[], b: number[]): number {
  const rank = (v: number[]): number[] => {
    const idx = v.map((x, i) => [x, i] as const).sort((p, q) => p[0] - q[0]);
    const r = new Array<number>(v.length);
    let i = 0;
    while (i < idx.length) {
      let j = i;
      while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
      const avg = (i + j) / 2;
      for (let k = i; k <= j; k++) r[idx[k][1]] = avg;
      i = j + 1;
    }
    return r;
  };
  return pearson(rank(a), rank(b));
}

// ─── Effective sample size ────────────────────────────────────────────

/**
 * Independent-observation equivalent of `n` series correlated at `rho`:
 *
 *     n_eff = n / (1 + (n - 1) * rho)
 *
 * This is the single most important number when deciding whether a
 * cross-sectional result means anything. In the crypto predecessor, 12 majors
 * correlating at 0.449 gave n_eff = 2.2 — and crucially it ASYMPTOTES at 1/rho,
 * so adding assets could never have fixed it. Twelve assets agreeing was closer
 * to two agreeing, which is why every "improvement" there was unfalsifiable.
 *
 * The main reason to prefer equities is that a genuine cross-section pushes this
 * number up. Check it before believing any breadth-based claim.
 */
export function effectiveSampleSize(n: number, rho: number): number {
  if (n <= 1) return n;
  const r = Math.max(0, Math.min(0.999, rho));
  return n / (1 + (n - 1) * r);
}

/** Ceiling on n_eff however many correlated series are added: 1/rho. */
export const effectiveSampleCeiling = (rho: number): number =>
  rho <= 0 ? Infinity : 1 / Math.max(1e-9, rho);

/** Mean pairwise Pearson correlation across a set of aligned series. */
export function meanPairwiseCorrelation(series: number[][]): number {
  const pairs: number[] = [];
  for (let i = 0; i < series.length; i++) {
    for (let j = i + 1; j < series.length; j++) pairs.push(pearson(series[i], series[j]));
  }
  return pairs.length ? mean(pairs) : 0;
}

// ─── Inference ────────────────────────────────────────────────────────

/**
 * Two-sided p-value for a t statistic, via an incomplete-beta continued
 * fraction. Accurate enough for screening; this harness uses p-values to reject
 * ideas rather than to publish, so a small tail approximation error is
 * acceptable where a wrong sign or magnitude would not be.
 */
export function tDistPValue(t: number, df: number): number {
  if (!Number.isFinite(t) || df <= 0) return 1;
  const x = df / (df + t * t);
  return Math.max(0, Math.min(1, incompleteBeta(x, df / 2, 0.5)));
}

function incompleteBeta(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const lbeta = lgamma(a) + lgamma(b) - lgamma(a + b);
  const front = Math.exp(Math.log(x) * a + Math.log(1 - x) * b - lbeta) / a;
  let f = 1, c = 1, d = 0;
  for (let i = 0; i <= 300; i++) {
    const m = Math.floor(i / 2);
    let numerator: number;
    if (i === 0) numerator = 1;
    else if (i % 2 === 0) numerator = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m));
    else numerator = -(((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1)));
    d = 1 + numerator * d;
    if (Math.abs(d) < 1e-30) d = 1e-30;
    d = 1 / d;
    c = 1 + numerator / c;
    if (Math.abs(c) < 1e-30) c = 1e-30;
    f *= c * d;
    if (Math.abs(1 - c * d) < 1e-10) break;
  }
  const res = front * (f - 1);
  return a + 1 > (a + b + 2) * x ? res : 1 - incompleteBetaComplement(1 - x, b, a, lbeta);
}

function incompleteBetaComplement(x: number, a: number, b: number, lbeta: number): number {
  const front = Math.exp(Math.log(x) * a + Math.log(1 - x) * b - lbeta) / a;
  let f = 1, c = 1, d = 0;
  for (let i = 0; i <= 300; i++) {
    const m = Math.floor(i / 2);
    let numerator: number;
    if (i === 0) numerator = 1;
    else if (i % 2 === 0) numerator = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m));
    else numerator = -(((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1)));
    d = 1 + numerator * d;
    if (Math.abs(d) < 1e-30) d = 1e-30;
    d = 1 / d;
    c = 1 + numerator / c;
    if (Math.abs(c) < 1e-30) c = 1e-30;
    f *= c * d;
    if (Math.abs(1 - c * d) < 1e-10) break;
  }
  return front * (f - 1);
}

export interface TTest { t: number; df: number; pValue: number; mean: number; stdErr: number }

/** One-sample t-test of mean(xs) against 0. */
export function tTest(xs: number[]): TTest {
  const n = xs.length;
  if (n < 2) return { t: 0, df: 0, pValue: 1, mean: mean(xs), stdErr: 0 };
  const m = mean(xs);
  const se = stdDev(xs) / Math.sqrt(n);
  const t = se > 0 ? m / se : 0;
  return { t, df: n - 1, pValue: tDistPValue(t, n - 1), mean: m, stdErr: se };
}

/**
 * t-test with the sample size DISCOUNTED for cross-sectional correlation.
 *
 * Use this, not `tTest`, whenever observations come from correlated series.
 * Treating correlated observations as independent is how a project convinces
 * itself that noise is significant.
 */
export function tTestEffective(xs: number[], rho: number): TTest & { nEff: number } {
  const base = tTest(xs);
  const nEff = effectiveSampleSize(xs.length, rho);
  if (nEff < 2) return { ...base, t: 0, df: 0, pValue: 1, nEff };
  // Rescale: t grows with sqrt(n), so shrink it by sqrt(n_eff / n).
  const t = base.t * Math.sqrt(nEff / xs.length);
  return { ...base, t, df: nEff - 1, pValue: tDistPValue(t, nEff - 1), nEff };
}

// ─── Regression with interaction (for the LAP test) ───────────────────

export interface InteractionFit {
  intercept: number;
  signal: number;
  lap: number;
  interaction: number;
  interactionT: number;
  interactionPValue: number;
  n: number;
}

/**
 * OLS of `y` on [1, signal, lap, signal*lap].
 *
 * The interaction coefficient is the lookahead-bias diagnostic: it asks whether
 * the signal predicts BETTER where the model remembers more. Solved by Gaussian
 * elimination on the 4x4 normal equations — small and well conditioned enough
 * that a full decomposition would be overkill.
 */
export function olsWithInteraction(signal: number[], lap: number[], y: number[]): InteractionFit {
  const n = Math.min(signal.length, lap.length, y.length);
  const empty: InteractionFit = {
    intercept: 0, signal: 0, lap: 0, interaction: 0,
    interactionT: 0, interactionPValue: 1, n,
  };
  if (n < 5) return empty;

  const X: number[][] = [];
  for (let i = 0; i < n; i++) X.push([1, signal[i], lap[i], signal[i] * lap[i]]);

  const k = 4;
  const xtx = Array.from({ length: k }, () => new Array(k).fill(0));
  const xty = new Array(k).fill(0);
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < k; a++) {
      xty[a] += X[i][a] * y[i];
      for (let b = 0; b < k; b++) xtx[a][b] += X[i][a] * X[i][b];
    }
  }

  const inv = invert(xtx);
  if (!inv) return empty;

  const beta = new Array(k).fill(0);
  for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) beta[a] += inv[a][b] * xty[b];

  let sse = 0;
  for (let i = 0; i < n; i++) {
    let yhat = 0;
    for (let a = 0; a < k; a++) yhat += beta[a] * X[i][a];
    sse += (y[i] - yhat) ** 2;
  }
  const df = n - k;
  if (df <= 0) return { ...empty, intercept: beta[0], signal: beta[1], lap: beta[2], interaction: beta[3] };

  const sigma2 = sse / df;
  const seInteraction = Math.sqrt(Math.max(0, sigma2 * inv[3][3]));
  const t = seInteraction > 0 ? beta[3] / seInteraction : 0;

  return {
    intercept: beta[0], signal: beta[1], lap: beta[2], interaction: beta[3],
    interactionT: t, interactionPValue: tDistPValue(t, df), n,
  };
}

function invert(m: number[][]): number[][] | null {
  const k = m.length;
  const a = m.map((row, i) => [...row, ...Array.from({ length: k }, (_, j) => (i === j ? 1 : 0))]);
  for (let col = 0; col < k; col++) {
    let pivot = col;
    for (let r = col + 1; r < k; r++) if (Math.abs(a[r][col]) > Math.abs(a[pivot][col])) pivot = r;
    if (Math.abs(a[pivot][col]) < 1e-12) return null; // singular: collinear inputs
    [a[col], a[pivot]] = [a[pivot], a[col]];
    const p = a[col][col];
    for (let j = 0; j < 2 * k; j++) a[col][j] /= p;
    for (let r = 0; r < k; r++) {
      if (r === col) continue;
      const f = a[r][col];
      if (f === 0) continue;
      for (let j = 0; j < 2 * k; j++) a[r][j] -= f * a[col][j];
    }
  }
  return a.map((row) => row.slice(k));
}

function lgamma(x: number): number {
  const c = [
    76.18009172947146, -86.50532032941677, 24.01409824083091,
    -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5,
  ];
  let y = x, tmp = x + 5.5;
  tmp -= (x + 0.5) * Math.log(tmp);
  let ser = 1.000000000190015;
  for (let j = 0; j < 6; j++) ser += c[j] / ++y;
  return -tmp + Math.log((2.5066282746310005 * ser) / x);
}
