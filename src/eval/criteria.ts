/**
 * Pre-registration: criteria are declared BEFORE a run and evaluated after.
 *
 * ── Why this is code rather than a convention ─────────────────────────
 * The predecessor project produced thirteen negative results that were
 * trustworthy precisely because their thresholds were written down first. It
 * also produced the counter-example: one criterion was mis-specified (a
 * memoryless model was asked to show persistence), and the honest move was to
 * record the error rather than quietly relax the bar. Both only worked because
 * the bar existed in advance.
 *
 * The failure mode this prevents is not dishonesty, it is drift — looking at a
 * result, noticing 7% is "nearly" 8%, and adjusting. Encoding the criteria and
 * hashing them makes that adjustment visible in a diff.
 */

import * as crypto from "crypto";

export interface Criterion {
  name: string;
  /** What must hold, in plain language, for a reader who was not there. */
  statement: string;
  /** Numeric bar. */
  threshold: number;
  direction: "gte" | "lte";
  /** Why THIS number. A threshold without a justification is a guess. */
  rationale: string;
}

export interface PreRegistration {
  study: string;
  /** ISO date the criteria were fixed — before any result was seen. */
  registeredAt: string;
  hypothesis: string;
  /** Honest prior that the study passes, in [0,1]. Forces a real expectation. */
  prior: number;
  criteria: Criterion[];
  /** What result would make you abandon the line entirely. */
  abandonIf: string;
}

export interface CriterionResult extends Criterion {
  observed: number;
  passed: boolean;
}

export interface StudyResult {
  study: string;
  /** Hash of the registration — changing any criterion changes this. */
  registrationHash: string;
  results: CriterionResult[];
  /** ALL criteria must pass. A partial pass is a fail. */
  passed: boolean;
  passedCount: number;
  totalCount: number;
}

/**
 * Stable fingerprint of a registration.
 *
 * Printed alongside every result so a reader can tell whether the bar moved
 * between runs. Deliberately excludes `registeredAt` so re-reading the same
 * criteria on a later date does not change the hash — the fingerprint is of the
 * BAR, not of when it was read.
 */
export function registrationHash(reg: PreRegistration): string {
  const material = JSON.stringify({
    study: reg.study,
    hypothesis: reg.hypothesis,
    criteria: reg.criteria.map((c) => ({
      name: c.name, threshold: c.threshold, direction: c.direction,
    })),
  });
  return crypto.createHash("sha256").update(material).digest("hex").slice(0, 12);
}

/**
 * Scores observations against a registration.
 *
 * A criterion with no observation FAILS rather than being skipped. An
 * unmeasured criterion is not a passed one, and silently dropping it would let
 * a study pass by omitting its hardest test.
 */
export function evaluate(reg: PreRegistration, observed: Record<string, number>): StudyResult {
  const results: CriterionResult[] = reg.criteria.map((c) => {
    const value = observed[c.name];
    if (value === undefined || !Number.isFinite(value)) {
      return { ...c, observed: NaN, passed: false };
    }
    return {
      ...c,
      observed: value,
      passed: c.direction === "gte" ? value >= c.threshold : value <= c.threshold,
    };
  });
  const passedCount = results.filter((r) => r.passed).length;
  return {
    study: reg.study,
    registrationHash: registrationHash(reg),
    results,
    passed: passedCount === results.length,
    passedCount,
    totalCount: results.length,
  };
}

/** Human-readable report. Prints the bar next to the observation, always. */
export function formatResult(reg: PreRegistration, res: StudyResult): string {
  const lines: string[] = [];
  lines.push("═".repeat(78));
  lines.push(`${res.study}   [registration ${res.registrationHash}]`);
  lines.push(`hypothesis: ${reg.hypothesis}`);
  lines.push(`prior: ${(reg.prior * 100).toFixed(0)}% · registered ${reg.registeredAt}`);
  lines.push("═".repeat(78));
  for (const r of res.results) {
    const arrow = r.direction === "gte" ? "≥" : "≤";
    const obs = Number.isFinite(r.observed) ? r.observed.toFixed(4) : "NOT MEASURED";
    lines.push(`${r.passed ? "PASS" : "FAIL"}  ${r.name.padEnd(28)} ${obs.padStart(14)}  (need ${arrow} ${r.threshold})`);
    lines.push(`      ${r.statement}`);
  }
  lines.push("─".repeat(78));
  lines.push(
    res.passed
      ? `ALL ${res.totalCount} CRITERIA PASSED`
      : `FAILED — ${res.passedCount}/${res.totalCount} criteria met (all are required)`
  );
  if (!res.passed) lines.push(`abandon condition: ${reg.abandonIf}`);
  lines.push("═".repeat(78));
  return lines.join("\n");
}
