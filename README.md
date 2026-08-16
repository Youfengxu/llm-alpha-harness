# llm-alpha-harness

**Phase 0 of the LLM trading research programme: the evaluation harness.**

This repository does not contain a trading strategy. It contains the apparatus
for deciding whether one is real — built first, and deliberately, because the
predecessor project established that the harness is where results are won and
lost, not the model.

## Why this exists before any strategy

A BTC trading system was taken through thirteen method families — eight threshold
rules, HMMs at 2/3/4 states, BOCPD at four hazard rates, GMM clustering, and exit
policy optimisation. Every one failed. The instructive part is *how* they failed:

| what went wrong | how it was caught |
|---|---|
| Optimiser reported 93% over six weeks against a true −1.6% | the split was not actually held out |
| 12 assets treated as 12 observations | correlation 0.449 → effective n ≈ **2.02** |
| Best historical parameters chosen | train and test rank were **anti**-correlated (ρ = −0.339) |
| "Improves in both regime windows" used as the bar | both windows were in-sample; a real held-out test killed the result |
| Three scripts gave different answers on consecutive runs | non-deterministic windows and silently truncated fetches |

None of those were modelling errors. Every one was a counting or plumbing error,
and each would have been just as fatal — and considerably more expensive — in
equities.

## The controls, and what enforces each

| control | module | enforcement |
|---|---|---|
| Score only after the model's knowledge cutoff | `src/llm/models.ts` | `assertScorable` **throws**; an unknown cutoff refuses to run at all |
| Detect lookahead contamination | `src/eval/lap.ts` | Lookahead Propensity + interaction test |
| Deterministic reruns | `src/llm/client.ts` | every response hashed and cached; temperature 0 |
| Beat a dumb baseline | `src/eval/baseline.ts` | finance lexicon, numeric surprise, incremental R² |
| Discount correlated observations | `src/eval/stats.ts` | `effectiveSampleSize`, `tTestEffective` |
| Fix the bar before the result | `src/eval/criteria.ts` | pre-registration with a hash that changes if a threshold moves |

### Lookahead bias is the headline risk

An LLM asked to forecast an event inside its training data may simply **recall
the outcome**. Following Gao, Jiang & Yan ([arXiv:2512.23847](https://arxiv.org/abs/2512.23847)),
this harness measures *Lookahead Propensity* — a date-only recall probe — and
tests whether a signal's accuracy **co-varies** with it. Skill that grows with
recall is memory, not forecasting.

Measured on this hardware, first attempt, with `qwen3.6-35b-a3b`:

```
LAP  Apple Inc, 2023-01-15 : 1.0     full recall of realised outcomes
LAP  Apple Inc, 2026-06-15 : 0.0     nothing
```

That collapse is the control working. It is also why **cutoffs in
`src/llm/models.ts` are all `null` until probed** — a plausible date recalled
from a model card is exactly the kind of thing that looks like rigour and isn't.

## Setup

```bash
pnpm install
pnpm verify        # typecheck + 38 tests
```

Inference is local and OpenAI-compatible. Override endpoints with
`MAC_LLAMASWAP_URL` / `GX10_VLLM_URL`.

### The serving layer is shared — eviction is the default

llama-swap holds one primary model per host and swaps on demand, so calling any
non-resident model **evicts whatever is loaded**. Two are pinned because other
services need them warm:

| host | pinned model | free to call |
|---|---|---|
| Mac Studio | `muse-glimmer-30b` | yes — already resident |
| GX10 | `gx10/Qwen3-Coder-Next-UD-Q4_K_M` | yes — `ttl: -1` |

Calling anything else on those hosts costs ~15s each way **and** takes the
pinned model away from whatever depends on it. A batch run over hundreds of
names would thrash continuously.

`assertSafeToCall` therefore **throws** rather than warns — eviction is invisible
at the call site, since the request still succeeds, just slower, with the damage
landing elsewhere. This harness's own smoke test evicted `muse-glimmer-30b` on
its first run, which is why the guard exists.

To proceed deliberately, pass `acknowledgeEviction: true` and call
`restorePinned(host)` afterwards. For heavy batch work prefer the GX10 vLLM
mode, which switches the host **once** by design instead of per request.

**Local models are preferred for reproducibility, not cost.** A pinned local
weight file has a fixed cutoff. Hosted APIs are updated silently, and if the
model behind a post-cutoff evaluation shifts mid-study the control evaporates
with no warning.

The GX10 vLLM endpoint (`qwen32`, ~500 tok/s at 64 concurrent) is the batch
workhorse, but it is **on-demand and conflicts with llama-swap on that host**:

```bash
ssh gx10 'sudo systemctl start vllm'   # serves :8000, stops llama-swap
ssh gx10 'sudo systemctl stop vllm'    # restores llama-swap on :8085
```

The client detects an unreachable endpoint and says this, rather than surfacing
a bare `ECONNREFUSED` from the middle of a long batch.

## Status

Phase 0 core is built and tested. Not yet built:

- `scripts/probeCutoff.ts` — sweep dates to establish each model's cutoff empirically
- Point-in-time data connectors (SEC EDGAR is free; news and transcripts are not)
- The first study (Option B — earnings events, incremental over post-earnings-announcement drift)

## Design rules

1. **Null, never zero, for "unknown".** A parse failure scored 0 becomes a
   confident neutral opinion; an unmeasured criterion counted as passed lets a
   study succeed by omitting its hardest test.
2. **Refuse rather than guess.** No cutoff means no evaluation.
3. **Every claim gets a dumb baseline.** In the predecessor project the dumb
   baseline won every single time. Assume it will again.
4. **Determinism is a correctness property.** If two runs disagree, both are
   unusable, and you will not know which.

## References

- Lopez-Lira & Tang, *Can ChatGPT Forecast Stock Price Movements?*, Journal of Financial Economics — [arXiv:2304.07619](https://arxiv.org/abs/2304.07619)
- Gao, Jiang & Yan, *Detecting Lookahead Bias in LLM Forecasts* — [arXiv:2512.23847](https://arxiv.org/abs/2512.23847)
- Full options analysis: `docs/llm-trading-proposal.md` in the `bitcoin-trading-tool` repo

**This repository is research tooling, not investment advice.**
