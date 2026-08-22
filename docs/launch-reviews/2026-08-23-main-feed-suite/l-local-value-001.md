# L-LOCAL-VALUE-001 — Feed-guarded Local Value Tree

Status: `hold_hlt_risk` on a fixed treatment catalog.

## Change

Within candidates no more than 0.03 below the base Feed score, add a 0.15 Local
Value proxy using interest, POI value, city match, and quality. Supply is frozen,
so viewer-level randomization is valid and does not contain author interference.

## Known DGP effect

| Metric | Relative effect |
|---|---:|
| Stay | -0.033% |
| LT | +0.067% |
| HLT | +0.412% |
| Anchor click | +1.064% |
| Local Service Value | +0.894% |
| Long-term Feed Value | +0.170% |

The 300-user observed A/B was noisy: HLT -10.19%, anchor click -10.35%, and
Local Value -4.16%, none significant. Randomization audit contains the paired
DGP truth, so the direction mismatch is attributed to low power rather than
silently presented as a regression or win.

## Decision

Hold. The treatment has a plausible sub-1% Local Value effect with neutral Feed
truth, but the current viewer sample cannot clear HLT or Local Value uncertainty.
Increase the fixed-catalog viewer sample and use CUPED before reconsidering.
