# PathSnap Project Status

**Review date:** 2026-11-18  
**Current stage:** Stage 1 validation review  
**Next stage:** Do not begin Stage 2 until the remaining alignment issue below has been reviewed.

## Scope and integrity

- Reviewed `src/pathsnap/alignment.py`, the Stage 1 readers/validation, diagnostic outputs, and targeted tests.
- Recomputed diagnostics using the 80 training trips only. No student-test route files were read during this review. The evaluator partition remains absent and was not substituted.
- Did not modify raw files, train models, or begin Stage 2.

## Findings and fixes

### Repeated route occurrences and the 20-state cap

The original code expanded every nearby route arc into all route occurrences, sorted those states by emission distance, then truncated the state list to 20. A later occurrence of a repeated arc could therefore be cut off even when the same road geometry was a viable candidate.

Training evidence: 9 trips contain repeated route arc IDs (67 additional occurrences total; maximum three occurrences for one arc). In the old implementation, the cap dropped repeated-occurrence states in trip `00000018` (188), `00000069` (878), and `00000095` (196). The updated selection prioritizes nearby arcs that repeat in the route and retains every occurrence for them. Post-fix checks found all offered repeated-occurrence states retained for those trips: 618/618, 3,800/3,800, and 1,544/1,544. A targeted unit test places a repeated candidate beyond the 20 nearest distinct candidates and verifies both route occurrences remain.

### Monotonic transitions and timestamp gaps

The old 2 m tolerance allowed real backward movement in route chainage or route position. Separately, the single unmatched DP state forgot the last matched route position, allowing backward re-entry after an unmatched observation. Among trips with supplied route breaks, the old saved alignments contained 392 backward transitions between consecutive matched observations, including transitions separated by unmatched rows.

The decoder now requires nondecreasing route position and chainage within a route occurrence. After an unmatched row, it chooses a legal forward candidate when available and flags that choice; if none is available, it leaves that observation unmatched and records the correction. For timestamp intervals greater than 5 seconds or nonpositive intervals, the displacement consistency term is skipped while route order remains enforced. The interval is still surfaced in the per-observation output.

### Low candidate coverage: trips 00000054 and 00000033

Candidate coverage uses the unchanged 80 m radius and is measured before alignment-state selection. The route arc IDs in both trips are unique, so repeated occurrences and state truncation do not explain their low candidate coverage.

| Training trip | Coverage | GPS points within 80 m of a reference-route edge | Nearest reference-edge distance (median / p90 / max) |
|---|---:|---:|---:|
| 00000054 | 14.8967% | 2,877 / 19,313 | 138.6 m / 196.6 m / 241.3 m |
| 00000033 | 48.7166% | 2,847 / 5,844 | 122.4 m / 139.7 m / 160.5 m |

Trip 00000054 has six timestamp gaps over five seconds; trip 00000033 has none. Their GPS points are often substantially farther than the configured candidate radius from the supplied reference-route geometry. This is an observed data/alignment mismatch, not a reason to enlarge the radius during this review. The existing route geometry and distance method should be visually and semantically reviewed before Stage 2 uses these labels.

### Invalid route transitions

The training partition contains 13 noncontiguous consecutive route transitions across 10 trips. Validation compares each route arc's destination node ID with the next route arc's source node ID and reports the break without modifying the route.

The old per-row flag only compared adjacent GPS observations. It missed 72 decoded non-null spans that crossed a supplied route break across an unmatched interval or backward re-entry. The decoder now blocks direct matched transitions across a broken route edge. A forward re-entry after an unmatched interval is explicitly marked. The refreshed training diagnostics contain 11 spans crossing a supplied break, all flagged, with zero unflagged spans, zero direct matched crossings, and zero backward route positions.

## Verification and current diagnostic outputs

- `python -m pytest -q --basetemp=.pytest-tmp`: **15 passed**.
- `python -m compileall -q src tests`: passed.
- Refreshed the full training-only diagnostic outputs for all 80 trips. Mean candidate coverage remains 98.02% (unweighted across trips).
- Updated per-observation CSVs now include `timestamp_order_violation_before`, `crosses_invalid_route_transition`, and `nonmonotonic_state_rejected`.
- No ML models were trained and no student-test routes were opened.

## Remaining issue before Stage 2

The corrected output flags **40,250** observations (about 19.3% of all training GPS observations) where the original Viterbi choice would have re-entered backward after an unmatched row and a forward candidate was selected instead. This is visible rather than silent, but the volume is high enough that the alignment method should be reviewed before treating the resulting labels as suitable ML targets. Trips 00000054 and 00000033 also retain poor geometric candidate coverage. Do not tune the radius or alignment costs just to raise coverage; decide and document the label policy first.

No Stage 2 work is started by this review.
