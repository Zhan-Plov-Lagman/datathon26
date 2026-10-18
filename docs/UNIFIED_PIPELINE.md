# CV and targeted VLM pipeline

Status: implemented integration path, **not an accepted detection release**.
Real weights run offline; tests use synthetic inference responses explicitly.
The original `LocalGPUWorker`/v1 coordinator remains available for comparison.

## Execution

1. `app/perception/scan.py` streams display-oriented RGB frames with source PTS.
   RT-DETR runs at 3 sampled frames/s; OWLv2 every 5 s. Track IDs are local
   geometric hypotheses, never facial identities or automatic employee labels.
2. `plan_reviews` preserves periodic/end context and motion/object proposals.
   Static scenes therefore still receive VLM review. Overlapping intervals are
   coalesced only when their full union fits within 20 s; every original interval,
   actor and reason is retained. Review uses 12 frames, an at-most 1.82 s target
   spacing for the largest window. All assigned windows run.
3. `app/inference/unified.py` extracts the selected review frames in one decode
   pass and calls the already loaded Qwen for each window. It does not run the
   old full-video Qwen overview and then repeat that overview inside `assess`.
4. Every proposed episode receives a dense window of up to 4 s / 6 actual PTS.
   Exactly identical refinement intervals share one call. Final fact packets
   come only from these dense windows. Every scheduled call completes or the
   job fails. There is one schema/JSON retry per call within the same deadline.
5. `app/jobs/unified.py` validates v2 provenance, projects selected observations
   to B's v1 **interval** interface, evaluates facts, and extracts source JPEGs
   only for supported public events. A CV scan is never renamed a full VLM scan.
6. The existing C queue, transport, API schema and UI consume `UnifiedProcessor`.
   A technical exception remains `failed`. A completed negative means no
   supported signs in the declared detector scope, not proof of perfect absence.

The initial implemented fact families are `cashier_leaves_service` and
`personal_items_storage`. This is an incremental implementation, not completion
of the full planned class coverage. Count-absence rules and valuables/storage
extensions still require adequate temporal evidence and applicable definitions.
No universal identity, ownership, workstation or service-completion heuristic is
assumed. Their visual hypotheses may remain null. No fake human review is added.

## Contracts and compatibility

`assessment_bundle.v2` contains:

- `report`: `observations.v2`, `coverage_scope=cv_full_video_vlm_targeted`, actual
  review/refinement PTS, observations, job/model identity and producer manifest.
- `perception`: the actual `perception.v1` scan and its selected windows.
- `assessments`: B's fact packets; bool/null, anchor PTS, observation references,
  `refinement_complete=true`, `full_sequence_visible=false`.
- `diagnostics.episodes`: every fact's visual references and supporting PTS,
  or explicit unknown provenance; optional geometric actor-track association.
- `diagnostics.search`: `detector_execution.v2`; separate CV frame lists,
  planned/executed review windows and refinements, attempts, completion status,
  scope and producer-manifest digest.

Validation rejects missing scan frames, missing assigned calls, missing dense
packets, wrong jobs/models, unbound booleans, references to unsampled times and
coarse-only evidence. A source conflict does not become a supported event.
Context facts may reference supplied dense frames before or after the event
itself. They receive separate feature observations with their own source PTS.
The primary event retains its original bounds and anchor; the rule adapter
restores those bounds after evaluating the wider supporting context. Previously,
valid context references were incorrectly rejected as outside the event.
Only overlapping events with the same known track and class are deduplicated;
unknown or different identities are retained. Fragmented tracks can still cause
duplicates and must be evaluated on real cases.

The model-facing geometry field is explicitly `actor_box_1000`, following the
[official Qwen3-VL grounding convention](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb).
Validated coordinates are converted to the shared 0..1 units by division by 1000,
without clipping or guessing a scale from the values. An all-null coordinate
array is canonicalized to null with a diagnostic counter. Mixed/invalid known
coordinates still fail validation. This fixes observed schema errors; it does
not prove geometric or semantic accuracy.

The old `observations.v1` rule engine is unchanged. The adapter explicitly maps
window field names and uses `coverage_scope=interval`; v2 independently verifies
the CV coverage. Sampling limitations are recorded at report level. Occlusion or
unknown applicability belongs in the affected null fact rather than marking
every unrelated fact invalid because the video was sampled.

`producer_manifest.v1` records the actual VLM, people and object models/revisions,
download-manifest digests, runtime versions, thresholds, OWL queries, NF4,
generation limit, preprocessing, prompt digest and hashes of implementation
files. The internal public-result envelope keeps `analysis.v1`; its private
`pipeline_version` includes the manifest digest. The external jury JSON is
unchanged. Code or configuration changes require a matching release record.

## Diagnostic execution without activating rules

```bash
cd /workspace/forte
set -a
source config/runpod-l40s.env
set +a
.venv/bin/python -m scripts.benchmark_unified \
  --video path/to/local-a.mov path/to/local-b.mov \
  --output-dir artifacts/new-run --timeout 240 --block-network
```

The models load once before the series. A failed input is recorded separately;
later inputs continue. Detailed bundles/decisions stay in the artifact directory.
Only technical aggregates are printed. The benchmark reads no labels and does
not turn hypothesis rules into reviewed ones. `fact_supported_candidates` is a
diagnostic count of packets whose remaining reason is `rule_not_reviewed`, not
a public verdict or measured accuracy. The socket block is a Python diagnostic,
not an OS firewall. Never display private bundles or exception payloads in an
external assistant, log collector or browser tool.

## Service activation after acceptance

`FORTE_PROCESSOR_FACTORY=app.jobs.unified:create_processor` selects this path
through the existing `python -m scripts.run_local` supervisor/worker. It requires
`FORTE_UNIFIED_RELEASE` pointing to a real acceptance record and optionally
`FORTE_RULES_PATH` pointing to its matching reviewed registry. No acceptance
record is fabricated by setup or benchmark commands.

The record has schema `unified_release.v1`, registry/manifest SHA-256, policy
`detection-targeted.v2`, the supported rule IDs and an actual validation reference.
The current implemented scope must be tested before activation. Do not use the
synthetic registries/releases from tests on bank inputs or in the jury service.

`FORTE_STARTUP_TIMEOUT=360` in the L40S profile accommodates the measured cold
loading delay. It changes only initialization; upload/queue/job deadlines remain
unchanged. The worker announces readiness after models and release validation.

## Current checks and limits

The real 32B + RT-DETR + OWLv2 pipeline completed a 12 s synthetic 1280x720 MOV:
194.406 s cold loading, 9.767 s processing (5.244 s scan, 0.250 s decode,
4.191 s targeted review), 37 people frames, 3 object frames, 2 VLM calls,
0 refinements/candidates/retries, peak reserved 22144 MiB. Networking was blocked.
This blank-scene run does not measure event refinement quality or bank accuracy.

Synthetic tests exercise actual decode/JPEG/rules, unknown/conflicting facts,
malformed output/recovery, coverage/provenance failures, and HTTP upload -> queue
-> processor -> done/failed. Full current test totals and real development
results are recorded in `INTEGRATION_STATUS.md`, not inferred from this example.

The final development run of `cv-targeted.v1.3` completed all four selected
inputs (5.359 / 139.301 / 6.300 / 147.083 s, in sample order). These numbers
exclude cold loading, upload, queue and public API. No packet had all the
required factual predicates for a supported violation. This is a substantive
quality/coverage gap, not evidence of a ready negative detector. See the status
report for sample-selection limits and the earlier reproducible failures.

Remaining work includes representative positive/hard-negative examples, event
type/time/JPEG checks, full video quality and latency, comparison with 8B and
alternative people detectors, remaining rule scope, and public service rollout.
Do not repeat model setup or all completed benchmarks without a diagnostic need.
