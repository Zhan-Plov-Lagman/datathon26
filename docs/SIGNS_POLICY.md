# Explicit signs detection policy

`FORTE_PROCESSOR_FACTORY=app.jobs.unified:create_signs_processor` selects
`signs-targeted.v1`. This implements the original case's two outputs: detected
signs of possible violations, or no signs detected by the configured detector.
It does not assert a final legal determination or measured detection accuracy.

The default `create_processor`, reviewed-release validator and `evaluate`
behavior remain unchanged. Signs mode does not create a release, mark a rule
reviewed, modify the registry, or set `validated_on_local_cases` to true.

Startup requires the two actually implemented `FACTS` classes, active
`hypothesis` or `reviewed` registry entries, verified public source text and a
nonempty document, clause and HTTPS citation. Source verification is distinct
from empirical validation of the visual detector. The registry retains its
original status and limitations.

The same bundle validation checks actual scheduled calls, PTS, dense/context
provenance, job/model identity, completed refinement and boolean evidence.
The same rule checks require all applicability and positive predicates true,
all exceptions false, adequate visibility and required sequence coverage.
Unknown facts remain unknown; contradictory service states cannot qualify.
Descriptions identify signs of a possible violation and cite the actual
registry source, without its categorical legal explanation.

Every assigned scan/review/refinement must complete. A completed scoped search
with no qualifying events returns `no_violation` and an empty list. An incomplete
or technically failed operation raises an error for the existing queue/API to
return `failed`. It cannot become a successful negative result. Source JPEGs
are extracted at the event's actual PTS with the same job deadline.

Private result metadata is `analysis_kind=signs_evaluated`,
`rules_reviewed=false`, `source_texts_verified=true`,
`accuracy_status=unmeasured`, and `negative_policy_version=signs-targeted.v1`.
The public jury JSON is unchanged. The supported scope remains departure and
personal-item storage; no new detector classes or accuracy claims are added.

Synthetic tests cover positive source JPEGs, registry immutability, scoped
negatives, unknown/conflicting facts, incomplete refinement, invalid sources,
explicit factory selection, unchanged default behavior and predicate gates.
Real precision, recall, temporal quality and complete public-API latency must
be reported from actual measurements, separately from this policy selection.
