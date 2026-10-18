"""Opt-in evaluator implementation. No release or default factory is supplied.

A/B/C must validate a real detector release before constructing this for use
on real videos. Tests supply synthetic releases only in memory.
"""
from copy import deepcopy
import hashlib
import json
import time
from typing import Annotated

from pydantic import Field

from app.api.schema import StrictModel, Text
from app.jobs.detection import finalize_bundle
from app.jobs.pipeline import RuleDecision, SupportedEvent
from app.rules.engine import validate_registry
from app.rules.policy import POLICY_VERSION


class ValidatedRelease(StrictModel):
    validation_reference: Text
    registry_sha256: Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]
    model_id: Text
    model_revision: Text
    detector_versions: Annotated[dict[Text, Text], Field(min_length=1)]
    policy_version: Text


def registry_digest(registry):
    return hashlib.sha256(json.dumps(registry, sort_keys=True).encode()).hexdigest()


def require(condition):
    if not condition:
        raise ValueError('Detection release does not match execution')


class DetectionEvaluator:
    """Implements C's evaluate interface using pinned, locally validated versions.

The release record binds versions, not proof of quality. Its reference must
come from real local acceptance, never from a generated placeholder.
"""
    reviewed = False

    def __init__(self, *, registry, release, context_provider=None):
        self._registry = deepcopy(registry)
        self._release = ValidatedRelease.model_validate(release).model_copy(deep=True)
        validate_registry(self._registry)
        require(registry_digest(self._registry) == self._release.registry_sha256)
        require(self._release.policy_version == POLICY_VERSION)
        rules = {r['rule_id']: r for r in self._registry['rules']}
        for rule_id in self._release.detector_versions:
            require(rule_id in rules and rules[rule_id]['status'] == 'reviewed'
                    and rules[rule_id].get('validated_on_local_cases') is True)
        self._context_provider = context_provider
        self.reviewed = True

    def evaluate(self, report, *, video_path, worker, deadline):
        require(report.get('model', {}).get('id') == self._release.model_id)
        require(report.get('model', {}).get('revision') == self._release.model_revision)
        context = self._context_provider(deepcopy(report)) if self._context_provider else None
        remaining = deadline - time.monotonic() - 5.0  # reserve for rules/JPEG
        if remaining <= 0:
            raise TimeoutError('Analysis budget exhausted')
        bundle = worker.assess(video_path, deepcopy(report), timeout_sec=remaining,
                               deadline=deadline, local_context=deepcopy(context))
        if time.monotonic() >= deadline:
            raise TimeoutError('Analysis budget exhausted')
        result = finalize_bundle(report, bundle, registry=self._registry,
                                 supported_rule_ids=list(self._release.detector_versions),
                                 local_context=context)
        runs = bundle['diagnostics']['search']['detector_runs']
        require({r['rule_id']: r['detector_version'] for r in runs} == self._release.detector_versions)
        require(result['policy_version'] == self._release.policy_version)
        require(result['registry_sha256'] == self._release.registry_sha256)
        windows = {w['window_id']: w for w in bundle['report']['windows']}
        events = []
        for event in result['supported']:
            descriptions = []
            for ref in event['observation_refs']:
                observation = windows[ref['window_id']]['observations'][ref['observation_index']]
                # Structured feature strings are diagnostics, not prose evidence.
                if ('feature_key' not in observation
                        and observation['start_sec'] <= event['anchor_sec'] <= observation['end_sec']):
                    description = observation.get('description')
                    require(isinstance(description, str) and bool(description.strip()))
                    descriptions.append(description.strip())
            require(bool(descriptions))
            events.append(SupportedEvent(rule_id=event['rule_id'], type=event['type'],
                anchor_sec=event['anchor_sec'],
                description=f'{descriptions[0]} {event["description"]}',
                evidence_timestamps=[event['anchor_sec']]))
        if time.monotonic() >= deadline:
            raise TimeoutError('Analysis budget exhausted')
        return RuleDecision(complete=True, rules_reviewed=True, rules_version=result['registry_version'],
                            supported=events, negative_policy_version=result['negative_policy_version'])
