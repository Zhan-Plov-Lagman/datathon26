"""Structural provenance checks; semantic accuracy is validated locally by A/B."""
from copy import deepcopy

from app.inference.departure import validate_context
from app.rules.engine import validate_report


def require(condition):
    if not condition:
        raise ValueError("Invalid fact provenance; inspect locally")


def grounded_packets(bundle, local_context=None):
    """Return a copy, withholding conflicted facts without rewriting A's report.

Missing/mismatched provenance is a broken contract, not semantic uncertainty.
Declared uncertainty and conflicts remain data and follow B's detection policy.
"""
    report = bundle['report']
    windows = validate_report(report)
    context = validate_context(local_context, report)
    packets = bundle.get('assessments')
    episodes = bundle.get('diagnostics', {}).get('episodes')
    require(isinstance(packets, list) and isinstance(episodes, list))
    by_key = {}
    for episode in episodes:
        require(isinstance(episode, dict))
        key = (episode.get('rule_id'), episode.get('event_id'))
        require(all(isinstance(x, str) and x.strip() for x in key))
        require(key not in by_key)
        by_key[key] = episode

    def references(refs):
        require(isinstance(refs, list))
        keys = set()
        for ref in refs:
            require(isinstance(ref, dict))
            wid, index = ref.get('window_id'), ref.get('observation_index')
            require(type(wid) is int and type(index) is int)
            require(wid in windows and 0 <= index < len(windows[wid]['observations']))
            require((wid, index) not in keys)
            keys.add((wid, index))
        return keys

    output, conflicts = deepcopy(packets), []
    for packet in output:
        require(isinstance(packet, dict))
        key = (packet.get('rule_id'), packet.get('event_id'))
        require(all(isinstance(x, str) and x.strip() for x in key) and key in by_key)
        episode = by_key[key]
        packet_refs = references(packet.get('observation_refs'))
        require(bool(packet_refs))
        facts, sources = packet.get('facts'), episode.get('fact_sources')
        require(isinstance(facts, dict) and isinstance(sources, dict))
        for name, value in facts.items():
            require(value is None or type(value) is bool)
            source = sources.get(name)
            require(isinstance(source, dict))
            origin = source.get('origin')
            refs = references(source.get('observation_refs', []))
            require(refs.issubset(packet_refs))
            if value is None:
                require(origin == 'unknown' and isinstance(source.get('reason'), str)
                        and bool(source['reason'].strip()))
                continue
            require(origin in ('visual', 'local_context', 'mixed'))
            if origin in ('visual', 'mixed'):
                require(bool(refs))
            if origin in ('local_context', 'mixed'):
                require(context is not None and isinstance(source.get('context_ref'), dict))
                for field in ('view_id', 'context_version', 'basis_ref', 'reviewer'):
                    require(source['context_ref'].get(field) == context[field])
        reasons = episode.get('conflicts')
        require(isinstance(reasons, list) and all(isinstance(x, str) and x.strip() for x in reasons))
        if reasons:
            # B must not support a packet while a source/episode conflict remains.
            packet['facts'] = dict.fromkeys(facts)
            conflicts.append({'rule_id': key[0], 'event_id': key[1], 'reasons': list(reasons)})
    return output, conflicts
