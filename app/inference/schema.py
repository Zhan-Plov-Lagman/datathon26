from __future__ import annotations

import json


class ModelOutputError(RuntimeError):
    """The model did not produce a grounded, valid observation structure."""


def parse_observations(raw: str, frame_times: list[float]) -> dict:
    # Never put raw model output in an exception or assistant-visible log.
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
    try:
        body = json.loads(text)
        if not isinstance(body, dict):
            raise ValueError()
        context = body["operation_context"]
        limits = body["visibility_limits"]
        observations = body["observations"]
        if not isinstance(context, str) or len(context) > 1000:
            raise ValueError()
        if not isinstance(limits, list) or len(limits) > 12:
            raise ValueError()
        if any(not isinstance(item, str) or len(item) > 500 for item in limits):
            raise ValueError()
        if not isinstance(observations, list) or len(observations) > 16:
            raise ValueError()
        grounded = []
        for observation in observations:
            start, end = observation["start_frame"], observation["end_frame"]
            description = observation["description"]
            if type(start) is not int or type(end) is not int or not 0 <= start <= end < len(frame_times):
                raise ValueError()
            if not isinstance(description, str) or not description.strip() or len(description) > 1500:
                raise ValueError()
            grounded.append({"start_frame": start, "end_frame": end,
                             "start_sec": frame_times[start], "end_sec": frame_times[end],
                             "description": description.strip()})
        return {"operation_context": context, "visibility_limits": limits, "observations": grounded}
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        raise ModelOutputError("Model output is not valid grounded observation JSON") from None
