"""Offline vLLM 0.11 backend; grammar never establishes visual truth."""
import json
import math
import os
from pathlib import Path
import re

from app.inference.qwen import QwenObserver, SYSTEM_PROMPT
from app.inference.schema import ModelOutputError, parse_observations
from app.inference.targeted import FACTS, parse_facts
from app.video.sampling import check_deadline


def facts_schema(frame_count):
    """Constrain shape/types; the existing parser checks semantic relations."""
    index = {'type': 'integer', 'minimum': 0, 'maximum': frame_count - 1}
    fact = {'type': 'object', 'additionalProperties': False,
            'required': ['value', 'frames'], 'properties': {
                'value': {'type': ['boolean', 'null']},
                'frames': {'type': 'array', 'items': index, 'maxItems': frame_count}}}
    events = []
    for rule_id, names in FACTS.items():
        properties = {
            'rule_id': {'type': 'string', 'enum': [rule_id]},
            'start_frame': index, 'end_frame': index, 'anchor_frame': index,
            'actor_box_1000': {'anyOf': [{'type': 'null'}, {
                'type': 'array', 'minItems': 4, 'maxItems': 4,
                'items': {'type': 'number', 'minimum': 0, 'maximum': 1000}}]},
            'description': {'type': 'string', 'minLength': 1, 'maxLength': 600},
            'facts': {'type': 'object', 'additionalProperties': False,
                      'required': list(names), 'properties': {name: fact for name in names}},
        }
        events.append({'type': 'object', 'additionalProperties': False,
                       'required': list(properties), 'properties': properties})
    return {'type': 'object', 'additionalProperties': False, 'required': ['events'],
            'properties': {'events': {'type': 'array', 'maxItems': 2,
                                       'items': {'anyOf': events}}}}


def local_model_info(model_dir):
    path = Path(model_dir).resolve(strict=True)
    try:
        info = json.loads((path / 'download_manifest.json').read_text(encoding='utf-8'))
        config = json.loads((path / 'config.json').read_text(encoding='utf-8'))
        if (not isinstance(info.get('repo_id'), str) or not info['repo_id'].strip()
                or not isinstance(info.get('revision'), str)
                or re.fullmatch(r'[0-9a-fA-F]{40}', info['revision']) is None
                or config.get('model_type') != 'qwen3_vl'
                or config.get('quantization_config', {}).get('quant_method') != 'fp8'):
            raise ValueError()
    except (OSError, ValueError, TypeError, AttributeError):
        raise ValueError('Pinned local Qwen3-VL FP8 model/config required') from None
    return path, info


class VllmObserver(QwenObserver):
    def __init__(self, model_dir, max_new_tokens=1024, *, memory_fraction=.90):
        path, info = local_model_info(model_dir)
        if type(max_new_tokens) is not int or not 0 < max_new_tokens < 8192:
            raise ValueError('Invalid output token limit')
        if not math.isfinite(memory_fraction) or not 0 < memory_fraction <= 1:
            raise ValueError('Invalid GPU memory fraction')
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
            HF_HUB_DISABLE_TELEMETRY='1', VLLM_NO_USAGE_STATS='1',
            DO_NOT_TRACK='1', VLLM_WORKER_MULTIPROC_METHOD='spawn')
        import torch
        from transformers import AutoProcessor
        from vllm import LLM
        from vllm.sampling_params import StructuredOutputsParams, SamplingParams
        self.model_id, self.model_revision = info['repo_id'], info['revision']
        self.torch, self.max_new_tokens, self.language = torch, max_new_tokens, 'en'
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.SamplingParams, self.StructuredOutputsParams = SamplingParams, StructuredOutputsParams
        self.model = LLM(model=str(path), tokenizer=str(path), trust_remote_code=False,
            dtype='half', max_model_len=8192, max_num_seqs=1,
            limit_mm_per_prompt={'image': 18, 'video': 0},
            gpu_memory_utilization=memory_fraction, enforce_eager=True,
            enable_prefix_caching=True, disable_log_stats=True, seed=0)

    def _observe_once(self, frames, previous_context, deadline, concise=False, *, system_prompt=None, parser=None):
        check_deadline(deadline)
        if not 1 <= len(frames) <= 18:
            raise ValueError('Between 1 and 18 frames required')
        content = []
        for index, _ in enumerate(frames):
            content.extend([{'type': 'text', 'text': f'Image index {index}:'}, {'type': 'image'}])
        content.append({'type': 'text', 'text': f'Exactly {len(frames)} images. Indices are image positions, never seconds. '
            'Return only the requested JSON. Prior tentative context: ' + previous_context[:600]})
        system = system_prompt or SYSTEM_PROMPT
        if concise:
            system += '\nKeep descriptions short. Match the requested schema and frame indices exactly.'
        prompt = self.processor.apply_chat_template([
            {'role': 'system', 'content': system}, {'role': 'user', 'content': content}],
            tokenize=False, add_generation_prompt=True)
        constraint = ({'json': facts_schema(len(frames))} if parser is parse_facts
                      else {'json_object': True})
        params = self.SamplingParams(temperature=0, max_tokens=self.max_new_tokens,
            structured_outputs=self.StructuredOutputsParams(**constraint))
        outputs = self.model.generate([{'prompt': prompt, 'multi_modal_data': {
            'image': [frame.image for frame in frames]}}], sampling_params=params, use_tqdm=False)
        check_deadline(deadline)
        if len(outputs) != 1 or len(outputs[0].outputs) != 1:
            raise ModelOutputError('invalid_schema')
        generated = outputs[0].outputs[0]
        try:
            # Even valid-looking truncated JSON must not become a negative verdict.
            if generated.finish_reason != 'stop':
                raise ModelOutputError('invalid_schema')
            return (parser or parse_observations)(generated.text, [f.timestamp_sec for f in frames])
        except ModelOutputError as exc:
            exc.generated_tokens = len(generated.token_ids)
            exc.hit_token_limit = generated.finish_reason == 'length'
            sink = getattr(self, '_invalid_response_sink', None)
            if sink is not None:
                sink(generated.text, {'reason': exc.reason, 'generated_tokens': exc.generated_tokens,
                     'hit_token_limit': exc.hit_token_limit, 'frame_count': len(frames)})
            raise

    def metrics(self):
        return {**super().metrics(), 'gpu_metrics_scope': 'caller_process_only_vllm_engine_separate'}
