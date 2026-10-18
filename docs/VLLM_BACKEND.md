# Offline vLLM candidate handoff

This is an isolated candidate, not a released analysis service. Keep the existing
Transformers environment. Install requirements-vllm.txt only in the separate Linux
environment and retain its resolved dependency lock with the benchmark artifacts.
The observed server environment uses PyTorch 2.8.0/cu128.

The candidate is Qwen/Qwen3-VL-32B-Instruct-FP8, revision
4bf2c2f39c37c0fede78bede4056e1f18cdf8109. Runtime requires an existing local
download_manifest.json and Qwen3-VL FP8 config. No runtime model download or HTTP
inference service is used. Manifest validation does not rehash all weight files.

vLLM 0.11.0 contains the native Qwen3-VL implementation and accepts local PIL
images in multi_modal_data. StructuredOutputsParams(json=...) constrains the
targeted-facts output shape, known rule identifiers and frame index ranges.
The ordinary parser still checks ordering, attribution, required fact references,
null semantics and boxes. Grammar does not validate visual truth or legal verdicts.
Other parsers retain JSON-object syntax constraints. Prompts are unchanged.

Settings: eager execution, max_model_len=8192, max_num_seqs=1, at most 18 images,
GPU memory utilization 0.90. At 0.88 startup failed: 2.00 GiB KV cache was required but only 1.56 GiB was available. The 0.90 profile preserves the 8192-token context. Actual combined
memory must be measured on the L40S. Large images can exceed the token budget;
errors must remain failures. Caller-process CUDA counters do not measure the
separate vLLM engine. Record whole-device memory separately.

Blocking LLM.generate checks the shared deadline before/after generation. It does
not interrupt an in-flight GPU call; the outer worker must enforce cancellation.
Missing/multiple completions, aborted generation and token-limit truncation fail.

Run from the server checkout with the isolated environment (substitute its path):

```bash
set -a
source config/runpod-l40s-vllm.env
set +a
/path/to/isolated-vllm/bin/python -m scripts.benchmark_unified --video /local/input.mov --output-dir /local/private-benchmark --timeout 240
```

Use synthetic data first. The existing socket-blocking option may also block
vLLM's local multiprocessing transport; offline HF flags alone do not prove a
network-isolated run. Keep bank content and raw exception traces private. Compare
technical success, complete wall time, peak GPU memory, fact-schema retries and
independently reviewed quality before selecting this backend.

Local Windows synthetic adapter checks: **10 passed in 0.09 s**. They cover
image/PTS forwarding, grammar selection, invalid-schema handling, frame cap,
expired and late deadlines, truncated/aborted output, pinned FP8 manifests and
unknown fact preservation. No real GPU inference or xgrammar compilation was
performed by this audit; those remain server benchmark requirements.

Audited against pinned primary sources:

- [v0.11.0 sampling parameters](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/sampling_params.py)
- [v0.11.0 native Qwen3-VL](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/model_executor/models/qwen3_vl.py)
- [v0.11.0 offline LLM API](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/entrypoints/llm.py)

The tested isolated interpreter is `/workspace/forte/.venv-vllm/bin/python`.
To reproduce the selected packages, create `python3 -m venv --system-site-packages .venv-vllm`, then run
`.venv-vllm/bin/python -m pip install --no-cache-dir -r requirements-vllm.txt`.
Do not force-reinstall the CUDA stack. The actual environment resolves
PyTorch 2.8.0+cu128, torchvision 0.23.0+cu128, Transformers 4.57.6,
vLLM 0.11.0, xgrammar 0.1.25 and xformers 0.0.32.post1.
The full resolved lock is retained privately as
`artifacts/server-setup/vllm-requirements-resolved.txt`; the selected direct
requirements are pinned in the repository. Fresh-machine installation from this
file has not been separately rerun after the measured environment was created.
Final measurements and failure history: `docs/BACKEND_COMPARISON.md`.
