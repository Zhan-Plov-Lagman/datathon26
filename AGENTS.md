# Instructions for coding assistants

Read `README.md`, `TEAM_WORKFLOW.md`, `docs/CONTRACTS.md`, and the assigned file in `prompts/` before implementing changes. The project has three human teammates using three separate laptops. Do not assume that you are responsible for all three roles.

- Role A, RTX 4050: video decoding, local inference, timestamps, evidence, and GPU worker. Owned areas: `app/video/`, `app/inference/`, `app/evidence/`, related tests and worker documentation.
- Role B, RTX 4050: regulatory rule catalog, rule evaluation, dataset annotation and quality evaluation. Owned areas: `app/rules/`, `evaluation/`, `config/rules.yaml` and related tests.
- Role C, RTX 3050: public API, durable queue, UI, deployment and integration. Owned areas: `app/api/`, `app/jobs/`, `web/` and deployment files.

The role is specified by the user's starting prompt or branch. If neither specifies it, ask which role before implementing a module; read-only setup checks can proceed. Do not spawn extra coding agents unless the user explicitly requests delegation. Work on your role branch and do not overwrite other roles' work. Coordinate changes to shared contracts and dependencies in documentation and review them before merging.

The deadline is 2026-10-18 12:00 Asia/Qyzylorda. Prefer one working end-to-end path, measured latency and tested evidence over extra features. Do not claim that a stub, an untested model or a partial run is a working analysis service.

Bank videos, frames, frame descriptions and model outputs containing scene descriptions must remain inside the team's local processing environment. Never send them to cloud AI APIs, browser tools, remote assistants or external logging services. Do not display actual bank frames or scene descriptions in the coding assistant's tool output. Local scripts may process them; return only technical statistics and aggregate metrics to the assistant. Synthetic fixtures without bank content may be used for code debugging. Do not identify faces or analyze audio.

Do not commit source data, labels, evidence, local audits, model weights, secrets or generated video reports. Use the organizer's source and `docs/DATA_SETUP.md` for data distribution. Public code, schemas, prompts without actual scene content, verified public rules and aggregate metrics belong in Git. Inspect the staged diff before committing. Never use filenames or hashes to substitute ground-truth labels for inference.

Runtime analysis must load local model files without network access. Model download is a separate setup command. Start with Qwen3-VL-2B-Instruct and measured 4-bit inference on a 4050; alternative configurations require an explicit benchmark. Do not silently fall back to a cloud model or return `no_violation` after a failure.

Follow the jury API exactly. Technical failure is `failed`; successful `done` has either `violation` with events or `no_violation` with an empty event list. Observation extraction alone cannot supply a legal verdict. Regulatory citations must come from the reviewed rule registry. Use actual video timestamps and source JPEG evidence. Keep model/rule/version information in internal results.

Run focused tests for changed behavior and report actual execution results, remaining limitations and next integration requirements. Update your role handoff document so the next teammate can continue without private chat history. Never reset or delete a teammate's changes or force-push main.
