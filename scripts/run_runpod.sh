#!/usr/bin/env bash
# Foreground supervisor for the existing /workspace/forte Runpod installation.
set +x
set -Eeuo pipefail
set +m # setsid children must not already be process-group leaders.
umask 077

die() { printf 'run_runpod: %s\n' "$*" >&2; exit 1; }
[[ $# == 0 ]] || die 'No arguments supported; run bash scripts/run_runpod.sh.'
[[ $(uname -s) == Linux ]] || die 'This launcher requires Linux.'
(( BASH_VERSINFO[0] > 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] >= 3) )) \
    || die 'Bash 4.3 or newer is required.'
for command_name in setsid flock ps; do
    command -v "$command_name" >/dev/null || die "Required command missing: $command_name"
done

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd -- "$project_root"
api_python="$project_root/.venv/bin/python"
worker_python="$project_root/.venv-vllm/bin/python"
[[ -x $api_python && -x $worker_python ]] || die 'Both existing Python environments are required.'
[[ -r .env ]] || die 'Existing .env is required; this launcher does not create keys.'
[[ -r config/runpod-l40s-vllm.env ]] || die 'Missing L40S vLLM configuration.'

# The tracked model configuration contains no secrets. Python loads .env itself.
set -a
source config/runpod-l40s-vllm.env
set +a
export FORTE_STORAGE="$project_root/.runtime"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# The outer queue worker already spawns the model process. Actual GPU validation
# of this topology is recorded separately; this is not a benchmark result.
export VLLM_ENABLE_V1_MULTIPROCESSING=0
for model_path in "${FORTE_MODEL_DIR:?}" "${FORTE_PEOPLE_MODEL_DIR:?}" "${FORTE_OBJECT_MODEL_DIR:?}"; do
    [[ -d $model_path ]] || die 'Configured local model directory is missing; run setup separately.'
done
[[ ${FORTE_VLM_BACKEND:-} == vllm && ${FORTE_MODEL_QUANTIZATION:-} == fp8 ]] \
    || die 'This launcher requires the explicit vLLM FP8 configuration.'

mkdir -p -- "$FORTE_STORAGE/worker" artifacts/server-setup
exec 9>"$FORTE_STORAGE/runpod-service.lock"
flock -n 9 || die 'Another supervisor holds the service lock; nothing was stopped.'

# Validate keys without displaying them. Refuse both occupied ports and an
# existing same-checkout worker, which has no listening port of its own.
"$api_python" - <<'PY'
import os
from pathlib import Path
import socket
from dotenv import load_dotenv
from app.api.settings import Settings

load_dotenv(override=False)
try:
    Settings.from_env()
except Exception:
    raise SystemExit('run_runpod: Invalid API/worker settings; check .env privately.') from None
root = Path.cwd().resolve()
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit() or int(entry.name) == os.getpid():
        continue
    try:
        if (entry / 'cwd').resolve() != root:
            continue
        args = (entry / 'cmdline').read_bytes().split(b'\0')
    except (OSError, RuntimeError):
        continue
    modules = (b'app.api', b'app.jobs.worker')
    if any(a == b'-m' and b in modules for a, b in zip(args, args[1:])):
        raise SystemExit('run_runpod: Existing API/worker in this checkout; nothing was stopped.')
for port in (8000, 8002):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(('0.0.0.0', port))
        except OSError:
            raise SystemExit(f'run_runpod: Port {port} unavailable; nothing was stopped.') from None
PY

run_dir=$(mktemp -d "$project_root/artifacts/server-setup/run-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
api_pid=''
worker_pid=''
signal_status=0
cleanup() {
    local status=$?
    trap - EXIT
    trap '' INT TERM HUP
    set +e
    # Only signal process groups created by this invocation; never pkill/fuser.
    for child_pid in "$worker_pid" "$api_pid"; do
        [[ -n $child_pid ]] && kill -TERM -- "-$child_pid" 2>/dev/null
    done
    for (( attempt=0; attempt<50; attempt++ )); do
        local alive=0
        for child_pid in "$worker_pid" "$api_pid"; do
            [[ -n $child_pid ]] && kill -0 -- "-$child_pid" 2>/dev/null && alive=1
        done
        (( alive == 0 )) && break
        sleep 0.2
    done
    for child_pid in "$worker_pid" "$api_pid"; do
        if [[ -n $child_pid ]]; then
            kill -KILL -- "-$child_pid" 2>/dev/null
            wait "$child_pid" 2>/dev/null
        fi
    done
    printf 'run_runpod: owned service groups stopped (exit %s).\n' "$status"
    exit "$status"
}
trap cleanup EXIT
# Defer exiting until the newly launched PID is recorded, avoiding orphaned
# children if a signal arrives between background launch and assignment.
trap 'signal_status=130' INT
trap 'signal_status=143' TERM
trap 'signal_status=129' HUP
check_signal() { (( signal_status == 0 )) || exit "$signal_status"; }
check_group() {
    local pid=$1 group=''
    for (( attempt=0; attempt<20; attempt++ )); do
        group=$(ps -o pgid= -p "$pid" 2>/dev/null || true)
        group=${group//[[:space:]]/}
        [[ $group == "$pid" ]] && return 0
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.05
    done
    die 'Child did not start in its own process group; inspect private logs.'
}

printf 'run_runpod: private logs: %s\n' "$run_dir"
check_signal
setsid "$api_python" -m app.api \
    --public-host 0.0.0.0 --public-port 8000 \
    --worker-host 127.0.0.1 --worker-port 8002 \
    >"$run_dir/api.log" 2>&1 9>&- &
api_pid=$!
check_signal
check_group "$api_pid"

# Wait for both listeners, not model readiness. The public readiness endpoint
# remains authoritative after the worker has loaded and registered.
# Use short probes so TERM does not wait for a 30-second foreground Python call.
listeners_ready=0
listener_deadline=$(( SECONDS + 30 ))
while (( SECONDS < listener_deadline )); do
    check_signal
    kill -0 "$api_pid" 2>/dev/null || die 'API exited during startup; inspect private logs.'
    if "$api_python" - <<'PY'
import socket
import sys
try:
    for port in (8000, 8002):
        with socket.create_connection(('127.0.0.1', port), timeout=0.2):
            pass
except OSError:
    sys.exit(1)
PY
    then
        listeners_ready=1
        break
    fi
    sleep 0.2
done
check_signal
(( listeners_ready == 1 )) || die 'API listeners timed out; inspect private logs.'
kill -0 "$api_pid" 2>/dev/null || die 'API stopped before worker startup.'
setsid "$worker_python" -m app.jobs.worker \
    --coordinator http://127.0.0.1:8002 --worker-id l40s \
    --factory app.jobs.unified:create_signs_processor \
    --startup-timeout 360 --work-dir "$FORTE_STORAGE/worker" \
    >"$run_dir/worker.log" 2>&1 9>&- &
worker_pid=$!
check_signal
check_group "$worker_pid"
printf 'run_runpod: API and worker launched; model readiness is not yet established.\n'
status=0
wait -n "$api_pid" "$worker_pid" || status=$?
check_signal
# A clean child exit is also an unexpected service stop; terminate its peer.
(( status != 0 )) || status=1
printf 'run_runpod: a service process exited; shutting down the owned groups.\n' >&2
exit "$status"
