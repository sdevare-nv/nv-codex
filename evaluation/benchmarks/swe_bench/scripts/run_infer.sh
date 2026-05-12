#!/usr/bin/env bash
# Bench entry script — invoked by gym's CodexHarnessProcessor.get_run_command().
#
# Args (positional, must match the order in app.py's get_run_command):
#   $1  COMMIT_HASH        codex commit (informational; checkout done at setup)
#   $2  AGENT              agent class name (informational)
#   $3  MAX_ITER           max agent turns
#   $4  DATASET            dataset name
#   $5  SPLIT              dataset split
#   $6  EVAL_OUTPUT_DIR    where output.jsonl lands (relative to codex dir)
#   $7  SELECTED_ID        instance_id to run
#   $8  INSTANCE_DICT_PATH /root/dataset/data.jsonl
#   $9  MODEL_NAME         model name to send to codex (resolved gym-side)
#   $10 WORKSPACE_ROOT     repo path inside the SIF
#   $11 USER_MESSAGE_PATH  pre-rendered user prompt file
#   $12 SYSTEM_PROMPT_PATH optional system-prompt override
#
# Environment (set by gym):
#   NEMO_GYM_MODEL_SERVER_NAME      proxy name on the gym head server
#   NEMO_GYM_MODEL_SERVER_BASE_URL  base http://host:port for the model server
#   NEMO_GYM_WIRE_API               "responses" (openai_model) or "chat" (vllm_model)
#   NEMO_GYM_METRICS_FPATH          path to the metrics JSON to update
#   POLICY_API_KEY                  bearer token (codex requires env_key set)
#   COMMAND_EXEC_TIMEOUT            per-bash-command timeout in seconds

set -eo pipefail

COMMIT_HASH="${1:-}"
AGENT="${2:-}"
MAX_ITER="${3:-100}"
DATASET="${4:-}"
SPLIT="${5:-test}"
EVAL_OUTPUT_DIR="${6:-evaluation/oh}"
SELECTED_ID="${7:-}"
INSTANCE_DICT_PATH="${8:-/root/dataset/data.jsonl}"
MODEL_NAME="${9:-}"
WORKSPACE_ROOT="${10:-}"
USER_MESSAGE_PATH="${11:-}"
SYSTEM_PROMPT_PATH="${12:-}"

for var in SELECTED_ID MODEL_NAME WORKSPACE_ROOT USER_MESSAGE_PATH \
           NEMO_GYM_MODEL_SERVER_BASE_URL NEMO_GYM_WIRE_API; do
    if [ -z "${!var:-}" ]; then
        echo "ERROR: $var is required" >&2
        exit 64
    fi
done

# Resolve the codex dir (the script lives at evaluation/.../scripts/run_infer.sh
# inside the staged tree; go up 4 levels).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODEX_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
CODEX_BIN="$CODEX_DIR/bin/codex"
PROXY_SCRIPT="$CODEX_DIR/evaluation/benchmarks/swe_bench/scripts/nemo_gym_proxy.py"

[ -x "$CODEX_BIN" ] || { echo "ERROR: codex binary missing at $CODEX_BIN"; exit 69; }
[ -f "$PROXY_SCRIPT" ] || { echo "ERROR: proxy script missing at $PROXY_SCRIPT"; exit 70; }

# Absolutize EVAL_OUTPUT_DIR.
case "$EVAL_OUTPUT_DIR" in
    /*) ABS_OUTPUT_DIR="$EVAL_OUTPUT_DIR" ;;
    *)  ABS_OUTPUT_DIR="$CODEX_DIR/$EVAL_OUTPUT_DIR" ;;
esac
RUN_DIR="$ABS_OUTPUT_DIR/$SELECTED_ID/bench_run"
COMPLETIONS_DIR="$RUN_DIR/llm_completions/$SELECTED_ID"
mkdir -p "$RUN_DIR" "$COMPLETIONS_DIR"

# Per-instance CODEX_HOME — keeps codex's session rollouts / auth / cache out
# of the user's real ~/.codex, and lets us point it at a writable path inside
# the SIF (Lustre would be a metadata nightmare for codex's many small writes).
CODEX_HOME_DIR="$(mktemp -d -t codex-home.XXXXXX)"
export CODEX_HOME="$CODEX_HOME_DIR"

# Resolve a Python 3 interpreter for the sidecar proxy. Probe in order:
#   1. The bundled python-build-standalone interpreter staged alongside codex
#      itself. Setup script downloads it; self-contained, runs on any SIF.
#   2. PATH-resolved python3 / python — works after `conda activate testbed`.
#   3. Known conda + system paths — last-ditch fallbacks.
PYTHON_BIN=""
for cand in \
    /codex_setup/codex/python/bin/python3 \
    /codex_setup/codex/python/bin/python3.12 \
    python3 \
    python \
    /opt/miniconda3/envs/testbed/bin/python \
    /opt/miniconda3/bin/python \
    /usr/bin/python3 \
    /usr/bin/python; do
    if command -v "$cand" >/dev/null 2>&1 || [ -x "$cand" ]; then
        # Sanity check: must be Python 3.
        if "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3,7) else 1)' 2>/dev/null; then
            PYTHON_BIN="$cand"
            break
        fi
    fi
done
if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: no Python 3 interpreter found (bundled or system)" >&2
    exit 72
fi
echo "Using PYTHON_BIN=$PYTHON_BIN ($("$PYTHON_BIN" --version 2>&1))"

# Start the sidecar proxy. It listens on a free localhost port, mirrors each
# request/response to disk in openhands-compatible llm_completions JSONs, then
# forwards to NEMO_GYM_MODEL_SERVER_BASE_URL.
PROXY_LOG="$RUN_DIR/nemo_gym_proxy.log"
PROXY_PORT=$("$PYTHON_BIN" - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
)
echo "Starting sidecar proxy on 127.0.0.1:$PROXY_PORT -> $NEMO_GYM_MODEL_SERVER_BASE_URL"
"$PYTHON_BIN" "$PROXY_SCRIPT" \
    --listen-port "$PROXY_PORT" \
    --upstream-url "$NEMO_GYM_MODEL_SERVER_BASE_URL" \
    --completions-dir "$COMPLETIONS_DIR" \
    --instance-id "$SELECTED_ID" \
    --model-name "$MODEL_NAME" \
    >"$PROXY_LOG" 2>&1 &
PROXY_PID=$!

# Poll for readiness — proxy writes a `ready` file when it's listening.
for i in $(seq 1 50); do
    if ! kill -0 "$PROXY_PID" 2>/dev/null; then
        echo "ERROR: proxy died during startup. Log tail:" >&2
        tail -20 "$PROXY_LOG" >&2 || true
        exit 71
    fi
    if "$PYTHON_BIN" -c "import socket,sys; s=socket.socket(); s.settimeout(0.2); s.connect(('127.0.0.1',$PROXY_PORT)); s.close()" 2>/dev/null; then
        echo "Proxy ready after ${i}*0.2s"
        break
    fi
    sleep 0.2
done

cleanup() {
    if kill -0 "$PROXY_PID" 2>/dev/null; then
        kill "$PROXY_PID" 2>/dev/null || true
        wait "$PROXY_PID" 2>/dev/null || true
    fi
    rm -rf "$CODEX_HOME_DIR"
}
trap cleanup EXIT

# Build the per-instance codex config.toml. The provider's `env_key` tells
# codex which env var to read the bearer token from; we already exported
# POLICY_API_KEY. `wire_api` is "responses" for openai_model, "chat" for
# vllm_model — chosen by the gym side.
CONFIG_TOML="$CODEX_HOME_DIR/config.toml"
cat >"$CONFIG_TOML" <<EOF
model = "$MODEL_NAME"
model_provider = "nemo-gym"

# Skip AGENTS.md auto-injection. Codex embeds <workspace>/AGENTS.md into the
# system prompt as a "<INSTRUCTIONS>" block. If the agent creates or edits an
# AGENTS.md mid-rollout the system prompt shifts on subsequent turns and the
# RL prompt-token-prefix invariant breaks.
project_doc_max_bytes = 0

# Push auto-compaction out of reach. Codex auto-summarizes when token usage
# nears \`model_auto_compact_token_limit\`; that injects a synthetic summary
# message and drops prior turns, which also breaks the prefix invariant.
# Setting both to a value larger than any plausible rollout context keeps the
# code path dormant.
model_context_window = 1000000
model_auto_compact_token_limit = 1000000

[model_providers.nemo-gym]
name = "nemo-gym"
base_url = "http://127.0.0.1:${PROXY_PORT}/v1"
env_key = "POLICY_API_KEY"
wire_api = "$NEMO_GYM_WIRE_API"
EOF
echo "Wrote codex config:"
cat "$CONFIG_TOML"

# Read the user prompt.
USER_PROMPT_CONTENT="$(cat "$USER_MESSAGE_PATH")"

# Assemble codex CLI args.
LAST_MSG_FILE="$RUN_DIR/last_message.txt"
JSONL_LOG="$RUN_DIR/codex_events.jsonl"

# Codex's CLI: `codex exec --json --cd <workspace> [prompt]`.
# --dangerously-bypass-approvals-and-sandbox: required so codex doesn't ask
#   the user for permission before each command (we're headless).
# --skip-git-repo-check: SWE-bench workspaces are git repos but codex'
#   default policy may still reject them in some configs.
# --ephemeral: don't persist session rollouts to CODEX_HOME/sessions/ (we
#   already mirror everything via the proxy).
cmd=(
    "$CODEX_BIN"
    exec
    --json
    --skip-git-repo-check
    --dangerously-bypass-approvals-and-sandbox
    --ephemeral
    --ignore-user-config        # don't merge user ~/.codex/config.toml
    --ignore-rules              # don't load user/project .rules files
    --cd "$WORKSPACE_ROOT"
    --output-last-message "$LAST_MSG_FILE"
)
if [ -n "$SYSTEM_PROMPT_PATH" ] && [ -f "$SYSTEM_PROMPT_PATH" ]; then
    # Codex doesn't expose a direct --system-prompt flag; prepend the override
    # to the user prompt so it's part of turn 0. (Future: write to
    # $CODEX_HOME/AGENTS.md, which codex auto-reads, for a cleaner injection.)
    USER_PROMPT_CONTENT="$(cat "$SYSTEM_PROMPT_PATH")"$'\n\n'"$USER_PROMPT_CONTENT"
fi
cmd+=("$USER_PROMPT_CONTENT")

echo "Executing: $CODEX_BIN exec --json --cd $WORKSPACE_ROOT (prompt $((${#USER_PROMPT_CONTENT})) chars)"
set +e
"${cmd[@]}" >"$JSONL_LOG" 2>&1
CODEX_EXIT=$?
set -e
echo "[bench] codex exec exit=$CODEX_EXIT (log: $JSONL_LOG)"

# Capture the final patch via git diff inside the workspace.
PATCH_FILE="$RUN_DIR/patch.diff"
GIT_PAGER=cat git -C "$WORKSPACE_ROOT" diff > "$PATCH_FILE" 2>/dev/null || echo "" > "$PATCH_FILE"
PATCH_BYTES=$(stat -c '%s' "$PATCH_FILE" 2>/dev/null || echo 0)

# Write output.jsonl in the openhands-compatible shape.
ERROR_FIELD="null"
if [ "$CODEX_EXIT" -ne 0 ]; then
    ERROR_FIELD="\"codex_exit_${CODEX_EXIT}\""
fi
OUTPUT_JSONL="$RUN_DIR/output.jsonl"
"$PYTHON_BIN" - <<PY > "$OUTPUT_JSONL"
import json
with open("$PATCH_FILE") as f:
    patch = f.read()
print(json.dumps({
    "instance_id": "$SELECTED_ID",
    "test_result": {"git_patch": patch},
    "metadata": {"llm_config": {"model": "$MODEL_NAME"}},
    "metrics": {"codex_exit_code": $CODEX_EXIT},
    "error": $ERROR_FIELD,
}))
PY

echo "[bench] wrote $OUTPUT_JSONL (patch=$PATCH_BYTES bytes, error=$ERROR_FIELD)"

# Always exit 0 — partial patches + error field are useful signals.
exit 0
