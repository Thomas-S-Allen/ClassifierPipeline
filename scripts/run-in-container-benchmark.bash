#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET="/app/logs/2023Sci_titles_abstracts_200.csv"
MODE="real"
BATCH_SIZE="100"
TIMEOUT="900"
POLL_INTERVAL="2.0"
OPERATION_STEP="classify_verify"
OUTPUT_DIR="/app/logs/benchmarks/container_benchmark_runs"
EVENTS_PATH="/app/logs/benchmark/perf_events.jsonl"
LABEL=""
RUN_STAMP=""
MODEL_INFERENCE_BATCH_SIZE=""
MODEL_NUM_THREADS=""
MODEL_NUM_INTEROP_THREADS=""
MODEL_DEVICE=""
TOKENIZERS_PARALLELISM=""

usage() {
  cat <<'EOF'
Usage: /app/scripts/run-in-container-benchmark.bash [options]

Options:
  --dataset PATH                     Dataset path inside the container
  --mode real|fake                   Benchmark mode
  --batch-size N                     Benchmark task batch size
  --timeout N                        Benchmark timeout in seconds
  --poll-interval FLOAT              Benchmark poll interval in seconds
  --operation-step STEP              Benchmark operation step
  --output-dir PATH                  Output directory inside the container
  --events-path PATH                 Perf events path inside the container
  --label TEXT                       Stable artifact label prefix
  --run-stamp TEXT                   Reuse an externally supplied run timestamp
  --model-inference-batch-size N     Override MODEL_INFERENCE_BATCH_SIZE
  --model-num-threads N              Override MODEL_NUM_THREADS
  --model-num-interop-threads N      Override MODEL_NUM_INTEROP_THREADS
  --model-device NAME                Override MODEL_DEVICE
  --tokenizers-parallelism VALUE     Override TOKENIZERS_PARALLELISM
  --help                             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT="$2"
      shift 2
      ;;
    --poll-interval)
      POLL_INTERVAL="$2"
      shift 2
      ;;
    --operation-step)
      OPERATION_STEP="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --events-path)
      EVENTS_PATH="$2"
      shift 2
      ;;
    --label)
      LABEL="$2"
      shift 2
      ;;
    --run-stamp)
      RUN_STAMP="$2"
      shift 2
      ;;
    --model-inference-batch-size)
      MODEL_INFERENCE_BATCH_SIZE="$2"
      shift 2
      ;;
    --model-num-threads)
      MODEL_NUM_THREADS="$2"
      shift 2
      ;;
    --model-num-interop-threads)
      MODEL_NUM_INTEROP_THREADS="$2"
      shift 2
      ;;
    --model-device)
      MODEL_DEVICE="$2"
      shift 2
      ;;
    --tokenizers-parallelism)
      TOKENIZERS_PARALLELISM="$2"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

RUN_STAMP="${RUN_STAMP:-$(date -u +"%Y%m%dT%H%M%SZ")}"
RUN_LABEL="${LABEL:-benchmark_${RUN_STAMP}}"
RUN_DIR="${OUTPUT_DIR}/run_${RUN_STAMP}"
RUN_LOG_DIR="${RUN_DIR}/run_logs"
STDOUT_PATH="${RUN_LOG_DIR}/${RUN_LABEL}.stdout.log"
RESULT_PATH="${RUN_LOG_DIR}/${RUN_LABEL}.result.json"
mkdir -p "${RUN_DIR}" "${RUN_LOG_DIR}"

if [[ ! -f "${DATASET}" ]]; then
  cat > "${RESULT_PATH}" <<EOF
{"status":"failed","error":"dataset not found: ${DATASET}"}
EOF
  cat "${RESULT_PATH}"
  exit 1
fi

BENCHMARK_CMD=(
  python3 -m ClassifierPipeline.benchmark run
  --dataset "${DATASET}"
  --mode "${MODE}"
  --batch-size "${BATCH_SIZE}"
  --timeout "${TIMEOUT}"
  --poll-interval "${POLL_INTERVAL}"
  --operation-step "${OPERATION_STEP}"
  --output-dir "${RUN_DIR}"
  --events-path "${EVENTS_PATH}"
)

REQUESTED_MODEL_INFERENCE_BATCH_SIZE="${MODEL_INFERENCE_BATCH_SIZE}"
REQUESTED_MODEL_NUM_THREADS="${MODEL_NUM_THREADS}"
REQUESTED_MODEL_NUM_INTEROP_THREADS="${MODEL_NUM_INTEROP_THREADS}"
REQUESTED_MODEL_DEVICE="${MODEL_DEVICE}"
REQUESTED_TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM}"

unset MODEL_INFERENCE_BATCH_SIZE MODEL_NUM_THREADS MODEL_NUM_INTEROP_THREADS MODEL_DEVICE TOKENIZERS_PARALLELISM OMP_NUM_THREADS MKL_NUM_THREADS

if [[ -n "${REQUESTED_MODEL_INFERENCE_BATCH_SIZE}" ]]; then
  export MODEL_INFERENCE_BATCH_SIZE="${REQUESTED_MODEL_INFERENCE_BATCH_SIZE}"
fi
if [[ -n "${REQUESTED_MODEL_NUM_THREADS}" ]]; then
  export MODEL_NUM_THREADS="${REQUESTED_MODEL_NUM_THREADS}"
  export OMP_NUM_THREADS="${REQUESTED_MODEL_NUM_THREADS}"
  export MKL_NUM_THREADS="${REQUESTED_MODEL_NUM_THREADS}"
fi
if [[ -n "${REQUESTED_MODEL_NUM_INTEROP_THREADS}" ]]; then
  export MODEL_NUM_INTEROP_THREADS="${REQUESTED_MODEL_NUM_INTEROP_THREADS}"
fi
if [[ -n "${REQUESTED_MODEL_DEVICE}" ]]; then
  export MODEL_DEVICE="${REQUESTED_MODEL_DEVICE}"
fi
if [[ -n "${REQUESTED_TOKENIZERS_PARALLELISM}" ]]; then
  export TOKENIZERS_PARALLELISM="${REQUESTED_TOKENIZERS_PARALLELISM}"
fi

(
  cd "${APP_DIR}"
  "${BENCHMARK_CMD[@]}"
) > "${STDOUT_PATH}" 2>&1 || BENCHMARK_EXIT_CODE=$?

BENCHMARK_EXIT_CODE="${BENCHMARK_EXIT_CODE:-0}"

python3 - "${STDOUT_PATH}" "${RESULT_PATH}" "${RUN_DIR}" "${RUN_LABEL}" "${BENCHMARK_EXIT_CODE}" "${REQUESTED_MODEL_INFERENCE_BATCH_SIZE}" "${REQUESTED_MODEL_NUM_THREADS}" "${REQUESTED_MODEL_NUM_INTEROP_THREADS}" "${REQUESTED_MODEL_DEVICE}" "${REQUESTED_TOKENIZERS_PARALLELISM}" <<'PY'
import json
import shutil
import sys
from pathlib import Path

stdout_path = Path(sys.argv[1])
result_path = Path(sys.argv[2])
run_dir = Path(sys.argv[3])
run_label = sys.argv[4]
benchmark_exit_code = int(sys.argv[5])
expected_batch = sys.argv[6] or None
expected_threads = sys.argv[7] or None
expected_interop = sys.argv[8] or None
expected_device = sys.argv[9] or None
expected_tokenizers = sys.argv[10] or None

payload = {
    "status": "failed" if benchmark_exit_code else "complete",
    "benchmark_exit_code": benchmark_exit_code,
    "config_validated": benchmark_exit_code == 0,
    "run_dir": str(run_dir),
    "stdout_log": str(stdout_path),
}

stdout = stdout_path.read_text() if stdout_path.exists() else ""
decoder = json.JSONDecoder()
benchmark_result = None
for index, char in enumerate(stdout):
    if char != "{":
        continue
    try:
        candidate, _ = decoder.raw_decode(stdout[index:])
    except json.JSONDecodeError:
        continue
    if isinstance(candidate, dict):
        benchmark_result = candidate

if benchmark_result is None:
    payload["status"] = "invalid" if benchmark_exit_code == 0 else payload["status"]
    payload["error"] = "failed to find final benchmark JSON object in stdout"
else:
    payload["status"] = benchmark_result.get("status", payload["status"])
    payload["raw_artifact_json"] = benchmark_result.get("json")
    payload["raw_artifact_markdown"] = benchmark_result.get("markdown")

    raw_json = Path(benchmark_result.get("json", ""))
    raw_md = Path(benchmark_result.get("markdown", ""))
    if raw_json.exists():
        summary = json.loads(raw_json.read_text())
        stable_json = run_dir / f"{run_label}.json"
        stable_md = run_dir / f"{run_label}.md"
        if raw_md.exists():
            shutil.copy2(raw_md, stable_md)
            payload["artifact_markdown"] = str(stable_md)
        stable_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        payload["artifact_json"] = str(stable_json)
        payload["run_id"] = ((summary.get("run_metadata") or {}).get("run_id"))
        payload["summary_status"] = summary.get("status")
        payload["throughput"] = ((summary.get("throughput") or {}).get("overall_records_per_minute"))
        payload["load_adjusted_throughput"] = ((summary.get("throughput") or {}).get("load_adjusted_records_per_minute"))
        payload["wall_duration_s"] = ((summary.get("duration_s") or {}).get("wall_clock"))
        payload["runtime_metadata"] = summary.get("runtime_metadata")
        payload["validated_model_inference_batch_size_mean"] = (((summary.get("classifier_batch_shapes") or {}).get("model_inference_batch_size") or {}).get("mean"))
        payload["model_forward_mean_ms"] = (((summary.get("classifier_timing_ms") or {}).get("model_forward") or {}).get("mean"))
        payload["classify_task_mean_ms"] = (((summary.get("task_timing_ms") or {}).get("task_send_input_record_to_classifier") or {}).get("mean"))

        runtime = summary.get("runtime_metadata") or {}
        if expected_batch is not None and payload["validated_model_inference_batch_size_mean"] != float(expected_batch):
            payload["config_validated"] = False
            payload["status"] = "invalid"
            payload["error"] = f"expected MODEL_INFERENCE_BATCH_SIZE={expected_batch}, saw {payload['validated_model_inference_batch_size_mean']}"
        if expected_threads is not None and runtime.get("torch_num_threads") != int(expected_threads):
            payload["config_validated"] = False
            payload["status"] = "invalid"
            payload["error"] = f"expected torch_num_threads={expected_threads}, saw {runtime.get('torch_num_threads')}"
        if expected_interop is not None and runtime.get("torch_num_interop_threads") != int(expected_interop):
            payload["config_validated"] = False
            payload["status"] = "invalid"
            payload["error"] = f"expected torch_num_interop_threads={expected_interop}, saw {runtime.get('torch_num_interop_threads')}"
        if expected_device is not None and str(runtime.get("device")) != str(expected_device):
            payload["config_validated"] = False
            payload["status"] = "invalid"
            payload["error"] = f"expected device={expected_device}, saw {runtime.get('device')}"
        if expected_tokenizers is not None and str(runtime.get("tokenizer_parallelism")) != str(expected_tokenizers).lower():
            payload["config_validated"] = False
            payload["status"] = "invalid"
            payload["error"] = f"expected tokenizer_parallelism={expected_tokenizers}, saw {runtime.get('tokenizer_parallelism')}"
        if expected_threads is not None:
            omp = runtime.get("omp_num_threads")
            mkl = runtime.get("mkl_num_threads")
            if omp is not None and str(omp) != str(expected_threads):
                payload["config_validated"] = False
                payload["status"] = "invalid"
                payload["error"] = f"expected omp_num_threads={expected_threads}, saw {omp}"
            if mkl is not None and str(mkl) != str(expected_threads):
                payload["config_validated"] = False
                payload["status"] = "invalid"
                payload["error"] = f"expected mkl_num_threads={expected_threads}, saw {mkl}"
    elif benchmark_exit_code == 0:
        payload["status"] = "invalid"
        payload["config_validated"] = False
        payload["error"] = f"benchmark JSON artifact not found: {benchmark_result.get('json')}"

if benchmark_exit_code and "error" not in payload:
    payload["error"] = f"benchmark exited with code {benchmark_exit_code}"

result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload))
PY
