#!/usr/bin/env bash

set -euo pipefail

OUTPUT="${OUTPUT:-/hy-tmp/veriseek/outputs/veriseek_rl_only_scifact_2a800}"
PID_FILE="${PID_FILE:-${OUTPUT}/pid}"
LOG_FILE="${LOG_FILE:-${OUTPUT}/nohup.log}"
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"
STALL_LIMIT="${STALL_LIMIT:-45}"
MIN_FREE_GB="${MIN_FREE_GB:-10}"
ALERT_FILE="${ALERT_FILE:-${OUTPUT}/monitor_alert.txt}"
STATE_FILE="${STATE_FILE:-${OUTPUT}/monitor_state.tsv}"
MONITOR_LOG="${MONITOR_LOG:-${OUTPUT}/monitor.log}"

mkdir -p "${OUTPUT}"

stop_training() {
    local reason="$1"
    local ts
    ts="$(date -Is)"
    {
        echo "${ts} ${reason}"
        echo "last_step=${last_step:-unknown}"
        echo "log=${LOG_FILE}"
    } > "${ALERT_FILE}"
    echo "${ts} STOP ${reason}" >> "${MONITOR_LOG}"

    if [[ -f "${PID_FILE}" ]]; then
        local pid
        pid="$(cat "${PID_FILE}" || true)"
        if [[ -n "${pid}" ]]; then
            pkill -TERM -P "${pid}" 2>/dev/null || true
            kill -TERM "${pid}" 2>/dev/null || true
            sleep 20
            pkill -KILL -P "${pid}" 2>/dev/null || true
            kill -KILL "${pid}" 2>/dev/null || true
        fi
    fi

    pkill -f "verl.trainer.main_ppo" 2>/dev/null || true
    exit 1
}

read_step() {
    if [[ ! -f "${LOG_FILE}" ]]; then
        echo 0
        return
    fi
    grep -ao "training/global_step:[0-9]*" "${LOG_FILE}" \
        | tail -n 1 \
        | sed 's/.*://' \
        || echo 0
}

check_error_patterns() {
    [[ -f "${LOG_FILE}" ]] || return 0
    grep -aE "CUDA out of memory|Error executing job|Traceback|ValueError|RuntimeError|Killed|No space left on device|AssertionError" "${LOG_FILE}" \
        | grep -av "DataLoader worker.*killed.*signal: Killed" \
        | tail -n 1
}

last_step="$(read_step)"
unchanged=0
echo "$(date -Is) START step=${last_step}" >> "${MONITOR_LOG}"

while true; do
    now="$(date -Is)"

    if [[ -f "${PID_FILE}" ]]; then
        pid="$(cat "${PID_FILE}" || true)"
        if [[ -n "${pid}" ]] && ! ps -p "${pid}" >/dev/null 2>&1; then
            latest_step="$(read_step)"
            if [[ "${latest_step}" -ge 400 ]] || grep -aq "Training Progress: 100%" "${LOG_FILE}" 2>/dev/null; then
                echo "${now} COMPLETE step=${latest_step}" >> "${MONITOR_LOG}"
                exit 0
            fi
            stop_training "training process exited early at step=${latest_step}"
        fi
    fi

    if err="$(check_error_patterns)" && [[ -n "${err}" ]]; then
        stop_training "matched error pattern: ${err}"
    fi

    free_gb="$(df -BG /hy-tmp | awk 'NR==2 {print $4}' | tr -dc '0-9')"
    if [[ -n "${free_gb}" && "${free_gb}" -lt "${MIN_FREE_GB}" ]]; then
        stop_training "low disk space: ${free_gb}GB free"
    fi

    current_step="$(read_step)"
    if [[ "${current_step}" == "${last_step}" ]]; then
        unchanged=$((unchanged + 1))
    else
        unchanged=0
        last_step="${current_step}"
    fi

    echo -e "${now}\tstep=${current_step}\tunchanged=${unchanged}\tfree_gb=${free_gb}" >> "${STATE_FILE}"

    if [[ "${current_step}" -gt 0 && "${unchanged}" -ge "${STALL_LIMIT}" ]]; then
        stop_training "training step stalled for $((CHECK_INTERVAL * STALL_LIMIT / 60)) minutes at step=${current_step}"
    fi

    sleep "${CHECK_INTERVAL}"
done
