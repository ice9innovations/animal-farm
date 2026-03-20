#!/bin/bash
# Animal Farm service manager — replaces systemd on RunPod
# Auto-discovers services by scanning for */run.sh
#
# Usage: ./farm.sh {start|stop|restart|status} [service]

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
LOG_DIR="$SCRIPT_DIR/logs"
PID_DIR="$SCRIPT_DIR/.pids"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

ACTION="$1"
TARGET="$2"

get_all_services() {
    for run_sh in "$SCRIPT_DIR"/*/run.sh; do
        [ -f "$run_sh" ] && basename "$(dirname "$run_sh")"
    done | sort
}

service_pid() {
    local name="$1"
    local dir="$SCRIPT_DIR/$name"
    local pid_file="$PID_DIR/${name}.pid"

    # Check PID file first
    if [ -f "$pid_file" ]; then
        local pid
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo "$pid"
            return 0
        else
            rm -f "$pid_file"
        fi
    fi

    # Fall back to pgrep on the service's venv python — catches manually-started instances
    local pid
    pid=$(pgrep -f "$dir/venv/bin/python" 2>/dev/null | head -1)
    if [ -n "$pid" ]; then
        echo "$pid"
        return 0
    fi

    return 1
}

start_service() {
    local name="$1"
    local dir="$SCRIPT_DIR/$name"

    if [ ! -f "$dir/run.sh" ]; then
        echo "❌ No run.sh for $name"
        return 1
    fi

    if service_pid "$name" >/dev/null; then
        echo "  $name already running (PID $(service_pid "$name"))"
        return 0
    fi

    mkdir -p "$LOG_DIR" "$PID_DIR"
    nohup bash "$dir/run.sh" >> "$LOG_DIR/${name}.log" 2>&1 &
    local pid=$!
    echo $pid > "$PID_DIR/${name}.pid"
    sleep 2
    if kill -0 "$pid" 2>/dev/null; then
        echo "✅ Started $name (PID $pid)"
    else
        rm -f "$PID_DIR/${name}.pid"
        echo "❌ $name failed to start — check logs/${name}.log"
        return 1
    fi
}

stop_service() {
    local name="$1"
    local pid
    pid=$(service_pid "$name")

    if [ -z "$pid" ]; then
        echo "❌ $name not running"
        return 1
    fi

    # Kill the process group so the python child dies with the bash wrapper
    local pgid
    pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    if [ -n "$pgid" ] && [ "$pgid" != "0" ]; then
        kill -- -"$pgid" 2>/dev/null
    else
        kill "$pid" 2>/dev/null
    fi

    # Also kill any matching venv python processes (handles untracked instances)
    local dir="$SCRIPT_DIR/$name"
    pkill -f "$dir/venv/bin/python" 2>/dev/null || true

    sleep 3

    # Force kill anything still alive
    pkill -9 -f "$dir/venv/bin/python" 2>/dev/null || true

    rm -f "$PID_DIR/${name}.pid"
    echo "✅ Stopped $name"
}

status_all() {
    echo "Service Status:"
    echo "===================="
    for name in $(get_all_services); do
        local pid
        pid=$(service_pid "$name")
        if [ -n "$pid" ]; then
            echo -e "${GREEN}✅ $name${NC} (PID: $pid)"
        else
            echo -e "${RED}❌ $name${NC} (not running)"
        fi
    done
}

start_all() {
    mkdir -p "$LOG_DIR" "$PID_DIR"
    echo "Starting all services..."
    for name in $(get_all_services); do
        start_service "$name"
    done
}

stop_all() {
    echo "Stopping all services..."
    for name in $(get_all_services); do
        stop_service "$name"
    done
}

case "$ACTION" in
    start)
        if [ -n "$TARGET" ]; then
            start_service "$TARGET"
        else
            start_all
        fi
        ;;
    stop)
        if [ -n "$TARGET" ]; then
            stop_service "$TARGET"
        else
            stop_all
        fi
        ;;
    restart)
        if [ -n "$TARGET" ]; then
            stop_service "$TARGET"
            sleep 1
            start_service "$TARGET"
        else
            stop_all
            sleep 2
            start_all
        fi
        echo ""
        status_all
        ;;
    status)
        status_all
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status} [service]"
        echo ""
        echo "Examples:"
        echo "  $0 start             # Start all services"
        echo "  $0 start BLIP2       # Start just BLIP2"
        echo "  $0 stop              # Stop all services"
        echo "  $0 stop BLIP2        # Stop just BLIP2"
        echo "  $0 restart BLIP2     # Restart just BLIP2"
        echo "  $0 status            # Show status of all services"
        exit 1
        ;;
esac
