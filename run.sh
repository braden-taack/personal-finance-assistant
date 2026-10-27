#!/usr/bin/env bash
# Start / stop the Finance Tracker app.
#
#   ./run.sh                    start in the background (or just open it if already running) and open the browser
#   ./run.sh stop|status|restart|logs
#   ./run.sh install-launcher   add "Finance Tracker" to the desktop application menu
#
# The app listens on localhost only (see .streamlit/config.toml). PORT overrides the port (default 8501).
set -euo pipefail

DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
PORT="${PORT:-8501}"
URL="http://localhost:${PORT}"
LOG="$DIR/data/streamlit.log"
LMS="$HOME/.lmstudio/bin/lms"
cd "$DIR"

say() {   # print, and show a desktop notification when there's no terminal (launched from the menu)
    echo "$1"
    if [ ! -t 1 ] && command -v notify-send >/dev/null; then notify-send "Finance Tracker" "$1"; fi
}

healthy() { curl -fs -o /dev/null --max-time 2 "$URL/_stcore/health"; }

app_pids() {   # streamlit processes started from this project folder
    for p in $(pgrep -f "streamlit run app.py" || true); do
        [ "$(readlink "/proc/$p/cwd" 2>/dev/null)" = "$DIR" ] && echo "$p"
    done
}

ensure_venv() {
    if [ ! -x .venv/bin/streamlit ]; then
        say "First run: creating .venv and installing requirements (a few minutes)…"
        python3 -m venv .venv
        .venv/bin/pip install -q -r requirements.txt
    fi
}

ensure_llm() {   # optional: the app works without it (uncertain rows just go to the review queue)
    if ! curl -fs -o /dev/null --max-time 2 http://127.0.0.1:1234/v1/models && [ -x "$LMS" ]; then
        echo "Starting the LM Studio server…"
        "$LMS" server start >/dev/null 2>&1 || echo "Couldn't start LM Studio; continuing without the LLM."
    fi
}

start() {
    if healthy; then
        xdg-open "$URL" >/dev/null 2>&1 &
        echo "Already running at $URL"
        return
    fi
    ensure_venv
    ensure_llm
    mkdir -p data
    nohup .venv/bin/streamlit run app.py --server.port "$PORT" >"$LOG" 2>&1 &
    for _ in $(seq 1 60); do
        if healthy; then
            xdg-open "$URL" >/dev/null 2>&1 &
            echo "Running at $URL (logs: data/streamlit.log; stop with ./run.sh stop)"
            return
        fi
        sleep 1
    done
    say "The app didn't start within a minute. See data/streamlit.log."
    exit 1
}

stop() {
    local pids
    pids="$(app_pids)"
    if [ -z "$pids" ]; then echo "Not running."; return; fi
    kill $pids
    for _ in $(seq 1 10); do [ -z "$(app_pids)" ] && break; sleep 1; done
    say "Finance Tracker stopped."
}

install_launcher() {
    local dest="$HOME/.local/share/applications/finance-tracker.desktop"
    mkdir -p "$(dirname "$dest")"
    cat >"$dest" <<EOF
[Desktop Entry]
Type=Application
Name=Finance Tracker
Comment=Personal finance dashboard (runs locally)
Exec="$DIR/run.sh"
Icon=accessories-calculator
Terminal=false
Categories=Office;Finance;
Actions=stop;

[Desktop Action stop]
Name=Stop Finance Tracker
Exec="$DIR/run.sh" stop
EOF
    command -v update-desktop-database >/dev/null && update-desktop-database "$(dirname "$dest")" >/dev/null 2>&1 || true
    echo "Installed $dest — look for \"Finance Tracker\" in the application menu."
}

case "${1:-start}" in
    start) start ;;
    stop) stop ;;
    restart) stop; start ;;
    status)
        if healthy; then echo "Running at $URL (pid $(app_pids | tr '\n' ' '))"; else echo "Not running."; fi ;;
    logs) tail -n 50 -f "$LOG" ;;
    install-launcher) install_launcher ;;
    *) echo "Usage: ./run.sh [start|stop|restart|status|logs|install-launcher]"; exit 2 ;;
esac
