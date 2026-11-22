#!/usr/bin/env bash
# Run jetson/bench.sh in a tmux session, so it runs to completion even if your SSH connection
# drops or you close the laptop. Takes the same arguments as bench.sh:
#
#   jetson/bench_detached.sh <video> [bench.sh options]
#
# It attaches to the session straight away so you can type the sudo password. Then detach with
# Ctrl-b then d (or just disconnect). Later:
#   tmux attach -t hv-bench                    # watch it again
#   tail -f bench_results/*/bench.log          # or follow the log
# The session stays open after bench.sh ends, showing its exit status, until you press Enter.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v tmux >/dev/null || sudo apt-get install -y tmux
SESSION=${HV_TMUX_SESSION:-hv-bench}
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "A benchmark is already running in tmux session '$SESSION': tmux attach -t $SESSION" >&2
    exit 1
fi
CMD="$(printf '%q ' jetson/bench.sh "$@")"
tmux new-session -d -s "$SESSION" -c "$PWD" \
    "$CMD; status=\$?; echo; echo \"bench.sh finished with exit status \$status. Press Enter to close.\"; read"
echo "Started in tmux session '$SESSION'. Type the sudo password if asked, then Ctrl-b d to detach."
if [ -t 1 ]; then
    exec tmux attach -t "$SESSION"
fi
