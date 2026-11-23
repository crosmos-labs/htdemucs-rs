#!/bin/bash
# measure.sh <out file> <cmd...>: runs cmd, writing "cpu_s=... peak_rss_mb=... wall_s=..." to <out file>
# (CPU: user + sys of cmd and its children, from bash's `times`; peak RSS: VmHWM of cmd's own
# process, polled from /proc - Linux only, and only that process, not its children).
out=$1; shift
t0=$(date +%s.%N)
"$@" & pid=$!; peak=0
while kill -0 $pid 2>/dev/null; do r=$(awk '/VmHWM/{print $2}' /proc/$pid/status 2>/dev/null); [ -n "$r" ] && peak=$r; sleep 0.5; done
wait $pid; code=$?
t1=$(date +%s.%N)
sec() { echo "$1" | awk -F'[ms]' '{print $1 * 60 + $2}'; }
times > "$out.times"  # not in a pipe or $(...): a subshell would report its own (zero) child times
read -r u s < <(tail -1 "$out.times"); rm -f "$out.times"
awk -v u="$(sec "$u")" -v s="$(sec "$s")" -v p="$peak" -v a="$t0" -v b="$t1" 'BEGIN {printf "cpu_s=%.1f peak_rss_mb=%.0f wall_s=%.1f\n", u + s, p / 1024, b - a}' > "$out"
exit $code
