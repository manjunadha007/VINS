#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  HyperFusion testing arena -- record, replay and score scenario takes
#
#    ./arena.sh list              scenarios, takes, references, latest scores
#    ./arena.sh record s1         guided recording of scenario s1 (+ 3DE plate)
#    ./arena.sh info s1           latest take of s1: marks, coverage, warnings
#    ./arena.sh export s1         rewrite the 3DE plate of the latest s1 take
#    ./arena.sh test s1 [--show]  replay the latest s1 take and score it
#    ./arena.sh score s1 --live   score what the dashboard did live
#
#  Recording needs the pipeline up (./launch_isaac.sh); it only subscribes.
#  Replays run in their own ROS domain beside the live pipeline and never
#  send FreeD to Unreal. Scenarios and pass criteria: arena_scenarios.yaml.
#
#  ~/workspaces/isaac_ros-dev/arena/   (also linked as ./arena in this repo)
#     inputs/<take>/        recordings + the 3DE plate
#     reference/<take>.abc  your 3DEqualizer cameras, named after the take
#     runs/<take>/<run>/    replays and their scores
# ═══════════════════════════════════════════════════════════════════
set -u

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WS="$HOME/workspaces/isaac_ros-dev"
ARENA="$WS/arena"
CONTAINER="isaac_ros_dev-x86_64-container"

if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "❌ Container '${CONTAINER}' is not running. Start the pipeline with ./launch_isaac.sh first."
    exit 1
fi

mkdir -p "$ARENA/inputs" "$ARENA/reference" "$ARENA/runs" || exit 1
# Visible in the project (Antigravity's file tree) but never committed: takes
# are ~9 GB a minute.
if [ ! -e "$DIR/arena" ]; then
    ln -s "$ARENA" "$DIR/arena"
fi
if ! grep -qx "arena" "$DIR/.gitignore" 2>/dev/null; then
    printf '\n# Testing arena data (symlink to ~/workspaces/isaac_ros-dev/arena)\narena\n' >> "$DIR/.gitignore"
fi

# The code under test, staged apart from the live deployment in $WS/src: a
# replay runs today's repo while the live dashboard keeps the code it was
# launched with. (The live deployment is still launch_isaac.sh's job.)
CODE="$ARENA/code"
rm -rf "$CODE" && mkdir -p "$CODE" || exit 1
cp "$DIR"/*.py "$DIR"/*.yaml "$CODE/" || exit 1

# 3DE cameras: Blender (host only) reads each .abc at its plate's frame rate
# into inputs/<take>/reference.json for the scorer. Only new or changed ones.
if [ -n "$(ls "$ARENA/reference/"*.abc 2>/dev/null)" ]; then
    for abc in "$ARENA/reference/"*.abc; do
        name=$(basename "$abc" .abc)
        plate="$ARENA/inputs/$name/plate.json"
        out="$ARENA/inputs/$name/reference.json"
        if [ ! -f "$plate" ]; then
            echo "⚠️  reference/$name.abc has no matching take (inputs/$name/plate.json); name it after the take"
            continue
        fi
        if [ ! -f "$out" ] || [ "$abc" -nt "$out" ]; then
            if ! command -v blender >/dev/null; then
                echo "⚠️  Blender is needed to read reference/$name.abc (snap install blender --classic)"
                continue
            fi
            echo "Reading reference/$name.abc with Blender..."
            blender -b --python "$DIR/arena_abc.py" -- "$abc" "$plate" "$out" 2>&1 \
                | grep -E "^\[arena_abc\]|rror" || true
            [ -f "$out" ] || echo "⚠️  could not read reference/$name.abc"
        fi
    done
fi

# Which code a take or run used: commit, plus "+dirty" for local edits.
GIT_REV=$(git -C "$DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)
git -C "$DIR" diff --quiet HEAD 2>/dev/null || GIT_REV="$GIT_REV+dirty"

# -it with a terminal on both ends, so Enter marks steps and Ctrl-C reaches the
# recorder (which then finalises the bag) instead of only this shell.
if [ -t 0 ] && [ -t 1 ]; then TTY=(-it); else TTY=(); fi

exec docker exec "${TTY[@]}" -u admin \
    -e ARENA_GIT_REV="$GIT_REV" \
    -e DISPLAY="${DISPLAY:-}" \
    -e VP_TRACKING_CONFIG=/workspaces/isaac_ros-dev/src/tracking.yaml \
    -w /workspaces/isaac_ros-dev "$CONTAINER" /bin/bash -c \
    'source /opt/ros/humble/setup.bash && source install/setup.bash && exec python3 -u arena/code/arena.py "$@"' \
    arena "$@"
