#!/usr/bin/env bash
# Swap the active prop profile used by the hw container.
#
# Profiles live in docker/hw/urdf_override/profiles/<name>/
# Each profile contains: tm12s.urdf.xacro + tm12s.srdf
#
# Usage:
#   ./scripts/set_profile.sh              # list available profiles
#   ./scripts/set_profile.sh <name>       # activate profile <name>
#
# After this runs: Ctrl+C your three ROS launches and re-run them.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTIVE_DIR="$REPO_ROOT/docker/hw/urdf_override"
PROFILE_DIR="$ACTIVE_DIR/profiles"
APPLY_SCRIPT="$REPO_ROOT/scripts/apply_prop_changes.sh"

list_profiles() {
    if [ ! -d "$PROFILE_DIR" ]; then
        echo "No profiles directory found at $PROFILE_DIR" >&2
        exit 1
    fi
    local active="(none)"
    if [ -f "$ACTIVE_DIR/.active_profile" ]; then
        active="$(cat "$ACTIVE_DIR/.active_profile")"
    fi
    echo "Active profile: $active"
    echo ""
    echo "Available profiles:"
    for p in "$PROFILE_DIR"/*/; do
        [ -d "$p" ] || continue
        local name
        name="$(basename "$p")"
        local marker="  "
        [ "$name" = "$active" ] && marker="* "
        echo "${marker}${name}"
    done
}

PROFILE="${1:-}"

if [ -z "$PROFILE" ]; then
    list_profiles
    exit 0
fi

PROFILE_XACRO="$PROFILE_DIR/$PROFILE/tm12s.urdf.xacro"
PROFILE_SRDF="$PROFILE_DIR/$PROFILE/tm12s.srdf"

if [ ! -f "$PROFILE_XACRO" ] || [ ! -f "$PROFILE_SRDF" ]; then
    echo "Error: profile '$PROFILE' is missing required files." >&2
    echo "  expected: $PROFILE_XACRO" >&2
    echo "  expected: $PROFILE_SRDF" >&2
    echo "" >&2
    list_profiles >&2
    exit 1
fi

echo "Activating profile: $PROFILE"
cp "$PROFILE_XACRO" "$ACTIVE_DIR/tm12s.urdf.xacro"
cp "$PROFILE_SRDF"  "$ACTIVE_DIR/tm12s.srdf"
echo "$PROFILE" > "$ACTIVE_DIR/.active_profile"

echo ""
echo "Regenerating flat URDF (no container restart needed -- cp preserves inode)..."
"$APPLY_SCRIPT"

echo ""
echo "Active profile is now: $PROFILE"
echo "Ctrl+C and re-run your three ROS launches to see the change (MoveIt caches the URDF at launch)."
