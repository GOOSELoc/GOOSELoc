#!/usr/bin/env bash
# Fetch the upstream Boosting3DoFAccuracy code (Shi et al., ICCV 2023) into
# ./upstream at the commit used for the GOOSELoc paper.
#
# The upstream sources are used unmodified: the few GOOSELoc accommodations
# (aerial resolution, camera height, a CPU-safe helper) are applied at import
# time by model.py, so no source patch is needed.
#
#   bash setup.sh                      # from any directory
#   SHI_UPSTREAM_URL=<mirror> bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"

URL=${SHI_UPSTREAM_URL:-https://github.com/YujiaoShi/Boosting3DoFAccuracy.git}
COMMIT=2dc4e93c64cda8c2ea2841cb58394ddbe16a8aaa

if [ ! -d upstream/.git ]; then
    git init -q upstream
    git -C upstream remote add origin "$URL"
    git -C upstream fetch -q --depth 1 origin "$COMMIT"
    git -C upstream checkout -q --detach FETCH_HEAD
fi

head=$(git -C upstream rev-parse HEAD)
if [ "$head" != "$COMMIT" ]; then
    echo "error: upstream/ is at $head, expected $COMMIT (remove upstream/ and rerun)" >&2
    exit 1
fi
if [ -n "$(git -C upstream status --porcelain --untracked-files=no)" ]; then
    echo "error: upstream/ has local modifications; expected a clean checkout" >&2
    exit 1
fi
echo "upstream ready: $(pwd)/upstream @ ${COMMIT:0:7}"
