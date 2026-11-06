#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# Fetch the upstream CCVPE code (Xia et al., T-PAMI 2023) into ./upstream at
# the commit used for the GOOSELoc paper.
#
# The upstream sources are used unmodified: the pseudo-panorama, resizing and
# label construction live in data.py, so no source patch is needed.
#
#   bash setup.sh                      # from any directory
#   CCVPE_UPSTREAM_URL=<mirror> bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"

URL=${CCVPE_UPSTREAM_URL:-https://github.com/tudelft-iv/CCVPE.git}
COMMIT=3028aa5c1585d3c274c3a531bdce19846c9c3bd2

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
