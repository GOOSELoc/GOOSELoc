#!/usr/bin/env bash
# Fetch the upstream HC-Net code (Wang et al., NeurIPS 2023) into ./upstream at
# the commit used for the GOOSELoc paper.
#
# HC-Net is published without a license, so this repository ships none of its
# code: the clone stays on your machine, is used unmodified and is imported at
# run time by data.py / predict.py. No source patch is needed.
#
#   bash setup.sh                      # from any directory
#   HCNET_UPSTREAM_URL=<mirror> bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"

URL=${HCNET_UPSTREAM_URL:-https://github.com/xlwangDev/HC-Net.git}
COMMIT=62329ddb05e721711ae81615a39cb48c2448ab85

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
