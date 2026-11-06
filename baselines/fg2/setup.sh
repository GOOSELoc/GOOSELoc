#!/usr/bin/env bash
# Fetch upstream FG² at the pinned commit into ./upstream and apply goose.patch.
#
#   bash baselines/fg2/setup.sh
#
# Safe to re-run: an existing checkout is reset to the pinned commit (edits to
# tracked upstream files are discarded) and the patch is applied afresh.
#
# Weights that are downloaded separately:
#   * DINOv2 ViT-L/14 (frozen backbone, ~1.2 GB): fetched automatically by
#     upstream's DinoExtractor on first use (torch hub cache).
#   * The FG² authors' KITTI checkpoint: only needed to re-run fine-tuning
#     (train.py), not for inference with the released GOOSELoc checkpoints.
#     Download KITTI/model.pt from the authors' Google Drive folder linked in
#     upstream/README.md ("Download pretrained models"),
#       https://drive.google.com/drive/folders/1WUViQcX9m0PE9FePbWklisBS88sHOyMK
#     e.g. to upstream/checkpoints/KITTI/model.pt and pass it to train.py via
#     --pretrained. The file the paper fine-tuned from has sha256
#       b1d461caf81333a4aa5af61aa05f3cda4ce290a87bef92e98824daf454a2d8ca
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URL="https://github.com/vita-epfl/FG2.git"
COMMIT="a55d9d13eb029ab2cccf21cff31b728c78d443b2"
DEST="$HERE/upstream"
PATCH="$HERE/goose.patch"

if [ ! -d "$DEST/.git" ]; then
    echo ">> cloning $URL into $DEST"
    git clone --quiet "$URL" "$DEST"
fi

if ! git -C "$DEST" cat-file -e "$COMMIT^{commit}" 2>/dev/null; then
    git -C "$DEST" fetch --quiet origin
fi
echo ">> checking out $COMMIT"
git -C "$DEST" -c advice.detachedHead=false checkout --quiet --force "$COMMIT"

echo ">> applying goose.patch"
git -C "$DEST" apply "$PATCH"

echo ">> FG² ready in $DEST (commit ${COMMIT:0:7} + goose.patch)"
