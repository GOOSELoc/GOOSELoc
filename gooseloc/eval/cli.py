"""Command line evaluation.

    gooseloc-eval predictions.csv --split test [--json summary.json] [--per-scene errors.csv]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gooseloc.eval.benchmark import evaluate, format_summary


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("predictions", type=Path, help="CSV with scene_id,x,y[,heading_deg]")
    p.add_argument("--split", default="test", help="train, dev, test or test_buffered")
    p.add_argument("--json", type=Path, default=None, help="write the summary as JSON")
    p.add_argument("--per-scene", type=Path, default=None, help="write per-scene errors as CSV")
    p.add_argument("--allow-partial", action="store_true",
                   help="evaluate only the predicted scenes instead of failing")
    args = p.parse_args(argv)

    res = evaluate(args.predictions, args.split, allow_partial=args.allow_partial)
    print(format_summary(res["summary"]))
    if args.json:
        args.json.write_text(json.dumps(res["summary"], indent=2))
    if args.per_scene:
        res["per_scene"].to_csv(args.per_scene, index=False)


if __name__ == "__main__":
    main()
