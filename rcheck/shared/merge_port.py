"""``FindAllInteractionPrograms``' merge in the port, for runs whose module
count makes upstream's R merge impractical (IBD: ~16k modules, days per seed).

Reads the per-sample `ip_runs.R` results in RUN_DIR (``<sample>.json``),
merges them with `scriabin.merge_programs_incremental` in the sample order
upstream's merge uses (sorted, as `merge_sig.R` lists them) and writes
``merged.json`` as `merge_sig.R merge` does. R-vs-port equality of the merge
is checked on IBD subsets of 10 / 20 / 40 samples (CONCORDANCE_PLAN.md §10).

Usage: uv run python rcheck/shared/merge_port.py RUN_DIR [SIM_THRESHOLD]
(default 0.15, upstream)
"""

import json
import sys
import time
from pathlib import Path

from cccrise.comparators import scriabin


def main(run: Path, threshold: float = scriabin.SIM_THRESHOLD) -> None:
    per_sample = {
        f.stem: {
            name: mods if isinstance(mods, list) else [mods]
            for name, mods in json.loads(f.read_text())["modules"].items()
        }
        for f in sorted(run.glob("*.json"))
        if f.stem not in ("merged", "kept")
    }
    t0 = time.time()
    merged = scriabin.merge_programs_incremental(per_sample, threshold)
    partial = run / "merged.json.partial"
    partial.write_text(json.dumps(merged))
    partial.replace(run / "merged.json")
    n = sum(len(m) for m in per_sample.values())
    print(
        f"{len(merged)} merged programs from {n} modules {time.time() - t0:.0f} secs (port)"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), *(float(x) for x in sys.argv[2:3]))
