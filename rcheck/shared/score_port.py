"""``ScoreInteractionPrograms`` in the port for an S1 run's kept programs, for
datasets whose full-gene matrix R cannot hold (IBD: ~24 GB in R).

Equal to `scriabin.score_programs` on the dataset with complex pseudo-genes
(`inputs.with_complex_pseudogenes`), without building that matrix: genes are
read from the sparse expression, complexes from a separate dense block. The
parity gate's scores stage checks the port against R (exact with R's control
genes; within the seed-to-seed spread with its own draws).

Draws every program's control genes first (the RNG never touches expression),
then computes row means as one sparse product per batch of programs,
checkpointed per batch (scores_port/<side>/<k>.npy), so an interruption
resumes with the same random stream. Writes
scores_port/scores_<side>.npy (cells x kept programs, float32), cells.txt and
programs.txt.

Usage: uv run python rcheck/shared/score_port.py RUN_DIR --dataset ibd [--kept FILE]
(FILE: kept.json, default RUN_DIR/kept.json, else
RUN_DIR/significance_batched/kept.json)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import dataset_inputs

from cccrise.comparators import scriabin
from cccrise.comparators.inputs import COMPLEX_SEP, entity_expression

SIDES = ("ligands", "receptors")
ROW_BLOCK = 50_000


class Features:
    """Column means and column-block means of expression plus complex
    pseudo-genes, without concatenating them."""

    def __init__(self, X, genes: pd.Index, lr: pd.DataFrame):
        entities = pd.unique(pd.concat([lr["ligand"], lr["receptor"]]))
        self.complexes = [e for e in entities if COMPLEX_SEP in e and e not in genes]
        self.X, self.genes = X, genes
        self.C = entity_expression(X, genes, self.complexes, "min")
        self.names = genes.append(pd.Index(self.complexes))
        # float64 accumulation, as on the (upcast) pseudo-gene matrix; sparse
        # views ignore mean(dtype=), so cast row blocks (no whole-matrix copy)
        gene_means = (
            sum(
                np.asarray(X[i : i + ROW_BLOCK].astype(np.float64).sum(axis=0)).ravel()
                for i in range(0, X.shape[0], ROW_BLOCK)
            )
            / X.shape[0]
        )
        self.means = pd.Series(
            np.concatenate([gene_means, self.C.mean(axis=0)]), index=self.names
        )

    def row_mean(self, features) -> np.ndarray:
        """Per cell, the mean over `features` (as `scriabin._column_means`)."""
        pos = self.names.get_indexer(list(features))
        g, c = pos[pos < len(self.genes)], pos[pos >= len(self.genes)] - len(self.genes)
        total = np.zeros(self.X.shape[0])
        if len(g):
            sub = self.X[:, g]
            total += np.asarray(sub.astype(np.float64).sum(axis=1)).ravel()
        if len(c):
            total += self.C[:, c].sum(axis=1)
        return total / len(pos)


def _weights(feat: Features, groups: list[list[str]]) -> tuple[np.ndarray, np.ndarray]:
    """Column-averaging weights (genes x groups, complexes x groups): 1/len on
    each group's features, so X @ Wg + C @ Wc is every group's row mean."""
    wg = np.zeros((len(feat.genes), len(groups)))
    wc = np.zeros((len(feat.complexes), len(groups)))
    for j, group in enumerate(groups):
        pos = feat.names.get_indexer(list(group))
        g, c = pos[pos < len(feat.genes)], pos[pos >= len(feat.genes)] - len(feat.genes)
        wg[g, j] = 1.0 / len(pos)
        wc[c, j] = 1.0 / len(pos)
    return wg, wc


def score_side(
    feat: Features, programs: dict, side: int, out: Path, seed: int = 1, batch: int = 32
):
    """One side's scores. All control sets are drawn first (the RNG never
    touches expression, so the stream is `score_programs`' exactly); the row
    means are then one sparse product per batch of programs, checkpointed
    per batch (one .npy per program)."""
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    names = list(programs)
    sets = []
    for name in names:
        feats = [
            x
            for x in scriabin.program_sides(programs[name])[side]
            if x in feat.means.index
        ]
        sets.append((feats, scriabin.control_genes(feat.means, feats, rng)))
    for start in range(0, len(names), batch):
        ks = [
            k
            for k in range(start, min(start + batch, len(names)))
            if not (out / f"{k}.npy").exists()
        ]
        if not ks:
            continue
        wg, wc = _weights(feat, [sets[k][0] for k in ks] + [sets[k][1] for k in ks])
        means = np.asarray(feat.X @ wg) + feat.C @ wc
        for i, k in enumerate(ks):
            np.save(out / f"{k}.partial.npy", means[:, i] - means[:, len(ks) + i])
            (out / f"{k}.partial.npy").replace(out / f"{k}.npy")
    return np.column_stack([np.load(out / f"{k}.npy") for k in range(len(names))])


def main(run: Path, dataset: str, kept_file: Path | None) -> None:
    kept_file = kept_file or next(
        p
        for p in (run / "kept.json", run / "significance_batched/kept.json")
        if p.exists()
    )
    kept = {
        k: v if isinstance(v, list) else [v]
        for k, v in json.loads(kept_file.read_text()).items()
    }
    inp = dataset_inputs(dataset)
    feat = Features(inp.X, inp.genes, inp.lr)
    out = run / "scores_port"
    for i, side in enumerate(SIDES):
        scores = score_side(feat, kept, i, out / side)
        np.save(out / f"scores_{side}.npy", scores.astype(np.float32))
        print(
            f"{side}: {scores.shape[1]} programs x {scores.shape[0]} cells", flush=True
        )
    (out / "cells.txt").write_text("\n".join(inp.obs.index) + "\n")
    (out / "programs.txt").write_text("\n".join(kept) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run", type=Path)
    p.add_argument("--dataset", required=True, choices=["alad", "ibd"])
    p.add_argument("--kept", type=Path)
    a = p.parse_args()
    main(a.run, a.dataset, a.kept)
