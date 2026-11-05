"""Seed runs of upstream InteractionPrograms (rcheck/faithful/ip_runs.R) for
the figureR_scriabin_model datasets, as job lists for rcheck/shared/loadq.py
(one single-threaded job per seed and sample or group; done-file = its .rds):

- ``alad_authors``: ALAD with the authors' preprocessing (SCTransform +
  ALRA, or SCTransform alone; OmniPath via the stub), from
  rcheck/faithful/preprocess.R (copied to singlecell_cache/authors/alad/),
  six samples per job -> rcheck/out/authors/alad/{alra,sct}/seed<k>/;
- ``panc8_authors``: panc8 with the vignette's preprocessing
  (rcheck/authors/panc8_prep.R), groups = donors or cell types ->
  rcheck/out/authors/panc8/{donor,bytype}/seed<k>/;
- ``panc8_cccrise``: panc8 on cccRISE's inputs
  (rcheck/authors/panc8_cccrise_inputs.py; CellChatDB as IP_LR_CSV) ->
  rcheck/out/shared/panc8/{donor,bytype}/seed<k>/;
- ``panc8_paper``: the Scriabin paper's own robustness analyses
  (figureR_scriabin_reproduced), from rcheck/authors/panc8_paper_prep.R:
  12 Fig. 2d-e subsamples (panc8_paper_noise.R) and 35 ED Fig. 8f
  subsamples each of 300 and 1,000 cells (panc8_paper_r2_sweep.R) ->
  rcheck/out/authors/panc8_paper/{noise,r2_sweep}/.

Seeds 1111 (upstream's own) and 2-10. ALAD and IBD on cccRISE's inputs run
through rcheck/shared/run_s1.sh.

  uv run python rcheck/authors/jobs.py panc8_authors   # writes <run root>/jobs.tsv
  uv run python rcheck/shared/loadq.py rcheck/out/authors/panc8/jobs.tsv --name panc8
"""

import subprocess
import sys
from pathlib import Path

SEEDS = (1111, 2, 3, 4, 5, 6, 7, 8, 9, 10)


def _samples(meta: str) -> list[str]:
    out = subprocess.run(
        ["Rscript", "-e", f"cat(sort(unique(readRDS('{meta}')$sample)), sep='\\n')"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return out.split()


def _job(
    name: str,
    out: str,
    samples: list[str],
    data: str,
    meta: str,
    seed: int,
    env: str = "",
) -> str:
    cmd = (
        f"mkdir -p {out} && {env}IP_SAMPLES={','.join(samples)} Rscript rcheck/faithful/ip_runs.R "
        f"programs {data} {meta} {out} {seed} 1 1"
    )
    return f"{name}\t{out}/{samples[-1]}.rds\t{cmd}"


def alad_authors() -> tuple[str, list[str]]:
    src, root = "singlecell_cache/authors/alad", "rcheck/out/authors/alad"
    samples = _samples(f"{src}/meta.rds")
    chunks = [samples[i : i + 6] for i in range(0, len(samples), 6)]
    return root, [
        _job(
            f"{pipe}_s{seed}_c{k}",
            f"{root}/{pipe}/seed{seed}",
            ch,
            f"{src}/{pipe}_lr.rds",
            f"{src}/meta.rds",
            seed,
        )
        for pipe in ("alra", "sct")
        for seed in SEEDS
        for k, ch in enumerate(chunks)
    ]


def _panc8(src: str, root: str, data: str, env: str) -> list[str]:
    lines = []
    for framing, meta in (("donor", "meta.rds"), ("bytype", "meta_bytype.rds")):
        for seed in SEEDS:
            lines += [
                _job(
                    f"{framing}_s{seed}_{g}",
                    f"{root}/{framing}/seed{seed}",
                    [g],
                    f"{src}/{data}",
                    f"{src}/{meta}",
                    seed,
                    env,
                )
                for g in _samples(f"{src}/{meta}")
            ]
    return lines


def panc8_authors() -> tuple[str, list[str]]:
    root = "rcheck/out/authors/panc8"
    return root, _panc8("singlecell_cache/authors/panc8", root, "alra_data.rds", "")


def panc8_cccrise() -> tuple[str, list[str]]:
    src, root = "singlecell_cache/shared/panc8", "rcheck/out/shared/panc8"
    return root, _panc8(src, root, "lr_data.rds", f"IP_LR_CSV={src}/lr.csv ")


def panc8_paper() -> tuple[str, list[str]]:
    root, env = (
        "rcheck/out/authors/panc8_paper",
        "R_LIBS=singlecell_cache/authors/panc8/rlib ",
    )
    noise = [
        f"noise_{i:02d}\t{root}/noise/sub{i:02d}.rds\t{env}Rscript rcheck/authors/panc8_paper_noise.R "
        f"{i} {root}/noise"
        for i in range(1, 13)
    ]
    sweep = [
        f"r2_n{n}_{i:02d}\t{root}/r2_sweep/n{n}_sub{i:02d}.rds\t{env}Rscript "
        f"rcheck/authors/panc8_paper_r2_sweep.R {n} {i} {root}/r2_sweep"
        for n in (300, 1000)
        for i in range(1, 36)
    ]
    return root, noise + sweep


def main(which: str) -> None:
    root, lines = {
        "alad_authors": alad_authors,
        "panc8_authors": panc8_authors,
        "panc8_cccrise": panc8_cccrise,
        "panc8_paper": panc8_paper,
    }[which]()
    Path(root).mkdir(parents=True, exist_ok=True)
    Path(root, "jobs.tsv").write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} jobs -> {root}/jobs.tsv")


if __name__ == "__main__":
    main(sys.argv[1])
