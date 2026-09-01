"""
Runner scripts: take data + user inputs, run a CCC-RISE-related computation,
and persist the result (plus a YAML provenance sidecar) for consumption by a
downstream runner or a figure module.

Nothing in here is imported by the library or the figure modules -- runners are
the *producers*, figures are the *consumers*. Each module is
runnable as a script:

    uv run ccc-rise-run      --dataset alad --rise-rank 25 --cp-rank 14
    uv run ccc-rise-riserank --dataset alad --ranks 5:41:5
    uv run ccc-rise-cpdrank  --dataset alad --rise-rank 25 --ranks 1:26

or equivalently ``uv run python -m cellcommunicationpf2.runners.<module>``.
"""
