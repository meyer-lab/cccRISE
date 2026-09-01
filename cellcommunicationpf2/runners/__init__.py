"""
Runner scripts: take data plus user inputs, run a CCC-RISE-related computation,
and persist the result -- plus a YAML provenance sidecar -- for a downstream
runner or a figure module to read.

Runners are the producers and are imported by nothing in the library or the
figure modules. Each is runnable as a script:

    uv run ccc-rise-run      --dataset alad --rise-rank 25 --cp-rank 14
    uv run ccc-rise-riserank --dataset alad --ranks 5:51:5
    uv run ccc-rise-cpdrank  --dataset alad --rise-rank 25 --ranks 1:26

or equivalently ``uv run python -m cellcommunicationpf2.runners.<module>``.
"""
