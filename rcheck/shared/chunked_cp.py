"""Chunked GPU CP for tensors whose full-size copies do not fit the card
(IBD 343 x 60 x 60 x 1,255 float32 = 6.2 GB: tensorly's ``parafac`` needs a
second 6.2 GB unfolding and runs the 16 GB RTX 5080 out of memory).

`chunked_cupy_cp()` is a context manager around a CuPy-backend ``parafac``
(e.g. `ccc_rise.fit_cp(..., device="gpu")`) that replaces only the three
places tensorly makes a full-size copy, keeping its ALS otherwise unchanged:

- the SVD start (``initialize_cp``, an unfolding per mode): computed on the
  host with tensorly's own ``initialize_cp`` (same algorithm, same NumPy
  random draws) and moved to the device;
- ``tl.norm(tensor, 2)``: summed over slices of the first mode;
- ``unfolding_dot_khatri_rao`` (MTTKRP, every iteration): accumulated over
  slices of a mode other than the one being solved (views; no copy of the
  tensor).

Only summation order changes (float32 rounding). Validated against the
standard GPU fit (ALAD) and the production IBD CP. Used by
rcheck/shared/cccrise_refits.py (``CP_CHUNKED=1``).
"""

from contextlib import contextmanager

import numpy as np
import tensorly as tl
import tensorly.decomposition._cp as tl_cp

#: Elements per tensor slice processed at once (float32: ~0.8 GB).
CHUNK_ELEMENTS = 200_000_000


def _slices(n: int, per_index: int):
    step = max(1, CHUNK_ELEMENTS // max(per_index, 1))
    for start in range(0, n, step):
        yield slice(start, min(start + step, n))


def chunked_mttkrp(tensor, cp_tensor, mode: int):
    """``unfold(tensor, mode) @ khatri_rao(factors, weights, skip=mode)``,
    summed over slices of another mode."""
    import cupy

    weights, factors = cp_tensor
    ndim = tensor.ndim
    axis = max((m for m in range(ndim) if m != mode), key=lambda m: tensor.shape[m])
    per_index = tensor.size // tensor.shape[axis]
    letters = "abcdefgh"[:ndim]
    rank = factors[0].shape[1]
    out = cupy.zeros((tensor.shape[mode], rank), dtype=tensor.dtype)
    others = [m for m in range(ndim) if m != mode]
    spec = (
        letters + "," + ",".join(f"{letters[m]}r" for m in others) + f"->{letters[mode]}r"
    )
    for s in _slices(tensor.shape[axis], per_index):
        index = [slice(None)] * ndim
        index[axis] = s
        part = [factors[m][s] if m == axis else factors[m] for m in others]
        out += cupy.einsum(spec, tensor[tuple(index)], *part)
    if weights is not None:
        out = out * weights.reshape(1, -1)
    return out


def chunked_norm(tensor, order=2, axis=None):
    """``tl.norm(tensor, 2)`` over the whole tensor, summed over slices."""
    import cupy

    total = cupy.zeros((), dtype=tensor.dtype)
    for s in _slices(tensor.shape[0], tensor.size // tensor.shape[0]):
        block = tensor[s]
        total += cupy.sum(block * block)
    return cupy.sqrt(total)


@contextmanager
def chunked_cupy_cp():
    """Patch tensorly's CP ALS for the duration (CuPy backend only)."""
    orig_mttkrp = tl_cp.unfolding_dot_khatri_rao
    orig_init = tl_cp.initialize_cp
    orig_norm = tl.norm

    def norm(tensor, order=2, axis=None):
        on_gpu = type(tensor).__module__.startswith("cupy")
        if on_gpu and order == 2 and axis is None and tensor.ndim >= 3:
            return chunked_norm(tensor)
        return orig_norm(tensor, order, axis)

    def initialize_cp(tensor, rank, init="svd", svd="truncated_svd", **kw):
        if init != "svd" or kw.get("mask") is not None:
            return orig_init(tensor, rank, init=init, svd=svd, **kw)
        host = tl.to_numpy(tensor)
        with tl.backend_context("numpy"):
            kt = orig_init(host, rank, init=init, svd=svd, **kw)
            weights, factors = kt
        return tl_cp.CPTensor(
            (None if weights is None else tl.tensor(weights, **tl.context(tensor)),
             [tl.tensor(np.asarray(f), **tl.context(tensor)) for f in factors])
        )

    tl_cp.unfolding_dot_khatri_rao = chunked_mttkrp
    tl_cp.initialize_cp = initialize_cp
    tl.norm = norm
    try:
        yield
    finally:
        tl_cp.unfolding_dot_khatri_rao = orig_mttkrp
        tl_cp.initialize_cp = orig_init
        tl.norm = orig_norm
