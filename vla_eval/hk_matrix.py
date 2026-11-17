"""Explicit H/K experiment cells; unavailable actions are never padded or repeated."""
from __future__ import annotations


def matrix_cells(horizons, execute, native_horizon):
    if not horizons or not execute or len(set(horizons))!=len(horizons) or len(set(execute))!=len(execute):
        raise ValueError('H and K axes must be nonempty and contain no duplicates')
    if any(type(v) is not int or v<1 for v in [*horizons,*execute]):
        raise ValueError('H and K must be positive integers')
    if any(h>native_horizon for h in horizons):
        raise ValueError('This checkpoint does not support a requested generation horizon')
    return [{'H':h,'K':k,'applicable':k<=h,
             'reason':None if k<=h else 'K > H: insufficient actions in one generated chunk'}
            for h in horizons for k in execute]
