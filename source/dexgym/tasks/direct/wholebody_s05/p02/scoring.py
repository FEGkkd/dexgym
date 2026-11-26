"""Independent score target semantics shared by later controlled comparisons."""
import numpy as np


def position_errors(mapped, feasible, measured=None):
    """Use immutable target vectors; measured geometry is an optional scorer input."""
    mapped,feasible=map(np.asarray,(mapped,feasible))
    if mapped.shape!=feasible.shape or mapped.shape[-1:]!=(3,) or not (
            np.isfinite(mapped).all() and np.isfinite(feasible).all()):
        raise ValueError('Expected finite shape-aligned labelled Cartesian targets')
    projection=feasible-mapped
    if measured is None:
        return {'projection_error':projection}
    q=np.asarray(measured)
    if q.shape!=mapped.shape or not np.isfinite(q).all():
        raise ValueError('Measured geometry must match the target scope and shape')
    return {'projection_error':projection,'execution_error':q-feasible,'end_to_end_error':q-mapped}
