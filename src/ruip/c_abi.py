"""ctypes layouts for the C17 Gaussian-mixture comparator interface.

Field order and C scalar types must match ``csrc/include/ruip/priors.h``.
Only the structures used by the published simulation runners are exposed.
"""

import ctypes as ct


class History(ct.Structure):
    """One historical estimate with per-subject observed information."""

    _fields_ = [
        ("n", ct.c_uint32),
        ("estimate", ct.c_double),
        ("unit_info", ct.c_double),
    ]


class NormalComponent(ct.Structure):
    """A Normal component returned by a C comparator prior."""

    _fields_ = [
        ("mean", ct.c_double), ("variance", ct.c_double),
        ("log_weight", ct.c_double), ("j_g", ct.c_double),
        ("m_eff", ct.c_double), ("retention_mask", ct.c_uint),
        ("tau", ct.c_double), ("robust_component", ct.c_double),
        ("borrowing_m", ct.c_double), ("source_weight", ct.c_double * 8),
    ]


class Mixture(ct.Structure):
    """Owned array of Normal components; release it through the C API."""

    _fields_ = [("items", ct.POINTER(NormalComponent)), ("count", ct.c_size_t)]
