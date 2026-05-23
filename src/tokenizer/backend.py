"""
Backend abstraction layer for tokenizer pipeline.

Selects GPU (cuDF/cuPy/cuML) or CPU (pandas/numpy/sklearn) based on
the TOKENIZER_BACKEND environment variable.

Usage:
    TOKENIZER_BACKEND=cpu   — Apple Silicon / CPU mode (default)
    TOKENIZER_BACKEND=gpu   — NVIDIA CUDA mode (original)
"""

import os
import logging

_BACKEND = os.environ.get("TOKENIZER_BACKEND", "cpu").lower()

if _BACKEND == "gpu":
    import cudf as _df_mod
    import cupy as _array_mod
    from cuml.preprocessing import KBinsDiscretizer as _KBinsDiscretizer

    _USE_GPU = True
else:
    import pandas as _df_mod
    import numpy as _array_mod
    from sklearn.preprocessing import KBinsDiscretizer as _KBinsDiscretizer

    _USE_GPU = False

logger = logging.getLogger(__name__)


def get_backend() -> str:
    return _BACKEND


def is_gpu() -> bool:
    return _USE_GPU


DataFrame = _df_mod.DataFrame
Series = _df_mod.Series
array_lib = _array_mod
KBinsDiscretizer = _KBinsDiscretizer


def to_datetime(arg, *args, **kwargs):
    if _USE_GPU:
        import cudf
        return cudf.to_datetime(arg, *args, **kwargs)
    else:
        import pandas as pd
        return pd.to_datetime(arg, *args, **kwargs)


def concat(objs, *args, **kwargs):
    if _USE_GPU:
        import cudf
        return cudf.concat(objs, *args, **kwargs)
    else:
        import pandas as pd
        return pd.concat(objs, *args, **kwargs)


def hash_series(ser):
    """Hash string elements to int64, compatible with cudf.Series.hash_values()."""
    if _USE_GPU:
        return ser.hash_values()

    import mmh3
    def _hash_to_int64(s):
        val = mmh3.hash64(str(s), signed=True)
        return val[0] if isinstance(val, tuple) else val
    return ser.astype(str).apply(_hash_to_int64).astype("int64")


def read_csv(filepath, *args, **kwargs):
    if _USE_GPU:
        import cudf
        return cudf.read_csv(filepath, *args, **kwargs)
    else:
        import pandas as pd
        return pd.read_csv(filepath, *args, **kwargs)


def read_parquet(filepath, *args, **kwargs):
    if _USE_GPU:
        import cudf
        return cudf.read_parquet(filepath, *args, **kwargs)
    else:
        import pandas as pd
        return pd.read_parquet(filepath, *args, **kwargs)


def to_pandas(obj):
    """Convert to pandas DataFrame/Series. On CPU backend, returns as-is."""
    if hasattr(obj, "to_pandas"):
        return obj.to_pandas()
    return obj


def get_values(arr):
    """Get numpy array from cupy/numpy array or Series values."""
    if hasattr(arr, "get"):
        return arr.get()
    if hasattr(arr, "values_host"):
        return arr.values_host
    if hasattr(arr, "values"):
        return arr.values
    return arr
