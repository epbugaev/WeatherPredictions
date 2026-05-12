"""Per-strategy channel-index maps for weatherbench_128_v* targets.

Each map points from a human-readable variable name (e.g. ``z500``) to the
position of that channel in the model's output tensor. The mappings come
directly from the LitModels they replace and are kept verbatim to preserve
numerical parity. Two layouts exist:

  * ``BASEMODEL_INDEX_MAP`` — used by ``SimpleStep`` (matches LitModels/mutiout_f).
  * ``MULTIOUT_INDEX_MAP`` — used by every strategy derived from
    LitModels/basemodel_no_history (mutiout, multiout_double, mutiout_imvp,
    mutiout_predrnn).
"""

BASEMODEL_INDEX_MAP: dict[str, int] = {
    "u10": 1,
    "v10": 2,
    "t2": 0,
    "z500": 11,
    "t500": 20,
    "t50": 23,
    "t1000": 19,
    "u500": 32,
    "v500": 45,
    "u50": 35,
    "v50": 48,
    "u1000": 31,
    "v1000": 44,
}


MULTIOUT_INDEX_MAP: dict[str, int] = {
    "u10": 1,
    "v10": 2,
    "t2": 0,
    "z500": 4 + 7,
    "t500": 17 + 7,
    "t50": 17 + 0,
    "t1000": 17 + 12,
    "u500": 43 + 7,
    "v500": 56 + 7,
    "u50": 43 + 0,
    "v50": 56 + 0,
    "u1000": 43 + 12,
    "v1000": 56 + 12,
}


VIS_VARS: tuple[str, ...] = ("u50", "v50", "z500", "u500", "v500")
