"""Per-strategy channel-index maps for weatherbench_128_v* targets.

Each map points from a human-readable variable name (e.g. ``z500``) to the
position of that channel in the model's output tensor. Two layouts exist:

  * ``BASEMODEL_INDEX_MAP`` — used by ``SimpleStep`` (PredFormer / SimVP on the
    v4 raw-memmap pipeline).
  * ``MULTIOUT_INDEX_MAP`` — used by every strategy derived from
    LitModels/basemodel_no_history (mutiout, multiout_double, mutiout_imvp,
    mutiout_predrnn).

Both now cover **all 69 channels** so per-variable val RMSE is observable for
every channel (Pareto monitoring), not just a 13-channel surrogate subset.

Channel layout (identical for both maps — the v4 memmap is the 69-filtered
ERA5 1.4° artefact; see ``tools.check_physics_common.CHANNEL_RANGES``)::

    0..3   surface : t2, u10, v10, tp
    4..16  z  @ 13 pressure levels (ascending hPa)
    17..29 t  @ 13 levels
    30..42 r  @ 13 levels
    43..55 u  @ 13 levels
    56..68 v  @ 13 levels
    levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

Note: the previous hand-written ``BASEMODEL_INDEX_MAP`` (13 entries) carried
*legacy LitModels* indices that were wrong for the v4 layout — e.g. ``t500``
pointed at channel 20 (= t@200 hPa) instead of 24, ``u500`` at 32 (= r@150 hPa)
instead of 50. Only ``z500`` (11) was coincidentally correct. The maps below
are now derived from the canonical layout above, so the labels match the
physics they measure.
"""

PRESSURE_LEVELS_HPA: tuple[int, ...] = (
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    850,
    925,
    1000,
)

# (var_name, channel index of that var's first/lowest-index level)
_SURFACE: tuple[tuple[str, int], ...] = (
    ("t2", 0),
    ("u10", 1),
    ("v10", 2),
    ("tp", 3),
)
_UPPER_BASES: tuple[tuple[str, int], ...] = (
    ("z", 4),
    ("t", 17),
    ("r", 30),
    ("u", 43),
    ("v", 56),
)


def _build_channel_index_map() -> dict[str, int]:
    """Construct the full 69-channel ``name -> index`` map from the layout."""
    index_map: dict[str, int] = {name: idx for name, idx in _SURFACE}
    for var, base in _UPPER_BASES:
        for level_offset, level_hpa in enumerate(PRESSURE_LEVELS_HPA):
            index_map[f"{var}{level_hpa}"] = base + level_offset
    return index_map


BASEMODEL_INDEX_MAP: dict[str, int] = _build_channel_index_map()

# MULTIOUT strategies share the same v4 layout; the previous hand-written
# MULTIOUT_INDEX_MAP already used these (correct) algorithmic indices.
MULTIOUT_INDEX_MAP: dict[str, int] = _build_channel_index_map()


VIS_VARS: tuple[str, ...] = ("u50", "v50", "z500", "u500", "v500")

# Default validation-channel curves. These are exact channels present in the
# 69-channel dataloader layout; intermediate levels such as ``t450`` are not
# represented by WeatherBench 1.40625deg and should not be logged as direct
# channel metrics without interpolation.
DEFAULT_VALIDATION_CHANNELS: tuple[str, ...] = tuple(BASEMODEL_INDEX_MAP)
