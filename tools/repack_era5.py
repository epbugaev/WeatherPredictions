"""Pack ERA5 netCDF tree into a single float32 memmap for fast Dataset access.

The output replicates the (channel, lat, lon) layout used by
``Data.weatherbench_128_v3.WeatherBench128.custom_np_load`` and then
pre-applies the channel filter (``variables_list``: 110->69 channels) and the
spatial cut. ``WeatherBench128Memmap.__getitem__`` becomes a slice +
normalisation, without per-sample netCDF parsing.

Output files:
    ``<prefix>.dat``       — contiguous C-order binary, shape
        ``(T, 69, H_cut, W_cut)``, dtype float32.
    ``<prefix>.meta.json`` — shape, dtype, years, hours_per_year, cut,
        variables_list.

Usage:
    python tools/repack_era5.py \\
        --src /home/fratnikov/weather_bench/1.40625deg \\
        --dst /home/fa.buzaev/era5_memmap \\
        --prefix predformer_usa_2000_2004 \\
        --start-year 2000 --end-year 2004 \\
        --cut 75 107 164 228
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
import time
from collections.abc import Sequence

import h5netcdf
import numpy as np

# Channel indices kept in the packed memmap. Mirrors
# ``Data.weatherbench_128_v3.WeatherBench128.variables_list`` so the on-disk
# layout matches the in-memory layout used by every downstream model.
VARIABLES_LIST = [
    0, 1, 2, 4,
    6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
    58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
    71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
]

# (long name on disk, short name inside the netCDF file, number of levels).
# Order and channel widths match ``custom_np_load``'s ``match_set`` / ``ids``.
VARIABLE_ORDER = [
    ("2m_temperature", "t2m", 1),
    ("10m_u_component_of_wind", "u10", 1),
    ("10m_v_component_of_wind", "v10", 1),
    ("total_cloud_cover", "tcc", 1),
    ("total_precipitation", "tp", 1),
    ("toa_incident_solar_radiation", "tisr", 1),
    ("geopotential", "z", 13),
    ("temperature", "t", 13),
    ("specific_humidity", "q", 13),
    ("relative_humidity", "r", 13),
    ("u_component_of_wind", "u", 13),
    ("v_component_of_wind", "v", 13),
    ("vorticity", "vo", 13),
    ("potential_vorticity", "pv", 13),
]
# Channel offsets within the 110-wide raw layout.
IDS = [0, 1, 2, 3, 4, 5, 6, 19, 32, 45, 58, 71, 84, 97, 110]
TOTAL_RAW_CHANNELS = 110


def hours_in_year(year: int) -> int:
    """Return 8784 for leap years, 8760 otherwise."""
    return 8784 if calendar.isleap(year) else 8760


def repack(
    src: str,
    dst_dat: str,
    dst_meta: str,
    start_year: int,
    end_year: int,
    cut: Sequence[int],
) -> None:
    """Pack the configured year range into a single memmap.

    Args:
        src: Root dir holding ``<variable>/<variable>_<year>_1.40625deg.nc``.
        dst_dat: Output path for the binary memmap file.
        dst_meta: Output path for the metadata JSON.
        start_year: First (inclusive) year to pack.
        end_year: Last (inclusive) year to pack.
        cut: Four ints ``[lat0, lat1, lon0, lon1]`` slicing the global grid.
    """
    lat0, lat1, lon0, lon1 = cut
    H = lat1 - lat0
    W = lon1 - lon0
    C = len(VARIABLES_LIST)
    years = list(range(start_year, end_year + 1))
    hours_per_year = [hours_in_year(y) for y in years]
    T = sum(hours_per_year)
    size_gb = T * C * H * W * 4 / 1e9
    print(f"[repack] T={T} hours, C={C} channels, H*W={H}x{W} -> {size_gb:.2f} GB")

    os.makedirs(os.path.dirname(dst_dat) or ".", exist_ok=True)
    arr = np.memmap(dst_dat, dtype=np.float32, mode="w+", shape=(T, C, H, W))

    sel = np.asarray(VARIABLES_LIST, dtype=np.int64)
    row_offset = 0
    for y, n_hours in zip(years, hours_per_year, strict=True):
        year_start = time.perf_counter()
        raw = np.empty((n_hours, TOTAL_RAW_CHANNELS, H, W), dtype=np.float32)
        for var_idx, (var_name, short, levels) in enumerate(VARIABLE_ORDER):
            path = os.path.join(src, var_name, f"{var_name}_{y}_1.40625deg.nc")
            with h5netcdf.File(path, "r") as f:
                dat = f[short]
                actual = dat.shape[0]
                n_read = min(actual, n_hours)
                if actual < n_hours:
                    print(
                        f"[repack] WARNING: {var_name} {y} has {actual} hours, "
                        f"expected {n_hours}; the tail will stay zero"
                    )
                if levels == 1:
                    raw[:n_read, IDS[var_idx], :, :] = dat[
                        :n_read, lat0:lat1, lon0:lon1
                    ]
                else:
                    raw[:n_read, IDS[var_idx]:IDS[var_idx + 1], :, :] = dat[
                        :n_read, 0:13, lat0:lat1, lon0:lon1
                    ]
        arr[row_offset:row_offset + n_hours] = raw[:, sel, :, :]
        row_offset += n_hours
        print(
            f"[repack] {y} written ({n_hours} rows, total {row_offset}/{T}) "
            f"in {time.perf_counter() - year_start:.1f}s"
        )

    arr.flush()
    del arr

    meta = {
        "shape": [T, C, H, W],
        "dtype": "float32",
        "cut": list(cut),
        "start_year": start_year,
        "end_year": end_year,
        "years": years,
        "hours_per_year": hours_per_year,
        "variables_list": VARIABLES_LIST,
    }
    with open(dst_meta, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[repack] done: {dst_dat} ({os.path.getsize(dst_dat) / 1e9:.2f} GB)")
    print(f"[repack] meta: {dst_meta}")


def main() -> None:
    """CLI entry point — see module docstring for examples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="/home/fratnikov/weather_bench/1.40625deg")
    parser.add_argument("--dst", default="/home/fa.buzaev/era5_memmap")
    parser.add_argument("--prefix", default="predformer_usa_2000_2004")
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2004)
    parser.add_argument("--cut", type=int, nargs=4, default=[75, 107, 164, 228])
    args = parser.parse_args()

    dst_dat = os.path.join(args.dst, args.prefix + ".dat")
    dst_meta = os.path.join(args.dst, args.prefix + ".meta.json")
    repack(args.src, dst_dat, dst_meta, args.start_year, args.end_year, args.cut)


if __name__ == "__main__":
    main()
