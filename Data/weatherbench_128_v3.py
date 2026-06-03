import json
import os

import h5netcdf
import numpy as np
import pandas as pd
import torch
from torch.profiler import record_function
from torch.utils.data import Dataset

from utils.paths import weatherbench_input_root, weatherbench_npy_root


class WeatherBench128(Dataset):
    returns_normalized = True

    def __init__(
        self,
        start_time: str = "2000-01-01 00:00:00",
        end_time: str = "2000-01-05 23:00:00",
        include_target: bool = False,
        lead_time: int = 6,
        interval: int = 6,
        sample_stride: int | None = None,
        frame_interval: int = 1,
        muti_target_steps: int = 1,
        # New parameters for time sequences
        start_time_x: int = 0,
        end_time_x: int = 1,
        start_time_y: int = 0,
        end_time_y: int = 1,
        cut=None,
        num_preload=12,
        data_folder: str | None = None,
        input_folder: str | None = None,
        mean_std_path: str | None = None,
    ):

        self.variables_list = [
            0,
            1,
            2,
            4,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            83,
        ]
        # ``data_folder`` is the dir whose basename pattern (YYYY-HHHH.npy) the
        # loader parses to extract year/hour; ``input_folder`` is the parent dir
        # of per-variable netCDF subfolders actually opened by ``custom_np_load``.
        self.data_folder = data_folder or weatherbench_npy_root()
        self.input_folder = input_folder or weatherbench_input_root()
        self.mean_std_path = mean_std_path
        self.start_time = start_time
        self.end_time = end_time
        self.include_target = include_target
        self.lead_time = int(lead_time)
        # Legacy configs used `interval`. In the refactored loader it is the
        # stride between sample starts; frame spacing is controlled separately.
        self.sample_stride = int(sample_stride if sample_stride is not None else interval)
        self.interval = self.sample_stride
        self.frame_interval = int(frame_interval)
        self.muti_target_steps = int(muti_target_steps)

        if self.sample_stride <= 0:
            raise ValueError("sample_stride/interval must be a positive number of hours")
        if self.frame_interval <= 0:
            raise ValueError("frame_interval must be a positive number of hours")

        # Store the new sequence parameters
        self.start_time_x = start_time_x
        self.end_time_x = end_time_x
        self.start_time_y = start_time_y
        self.end_time_y = end_time_y

        # +1 because the ranges are now inclusive
        self.x_sequence_length = end_time_x - start_time_x + 1
        self.y_sequence_length = end_time_y - start_time_y + 1

        if self.x_sequence_length <= 0:
            raise ValueError("end_time_x must be greater than or equal to start_time_x")
        if self.y_sequence_length <= 0:
            raise ValueError("end_time_y must be greater than or equal to start_time_y")

        self.init_time_list()
        self.init_file_list()
        self.get_mean_std()
        # `x_time_ilst` is an hourly calendar. `sample_start_indices` selects
        # which hours may start a training sample; frames inside each sample are
        # still hour-by-hour unless `frame_interval` is explicitly changed.
        self.max_sequence_offset = max(
            self.end_time_x * self.frame_interval,
            self.end_time_y * self.frame_interval + self.muti_target_steps * self.lead_time,
        )
        max_start_idx = len(self.x_time_ilst) - 1 - self.max_sequence_offset
        self.length = max_start_idx // self.sample_stride + 1 if max_start_idx >= 0 else 0
        self.sample_start_indices = [
            i * self.sample_stride for i in range(self.length)
        ]

        if self.length <= 0:
            raise ValueError("Not enough time steps available for the requested sequence lengths")
        self.max_required_time = self.x_time_ilst[self.sample_start_indices[-1]] + pd.Timedelta(
            hours=self.max_sequence_offset
        )

        self.cut = cut
        if self.cut is None:
            self.lat_slice = slice(None)
            self.lon_slice = slice(None)
            self.spatial_shape = (128, 256)
        else:
            self.lat_slice = slice(self.cut[0][0], self.cut[0][1])
            self.lon_slice = slice(self.cut[1][0], self.cut[1][1])
            self.spatial_shape = (
                self.cut[0][1] - self.cut[0][0],
                self.cut[1][1] - self.cut[1][0],
            )

        self.preload = {}
        self.num_preload = num_preload

        self.hit = 0
        self.not_hit = 0

    def custom_np_load(self, file_path):
        input_folder = self.input_folder
        match_set = {
            "2m_temperature": "t2m",
            "10m_u_component_of_wind": "u10",
            "10m_v_component_of_wind": "v10",
            "total_cloud_cover": "tcc",
            "total_precipitation": "tp",
            "toa_incident_solar_radiation": "tisr",
            "geopotential": "z",
            "temperature": "t",
            "specific_humidity": "q",
            "relative_humidity": "r",
            "u_component_of_wind": "u",
            "v_component_of_wind": "v",
            "vorticity": "vo",
            "potential_vorticity": "pv",
        }

        ids = [0, 1, 2, 3, 4, 5, 6, 19, 32, 45, 58, 71, 84, 97, 110]
        year = file_path.split("/")[-1][0:4]
        hour = int(file_path.split("/")[-1][5:9])

        if (
            self.preload is not None and (year + "-" + str(hour)) in self.preload
        ):  # If already loaded, do not load again
            self.hit += 1
            return self.preload[year + "-" + str(hour)]
        else:
            self.not_hit += 1
            self.preload = {}

        match_set_files = {}
        for key in match_set:
            path = input_folder + key + "/" + key + "_" + year + "_1.40625deg.nc"
            match_set_files[key] = h5netcdf.File(path, "r")

        right_bound_hours = max(hour + 1, min(7861, hour + self.num_preload))

        height, width = self.spatial_shape
        res = np.empty([right_bound_hours - hour, 110, height, width], dtype=np.float32)

        for ind, key in enumerate(match_set):
            start_id = ids[ind]
            end_id = ids[ind + 1]

            if end_id - start_id == 1:
                res[:, start_id, :, :] = match_set_files[key][match_set[key]][
                    hour:right_bound_hours,
                    self.lat_slice,
                    self.lon_slice,
                ]
            else:
                res[:, start_id:end_id, :, :] = match_set_files[key][match_set[key]][
                    hour:right_bound_hours,
                    0:13,
                    self.lat_slice,
                    self.lon_slice,
                ]

        for cur_hour in range(hour, right_bound_hours):
            self.preload[year + "-" + str(cur_hour)] = res[cur_hour - hour, ...]

        for key in match_set:
            match_set_files[key].close()

        return self.preload[year + "-" + str(hour)]

    def init_time_list(self):
        self.x_time_ilst = pd.date_range(self.start_time, self.end_time, freq="1h")

    def idx_in_year(self, time_stamp):
        year = time_stamp.year
        first_day = pd.to_datetime(f"{year}-01-01 00:00:00")
        idx = int((time_stamp - first_day).total_seconds() / 3600)
        return idx

    def init_file_list(self):
        # your_weatherbench_data_path/1979/1979-0000.npy
        self.x_file_list = [
            os.path.join(
                self.data_folder,  # str(time_stamp.year),
                str(time_stamp.year) + f"-{self.idx_in_year(time_stamp):04d}" + ".npy",
            )
            for time_stamp in self.x_time_ilst
        ]

    def get_mean_std(self):
        mean_std_path = self.mean_std_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "example_data",
            "mean_std.json",
        )

        if mean_std_path.endswith(".npy"):
            mean_std = np.load(mean_std_path)
            self.the_mean = mean_std[0, self.variables_list].astype(np.float32)
            self.the_std = mean_std[1, self.variables_list].astype(np.float32)
        else:
            with open(mean_std_path) as f:
                mean_std = json.load(f)
            # mean_std.json already contains only the 69 selected variables,
            # in the same order as self.variables_list.
            self.the_mean = np.array(mean_std["mean"], dtype=np.float32)
            self.the_std = np.array(mean_std["std"], dtype=np.float32)

        self.data_mean_tensor = torch.from_numpy(self.the_mean).float()
        self.data_std_tensor = torch.from_numpy(self.the_std).float()

    def normalization(self, sample):
        return (sample[self.variables_list] - self.the_mean[:, None, None]) / self.the_std[
            :, None, None
        ]

    def __len__(self):
        return self.length

    # ------------------------------------------------------------------
    # Per-step reader hooks; overridden by Memmap/V4 subclasses.
    # ``record_function`` tags выставляются здесь, в overrides — со своими
    # именами; общий стэк остаётся в `_build_sample_pair`.
    # ------------------------------------------------------------------

    def _read_x(self, time_idx: int) -> torch.Tensor:
        """Прочитать вход на абсолютном time_idx; (C, H, W), float32."""
        file_path = self.x_file_list[time_idx]
        with record_function("custom_np_load_x"):
            sample = self.custom_np_load(file_path)
        with record_function("normalization_x"):
            sample = self.normalization(sample)
        with record_function("from_numpy_x"):
            return torch.from_numpy(sample).float()

    def _read_y(self, y_time: pd.Timestamp) -> torch.Tensor:
        """Прочитать таргет в момент времени y_time; (C, H, W), float32."""
        y_file_path = os.path.join(
            self.data_folder,
            str(y_time.year) + f"-{self.idx_in_year(y_time):04d}" + ".npy",
        )
        with record_function("custom_np_load_y"):
            sample = self.custom_np_load(y_file_path)
        with record_function("normalization_y"):
            sample = self.normalization(sample)
        with record_function("from_numpy_y"):
            return torch.from_numpy(sample).float()

    def _build_sample_pair(self, index):
        """Общий цикл по таймстепам x/y, сборка и опциональный stack по muti_target_steps."""
        sample_start_idx = self.sample_start_indices[index]
        x_sequence = [
            self._read_x(sample_start_idx + i * self.frame_interval)
            for i in range(self.start_time_x, self.end_time_x + 1)
        ]
        with record_function("stack_x"):
            sample_x_sequence = torch.stack(x_sequence, dim=0)  # [T, C, H, W]

        y_sequences = []
        for steps in range(self.muti_target_steps):
            offset = pd.Timedelta(hours=(steps + 1) * self.lead_time)
            y_sequence = [
                self._read_y(self.x_time_ilst[sample_start_idx + i * self.frame_interval] + offset)
                for i in range(self.start_time_y, self.end_time_y + 1)
            ]
            with record_function("stack_y"):
                y_sequences.append(torch.stack(y_sequence, dim=0))  # [T, C, H, W]

        if self.muti_target_steps > 1:
            # [muti_target_steps, T, C, H, W]
            sample_y_all = torch.stack(y_sequences, dim=0)
        else:
            # [T, C, H, W]
            sample_y_all = y_sequences[0]
        return sample_x_sequence, sample_y_all

    def __getitem__(self, index):
        return self._build_sample_pair(index)


class WeatherBench128Memmap(WeatherBench128):
    """Memmap-backed v3 dataset (§2.5 of the dataloader plan).

    Reads from a single contiguous ``np.memmap`` produced by
    ``tools/repack_era5.py``: shape ``(T, C_selected, H_cut, W_cut)`` already
    channel-filtered (69 channels via ``variables_list``) and spatially
    cropped (per the ``cut`` used at repack time). ``__getitem__`` becomes a
    pair of row slices plus normalisation, so the parent's 24 per-sample
    ``h5netcdf`` opens are skipped entirely.

    Parent ``__init__`` still runs (time list, valid_idx, mean/std, file list
    for ``self.data_folder``); those structures are reused, but
    ``custom_np_load`` is never called in this branch.

    Args:
        memmap_path: Path to the ``.dat`` produced by ``tools/repack_era5.py``.
        memmap_meta_path: Path to the matching ``.meta.json``. Defaults to
            replacing the ``.dat`` suffix with ``.meta.json``.
        **kwargs: Forwarded to ``WeatherBench128.__init__``.
    """

    returns_normalized = True

    def __init__(
        self,
        memmap_path: str,
        memmap_meta_path: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if memmap_meta_path is None:
            memmap_meta_path = (
                memmap_path[:-4] + ".meta.json"
                if memmap_path.endswith(".dat")
                else memmap_path + ".meta.json"
            )
        with open(memmap_meta_path) as f:
            meta = json.load(f)
        self._memmap_shape: tuple[int, ...] = tuple(meta["shape"])
        self._memmap_dtype = np.dtype(meta["dtype"])
        if len(self._memmap_shape) != 4:
            raise ValueError(f"Memmap shape must be (T, C, H, W), got {self._memmap_shape}")
        if self._memmap_shape[1] != len(self.the_mean):
            raise ValueError(
                f"Memmap channel count {self._memmap_shape[1]} does not match "
                f"normalization stats ({len(self.the_mean)} channels)"
            )
        if "variables_list" in meta and list(meta["variables_list"]) != self.variables_list:
            raise ValueError("Memmap variables_list does not match WeatherBench128.variables_list")
        if self.cut is not None:
            expected_cut = [
                self.cut[0][0],
                self.cut[0][1],
                self.cut[1][0],
                self.cut[1][1],
            ]
            if "cut" in meta and list(meta["cut"]) != expected_cut:
                raise ValueError(f"Memmap cut {meta['cut']} does not match config cut {expected_cut}")
            expected_hw = (
                self.cut[0][1] - self.cut[0][0],
                self.cut[1][1] - self.cut[1][0],
            )
            if self._memmap_shape[-2:] != expected_hw:
                raise ValueError(
                    f"Memmap spatial shape {self._memmap_shape[-2:]} does not match "
                    f"config cut shape {expected_hw}"
                )
        self._memmap_path = memmap_path
        self._memmap = np.memmap(
            memmap_path,
            dtype=self._memmap_dtype,
            mode="r",
            shape=self._memmap_shape,
        )
        # Year -> first-row offset; idx = row_starts[year] + hour_in_year.
        row_starts: dict[int, int] = {}
        offset = 0
        for y, n in zip(meta["years"], meta["hours_per_year"], strict=True):
            row_starts[int(y)] = offset
            offset += int(n)
        self._memmap_row_starts = row_starts
        max_target_time = self.max_required_time
        required_years = set(range(self.x_time_ilst[0].year, max_target_time.year + 1))
        missing_years = sorted(required_years.difference(row_starts))
        if missing_years:
            raise ValueError(f"Memmap is missing required years: {missing_years}")

    def _memmap_row(self, timestamp: pd.Timestamp) -> int:
        """Return the memmap row corresponding to ``timestamp``."""
        return self._memmap_row_starts[timestamp.year] + self.idx_in_year(timestamp)

    # Memmap-чтение: один slice вместо h5netcdf parse, без channel-filter
    # (он уже применён на этапе repack).

    def _read_x(self, time_idx: int) -> torch.Tensor:
        t = self.x_time_ilst[time_idx]
        with record_function("memmap_read_x"):
            raw = np.asarray(self._memmap[self._memmap_row(t)])
        with record_function("normalization_x"):
            normed = (raw - self.the_mean[:, None, None]) / self.the_std[:, None, None]
        with record_function("from_numpy_x"):
            return torch.from_numpy(normed).float()

    def _read_y(self, y_time: pd.Timestamp) -> torch.Tensor:
        with record_function("memmap_read_y"):
            raw = np.asarray(self._memmap[self._memmap_row(y_time)])
        with record_function("normalization_y"):
            normed = (raw - self.the_mean[:, None, None]) / self.the_std[:, None, None]
        with record_function("from_numpy_y"):
            return torch.from_numpy(normed).float()
