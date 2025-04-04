import random
import numpy as np
import os.path as osp
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import xarray as xr

from timm.data.distributed_sampler import OrderedDistributedSampler, RepeatAugSampler
from functools import partial
from itertools import repeat
from typing import Callable
import tqdm


def expand_to_chs(x, n):
    if not isinstance(x, (tuple, list)):
        x = tuple(repeat(x, n))
    elif len(x) == 1:
        x = x * n
    else:
        assert len(x) == n, 'normalization stats must match image channels'
    return x

class PrefetchLoader:

    def __init__(self,
                 loader,
                 mean=None,
                 std=None,
                 channels=3,
                 fp16=False):

        self.fp16 = fp16
        self.loader = loader
        if mean is not None and std is not None:
            mean = expand_to_chs(mean, channels)
            std = expand_to_chs(std, channels)
            normalization_shape = (1, channels, 1, 1)

            self.mean = torch.tensor([x * 255 for x in mean]).cuda().view(normalization_shape)
            self.std = torch.tensor([x * 255 for x in std]).cuda().view(normalization_shape)
            if fp16:
                self.mean = self.mean.half()
                self.std = self.std.half()
        else:
            self.mean, self.std = None, None

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                if self.fp16:
                    if self.mean is not None:
                        next_input = next_input.half().sub_(self.mean).div_(self.std)
                        next_target = next_target.half().sub_(self.mean).div_(self.std)
                    else:
                        next_input = next_input.half()
                        next_target = next_target.half()
                else:
                    if self.mean is not None:
                        next_input = next_input.float().sub_(self.mean).div_(self.std)
                        next_target = next_target.float().sub_(self.mean).div_(self.std)
                    else:
                        next_input = next_input.float()
                        next_target = next_target.float()

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset

def create_loader_(dataset,
                  batch_size,
                  shuffle=True,
                  is_training=False,
                  mean=None,
                  std=None,
                  num_workers=1,
                  num_aug_repeats=0,
                  input_channels=1,
                  use_prefetcher=False,
                  distributed=False,
                  pin_memory=False,
                  drop_last=False,
                  fp16=False,
                  collate_fn=None,
                  persistent_workers=True,
                  worker_seeding='all'):
    sampler = None
    if distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
        if is_training:
            if num_aug_repeats:
                sampler = RepeatAugSampler(dataset, num_repeats=num_aug_repeats)
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            sampler = OrderedDistributedSampler(dataset)
    else:
        assert num_aug_repeats==0, "RepeatAugment is not supported in non-distributed or IterableDataset"

    if collate_fn is None:
        collate_fn = torch.utils.data.dataloader.default_collate
    loader_class = torch.utils.data.DataLoader

    loader_args = dict(
        batch_size=batch_size,
        shuffle=shuffle and (not isinstance(dataset, torch.utils.data.IterableDataset)) and sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=partial(worker_init, worker_seeding=worker_seeding),
        persistent_workers=persistent_workers
    )
    try:
        loader = loader_class(dataset, **loader_args)
    except TypeError:
        loader_args.pop('persistent_workers')  # only in Pytorch 1.7+
        loader = loader_class(dataset, **loader_args)

    if use_prefetcher:
        loader = PrefetchLoader(
            loader,
            mean=mean,
            std=std,
            channels=input_channels,
            fp16=fp16,
        )

    return loader


d2r = np.pi / 180

def worker_init(worker_id, worker_seeding='all'):
    worker_info = torch.utils.data.get_worker_info()
    assert worker_info.id == worker_id
    if isinstance(worker_seeding, Callable):
        seed = worker_seeding(worker_info)
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
    else:
        assert worker_seeding in ('all', 'part')
        # random / torch seed already called in dataloader iter class w/ worker_info.seed
        # to reproduce some old results (same seed + hparam combo), partial seeding
        # is required (skip numpy re-seed)
        if worker_seeding == 'all':
            np.random.seed(worker_info.seed % (2 ** 32 - 1))


def create_loader(dataset,
                  batch_size,
                  shuffle=True,
                  is_training=False,
                  mean=None,
                  std=None,
                  num_workers=1,
                  num_aug_repeats=0,
                  input_channels=1,
                  use_prefetcher=False,
                  distributed=False,
                  pin_memory=False,
                  drop_last=False,
                  fp16=False,
                  collate_fn=None,
                  persistent_workers=True,
                  worker_seeding='all'):
    sampler = None
    if distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
        if is_training:
            if num_aug_repeats:
                sampler = RepeatAugSampler(dataset, num_repeats=num_aug_repeats)
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            sampler = OrderedDistributedSampler(dataset)
    else:
        assert num_aug_repeats==0, "RepeatAugment is not supported in non-distributed or IterableDataset"

    if collate_fn is None:
        collate_fn = torch.utils.data.dataloader.default_collate
    loader_class = torch.utils.data.DataLoader

    loader_args = dict(
        batch_size=batch_size,
        shuffle=shuffle and (not isinstance(dataset, torch.utils.data.IterableDataset)) and sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=partial(worker_init, worker_seeding=worker_seeding),
        persistent_workers=persistent_workers
    )
    try:
        loader = loader_class(dataset, **loader_args)
    except TypeError:
        loader_args.pop('persistent_workers')  # only in Pytorch 1.7+
        loader = loader_class(dataset, **loader_args)

    return loader

def latlon2xyz(lat, lon):
    if type(lat) == torch.Tensor:
        x = -torch.cos(lat)*torch.cos(lon)
        y = -torch.cos(lat)*torch.sin(lon)
        z = torch.sin(lat)

    if type(lat) == np.ndarray:
        x = -np.cos(lat)*np.cos(lon)
        y = -np.cos(lat)*np.sin(lon)
        z = np.sin(lat)
    return x, y, z


def xyz2latlon(x, y, z):
    if type(x) == torch.Tensor:
        lat = torch.arcsin(z)
        lon = torch.atan2(-y, -x)

    if type(x) == np.ndarray:
        lat = np.arcsin(z)
        lon = np.arctan2(-y, -x)
    return lat, lon


data_map = {
    'g': 'geopotential',
    'z': 'geopotential_500',
    't': 'temperature',
    't850': 'temperature_850',
    'tp': 'total_precipitation',
    't2m': '2m_temperature',
    'r': 'relative_humidity',
    'u10': '10m_u_component_of_wind',
    'u': 'u_component_of_wind',
    'v10': '10m_v_component_of_wind',
    'v': 'v_component_of_wind',
    'tcc': 'total_cloud_cover',
}

mv_data_map = {
    **dict.fromkeys(['mv', 'mv4'], ['r', 't', 'u', 'v']),
    'mv5': ['g', 'r', 't', 'u', 'v'],
    'mv8': ['u10', 'v10', 't2m', 'g', 'r', 't', 'u', 'v'],
    'mv6': ['u10', 'v10', 't2m', 't', 'u', 'v'],
    'mv3': ['u10', 'v10', 't2m'],
    'uv10': ['u10', 'v10'],
    'uv': ['u', 'v'],
    'mv12': list(data_map.keys()),
}


class WeatherBenchDataset(Dataset):
    """Wheather Bench Dataset <http://arxiv.org/abs/2002.00469>`_

    Args:
        data_root (str): Path to the dataset.
        data_name (str): Name of the weather modality in Wheather Bench.
        training_time (list): The arrange of years for training.
        idx_in (list): The list of input indices.
        idx_out (list): The list of output indices to predict.
        step (int): Sampling step in the time dimension.
        level (int): Used level in the multi-variant version.
        data_split (str): The resolution (degree) of Wheather Bench splits.
        use_augment (bool): Whether to use augmentations (defaults to False).
    """

    def __init__(self, data_root, data_name, training_time,
                 idx_in, idx_out, step=1, level=1, data_split='5_625',
                 mean=None, std=None,
                 transform_data=None, transform_labels=None, use_augment=False):
        super().__init__()
        self.data_root = data_root
        self.data_name = data_name
        self.data_split = data_split
        self.training_time = training_time
        self.idx_in = np.array(idx_in)
        self.idx_out = np.array(idx_out)
        self.step = step
        self.level = level
        self.data = None
        self.mean = mean
        self.std = std
        self.transform_data = transform_data
        self.transform_labels = transform_labels
        self.use_augment = use_augment
        assert isinstance(level, (int, list))

        self.time = None
        shape = int(32 * 5.625 / float(data_split.replace('_', '.')))
        self.shape = (shape, shape * 2)

        if isinstance(data_name, list):
            data_name = data_name[0]
        if 'mv' in data_name:  # multi-variant version
            self.data_name = mv_data_map[data_name]
            self.data, self.mean, self.std = [], [], []
            for name in self.data_name:
                data, mean, std = self._load_data_xarray(data_name=name, single_variant=False)
                self.data.append(data)
                self.mean.append(mean)
                self.std.append(std)
            print(len(self.mean))
            self.data = np.concatenate(self.data, axis=1)
            self.mean = np.concatenate(self.mean, axis=1)
            self.std = np.concatenate(self.std, axis=1)
        else:  # single variant
            self.data_name = data_name
            self.data, mean, std = self._load_data_xarray(data_name, single_variant=True)
            if self.mean is None:
                self.mean, self.std = mean, std

        self.valid_idx = np.array(
            range(-idx_in[0], self.data.shape[0]-idx_out[-1]-1))

    def _load_data_xarray(self, data_name, single_variant=True):
        """Loading full data with xarray"""
        if data_name != 'uv10':
            try:
                dataset = xr.open_mfdataset(self.data_root+'/{}/{}*.nc'.format(
                    data_map[data_name], data_map[data_name]), combine='by_coords')
            except (AttributeError, ValueError):
                assert False and 'Please install xarray and its dependency (e.g., netcdf4), ' \
                                    'pip install xarray==0.19.0,' \
                                    'pip install netcdf4 h5netcdf dask'
            except OSError:
                print("OSError: Invalid path {}/{}/*.nc".format(self.data_root, data_map[data_name]))
                assert False
            dataset = dataset.sel(time=slice(*self.training_time))
            dataset = dataset.isel(time=slice(None, -1, self.step))
            if self.time is None and single_variant:
                self.week = dataset['time.week']
                self.month = dataset['time.month']
                self.year = dataset['time.year']
                self.time = np.stack(
                    [self.week, self.month, self.year], axis=1)
                lon, lat = np.meshgrid(
                    (dataset.lon-180) * d2r, dataset.lat*d2r)
                x, y, z = latlon2xyz(lat, lon)
                self.V = np.stack([x, y, z]).reshape(3, self.shape[0]*self.shape[1]).T
            if not single_variant and isinstance(self.level, list):
                dataset = dataset.sel(level=np.array(self.level))
            data = dataset.get(data_name).values[:, np.newaxis, :, :]

        elif data_name == 'uv10':
            input_datasets = []
            for key in ['u10', 'v10']:
                try:
                    dataset = xr.open_mfdataset(self.data_root+'/{}/{}*.nc'.format(
                        data_map[key], data_map[key]), combine='by_coords')
                except (AttributeError, ValueError):
                    assert False and 'Please install xarray and its dependency (e.g., netcdf4), ' \
                                     'pip install xarray==0.19.0,' \
                                     'pip install netcdf4 h5netcdf dask'
                except OSError:
                    print("OSError: Invalid path {}/{}/*.nc".format(self.data_root, data_map[key]))
                    assert False
                dataset = dataset.sel(time=slice(*self.training_time))
                dataset = dataset.isel(time=slice(None, -1, self.step))
                if self.time is None and single_variant:
                    self.week = dataset['time.week']
                    self.month = dataset['time.month']
                    self.year = dataset['time.year']
                    self.time = np.stack(
                        [self.week, self.month, self.year], axis=1)
                    lon, lat = np.meshgrid(
                        (dataset.lon-180) * d2r, dataset.lat*d2r)
                    x, y, z = latlon2xyz(lat, lon)
                    self.V = np.stack([x, y, z]).reshape(3, self.shape[0]*self.shape[1]).T
                input_datasets.append(dataset.get(key).values[:, np.newaxis, :, :])
            data = np.concatenate(input_datasets, axis=1)

        # uv10
        if len(data.shape) == 5:
            data = data.squeeze(1)
        # humidity
        if data_name == 'r' and single_variant:
            data = data[:, -1:, ...]
        # multi-variant level
        if not single_variant and isinstance(self.level, int):
            data = data[:, -self.level:, ...]

        mean = data.mean(axis=(0, 2, 3)).reshape(1, data.shape[1], 1, 1)
        std = data.std(axis=(0, 2, 3)).reshape(1, data.shape[1], 1, 1)
        data = (data - mean) / std

        return data, mean, std

    def _augment_seq(self, seqs, crop_scale=0.96):
        """Augmentations as a video sequence"""
        _, _, h, w = seqs.shape  # original shape, e.g., [4, 1, 128, 256]
        seqs = F.interpolate(seqs, scale_factor=1 / crop_scale, mode='bilinear')
        _, _, ih, iw = seqs.shape
        # Random Crop
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        seqs = seqs[:, :, x:x+h, y:y+w]
        # Random Flip
        if random.randint(0, 1):
            seqs = torch.flip(seqs, dims=(3, ))  # horizontal flip
        return seqs

    def __len__(self):
        return self.valid_idx.shape[0]

    def __getitem__(self, index):
        index = self.valid_idx[index]
        data = torch.tensor(self.data[index+self.idx_in])
        labels = torch.tensor(self.data[index+self.idx_out])
        if self.use_augment:
            len_data = self.idx_in.shape[0]
            seqs = self._augment_seq(torch.cat([data, labels], dim=0), crop_scale=0.96)
            data = seqs[:len_data, ...]
            labels = seqs[len_data:, ...]
        return data, labels


def load_data(batch_size,
              val_batch_size,
              data_root,
              num_workers=4,
              data_split='5_625',
              data_name='t2m',
              train_time=['1979', '2015'],
              val_time=['2016', '2016'],
              test_time=['2017', '2018'],
              idx_in=[*range(0, 12)],
              idx_out=[*range(12, 24)],
              step=1,
              level=1,
              distributed=False, use_augment=False, use_prefetcher=False, drop_last=False, fp16=False,
              **kwargs):

    assert data_split in ['5_625', '2_8125', '1_40625']
    _dataroot = osp.join(data_root, f'{data_split.replace("_", ".")}deg')
    weather_dataroot = _dataroot if osp.exists(_dataroot) else osp.join(data_root, f'{data_split}deg')

    train_set = WeatherBenchDataset(data_root=weather_dataroot,
                                    data_name=data_name, data_split=data_split,
                                    training_time=train_time,
                                    idx_in=idx_in,
                                    idx_out=idx_out,
                                    step=step, level=level, use_augment=use_augment)
    vali_set = WeatherBenchDataset(weather_dataroot,
                                    data_name=data_name, data_split=data_split,
                                    training_time=val_time,
                                    idx_in=idx_in,
                                    idx_out=idx_out,
                                    step=step, level=level, use_augment=False,
                                    mean=train_set.mean,
                                    std=train_set.std)
    test_set = WeatherBenchDataset(weather_dataroot,
                                    data_name, data_split=data_split,
                                    training_time=test_time,
                                    idx_in=idx_in,
                                    idx_out=idx_out,
                                    step=step, level=level, use_augment=False,
                                    mean=train_set.mean,
                                    std=train_set.std)

    dataloader_train = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=True,
                                     num_workers=num_workers,
                                     distributed=distributed, use_prefetcher=use_prefetcher, fp16 = fp16)
    dataloader_vali = create_loader(vali_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher, fp16 = fp16)
    dataloader_test = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher, fp16 = fp16)

    return dataloader_train, dataloader_vali, dataloader_test, train_set.mean, train_set.std


data_map2 = {
    'z': 'geopotential',
    't': 'temperature',
    'tp': 'total_precipitation',
    't2m': '2m_temperature',
    'r': 'relative_humidity',
    's': 'specific_humidity',
    'u10': '10m_u_component_of_wind',
    'u': 'u_component_of_wind',
    'v10': '10m_v_component_of_wind',
    'v': 'v_component_of_wind',
    'tcc': 'total_cloud_cover',
    "lsm": "constants",
    "o": "orography",
    "l": "lat2d",
}

mv_data_map2 = {
    **dict.fromkeys(['mv', 'mv4'], ['u', 'v', 'z', 't']),
    'mv5': ['z', 'r', 't', 'u', 'v'],
    'uv10': ['u10', 'v10'],
    'uv': ['u', 'v'],
    'mv12': ['lsm', 'o', 't2m', 'u10', 'v10', 'l', 'z', 'u', 'v', 't', 'r', 's'],
    'gft': ['t2m', 'u10', 'v10', 'tp', 'z', 't', 'r', 'u', 'v'],
}

data_keys_map2 = {
    'o': 'orography',
    'l': 'lat2d',
    's': 'q'
}


class WeatherBenchDataset2(Dataset):
    """Wheather Bench Dataset <http://arxiv.org/abs/2002.00469>`_

    Args:
        data_root (str): Path to the dataset.
        data_name (str|list): Name(s) of the weather modality in Wheather Bench.
        training_time (list): The arrange of years for training.
        idx_in (list): The list of input indices.
        idx_out (list): The list of output indices to predict.
        step (int): Sampling step in the time dimension.
        level (int|list|"all"): Level(s) to use.
        data_split (str): The resolution (degree) of Wheather Bench splits.
        use_augment (bool): Whether to use augmentations (defaults to False).
    """

    def __init__(self, data_root, data_name, training_time,
                 idx_in, idx_out, step=1, levels=['50'], data_split='5_625',
                 mean=None, std=None,
                 transform_data=None, transform_labels=None, use_augment=False):
        super().__init__()
        self.data_root = data_root
        self.data_split = data_split
        self.training_time = training_time
        self.idx_in = np.array(idx_in)
        self.idx_out = np.array(idx_out)
        self.step = step
        self.data = None
        self.mean = mean
        self.std = std
        self.transform_data = transform_data
        self.transform_labels = transform_labels
        self.use_augment = use_augment

        self.time = None
        self.time_size = self.training_time
        shape = int(32 * 5.625 / float(data_split.replace('_', '.')))
        self.shape = (shape, shape * 2)

        self.data, self.mean, self.std = [], [], []
        
        if levels == 'all':
            # levels = ['50', '250', '500', '600', '700', '850', '925']
            levels = ['50', '100', '150', '200', '250', '300', '400', '500', '600', '700', '850', '925', '1000']
        levels = levels if isinstance(levels, list) else [levels]
        levels = [int(level) for level in levels]
        if isinstance(data_name, str) and data_name in mv_data_map2:
            data_names = mv_data_map2[data_name]
        else:
            data_names = data_name if isinstance(data_name, list) else [data_name]

        for name in tqdm.tqdm(data_names):
            data, mean, std = self._load_data_xarray(data_name=name, levels=levels)
            self.data.append(data)
            self.mean.append(mean)
            self.std.append(std)

        for i, data in enumerate(self.data):
            if data.shape[0] != self.time_size:
                self.data[i] = data.repeat(self.time_size, axis=0)

        self.data = np.concatenate(self.data, axis=1)
        self.mean = np.concatenate(self.mean, axis=1)
        self.std = np.concatenate(self.std, axis=1)

        self.valid_idx = np.array(
            range(-idx_in[0], self.data.shape[0]-idx_out[-1]-1))

    def _load_data_xarray(self, data_name, levels):
        """Loading full data with xarray"""
        try:
            dataset = xr.open_mfdataset(self.data_root+'/{}/{}*.nc'.format(
                data_map2[data_name], data_map2[data_name]), combine='by_coords')
        except (AttributeError, ValueError):
            assert False and 'Please install xarray and its dependency (e.g., netcdf4), ' \
                                'pip install xarray==0.19.0,' \
                                'pip install netcdf4 h5netcdf dask'
        except OSError:
            print("OSError: Invalid path {}/{}/*.nc".format(self.data_root, data_map2[data_name]))
            assert False

        if 'time' not in dataset.indexes:
            dataset = dataset.expand_dims(dim={"time": 1}, axis=0)
        else:
            dataset = dataset.sel(time=slice(*self.training_time))
            dataset = dataset.isel(time=slice(None, -1, self.step))
            self.time_size = dataset.dims['time']

        if 'level' not in dataset.indexes:
            dataset = dataset.expand_dims(dim={"level": 1}, axis=1)
        else:
            dataset = dataset.sel(level=np.array(levels))

        if data_name in data_keys_map2:
            data = dataset.get(data_keys_map2[data_name]).values
        else:
            data = dataset.get(data_name).values

        mean = data.mean().reshape(1, 1, 1, 1)
        std = data.std().reshape(1, 1, 1, 1)
        # mean = dataset.mean('time').mean(('lat', 'lon')).compute()[data_name].values
        # std = dataset.std('time').mean(('lat', 'lon')).compute()[data_name].values
        # data = (data - mean) / std

        return data, mean, std

    def _augment_seq(self, seqs, crop_scale=0.96):
        """Augmentations as a video sequence"""
        _, _, h, w = seqs.shape  # original shape, e.g., [4, 1, 128, 256]
        seqs = F.interpolate(seqs, scale_factor=1 / crop_scale, mode='bilinear')
        _, _, ih, iw = seqs.shape
        # Random Crop
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        seqs = seqs[:, :, x:x+h, y:y+w]
        # Random Flip
        if random.randint(0, 1):
            seqs = torch.flip(seqs, dims=(3, ))  # horizontal flip
        return seqs

    def __len__(self):
        return self.valid_idx.shape[0]

    def __getitem__(self, index):
        index = self.valid_idx[index]
        data = torch.tensor(self.data[index+self.idx_in])
        labels = torch.tensor(self.data[index+self.idx_out])
        if self.use_augment:
            len_data = self.idx_in.shape[0]
            seqs = self._augment_seq(torch.cat([data, labels], dim=0), crop_scale=0.96)
            data = seqs[:len_data, ...]
            labels = seqs[len_data:, ...]
        return data, labels


def load_data2(batch_size,
              val_batch_size,
              data_root,
              num_workers=4,
              data_split='5_625',
              data_name='t2m',
              train_time=['1979', '2015'],
              val_time=['2016', '2016'],
              test_time=['2017', '2018'],
              idx_in=[*range(0, 12)],
              idx_out=[*range(12, 24)],
              step=1,
              levels=['50'],
              distributed=False, use_augment=False, use_prefetcher=False, drop_last=False,
              **kwargs):

    assert data_split in ['5_625', '2_8125', '1_40625']
    for suffix in [f'weather_{data_split.replace("_", ".")}deg', f'weather', f'{data_split.replace("_", ".")}deg']:
        if osp.exists(osp.join(data_root, suffix)):
            weather_dataroot = osp.join(data_root, suffix)
    train_set = WeatherBenchDataset2(data_root=weather_dataroot,
                                    data_name=data_name, data_split=data_split,
                                    training_time=train_time,
                                    idx_in=idx_in,
                                    idx_out=idx_out,
                                    step=step, levels=levels, use_augment=use_augment)
    vali_set = WeatherBenchDataset2(weather_dataroot,
                                    data_name=data_name, data_split=data_split,
                                    training_time=val_time,
                                    idx_in=idx_in,
                                    idx_out=idx_out,
                                    step=step, levels=levels, use_augment=False,
                                    mean=train_set.mean,
                                    std=train_set.std)
    test_set = WeatherBenchDataset2(weather_dataroot,
                                    data_name, data_split=data_split,
                                    training_time=test_time,
                                    idx_in=idx_in,
                                    idx_out=idx_out,
                                    step=step, levels=levels, use_augment=False,
                                    mean=train_set.mean,
                                    std=train_set.std)

    dataloader_train = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=True,
                                     num_workers=num_workers,
                                     distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_vali = create_loader(vali_set, # validation_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_test = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)

    return dataloader_train, dataloader_vali, dataloader_test, train_set.mean, train_set.std