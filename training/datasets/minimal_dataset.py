#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from typing import Any, Optional

from torch.utils.data import Dataset
import numpy as np
from training.datasets.augmentations import TimeseriesAugmentations
from sklearn.preprocessing import StandardScaler


class RasouliPPGDataset(Dataset):
    """
    Dataset for Rasouli pregnancy PPG segments.

    Expects 1-minute PPG segments sampled at 20 Hz:
        - 1D PPG array of length 1200 per sample
        - corresponding integer participant ID `pid`

    Data can be provided either as:
        - `data_path`: path to a file that can be loaded with `np.load(..., allow_pickle=True)`
          and yields a list/array of dict-like records or a DataFrame-like object, or
        - `records`: a list/array of dictionaries or a DataFrame-like object.

    Each record must expose:
        - a 1D PPG array via key `ppg_key` (default: "ppg")
        - an integer participant ID via key `pid_key` (default: "pid")
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        records: Optional[Any] = None,
        do_zscore: bool = True,
        augmentation_name: str = "identity",
        augmentation_config: Optional[dict] = None,
        ppg_key: str = "ppg",
        pid_key: str = "pid",
    ):
        if data_path is None and records is None:
            raise ValueError("Either `data_path` or `records` must be provided.")

        if records is None:
            # Generic loader: expects np.load to return an array-like of dicts or a DataFrame-like object.
            loaded = np.load(data_path, allow_pickle=True)
            records = loaded

        self.records = records
        self.ppg_key = ppg_key
        self.pid_key = pid_key

        self.do_zscore = do_zscore
        # augmentation params
        self.augmentation_tool = TimeseriesAugmentations()
        self.augmentation_name = augmentation_name
        self.augmentation_config = augmentation_config or {}

    def __len__(self):
        # Handle list/array-like or DataFrame-like containers
        if hasattr(self.records, "__len__"):
            return len(self.records)
        raise TypeError("Unsupported records container type for RasouliPPGDataset.")

    def _get_record(self, idx: int):
        # DataFrame-like (e.g., pandas) objects usually expose `.iloc`
        if hasattr(self.records, "iloc"):
            row = self.records.iloc[idx]
        else:
            row = self.records[idx]
        return row

    def __getitem__(self, idx):
        # 1) Load real 1D PPG array of length 1200 and reshape to (1, 1200)
        record = self._get_record(idx)

        # dict-like (supports key access) or Series-like
        ppg = record[self.ppg_key]
        pid = int(record[self.pid_key])

        ppg = np.asarray(ppg, dtype=np.float32)
        if ppg.ndim != 1:
            raise ValueError(f"PPG segment must be 1D, got shape {ppg.shape}")
        if ppg.shape[0] != 1200:
            raise ValueError(f"PPG segment must have length 1200, got {ppg.shape[0]}")

        segment = ppg.reshape(1, 1200)  # (channels=1, time=1200)

        # 3) Call self.transform() twice on the same array to get two views
        signal_1 = self.transform(segment.copy())
        signal_2 = self.transform(segment.copy())

        # 4) Stack into shape (1, 1200, 2)
        views = np.stack([signal_1, signal_2], axis=-1).astype(np.float32)

        return views, pid, idx

    def transform(self, segment: np.ndarray) -> np.ndarray:
        if self.do_zscore:
            segment = zscore_segment(segment)
        if self.augmentation_name != "identity":
            # create a view/augmentation of the current segment
            # the augmentation functions receive (time, num_channels),
            # hence the .T operators
            segment = getattr(self.augmentation_tool, self.augmentation_name)(
                segment.T, **self.augmentation_config
            ).T

        return segment


#
# utility functions
#


def zscore_segment(x: np.ndarray) -> np.ndarray:
    """Z-score a segment, per-channel basis"""
    zscore_tool = StandardScaler()
    x = zscore_tool.fit_transform(x.T).T

    return x