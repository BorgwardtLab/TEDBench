import math
from typing import Optional, Union

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset

from .afdb import AFDBDataset, AFDBStreamingDataset
from .afdb_ted import AFDBTEDStreamingDataset
from .cath_test import CATHTestDataset
from .hf_datasets import HFAFDBDataset, HFTEDDataset, HFCATHTestDataset


class LightningStructureDataset(pl.LightningDataModule):
    """PyTorch Lightning data module for protein structure datasets.

    Maps dataset names to their implementations and handles train/val/test splits
    with optional data transforms and subsampling.

    Args:
        root: Root directory for local datasets **or** a HuggingFace repo ID /
            local HF directory for the ``hf_*`` variants.
        dataset_name: Which dataset to load.  Valid values:

            Local datasets (downloaded automatically on first use):

            * ``"ted"``         — TEDBench train/val/test (labeled, 5-tuple)
            * ``"afdb_stream"`` — AFDB pretraining corpus (unlabeled, 4-tuple)
            * ``"cath4.4"``     — CATH 4.4 experimental test set (labeled, 5-tuple)
            * ``"afdb"``        — Legacy AFDB dataset (pre-processed tensors)

            HuggingFace datasets (``root`` = HF repo ID or local HF directory):

            * ``"hf_ted"``      — TEDBench from ``TEDBench/ted``
            * ``"hf_afdb"``     — AFDB pretraining from ``TEDBench/afdb``
            * ``"hf_cath4.4"``  — CATH 4.4 test set from ``TEDBench/cath``

        train_transform: Optional transform applied to training samples.
        transform: Optional transform applied to validation/test samples.
        subsample: If an int, keep that many training samples; if a float in
            (0, 1), keep that fraction of training samples.
        **kwargs: Passed to the underlying :class:`~torch.utils.data.DataLoader`
            (e.g. ``batch_size``, ``num_workers``).
    """

    datasets_map = {
        "afdb": AFDBDataset,
        "afdb_stream": AFDBStreamingDataset,
        "ted": AFDBTEDStreamingDataset,
        "cath4.4": CATHTestDataset,
        "hf_afdb": HFAFDBDataset,
        "hf_ted": HFTEDDataset,
        "hf_cath4.4": HFCATHTestDataset,
    }

    def __init__(
        self,
        root: str,
        dataset_name: str = "ted",
        train_transform=None,
        transform=None,
        subsample: Optional[Union[int, float]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.root = root
        self.dataset_name = dataset_name
        self.train_transform = train_transform
        self.transform = transform
        self.subsample = subsample
        self.kwargs = kwargs
        self.train_dataset = self.val_dataset = self.test_dataset = None

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str = "fit") -> None:
        if stage == "fit":
            if self.train_dataset is None:
                self.train_dataset = self.datasets_map[self.dataset_name](
                    root=self.root,
                    split="train",
                    transform=self.train_transform,
                )
                if self.subsample is not None:
                    n = len(self.train_dataset)
                    k = math.floor(n * self.subsample) if 0 < self.subsample < 1 else int(self.subsample)
                    if k < n:
                        indices = torch.randperm(n)[:k].tolist()
                        self.train_dataset = Subset(self.train_dataset, indices)
                        self.train_dataset.collate_fn = self.train_dataset.dataset.collate_fn
            if self.val_dataset is None:
                self.val_dataset = self.datasets_map[self.dataset_name](
                    root=self.root,
                    split="val",
                    transform=self.transform,
                )
            print(
                f"# Training samples: {len(self.train_dataset)} \n"
                f"# Val samples: {len(self.val_dataset)} \n"
            )
        elif stage == "val":
            if self.val_dataset is None:
                self.val_dataset = self.datasets_map[self.dataset_name](
                    root=self.root,
                    split="val",
                    transform=self.transform,
                )
        if stage == "test":
            if self.test_dataset is None:
                self.test_dataset = self.datasets_map[self.dataset_name](
                    root=self.root,
                    split="test",
                    transform=self.transform,
                )
            print(f"# Test samples: {len(self.test_dataset)}")

    def dataloader(self, dataset, **kwargs) -> DataLoader:
        return DataLoader(dataset, **kwargs)

    def train_dataloader(self, shuffle=True) -> DataLoader:
        return self.dataloader(
            self.train_dataset,
            shuffle=shuffle,
            collate_fn=self.train_dataset.collate_fn,
            **self.kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_dataset is not None
        return self.dataloader(
            self.val_dataset,
            shuffle=False,
            collate_fn=self.val_dataset.collate_fn,
            **self.kwargs,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None
        return self.dataloader(
            self.test_dataset,
            shuffle=False,
            collate_fn=self.test_dataset.collate_fn,
            **self.kwargs,
        )


class TEDLightningDataset(LightningStructureDataset):
    """Variant of :class:`LightningStructureDataset` for the streaming TED dataset.

    Overrides :meth:`train_dataloader` to expose a ``drop_last`` argument needed
    when the streaming dataset is used with fixed-length chunks.
    """

    def train_dataloader(self, drop_last=True, shuffle=True) -> DataLoader:
        return self.dataloader(
            self.train_dataset,
            shuffle=shuffle,
            collate_fn=self.train_dataset.collate_fn,
            drop_last=drop_last,
            **self.kwargs,
        )
