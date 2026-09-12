"""The batch gather must emit the batches the per-sequence path emitted.

``FoldSequenceDataset.__getitems__`` replaced one Python call per sequence with one
strided gather. Sequence runs already registered under the old path stay valid only if
the change is invisible to the model: the same sequences in the same batches in the same
order, with the same bits in every window. That is what these tests assert, by running
the same sampler over the gather and over a view of the dataset that hides
``__getitems__`` and so takes the path the loader took before.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from case_studies.utils.sequence_dataset import (
    FoldSequenceDataset,
    SequenceStore,
    _device_available_bytes,
    _resolve_gather_device,
    collate_sequences,
    collate_with_metadata,
)

LOOKBACK = 6
SYMBOL_LENGTHS = (20, 35, 12, 28)
N_FEATURES = 5


class _PerItemView(Dataset):
    """The dataset without ``__getitems__``, which is the path the loader took before."""

    def __init__(self, dataset: FoldSequenceDataset) -> None:
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int):
        return self._dataset[idx]


def _synthetic_store() -> SequenceStore:
    """A store with uneven symbols and missing cells, which is the shape folds have."""

    rng = np.random.default_rng(11)
    features, targets, timestamps, entities = [], [], [], []
    symbol_idx, end_idx = [], []
    for symbol, length in enumerate(SYMBOL_LENGTHS):
        feats = rng.standard_normal((length, N_FEATURES)).astype(np.float32)
        # Unobserved cells reach the gather as NaN, so they have to survive it bit for bit.
        feats[rng.random(feats.shape) < 0.1] = np.nan
        features.append(feats)
        targets.append(rng.standard_normal(length).astype(np.float32))
        timestamps.append(np.datetime64("2024-01-01") + np.arange(length, dtype="timedelta64[D]"))
        entities.append(f"SYM{symbol}")
        for end in range(LOOKBACK, length):
            symbol_idx.append(symbol)
            end_idx.append(end)

    return SequenceStore(
        features=features,
        targets=targets,
        timestamps=timestamps,
        entities=entities,
        symbol_idx=np.asarray(symbol_idx, dtype=np.int64),
        end_idx=np.asarray(end_idx, dtype=np.int64),
        lookback=LOOKBACK,
    )


def _assert_bit_identical(gathered: torch.Tensor, per_item: torch.Tensor) -> None:
    """Compare the raw bits, so a NaN counts as equal to itself and to nothing else."""

    assert gathered.dtype == per_item.dtype
    assert gathered.shape == per_item.shape
    left = np.ascontiguousarray(gathered.cpu().numpy())
    right = np.ascontiguousarray(per_item.cpu().numpy())
    np.testing.assert_array_equal(left.view(np.uint8), right.view(np.uint8))


def _loaders(dataset: FoldSequenceDataset, collate, *, batch_size: int, shuffle: bool):
    """The same sampler over the gather and over the per-sequence path."""

    def build(source, seed: int = 1234):
        return DataLoader(
            source,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=collate,
            generator=torch.Generator().manual_seed(seed),
        )

    return build(dataset), build(_PerItemView(dataset))


def _devices() -> list[str]:
    return ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


@pytest.mark.parametrize("device", _devices())
def test_the_gathered_training_batches_are_the_per_sequence_batches(device: str) -> None:
    dataset = FoldSequenceDataset(_synthetic_store(), device=device)
    gathered_loader, per_item_loader = _loaders(
        dataset, collate_sequences, batch_size=7, shuffle=True
    )

    batches = 0
    for (X, y), (X_ref, y_ref) in zip(gathered_loader, per_item_loader, strict=True):
        _assert_bit_identical(X, X_ref)
        _assert_bit_identical(y, y_ref)
        batches += 1
    assert batches > 1


@pytest.mark.parametrize("device", _devices())
def test_the_gathered_evaluation_batches_keep_their_timestamps_and_entities(
    device: str,
) -> None:
    dataset = FoldSequenceDataset(_synthetic_store(), include_metadata=True, device=device)
    gathered_loader, per_item_loader = _loaders(
        dataset, collate_with_metadata, batch_size=7, shuffle=False
    )

    for gathered, reference in zip(gathered_loader, per_item_loader, strict=True):
        X, y, timestamps, entities = gathered
        X_ref, y_ref, timestamps_ref, entities_ref = reference
        _assert_bit_identical(X, X_ref)
        _assert_bit_identical(y, y_ref)
        np.testing.assert_array_equal(timestamps, timestamps_ref)
        np.testing.assert_array_equal(entities, entities_ref)
        assert timestamps.dtype == timestamps_ref.dtype
        assert entities.dtype == entities_ref.dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_a_gathered_batch_needs_no_copy_to_reach_the_model() -> None:
    """The training loop's ``.to(device)`` must be a no-op, not a second transfer."""

    dataset = FoldSequenceDataset(_synthetic_store(), device="cuda")
    assert dataset.returns_device_tensors
    X, _y = dataset.__getitems__(list(range(16))).payload
    assert X.device.type == "cuda"
    assert X.to(torch.device("cuda"), non_blocking=True) is X


def test_a_store_too_big_for_the_card_is_gathered_on_the_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback is chosen by measuring, before anything is allocated."""

    store = _synthetic_store()
    monkeypatch.setattr(
        "case_studies.utils.sequence_dataset._device_available_bytes", lambda device: 1
    )
    assert _resolve_gather_device(store, "cuda").type == "cpu"

    dataset = FoldSequenceDataset(store, device="cuda")
    assert not dataset.returns_device_tensors
    assert dataset.gather_device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_the_fit_measurement_counts_what_this_process_can_reuse() -> None:
    """A released fold's blocks are still this process's to allocate."""

    device = torch.device("cuda")
    assert _device_available_bytes(device) >= torch.cuda.mem_get_info(device)[0]
