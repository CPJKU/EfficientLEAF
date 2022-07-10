import collections
from functools import partial

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


#SOURCE: https://github.com/tensorflow/datasets/blob/17b40dfdf6ce13adde74f82dd1214fe26545b0d3/tensorflow_datasets/audio/crema_d.py#L56
def _compute_split_boundaries(split_probs, n_items):
    """Computes boundary indices for each of the splits in split_probs.
    Args:
      split_probs: List of (split_name, prob), e.g. [('train', 0.6), ('dev', 0.2),
        ('test', 0.2)]
      n_items: Number of items we want to split.
    Returns:
      The item indices of boundaries between different splits. For the above
      example and n_items=100, these will be
      [('train', 0, 60), ('dev', 60, 80), ('test', 80, 100)].
    """
    if len(split_probs) > n_items:
        raise ValueError('Not enough items for the splits. There are {splits} '
                         'splits while there are only {items} items'.format(
            splits=len(split_probs), items=n_items))
    total_probs = sum(p for name, p in split_probs)
    if abs(1 - total_probs) > 1E-8:
        raise ValueError('Probs should sum up to 1. probs={}'.format(split_probs))
    split_boundaries = []
    sum_p = 0.0
    for name, p in split_probs:
        prev = sum_p
        sum_p += p
        split_boundaries.append((name, int(prev * n_items), int(sum_p * n_items)))

    # Guard against rounding errors.
    split_boundaries[-1] = (split_boundaries[-1][0], split_boundaries[-1][1],
                            n_items)
    return split_boundaries


#SOURCE: https://github.com/tensorflow/datasets/blob/17b40dfdf6ce13adde74f82dd1214fe26545b0d3/tensorflow_datasets/audio/crema_d.py#L90
def _get_inter_splits_by_group(items_and_groups, split_probs, split_number):
    """Split items to train/dev/test, so all items in group go into same split.
    Each group contains all the samples from the same speaker ID. The samples are
    splitted between train, validation and testing so that samples from each
    speaker belongs to exactly one split.
    Args:
      items_and_groups: Sequence of (item_id, group_id) pairs.
      split_probs: List of (split_name, prob), e.g. [('train', 0.6), ('dev', 0.2),
        ('test', 0.2)]
      split_number: Generated splits should change with split_number.
    Returns:
      Dictionary that looks like {split name -> set(ids)}.
    """
    groups = sorted(set(group_id for item_id, group_id in items_and_groups))
    rng = np.random.RandomState(split_number)
    rng.shuffle(groups)

    split_boundaries = _compute_split_boundaries(split_probs, len(groups))
    group_id_to_split = {}
    for split_name, i_start, i_end in split_boundaries:
        for i in range(i_start, i_end):
            group_id_to_split[groups[i]] = split_name

    split_to_ids = collections.defaultdict(set)
    for item_id, group_id in items_and_groups:
        split = group_id_to_split[group_id]
        split_to_ids[split].add(item_id)
    return split_to_ids


class WindowedDataset(IterableDataset):
    """
    Iterates over recordings of an audio dataset in chunks of a given length,
    with a given amount or fraction of overlap. If `pad_incomplete` is "zero",
    the last chunk will be zero-padded; if "overlap", it will be overlapped
    with the previous chunk; if "drop", it will be omitted.
    """
    def __init__(self, dataset, window_size, overlap=0, pad_incomplete='zero'):
        super().__init__()
        self.dataset = dataset
        self.window_size = window_size
        self.pad_incomplete = pad_incomplete
        if 0 < abs(overlap) < 1:
            self.overlap = int(overlap * window_size)  # interpret as fraction
        else:
            self.overlap = int(overlap)

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            offset = 0
            stride = 1
        else:
            offset = worker_info.id
            stride = worker_info.num_workers
        for idx in range(offset, len(self.dataset), stride):
            audio, label = self.dataset[idx]
            audio_size = audio.shape[1]
            hop_size = self.window_size - self.overlap
            start_pos = 0
            while audio_size - start_pos >= self.window_size:
                # yield all complete chunks, with the given amount of overlap
                yield audio[:, start_pos:start_pos + self.window_size], label, idx
                start_pos += hop_size
            if self.pad_incomplete == 'drop' and start_pos > 0:
                # drop any remainder of the recording, move to the next file
                continue
            elif self.pad_incomplete == 'overlap' and start_pos < audio_size:
                # overlap last chunk with the previous to last chunk
                start_pos = max(0, audio_size - self.window_size)
            if start_pos < audio_size:
                # return last chunk, zero-padded at the end if needed
                chunk = audio[:, start_pos:]
                if chunk.shape[1] < self.window_size:
                    chunk = torch.nn.functional.pad(
                        chunk, (0, self.window_size - chunk.shape[1]))
                yield chunk, label, idx


def align_sample(sample: torch.Tensor, seq_len: int=16000): #sample shape: (channels, seq_len)
    pad_length = seq_len - sample.shape[1]
    if pad_length == 0:
        return sample
    elif pad_length > 0: #padding
        return torch.nn.functional.pad(sample, pad=(0, pad_length), mode='constant', value=0.)
    else: #cropping
        max_start_pos = (pad_length * -1) + 1 #draw from 0 to max_start_pos
        pos = np.random.randint(max_start_pos)
        return sample[:, pos:pos + seq_len]


def db_to_linear(samples):
    return 10.0 ** (samples / 20.0)


def loudness_normalization(samples: torch.Tensor,
                           target_db: float=15.0,
                           max_gain_db: float=30.0):
    """Normalizes the loudness of the input signal."""
    std = torch.std(samples) + 1e-9
    gain = np.minimum(db_to_linear(max_gain_db), db_to_linear(target_db) / std)
    return gain * samples


def collate_fn(batch, seq_len, label_transform_fn=None):
    # A data tuple has the form:
    # waveform, label, *anything else
    tensors, targets, anythings = [], [], []

    # Gather in lists, normalize waveforms, encode labels as indices
    for waveform, label, *anything in batch:
        norm_wave = waveform.float()
        if label_transform_fn:
            label = label_transform_fn(label)
        tensors.append(loudness_normalization(align_sample(norm_wave, seq_len=seq_len)))
        targets.append(label)
        anythings.append(anything)

    # Group the list of tensors into a batched tensor
    #and loudness normalization
    data = torch.stack(tensors)
    targets = torch.stack(targets)
    return (data, targets) + tuple(zip(*anythings))


def build_dataloaders(train_set, val_set, test_set, label_transform_fn, args):
    collate = partial(collate_fn, seq_len=args.input_size,
                      label_transform_fn=label_transform_fn)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )
    val_loader = torch.utils.data.DataLoader(
        WindowedDataset(val_set, args.input_size, overlap=args.eval_overlap,
                        pad_incomplete=args.eval_pad),
        batch_size=args.batch_size_eval or args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=min(1, args.num_workers),  # must keep order
        pin_memory=args.pin_mem,
    )
    test_loader = torch.utils.data.DataLoader(
        WindowedDataset(test_set, args.input_size, overlap=args.eval_overlap,
                        pad_incomplete=args.eval_pad),
        batch_size=args.batch_size_eval or args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=min(1, args.num_workers),  # must keep order
        pin_memory=args.pin_mem,
    )
    return train_loader, val_loader, test_loader
