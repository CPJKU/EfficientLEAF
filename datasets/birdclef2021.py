## imports
#basic
import os
import warnings

#processing
from scipy.io.wavfile import read as wav_read
import numpy as np
import pandas as pd

#torch
import torch
from torch.utils.data import Dataset

#shared
from . import (_compute_split_boundaries, _get_inter_splits_by_group,
               build_dataloaders)

## BirdCLEF 2021
### Dataset
class BirdClef(Dataset):
    def __init__(self, root = '/content', split='train', seed=0, sample_rate=16000,
                 fixed_crop=None, random_crop=None):
        #same seed for train and test(!)
        assert split == 'test' or split == 'train' or split == 'validation', 'split must be "train", "validation" or "test"'

        # inits
        #paths
        self.root_data = os.path.join(root, 'birdclef2021')
        if not os.path.isdir(self.root_data):
            raise RuntimeError("BirdCLEF dataset not found in '%s', please "
                               "run prepare_birdclef2021.sh" % self.root_data)
        #metadata
        self.metadata = pd.read_csv(os.path.join(self.root_data,
                                                 'train_metadata.csv'))
        #parameters
        self.split = split
        self.seed = seed
        self.sample_rate = sample_rate
        self.fixed_crop = fixed_crop
        self.random_crop = random_crop
        #x,y
        self.filenames = []
        self.labels = []

        #fill x,y
        self.split_load()

    def split_load(self):
        filenames = [os.path.join(label, fn[:-3] + 'wav')
                     for label, fn in zip(self.metadata.primary_label,
                                          self.metadata.filename)]
        split_probs = [('train', 0.7), ('validation', 0.1), ('test', 0.2)]
        splits = _get_inter_splits_by_group(list(zip(filenames,
                                                     self.metadata.author)),
                                            split_probs, 0)
        self.filenames = sorted(splits[self.split])
        label_ids = dict((label, torch.tensor(idx)) for idx, label
                         in enumerate(sorted(set(self.metadata.primary_label))))
        self.labels = [label_ids[os.path.dirname(fn)] for fn in self.filenames]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        _, audio = wav_read(os.path.join(self.root_data, self.filenames[idx]),
                            mmap=True)
        if self.fixed_crop:
            audio = audio[:self.fixed_crop]
        if self.random_crop:
            if len(audio) > self.random_crop:
                pos = np.random.randint(len(audio) - self.random_crop)
                audio = audio[pos:pos + self.random_crop]
            elif len(audio) < self.random_crop:
                audio = np.concatenate((audio,
                                        np.zeros(self.random_crop - len(audio),
                                                 dtype=audio.dtype)))
        if audio.ndim == 2:
            audio = audio.T  # move channels first
        elif audio.ndim == 1:
            audio = audio[np.newaxis]  # add channels dimension
        if not np.issubdtype(audio.dtype, np.floating):
            audio = np.divide(audio, np.iinfo(audio.dtype).max, dtype=np.float32)
        audio = torch.as_tensor(audio, dtype=torch.float32)
        label = self.labels[idx]

        return audio, label


## Build dataset function returns all 3 dataloaders
def build_dataset(args):
    # test on full recordings
    test_set = BirdClef(root=args.data_path, split='test', seed=0, sample_rate=16000)
    # validate on first 16 seconds
    val_set = BirdClef(root=args.data_path, split='validation', seed=0, sample_rate=16000, fixed_crop=16*16000)
    # train on random excerpt (args.input_size)
    train_set = BirdClef(root=args.data_path, split='train', seed=0, sample_rate=16000, random_crop=args.input_size)
    train_loader, val_loader, test_loader = build_dataloaders(
        train_set, val_set, test_set,
        label_transform_fn=None, args=args)
    nb_classes = 397
    return train_loader, val_loader, test_loader, nb_classes
