## imports
#torch
import torch
from torch.utils.data import Dataset

#torchaudio
import torchaudio
import platform
if platform.system() == 'Windows':
    torchaudio.set_audio_backend("soundfile") # using torchaudio on a windows machine
from torchaudio.datasets import SPEECHCOMMANDS

#shared
from . import build_dataloaders


## Speechcommands Dataset
## Speech Commands
def label_to_index(word):
    labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


def index_to_label(index):
    labels = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]


def select_columns(datatuple):
    waveform, sample_rate, label, speaker_id, utterance_number = datatuple
    return waveform, label


class TransformedDataset(Dataset):
    def __init__(self, dataset, transform_fn):
        super().__init__()
        self.dataset = dataset
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.transform_fn(self.dataset[idx])


## Build dataset function returns all 3 dataloaders
def build_dataset(args):
    test_set = SPEECHCOMMANDS(root= args.data_path, subset = 'testing', download=True)
    val_set = SPEECHCOMMANDS(root= args.data_path, subset = 'validation', download=True)
    train_set = SPEECHCOMMANDS(root= args.data_path, subset = 'training', download=True)
    test_set = TransformedDataset(test_set, select_columns)
    val_set = TransformedDataset(val_set, select_columns)
    train_set = TransformedDataset(train_set, select_columns)
    train_loader, val_loader, test_loader = build_dataloaders(
        train_set, val_set, test_set,
        label_transform_fn=label_to_index, args=args)
    nb_classes = 35

    return train_loader, val_loader, test_loader, nb_classes
