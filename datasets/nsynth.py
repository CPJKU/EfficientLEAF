## imports
#basic
import os
import json
import urllib
import tarfile

#torch
import torch
from torch.utils.data import Dataset

#torchaudio
import torchaudio
import platform
if platform.system() == 'Windows':
    torchaudio.set_audio_backend("soundfile") # using torchaudio on a windows machine

#shared
from . import build_dataloaders

## Nsynth Dataset
### Labels for Instruments (are already encoded)
def label_to_index(word):
    labels = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal']
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


def index_to_label(index):
    labels = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal']
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]

class Nsynth(Dataset):
    def __init__(self, root='/content', subset='instrument', split='train'):
        '''
        subset can be 'instrument' (11 classes) or 'pitch' (128 classes)
        '''
        assert split == 'test' or split == 'train' or split == 'validation', 'split must be "train", "validation" or "test"'
        assert subset == 'instrument' or subset == 'pitch', 'subset must be "instrument" or "pitch"'
        if split == 'validation': split = 'valid'

        # inits
        #parameters
        self.split = split
        self.subset = subset

        #paths
        self.root_data = os.path.join(root, 'nsynth', 'nsynth-{}'.format(split))

        #data
        if not os.path.isdir(self.root_data):
            os.makedirs(os.path.join(root, 'nsynth'), exist_ok=True) #init folder
            durl = 'http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-{}.jsonwav.tar.gz'.format(split)
            #download
            with urllib.request.urlopen(durl) as u:
                with tarfile.open(fileobj=u, mode='r|gz', errorlevel=1) as t:
                    t.extractall(os.path.join(root, 'nsynth'))

        #load description json
        with open(os.path.join(self.root_data, 'examples.json'), 'r') as f:
            self.example_desc = json.load(f)

        #get data
        self.waveforms_locations = []
        self.labels = []
        for name, attr in self.example_desc.items():
            if self.subset == 'instrument':
                label = attr['instrument_family']
            if self.subset == 'pitch':
                label = attr['pitch']
            self.labels.append(torch.tensor(label))
            self.waveforms_locations.append(os.path.join(self.root_data, 'audio', '{}.wav'.format(name)))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio, _ = torchaudio.load(self.waveforms_locations[idx])
        label = self.labels[idx]

        return audio, label

## Build dataset function returns all 3 dataloaders
def build_dataset_inst(args):
    test_set = Nsynth(root=args.data_path, subset='instrument', split='test')
    val_set = Nsynth(root=args.data_path, subset='instrument', split='validation')
    train_set = Nsynth(root=args.data_path, subset='instrument', split='train')
    train_loader, val_loader, test_loader = build_dataloaders(
        train_set, val_set, test_set,
        label_transform_fn=None, args=args)
    nb_classes = 11
    return train_loader, val_loader, test_loader, nb_classes

def build_dataset_pitch(args):
    test_set = Nsynth(root=args.data_path, subset='pitch', split='test')
    val_set = Nsynth(root=args.data_path, subset='pitch', split='validation')
    train_set = Nsynth(root=args.data_path, subset='pitch', split='train')
    train_loader, val_loader, test_loader = build_dataloaders(
        train_set, val_set, test_set,
        label_transform_fn=None, args=args)
    nb_classes = 128
    return train_loader, val_loader, test_loader, nb_classes
