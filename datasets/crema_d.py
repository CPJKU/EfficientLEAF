## imports
#basic
import os
import collections
import requests

#processing
from tqdm.auto import tqdm
import numpy as np
import pandas as pd

#torch
import torch
from torch.utils.data import Dataset

#torchaudio
import torchaudio
import platform
if platform.system() == 'Windows':
    torchaudio.set_audio_backend("soundfile") # using torchaudio on a windows machine

#shared
from . import (_compute_split_boundaries, _get_inter_splits_by_group,
               build_dataloaders)


## Crema-D
class Crema_D(Dataset):
    def __init__(self, root = '/content', split='train', seed=10):
        #same seed for train and test(!)
        assert split == 'test' or split == 'train' or split == 'validation', 'split must be "train" or "test"'

        # inits
        #urls
        self.wav_data_url = 'https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/AudioWAV/'
        self.csv_summary_url = 'https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master/processedResults/summaryTable.csv'
        #paths
        self.root_data = os.path.join(root, 'crema_d')
        if not os.path.isdir(self.root_data): os.mkdir(self.root_data) #init folder
        #parameters
        self.split = split
        self.seed = seed
        #x,y
        self.waveforms = []
        self.labels = []

        #fill x,y
        self.split_load()

    def split_load(self):
        csv_summary = pd.read_csv(self.csv_summary_url, index_col=0)
        all_wav_files = []
        speaker_ids = []
        wav_names = []
        labels = []
        # These are file names which do do not exist in the github
        bad_files = set([
            'FileName', '1040_ITH_SAD_XX', '1006_TIE_NEU_XX', '1013_WSI_DIS_XX',
            '1017_IWW_FEA_XX', '1010_IWL_SAD_XX'
        ])

        # get info from summary csv
        for _, row in csv_summary.iterrows():
            wav_name = row['FileName']

            if (not wav_name) or (wav_name in bad_files): #skip badfiles (by google)
                continue

            wav_path = os.path.join(self.wav_data_url, '%s.wav' % wav_name)
            all_wav_files.append(wav_path)
            speaker_ids.append(wav_name.split('_')[0])
            wav_names.append(wav_name)
            labels.append(wav_name.split('_')[2])

        #splitting train/test
        items_and_groups =  list(zip(wav_names, speaker_ids))
        all_wav_info = list(zip(all_wav_files, wav_names, speaker_ids, labels))
        items_and_groups = list(zip(all_wav_info, speaker_ids))
        split_probs = [('train', 0.7), ('validation', 0.1), ('test', 0.2)]
        split_to_ids = _get_inter_splits_by_group(items_and_groups, split_probs, split_number=0)

        #download file if not already in crema_d folder
        tqdm_dl = tqdm(range(1, len(split_to_ids[self.split])), desc= f'Downloading {self.split}')
        for wave_url, wave_name, speaker_id, label in split_to_ids[self.split]:

            #download
            wave_file_location = os.path.join(self.root_data, wave_name+'.wav')
            if not os.path.isfile(wave_file_location):
                with open(wave_file_location, 'wb') as file_:
                    file_.write(requests.get(wave_url).content)

            waveform, sample_rate = torchaudio.load(wave_file_location)
            if sample_rate != 16000:
                print('Samplerate of', wave_name, 'is', sample_rate)
            self.waveforms.append(waveform)
            self.labels.append(label)
            tqdm_dl.update()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio = self.waveforms[idx]
        label = self.labels[idx]

        return audio, label


def label_to_index(word):
    labels = ['NEU', 'HAP', 'SAD', 'ANG', 'FEA', 'DIS']
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))


def index_to_label(index):
    labels = ['NEU', 'HAP', 'SAD', 'ANG', 'FEA', 'DIS']
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]


## Build dataset function returns all 3 dataloaders
def build_dataset(args):
    test_set = Crema_D(root = args.data_path, split='test', seed=args.seed)
    val_set = Crema_D(root = args.data_path, split='validation', seed=args.seed)
    train_set = Crema_D(root = args.data_path, split='train', seed=args.seed)
    train_loader, val_loader, test_loader = build_dataloaders(
        train_set, val_set, test_set,
        label_transform_fn=label_to_index, args=args)
    nb_classes = 6
    return train_loader, val_loader, test_loader, nb_classes
