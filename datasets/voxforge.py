## imports
#basic
import os
import warnings

#processing
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
##for this set needed
from six.moves import urllib
import subprocess
import tarfile
from urllib.error import HTTPError

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

## Voxforge
### Helper Functions
#check if ffmpeg or sox are installed
def ffmpeg_or_sox_installed():
    import subprocess
    try:
        subprocess.check_output(['ffmpeg', '-version'])
    except FileNotFoundError:
        try:
            subprocess.check_output(['sox', '-h'])
        except FileNotFoundError:
            raise RuntimeError("Neither ffmpeg nor sox are installed, one is needed!")


def prepare_sample(url, recording_path, label, recording_name, sample_rate):
    """
    Downloads and extracts a sample from VoxForge. Gets the LABEL (Language ID) and puts the wav into:target_folder/recording_name_LABEL.
    """
    # download only if still needed
    if not os.path.exists(recording_path):
        # open compressed file from web
        try:
            with urllib.request.urlopen('http://' + url) as u:
                with tarfile.open(fileobj=u, mode='r|gz', errorlevel=1) as t:
                    # extract all .wav and .flac files
                    for member in t:
                        if member.name.endswith('.wav') or member.name.endswith('.flac'):
                            os.makedirs(recording_path, exist_ok=True)
                            outfile = os.path.join(recording_path,
                                                   os.path.basename(member.name))
                            with open(outfile, 'wb') as f:
                                f.write(t.extractfile(member).read())
                            # TODO: check whether sample rate is correct for .wav
                            # convert .flac to .wav
                            if outfile.endswith('.flac'):
                                wavoutfile = outfile[:-len('.flac')] + '.wav'
                                try:
                                    subprocess.call(['ffmpeg', '-v', 'fatal', '-i', outfile, '-c:a', 'pcm_s16le', '-ac', '1', '-ar', str(sample_rate), wavoutfile])
                                except FileNotFoundError:
                                    subprocess.call(['sox', outfile, '-b', '16', '-c', '1', '-r', str(sample_rate), wavoutfile])
                                if os.path.exists(wavoutfile):
                                    os.unlink(outfile)
        except urllib.error.HTTPError as e:
            print("Error %d for %s; skipping" % (e.code, url))
            with open(recording_path, 'w') as f:
                f.write('Failed with HTTP Error %d' % e.code)

    if os.path.isdir(recording_path):
        # get all wav files from this sample
        wav_files = [os.path.join(recording_path, fn)
                     for fn in os.listdir(recording_path)]
        # make labels
        labels = [label] * len(wav_files)
        return wav_files, labels
    else:
        return [], []


### Dataset
class Voxforge(Dataset):
    def __init__(self, root = '/content', split='train', seed=0, sample_rate = 16000):
        #same seed for train and test(!)
        assert split == 'test' or split == 'train' or split == 'validation', 'split must be "train", "validation" or "test"'
        ffmpeg_or_sox_installed() #check if ffmpeg or sox are installed

        # inits
        #paths
        self.root_data = os.path.join(root, 'voxforge')
        if not os.path.isdir(self.root_data): os.mkdir(self.root_data) #init folder
        #urls
        self.urls_list_file = os.path.join(self.root_data, 'voxforge_urls.txt')
        if not os.path.exists(self.urls_list_file):
            with open(self.urls_list_file, 'wb') as f:
                with urllib.request.urlopen('https://storage.googleapis.com/tfds-data/downloads/voxforge/voxforge_urls.txt') as u:
                    f.write(u.read())
        #parameters
        self.split = split
        self.seed = seed
        self.sample_rate = sample_rate
        #x,y
        self.waveforms_locations = []
        self.labels = []

        #fill x,y
        self.split_load()

    def split_load(self):
        file_urls = pd.read_csv(self.urls_list_file, header=None) #read in txt file
        LABELS = ['DE', 'EN', 'ES', 'FR', 'IT', 'RU']

        archives_and_speaker_ids = []
        for _, row in file_urls.iterrows():
            url = row[0].strip().replace('"', '').replace('\'', '')
            recording_name = url.split('/')[-1][:-4]
            label = url.split('/')[2].upper()
            if label not in LABELS: #check for label
                warnings.warn("Language found, but it is non of the set labels: {} - maybe add manually".format(label))
            speaker_id = recording_name.split('-')[0]
            recording_path = os.path.join(self.root_data, '{}_{}'.format(recording_name, label))
            archives_and_speaker_ids.append(((url, recording_path, label, recording_name), speaker_id))


        split_probs = [('train', 0.7), ('validation', 0.1), ('test', 0.2)]
        splits = _get_inter_splits_by_group(archives_and_speaker_ids, split_probs, 0)


        #download file if not already in crema_d folder
        tqdm_dl = tqdm(range(1, len(splits[self.split])), f'Downloading {self.split}')
        for url, recording_path, label, recording_name in splits[self.split]:
            wave_form_location, labels = prepare_sample(url, recording_path, label, recording_name, self.sample_rate)


            self.waveforms_locations.extend(wave_form_location)
            self.labels.extend(labels)
            tqdm_dl.update()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio, _ = torchaudio.load(self.waveforms_locations[idx])
        label = self.labels[idx]

        return audio, label


def label_to_index(word):
    labels = ['DE', 'EN', 'ES', 'FR', 'IT', 'RU']
    # Return the position of the word in labels
    return torch.tensor(labels.index(word))

def index_to_label(index):
    labels = ['DE', 'EN', 'ES', 'FR', 'IT', 'RU']
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]


## Build dataset function returns all 3 dataloaders
def build_dataset(args):
    test_set = Voxforge(root=args.data_path, split='test', seed=0, sample_rate=16000)
    val_set = Voxforge(root=args.data_path, split='validation', seed=0, sample_rate=16000)
    train_set = Voxforge(root=args.data_path, split='train', seed=0, sample_rate=16000)
    train_loader, val_loader, test_loader = build_dataloaders(
        train_set, val_set, test_set,
        label_transform_fn=label_to_index, args=args)
    nb_classes = 6
    return train_loader, val_loader, test_loader, nb_classes
