import librosa
import pysptk
import scipy.io.wavfile as wav
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class AudioTrainDataset(Dataset):
    def __init__(self, root_dir, index_list):
        self.data = []
        data = pd.read_excel(root_dir)
        for i in index_list:
            self.data.append((data.iloc[i]['filename'], data.iloc[i]['label'], data.iloc[i]['bf']))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index: int):
        data = self.data[index]
        audio_dir = os.path.join("data/train",data[0])
        label = data[1]
        bf = np.array([data[2]])
        mfcc = self.getmfcc(audio_dir)
        input = np.concatenate((mfcc, bf))
        return input, np.int64(label)
    def getmfcc(self, wav_file_path):
        y_ps, sr = librosa.load(wav_file_path, sr=None, duration=2)
        mfcc = np.mean(librosa.feature.mfcc(y=y_ps, sr=sr, n_mfcc=99, n_fft=1024, hop_length=512), axis=1)
        return mfcc

class AudioTestDataset(Dataset):
    def __init__(self, root_dir):
        self.data = []
        data = pd.read_excel(root_dir)
        for i in range(data.shape[0]):
            self.data.append((data.iloc[i]['filename'], data.iloc[i]['bf']))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index: int):
        data = self.data[index]
        audio_dir = os.path.join("data/test",data[0])
        bf = np.array([data[1]])
        mfcc = self.getmfcc(audio_dir)
        input = np.concatenate((mfcc, bf))
        return input
    def getmfcc(self, wav_file_path):
        y_ps, sr = librosa.load(wav_file_path, sr=None, duration=2)
        mfcc = np.mean(librosa.feature.mfcc(y=y_ps, sr=sr, n_mfcc=99, n_fft=1024, hop_length=512), axis=1)
        return mfcc