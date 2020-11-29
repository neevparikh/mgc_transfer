import os
import pickle
import os.path
import re
import warnings
import sys
import glob

import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm 


def make_label_dic(path_to_metadata):
    '''
    This function takes in a directory of audio files in .wav format, computes the
    mel spectrogram for each audio file, reshapes them so that they are all the
    same size, flattens them, and stores them in a dataframe.

    Genre labels are also computed and added to the dataframe.

    Parameters:
    directory (int): a directory of audio files in .wav format

    Returns:
    df (DataFrame): a dataframe of flattened mel spectrograms and their
    corresponding genre labels
    '''

    num_to_label_dic = {}
    genres_tracks = pd.read_csv(path_to_metadata, index_col=0, header=[0, 1])

    keep_cols = [('track', 'genre_top')]

    df_all = genres_tracks[keep_cols]

    df_all['track_id'] = df_all.index

    for _, row in df_all.iterrows():
        track_id = int(row['track_id'])
        genre = str(row[('track', 'genre_top')])
        num_to_label_dic[track_id] = genre

    with open('label_dic.pkl', 'wb') as f:
        pickle.dump(num_to_label_dic, f)

def make_mel_spectrogram_df(directory):

    # Creating empty lists for mel spectrograms and labels
    labels = []
    mel_specs = []

    with open('label_dic.pkl', 'rb') as f:
        num_to_label_dic = pickle.load(f)

    # Looping through each file in the directory
    for f in tqdm(glob.glob('{}/*.mp3'.format(directory))):
        try:
            y, sr = librosa.load(f)

            # Extracting the label and adding it to the list
            curr_name = os.path.basename(f)
            label = num_to_label_dic.get(int(curr_name[:-4]))

            tqdm.write(label)
            labels.append(label)

            # Computing the mel spectrograms
            spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
            spect = librosa.power_to_db(spect, ref=np.max)

            # Adjusting the size to be 128 x 640
            if spect.shape[1] != 640:
                spect.resize(128, 640, refcheck=False)

            # Flattening to fit into dataframe and adding to the list
            spect = spect.flatten()
            mel_specs.append(spect)
        except Exception as e:
            tqdm.write(str(e))
            continue

    # Converting the lists to arrays so we can stack them
    mel_specs = np.array(mel_specs)
    labels = np.array(labels).reshape(len(labels), 1)

    # Create dataframe
    df = pd.DataFrame(np.hstack((mel_specs, labels)))

    return df


warnings.simplefilter("ignore")
if not os.path.isfile('label_dic.pkl'):
    print('making label dict')
    make_label_dic('tracks.csv')

df = make_mel_spectrogram_df(sys.argv[1])
df.to_csv('{}.csv'.format(sys.argv[1]))
