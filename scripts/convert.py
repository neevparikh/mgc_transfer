import pandas as pd
import os
import numpy as np

DATA_DIR = './data'
files = {
    # 'test.csv': '81920',
    # 'gtzan.csv': '84480',
    # 'fma_small_combined.csv': '81920',
    'fma_large_combined.csv': '81920',
}

for f, lk in files.items():
    print("Reading", f)
    reader = pd.read_csv(os.path.join(DATA_DIR, f), chunksize=50)
    labels = []
    spects = []
    for df in reader:
        if 'fma' in f or 'test' in f:
            del df['Unnamed: 0']
            del df['Unnamed: 0.1']
        print("{} of the dataset is missing, dropping".format(
            df[lk].isna().sum() / len(df)))
        df = df.dropna(axis=0)
        label = df[lk].to_numpy()
        del df[lk]
        spect = df.to_numpy()
        spect = spect.reshape(spect.shape[0], 128, -1)
        spect = np.abs(spect)
        labels.append(label)
        spects.append(spect)

    print("Saving to disk")
    labels = np.concatenate(labels, axis=0)
    spects = np.concatenate(spects, axis=0)
    np.savez_compressed(os.path.join(DATA_DIR,
                                     os.path.splitext(f)[0]),
                        spects=spects,
                        labels=labels)
