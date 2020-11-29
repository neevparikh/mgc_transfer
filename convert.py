import pandas as pd
import os 
import numpy as np

DATA_DIR = './data'
files = {
    'test.csv': '81920',
    'gtzan.csv': '84480',
    'fma_small_combined.csv': '81920',
    'fma_large_combined.csv': '81920',
}

for f, lk in files.items():
    print("Reading", f)
    df = pd.read_csv(os.path.join(DATA_DIR, f))
    if 'fma' in f or 'test' in f:
        del df['Unnamed: 0']
        del df['Unnamed: 0.1']
    print("{} of the dataset is not missing, dropping others".format(len(df.isna()) / len(df)))
    df = df.dropna(axis=0)
    labels = df[lk].to_numpy()
    del df[lk]
    spects = df.to_numpy()
    spects = spects.reshape(spects.shape[0], 128, -1)
    spects = np.abs(spects)
    print("Saving to disk")
    np.savez_compressed(
            os.path.join(DATA_DIR, os.path.splitext(f)[0]),
            spects=spects,
            labels=labels
        )

