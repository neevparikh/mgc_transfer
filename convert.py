import pandas as pd
import os 
import numpy as np

DATA_DIR = './data'
files = [
    'gtzan.csv',
    # 'fma_small_combined.csv',
    # 'fma_large_combined.csv',
]

for f in files:
    df = pd.read_csv(os.path.join(DATA_DIR, f))
    labels = df['84480'].to_numpy()
    del df['84480']
    spects = df.to_numpy()
    spects = spects.reshape(spects.shape[0], 128, -1)
    np.savez_compressed(
            os.path.join(DATA_DIR, os.path.splitext(f)[0]),
            spects=spects,
            labels=labels
        )

