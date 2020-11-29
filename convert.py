import pandas as pd
import os 
import numpy as np

DATA_DIR = './data'
files = {
    # 'gtzan.csv': '84480',
    'fma_small_combined.csv': '81920',
    'fma_large_combined.csv': '81920',
}

for f, lk in files.items():
    df = pd.read_csv(os.path.join(DATA_DIR, f))
    labels = df[lk].to_numpy()
    del df[lk]
    spects = df.to_numpy()
    spects = spects.reshape(spects.shape[0], 128, -1)
    np.savez_compressed(
            os.path.join(DATA_DIR, os.path.splitext(f)[0]),
            spects=spects,
            labels=labels
        )

