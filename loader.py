import pandas as pd
import numpy as np


def load_ratings():
    df = pd.read_excel('jester-data-1.xls', header=None)
    df = df.drop(columns=[0])  # col 0 is the rated-joke count, not a rating
    return df.to_numpy(dtype=float)
