from suggester.objects import METADATA_COLS

from sklearn.manifold import trustworthiness

import pandas as pd

# TODO: This needs to be somewhere else (reducer module?)

def trust_cont(original: pd.DataFrame, reduced: pd.DataFrame):
    original = original.drop(columns=METADATA_COLS)
    reduced = reduced.drop(columns=METADATA_COLS)
    return trustworthiness(original, reduced), trustworthiness(reduced, original)
