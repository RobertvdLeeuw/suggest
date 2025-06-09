from objects import METADATA_COLS

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from umap import UMAP

import pandas as pd


def pca(embeddings: pd.DataFrame, n_dim: int) -> pd.DataFrame:
    metadata = embeddings[METADATA_COLS]
    embeddings = embeddings.drop(columns=METADATA_COLS)

    for col in embeddings.columns:
        assert embeddings[col].isna().sum() == 0, f"{col} with NaN's left in data."
    
    scaled = StandardScaler().fit_transform(embeddings)
    reduced = PCA(n_components=n_dim, svd_solver='arpack').fit_transform(scaled)

    return pd.DataFrame(reduced).join(metadata)

    
def umap(embeddings: pd.DataFrame, n_dim: int, densmap=False, n_neighbors=15, min_dist=0.1, spread=1) -> pd.DataFrame:
    metadata = embeddings[METADATA_COLS]
    embeddings = embeddings.drop(columns=METADATA_COLS)

    for col in embeddings.columns:
        assert embeddings[col].isna().sum() == 0, f"{col} with NaN's left in data."

    # embeddings.columns = embeddings.columns.astype(str)

    model = UMAP(n_components=n_dim, 
                 densmap=densmap, 
                 n_neighbors=n_neighbors, 
                 min_dist=min_dist,
                 spread=spread)
    reduced = model.fit_transform(embeddings)

    return pd.DataFrame(reduced).join(metadata)
    
