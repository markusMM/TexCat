from sklearn.decomposition import PCA
import pandas as pd


def ca_pca(df: pd.DataFrame, pca_model: PCA, column: str = "text_enc") -> pd.DataFrame:
    df[column + "_pc"] = df[column].apply(pca_model.transform)
    return df
