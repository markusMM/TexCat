from sklearn.decomposition import PCA
import pandas as pd


def ca_pca(df: pd.DataFrame, pca_model: PCA, column: str = "text_enc") -> pd.DataFrame:
    """
    Apply Principal Component Analysis (PCA) to the embeddings in a DataFrame column.

    This function applies PCA transformation to the embeddings stored in the specified DataFrame column
    using the provided PCA model, and adds a new column with the transformed embeddings to the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the embeddings data.
        pca_model (PCA): The pre-trained PCA model to be applied.
        column (str, optional): The name of the column containing the embeddings data. Defaults to 'text_enc'.

    Returns:
        pd.DataFrame: The input DataFrame with an additional column containing the PCA-transformed embeddings.
    """
    df[column + "_pc"] = df[column].apply(pca_model.transform)
    return df
