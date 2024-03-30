from typing import Union, List
import numpy as np
import pandas as pd
from src.preprocessing import pp_detect_language, pp_numeric_label_with_missings
from src.text_encoding import enc_herbert
from src.compression import ca_pca
from src.common import model, inv_label_mapping, features, pca


def categorize(text: Union[str, List[str]]) -> np.ndarray:
    if not isinstance(text, list):
        text = [text]
    df = pd.DataFrame([text], cloumns="text")
    df = pp_detect_language(df)
    df = pp_numeric_label_with_missings(df)
    df = enc_herbert(df)
    df = ca_pca(df, pca).drop("text_enc", axis=1)
    pred = model.predict(df[features].values.fillna(0))
    return pd.Series(pred).map(inv_label_mapping).values
