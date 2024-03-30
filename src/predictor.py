from typing import Union, List
import joblib
import numpy as np
import pandas as pd
from src.preprocessing import pp_detect_language, pp_label_with_missings
from src.text_encoding import enc_herbert
from common import model, inv_label_mapping, features

def categorize(text: Union[str, List[str]]) -> np.ndarray:
    if not isinstance(text, list):
        text = [text]
    df = pd.DataFrame([text], cloumns='text')
    df = pp_detect_language(df)
    df = pp_label_with_missings(df)
    df = enc_herbert(df)
    pred = model.predict(df[features].values.fillna(0))
    return pd.Series(pred).map(inv_label_mapping).values
