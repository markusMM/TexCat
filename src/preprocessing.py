import numpy as np
import pandas as pd
from src.common import nlp_fast_lang, nlp_slow_lang


def detect_language(
    text: str, nlp_fast=nlp_fast_lang(), nlp_slow=nlp_slow_lang()
) -> pd.Series:
    """
    Detect Language of given `text`.

    Uses both a fast first detector and a fallback model on less than 50%
    score (rounded).

    **NOTE: Bost models need a final language detection in the pipeline!**

    Args:
        text (str): _description_
        nlp_fast ("spacy.NLP"): _description_. Defaults to nlp_fast_lang().
        nlp_slow ("spcay.NLP"): _description_. Defaults to nlp_slow_lang().

    Returns:
        pd.Series: A Pandas Series with "lang" being the language short and
            "lang_acc" being the score of the final language detection.
    """
    doc = nlp_fast(text)
    # fallback detector
    if np.round(100 * doc._.language_score) < 50:
        doc = nlp_slow(text)
    if isinstance(doc._.language, str):
        lng = doc._.language
        scr = doc._.language_score
    else:
        lng = doc._.language["language"]
        scr = doc._.language["score"]
    return pd.Series([lng, np.round(100 * float(scr))], index=["lang", "Lang_acc"])


def pp_detect_language(df: pd.DataFrame, column: str = "text") -> pd.DataFrame:
    """
    PP Detect Language of text `column`.

    Runs the function `detect_language` with default models.
    Here those models come from `src.common`.

    This adds 'lang', the language, and 'lang_acc', the detector score,
    to the data frame and returns it back.

    Args:
        df (pd.DataFrame): data to be processed
        column (str, optional): name of the text column. Defaults to 'text'.

    Returns:
        pd.DataFrame: data with
    """
    return pd.concat([df, df[column].apply(detect_language)], axis=1)


def pp_numeric_label_with_missings(
    df: pd.DataFrame,
    ycol: str = "label",
    mapping: dict = {
        "ft": 0,
        "mr": 1,
        "ct": 2,
        "pkg": 3,
        "ch": 4,
        "cnc": 5,
    },
    missing_class: int = -1,
) -> pd.DataFrame:
    """
    PP Numeric Labels with Missings.

    This maps a str label `column` to a numberic `y` column Using
    `mappings`.
    `missing_class` is the default mapping for NaN.

    This is intended to be used with semi-supervised learning.

    The added new column is named "y"!

    Args:
        df (pd.DataFrame): the data
        ycol (str, optional): The label column. Defaults to 'label'.
        mapping (dict): The mapping for each label.
            Defaults to { 'ft': 0, 'mr': 1, 'ct': 2, 'pkg': 3, 'ch': 4, 'cnc': 5, }.
        missing_class (int, optional): mapping for NaN. Defaults to -1.

    Returns:
        pd.DataFrame: data with "y", the numerical mapping
    """
    df["y"] = df[ycol].map(mapping).fillna(missing_class).astype(int)
    return df
