import pandas as pd
import spacy_fastlang
from spacy_language_detection import LanguageDetector
from spacy.language import Language

nlp_model="xx_sent_ud_sm"
try:
    nlp_test = spacy.load(nlp_model)
    del nlp_test
except:
    !python -m spacy download $nlp_model


def get_lang_detector(nlp, name):
    return LanguageDetector(seed=42)  # We use the seed 42


def nlp_fast_lang(nlp_model=nlp_model):
    nlp = spacy.load(nlp_model)
    nlp.add_pipe("language_detector")
    return nlp


def nlp_slow_lang(nlp_model=nlp_model):
    nlp = spacy.load(nlp_model)
    t = int(time.time())
    Language.factory(f"lang_det_{t}", func=get_lang_detector)
    nlp.add_pipe(f'lang_det_{t}', last=True)
    return nlp


def detect_language(
    text,
    nlp_fast=nlp_fast_lang(),
    nlp_slow=nlp_slow_lang()
):
    doc = nlp_fast(text)
    # fallback detector
    if np.round(100*doc._.language_score) < 50:
        doc = nlp_slow(text)
    if isinstance(doc._.language, str):
        lng = doc._.language
        scr = doc._.language_score
    else:
        lng = doc._.language['language']
        scr = doc._.language['score']
    return pd.Series([
        lng,
        np.round(100*float(scr))
    ], index=['lang', 'Lang_acc'])


def pp_detect_language(df: pd.DataFrame, column:str = 'text') -> pd.DataFrame:
    return pd.concat({
        df,
        df[column].apply(detect_language)
    ], axis=1)


def pp_numeric_label_with_missings(
    df: pd.DataFrame,
    ycol:str = 'label',
    mapping: dict = {
        'ft': 0,
        'mr': 1,
        'ct': 2,
        'pkg': 3,
        'ch': 4,
        'cnc': 5,
    },
    missing_class:int = -1
) -> pd.DataFrame:
    df['y'] = df[ycol].map(mapping).fillna(missing_class).astype(int)
    return df
    
