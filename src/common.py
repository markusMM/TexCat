import os
import json
import time
import spacy

import spacy_fastlang
from spacy_language_detection import LanguageDetector
from spacy.language import Language
import joblib

config = json.load(open("config.json", "r+"))

# for much more consistent model deployment / usage,
# other binary formats are recommended, like ONNX!
model_path = config.get("model_path", "model/texcat.pkl")
model = joblib.load(model_path)
pca_path = config.get("pca_path", "model/pca_22.pkl")
pca = joblib.load(pca_path)

features = config.get("features", "text_enc_pc")
label_mapping = config.get(
    "label_mapping", {"ft": 0, "mr": 1, "ct": 2, "pkg": 3, "ch": 4, "cnc": 5}
)
inv_label_mapping = {}
for k in label_mapping.keys():
    inv_label_mapping |= {label_mapping[k]: k}

nlp_model = config.get("spacy_model", "xx_sent_ud_sm")
try:
    nlp_test = spacy.load(nlp_model)
    del nlp_test
except:
    os.system(f"python -m spacy download {nlp_model}")


def get_lang_detector(nlp, name):
    return LanguageDetector(seed=42)  # We use the seed 42


def nlp_fast_lang(nlp_model: str = nlp_model):
    """
    NLP Fast Language Detector

    This initializese an NLP model with Spacy-FastLang
    as backend which is supposed to be a little less accurate, but
    faster, than Spacy-Language-Detection.

    Args:
        nlp_model (str): original NLP model name. Defaults to "xx_sent_ud_sm".

    Returns:
        "spacy.NLP": NLP model with final language detection added to the pipe.
    """
    nlp = spacy.load(nlp_model)
    nlp.add_pipe("language_detector")
    return nlp


def nlp_slow_lang(nlp_model: str = nlp_model):
    """
    NLP Slow Language Detector

    This initializese an NLP model with Spacy-Language-Detection
    as backend which is supposed to be a little more accurate, but
    slower, than Spacy-FastLang.

    The time is added to the name of the detection module to prevent
    naming issues on multiple initializations.

    Args:
        nlp_model (str): original NLP model name. Defaults to "xx_sent_ud_sm".

    Returns:
        "spacy.NLP": NLP model with final language detection added to the pipe.
    """
    nlp = spacy.load(nlp_model)
    # time.time is used to prevent multiple workers with same name
    t = int(time.time())
    Language.factory(f"lang_det_{t}", func=get_lang_detector)
    nlp.add_pipe(f"lang_det_{t}", last=True)
    return nlp
