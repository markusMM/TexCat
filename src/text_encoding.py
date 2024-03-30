import torch
from torch import Tensor
from transformers import HerbertTokenizer, RobertaModel
import pandas as pd
tokenizer = HerbertTokenizer.from_pretrained("allegro/herbert-klej-cased-tokenizer-v1")
model = RobertaModel.from_pretrained("allegro/herbert-klej-cased-v1")


def average_pool(
    last_hidden_states: Tensor,
    attention_mask: Tensor = None
) -> Tensor:
    if attention_mask is None:
        attention_mask = torch.ones(last_hidden_states.shape[:-1])
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def encode_text(text: str) -> Tensor:
    encoding = model(tokenizer.encode(
        text, return_tensors="pt"
    )).last_hidden_state
    encoding = average_pool(encoding)
    return encoding / encoding.sum(dim=-1)[..., None]


def enc_herbert(df: pd.DataFrame, column: str = 'text') -> pd.DataFrame:
    df[column + '_enc'] = df[column].apply(
        lambda x: encode_text(x).detach().numpy()
    )
    return df
