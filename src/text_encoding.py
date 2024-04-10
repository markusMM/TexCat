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
    """
    Compute average pooling over the last dimension of a tensor.

    Args:
        last_hidden_states (Tensor): Tensor containing the last hidden states from a model.
        attention_mask (Tensor, optional): Tensor indicating the positions of the tokens.
            Defaults to None.

    Returns:
        Tensor: Tensor resulting from average pooling operation.
    """
    if attention_mask is None:
        attention_mask = torch.ones(last_hidden_states.shape[:-1])
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def encode_text(text: str) -> Tensor:
    """
    Encode input text using a pre-trained HerBERT model and return the resulting tensor.

    Args:
        text (str): Input text to be encoded.

    Returns:
        Tensor: Encoded tensor representing the input text.
    """
    encoding = model(tokenizer.encode(
        text, return_tensors="pt"
    )).last_hidden_state
    encoding = average_pool(encoding)
    return encoding / encoding.sum(dim=-1)[..., None]


def enc_herbert(df: pd.DataFrame, column: str = 'text') -> pd.DataFrame:
    """
    Encode HerBERT
    
    Encode text in a DataFrame column using a pre-trained HerBERT model and add the
    resulting encoded representations as a new column.

    Args:
        df (pd.DataFrame): DataFrame containing the text data.
        column (str, optional): Name of the column containing the text data. Defaults to 'text'.

    Returns:
        pd.DataFrame: DataFrame with the encoded representations added as a new column.
    """
    df[column + '_enc'] = df[column].apply(
        lambda x: encode_text(x).detach().numpy()
    )
    return df
