from typing import Optional

import torch
import torch.nn as nn
from rex.utils.position import extract_positions_from_start_end_label
from transformers import BertModel

from src.utils import decode_nnw_thw_mat


class Biaffine(nn.Module):
    """Biaffine transformation

    References:
        - https://github.com/yzhangcs/parser/blob/main/supar/modules/mlp.py
        - https://github.com/ljynlp/W2NER
    """

    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros((n_out, n_in + int(bias_x), n_in + int(bias_y)))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum("bxi,oij,byj->boxy", x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.permute(0, 2, 3, 1)

        return s


class LinearWithAct(nn.Module):
    def __init__(self, n_in, n_out, dropout=0) -> None:
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        return x


class PointerMatrix(nn.Module):
    """Pointer Matrix Prediction

    References:
        - https://github.com/ljynlp/W2NER
    """

    def __init__(self, cls_num, hidden_size, biaffine_size, dropout=0):
        super().__init__()
        self.linear_h = LinearWithAct(
            n_in=hidden_size, n_out=biaffine_size, dropout=dropout
        )
        self.linear_t = LinearWithAct(
            n_in=hidden_size, n_out=biaffine_size, dropout=dropout
        )
        self.biaffine = Biaffine(n_in=biaffine_size, n_out=cls_num)

    def forward(self, x):
        h = self.linear_h(x)
        t = self.linear_t(x)
        o = self.biaffine(h, t)
        return o


class MrcPointerMatrixModel(nn.Module):
    def __init__(
        self,
        plm_dir: str,
        cls_num: int = 3,
        none_type_id: int = 0,
        text_mask_id: int = 4,
        dropout: Optional[float] = 0.3,
    ):
        super().__init__()

        # num of predicted classes, default is 3: None, NNW and THW
        self.cls_num = cls_num
        # None type id: 0, Next Neighboring Word (NNW): 1, Tail Head Word (THW): 2
        self.none_type_id = none_type_id
        # input: cls instruction sep text sep pad
        # mask:   1       2       3   4    5   0
        self.text_mask_id = text_mask_id

        self.plm = BertModel.from_pretrained(plm_dir)
        hidden_size = self.plm.config.hidden_size
        self.pointer_matrix = PointerMatrix(cls_num, hidden_size, dropout=dropout)

        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def input_encoding(self, input_ids, mask):
        attention_mask = mask.gt(0).float()
        plm_outputs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return plm_outputs.last_hidden_state

    def forward(self, input_ids, mask, labels=None, decode=True, **kwargs):
        hidden = self.input_encoding(input_ids, mask)
        hidden = self.pointer_matrix(hidden)

        results = {"logits": hidden}
        if labels is not None:
            loss = self.criterion(hidden.reshape(-1, self.cls_num), labels.reshape(-1))
            results["loss"] = loss

        if decode:
            batch_positions = self.decode(hidden, mask, **kwargs)
            results["pred"] = batch_positions
        return results

    def decode(self, logits: torch.Tensor, mask: torch.Tensor, **kwargs):
        dtype = logits.dtype
        logits = logits.detach()
        logits[..., self.none_type_id].masked_fill_(
            mask.ne(self.text_mask_id), torch.finfo(dtype).max
        )
        pred = logits.max(dim=-1)[1]
        batch_preds = decode_nnw_thw_mat(pred)

        return batch_preds
