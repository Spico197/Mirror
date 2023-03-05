import torch
import torch.nn as nn
from transformers import BertModel

from src.utils import decode_nnw_thw_mat


class Biaffine(nn.Module):
    """Biaffine transformation

    References:
        - https://github.com/yzhangcs/parser/blob/main/supar/modules/affine.py
        - https://github.com/ljynlp/W2NER
    """

    def __init__(self, n_in, n_out=2, bias_x=True, bias_y=True):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros(n_out, n_in + int(bias_x), n_in + int(bias_y))
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
        # s = s.permute(0, 2, 3, 1)

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

    def __init__(self, hidden_size, biaffine_size, cls_num=2, dropout=0):
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


def multilabel_categorical_crossentropy(y_pred, y_true):
    """
    https://kexue.fm/archives/7359
    https://github.com/gaohongkui/GlobalPointer_pytorch/blob/main/common/utils.py
    """
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = y_pred - (1 - y_true) * 1e12  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    return (neg_loss + pos_loss).mean()


class MrcPointerMatrixModel(nn.Module):
    def __init__(
        self,
        plm_dir: str,
        cls_num: int = 2,
        none_type_id: int = 0,
        text_mask_id: int = 4,
        dropout: float = 0.3,
        pred_threshold: float = 0.5,
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
        # self.nnw_mat = PointerMatrix(
        #     hidden_size, hidden_size, cls_num=1, dropout=dropout
        # )
        # self.thw_mat = PointerMatrix(
        #     hidden_size, hidden_size, cls_num=1, dropout=dropout
        # )
        self.pointer_mat = PointerMatrix(
            hidden_size, hidden_size // 2, cls_num=2, dropout=dropout
        )

        self.pred_threshold = pred_threshold
        # self.criterion = nn.BCEWithLogitsLoss(reduction="sum")
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        # self.criterion = nn.BCEWithLogitsLoss()
        # self.criterion = nn.CrossEntropyLoss()

    def input_encoding(self, input_ids, mask):
        attention_mask = mask.gt(0).float()
        plm_outputs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return plm_outputs.last_hidden_state

    def build_bit_mask(self, mask: torch.Tensor) -> torch.Tensor:
        # mask: (batch_size, seq_len)
        bs, seq_len = mask.shape
        mask_mat = mask.eq(4).unsqueeze(-1).expand((bs, seq_len, seq_len))
        # bit_mask: (batch_size, seq_len, seq_len, 1)
        bit_mask = (
            torch.logical_and(mask_mat, mask_mat.transpose(1, 2))
            .unsqueeze(1)
            .expand((bs, 2, seq_len, seq_len))
            .float()
        )
        return bit_mask

    def forward(self, input_ids, mask, labels=None, is_eval=False, **kwargs):
        hidden = self.input_encoding(input_ids, mask)
        logits = self.pointer_mat(hidden)
        # nnw_hidden = self.nnw_mat(hidden)
        # thw_hidden = self.thw_mat(hidden)
        # # (bs, 2, seq_len, seq_len)
        # logits = torch.cat([nnw_hidden, thw_hidden], dim=1)
        bs, _, seq_len, seq_len = logits.shape

        bit_mask = self.build_bit_mask(mask)

        results = {"logits": logits}
        if labels is not None:
            # # multi-label cross entropy
            # y_pred = logits.reshape(bs * 2, -1)
            # y_true = labels.reshape(bs * 2, -1)
            # loss = multilabel_categorical_crossentropy(y_pred, y_true)
            # results["loss"] = loss

            # mean
            loss = self.criterion(logits.reshape(bs, -1), labels.reshape(bs, -1))
            masked_loss = (loss * bit_mask.reshape(bs, -1)).sum()
            masked_loss = masked_loss / bit_mask.sum()
            results["loss"] = masked_loss

        if is_eval:
            batch_positions = self.decode(logits, bit_mask, **kwargs)
            results["pred"] = batch_positions
        return results

    def decode(self, logits: torch.Tensor, bit_mask: torch.Tensor, **kwargs):
        logits *= bit_mask
        pred = logits.sigmoid().ge(self.pred_threshold).long()
        batch_preds = decode_nnw_thw_mat(pred, offsets=kwargs.get("offset"))

        return batch_preds
